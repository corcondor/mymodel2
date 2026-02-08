#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Grammar V7 - Reward-ready (based on V6 ~72%)
- V6 program set (mymodel2-like masks + stats + LBP/RGB)
- Reward logging hooks for program-level scoring
- Tokenization (text + ID) for program tracing
- Ready for Transformer/LLM-driven program search
"""

from dataclasses import dataclass
import os
import json
import uuid
from datetime import datetime, timezone
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import convolve
from joblib import Parallel, delayed
import argparse
import warnings
import threading
from queue import Queue
import shutil
import tempfile

# ========== Prefetching Feature Loader (GD → local pipeline) ==========

_SENTINEL = object()

class PrefetchingFeatureLoader:
    """
    遅いストレージ(GD同期領域) → 速いローカル(/tmp) への先読みパイプライン。
    
    仕組み:
      1. バックグラウンドスレッドがGDからローカルにコピー
      2. メインスレッドはローカルの.npyをmmap_mode='r'で高速アクセス
      3. 使用済みファイルは削除して容量回収
    
    前提:
      - .npy ファイル専用（.npzはmmap非対応なので非推奨）
      - チャンクサイズの目安: 256MB〜1GB
    """
    def __init__(self, feature_files, gd_base, cache_dir='/tmp/feat_cache_prefetch',
                 queue_size=2, max_cache_bytes=None):
        """
        Args:
            feature_files: GD上の.npyファイル名リスト
            gd_base: GD上のキャッシュディレクトリパス
            cache_dir: ローカル一時ディレクトリ
            queue_size: 先読みバッファ数（=自然な背圧制御）
            max_cache_bytes: ローカルキャッシュ総量上限（Noneなら無制限）
        """
        self.feature_files = list(feature_files)
        self.gd_base = gd_base
        self.cache_dir = cache_dir
        self.queue = Queue(maxsize=queue_size)
        self.max_cache_bytes = max_cache_bytes
        self._thread = None
        self._stop = threading.Event()
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def start(self):
        """プリフェッチスレッドを開始"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._thread.start()
    
    def _prefetch_worker(self):
        """バックグラウンドでGD→ローカルにコピー"""
        try:
            for fname in self.feature_files:
                if self._stop.is_set():
                    break
                
                src = os.path.join(self.gd_base, fname)
                dst = os.path.join(self.cache_dir, fname)
                
                # 既にローカルにあればスキップ
                if not os.path.exists(dst):
                    print(f"  [Prefetch] Copying {fname}...", flush=True)
                    shutil.copy2(src, dst)
                    print(f"  [Prefetch] {fname} ready ({os.path.getsize(dst)/1e6:.0f}MB)", flush=True)
                else:
                    print(f"  [Prefetch] {fname} already cached locally", flush=True)
                
                # キューに入れる（満杯なら自然にブロック = 背圧制御）
                self.queue.put(dst)
        except Exception as e:
            # エラーでもconsumerが永久待ちしないように
            self.queue.put((_SENTINEL, e))
            return
        
        self.queue.put(_SENTINEL)
    
    def stop(self):
        """プリフェッチを停止"""
        self._stop.set()
        try:
            self.queue.put_nowait(_SENTINEL)
        except Exception:
            pass
    
    def __iter__(self):
        self.start()
        return self
    
    def __next__(self):
        item = self.queue.get()
        if item is _SENTINEL:
            raise StopIteration
        if isinstance(item, tuple) and len(item) == 2 and item[0] is _SENTINEL:
            raise RuntimeError(f"Prefetch worker failed: {item[1]}")
        
        local_path = item
        # .npy なら memmap で本当にページング可能
        arr = np.load(local_path, mmap_mode='r', allow_pickle=False)
        return local_path, arr
    
    def cleanup_file(self, local_path):
        """使用済みファイルを削除して容量回収"""
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except OSError:
            pass
    
    def cleanup_all(self):
        """ローカルキャッシュディレクトリごと削除"""
        self.stop()
        try:
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
        except OSError:
            pass


def _is_google_drive_path(path):
    """パスがGoogle Drive同期領域かどうか判定"""
    return "GoogleDrive" in path or "Google Drive" in path


def save_as_chunks(data, chunk_dir, prefix, chunk_cols=1000):
    """
    大きなnp.arrayを.npyチャンクに分割保存。
    
    Args:
        data: (N, D) のnp.array
        chunk_dir: チャンク保存先ディレクトリ
        prefix: ファイル名プレフィックス（例: 'train', 'test'）
        chunk_cols: 1チャンクあたりの列数
    
    Returns:
        chunk_files: 保存したチャンクファイル名のリスト
    """
    os.makedirs(chunk_dir, exist_ok=True)
    n_samples, n_cols = data.shape
    chunk_files = []
    
    col = 0
    chunk_idx = 0
    while col < n_cols:
        end_col = min(col + chunk_cols, n_cols)
        fname = f"{prefix}_chunk_{chunk_idx:03d}.npy"
        fpath = os.path.join(chunk_dir, fname)
        
        chunk_data = np.ascontiguousarray(data[:, col:end_col].astype(np.float32))
        np.save(fpath, chunk_data)
        chunk_files.append(fname)
        
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  Saved {fname}: cols [{col}:{end_col}] = {end_col-col}d, {size_mb:.0f}MB")
        
        col = end_col
        chunk_idx += 1
    
    # メタ情報を保存
    meta = {
        'prefix': prefix,
        'n_samples': n_samples,
        'n_cols': n_cols,
        'chunk_cols': chunk_cols,
        'n_chunks': len(chunk_files),
        'chunk_files': chunk_files,
        'dtype': str(data.dtype),
    }
    meta_path = os.path.join(chunk_dir, f"{prefix}_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    return chunk_files


def load_chunks_with_prefetch(chunk_dir, prefix, gd_base=None):
    """
    チャンクをパイプライン読み込みして結合。
    GDパスならprefetch、ローカルなら直接mmap。
    
    Returns:
        結合されたnp.array (N, D)
    """
    meta_path = os.path.join(chunk_dir if gd_base is None else gd_base, f"{prefix}_meta.json")
    
    # メタ情報がGDにある場合はまずコピー
    if gd_base and _is_google_drive_path(gd_base):
        local_meta = os.path.join('/tmp/feat_cache_prefetch', f"{prefix}_meta.json")
        os.makedirs('/tmp/feat_cache_prefetch', exist_ok=True)
        if not os.path.exists(local_meta):
            shutil.copy2(os.path.join(gd_base, f"{prefix}_meta.json"), local_meta)
        meta_path = local_meta
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    chunk_files = meta['chunk_files']
    n_samples = meta['n_samples']
    n_cols = meta['n_cols']
    
    source_dir = gd_base if gd_base else chunk_dir
    
    if _is_google_drive_path(source_dir):
        # GDパス → パイプラインプリフェッチ
        print(f"  Using prefetch pipeline for {prefix} ({len(chunk_files)} chunks from GD)")
        loader = PrefetchingFeatureLoader(
            chunk_files, gd_base=source_dir,
            cache_dir='/tmp/feat_cache_prefetch', queue_size=2
        )
        
        result = np.empty((n_samples, n_cols), dtype=np.float32)
        col = 0
        for local_path, chunk_arr in loader:
            d = chunk_arr.shape[1]
            result[:, col:col+d] = chunk_arr[:]  # mmapから実体にコピー
            col += d
            loader.cleanup_file(local_path)  # 即座に容量回収
        
        loader.cleanup_all()
        return result
    else:
        # ローカルパス → 直接mmap
        print(f"  Direct mmap load for {prefix} ({len(chunk_files)} chunks)")
        result = np.empty((n_samples, n_cols), dtype=np.float32)
        col = 0
        for fname in chunk_files:
            fpath = os.path.join(source_dir, fname)
            chunk_arr = np.load(fpath, mmap_mode='r')
            d = chunk_arr.shape[1]
            result[:, col:col+d] = chunk_arr[:]
            col += d
        return result

# ========== AST Nodes ==========

@dataclass(frozen=True)
class ASTNode:
    pass

@dataclass(frozen=True)
class Img(ASTNode):
    def __repr__(self): return "IMG"

@dataclass(frozen=True)
class R(ASTNode):
    img: ASTNode
    def __repr__(self): return f"R({self.img})"

@dataclass(frozen=True)
class G(ASTNode):
    img: ASTNode
    def __repr__(self): return f"G({self.img})"

@dataclass(frozen=True)
class B(ASTNode):
    img: ASTNode
    def __repr__(self): return f"B({self.img})"

@dataclass(frozen=True)
class Gray(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Gray({self.img})"

@dataclass(frozen=True)
class OppRG(ASTNode):
    img: ASTNode
    def __repr__(self): return f"OppRG({self.img})"

@dataclass(frozen=True)
class OppYB(ASTNode):
    img: ASTNode
    def __repr__(self): return f"OppYB({self.img})"

@dataclass(frozen=True)
class OppRB(ASTNode):
    img: ASTNode
    def __repr__(self): return f"OppRB({self.img})"

@dataclass(frozen=True)
class OppGB(ASTNode):
    img: ASTNode
    def __repr__(self): return f"OppGB({self.img})"

@dataclass(frozen=True)
class Sat(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Sat({self.img})"

@dataclass(frozen=True)
class Hue(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Hue({self.img})"

@dataclass(frozen=True)
class SatHSV(ASTNode):
    img: ASTNode
    def __repr__(self): return f"SatHSV({self.img})"

@dataclass(frozen=True)
class ValHSV(ASTNode):
    img: ASTNode
    def __repr__(self): return f"ValHSV({self.img})"

@dataclass(frozen=True)
class YCh(ASTNode):
    img: ASTNode
    def __repr__(self): return f"YCh({self.img})"

@dataclass(frozen=True)
class Cb(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Cb({self.img})"

@dataclass(frozen=True)
class Cr(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Cr({self.img})"

@dataclass(frozen=True)
class Neg(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Neg({self.img})"

@dataclass(frozen=True)
class Grad(ASTNode):
    img: ASTNode
    th: int
    mode: str  # 'dx_pos'|'dx_neg'|'dy_pos'|'dy_neg'
    def __repr__(self): return f"Grad({self.img},th={self.th},{self.mode})"

@dataclass(frozen=True)
class Threshold(ASTNode):
    img: ASTNode
    thresh: int
    def __repr__(self): return f"Threshold({self.img},{self.thresh})"

@dataclass(frozen=True)
class Edge(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Edge({self.img})"

@dataclass(frozen=True)
class CA(ASTNode):
    mask: ASTNode
    steps: int
    rule: str = "life"
    def __repr__(self): return f"CA({self.mask},{self.steps},{self.rule})"

@dataclass(frozen=True)
class GridStats(ASTNode):
    mask: ASTNode
    grid_n: int = 8
    def __repr__(self): return f"GridStats({self.mask},g{self.grid_n})"

@dataclass(frozen=True)
class Pat2x2(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Pat2x2({self.mask})"

@dataclass(frozen=True)
class Markov4(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Markov4({self.mask})"

@dataclass(frozen=True)
class Moments(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Moments({self.mask})"

@dataclass(frozen=True)
class Mean(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Mean({self.mask})"

@dataclass(frozen=True)
class Std(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Std({self.mask})"

@dataclass(frozen=True)
class Skewness(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Skew({self.mask})"

@dataclass(frozen=True)
class Kurtosis(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Kurt({self.mask})"

@dataclass(frozen=True)
class Entropy(ASTNode):
    mask: ASTNode
    def __repr__(self): return f"Entropy({self.mask})"

@dataclass(frozen=True)
class Histogram(ASTNode):
    mask: ASTNode
    bins: int = 2
    vmin: float = 0.0
    vmax: float = 1.0
    def __repr__(self): return f"Histogram({self.mask},bins={self.bins})"

@dataclass(frozen=True)
class LBP(ASTNode):
    img: ASTNode
    eps: int = 0
    def __repr__(self): return f"LBP({self.img},eps={self.eps})"

@dataclass(frozen=True)
class LBPriu2(ASTNode):
    img: ASTNode
    eps: int = 0
    def __repr__(self): return f"LBPriu2({self.img},eps={self.eps})"

@dataclass(frozen=True)
class LTP(ASTNode):
    img: ASTNode
    eps: int = 0
    def __repr__(self): return f"LTP({self.img},eps={self.eps})"

@dataclass(frozen=True)
class CLBP(ASTNode):
    img: ASTNode
    eps: int = 0
    def __repr__(self): return f"CLBP({self.img},eps={self.eps})"

@dataclass(frozen=True)
class RGBHist(ASTNode):
    img: ASTNode
    bins: int = 4
    def __repr__(self): return f"RGBHist({self.img},bins={self.bins})"

@dataclass(frozen=True)
class RGBBlocks(ASTNode):
    img: ASTNode
    blocks: int = 4
    def __repr__(self): return f"RGBBlocks({self.img},b={self.blocks})"

@dataclass(frozen=True)
class Concat(ASTNode):
    left: ASTNode
    right: ASTNode
    def __repr__(self): return f"Concat({self.left},{self.right})"

# ========== Tokenization + Reward Logging ==========

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _num_token(value: int) -> str:
    return f"NUM_{int(value)}"

def ast_to_tokens(node: ASTNode) -> list[str]:
    """Deterministic tokenization (prefix with explicit delimiters)."""
    if isinstance(node, Img):
        return ["IMG"]
    if isinstance(node, R):
        return ["R", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, G):
        return ["G", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, B):
        return ["B", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, Gray):
        return ["GRAY", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, OppRG):
        return ["OPPRG", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, OppYB):
        return ["OPPYB", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, OppRB):
        return ["OPPRB", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, OppGB):
        return ["OPPGB", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, Sat):
        return ["SAT", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, Hue):
        return ["HUE", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, SatHSV):
        return ["HSV_S", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, ValHSV):
        return ["HSV_V", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, YCh):
        return ["YCH", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, Cb):
        return ["CB", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, Cr):
        return ["CR", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, Neg):
        return ["NEG", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, Grad):
        mode_tok = f"MODE_{str(node.mode).upper()}"
        return ["GRAD", "(", *ast_to_tokens(node.img), "TH", _num_token(node.th), "MODE", mode_tok, ")"]
    if isinstance(node, Threshold):
        return ["THRESHOLD", "(", *ast_to_tokens(node.img), "TH", _num_token(node.thresh), ")"]
    if isinstance(node, Edge):
        return ["EDGE", "(", *ast_to_tokens(node.img), ")"]
    if isinstance(node, CA):
        rule_tok = f"RULE_{str(node.rule).upper()}"
        return ["CA", "(", *ast_to_tokens(node.mask), "STEPS", _num_token(node.steps), "RULE", rule_tok, ")"]
    if isinstance(node, GridStats):
        return ["GRIDSTATS", "(", *ast_to_tokens(node.mask), "G", _num_token(node.grid_n), ")"]
    if isinstance(node, Pat2x2):
        return ["PAT2X2", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Markov4):
        return ["MARKOV4", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Moments):
        return ["MOMENTS", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Mean):
        return ["MEAN", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Std):
        return ["STD", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Skewness):
        return ["SKEW", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Kurtosis):
        return ["KURT", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Entropy):
        return ["ENTROPY", "(", *ast_to_tokens(node.mask), ")"]
    if isinstance(node, Histogram):
        return ["HIST", "(", *ast_to_tokens(node.mask), "BINS", _num_token(node.bins), ")"]
    if isinstance(node, LBP):
        return ["LBP", "(", *ast_to_tokens(node.img), "EPS", _num_token(node.eps), ")"]
    if isinstance(node, LBPriu2):
        return ["LBPRIU2", "(", *ast_to_tokens(node.img), "EPS", _num_token(node.eps), ")"]
    if isinstance(node, LTP):
        return ["LTP", "(", *ast_to_tokens(node.img), "EPS", _num_token(node.eps), ")"]
    if isinstance(node, CLBP):
        return ["CLBP", "(", *ast_to_tokens(node.img), "EPS", _num_token(node.eps), ")"]
    if isinstance(node, RGBHist):
        return ["RGBHIST", "(", *ast_to_tokens(node.img), "BINS", _num_token(node.bins), ")"]
    if isinstance(node, RGBBlocks):
        return ["RGBBLOCKS", "(", *ast_to_tokens(node.img), "BLOCKS", _num_token(node.blocks), ")"]
    if isinstance(node, Concat):
        return ["CONCAT", "(", *ast_to_tokens(node.left), ",", *ast_to_tokens(node.right), ")"]
    raise TypeError(f"Unsupported AST node: {type(node)}")

def build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    uniq = sorted({tok for tokens in token_lists for tok in tokens})
    return {tok: i + 1 for i, tok in enumerate(uniq)}  # reserve 0 for PAD if needed

def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    return [int(vocab[tok]) for tok in tokens]

class RewardLogger:
    def __init__(self, path: str, run_id: str, vocab: dict[str, int], args: dict, program_count: int):
        self.path = path
        self.run_id = run_id
        self._fh = None
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8")
        self._write({
            "type": "meta",
            "run_id": run_id,
            "timestamp": _utc_now(),
            "vocab": vocab,
            "args": args,
            "program_count": program_count,
        })

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._fh.flush()

    def log_program(self, index: int, node: ASTNode, tokens: list[str], token_ids: list[int], dim: int, reward: float | None) -> None:
        self._write({
            "type": "program",
            "run_id": self.run_id,
            "index": int(index),
            "repr": repr(node),
            "tokens": tokens,
            "token_ids": token_ids,
            "dim": int(dim),
            "reward": None if reward is None else float(reward),
        })

    def log_summary(self, summary: dict) -> None:
        payload = {"type": "summary", "run_id": self.run_id, "timestamp": _utc_now()}
        payload.update(summary)
        self._write(payload)

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

# ========== Evaluator ==========

class FullEvaluator:
    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale"""
        if len(img.shape) == 2:
            return img.astype(np.float32)
        return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.float32)

    def _rgb_to_hsv(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB (0..255) -> HSV scaled to 0..255"""
        r = rgb[..., 0].astype(np.float32) / 255.0
        g = rgb[..., 1].astype(np.float32) / 255.0
        b = rgb[..., 2].astype(np.float32) / 255.0
        mx = np.maximum.reduce([r, g, b])
        mn = np.minimum.reduce([r, g, b])
        diff = mx - mn

        h = np.zeros_like(mx)
        mask = diff > 1e-8
        r_mask = (mx == r) & mask
        g_mask = (mx == g) & mask
        b_mask = (mx == b) & mask
        h[r_mask] = ((g[r_mask] - b[r_mask]) / diff[r_mask]) % 6.0
        h[g_mask] = ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 2.0
        h[b_mask] = ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 4.0
        h = (h / 6.0)  # 0..1
        s = np.where(mx > 1e-8, diff / (mx + 1e-8), 0.0)
        v = mx
        return (h * 255.0).astype(np.float32), (s * 255.0).astype(np.float32), (v * 255.0).astype(np.float32)

    def _rgb_to_ycbcr(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RGB (0..255) -> YCbCr (0..255)"""
        r = rgb[..., 0].astype(np.float32)
        g = rgb[..., 1].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return y.astype(np.float32), cb.astype(np.float32), cr.astype(np.float32)
    
    def evaluate(self, node: ASTNode, img: np.ndarray) -> np.ndarray:
        if isinstance(node, Img):
            return img.astype(np.float32)
        
        elif isinstance(node, R):
            rgb = self.evaluate(node.img, img)
            return rgb[:,:,0]
        
        elif isinstance(node, G):
            rgb = self.evaluate(node.img, img)
            return rgb[:,:,1]
        
        elif isinstance(node, B):
            rgb = self.evaluate(node.img, img)
            return rgb[:,:,2]
        
        elif isinstance(node, Gray):
            rgb = self.evaluate(node.img, img)
            # mymodel2-style integer grayscale (approx ITU-R BT.601)
            r = rgb[:, :, 0].astype(np.uint16)
            g = rgb[:, :, 1].astype(np.uint16)
            b = rgb[:, :, 2].astype(np.uint16)
            y = (77 * r + 150 * g + 29 * b) >> 8
            return y.astype(np.float32)
        
        elif isinstance(node, OppRG):
            rgb = self.evaluate(node.img, img)
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            return (r - g).astype(np.float32)
        
        elif isinstance(node, OppYB):
            rgb = self.evaluate(node.img, img)
            # mymodel2-style: yb = 2*B - R - G
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)
            return (2 * b - r - g).astype(np.float32)

        elif isinstance(node, OppRB):
            rgb = self.evaluate(node.img, img)
            r = rgb[:, :, 0].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)
            return (r - b).astype(np.float32)

        elif isinstance(node, OppGB):
            rgb = self.evaluate(node.img, img)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)
            return (g - b).astype(np.float32)
        
        elif isinstance(node, Sat):
            rgb = self.evaluate(node.img, img)
            # mymodel2-style saturation proxy: max(R,G,B) - min(R,G,B) in [0..255]
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)
            mx = np.maximum.reduce([r, g, b])
            mn = np.minimum.reduce([r, g, b])
            return (mx - mn).astype(np.float32)

        elif isinstance(node, Hue):
            rgb = self.evaluate(node.img, img)
            h, _, _ = self._rgb_to_hsv(rgb)
            return h

        elif isinstance(node, SatHSV):
            rgb = self.evaluate(node.img, img)
            _, s, _ = self._rgb_to_hsv(rgb)
            return s

        elif isinstance(node, ValHSV):
            rgb = self.evaluate(node.img, img)
            _, _, v = self._rgb_to_hsv(rgb)
            return v

        elif isinstance(node, YCh):
            rgb = self.evaluate(node.img, img)
            y, _, _ = self._rgb_to_ycbcr(rgb)
            return y

        elif isinstance(node, Cb):
            rgb = self.evaluate(node.img, img)
            _, cb, _ = self._rgb_to_ycbcr(rgb)
            return cb

        elif isinstance(node, Cr):
            rgb = self.evaluate(node.img, img)
            _, _, cr = self._rgb_to_ycbcr(rgb)
            return cr

        elif isinstance(node, Neg):
            ch = self.evaluate(node.img, img)
            return (-ch).astype(np.float32)
        
        elif isinstance(node, Threshold):
            channel = self.evaluate(node.img, img)
            return (channel >= node.thresh).astype(np.uint8)

        elif isinstance(node, Grad):
            g = self.evaluate(node.img, img).astype(np.int16)
            th = int(node.th)
            dx = np.zeros((32, 32), dtype=np.int16)
            dy = np.zeros((32, 32), dtype=np.int16)
            dx[:, :-1] = g[:, 1:] - g[:, :-1]
            dy[:-1, :] = g[1:, :] - g[:-1, :]
            if node.mode == "dx_pos":
                return (dx > th).astype(np.uint8)
            if node.mode == "dx_neg":
                return (dx < -th).astype(np.uint8)
            if node.mode == "dy_pos":
                return (dy > th).astype(np.uint8)
            if node.mode == "dy_neg":
                return (dy < -th).astype(np.uint8)
            raise ValueError(f"Unknown Grad mode: {node.mode}")
        
        elif isinstance(node, Edge):
            # mymodel2-style edge map on a binary mask: dilate4 XOR erode4
            B01 = self.evaluate(node.img, img).astype(np.uint8)
            up = np.zeros_like(B01); up[1:, :] = B01[:-1, :]
            dn = np.zeros_like(B01); dn[:-1, :] = B01[1:, :]
            lf = np.zeros_like(B01); lf[:, 1:] = B01[:, :-1]
            rt = np.zeros_like(B01); rt[:, :-1] = B01[:, 1:]
            dil = np.maximum.reduce([B01, up, dn, lf, rt]).astype(np.uint8)

            up = np.ones_like(B01); up[1:, :] = B01[:-1, :]
            dn = np.ones_like(B01); dn[:-1, :] = B01[1:, :]
            lf = np.ones_like(B01); lf[:, 1:] = B01[:, :-1]
            rt = np.ones_like(B01); rt[:, :-1] = B01[:, 1:]
            ero = np.minimum.reduce([B01, up, dn, lf, rt]).astype(np.uint8)

            return (dil ^ ero).astype(np.uint8)
        
        elif isinstance(node, CA):
            mask = self.evaluate(node.mask, img)
            
            if node.rule == "life":
                kernel = np.array([[1,1,1],[1,0,1],[1,1,1]])
                for _ in range(node.steps):
                    neighbors = convolve(mask, kernel, mode='constant')
                    mask = ((mask == 1) & ((neighbors == 2) | (neighbors == 3))) | \
                           ((mask == 0) & (neighbors == 3))
                    mask = mask.astype(np.uint8)
            
            elif node.rule == "erosion":
                kernel = np.ones((3,3), dtype=np.uint8)
                for _ in range(node.steps):
                    neighbors = convolve(mask, kernel, mode='constant')
                    mask = (neighbors == 9).astype(np.uint8)
            
            elif node.rule == "dilation":
                kernel = np.ones((3,3), dtype=np.uint8)
                for _ in range(node.steps):
                    neighbors = convolve(mask, kernel, mode='constant')
                    mask = (neighbors > 0).astype(np.uint8)
            
            elif node.rule == "opening":
                kernel = np.ones((3,3), dtype=np.uint8)
                for _ in range(node.steps):
                    neighbors = convolve(mask, kernel, mode='constant')
                    mask = (neighbors == 9).astype(np.uint8)
                for _ in range(node.steps):
                    neighbors = convolve(mask, kernel, mode='constant')
                    mask = (neighbors > 0).astype(np.uint8)
            
            elif node.rule == "closing":
                kernel = np.ones((3,3), dtype=np.uint8)
                for _ in range(node.steps):
                    neighbors = convolve(mask, kernel, mode='constant')
                    mask = (neighbors > 0).astype(np.uint8)
                for _ in range(node.steps):
                    neighbors = convolve(mask, kernel, mode='constant')
                    mask = (neighbors == 9).astype(np.uint8)
            
            return mask
        
        elif isinstance(node, GridStats):
            mask = self.evaluate(node.mask, img)
            return self._grid_stats(mask, node.grid_n)
        
        elif isinstance(node, Pat2x2):
            mask = self.evaluate(node.mask, img)
            return self._pat2x2(mask)
        
        elif isinstance(node, Markov4):
            mask = self.evaluate(node.mask, img)
            return self._markov4(mask)
        
        elif isinstance(node, Moments):
            mask = self.evaluate(node.mask, img)
            flat = mask.flatten()
            return np.array([
                np.mean(flat),
                np.std(flat),
                skew(flat),
                kurtosis(flat)
            ], dtype=np.float32)

        elif isinstance(node, Mean):
            mask = self.evaluate(node.mask, img)
            flat = mask.flatten()
            return np.array([np.mean(flat)], dtype=np.float32)

        elif isinstance(node, Std):
            mask = self.evaluate(node.mask, img)
            flat = mask.flatten()
            return np.array([np.std(flat)], dtype=np.float32)

        elif isinstance(node, Skewness):
            mask = self.evaluate(node.mask, img)
            flat = mask.flatten()
            val = skew(flat)
            return np.array([np.nan_to_num(val)], dtype=np.float32)

        elif isinstance(node, Kurtosis):
            mask = self.evaluate(node.mask, img)
            flat = mask.flatten()
            val = kurtosis(flat)
            return np.array([np.nan_to_num(val)], dtype=np.float32)

        elif isinstance(node, Entropy):
            mask = self.evaluate(node.mask, img)
            flat = mask.ravel()
            if flat.size == 0:
                return np.array([0.0], dtype=np.float32)
            if flat.min() >= 0.0 and flat.max() <= 1.0:
                cnt1 = int((flat > 0.5).sum())
                cnt0 = int(flat.size - cnt1)
                total = float(flat.size)
                p0 = cnt0 / total
                p1 = cnt1 / total
                ent = 0.0
                if p0 > 0:
                    ent -= p0 * np.log2(p0)
                if p1 > 0:
                    ent -= p1 * np.log2(p1)
                return np.array([ent], dtype=np.float32)
            hist, _ = np.histogram(flat, bins=16)
            total = float(hist.sum()) + 1e-9
            p = hist / total
            ent = -np.sum(p * np.log2(p + 1e-12))
            return np.array([ent], dtype=np.float32)

        elif isinstance(node, Histogram):
            mask = self.evaluate(node.mask, img)
            flat = mask.ravel()
            if node.bins == 2 and flat.min() >= 0.0 and flat.max() <= 1.0:
                cnt1 = int((flat > 0.5).sum())
                cnt0 = int(flat.size - cnt1)
                return np.array([cnt0, cnt1], dtype=np.int16)
            hist, _ = np.histogram(flat, bins=node.bins, range=(node.vmin, node.vmax))
            return hist.astype(np.int16)
        
        elif isinstance(node, LBP):
            gray = self._to_gray(img)
            return self._lbp_hist8(gray, eps=node.eps)

        elif isinstance(node, LBPriu2):
            gray = self._to_gray(img)
            return self._lbp_riu2_hist(gray, eps=node.eps)

        elif isinstance(node, LTP):
            gray = self._to_gray(img)
            return self._ltp_hist8(gray, eps=node.eps)

        elif isinstance(node, CLBP):
            gray = self._to_gray(img)
            return self._clbp_hist8(gray, eps=node.eps)
        
        elif isinstance(node, RGBHist):
            return self._rgb_coarse_hist(img, bins=node.bins)
        
        elif isinstance(node, RGBBlocks):
            return self._rgb_block_means(img, blocks=node.blocks)
        
        elif isinstance(node, Concat):
            left = self.evaluate(node.left, img)
            right = self.evaluate(node.right, img)
            return np.concatenate([left, right])
        
        else:
            raise ValueError(f"Unknown node: {type(node)}")
    
    def _grid_stats(self, B01: np.ndarray, grid_n: int) -> np.ndarray:
        h, w = B01.shape
        grid_h, grid_w = h // grid_n, w // grid_n

        # mymodel2-style: use counts (not normalized)
        B = B01.astype(np.uint16, copy=False)
        cnt = float(B.sum())
        
        row_proj = []
        for i in range(grid_n):
            row_sum = B[i*grid_h:(i+1)*grid_h, :].sum()
            row_proj.append(float(row_sum))
        
        col_proj = []
        for j in range(grid_n):
            col_sum = B[:, j*grid_w:(j+1)*grid_w].sum()
            col_proj.append(float(col_sum))
        
        grid = []
        for i in range(grid_n):
            for j in range(grid_n):
                cell_sum = B[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w].sum()
                grid.append(float(cell_sum))
        
        return np.array([cnt] + row_proj + col_proj + grid, dtype=np.float32)
    
    def _pat2x2(self, B01: np.ndarray) -> np.ndarray:
        # mymodel2-style overlapping 2x2 histogram
        B = (B01 > 0.5).astype(np.uint8)
        if B.shape[0] < 2 or B.shape[1] < 2:
            return np.zeros(16, dtype=np.float32)
        a = B[:-1, :-1]
        b = B[:-1, 1:]
        c = B[1:, :-1]
        d = B[1:, 1:]
        code = (a | (b << 1) | (c << 2) | (d << 3)).ravel().astype(np.int32)
        h = np.bincount(code, minlength=16).astype(np.int16)
        return h
    
    def _markov4(self, B01: np.ndarray) -> np.ndarray:
        # mymodel2-style Markov transitions in 4 directions, 4 bins each => 16 dims
        B = (B01 > 0.5).astype(np.uint8)
        out = []
        A = B[:, :-1].ravel(); C = B[:, 1:].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        A = B[:-1, :].ravel(); C = B[1:, :].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        A = B[:-1, :-1].ravel(); C = B[1:, 1:].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        A = B[:-1, 1:].ravel(); C = B[1:, :-1].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        h = np.concatenate(out, axis=0).astype(np.int16)
        return h
    
    def _lbp_hist8(self, gray: np.ndarray, eps: int = 0) -> np.ndarray:
        """LBP (Local Binary Pattern) - mymodel2 implementation"""
        g = gray.astype(np.int16)
        c = g[1:-1, 1:-1]
        E = g[1:-1, 2:]; NE = g[0:-2, 2:]; N = g[0:-2, 1:-1]; NW = g[0:-2, 0:-2]
        W = g[1:-1, 0:-2]; SW = g[2:, 0:-2]; S = g[2:, 1:-1]; SE = g[2:, 2:]
        
        code = np.zeros_like(c, dtype=np.uint8)
        code |= ((E >= c + eps) << 0).astype(np.uint8)
        code |= ((NE >= c + eps) << 1).astype(np.uint8)
        code |= ((N >= c + eps) << 2).astype(np.uint8)
        code |= ((NW >= c + eps) << 3).astype(np.uint8)
        code |= ((W >= c + eps) << 4).astype(np.uint8)
        code |= ((SW >= c + eps) << 5).astype(np.uint8)
        code |= ((S >= c + eps) << 6).astype(np.uint8)
        code |= ((SE >= c + eps) << 7).astype(np.uint8)
        
        v = code.ravel()
        h = np.bincount(v, minlength=256).astype(np.int16)
        return h

    def _lbp_riu2_hist(self, gray: np.ndarray, eps: int = 0) -> np.ndarray:
        """LBP riu2 (P=8) - 10 bins"""
        g = gray.astype(np.int16)
        c = g[1:-1, 1:-1]
        E = g[1:-1, 2:]; NE = g[0:-2, 2:]; N = g[0:-2, 1:-1]; NW = g[0:-2, 0:-2]
        W = g[1:-1, 0:-2]; SW = g[2:, 0:-2]; S = g[2:, 1:-1]; SE = g[2:, 2:]
        code = np.zeros_like(c, dtype=np.uint8)
        code |= ((E >= c + eps) << 0).astype(np.uint8)
        code |= ((NE >= c + eps) << 1).astype(np.uint8)
        code |= ((N >= c + eps) << 2).astype(np.uint8)
        code |= ((NW >= c + eps) << 3).astype(np.uint8)
        code |= ((W >= c + eps) << 4).astype(np.uint8)
        code |= ((SW >= c + eps) << 5).astype(np.uint8)
        code |= ((S >= c + eps) << 6).astype(np.uint8)
        code |= ((SE >= c + eps) << 7).astype(np.uint8)

        v = code.ravel().astype(np.uint8)
        # build mapping table lazily
        if not hasattr(self, "_lbp_riu2_map"):
            m = np.zeros(256, dtype=np.uint8)
            for ccode in range(256):
                bits = [(ccode >> i) & 1 for i in range(8)]
                transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
                if transitions <= 2:
                    m[ccode] = sum(bits)
                else:
                    m[ccode] = 9
            self._lbp_riu2_map = m
        mapped = self._lbp_riu2_map[v]
        h = np.bincount(mapped, minlength=10).astype(np.int16)
        return h

    def _ltp_hist8(self, gray: np.ndarray, eps: int = 1) -> np.ndarray:
        """LTP (Local Ternary Pattern) - split into positive/negative LBP histograms"""
        g = gray.astype(np.int16)
        c = g[1:-1, 1:-1]
        E = g[1:-1, 2:]; NE = g[0:-2, 2:]; N = g[0:-2, 1:-1]; NW = g[0:-2, 0:-2]
        W = g[1:-1, 0:-2]; SW = g[2:, 0:-2]; S = g[2:, 1:-1]; SE = g[2:, 2:]

        code_pos = np.zeros_like(c, dtype=np.uint8)
        code_neg = np.zeros_like(c, dtype=np.uint8)
        code_pos |= ((E >= c + eps) << 0).astype(np.uint8)
        code_pos |= ((NE >= c + eps) << 1).astype(np.uint8)
        code_pos |= ((N >= c + eps) << 2).astype(np.uint8)
        code_pos |= ((NW >= c + eps) << 3).astype(np.uint8)
        code_pos |= ((W >= c + eps) << 4).astype(np.uint8)
        code_pos |= ((SW >= c + eps) << 5).astype(np.uint8)
        code_pos |= ((S >= c + eps) << 6).astype(np.uint8)
        code_pos |= ((SE >= c + eps) << 7).astype(np.uint8)

        code_neg |= ((E <= c - eps) << 0).astype(np.uint8)
        code_neg |= ((NE <= c - eps) << 1).astype(np.uint8)
        code_neg |= ((N <= c - eps) << 2).astype(np.uint8)
        code_neg |= ((NW <= c - eps) << 3).astype(np.uint8)
        code_neg |= ((W <= c - eps) << 4).astype(np.uint8)
        code_neg |= ((SW <= c - eps) << 5).astype(np.uint8)
        code_neg |= ((S <= c - eps) << 6).astype(np.uint8)
        code_neg |= ((SE <= c - eps) << 7).astype(np.uint8)

        h_pos = np.bincount(code_pos.ravel(), minlength=256).astype(np.int16)
        h_neg = np.bincount(code_neg.ravel(), minlength=256).astype(np.int16)
        return np.concatenate([h_pos, h_neg], axis=0).astype(np.int16)

    def _clbp_hist8(self, gray: np.ndarray, eps: int = 0) -> np.ndarray:
        """CLBP (sign + magnitude + center) histogram"""
        g = gray.astype(np.int16)
        c = g[1:-1, 1:-1]
        E = g[1:-1, 2:]; NE = g[0:-2, 2:]; N = g[0:-2, 1:-1]; NW = g[0:-2, 0:-2]
        W = g[1:-1, 0:-2]; SW = g[2:, 0:-2]; S = g[2:, 1:-1]; SE = g[2:, 2:]

        code_s = np.zeros_like(c, dtype=np.uint8)
        code_s |= ((E >= c + eps) << 0).astype(np.uint8)
        code_s |= ((NE >= c + eps) << 1).astype(np.uint8)
        code_s |= ((N >= c + eps) << 2).astype(np.uint8)
        code_s |= ((NW >= c + eps) << 3).astype(np.uint8)
        code_s |= ((W >= c + eps) << 4).astype(np.uint8)
        code_s |= ((SW >= c + eps) << 5).astype(np.uint8)
        code_s |= ((S >= c + eps) << 6).astype(np.uint8)
        code_s |= ((SE >= c + eps) << 7).astype(np.uint8)
        h_s = np.bincount(code_s.ravel(), minlength=256).astype(np.int16)

        diffs = np.stack([
            np.abs(E - c), np.abs(NE - c), np.abs(N - c), np.abs(NW - c),
            np.abs(W - c), np.abs(SW - c), np.abs(S - c), np.abs(SE - c)
        ], axis=0)
        th = float(np.mean(diffs))
        code_m = np.zeros_like(c, dtype=np.uint8)
        code_m |= ((np.abs(E - c) >= th) << 0).astype(np.uint8)
        code_m |= ((np.abs(NE - c) >= th) << 1).astype(np.uint8)
        code_m |= ((np.abs(N - c) >= th) << 2).astype(np.uint8)
        code_m |= ((np.abs(NW - c) >= th) << 3).astype(np.uint8)
        code_m |= ((np.abs(W - c) >= th) << 4).astype(np.uint8)
        code_m |= ((np.abs(SW - c) >= th) << 5).astype(np.uint8)
        code_m |= ((np.abs(S - c) >= th) << 6).astype(np.uint8)
        code_m |= ((np.abs(SE - c) >= th) << 7).astype(np.uint8)
        h_m = np.bincount(code_m.ravel(), minlength=256).astype(np.int16)

        c_mean = float(np.mean(g))
        c_bin = (c >= c_mean).astype(np.uint8)
        h_c = np.bincount(c_bin.ravel(), minlength=2).astype(np.int16)
        return np.concatenate([h_s, h_m, h_c], axis=0).astype(np.int16)
    
    def _rgb_coarse_hist(self, img_u8: np.ndarray, bins: int = 4) -> np.ndarray:
        """RGB histogram - mymodel2 implementation"""
        if bins <= 1:
            return np.array([img_u8.shape[0] * img_u8.shape[1]], dtype=np.int16)
        r = (img_u8[..., 0].astype(np.uint16) * bins) >> 8
        g = (img_u8[..., 1].astype(np.uint16) * bins) >> 8
        b = (img_u8[..., 2].astype(np.uint16) * bins) >> 8
        idx = (r * (bins * bins) + g * bins + b).ravel().astype(np.int32)
        h = np.bincount(idx, minlength=bins * bins * bins).astype(np.int16)
        return h
    
    def _rgb_block_means(self, img_u8: np.ndarray, blocks: int = 4) -> np.ndarray:
        """RGB block means - mymodel2 implementation"""
        B = blocks
        k = 32 // B
        x = img_u8.reshape(B, k, B, k, 3).mean(axis=(1, 3))
        return np.round(x).reshape(-1).astype(np.int16)


def diag_scale(X, eps=1e-6, scale_factor=16.0, use_var=True, scale_vec=None, inplace=False):
    """mymodel2-style diagonal scaling (float32-optimized, supports in-place)"""
    if not inplace:
        X = X.astype(np.float32)
    if scale_vec is not None:
        scale = scale_vec
    elif use_var:
        var = np.var(X, axis=0, dtype=np.float32) + np.float32(eps)
        scale = np.sqrt(var) / np.float32(scale_factor)
    else:
        std = np.std(X, axis=0, dtype=np.float32) + np.float32(eps)
        scale = std / np.float32(scale_factor)
    if inplace:
        X /= scale
        return X, scale
    else:
        return (X / scale).astype(np.float32), scale


if __name__ == "__main__":
    from mymodel3 import load_cifar10_numpy
    from lightgbm import LGBMClassifier
    import time

    parser = argparse.ArgumentParser(description="Feature Grammar V7 (reward-ready, based on V6)")
    parser.add_argument("--data-dir", type=str, default="./cifar10_data")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--subset-train", type=int, default=0, help="0=full, else number of train samples")
    parser.add_argument("--subset-test", type=int, default=0, help="0=full, else number of test samples")
    parser.add_argument("--max-programs", type=int, default=0, help="0=all, else use first N programs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run a small stratified subset for a fast sanity check")
    parser.add_argument("--reward-out", type=str, default="reward_logs/fgv7_reward.jsonl", help="JSONL reward log path")
    parser.add_argument("--reward-metric", type=str, default="var", choices=["none", "var", "mi"], help="Program reward metric")
    parser.add_argument("--reward-subsample", type=int, default=5000, help="Subsample size for reward metric (0=full)")
    parser.add_argument("--reward-seed", type=int, default=123, help="Seed for reward subsampling")
    parser.add_argument("--no-reward-log", action="store_true", help="Disable reward logging")
    parser.add_argument("--resume", action="store_true", help="Resume from cached features (skip extraction)")
    parser.add_argument("--feat-cache", type=str, default="/Users/yuta/Library/CloudStorage/GoogleDrive-imtceed@gmail.com/マイドライブ/cifar10_cache", help="Directory for feature cache .npy files")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=Warning, module=r"mymodel3")
    
    print("=" * 70)
    print("Feature Grammar V7 - Reward-ready (V6 base)")
    print("=" * 70)
    print(f"Args: {args}\n")
    
    if args.quick:
        args.subset_train = args.subset_train or 5000
        args.subset_test = args.subset_test or 2000
        print(f"Quick mode enabled: subset_train={args.subset_train}, subset_test={args.subset_test}")

    # Load data
    Xtr, ytr, Xte, yte = load_cifar10_numpy(args.data_dir)

    # Optional stratified subsetting
    if args.subset_train and args.subset_train < len(Xtr):
        from sklearn.model_selection import train_test_split
        Xtr, _, ytr, _ = train_test_split(
            Xtr, ytr,
            train_size=args.subset_train,
            random_state=args.seed,
            stratify=ytr,
        )
    if args.subset_test and args.subset_test < len(Xte):
        from sklearn.model_selection import train_test_split
        Xte, _, yte, _ = train_test_split(
            Xte, yte,
            train_size=args.subset_test,
            random_state=args.seed,
            stratify=yte,
        )

    print(f"Dataset: Train={len(Xtr)}, Test={len(Xte)}\n")
    
    evaluator = FullEvaluator()
    
    # Auto-generate programs combinatorially
    img = Img()
    
    # Build mymodel2-like binary masks, then apply per-mask statistics.
    # Key fix: thresholds must match channel scales (gray 0..255, rg ~[-255..255], yb ~[-510..510], sat 0..255)
    g = Gray(img)
    rg = OppRG(img)
    yb = OppYB(img)
    rb = OppRB(img)
    gb = OppGB(img)
    sat = Sat(img)
    h = Hue(img)
    hsv_s = SatHSV(img)
    hsv_v = ValHSV(img)
    ych = YCh(img)
    cb = Cb(img)
    cr = Cr(img)

    gray_thresholds = [40, 60, 80, 100, 120, 140]  # mymodel2 baseline
    rg_tpos = [20, 50, 80]  # mymodel2 baseline
    yb_tpos = [20, 50, 80]  # mymodel2 uses 2*th for yb
    rb_tpos = [10, 25, 40, 55, 70, 85]
    gb_tpos = [10, 25, 40, 55, 70, 85]
    sat_th = [30, 60, 90]  # mymodel2 baseline
    h_th = [32, 64, 96, 128]
    s_th = [32, 64, 96, 128]
    v_th = [32, 64, 96, 128]
    y_th = [40, 80, 120, 160]
    cb_th = [80, 100, 120, 140]
    cr_th = [80, 100, 120, 140]
    grad_ths = [8, 12]

    masks = []
    # gray masks + edges
    for th in gray_thresholds:
        m = Threshold(g, th)
        masks.append(m)
        masks.append(Edge(m))

    # grad direction masks (4 dirs)
    for th in grad_ths:
        masks.append(Grad(g, th, "dx_pos"))
        masks.append(Grad(g, th, "dx_neg"))
        masks.append(Grad(g, th, "dy_pos"))
        masks.append(Grad(g, th, "dy_neg"))

    # opponent masks (pos/neg)
    for th in rg_tpos:
        masks.append(Threshold(rg, th))
        masks.append(Threshold(Neg(rg), th))
    for th in yb_tpos:
        t2 = 2 * th
        masks.append(Threshold(yb, t2))
        masks.append(Threshold(Neg(yb), t2))
    for th in rb_tpos:
        masks.append(Threshold(rb, th))
        masks.append(Threshold(Neg(rb), th))
    for th in gb_tpos:
        masks.append(Threshold(gb, th))
        masks.append(Threshold(Neg(gb), th))

    # saturation masks
    for th in sat_th:
        masks.append(Threshold(sat, th))

    # HSV thresholds
    for th in h_th:
        masks.append(Threshold(h, th))
    for th in s_th:
        masks.append(Threshold(hsv_s, th))
    for th in v_th:
        masks.append(Threshold(hsv_v, th))

    # YCbCr thresholds
    for th in y_th:
        masks.append(Threshold(ych, th))
    for th in cb_th:
        masks.append(Threshold(cb, th))
    for th in cr_th:
        masks.append(Threshold(cr, th))

    # per-mask statistics (mymodel2-style core)
    grid_sizes = [4, 8, 12, 16]
    stats_ops = []
    for gsz in grid_sizes:
        stats_ops.append((f"GridStats{gsz}", lambda x, g=gsz: GridStats(x, grid_n=g)))
    stats_ops.extend([
        ("Pat2x2", Pat2x2),
        ("Markov4", Markov4),
        ("Moments", Moments),
        ("Mean", Mean),
        ("Std", Std),
        ("Skew", Skewness),
        ("Kurt", Kurtosis),
        ("Entropy", Entropy),
        ("Histogram", lambda x: Histogram(x, bins=2, vmin=0.0, vmax=1.0)),
    ])

    programs = []
    for m in masks:
        for _, stat_fn in stats_ops:
            programs.append(stat_fn(m))

    # Add mymodel2-style global features + variants
    for eps in [0, 1, 2]:
        programs.append(LBP(g, eps=eps))               # 256 dims
        programs.append(LBPriu2(g, eps=eps))           # 10 dims
        programs.append(LTP(g, eps=eps))               # 512 dims
        programs.append(CLBP(g, eps=eps))              # 514 dims
    programs.append(RGBHist(img, bins=4))              # 64 dims
    programs.append(RGBHist(img, bins=8))              # 512 dims
    programs.append(RGBBlocks(img, blocks=4))          # 48 dims
    programs.append(RGBBlocks(img, blocks=8))          # 192 dims

    if args.max_programs and args.max_programs < len(programs):
        programs = programs[: args.max_programs]

    reward_logger = None
    token_lists = []
    token_ids_list = []
    vocab = {}
    reward_out = "" if args.no_reward_log else (args.reward_out or "")
    if reward_out:
        token_lists = [ast_to_tokens(p) for p in programs]
        vocab = build_vocab(token_lists)
        token_ids_list = [tokens_to_ids(toks, vocab) for toks in token_lists]
        run_id = uuid.uuid4().hex
        reward_logger = RewardLogger(
            path=reward_out,
            run_id=run_id,
            vocab=vocab,
            args=vars(args),
            program_count=len(programs),
        )

    per_mask_programs = len(masks) * len(stats_ops)
    global_programs = len(programs) - per_mask_programs
    print(f"Using {len(programs)} auto-generated feature programs\n")
    print(
        f"  Masks: {len(masks)} (gray+edge={2*len(gray_thresholds)}, grad={4*len(grad_ths)}, "
        f"rg=2*{len(rg_tpos)}, yb=2*{len(yb_tpos)}, rb=2*{len(rb_tpos)}, gb=2*{len(gb_tpos)}, "
        f"sat={len(sat_th)}, hsv={len(h_th)+len(s_th)+len(v_th)}, ycbcr={len(y_th)+len(cb_th)+len(cr_th)})"
    )
    print(f"  Per-mask stats: {len(stats_ops)} (grid sizes={grid_sizes}) => {len(masks)}×{len(stats_ops)} = {per_mask_programs} programs")
    print(f"  Global features: {global_programs} programs (LBP/LTP/CLBP/riu2 + RGB hist/blocks variants)")
    print(f"  Total: {len(programs)} programs\n")
    
    # --- Feature extraction (with cache support + GD prefetch pipeline) ---
    import os as _os
    cache_dir = args.feat_cache
    is_gd = _is_google_drive_path(cache_dir)
    
    # チャンクベースのキャッシュ検出
    chunk_meta_train = _os.path.join(cache_dir, "train_meta.json")
    chunk_meta_test = _os.path.join(cache_dir, "test_meta.json")
    has_chunk_cache = args.resume and _os.path.exists(chunk_meta_train) and _os.path.exists(chunk_meta_test)
    
    # 旧形式（単一.npy）の検出
    cache_train = _os.path.join(cache_dir, "feats_train.npy")
    cache_test = _os.path.join(cache_dir, "feats_test.npy")
    have_train_cache = args.resume and _os.path.exists(cache_train)
    have_test_cache = args.resume and _os.path.exists(cache_test)

    def _extract_test_only(programs, Xte, evaluator, n_jobs):
        """Extract test features only (much faster: 10K vs 50K)"""
        print(f"Extracting TEST features only ({len(Xte)} samples, n_jobs={n_jobs})...")
        def _ext_te(prog):
            return np.array([evaluator.evaluate(prog, img) for img in Xte])
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_ext_te)(prog) for prog in programs
        )
        return np.hstack(results).astype(np.float32)

    if has_chunk_cache:
        # === .npy チャンクキャッシュ (推奨形式) ===
        print(f"\n*** RESUME: Loading chunked .npy features from {cache_dir}/ ***")
        if is_gd:
            print(f"  Detected Google Drive path → using PREFETCH pipeline")
        
        import time as _time
        t0 = _time.time()
        feats_train = load_chunks_with_prefetch(cache_dir, 'train', gd_base=cache_dir if is_gd else None)
        t1 = _time.time()
        print(f"  Train loaded: {feats_train.shape} in {t1-t0:.1f}s")
        
        feats_test = load_chunks_with_prefetch(cache_dir, 'test', gd_base=cache_dir if is_gd else None)
        t2 = _time.time()
        print(f"  Test loaded: {feats_test.shape} in {t2-t1:.1f}s\n")
        
        scale_vec = "compute"

    elif have_train_cache and have_test_cache:
        # === 旧形式: 単一 .npy ファイル ===
        if is_gd:
            print(f"\n*** RESUME: Loading .npy from GD with prefetch: {cache_dir}/ ***")
            print(f"  (次回以降はチャンク形式に自動変換します)")
            
            import time as _time
            t0 = _time.time()
            
            # train/testを並行でプリフェッチ
            loader = PrefetchingFeatureLoader(
                ['feats_train.npy', 'feats_test.npy'],
                gd_base=cache_dir,
                cache_dir='/tmp/feat_cache_prefetch',
                queue_size=2,
            )
            
            feats_train = None
            feats_test = None
            for local_path, arr in loader:
                fname = os.path.basename(local_path)
                if 'train' in fname:
                    # mmapからRAMにコピー
                    feats_train = np.array(arr)
                    print(f"  Train loaded: {feats_train.shape} ({_time.time()-t0:.1f}s)")
                elif 'test' in fname:
                    feats_test = np.array(arr)
                    print(f"  Test loaded: {feats_test.shape} ({_time.time()-t0:.1f}s)")
                # mmapを閉じてからファイル削除
                del arr
                loader.cleanup_file(local_path)
            
            loader.cleanup_all()
            
            # 旧形式→チャンク形式に自動変換（次回以降高速化）
            print(f"\n  Converting to chunked .npy format for next time...")
            chunk_cols = 1000  # ~256MB per chunk (50000 samples * 1000 cols * 4 bytes)
            save_as_chunks(feats_train, cache_dir, 'train', chunk_cols=chunk_cols)
            save_as_chunks(feats_test, cache_dir, 'test', chunk_cols=chunk_cols)
            print(f"  Chunk conversion complete! Next run will be much faster.\n")
        else:
            # OneDrive or local → direct mmap
            print(f"\n*** RESUME: Loading raw features from {cache_dir}/ ***")
            feats_train = np.load(cache_train, mmap_mode='r')
            print(f"  Train loaded: {feats_train.shape}")
            feats_test = np.load(cache_test)
            print(f"  Test loaded: {feats_test.shape}\n")
        
        scale_vec = "compute"

    elif have_train_cache:
        # Train cached, test missing
        print(f"\n*** RESUME: Loading TRAIN from cache, re-extracting TEST only ***")
        if is_gd:
            loader = PrefetchingFeatureLoader(
                ['feats_train.npy'], gd_base=cache_dir,
                cache_dir='/tmp/feat_cache_prefetch', queue_size=1,
            )
            for local_path, arr in loader:
                feats_train = np.array(arr)
                del arr
                loader.cleanup_file(local_path)
            loader.cleanup_all()
        else:
            feats_train = np.load(cache_train, mmap_mode='r')
        print(f"  Loaded train: {feats_train.shape}")
        
        import time as _time
        start_time = _time.time()
        feats_test = _extract_test_only(programs, Xte, evaluator, args.n_jobs)
        elapsed = _time.time() - start_time
        print(f"  Test extraction done in {elapsed:.1f}s  Test={feats_test.shape}\n")
        try:
            _os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_test, feats_test)
            print(f"  Test cache saved to {cache_test}")
        except OSError:
            print("  Warning: could not save test cache (disk full?), continuing...")
        scale_vec = "compute"

    else:
        # Full extraction → チャンク単位で処理してGoogle Driveに.npy保存
        print(f"Extracting features from {len(Xtr)} training samples (chunked, n_jobs={args.n_jobs})...")
        import time as _time
        start_time = _time.time()
        
        _os.makedirs(cache_dir, exist_ok=True)
        
        # プログラムを100個ずつのチャンクに分割
        CHUNK_SIZE = 100
        num_chunks = (len(programs) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        all_train_chunks = []
        all_test_chunks = []
        total_dims = 0
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min(start_idx + CHUNK_SIZE, len(programs))
            chunk_programs = programs[start_idx:end_idx]
            
            print(f"\nChunk {chunk_idx+1}/{num_chunks} (programs {start_idx}-{end_idx})...")
            
            def extract_program(prog_idx, prog):
                """Extract features for one program"""
                feats_tr = np.array([evaluator.evaluate(prog, img) for img in Xtr])
                feats_te = np.array([evaluator.evaluate(prog, img) for img in Xte])
                return feats_tr, feats_te
            
            chunk_results = Parallel(n_jobs=args.n_jobs, verbose=5)(
                delayed(extract_program)(i, prog) for i, prog in enumerate(chunk_programs)
            )
            
            # チャンク内の次元数を計算
            chunk_dims = [r[0].shape[1] if r[0].ndim > 1 else 1 for r in chunk_results]
            chunk_total_dims = sum(chunk_dims)
            total_dims += chunk_total_dims
            
            # /tmpにmmap作成（メモリ節約、OneDrive容量節約）
            tmp_train = f"/tmp/train_chunk_{chunk_idx:03d}.npy"
            tmp_test = f"/tmp/test_chunk_{chunk_idx:03d}.npy"
            
            chunk_train = np.lib.format.open_memmap(
                tmp_train, mode='w+', dtype=np.float32,
                shape=(len(Xtr), chunk_total_dims)
            )
            chunk_test = np.lib.format.open_memmap(
                tmp_test, mode='w+', dtype=np.float32,
                shape=(len(Xte), chunk_total_dims)
            )
            
            col = 0
            for tr, te in chunk_results:
                tr = tr.astype(np.float32)
                te = te.astype(np.float32)
                if tr.ndim == 1:
                    tr = tr.reshape(-1, 1)
                    te = te.reshape(-1, 1)
                d = tr.shape[1]
                chunk_train[:, col:col+d] = tr
                chunk_test[:, col:col+d] = te
                col += d
            
            # mmapファイルをflushして確実にディスクに書き込む
            if hasattr(chunk_train, 'flush'):
                chunk_train.flush()
            if hasattr(chunk_test, 'flush'):
                chunk_test.flush()
            
            del chunk_results, chunk_train, chunk_test
            import gc
            gc.collect()
            
            # /tmpからGoogle Drive (2TB空き) に移動
            train_chunk_path = _os.path.join(cache_dir, f"train_chunk_{chunk_idx:03d}.npy")
            test_chunk_path = _os.path.join(cache_dir, f"test_chunk_{chunk_idx:03d}.npy")
            
            shutil.move(tmp_train, train_chunk_path)
            shutil.move(tmp_test, test_chunk_path)
            
            all_train_chunks.append(train_chunk_path)
            all_test_chunks.append(test_chunk_path)
            
            print(f"  Chunk {chunk_idx+1} saved to GD: {chunk_total_dims} dims (via /tmp mmap)")
            gc.collect()
        
        extraction_time = _time.time() - start_time
        print(f"\nFeature extraction completed in {extraction_time:.1f}s")
        print(f"Total chunks: {num_chunks}, Total dims: {total_dims}")
        
        # メタデータ保存（読み込み時に必要）
        meta = {
            'num_chunks': num_chunks,
            'total_dims': total_dims,
            'n_train': len(Xtr),
            'n_test': len(Xte)
        }
        import json
        with open(_os.path.join(cache_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f)
        
        # チャンクをmmapで読み込みながら結合（パイプライン処理）
        print("\nLoading chunks with mmap (pipeline from GD)...")
        feats_train = np.zeros((len(Xtr), total_dims), dtype=np.float32)
        feats_test = np.zeros((len(Xte), total_dims), dtype=np.float32)
        
        col = 0
        for chunk_idx in range(num_chunks):
            # mmap_mode='r' でGDから読み込み（プリフェッチしながら次のchunkが準備される）
            train_chunk = np.load(all_train_chunks[chunk_idx], mmap_mode='r')
            test_chunk = np.load(all_test_chunks[chunk_idx], mmap_mode='r')
            d = train_chunk.shape[1]
            # コピー時にmmapからメモリへ転送（この間にOSがGDから次をプリフェッチ）
            feats_train[:, col:col+d] = train_chunk
            feats_test[:, col:col+d] = test_chunk
            col += d
            print(f"  Loaded chunk {chunk_idx+1}/{num_chunks} via mmap")
        
        scale_vec = "compute"

    import gc
    gc.collect()
    
    # --- Diagonal scaling (compute scale vector only, no disk writes) ---
    if scale_vec is None:
        # Already scaled (loaded from scaled cache)
        print("Features already scaled (from cache).")
        feats_train_scaled = feats_train
        feats_test_scaled = feats_test
        scale_vector = None
    elif scale_vec == "compute":
        # Compute scale vector from train mmap (chunked, no disk write)
        print("Computing diagonal scaling (chunked, eps=1e-6, scale_factor=16.0)...")
        chunk_size = 5000
        n_cols = feats_train.shape[1]
        n_samples = feats_train.shape[0]
        
        # Two-pass chunked variance
        mean_acc = np.zeros(n_cols, dtype=np.float64)
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            mean_acc += np.array(feats_train[start:end], dtype=np.float64).sum(axis=0)
        mean_acc /= n_samples
        
        var_acc = np.zeros(n_cols, dtype=np.float64)
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = np.array(feats_train[start:end], dtype=np.float64)
            var_acc += ((chunk - mean_acc) ** 2).sum(axis=0)
        var_acc /= n_samples
        scale_vector = (np.sqrt(var_acc.astype(np.float32)) + np.float32(1e-6)) / np.float32(16.0)
        del mean_acc, var_acc
        print(f"  Scale vector computed (shape={scale_vector.shape}), will scale on-the-fly during training")
        
        # Keep raw features as-is (will scale during LightGBM dataset construction)
        feats_train_scaled = feats_train
        feats_test_scaled = feats_test
    else:
        # scale_vec already computed (shouldn't reach here with mmap)
        feats_train_scaled = feats_train
        feats_test_scaled = feats_test
        scale_vector = scale_vec

    # Note: Keep feats_train/test in RAM for evaluation later
    # feats_train_scaled is just an alias (they are the same unscaled raw data)
    feats_train_raw_for_eval = feats_train_scaled  # Keep reference for evaluation
    feats_test_raw_for_eval = feats_test_scaled
    gc.collect()

    # Train/val split
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    
    print("\nSplitting train/val (90/10) for early stopping...")
    train_idx, val_idx = train_test_split(
        np.arange(feats_train_scaled.shape[0]), test_size=0.1, random_state=42, stratify=ytr
    )
    train_idx_sorted = np.sort(train_idx)
    val_idx_sorted = np.sort(val_idx)
    y_train = ytr[train_idx_sorted]
    y_val = ytr[val_idx_sorted]
    
    # Build LightGBM datasets from mmap (with on-the-fly scaling if needed)
    print(f"  Building LightGBM Dataset from mmap ({feats_train_scaled.shape[0]}×{feats_train_scaled.shape[1]})...")
    print(f"  Train: {len(train_idx_sorted)} samples,  Val: {len(val_idx_sorted)} samples")
    n_dims = feats_train_scaled.shape[1]
    
    if scale_vector is not None:
        # Need to scale on-the-fly - load train into RAM with scaling
        print(f"  Scaling train data on-the-fly (requires ~{feats_train_scaled.nbytes / 1e9:.1f}GB RAM)...")
        X_train_full = np.array(feats_train_scaled, dtype=np.float32) / scale_vector
        dtrain_full = lgb.Dataset(X_train_full, ytr, free_raw_data=True, max_bin=255)
        dtrain = dtrain_full.subset(train_idx_sorted)
        dval = dtrain_full.subset(val_idx_sorted)
        dtrain.construct()
        dval.construct()
        del X_train_full
    else:
        # Already scaled - use mmap directly (RAM-efficient)
        dtrain_full = lgb.Dataset(feats_train_scaled, ytr, free_raw_data=False, max_bin=255)
        dtrain = dtrain_full.subset(train_idx_sorted)
        dval = dtrain_full.subset(val_idx_sorted)
        dtrain.construct()
        dval.construct()
    
    del feats_train_scaled
    gc.collect()
    
    # Train with early stopping (AUTO tree count)
    max_estimators = 1000 if args.quick else 5000
    stopping_rounds = 100 if args.quick else 200

    print(f"\nTraining LightGBM with Early Stopping + Auto-Regularization...")
    print(f"  Max estimators: {max_estimators} (will stop early)")
    print(f"  Stopping rounds: {stopping_rounds}")
    print(f"  Max depth: 6, Num leaves: 63")
    print(f"  subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_lambda=1.0")
    
    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'max_depth': 6,
        'num_leaves': 63,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_lambda': 1.0,
        'max_bin': 255,
        'n_jobs': args.n_jobs,
        'seed': 42,
        'verbose': -1,
    }
    
    train_start = time.time()
    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=max_estimators,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=True)]
    )
    train_time = time.time() - train_start
    
    trees_used = bst.best_iteration if bst.best_iteration else max_estimators
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Best iteration: {trees_used}")
    
    # Evaluate - chunked prediction from raw data with on-the-fly scaling
    def _predict_accuracy_scaled(bst, raw_data, labels, scale_vec, chunk_size=2000):
        """Predict in chunks from raw data with on-the-fly scaling"""
        correct = 0
        total = len(labels)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = np.array(raw_data[start:end], dtype=np.float32)
            if scale_vec is not None:
                chunk = chunk / scale_vec
            preds = bst.predict(chunk)
            pred_labels = np.argmax(preds, axis=1)
            correct += (pred_labels == labels[start:end]).sum()
        return correct / total
    
    # Use raw features already in RAM for evaluation
    print("Evaluating train accuracy (chunked, scaled on-the-fly)...")
    acc_train = _predict_accuracy_scaled(bst, feats_train_raw_for_eval[train_idx_sorted], y_train, scale_vector)
    print("Evaluating val accuracy (chunked, scaled on-the-fly)...")
    acc_val = _predict_accuracy_scaled(bst, feats_train_raw_for_eval[val_idx_sorted], y_val, scale_vector)
    del feats_train_raw_for_eval
    gc.collect()
    
    # Evaluate test accuracy (chunked from raw data with scaling)
    print("Evaluating test accuracy (chunked, scaled on-the-fly)...")
    acc_test = _predict_accuracy_scaled(bst, feats_test_raw_for_eval, yte, scale_vector)
    del feats_test_raw_for_eval
    gc.collect()
    
    print(f"\n{'='*70}")
    print(f"Final Results - V7 (Reward-ready, V6 base)")
    print(f"  Train Accuracy:     {acc_train*100:.2f}%")
    print(f"  Val Accuracy:       {acc_val*100:.2f}%")
    print(f"  Test Accuracy:      {acc_test*100:.2f}%")
    print(f"  Train-Test Gap:     {(acc_train - acc_test)*100:.2f}%")
    print(f"  Trees Used:         {trees_used} (stopped early from {max_estimators})")
    print(f"  Feature Dims:       {n_dims}")
    print(f"  Programs:           {len(programs)}")
    print(f"  Train Size:         {len(Xtr)}")
    print(f"  Training Time:      {train_time:.1f}s")
    print(f"{'='*70}")

    print(f"\nComparison:")
    print(f"  V5 (3000 trees):    63.08%")
    print(f"  V7 (auto trees):    {acc_test*100:.2f}%")
    print(f"  Target (mymodel2):  77.00%")
    print(f"  Gap remaining:      {77.0 - acc_test*100:.2f}%")

    if reward_logger is not None:
        reward_logger.log_summary({
            "train_acc": float(acc_train),
            "val_acc": float(acc_val),
            "test_acc": float(acc_test),
            "train_time_s": float(train_time),
            "extraction_time_s": 0.0,
            "total_time_s": float(train_time),
            "trees_used": int(trees_used),
            "feature_dims": int(n_dims),
            "reward_metric": args.reward_metric,
            "reward_subsample": int(args.reward_subsample),
        })
        reward_logger.close()
        print(f"\nReward log saved to: {reward_out}")
