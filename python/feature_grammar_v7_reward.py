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


def diag_scale(X, eps=1e-6, scale_factor=16.0, use_var=True):
    """mymodel2-style diagonal scaling (float32-optimized)"""
    X = X.astype(np.float32)
    if use_var:
        var = np.var(X, axis=0, dtype=np.float32) + np.float32(eps)
        scale = np.sqrt(var) / np.float32(scale_factor)
    else:
        std = np.std(X, axis=0, dtype=np.float32) + np.float32(eps)
        scale = std / np.float32(scale_factor)
    result = (X / scale).astype(np.float32)
    return result


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
    
    # Extract features with parallelization
    print(f"Extracting features from {len(Xtr)} training samples (parallel, n_jobs={args.n_jobs})...")
    start_time = time.time()
    
    def extract_program(prog_idx, prog):
        """Extract features for one program"""
        feats_tr = np.array([evaluator.evaluate(prog, img) for img in Xtr])
        feats_te = np.array([evaluator.evaluate(prog, img) for img in Xte])
        return feats_tr, feats_te
    
    # Parallel extraction
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(extract_program)(i, prog) for i, prog in enumerate(programs)
    )
    
    all_feats_train = [r[0] for r in results]
    all_feats_test = [r[1] for r in results]
    
    # Convert to float32 to save memory (55K dims)
    feats_train = np.hstack(all_feats_train).astype(np.float32)
    feats_test = np.hstack(all_feats_test).astype(np.float32)
    
    # Free memory
    del all_feats_train, all_feats_test
    import gc
    gc.collect()
    
    extraction_time = time.time() - start_time
    print(f"\nFeature extraction completed in {extraction_time:.1f}s")
    print(f"Combined: Train={feats_train.shape}, Test={feats_test.shape}\n")

    if reward_logger is not None:
        reward_idx = None
        if args.reward_subsample and args.reward_subsample < len(Xtr):
            rng = np.random.default_rng(args.reward_seed)
            reward_idx = rng.choice(len(Xtr), size=args.reward_subsample, replace=False)
        y_reward = ytr if reward_idx is None else ytr[reward_idx]

        mi_fn = None
        if args.reward_metric == "mi":
            try:
                from sklearn.feature_selection import mutual_info_classif
                mi_fn = mutual_info_classif
            except Exception:
                print("Warning: sklearn mutual_info_classif unavailable; falling back to variance reward.")
                args.reward_metric = "var"

        for i, prog in enumerate(programs):
            feats = all_feats_train[i]
            if reward_idx is not None:
                feats = feats[reward_idx]

            if feats.ndim == 1:
                dim = 1
                feats_2d = feats.reshape(-1, 1)
            else:
                dim = feats.shape[1]
                feats_2d = feats

            reward_val = None
            if args.reward_metric == "var":
                reward_val = float(np.mean(np.var(feats_2d, axis=0)))
            elif args.reward_metric == "mi" and mi_fn is not None:
                mi = mi_fn(feats_2d, y_reward, discrete_features=False, random_state=args.reward_seed)
                reward_val = float(np.mean(mi))

            tokens = token_lists[i] if token_lists else ast_to_tokens(prog)
            token_ids = token_ids_list[i] if token_ids_list else tokens_to_ids(tokens, vocab)
            reward_logger.log_program(i, prog, tokens, token_ids, dim, reward_val)
    
    # Apply diagonal scaling (mymodel2-style)
    print("Applying diagonal scaling (eps=1e-6, scale_factor=16.0, use_var=True)...")
    feats_train_scaled = diag_scale(feats_train, eps=1e-6, scale_factor=16.0, use_var=True)
    feats_test_scaled = diag_scale(feats_test, eps=1e-6, scale_factor=16.0, use_var=True)
    
    # Train/val split for early stopping
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    
    print("\nSplitting train/val (90/10) for early stopping...")
    X_train, X_val, y_train, y_val = train_test_split(
        feats_train_scaled, ytr, test_size=0.1, random_state=42, stratify=ytr
    )
    print(f"  Train: {len(y_train)} samples")
    print(f"  Val:   {len(y_val)} samples")
    
    # Train with early stopping (AUTO tree count)
    print(f"\nTraining LightGBM with Early Stopping + Auto-Regularization...")

    max_estimators = 1000 if args.quick else 5000  # Reduce max_estimators for quick mode
    stopping_rounds = 100 if args.quick else 200   # Reduce patience for quick mode

    print(f"  Max estimators: {max_estimators} (will stop early)")
    print(f"  Stopping rounds: {stopping_rounds}")
    print(f"  Max depth: 6 (reduced from 8 to prevent overfitting)")
    print(f"  Num leaves: 63 (reduced from 255)")
    print(f"  Auto-regularization parameters:")
    print(f"    subsample=0.8 (use 80% random samples per tree)")
    print(f"    colsample_bytree=0.8 (use 80% random features per tree)")
    print(f"    min_child_samples=20 (min 20 samples per leaf)")
    print(f"    reg_lambda=1.0 (L2 regularization)")
    
    clf = LGBMClassifier(
        n_estimators=max_estimators,  # Large number, will stop early
        max_depth=6,        # Reduced from 8 to prevent overfitting
        num_leaves=63,      # Reduced from 255 (2^6-1)
        learning_rate=0.1,  # Default (faster convergence with early stopping)
        # Auto-regularization to prevent overfitting (no manual tuning needed)
        subsample=0.8,            # Random 80% samples per tree
        colsample_bytree=0.8,     # Random 80% features per tree  
        min_child_samples=20,     # Min 20 samples per leaf (avoid tiny leaves)
        reg_lambda=1.0,           # L2 regularization (default but explicit)
        # Memory efficiency for large feature sets
        max_bin=255,              # Reduce histogram bins if needed
        n_jobs=args.n_jobs,
        random_state=42,
        verbose=-1  # Suppress output
    )
    
    train_start = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
    )
    train_time = time.time() - train_start
    
    # Get actual number of trees used
    trees_used = clf.best_iteration_ if hasattr(clf, 'best_iteration_') else clf.n_estimators
    
    print(f"\nTraining completed in {train_time:.1f}s")
    print(f"Best iteration: {trees_used}")
    print(f"Trees actually used: {trees_used} / 5000")
    
    # Evaluate on all sets
    acc_train = clf.score(X_train, y_train)
    acc_val = clf.score(X_val, y_val)
    acc_test = clf.score(feats_test_scaled, yte)
    
    print(f"\n{'='*70}")
    print(f"Final Results - V7 (Reward-ready, V6 base)")
    print(f"  Train Accuracy:     {acc_train*100:.2f}%")
    print(f"  Val Accuracy:       {acc_val*100:.2f}%")
    print(f"  Test Accuracy:      {acc_test*100:.2f}%")
    print(f"  Train-Test Gap:     {(acc_train - acc_test)*100:.2f}%")
    print(f"  Trees Used:         {trees_used} (stopped early from {max_estimators})")
    print(f"  Feature Dims:       {feats_train.shape[1]}")
    print(f"  Programs:           {len(programs)}")
    print(f"  Train Size:         {len(Xtr)}")
    print(f"  Extraction Time:    {extraction_time:.1f}s")
    print(f"  Training Time:      {train_time:.1f}s")
    print(f"  Total Time:         {extraction_time + train_time:.1f}s")
    print(f"{'='*70}")
    
    # Target comparison
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
            "extraction_time_s": float(extraction_time),
            "total_time_s": float(extraction_time + train_time),
            "trees_used": int(trees_used),
            "feature_dims": int(feats_train.shape[1]),
            "reward_metric": args.reward_metric,
            "reward_subsample": int(args.reward_subsample),
        })
        reward_logger.close()
        print(f"\nReward log saved to: {reward_out}")
