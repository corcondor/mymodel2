from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import pickle
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# -------------------------
# Configuration / utils
# -------------------------
_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def now() -> float:
    return time.time()


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def sha1_of_json(obj) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def parse_int_list(s: Optional[str]) -> list[int]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def set_seed(seed: int):
    np.random.seed(int(seed))


def flip_images(X: np.ndarray) -> np.ndarray:
    return X[:, :, ::-1, :].copy()


def d4_group_transforms(img: np.ndarray) -> list[np.ndarray]:
    """
    Generate all 8 D4 group transformations of an image.
    D4 = dihedral group of order 8 (symmetries of a square)
    Elements: {e, r, r², r³, s, sr, sr², sr³}
    where r = 90° rotation (counterclockwise), s = horizontal flip
    
    Args:
        img: (H, W, C) numpy array
    Returns:
        list of 8 transformed images
    """
    transforms = []
    for k in range(4):  # r^0, r^1, r^2, r^3 (0°, 90°, 180°, 270°)
        rot = np.rot90(img, k, axes=(0, 1))
        transforms.append(rot.copy())
        # Apply horizontal flip to each rotation: sr^k
        transforms.append(rot[:, ::-1, :].copy())
    return transforms


# -------------------------
# CIFAR-10 loader (torchvision optional; fallback to tarball)
# -------------------------
def _try_load_torchvision(data_root: str):
    try:
        from torchvision import datasets  # type: ignore

        root = os.path.abspath(data_root)
        tr = datasets.CIFAR10(root=root, train=True, download=True)
        te = datasets.CIFAR10(root=root, train=False, download=True)
        Xtr = np.array(tr.data, dtype=np.uint8)
        ytr = np.array(tr.targets, dtype=np.int64)
        Xte = np.array(te.data, dtype=np.uint8)
        yte = np.array(te.targets, dtype=np.int64)
        return Xtr, ytr, Xte, yte
    except Exception:
        return None


def _download_and_extract_cifar10(data_root: str) -> Path:
    root = Path(data_root).resolve()
    ensure_dir(str(root))
    tar_path = root / "cifar-10-python.tar.gz"
    extract_dir = root / "cifar-10-batches-py"

    if not extract_dir.exists():
        if not tar_path.exists():
            print(f"[data] downloading CIFAR-10 to {tar_path} ...")
            urllib.request.urlretrieve(_CIFAR10_URL, str(tar_path))
        print(f"[data] extracting {tar_path} ...")
        with tarfile.open(str(tar_path), "r:gz") as tf:
            tf.extractall(str(root))
    return extract_dir


def _load_cifar10_from_batches(extract_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def unpickle(p):
        with open(p, "rb") as f:
            return pickle.load(f, encoding="bytes")

    xs = []
    ys = []
    for i in range(1, 6):
        d = unpickle(extract_dir / f"data_batch_{i}")
        xs.append(d[b"data"])
        ys.append(d[b"labels"] if b"labels" in d else d[b"fine_labels"])
    Xtr = np.concatenate(xs, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    ytr = np.array(sum(ys, []), dtype=np.int64)

    d = unpickle(extract_dir / "test_batch")
    Xte = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    yte = np.array(d[b"labels"] if b"labels" in d else d[b"fine_labels"], dtype=np.int64)

    return Xtr, ytr, Xte, yte


def load_cifar10_numpy(data_root: str = "./cifar10_data"):
    # tv = _try_load_torchvision(data_root)
    # if tv is not None:
    #     return tv
    extract_dir = _download_and_extract_cifar10(data_root)
    return _load_cifar10_from_batches(extract_dir)


# -------------------------
# Low-level image ops (integer-friendly)
# -------------------------
def gray_u8(img_u8: np.ndarray) -> np.ndarray:
    r = img_u8[..., 0].astype(np.uint16)
    g = img_u8[..., 1].astype(np.uint16)
    b = img_u8[..., 2].astype(np.uint16)
    y = (77 * r + 150 * g + 29 * b) >> 8
    return y.astype(np.uint8)


def _dilate4(B01: np.ndarray) -> np.ndarray:
    B = B01
    up = np.zeros_like(B); up[1:, :] = B[:-1, :]
    dn = np.zeros_like(B); dn[:-1, :] = B[1:, :]
    lf = np.zeros_like(B); lf[:, 1:] = B[:, :-1]
    rt = np.zeros_like(B); rt[:, :-1] = B[:, 1:]
    return np.maximum.reduce([B, up, dn, lf, rt]).astype(np.uint8)


def _erode4(B01: np.ndarray) -> np.ndarray:
    B = B01
    up = np.ones_like(B); up[1:, :] = B[:-1, :]
    dn = np.ones_like(B); dn[:-1, :] = B[1:, :]
    lf = np.ones_like(B); lf[:, 1:] = B[:, :-1]
    rt = np.ones_like(B); rt[:, :-1] = B[:, 1:]
    return np.minimum.reduce([B, up, dn, lf, rt]).astype(np.uint8)


def edge_map01(B01: np.ndarray) -> np.ndarray:
    d = _dilate4(B01)
    e = _erode4(B01)
    return (d ^ e).astype(np.uint8)


def _neighbor_count(B01: np.ndarray, use_diag: bool = True) -> np.ndarray:
    """
    Count neighbors for each pixel in a binary mask.
    The mask is treated with zero padding (no wrap-around).
    """
    B = B01.astype(np.uint8, copy=False)
    c = np.zeros_like(B, dtype=np.uint8)
    c[1:, :] += B[:-1, :]
    c[:-1, :] += B[1:, :]
    c[:, 1:] += B[:, :-1]
    c[:, :-1] += B[:, 1:]
    if use_diag:
        c[1:, 1:] += B[:-1, :-1]
        c[1:, :-1] += B[:-1, 1:]
        c[:-1, 1:] += B[1:, :-1]
        c[:-1, :-1] += B[1:, 1:]
    return c


def morph_ca_step(
    B01: np.ndarray,
    birth_min: int = 5,
    birth_max: int = 8,
    survive_min: int = 4,
    survive_max: int = 8,
    use_diag: bool = True,
) -> np.ndarray:
    """
    One step of a simple morphological cellular automaton on a binary mask.
    - birth: 0 -> 1 if neighbor count in [birth_min, birth_max]
    - survive: 1 -> 1 if neighbor count in [survive_min, survive_max]
    Zero padding is used (no wrap-around).
    """
    B = B01.astype(np.uint8, copy=False)
    neigh = _neighbor_count(B, use_diag=use_diag)
    born = (B == 0) & (neigh >= birth_min) & (neigh <= birth_max)
    survive = (B == 1) & (neigh >= survive_min) & (neigh <= survive_max)
    out = np.where(born | survive, 1, 0).astype(np.uint8)
    return out


def run_morph_ca(
    B01: np.ndarray,
    steps: int,
    birth_min: int = 5,
    birth_max: int = 8,
    survive_min: int = 4,
    survive_max: int = 8,
    use_diag: bool = True,
) -> list[np.ndarray]:
    """
    Run multiple CA steps, returning each intermediate mask (excluding the seed).
    """
    out = []
    B = B01.astype(np.uint8, copy=False)
    for _ in range(max(0, int(steps))):
        B = morph_ca_step(B, birth_min, birth_max, survive_min, survive_max, use_diag=use_diag)
        out.append(B.copy())
    return out


def grad_dir_masks(gray: np.ndarray, th: int):
    g = gray.astype(np.int16)
    dx = np.zeros((32, 32), dtype=np.int16)
    dy = np.zeros((32, 32), dtype=np.int16)
    dx[:, :-1] = g[:, 1:] - g[:, :-1]
    dy[:-1, :] = g[1:, :] - g[:-1, :]
    m1 = (dx > th).astype(np.uint8)
    m2 = (dx < -th).astype(np.uint8)
    m3 = (dy > th).astype(np.uint8)
    m4 = (dy < -th).astype(np.uint8)
    return [m1, m2, m3, m4]


# -------------------------
# Per-mask motif statistics
# -------------------------
def pooled_proj_and_grid(B01: np.ndarray, grid_n: int = 8):
    """
    cnt, rs(grid_n), cs(grid_n), grid (grid_n*grid_n)
    """
    B = B01.astype(np.uint16)
    cnt = int(B.sum())
    rs32 = B.sum(axis=1)  # (32,)
    cs32 = B.sum(axis=0)  # (32,)
    k = 32 // grid_n
    rs = rs32.reshape(grid_n, k).sum(axis=1).astype(np.int16)
    cs = cs32.reshape(grid_n, k).sum(axis=1).astype(np.int16)
    grid = B.reshape(grid_n, k, grid_n, k).sum(axis=(1, 3)).astype(np.int16).reshape(-1)
    return cnt, rs, cs, grid


def pat2x2_hist16(B01: np.ndarray) -> np.ndarray:
    B = B01.astype(np.uint8, copy=False)
    if B.shape[0] < 2 or B.shape[1] < 2:
        return np.zeros(16, dtype=np.int16)
    a = B[:-1, :-1]
    b = B[:-1, 1:]
    c = B[1:, :-1]
    d = B[1:, 1:]
    code = (a | (b << 1) | (c << 2) | (d << 3)).ravel().astype(np.int32)
    h = np.bincount(code, minlength=16).astype(np.int32)
    return np.clip(h, -32768, 32767).astype(np.int16)


def markov_hist_4dirs(B01: np.ndarray) -> np.ndarray:
    B = B01.astype(np.uint8, copy=False)
    out = []
    A = B[:, :-1].ravel(); C = B[:, 1:].ravel(); out.append(np.bincount((A << 1) | C, minlength=4))
    A = B[:-1, :].ravel(); C = B[1:, :].ravel(); out.append(np.bincount((A << 1) | C, minlength=4))
    A = B[:-1, :-1].ravel(); C = B[1:, 1:].ravel(); out.append(np.bincount((A << 1) | C, minlength=4))
    A = B[:-1, 1:].ravel(); C = B[1:, :-1].ravel(); out.append(np.bincount((A << 1) | C, minlength=4))
    h = np.concatenate(out, axis=0).astype(np.int16)
    return h


# -------------------------
# LBP (8-neighbor)
# -------------------------
def _lbp_mirror_lut():
    lut = np.zeros(256, dtype=np.uint8)
    for code in range(256):
        bits = [(code >> k) & 1 for k in range(8)]
        mbits = [0] * 8
        mbits[0] = bits[4]; mbits[4] = bits[0]
        mbits[1] = bits[3]; mbits[3] = bits[1]
        mbits[7] = bits[5]; mbits[5] = bits[7]
        mbits[2] = bits[2]; mbits[6] = bits[6]
        mcode = 0
        for k in range(8):
            mcode |= (mbits[k] << k)
        lut[code] = mcode
    return lut


_LBP_MIRROR = _lbp_mirror_lut()


def lbp_hist8(gray: np.ndarray, eps: int = 0, flip_invariant: bool = False) -> np.ndarray:
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
    if flip_invariant:
        mv = _LBP_MIRROR[v]
        v = np.minimum(v, mv)
    h = np.bincount(v, minlength=256).astype(np.int16)
    return h


# -------------------------
# Color features
# -------------------------
def rgb_block_means(img_u8: np.ndarray, blocks: int = 4) -> np.ndarray:
    B = blocks
    k = 32 // B
    x = img_u8.reshape(B, k, B, k, 3).mean(axis=(1, 3))
    return np.round(x).astype(np.int16).reshape(-1)


def rgb_coarse_hist(img_u8: np.ndarray, bins: int = 4) -> np.ndarray:
    if bins <= 1:
        return np.zeros(1, dtype=np.int16)
    r = (img_u8[..., 0].astype(np.uint16) * bins) >> 8
    g = (img_u8[..., 1].astype(np.uint16) * bins) >> 8
    b = (img_u8[..., 2].astype(np.uint16) * bins) >> 8
    idx = (r * (bins * bins) + g * bins + b).ravel().astype(np.int32)
    h = np.bincount(idx, minlength=bins * bins * bins).astype(np.int16)
    return h


# -------------------------
# Featurizer (single-image) -- unified layout
# -------------------------
class CIFARFeaturizer:
    def __init__(
        self,
        gray_thresholds: Sequence[int],
        grad_th: int = 12,
        rg_tpos: Sequence[int] = (20, 50, 80),
        yb_tpos: Sequence[int] = (20, 50, 80),
        sat_th: Sequence[int] = (30, 60, 90),
        blocks: int = 4,
        use_edges: bool = True,
        use_grad: bool = True,
        use_markov: bool = True,
        use_lbp: bool = True,
        lbp_eps: int = 0,
        lbp_flip_invariant: bool = False,
        color_hist_bins: int = 4,
        use_bias: bool = True,
        ca_steps: int = 0,
        ca_birth_min: int = 5,
        ca_birth_max: int = 8,
        ca_survive_min: int = 4,
        ca_survive_max: int = 8,
        ca_use_diag: bool = True,
    ):
        self.gray_thresholds = list(gray_thresholds)
        self.grad_th = int(grad_th)
        self.rg_tpos = list(rg_tpos)
        self.yb_tpos = list(yb_tpos)
        self.sat_th = list(sat_th)
        self.blocks = int(blocks)
        self.use_edges = bool(use_edges)
        self.use_grad = bool(use_grad)
        self.use_markov = bool(use_markov)
        self.use_lbp = bool(use_lbp)
        self.lbp_eps = int(lbp_eps)
        self.lbp_flip_invariant = bool(lbp_flip_invariant)
        self.color_hist_bins = int(color_hist_bins)
        self.use_bias = bool(use_bias)
        self.ca_steps = int(ca_steps)
        self.ca_birth_min = int(ca_birth_min)
        self.ca_birth_max = int(ca_birth_max)
        self.ca_survive_min = int(ca_survive_min)
        self.ca_survive_max = int(ca_survive_max)
        self.ca_use_diag = bool(ca_use_diag)

        # per mask default dims: cnt + rs8 + cs8 + grid64 + pat16 + markov16(opt)
        self.grid_n = 8
        self._per_mask_base = 1 + 8 + 8 + 64 + 16
        self._per_mask = self._per_mask_base + (16 if self.use_markov else 0)
        self.n_masks = self._count_masks()
        self.dim_lbp = 256 if self.use_lbp else 0
        self.dim_rgb_means = (self.blocks * self.blocks * 3)
        self.dim_rgb_hist = (self.color_hist_bins ** 3) if self.color_hist_bins > 1 else 1
        self.dim_bias = 1 if self.use_bias else 0
        self.dim = self.n_masks * self._per_mask + self.dim_lbp + self.dim_rgb_means + self.dim_rgb_hist + self.dim_bias

    def cfg(self) -> dict:
        return {
            "gray_thresholds": self.gray_thresholds,
            "grad_th": self.grad_th,
            "rg_tpos": self.rg_tpos,
            "yb_tpos": self.yb_tpos,
            "sat_th": self.sat_th,
            "blocks": self.blocks,
            "use_edges": self.use_edges,
            "use_grad": self.use_grad,
            "use_markov": self.use_markov,
            "use_lbp": self.use_lbp,
            "lbp_eps": self.lbp_eps,
            "lbp_flip_invariant": self.lbp_flip_invariant,
            "color_hist_bins": self.color_hist_bins,
            "use_bias": self.use_bias,
            "dim": self.dim,
            "ca_steps": self.ca_steps,
            "ca_birth_min": self.ca_birth_min,
            "ca_birth_max": self.ca_birth_max,
            "ca_survive_min": self.ca_survive_min,
            "ca_survive_max": self.ca_survive_max,
            "ca_use_diag": self.ca_use_diag,
        }

    def _count_masks(self) -> int:
        base = 0
        base += len(self.gray_thresholds)
        if self.use_edges:
            base += len(self.gray_thresholds)
        if self.use_grad:
            base += 4
        base += 2 * len(self.rg_tpos)
        base += 2 * len(self.yb_tpos)
        base += len(self.sat_th)
        mult = 1 + max(0, self.ca_steps)
        return base * mult

    def _mask_features(self, B01: np.ndarray) -> np.ndarray:
        cnt, rs, cs, grid = pooled_proj_and_grid(B01, grid_n=self.grid_n)
        pat = pat2x2_hist16(B01)
        mk = markov_hist_4dirs(B01) if self.use_markov else None
        parts = [np.array([cnt], dtype=np.int16), rs.astype(np.int16), cs.astype(np.int16), grid.astype(np.int16), pat]
        if mk is not None:
            parts.append(mk)
        return np.concatenate(parts, axis=0).astype(np.int16)

    def _add_mask_with_ca(self, feats: list[np.ndarray], B01: np.ndarray):
        feats.append(self._mask_features(B01))
        if self.ca_steps > 0:
            cas = run_morph_ca(
                B01,
                steps=self.ca_steps,
                birth_min=self.ca_birth_min,
                birth_max=self.ca_birth_max,
                survive_min=self.ca_survive_min,
                survive_max=self.ca_survive_max,
                use_diag=self.ca_use_diag,
            )
            for C in cas:
                feats.append(self._mask_features(C))

    def extract(self, img_u8: np.ndarray) -> np.ndarray:
        g = gray_u8(img_u8)
        feats = []
        # gray masks & edges
        for th in self.gray_thresholds:
            B = (g >= th).astype(np.uint8)
            self._add_mask_with_ca(feats, B)
            if self.use_edges:
                self._add_mask_with_ca(feats, edge_map01(B))
        # grad masks
        if self.use_grad:
            for B in grad_dir_masks(g, self.grad_th):
                self._add_mask_with_ca(feats, B)
        # opponent & sat
        R = img_u8[..., 0].astype(np.int16)
        G = img_u8[..., 1].astype(np.int16)
        Bc = img_u8[..., 2].astype(np.int16)
        rg = (R - G)
        yb = (2 * Bc - R - G)
        for th in self.rg_tpos:
            self._add_mask_with_ca(feats, (rg >= th).astype(np.uint8))
            self._add_mask_with_ca(feats, (rg <= -th).astype(np.uint8))
        for th in self.yb_tpos:
            t2 = 2 * th
            self._add_mask_with_ca(feats, (yb >= t2).astype(np.uint8))
            self._add_mask_with_ca(feats, (yb <= -t2).astype(np.uint8))
        mx = np.maximum.reduce([R, G, Bc])
        mn = np.minimum.reduce([R, G, Bc])
        sat = (mx - mn).astype(np.int16)
        for th in self.sat_th:
            self._add_mask_with_ca(feats, (sat >= th).astype(np.uint8))
        # lbp, color, bias
        if self.use_lbp:
            feats.append(lbp_hist8(g, eps=self.lbp_eps, flip_invariant=self.lbp_flip_invariant))
        feats.append(rgb_block_means(img_u8, blocks=self.blocks))
        feats.append(rgb_coarse_hist(img_u8, bins=self.color_hist_bins))
        if self.use_bias:
            feats.append(np.array([1], dtype=np.int16))
        v = np.concatenate(feats, axis=0).astype(np.int16)
        return v

    def extract_d4_invariant(self, img_u8: np.ndarray, pooling: str = "mean") -> np.ndarray:
        """
        Extract D4-invariant features by aggregating over all 8 group transformations.
        
        Args:
            img_u8: (H, W, C) uint8 image
            pooling: aggregation method - 'mean', 'max', or 'median'
        Returns:
            D4-invariant feature vector (same dimension as extract())
        """
        variants = d4_group_transforms(img_u8)
        features = np.stack([self.extract(v) for v in variants], axis=0)  # (8, D)
        
        if pooling == "mean":
            result = features.mean(axis=0)
        elif pooling == "max":
            result = features.max(axis=0)
        elif pooling == "median":
            result = np.median(features, axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return np.round(result).astype(np.int16)

    def extract_d4_equivariant(self, img_u8: np.ndarray) -> np.ndarray:
        """
        Extract D4-equivariant features by concatenating all 8 group transformations.
        This preserves full group structure but increases dimensionality by 8x.
        Useful for Phase 3 (Cayley graph / group convolution).
        
        Args:
            img_u8: (H, W, C) uint8 image
        Returns:
            Equivariant feature vector (8 × original dimension)
        """
        variants = d4_group_transforms(img_u8)
        features = [self.extract(v) for v in variants]
        return np.concatenate(features, axis=0).astype(np.int16)


# -------------------------
# Feature caching (memmap) with chunked build
# -------------------------
def _memmap_paths(cache_dir: str, tag: str, cfg_hash: str):
    base = Path(cache_dir) / f"{tag}_{cfg_hash}"
    return str(base) + ".mmap", str(base) + ".meta.json"


def build_features_memmap(
    X: np.ndarray,
    fe: CIFARFeaturizer,
    cache_dir: str,
    tag: str,
    force: bool = False,
    chunk: int = 512,
    num_workers: int = 1,
    verbose: bool = True,
    use_d4: bool = False,
    d4_pooling: str = "mean",
):
    ensure_dir(cache_dir)
    cfg_dict = fe.cfg() | {"tag": tag, "use_d4": use_d4, "d4_pooling": d4_pooling}
    h = sha1_of_json(cfg_dict)
    mmap_path, meta_path = _memmap_paths(cache_dir, tag, h)
    if (not force) and os.path.exists(mmap_path) and os.path.exists(meta_path):
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        if meta.get("hash") == h:
            if verbose:
                print(f"[feat] load {tag} from cache: {mmap_path} shape={meta['shape']}")
            F = np.memmap(mmap_path, dtype=np.int16, mode="r", shape=tuple(meta["shape"]))
            return F, h, mmap_path, meta_path

    N = X.shape[0]
    D = fe.dim
    if verbose:
        d4_str = f" [D4-{d4_pooling}]" if use_d4 else ""
        print(f"[feat] build {tag}{d4_str}: N={N} D={D} -> {mmap_path} chunk={chunk} workers={num_workers}")

    F = np.memmap(mmap_path, dtype=np.int16, mode="w+", shape=(N, D))
    t0 = now()
    slices = [(i, min(N, i + chunk)) for i in range(0, N, chunk)]

    def work_slice(i0: int, i1: int):
        for i in range(i0, i1):
            if use_d4:
                F[i, :] = fe.extract_d4_invariant(X[i], pooling=d4_pooling)
            else:
                F[i, :] = fe.extract(X[i])
        return i1 - i0

    if num_workers <= 1:
        done = 0
        for i0, i1 in slices:
            work_slice(i0, i1)
            done += (i1 - i0)
            if verbose and done % max(chunk, 1000) == 0:
                dt = now() - t0
                print(f"  {tag}: {done}/{N} ({dt:.1f}s)")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(work_slice, i0, i1) for (i0, i1) in slices]
            done = 0
            for f in futures:
                cnt = f.result()
                done += cnt
                if verbose:
                    dt = now() - t0
                    print(f"  {tag}: {done}/{N} ({dt:.1f}s)")

    F.flush()
    meta = {"hash": h, "shape": [N, D], "dtype": "int16", "tag": tag, "cfg": cfg_dict}
    Path(meta_path).write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    if verbose:
        print(f"[feat] done {tag}: time={now()-t0:.1f}s")
    F = np.memmap(mmap_path, dtype=np.int16, mode="r", shape=(N, D))
    return F, h, mmap_path, meta_path


# -------------------------
# Streaming stats (Welford merge)
# -------------------------
class RunningStats:
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)

    def update_batch(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        if X.size == 0:
            return
        nb = X.shape[0]
        mean_b = X.mean(axis=0)
        M2_b = ((X - mean_b) ** 2).sum(axis=0)
        self.merge(nb, mean_b, M2_b)

    def merge(self, n_b: int, mean_b: np.ndarray, M2_b: np.ndarray):
        if n_b == 0:
            return
        if self.n == 0:
            self.n = int(n_b)
            self.mean = mean_b.astype(np.float64).copy()
            self.M2 = M2_b.astype(np.float64).copy()
            return
        n_a = self.n
        mean_a = self.mean
        M2_a = self.M2
        n = n_a + n_b
        delta = mean_b - mean_a
        mean = (n_a * mean_a + n_b * mean_b) / n
        M2 = M2_a + M2_b + (delta * delta) * (n_a * n_b / n)
        self.n = int(n)
        self.mean = mean
        self.M2 = M2

    def var(self, ddof: int = 0) -> np.ndarray:
        denom = max(self.n - ddof, 1)
        return np.maximum(self.M2 / denom, 0.0)


# -------------------------
# Diagonal scaling (streaming)
# -------------------------
def fit_diag_invstd_stream(F: np.memmap, use_var: bool = True, eps: float = 1e-6, verbose: bool = True) -> np.ndarray:
    N, D = F.shape
    bs = 4096
    t0 = now()
    if use_var:
        rs = RunningStats(D)
        for i in range(0, N, bs):
            X = F[i:i+bs].astype(np.float64)
            rs.update_batch(X)
        var = rs.var(ddof=0)
        invstd = 1.0 / np.sqrt(var + eps)
    else:
        sum2 = np.zeros(D, dtype=np.float64)
        for i in range(0, N, bs):
            X = F[i:i+bs].astype(np.float64)
            sum2 += (X * X).sum(axis=0)
        mean2 = sum2 / float(N)
        invstd = 1.0 / np.sqrt(mean2 + eps)
    if verbose:
        print(f"[diag] fit invstd (use_var={use_var}, eps={eps}) time={now()-t0:.2f}s")
    return invstd.astype(np.float32)


def apply_diag_scale_memmap_stream(F_in: np.memmap, invstd: np.ndarray, out_path: str, scale: float = 16.0, verbose: bool = True) -> np.memmap:
    N, D = F_in.shape
    # Remove existing file if it exists to avoid OSError on Windows
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
            time.sleep(0.1)  # Give OS time to release file handle
        except Exception as e:
            # If removal fails, try with a slightly different name
            import uuid
            out_path = out_path.replace(".mmap", f"_{uuid.uuid4().hex[:8]}.mmap")
    F_out = np.memmap(out_path, dtype=np.int16, mode="w+", shape=(N, D))
    t0 = now()
    bs = 4096
    inv = invstd.reshape(1, -1).astype(np.float32)
    for i in range(0, N, bs):
        X = F_in[i:i+bs].astype(np.float32)
        Y = X * inv * float(scale)
        Y = np.clip(np.round(Y), -32768, 32767).astype(np.int16)
        F_out[i:i+bs] = Y
    F_out.flush()
    if verbose:
        print(f"[diag] wrote scaled feats: {out_path} time={now()-t0:.2f}s")
    return np.memmap(out_path, dtype=np.int16, mode="r", shape=(N, D))


# -------------------------
# Classifiers
# -------------------------
class AveragedClippedPerceptron:
    def __init__(self, n_classes: int, dim: int, step: int = 1, wmax: int = 12000, margin: int = 80):
        self.C = int(n_classes)
        self.D = int(dim)
        self.step = int(step)
        self.wmax = int(wmax)
        self.margin = int(margin)
        self.W = np.zeros((self.C, self.D), dtype=np.int64)
        self.Wsum = np.zeros((self.C, self.D), dtype=np.int64)
        self.t = 0

    def score(self, x_i16: np.ndarray) -> np.ndarray:
        return (self.W @ x_i16.astype(np.int64, copy=False)).astype(np.int64, copy=False)

    def update(self, x_i16: np.ndarray, y: int):
        self.t += 1
        y = int(y)
        s = self.score(x_i16)
        yhat = int(np.argmax(s))
        if yhat == y:
            s2 = s.copy()
            s2[y] = np.iinfo(np.int64).min // 4
            runner = int(np.argmax(s2))
            if s[y] >= s[runner] + self.margin:
                self.Wsum += self.W
                return
            yhat = runner
        x = x_i16.astype(np.int64, copy=False)
        self.W[y, :] += self.step * x
        self.W[yhat, :] -= self.step * x
        np.clip(self.W, -self.wmax, self.wmax, out=self.W)
        self.Wsum += self.W

    def finalize(self):
        if self.t > 0:
            self.W = (self.Wsum // self.t).astype(np.int64, copy=False)


class DiagLDA:
    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)
        self.classes_ = None
        self.mu_ = None
        self.sigma2_ = None
        self.logpriors_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        C = len(classes); D = X.shape[1]
        mus = np.zeros((C, D), dtype=np.float64)
        counts = np.zeros(C, dtype=np.int64)
        ss = np.zeros((C, D), dtype=np.float64)
        for i, c in enumerate(classes):
            Xi = X[y == c]
            counts[i] = Xi.shape[0]
            if counts[i] == 0:
                continue
            mus[i] = Xi.mean(axis=0)
            ss[i] = ((Xi - mus[i]) ** 2).sum(axis=0)
        pooled = ss.sum(axis=0) / max(X.shape[0], 1)
        pooled = np.maximum(pooled, self.eps)
        self.classes_ = classes
        self.mu_ = mus
        self.sigma2_ = pooled
        priors = (counts + 1e-12) / counts.sum()
        self.logpriors_ = np.log(priors)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        invsig = 1.0 / self.sigma2_
        const = -0.5 * (self.mu_ ** 2 * invsig).sum(axis=1) + self.logpriors_
        w = self.mu_ * invsig[None, :]
        scores = X.dot(w.T) + const[None, :]
        return scores


# -------------------------
# Evaluation helpers
# -------------------------
def eval_model_perceptron(W: np.ndarray, F: np.memmap, y: np.ndarray, Fflip: Optional[np.memmap] = None, batch: int = 2048):
    N = F.shape[0]
    correct = 0
    W64 = W.astype(np.int64, copy=False)
    for i in range(0, N, batch):
        X = F[i:i+batch].astype(np.int64)
        S = (W64 @ X.T)
        if Fflip is not None:
            Xf = Fflip[i:i+batch].astype(np.int64)
            S += (W64 @ Xf.T)
        yhat = np.argmax(S, axis=0).astype(np.int64)
        correct += int(np.sum(yhat == y[i:i+batch]))
    return correct / float(N)


def eval_model_gen(decision_fn, F: np.memmap, y: np.ndarray, Fflip: Optional[np.memmap] = None, batch: int = 2048):
    N = F.shape[0]
    correct = 0
    for i in range(0, N, batch):
        X = F[i:i+batch].astype(np.float64)
        S = decision_fn(X)
        if Fflip is not None:
            Xf = Fflip[i:i+batch].astype(np.float64)
            S += decision_fn(Xf)
        yhat = np.argmax(S, axis=1).astype(np.int64)
        correct += int(np.sum(yhat == y[i:i+batch]))
    return correct / float(N)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="mymodel4 - compact unified CIFAR-10 baseline")
    ap.add_argument("--data_root", type=str, default="./cifar10_data")
    ap.add_argument("--cache_dir", type=str, default="./feat_cache")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--force_feat", action="store_true")

    ap.add_argument("--flip_eval", action="store_true")
    ap.add_argument("--flip_train", action="store_true")

    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--wmax", type=int, default=12000)
    ap.add_argument("--margin", type=int, default=80)

    ap.add_argument("--gray_thresholds", type=str, default="40,60,80,100,120,140")
    ap.add_argument("--grad_th", type=int, default=12)
    ap.add_argument("--rg_tpos", type=str, default="20,50,80")
    ap.add_argument("--yb_tpos", type=str, default="20,50,80")
    ap.add_argument("--sat_th", type=str, default="30,60,90")
    ap.add_argument("--blocks", type=int, default=4)

    ap.add_argument("--no_edges", action="store_true")
    ap.add_argument("--no_grad", action="store_true")
    ap.add_argument("--no_markov", action="store_true")
    ap.add_argument("--no_lbp", action="store_true")
    ap.add_argument("--lbp_eps", type=int, default=0)
    ap.add_argument("--lbp_flip_invariant", action="store_true")
    ap.add_argument("--color_hist_bins", type=int, default=4)
    ap.add_argument("--no_bias", action="store_true")
    ap.add_argument("--ca_steps", type=int, default=0, help="number of morphological CA steps to run per mask")
    ap.add_argument("--ca_birth_min", type=int, default=5, help="CA birth lower threshold (inclusive)")
    ap.add_argument("--ca_birth_max", type=int, default=8, help="CA birth upper threshold (inclusive)")
    ap.add_argument("--ca_survive_min", type=int, default=4, help="CA survive lower threshold (inclusive)")
    ap.add_argument("--ca_survive_max", type=int, default=8, help="CA survive upper threshold (inclusive)")
    ap.add_argument("--ca_no_diag", action="store_true", help="disable diagonal neighbors in CA updates")

    ap.add_argument("--diag_scale", action="store_true")
    ap.add_argument("--diag_use_var", action="store_true", default=True)
    ap.add_argument("--no_diag_use_var", action="store_false", dest="diag_use_var")
    ap.add_argument("--diag_eps", type=float, default=1e-6)
    ap.add_argument("--diag_scale_factor", type=float, default=16.0)

    ap.add_argument("--classifier", choices=["perceptron", "diaglda", "lightgbm"], default="perceptron")
    
    # LightGBM specific parameters
    ap.add_argument("--lgbm_n_estimators", type=int, default=500, help="Number of boosting iterations for LightGBM")
    ap.add_argument("--lgbm_max_depth", type=int, default=8, help="Maximum tree depth for LightGBM")
    ap.add_argument("--lgbm_num_leaves", type=int, default=256, help="Maximum number of leaves for LightGBM")
    ap.add_argument("--lgbm_learning_rate", type=float, default=0.05, help="Learning rate for LightGBM")
    
    # Model persistence
    ap.add_argument("--save_model", type=str, default=None, help="Path to save trained model (pkl format)")
    ap.add_argument("--load_model", type=str, default=None, help="Path to load pretrained model (skip training)")

    ap.add_argument("--chunk_feat", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=1)

    ap.add_argument("--use_d4", action="store_true", help="Use D4-invariant features (8x slower extraction, better accuracy)")
    ap.add_argument("--d4_pooling", choices=["mean", "max", "median"], default="mean", help="Pooling method for D4 orbit aggregation")

    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.cache_dir)

    gray_thresholds = [int(x) for x in args.gray_thresholds.split(",") if x.strip()]
    rg_tpos = [int(x) for x in args.rg_tpos.split(",") if x.strip()]
    yb_tpos = [int(x) for x in args.yb_tpos.split(",") if x.strip()]
    sat_th = [int(x) for x in args.sat_th.split(",") if x.strip()]

    fe = CIFARFeaturizer(
        gray_thresholds=gray_thresholds,
        grad_th=args.grad_th,
        rg_tpos=rg_tpos,
        yb_tpos=yb_tpos,
        sat_th=sat_th,
        blocks=args.blocks,
        use_edges=(not args.no_edges),
        use_grad=(not args.no_grad),
        use_markov=(not args.no_markov),
        use_lbp=(not args.no_lbp),
        lbp_eps=args.lbp_eps,
        lbp_flip_invariant=args.lbp_flip_invariant,
        color_hist_bins=args.color_hist_bins,
        use_bias=(not args.no_bias),
        ca_steps=args.ca_steps,
        ca_birth_min=args.ca_birth_min,
        ca_birth_max=args.ca_birth_max,
        ca_survive_min=args.ca_survive_min,
        ca_survive_max=args.ca_survive_max,
        ca_use_diag=(not args.ca_no_diag),
    )

    print("[cfg] featurizer dim =", fe.dim, " masks =", fe.n_masks, " per-mask-dim =", fe._per_mask)

    t0 = now()
    Xtr, ytr, Xte, yte = load_cifar10_numpy(args.data_root)
    print(f"[data] loaded: train={Xtr.shape} test={Xte.shape} time={now()-t0:.2f}s")

    need_flip_feats = bool(args.flip_train or args.flip_eval)
    if need_flip_feats:
        Xtr_f = flip_images(Xtr)
        Xte_f = flip_images(Xte)
    else:
        Xtr_f = Xte_f = None

    Ftr, htr, _, _ = build_features_memmap(Xtr, fe, args.cache_dir, "train", force=args.force_feat, chunk=args.chunk_feat, num_workers=args.num_workers, use_d4=args.use_d4, d4_pooling=args.d4_pooling)
    Fte, hte, _, _ = build_features_memmap(Xte, fe, args.cache_dir, "test", force=args.force_feat, chunk=args.chunk_feat, num_workers=args.num_workers, use_d4=args.use_d4, d4_pooling=args.d4_pooling)

    Ftr_f = Fte_f = None
    if need_flip_feats:
        Ftr_f, htr_f, _, _ = build_features_memmap(Xtr_f, fe, args.cache_dir, "trainflip", force=args.force_feat, chunk=args.chunk_feat, num_workers=args.num_workers, use_d4=args.use_d4, d4_pooling=args.d4_pooling)
        Fte_f, hte_f, _, _ = build_features_memmap(Xte_f, fe, args.cache_dir, "testflip", force=args.force_feat, chunk=args.chunk_feat, num_workers=args.num_workers, use_d4=args.use_d4, d4_pooling=args.d4_pooling)

    # diag scaling (streaming)
    if args.diag_scale:
        invstd = fit_diag_invstd_stream(Ftr, use_var=bool(args.diag_use_var), eps=float(args.diag_eps), verbose=True)

        def scaled_path(tag, h):
            eps_str = str(int(args.diag_eps)) if args.diag_eps >= 1 else f"{args.diag_eps:.0e}"
            scale_str = str(int(args.diag_scale_factor))
            base = Path(args.cache_dir) / f"{tag}_{h}_diag{int(bool(args.diag_use_var))}_eps{eps_str}_s{scale_str}"
            return str(base) + ".mmap"

        Ftr = apply_diag_scale_memmap_stream(Ftr, invstd, scaled_path("train", htr), scale=args.diag_scale_factor)
        Fte = apply_diag_scale_memmap_stream(Fte, invstd, scaled_path("test", hte), scale=args.diag_scale_factor)
        if need_flip_feats:
            Ftr_f = apply_diag_scale_memmap_stream(Ftr_f, invstd, scaled_path("trainflip", htr_f), scale=args.diag_scale_factor)
            Fte_f = apply_diag_scale_memmap_stream(Fte_f, invstd, scaled_path("testflip", hte_f), scale=args.diag_scale_factor)

    # train / eval
    if args.classifier == "perceptron":
        model = AveragedClippedPerceptron(n_classes=10, dim=fe.dim, step=args.step, wmax=args.wmax, margin=args.margin)
        N = Ftr.shape[0]
        idx = np.arange(N, dtype=np.int64)
        t0 = now()
        for ep in range(args.epochs):
            np.random.shuffle(idx)
            for ii in idx:
                if args.flip_train and (np.random.rand() < 0.5) and (Ftr_f is not None):
                    x = Ftr_f[ii]
                else:
                    x = Ftr[ii]
                model.update(x, int(ytr[ii]))
            acc = eval_model_perceptron(model.W, Fte, yte, Fflip=(Fte_f if args.flip_eval else None))
            print(f"[epoch {ep+1}/{args.epochs}] test_acc={acc*100:.2f}%")
        model.finalize()
        acc_final = eval_model_perceptron(model.W, Fte, yte, Fflip=(Fte_f if args.flip_eval else None))
        print(f"[done] time={now()-t0:.1f}s final_test_acc={acc_final*100:.2f}%")
    
    elif args.classifier == "lightgbm":
        if not HAS_LIGHTGBM:
            print("[error] lightgbm not installed. Install with: pip install lightgbm")
            sys.exit(1)
        
        # Load pretrained model if specified
        if args.load_model and os.path.exists(args.load_model):
            print(f"[lightgbm] loading pretrained model from {args.load_model}")
            import pickle
            with open(args.load_model, 'rb') as f:
                model = pickle.load(f)
            print(f"[lightgbm] model loaded successfully")
        else:
            # Train new model
            print(f"[lightgbm] n_estimators={args.lgbm_n_estimators} max_depth={args.lgbm_max_depth} num_leaves={args.lgbm_num_leaves} lr={args.lgbm_learning_rate}")
            
            # Prepare training data with flip augmentation
            if args.flip_train and Ftr_f is not None:
                print("[lightgbm] using flip augmentation for training (concatenating train + trainflip)")
                X_train = np.vstack([Ftr[:], Ftr_f[:]])
                y_train = np.concatenate([ytr, ytr])
            else:
                X_train = Ftr[:]
                y_train = ytr
            
            model = lgb.LGBMClassifier(
                n_estimators=args.lgbm_n_estimators,
                max_depth=args.lgbm_max_depth,
                num_leaves=args.lgbm_num_leaves,
                learning_rate=args.lgbm_learning_rate,
                n_jobs=-1,
                verbose=-1
            )
            
            t0 = now()
            model.fit(X_train, y_train)
            print(f"[lightgbm] training done in {now()-t0:.1f}s")
            
            # Save model if requested
            if args.save_model:
                import pickle
                with open(args.save_model, 'wb') as f:
                    pickle.dump(model, f)
                print(f"[lightgbm] model saved to {args.save_model}")
        
        # Evaluation with flip ensemble (suppress sklearn warnings)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


            if args.flip_eval and Fte_f is not None:
                print("[lightgbm] using flip ensemble for evaluation")
                proba = model.predict_proba(Fte[:])
                proba_flip = model.predict_proba(Fte_f[:])
                proba_avg = (proba + proba_flip) / 2.0
                pred = np.argmax(proba_avg, axis=1)
            else:
                pred = model.predict(Fte[:])
        
        acc_final = np.mean(pred == yte)
        print(f"[done] final_test_acc={acc_final*100:.2f}%")
    
    else:
        model_gen = DiagLDA()
        print("[train] fitting DiagLDA on train...")
        model_gen.fit(Ftr.astype(np.float64), ytr)
        acc = eval_model_gen(model_gen.decision_function, Fte, yte, Fflip=(Fte_f if args.flip_eval else None))
        print(f"[done] diagLDA_test_acc={acc*100:.2f}%")

if __name__ == "__main__":
    main()
