#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Grammar V6 - LBP + RGB Histograms + Auto-generation
- LBP (Local Binary Pattern) - 256 dims
- RGB Histograms - 64 dims  
- RGB Block Means - 48 dims
- Auto-generated programs (combinatorial)
- 50k training samples, 3000 trees, early stopping
"""

from dataclasses import dataclass
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
class Sat(ASTNode):
    img: ASTNode
    def __repr__(self): return f"Sat({self.img})"

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
class LBP(ASTNode):
    img: ASTNode
    eps: int = 0
    def __repr__(self): return f"LBP({self.img},eps={self.eps})"

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

# ========== Evaluator ==========

class FullEvaluator:
    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        """Convert RGB to grayscale"""
        if len(img.shape) == 2:
            return img.astype(np.float32)
        return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.float32)
    
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
        
        elif isinstance(node, Sat):
            rgb = self.evaluate(node.img, img)
            # mymodel2-style saturation proxy: max(R,G,B) - min(R,G,B) in [0..255]
            r = rgb[:, :, 0].astype(np.int16)
            g = rgb[:, :, 1].astype(np.int16)
            b = rgb[:, :, 2].astype(np.int16)
            mx = np.maximum.reduce([r, g, b])
            mn = np.minimum.reduce([r, g, b])
            return (mx - mn).astype(np.float32)

        elif isinstance(node, Neg):
            ch = self.evaluate(node.img, img)
            return (-ch).astype(np.float32)
        
        elif isinstance(node, Threshold):
            channel = self.evaluate(node.img, img)
            return (channel > node.thresh).astype(np.uint8)

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
        
        elif isinstance(node, LBP):
            gray = self._to_gray(img)
            return self._lbp_hist8(gray, eps=node.eps)
        
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
        h = np.bincount(code, minlength=16).astype(np.float32)
        return h / (h.sum() + 1e-9)
    
    def _markov4(self, B01: np.ndarray) -> np.ndarray:
        # mymodel2-style Markov transitions in 4 directions, 4 bins each => 16 dims
        B = (B01 > 0.5).astype(np.uint8)
        out = []
        A = B[:, :-1].ravel(); C = B[:, 1:].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        A = B[:-1, :].ravel(); C = B[1:, :].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        A = B[:-1, :-1].ravel(); C = B[1:, 1:].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        A = B[:-1, 1:].ravel(); C = B[1:, :-1].ravel(); out.append(np.bincount(((A << 1) | C).astype(np.int32), minlength=4))
        h = np.concatenate(out, axis=0).astype(np.float32)
        return h / (h.sum() + 1e-9)
    
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
        h = np.bincount(v, minlength=256).astype(np.float32)
        return h / (np.sum(h) + 1e-9)  # Normalize
    
    def _rgb_coarse_hist(self, img_u8: np.ndarray, bins: int = 4) -> np.ndarray:
        """RGB histogram - mymodel2 implementation"""
        if bins <= 1:
            return np.array([1.0], dtype=np.float32)
        r = (img_u8[..., 0].astype(np.uint16) * bins) >> 8
        g = (img_u8[..., 1].astype(np.uint16) * bins) >> 8
        b = (img_u8[..., 2].astype(np.uint16) * bins) >> 8
        idx = (r * (bins * bins) + g * bins + b).ravel().astype(np.int32)
        h = np.bincount(idx, minlength=bins * bins * bins).astype(np.float32)
        return h / (np.sum(h) + 1e-9)  # Normalize
    
    def _rgb_block_means(self, img_u8: np.ndarray, blocks: int = 4) -> np.ndarray:
        """RGB block means - mymodel2 implementation"""
        B = blocks
        k = 32 // B
        x = img_u8.reshape(B, k, B, k, 3).mean(axis=(1, 3))
        return x.reshape(-1).astype(np.float32) / 255.0  # Normalize to [0,1]


def diag_scale(X, eps=10, scale_factor=32, use_var=True):
    """mymodel2-style diagonal scaling"""
    if use_var:
        var = np.var(X, axis=0) + eps
        scale = np.sqrt(var) / scale_factor
    else:
        std = np.std(X, axis=0) + eps
        scale = std / scale_factor
    return X / scale


if __name__ == "__main__":
    from mymodel3 import load_cifar10_numpy
    from lightgbm import LGBMClassifier
    import time

    parser = argparse.ArgumentParser(description="Feature Grammar V6 (mymodel2-like masks + stats)")
    parser.add_argument("--data-dir", type=str, default="./cifar10_data")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--subset-train", type=int, default=0, help="0=full, else number of train samples")
    parser.add_argument("--subset-test", type=int, default=0, help="0=full, else number of test samples")
    parser.add_argument("--max-programs", type=int, default=0, help="0=all, else use first N programs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run a small stratified subset for a fast sanity check")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=Warning, module=r"mymodel3")
    
    print("=" * 70)
    print("Feature Grammar V6 - LBP + RGB Histograms + Auto-generation")
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
    sat = Sat(img)

    gray_thresholds = [60, 100, 140, 180]
    rg_tpos = [20, 50, 80]
    yb_tpos = [20, 50, 80]      # mymodel2 uses 2*th for yb
    sat_th = [30, 60, 90]
    grad_th = 12

    masks = []
    # gray masks + edges
    for th in gray_thresholds:
        m = Threshold(g, th)
        masks.append(m)
        masks.append(Edge(m))

    # grad direction masks (4 dirs)
    masks.append(Grad(g, grad_th, "dx_pos"))
    masks.append(Grad(g, grad_th, "dx_neg"))
    masks.append(Grad(g, grad_th, "dy_pos"))
    masks.append(Grad(g, grad_th, "dy_neg"))

    # opponent masks (pos/neg)
    for th in rg_tpos:
        masks.append(Threshold(rg, th))
        masks.append(Threshold(Neg(rg), th))
    for th in yb_tpos:
        t2 = 2 * th
        masks.append(Threshold(yb, t2))
        masks.append(Threshold(Neg(yb), t2))

    # saturation masks
    for th in sat_th:
        masks.append(Threshold(sat, th))

    # per-mask statistics (mymodel2-style core)
    stats_ops = [
        ("GridStats", lambda x: GridStats(x, grid_n=8)),
        ("Pat2x2", Pat2x2),
        ("Markov4", Markov4),
    ]

    programs = []
    for m in masks:
        for _, stat_fn in stats_ops:
            programs.append(stat_fn(m))

    # Add mymodel2-style global features
    programs.append(LBP(g, eps=0))               # 256 dims
    programs.append(RGBHist(img, bins=4))        # 64 dims
    programs.append(RGBBlocks(img, blocks=4))    # 48 dims

    if args.max_programs and args.max_programs < len(programs):
        programs = programs[: args.max_programs]

    print(f"Using {len(programs)} auto-generated feature programs\n")
    print(f"  Masks: {len(masks)} (gray+edge={2*len(gray_thresholds)}, grad=4, rg=2*{len(rg_tpos)}, yb=2*{len(yb_tpos)}, sat={len(sat_th)})")
    print(f"  Per-mask stats: {len(stats_ops)} => {len(masks)}×{len(stats_ops)} = {len(masks)*len(stats_ops)} programs")
    print(f"  mymodel2 features: LBP (256d) + RGBHist (64d) + RGBBlocks (48d) = 3 programs")
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
    
    feats_train = np.hstack(all_feats_train)
    feats_test = np.hstack(all_feats_test)
    
    extraction_time = time.time() - start_time
    print(f"\nFeature extraction completed in {extraction_time:.1f}s")
    print(f"Combined: Train={feats_train.shape}, Test={feats_test.shape}\n")
    
    # Apply diagonal scaling (mymodel2-style)
    print("Applying diagonal scaling (eps=10, scale_factor=32, use_var=True)...")
    feats_train_scaled = diag_scale(feats_train, eps=10, scale_factor=32, use_var=True)
    feats_test_scaled = diag_scale(feats_test, eps=10, scale_factor=32, use_var=True)
    
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
    print(f"Final Results - V6 (LBP + RGB Histograms + Auto-generation)")
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
    print(f"  V6 (auto trees):    {acc_test*100:.2f}%")
    print(f"  Target (mymodel2):  77.00%")
    print(f"  Gap remaining:      {77.0 - acc_test*100:.2f}%")
