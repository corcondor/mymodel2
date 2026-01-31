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
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            return 0.299 * r + 0.587 * g + 0.114 * b
        
        elif isinstance(node, OppRG):
            rgb = self.evaluate(node.img, img)
            r, g = rgb[:,:,0], rgb[:,:,1]
            return r - g
        
        elif isinstance(node, OppYB):
            rgb = self.evaluate(node.img, img)
            r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
            return (r + g) / 2.0 - b
        
        elif isinstance(node, Sat):
            rgb = self.evaluate(node.img, img)
            mx = np.max(rgb, axis=2)
            mn = np.min(rgb, axis=2)
            delta = mx - mn
            return np.where(mx > 0, delta / (mx + 1e-9), 0.0)
        
        elif isinstance(node, Threshold):
            channel = self.evaluate(node.img, img)
            return (channel > node.thresh).astype(np.uint8)
        
        elif isinstance(node, Edge):
            channel = self.evaluate(node.img, img)
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            gx = convolve(channel, sobel_x, mode='constant')
            gy = convolve(channel, sobel_y, mode='constant')
            magnitude = np.sqrt(gx**2 + gy**2)
            return (magnitude > 50).astype(np.uint8)
        
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
        
        cnt = np.sum(B01) / (h * w)
        
        row_proj = []
        for i in range(grid_n):
            row_sum = np.sum(B01[i*grid_h:(i+1)*grid_h, :])
            row_proj.append(row_sum / (grid_h * w))
        
        col_proj = []
        for j in range(grid_n):
            col_sum = np.sum(B01[:, j*grid_w:(j+1)*grid_w])
            col_proj.append(col_sum / (h * grid_w))
        
        grid = []
        for i in range(grid_n):
            for j in range(grid_n):
                cell_sum = np.sum(B01[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w])
                grid.append(cell_sum / (grid_h * grid_w))
        
        return np.array([cnt] + row_proj + col_proj + grid, dtype=np.float32)
    
    def _pat2x2(self, B01: np.ndarray) -> np.ndarray:
        h, w = B01.shape
        counts = np.zeros(16, dtype=np.int32)
        # Ensure binary 0/1 values (input might be float or non-binary)
        B01_bin = (B01 > 0.5).astype(np.uint8)
        
        for i in range(0, h-1, 2):
            for j in range(0, w-1, 2):
                p = (int(B01_bin[i, j]) << 3) | (int(B01_bin[i, j+1]) << 2) | \
                    (int(B01_bin[i+1, j]) << 1) | int(B01_bin[i+1, j+1])
                counts[p] += 1
        
        total = np.sum(counts)
        return (counts / (total + 1e-9)).astype(np.float32)
    
    def _markov4(self, B01: np.ndarray) -> np.ndarray:
        h, w = B01.shape
        trans = np.zeros((2, 2), dtype=np.int32)
        # Ensure binary 0/1 values
        B01_bin = (B01 > 0.5).astype(np.int32)
        
        for i in range(h):
            for j in range(w-1):
                trans[B01_bin[i,j], B01_bin[i,j+1]] += 1
        for i in range(h-1):
            for j in range(w):
                trans[B01_bin[i,j], B01_bin[i+1,j]] += 1
        
        total = np.sum(trans)
        probs = (trans / (total + 1e-9)).flatten().astype(np.float32)
        return probs
    
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
    
    print("=" * 70)
    print("Feature Grammar V6 - LBP + RGB Histograms + Auto-generation")
    print("=" * 70)
    
    # Load full data
    Xtr, ytr, Xte, yte = load_cifar10_numpy("./cifar10_data")
    print(f"Dataset: Train={len(Xtr)}, Test={len(Xte)}\n")
    
    evaluator = FullEvaluator()
    
    # Auto-generate programs combinatorially
    img = Img()
    
    # Define channels and operators
    channels = [
        ('R', R(img)),
        ('G', G(img)),
        ('B', B(img)),
        ('Gray', Gray(img)),
        ('OppRG', OppRG(img)),
        ('OppYB', OppYB(img)),
        ('Sat', Sat(img))
    ]
    
    thresholds = [60, 100, 140, 180]
    stats_ops = [
        ('GridStats', lambda x: GridStats(x, grid_n=8)),
        ('Pat2x2', Pat2x2),
        ('Markov4', Markov4),
        ('Moments', Moments)
    ]
    
    programs = []
    
    # 1. Thresholded features (channel × threshold × stat)
    for ch_name, ch in channels:
        for th in thresholds:
            for stat_name, stat_fn in stats_ops:
                programs.append(stat_fn(Threshold(ch, th)))
    
    # 2. Edge features (channel × stat)
    for ch_name, ch in channels:
        for stat_name, stat_fn in stats_ops:
            programs.append(stat_fn(Edge(ch)))
    
    # 3. Direct channel statistics (channel × stat)
    for ch_name, ch in channels:
        for stat_name, stat_fn in stats_ops:
            programs.append(stat_fn(ch))
    
    # 4. Add mymodel2-style features
    programs.append(LBP(Gray(img), eps=0))       # 256 dims
    programs.append(RGBHist(img, bins=4))        # 64 dims (4^3)
    programs.append(RGBBlocks(img, blocks=4))    # 48 dims (4×4×3)
    
    print(f"Using {len(programs)} auto-generated feature programs\n")
    print(f"  Combinatorial: 7 channels × 4 thresholds × 4 stats = {7*4*4} programs")
    print(f"  Edge features: 7 channels × 4 stats = {7*4} programs")
    print(f"  Direct stats:  7 channels × 4 stats = {7*4} programs")
    print(f"  mymodel2 features: LBP (256d) + RGBHist (64d) + RGBBlocks (48d)")
    print(f"  Total: {len(programs)} programs\n")
    
    # Extract features with parallelization
    print("Extracting features from 50k training samples (parallel, n_jobs=4)...")
    start_time = time.time()
    
    def extract_program(prog_idx, prog):
        """Extract features for one program"""
        feats_tr = np.array([evaluator.evaluate(prog, img) for img in Xtr])
        feats_te = np.array([evaluator.evaluate(prog, img) for img in Xte])
        return feats_tr, feats_te
    
    # Parallel extraction (n_jobs=4)
    results = Parallel(n_jobs=4, verbose=10)(
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
    print(f"\nTraining LightGBM with Early Stopping...")
    print(f"  Max estimators: 5000 (will stop early)")
    print(f"  Stopping rounds: 100")
    
    clf = LGBMClassifier(
        n_estimators=5000,  # Large number, will stop early
        max_depth=8,
        num_leaves=255,
        learning_rate=0.1,  # Default (faster convergence with early stopping)
        random_state=42,
        verbose=-1  # Suppress output
    )
    
    train_start = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
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
    print(f"  Trees Used:         {trees_used} (stopped early from 5000)")
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
