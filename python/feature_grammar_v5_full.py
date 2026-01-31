#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature Grammar V5 - Full Scale with Diagonal Scaling + RGB
- RGB channels (R, G, B) individually
- Diagonal scaling (mymodel2-style)
- Multi-scale grid statistics
- Morphological operations
- 50k training samples, 3000 trees
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
class Concat(ASTNode):
    left: ASTNode
    right: ASTNode
    def __repr__(self): return f"Concat({self.left},{self.right})"

# ========== Evaluator ==========

class FullEvaluator:
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
        
        for i in range(0, h-1, 2):
            for j in range(0, w-1, 2):
                p = (B01[i, j] << 3) | (B01[i, j+1] << 2) | \
                    (B01[i+1, j] << 1) | B01[i+1, j+1]
                counts[p] += 1
        
        total = np.sum(counts)
        return (counts / (total + 1e-9)).astype(np.float32)
    
    def _markov4(self, B01: np.ndarray) -> np.ndarray:
        h, w = B01.shape
        trans = np.zeros((2, 2), dtype=np.int32)
        
        for i in range(h):
            for j in range(w-1):
                trans[B01[i,j], B01[i,j+1]] += 1
        for i in range(h-1):
            for j in range(w):
                trans[B01[i,j], B01[i+1,j]] += 1
        
        total = np.sum(trans)
        probs = (trans / (total + 1e-9)).flatten().astype(np.float32)
        return probs


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
    print("Feature Grammar V5 - Full Scale (50k + Diagonal Scaling + RGB)")
    print("=" * 70)
    
    # Load full data
    Xtr, ytr, Xte, yte = load_cifar10_numpy("./cifar10_data")
    print(f"Dataset: Train={len(Xtr)}, Test={len(Xte)}\n")
    
    evaluator = FullEvaluator()
    
    # Build programs with RGB channels
    img = Img()
    r = R(img)
    g = G(img)
    b = B(img)
    gray = Gray(img)
    opprg = OppRG(img)
    oppyb = OppYB(img)
    sat = Sat(img)
    
    programs = [
        # RGB channels (NEW!)
        GridStats(Threshold(r, 80), grid_n=8),   # 81 dims
        GridStats(Threshold(r, 120), grid_n=8),  # 81 dims
        GridStats(Threshold(g, 80), grid_n=8),   # 81 dims
        GridStats(Threshold(g, 120), grid_n=8),  # 81 dims
        GridStats(Threshold(b, 80), grid_n=8),   # 81 dims
        GridStats(Threshold(b, 120), grid_n=8),  # 81 dims
        
        # Gray with multiple thresholds
        GridStats(Threshold(gray, 60), grid_n=8),   # 81 dims
        GridStats(Threshold(gray, 100), grid_n=8),  # 81 dims
        GridStats(Threshold(gray, 140), grid_n=8),  # 81 dims
        GridStats(Threshold(gray, 180), grid_n=8),  # 81 dims
        
        # Opponent colors
        GridStats(Threshold(opprg, 80), grid_n=8),   # 81 dims
        GridStats(Threshold(opprg, 120), grid_n=8),  # 81 dims
        GridStats(Threshold(oppyb, 80), grid_n=8),   # 81 dims
        GridStats(Threshold(oppyb, 120), grid_n=8),  # 81 dims
        
        # Saturation
        GridStats(Threshold(sat, 0.3*255), grid_n=8),  # 81 dims
        GridStats(Threshold(sat, 0.5*255), grid_n=8),  # 81 dims
        
        # Pat2x2 features
        Pat2x2(Threshold(gray, 120)),   # 16 dims
        Pat2x2(Threshold(opprg, 100)),  # 16 dims
        Pat2x2(Threshold(r, 120)),      # 16 dims
        Pat2x2(Threshold(g, 120)),      # 16 dims
        Pat2x2(Threshold(b, 120)),      # 16 dims
        
        # Morphological operations
        GridStats(CA(Threshold(gray, 120), 1, "life"), grid_n=8),     # 81 dims
        GridStats(CA(Threshold(gray, 120), 2, "life"), grid_n=8),     # 81 dims
        GridStats(CA(Threshold(gray, 120), 1, "erosion"), grid_n=8),  # 81 dims
        GridStats(CA(Threshold(gray, 120), 1, "dilation"), grid_n=8), # 81 dims
        GridStats(CA(Threshold(gray, 120), 1, "opening"), grid_n=8),  # 81 dims
        GridStats(CA(Threshold(gray, 120), 1, "closing"), grid_n=8),  # 81 dims
        
        # RGB morphology
        GridStats(CA(Threshold(r, 120), 1, "erosion"), grid_n=8),  # 81 dims
        GridStats(CA(Threshold(g, 120), 1, "erosion"), grid_n=8),  # 81 dims
        GridStats(CA(Threshold(b, 120), 1, "erosion"), grid_n=8),  # 81 dims
        
        # Markov4
        Markov4(Threshold(gray, 120)),   # 4 dims
        Markov4(Threshold(opprg, 100)),  # 4 dims
        Markov4(Threshold(r, 120)),      # 4 dims
        Markov4(Threshold(g, 120)),      # 4 dims
        Markov4(Threshold(b, 120)),      # 4 dims
        
        # Moments
        Moments(Threshold(gray, 120)),   # 4 dims
        Moments(Threshold(opprg, 100)),  # 4 dims
        
        # Combined features
        Concat(
            GridStats(Threshold(gray, 120), grid_n=8),
            Markov4(Threshold(gray, 120))
        ),  # 85 dims
        Concat(
            GridStats(Threshold(r, 120), grid_n=8),
            Pat2x2(Threshold(r, 120))
        ),  # 97 dims
        Concat(
            GridStats(Threshold(g, 120), grid_n=8),
            Pat2x2(Threshold(g, 120))
        ),  # 97 dims
        Concat(
            GridStats(Threshold(b, 120), grid_n=8),
            Pat2x2(Threshold(b, 120))
        ),  # 97 dims
    ]
    
    print(f"Using {len(programs)} feature programs\n")
    for i, prog in enumerate(programs[:10], 1):
        print(f"  {i}. {prog}")
    print(f"  ... ({len(programs)-10} more programs)")
    print()
    
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
    
    # Train
    print(f"\nTraining LightGBM (n_estimators=3000)...")
    clf = LGBMClassifier(
        n_estimators=3000,
        max_depth=8,
        num_leaves=255,
        learning_rate=0.05,
        random_state=42,
        verbose=200
    )
    
    train_start = time.time()
    clf.fit(feats_train_scaled, ytr)
    train_time = time.time() - train_start
    
    acc_train = clf.score(feats_train_scaled, ytr)
    acc_test = clf.score(feats_test_scaled, yte)
    
    print(f"\n{'='*70}")
    print(f"Final Results - V5 Full Scale")
    print(f"  Train Accuracy:     {acc_train*100:.2f}%")
    print(f"  Test Accuracy:      {acc_test*100:.2f}%")
    print(f"  Feature Dims:       {feats_train.shape[1]}")
    print(f"  Programs:           {len(programs)}")
    print(f"  Train Size:         {len(Xtr)}")
    print(f"  Trees:              3000")
    print(f"  Extraction Time:    {extraction_time:.1f}s")
    print(f"  Training Time:      {train_time:.1f}s")
    print(f"  Total Time:         {extraction_time + train_time:.1f}s")
    print(f"{'='*70}")
    
    # Target comparison
    print(f"\nTarget (mymodel2):    77.00%")
    print(f"Gap remaining:        {77.0 - acc_test*100:.2f}%")
