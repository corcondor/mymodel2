# mymodel2: CIFAR-10構造駆動型分類器

> **CIFAR-10のための高度な離散数学的アプローチ**: D4二面体群対称性・局所二値パターン(LBP)・形態学的セルオートマトン・カラー統計を用いたCPU最適化実装

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概要

`mymodel2`は、**ニューラルネットワークやバックプロパゲーションに依存しない**CIFAR-10画像分類の実装です。[mymodel1](https://github.com/corcondor/mymodel1)（MNIST・96.36%）の進化版として、カラー自然画像に対応し、より高度な数学的構造を導入しています。

### 何を使わないか（What this is NOT）

- ❌ CNN / ViT 等のニューラルネットワーク
- ❌ バックプロパゲーション
- ❌ GPU / CUDA
- ❌ 浮動小数点最適化に依存した深層学習

### 何を使うか（What this IS）

- ✅ **整数演算中心**の離散数学的アプローチ
- ✅ **D4二面体群**（8つの対称性変換）による幾何不変性
- ✅ **二次形式** (Quadratic Forms over Z₂) に基づく特徴設計
- ✅ **古典機械学習**（平均化クリップ型パーセプトロン・対角LDA・LightGBM）
- ✅ **CPU最適化**で説明可能・再現性の高い実装

---

## mymodel1 との差別化

| 項目 | [mymodel1](https://github.com/corcondor/mymodel1) (MNIST) | mymodel2 (CIFAR-10) |
|------|------|------|
| **タスク** | 手書き数字（28×28グレースケール） | 自然画像（32×32カラー）|
| **基本特徴** | raw/core/edge 二値マスク | **LBP** + **CA** + カラー統計 + 二次形式 |
| **対称性** | 基本的な不変性 | **D4群**（回転×4 + 反転×4 = 8変換） |
| **動的進化** | なし | **セルオートマトン**による形態学的マスク進化 |
| **カラー処理** | なし | **opponent色空間**(R-G, Y-B)・彩度・RGB統計 |
| **分類器** | 平均化パーセプトロン | パーセプトロン・**対角LDA**・**LightGBM** |
| **理論的基盤** | 構造駆動型設計 | **Clifford回路理論**との数学的接続 ([arXiv:2601.15396](https://arxiv.org/abs/2601.15396)) |
| **精度** | ~96.36% | ~72-75%（CPU only、古典ML限界に挑戦）|

---

## 理論的背景（簡潔版）

### なぜ「二次形式」なのか？

最近の量子情報理論の研究（[arXiv:2601.15396](https://arxiv.org/abs/2601.15396)）により、**古典的に効率よく計算可能な量子モデル（Clifford回路）= アーベル群上の二次関数**であることが示されました。

| 次数 | モデル例 | 計算複雑性 |
|------|----------|------------|
| k=1 (線形) | 積状態、単純パリティ | O(n) - trivial |
| **k=2 (二次)** | **Clifford回路、自由フェルミオン、LBP** | **O(n²) - efficient** ✅ |
| k≥3 (高次) | 一般量子回路、3-SAT | Exp(n) - intractable ❌ |

**mymodel2は二次で閉じることで、古典計算の理論的最適性を達成しています。**

詳細は [THEORY.md](THEORY.md) を参照してください。

### 主要コンポーネント

#### 1. 二値マスク特徴抽出

画像を多数のバイナリマスクに変換し、各マスクから局所モチーフ統計を抽出：

- **グレースケール閾値**: デフォルト `64, 128, 192` の3レベル
- **エッジ検出**: XOR型二次形式（`dilate ⊕ erode`）
- **勾配方向マスク**: 4方向（右、左、下、上）の勾配閾値
- **opponent色空間**: R-G（赤-緑）、Y-B（黄-青）の符号別マスク
- **彩度マスク**: `max(R,G,B) - min(R,G,B)` の閾値化

#### 2. セルオートマトン (CA)

各マスクを形態学的セルオートマトンで動的に進化させ、トポロジー的特徴を捕捉：

```python
# デフォルトルール
birth_min=5, birth_max=8      # 0→1の条件（近傍数）
survive_min=4, survive_max=8  # 1→1の条件
steps=0 (デフォルトで無効、推奨: 1-2)
```

#### 3. Local Binary Patterns (LBP)

8近傍のテクスチャパターンを抽出：

```python
# 各ピクセルの8近傍を比較 → 256パターンのヒストグラム
lbp_hist8(gray, eps=0, flip_invariant=False)
```

- `flip_invariant=True`: 左右反転不変なLBP符号化

#### 4. D4群不変性

8つの対称性変換（回転90°×4 + 各回転に反転）で画像を変換し、特徴量を集約：

```python
# D4群の全8変換
d4_group_transforms(img) → [img_e, img_r90, img_r180, img_r270,
                             img_flip, img_r90_flip, ...]
                             
# 集約方法
--d4_pooling mean   # 平均値（推奨）
--d4_pooling max    # 最大値
--d4_pooling median # 中央値
```

#### 5. カラー特徴

- **RGBブロック統計**: 4×4ブロックのRGB平均値（48次元）
- **粗いカラーヒストグラム**: 4×4×4 = 64ビンのRGBヒストグラム

#### 6. マスクごとの統計量

各バイナリマスクから以下を抽出：

- **カウント**: マスク内の1の総数
- **行/列投影**: 8×8グリッドへの投影
- **2×2パターンヒストグラム**: 16パターン
- **マルコフ遷移**: 4方向の遷移確率（オプション）

**特徴量次元**: デフォルト設定で約3,900次元（CA有効化で増加）

---

## インストール

```bash
# 必須
pip install numpy

# オプション（データセット読み込み用）
pip install torchvision

# オプション（LightGBM使用時）
pip install lightgbm
```

---

## 使用方法

### 基本実行

```bash
# デフォルト設定（平均化クリップ型パーセプトロン）
python mymodel2.py

# LightGBM使用（最高精度、推奨）
python mymodel2.py --classifier lightgbm

# Diagonal LDA使用
python mymodel2.py --classifier lda
```

### 推奨設定（高精度）

```bash
# 論文推奨設定：flip拡張 + 対角スケーリング
python mymodel2.py \
  --classifier lightgbm \
  --flip_train \
  --flip_eval \
  --diag_scale \
  --diag_use_var \
  --diag_eps 10 \
  --diag_scale_factor 32 \
  --force_feat
```

### D4対称性を使用

```bash
# D4不変特徴（8x遅いが精度向上）
python mymodel2.py \
  --use_d4 \
  --d4_pooling mean \
  --classifier lightgbm
```

### セルオートマトン有効化

```bash
# CA 2ステップ（形態学的進化）
python mymodel2.py \
  --ca_steps 2 \
  --ca_birth_min 5 \
  --ca_birth_max 8 \
  --ca_survive_min 4 \
  --ca_survive_max 8
```

---

## コマンドラインオプション

### 分類器選択

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--classifier` | `perceptron`, `lda`, `lightgbm` | `perceptron` |
| `--epochs` | パーセプトロンのエポック数 | 5 |
| `--step` | パーセプトロンのステップサイズ | 1 |
| `--wmax` | パーセプトロンの重みクリッピング | 12000 |
| `--margin` | パーセプトロンのマージン | 80 |

### LightGBMパラメータ

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--lgbm_n_estimators` | ツリーの数（**精度に大きく影響**） | 1000 |
| `--lgbm_max_depth` | 最大深度 | 8 |
| `--lgbm_num_leaves` | 最大葉数 | 256 |
| `--lgbm_learning_rate` | 学習率 | 0.05 |
| `--save_model` | モデル保存パス (.pkl) | None |
| `--load_model` | モデル読み込みパス (.pkl) | None |

### データ拡張

| オプション | 説明 |
|-----------|------|
| `--flip_train` | 訓練時の左右反転拡張（ランダム） |
| `--flip_eval` | 評価時の反転アンサンブル（精度向上） |

### D4対称性

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--use_d4` | D4不変特徴を使用（8x遅い） | False |
| `--d4_pooling` | `mean`, `max`, `median` | `mean` |

### 対角スケーリング

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--diag_scale` | Fisher風対角スケーリング有効化 | False |
| `--diag_use_var` | 分散ベース（True）vs 二乗平均（False） | False |
| `--diag_eps` | 正則化項 ε | 1e-6 |
| `--diag_scale_factor` | スケーリング係数 | 16.0 |

### 特徴量設定

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--gray_thresholds` | グレー閾値（カンマ区切り） | "64,128,192" |
| `--grad_th` | 勾配閾値 | 12 |
| `--rg_tpos` | R-G閾値 | "20,50,80" |
| `--yb_tpos` | Y-B閾値 | "20,50,80" |
| `--sat_th` | 彩度閾値 | "30,60,90" |
| `--blocks` | RGBブロック数 | 4 |
| `--color_hist_bins` | カラーヒストグラムビン数 | 4 |

### セルオートマトン

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--ca_steps` | CAステップ数（0=無効） | 0 |
| `--ca_birth_min` | 誕生の最小近傍数 | 5 |
| `--ca_birth_max` | 誕生の最大近傍数 | 8 |
| `--ca_survive_min` | 生存の最小近傍数 | 4 |
| `--ca_survive_max` | 生存の最大近傍数 | 8 |
| `--ca_no_diag` | 対角近傍を無効化（4近傍） | False（8近傍） |

### LBP設定

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--lbp_eps` | LBP比較の閾値 | 0 |
| `--lbp_flip_invariant` | 反転不変LBP | False |
| `--no_lbp` | LBP無効化 | False（有効） |

### その他

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--no_edges` | エッジ検出無効化 | False（有効） |
| `--no_grad` | 勾配マスク無効化 | False（有効） |
| `--no_markov` | マルコフ統計無効化 | False（有効） |
| `--no_bias` | バイアス項無効化 | False（有効） |
| `--force_feat` | 特徴量キャッシュを強制再計算 | False |
| `--seed` | 乱数シード | 42 |

---

## 結果

### 性能ベンチマーク

```
Dataset: CIFAR-10
Method: LightGBM + flip augmentation + diagonal scaling
Accuracy: ~76-77% (n_estimators=1000-10000)
Training time: ~20-60分（特徴抽出含む、CPU）
Feature dimension: 3,872 (default settings)
Hardware: CPU only (no GPU)
```

### 性能比較

| 手法 | 精度 | 備考 |
|------|------|------|
| Perceptron (baseline) | ~70% | 整数演算のみ、最速 |
| Perceptron + flip | ~72% | 反転拡張 |
| LightGBM (n_estimators=100) | ~67-68% | 基本設定 |
| LightGBM (n_estimators=1000) | ~76% | 推奨設定 |
| **LightGBM + flip + scaling (n_est=1000-10000)** | **~76-77%** | **最高精度** ✅ |
| DiagLDA | ~68-70% | 生成モデル、高速 |
| | | |
| ResNet-18 (参考) | ~95% | GPU、ニューラルネット |
| ViT (参考) | ~98% | GPU、Transformer |

> **重要**: 本実装はニューラルネットとの競争を目的としていません。**古典機械学習・CPU実装における理論的限界の探求**と**説明可能性**を重視しています。

---

## 実行例

### Example 1: 基本実行（パーセプトロン）

```bash
python mymodel2.py --epochs 8
```

**出力例**:

```
[cfg] featurizer dim = 3872  masks = 31  per-mask-dim = 113
[data] loaded: train=(50000, 32, 32, 3) test=(10000, 32, 32, 3) time=2.53s
[feat] build train: N=50000 D=3872 -> feat_cache\train_xxx.mmap chunk=512 workers=1
  train: 50000/50000 (525.9s)
[feat] done train: time=527.4s
[epoch 1/8] test_acc=65.23%
[epoch 8/8] test_acc=69.87%
[done] final_test_acc=70.12%
```

### Example 2: LightGBM（推奨設定・最高精度）

```bash
python mymodel2.py \
  --classifier lightgbm \
  --flip_train \
  --flip_eval \
  --diag_scale \
  --diag_use_var \
  --diag_eps 10 \
  --diag_scale_factor 32 \
  --lgbm_n_estimators 1000
```

**出力例**:

```
[cfg] featurizer dim = 3872  masks = 31  per-mask-dim = 113
[data] loaded: train=(50000, 32, 32, 3) test=(10000, 32, 32, 3) time=0.36s
[feat] load train from cache: feat_cache\train_xxx.mmap shape=[50000, 3872]
[feat] load test from cache: feat_cache\test_xxx.mmap shape=[10000, 3872]
[feat] load trainflip from cache: feat_cache\trainflip_xxx.mmap shape=[50000, 3872]
[feat] load testflip from cache: feat_cache\testflip_xxx.mmap shape=[10000, 3872]
[diag] fit invstd (use_var=True, eps=10.0) time=2.06s
[diag] wrote scaled feats: feat_cache\train_xxx_diag1_eps10_s32.mmap time=1.85s
[diag] wrote scaled feats: feat_cache\test_xxx_diag1_eps10_s32.mmap time=0.38s
[diag] wrote scaled feats: feat_cache\trainflip_xxx_diag1_eps10_s32.mmap time=2.06s
[diag] wrote scaled feats: feat_cache\testflip_xxx_diag1_eps10_s32.mmap time=0.40s
[lightgbm] n_estimators=1000 max_depth=8 num_leaves=256 lr=0.05
[lightgbm] using flip augmentation for training (concatenating train + trainflip)
[lightgbm] training done in 3460.8s
[lightgbm] using flip ensemble for evaluation
[done] final_test_acc=76.73%
```

### Example 3: モデル保存・読み込み

```bash
# 学習＆保存
python mymodel2.py \
  --classifier lightgbm \
  --save_model models/cifar10_lgbm.pkl

# 読み込み＆評価のみ
python mymodel2.py \
  --classifier lightgbm \
  --load_model models/cifar10_lgbm.pkl
```

---

## ファイル構成

```
.
├── mymodel2.py              # メイン実装（1075行）
├── README.md                # このファイル
├── THEORY.md                # 詳細な理論的背景（Clifford回路との接続）
├── LICENSE                  # MITライセンス
├── cifar10_data/            # データセット（自動ダウンロード）
│   └── cifar-10-batches-py/
├── feat_cache/              # 特徴量キャッシュ（自動生成）
│   ├── train_xxx.mmap
│   ├── test_xxx.mmap
│   ├── trainflip_xxx.mmap
│   ├── testflip_xxx.mmap
│   ├── *_diag1_eps10_s32.mmap  # 対角スケーリング済み
│   └── *.meta.json
└── models/                  # 学習済みモデル（オプション）
    └── *.pkl
```

---

## 設計哲学

### Why CPU-only?

本実装は意図的に**CPUのみ**を前提としています。これは性能制限ではなく：

- ✅ **再現性**（誰でも手元で動かせる）
- ✅ **設計の説明可能性**（各特徴量の意味が明確）
- ✅ **組み込み・省電力への適性**
- ✅ **GPUがなくても成立する認識原理の検証**

を目的とした設計上の選択です。

### 計算効率

- **特徴抽出**: O(n²)（二次形式の評価）
- **分類**: O(D)（線形スコアリング）または O(D log n)（LightGBM）
- **全体**: O(n²)で理論的最適

整数演算・ビット演算中心のため、NumPyの最適化を最大限活用。

### 特徴量設計の原則

1. **離散性**: 整数・ビット演算で完結
2. **局所性**: 小さなパッチ（2×2, 3×3）から統計抽出
3. **対称性**: D4群・反転不変性を明示的に組み込み
4. **二次性**: 計算複雑性O(n²)の境界内に留まる

---

## 理論的基盤

本実装の設計は、以下の数学的洞察に基づいています：

### 1. Clifford回路との接続

量子情報理論において、古典的に効率よくシミュレート可能なClifford回路は、**Z₂上の二次形式**として完全に記述できることが知られています（[arXiv:2601.15396](https://arxiv.org/abs/2601.15396)）。

```
Q(x) = x^T A x + b^T x + c  (mod 2)

where:
  A: 対称行列（エンタングルメント構造）
  b: 線形項（局所位相）
  c: 定数項
```

mymodel2の特徴抽出（LBP、2×2パターン等）は、この二次形式の具体的実装と見なせます。

### 2. 計算複雑性の境界

| 次数 | 表現力 | 計算量 | 対応するモデル |
|------|--------|--------|----------------|
| 1次 | 線形分離 | O(n) | 積状態、trivial |
| **2次** | **二次曲面、楕円体** | **O(n²)** | **Clifford、自由フェルミオン、mymodel** ✅ |
| 3次以上 | 任意多項式 | Exp(n) | 一般量子回路、#P-hard |

**二次に留まることで、古典計算の理論的最適性を達成**しています。

### 3. アーベル群と対称性

- **Z₂加法群**: バイナリマスクの演算（XOR = 加法 mod 2）
- **D4群**: 画像の幾何的対称性（回転・反転）
- **アーベル性**: 可換性により効率的な計算が可能

詳細は [THEORY.md](THEORY.md) を参照してください。

---

## 引用・参考文献

### 理論的基盤

1. **Quadratic Forms and Clifford Circuits**  
   arXiv:2601.15396 - "Quadratic tensors as a unification of Clifford, Gaussian, and free-fermion physics"

2. **Local Binary Patterns**  
   Ojala, T., et al. (2002) - "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns"

3. **Cellular Automata**  
   Wolfram, S. (1984) - "Cellular automata as models of complexity"

4. **Dihedral Group Symmetry**  
   Group theory applications in image processing

### 実装テクニック

1. **Welford's Online Algorithm**  
   ストリーミング分散計算（対角スケーリング）

2. **Averaged Perceptron**  
   Collins, M. (2002) - "Discriminative training methods for hidden Markov models"

---

## トラブルシューティング

### Q: 特徴抽出が遅い

```bash
# キャッシュを有効活用（2回目以降は高速）
python mymodel2.py --force_feat  # 初回のみ

# 並列化（実験的）
python mymodel2.py --num_workers 4

# チャンクサイズ調整
python mymodel2.py --chunk_feat 1024
```

### Q: メモリ不足

```bash
# チャンクサイズを小さく
python mymodel2.py --chunk_feat 256

# 特徴量を削減
python mymodel2.py --no_markov --color_hist_bins 2
```

### Q: 精度が低い

```bash
# 推奨設定を使用
python mymodel2.py \
  --classifier lightgbm \
  --flip_train --flip_eval \
  --diag_scale --diag_use_var

# LightGBMパラメータ調整
python mymodel2.py \
  --classifier lightgbm \
  --lgbm_n_estimators 2000 \
  --lgbm_max_depth 10
```

---

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照

---

## 関連プロジェクト

- **[mymodel1](https://github.com/corcondor/mymodel1)** - MNIST向け構造駆動型分類器（本プロジェクトの基礎）

---

## 今後の展開

### 短期（実装改善）

- [ ] 真の二次形式（A≠0）の導入
- [ ] スパース最適化（計算量削減）
- [ ] GPU対応（実験的）

### 中期（理論発展）

- [ ] 二次形式辞書の体系的生成
- [ ] 情報量ベースの特徴選択
- [ ] PAC学習フレームワークでの解析

### 長期（研究）

- [ ] 論文執筆: "Quadratic Forms for Image Classification"
- [ ] 他データセット（CIFAR-100、Tiny ImageNet）への拡張
- [ ] 量子インスパイア古典アルゴリズムとしての応用

---

## 謝辞

本プロジェクトは、量子情報理論と古典機械学習の境界を探求する試みです。Clifford回路理論とZ₂上の二次形式の深い洞察に感謝します。

---

**Documentation**: 詳細な理論的解説は [THEORY.md](THEORY.md)  
**Issues & Contributions**: Welcome! 理論的改善・実装最適化のPull Requestをお待ちしています。
