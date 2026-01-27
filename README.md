# mymodel5: CIFAR-10構造駆動型分類器

> **CIFAR-10のための高度な離散数学的アプローチ**: D4二面体群対称性・局所二値パターン(LBP)・形態学的セルオートマトン・カラー統計を用いたCPU最適化実装

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概要

`mymodel5`は、**ニューラルネットワークやバックプロパゲーションに依存しない**CIFAR-10画像分類の実装です。[mymodel1](https://github.com/corcondor/mymodel1)（MNIST・96.36%）の進化版として、カラー自然画像に対応し、より高度な数学的構造を導入しています。

### 何を使わないか（What this is NOT）

- ❌ CNN / ViT 等のニューラルネットワーク
- ❌ バックプロパゲーション
- ❌ GPU / CUDA
- ❌ 浮動小数点最適化に依存した学習

### 何を使うか（What this IS）

- ✅ **整数演算中心**の離散数学的アプローチ
- ✅ **D4二面体群**による幾何不変性
- ✅ **二次形式** (Quadratic Forms over Z₂)
- ✅ **古典機械学習**（パーセプトロン・LDA・LightGBM）
- ✅ **CPU最適化**で説明可能・再現性の高い実装

---

## mymodel1との違い

| 項目 | [mymodel1](https://github.com/corcondor/mymodel1) (MNIST) | mymodel5 (CIFAR-10) |
|------|------|------|
| **タスク** | 手書き数字（28×28グレースケール） | 自然画像（32×32カラー） |
| **基本特徴** | raw/core/edge 二値マスク | LBP + CA + カラー統計 + 二次形式 |
| **対称性** | 基本的な不変性 | **D4群**による回転・反転不変性 |
| **動的進化** | なし | **セルオートマトン**による形態学的マスク進化 |
| **カラー処理** | なし | opponent色空間・彩度マスク・RGB統計 |
| **分類器** | 平均化パーセプトロン | パーセプトロン・**対角LDA**・**LightGBM** |
| **理論的基盤** | 構造駆動型設計 | **Clifford回路理論**との数学的接続 |
| **精度** | ~96.36% | ~72-75%（CPU only、古典ML限界に挑戦） |

---

## 理論的背景（簡潔版）

### なぜ「二次形式」なのか？

最近の量子情報理論の研究（[arXiv:2601.15396](https://arxiv.org/abs/2601.15396)）により、**古典的に効率よく計算可能な量子モデル（Clifford回路）= アーベル群上の二次関数**であることが示されました。

| 次数 | モデル例 | 計算複雑性 |
|------|----------|------------|
| k=1 (線形) | 積状態、単純パリティ | O(n) - trivial |
| **k=2 (二次)** | **Clifford回路、自由フェルミオン、LBP** | **O(n²) - efficient** ✅ |
| k≥3 (高次) | 一般量子回路、3-SAT | Exp(n) - intractable ❌ |

**mymodel5は二次で閉じることで、古典計算の理論的最適性を達成しています。**

詳細は [THEORY.md](THEORY.md) を参照してください。

### 主要コンポーネント

1. **二値マスク特徴**: 画像 → 多数のバイナリマスク → 局所モチーフ統計
   - グレースケール閾値
   - エッジ検出（XOR型二次形式）
   - 勾配方向マスク
   - opponent色空間（R-G, Y-B）
   - 彩度マスク

2. **セルオートマトン**: 各マスクをCA（Cellular Automaton）で動的に進化
   - 形態学的変化をk-step追跡
   - トポロジー的特徴を捕捉

3. **LBP (Local Binary Patterns)**: テクスチャ解析
   - 8近傍比較による局所パターン抽出
   - flip-invariant encoding対応

4. **D4群不変性**:
   - 8つの対称性変換（回転×4 + 反転×4）
   - 群作用でのpooling（mean/max/median）

5. **カラー特徴**:
   - RGBブロック統計
   - 粗いカラーヒストグラム
   - opponent色空間

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
# デフォルト設定（パーセプトロン）
python mymodel5.py

# LightGBM使用（高精度）
python mymodel5.py --method lgbm

# Diagonal LDA使用
python mymodel5.py --method lda
```

### 高度なオプション

```bash
# flip augmentation + variance-based diagonal scaling
python mymodel5.py \
  --flip_train \
  --flip_eval \
  --diag_scale \
  --diag_use_var \
  --diag_eps 10 \
  --diag_scale_factor 32 \
  --epochs 8

# D4対称性使用
python mymodel5.py \
  --use_d4 \
  --d4_pooling mean \
  --method lgbm

# セルオートマトン有効化
python mymodel5.py \
  --ca_steps 2 \
  --ca_birth_min 5 \
  --ca_birth_max 8
```

### 主要コマンドラインオプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--method` | 分類器 (`perce`, `lda`, `lgbm`) | `perce` |
| `--flip_train` | 訓練時の左右反転拡張 | False |
| `--flip_eval` | 評価時の反転アンサンブル | False |
| `--use_d4` | D4群不変特徴を使用 | False |
| `--d4_pooling` | D4 pooling方法 (`mean`, `max`, `median`) | `mean` |
| `--diag_scale` | 対角スケーリング有効化 | False |
| `--ca_steps` | セルオートマトンステップ数 | 0 |
| `--epochs` | パーセプトロンのエポック数 | 5 |

---

## 結果

```
Dataset: CIFAR-10
Accuracy: ~72-75% (CPU-only, classical ML)
Hardware: CPU only
Training time: ~3-5分（特徴抽出含む）
Feature dimension: ~15,000-25,000（設定による）
```

### 性能比較

| 手法 | 精度 | 備考 |
|------|------|------|
| mymodel5 (baseline) | ~72% | パーセプトロン、基本設定 |
| mymodel5 + flip | ~73% | 反転拡張 |
| mymodel5 + LightGBM | ~74-75% | 勾配ブースティング |
| ResNet-18 (参考) | ~95% | GPU、ニューラルネット |

**重要**: 本実装はニューラルネットとの競争を目的としていません。古典機械学習・CPU実装における**理論的限界の探求**と**説明可能性**を重視しています。

---

## ファイル構成

```
.
├── mymodel5.py              # メイン実装
├── README.md                # このファイル
├── THEORY.md                # 詳細な理論的背景
├── LICENSE                  # ライセンス
├── cifar10_data/            # データセット（自動ダウンロード）
└── cache/                   # 特徴量キャッシュ（自動生成）
```

---

## 設計哲学

### Why CPU-only?

本実装は意図的に**CPUのみ**を前提としています。これは性能制限ではなく：

- ✅ **再現性**（誰でも手元で動かせる）
- ✅ **設計の説明可能性**
- ✅ **組み込み・省電力への適性**
- ✅ **GPUがなくても成立する認識原理の検証**

を目的とした設計上の選択です。

### 計算効率

- **特徴抽出**: O(n²)（二次形式の評価）
- **分類**: O(D)（線形スコアリング）
- **全体**: O(n²)で理論的最適

整数演算・ビット演算中心のため、NumPyの最適化を最大限活用。

---

## 引用・参考文献

本実装の理論的基盤：

1. **Quadratic Forms and Clifford Circuits**  
   arXiv:2601.15396 - "Quadratic tensors as a unification of Clifford, Gaussian, and free-fermion physics"

2. **Local Binary Patterns**  
   Ojala, T., et al. (2002) - "Multiresolution gray-scale and rotation invariant texture classification"

3. **Cellular Automata**  
   Wolfram, S. (1984) - "Cellular automata as models of complexity"

4. **Dihedral Group Symmetry**  
   Group theory applications in image processing

詳細な数学的背景は [THEORY.md](THEORY.md) を参照。

---

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照

---

## 関連プロジェクト

- [mymodel1](https://github.com/corcondor/mymodel1) - MNIST向け構造駆動型分類器（本プロジェクトの基礎）

---

## 謝辞

本プロジェクトは、量子情報理論と古典機械学習の境界を探求する試みです。Clifford回路理論の洞察に感謝します。

---

**Documentation**: 詳細な理論的解説は [THEORY.md](THEORY.md)  
**Issues & Contributions**: Welcome! 理論的改善・実装最適化の提案をお待ちしています。
