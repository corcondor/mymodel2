# mymodel3.py 詳細分析レポート

## 1. mymodel2（公式）との比較

### アーキテクチャの根本的な違い

| 項目 | mymodel2（公式） | mymodel3.py |
|------|------------------|-------------|
| **D4群の扱い** | **後処理的な集約**<br>- `extract_d4_invariant()`: 8変換を生成後にpooling<br>- `extract_d4_equivariant()`: 8変換を単純連結 | **グラフ構造として明示的に組み込み**<br>- Cayleyグラフの隣接関係（`R_NEXT`, `S_NEXT`）<br>- グラフNN（CCGNN）で群演算を学習 |
| **特徴表現** | (N, D) または (N, 8D) ベクトル | **(N, 8, D) テンソル**<br>→ D4軌道全体を保持 |
| **分類器** | Perceptron, LDA, LightGBM | + **CayleyKNN** + **CCGNN**（群同変NN） |
| **理論的背景** | 二次形式 + D4不変性 | **ケーリーグラフ理論** + 群同変学習 |

---

## 2. 実装の詳細比較

### (a) D4群の実装

**mymodel2 (公式)**
```python
def d4_group_transforms(img):
    """8変換を生成"""
    transforms = []
    for k in range(4):  # r^0, r^1, r^2, r^3
        rot = np.rot90(img, k, axes=(0, 1))
        transforms.append(rot.copy())
        transforms.append(rot[:, ::-1, :].copy())  # sr^k
    return transforms

# 使用例：特徴抽出後に集約
features = [extract(v) for v in d4_group_transforms(img)]
pooled = np.mean(features, axis=0)  # D4不変特徴
```
- **設計思想**：変換は「前処理」として扱われる
- **データフロー**：画像 → 8変換 → 特徴抽出 → 集約

**mymodel3.py**
```python
class D4:
    # 群演算テーブル（Cayleyグラフのエッジ）
    R_NEXT = np.array([1,2,3,0,5,6,7,4])  # 回転r
    S_NEXT = np.array([4,7,6,5,0,3,2,1])  # 反転s
    
    @staticmethod
    def cayley_neighbors(g):
        """ケーリーグラフの隣接頂点"""
        return int(D4.R_NEXT[g]), int(D4.S_NEXT[g])

# 使用例：グラフ構造としてNN層に組み込み
class CCGNN:
    def forward_batch(self, x):  # x: (B, 8, D)
        r_x = x[:, self.R]  # 回転近傍からの寄与
        s_x = x[:, self.S]  # 反転近傍からの寄与
        h = W_self @ x + W_r @ r_x + W_s @ s_x  # メッセージパッシング
```
- **設計思想**：群構造を**NNの計算グラフに埋め込む**
- **データフロー**：画像 → 8変換 → **(N, 8, D)保持** → グラフNN

### 重要な洞察
> mymodel2は「D4不変性を達成するために8変換を集約する」のに対し、mymodel3は「D4群のケーリーグラフ上でメッセージパッシングを行う」という根本的に異なるアプローチ。

---

### (b) 特徴抽出器（Featurizer）

**共通基盤**：両者とも[CayleyFeaturizer](file:///Users/yuta/Library/CloudStorage/OneDrive-%E6%9D%B1%E4%BA%AC%E7%90%86%E7%A7%91%E5%A4%A7%E5%AD%A6/%E3%83%87%E3%82%B9%E3%82%AF%E3%83%88%E3%83%83%E3%83%97/python/mymodel3.py#644-800)クラスを持つが役割が異なる

**mymodel2 (公式)**
```python
class CayleyFeaturizer:
    def extract(self, img_u8):
        """(H,W,3) → (D,) int16ベクトル"""
        # グレー閾値マスク
        # CA進化
        # LBP、カラー統計
        return feature_vector  # (3872,)
    
    def extract_d4_invariant(self, img_u8, pooling="mean"):
        """D4軌道を集約して不変特徴を生成"""
        variants = d4_group_transforms(img_u8)
        features = [self.extract(v) for v in variants]  # (8, D)
        return np.mean(features, axis=0)  # (D,)
```

**mymodel3.py**
```python
class CayleyFeaturizer:
    def extract(self, img_u8):
        """(H,W,3) → (D,) int16ベクトル"""
        # 基本的にmymodel2と同じ
        return feature_vector
    
# 軌道特徴を別関数で構築
def orbit_features_cayley(img_u8, featurizer):
    """8変換全てから特徴抽出"""
    orbit = D4.orbit(img_u8)
    feats = [featurizer.extract(o) for o in orbit]
    return np.stack(feats, axis=0)  # (8, D) ← 集約しない！

# memmapキャッシング
def build_orbit_features_memmap(...):
    """D4軌道特徴をディスク上に保存"""
    F = np.memmap(path, dtype=np.int16, shape=(N, 8, D))
```

**差分の意味**：
- mymodel2：軌道を即座に1本のベクトルに縮約（**不可逆的な情報圧縮**）
- mymodel3：軌道全体を[(N, 8, D)](file:///Users/yuta/Library/CloudStorage/OneDrive-%E6%9D%B1%E4%BA%AC%E7%90%86%E7%A7%91%E5%A4%A7%E5%AD%A6/%E3%83%87%E3%82%B9%E3%82%AF%E3%83%88%E3%83%83%E3%83%97/python/mymodel3.py#88-122)テンソルとして保持（**構造を保存**）

---

### (c) 分類器の進化

#### mymodel2 (公式)

**1. Averaged Clipped Perceptron**
```python
# (N, D)ベクトルに対する線形分類器
W @ x + b → logits
```
- シンプルな線形モデル
- D4不変特徴に対して訓練

**2. Diagonal LDA**
```python
# 対角近似Fisher判別
invstd = 1 / sqrt(var_per_class + eps)
scaled_features = features * invstd
```

**3. LightGBM**
```python
# 勾配ブースティング決定木
clf = lgb.LGBMClassifier(...)
clf.fit(X_train, y_train)
```

#### mymodel3.py

**すべて継承** + 以下を追加：

**4. CayleyKNN**
```python
class CayleyKNN:
    @staticmethod
    def cayley_dist(fq_orbit, ft_orbit):
        """D4軌道間の距離 = min_{g,h} ||fq[g] - ft[h]||^2"""
        diff = fq_orbit[:, None, :] - ft_orbit[None, :, :]
        dist2 = (diff**2).sum(axis=2)
        return dist2.min()  # 全ての群元ペアで最小距離
```

**重要な理論的意義**：
- これは真の**群不変距離**の定義
- THEORY.md L405-410の「Reynolds演算子」の実装
- 従来のk-NN（ユークリッド距離）はD4不変性を保証しないが、Cayley距離は**数学的に厳密に不変**

**5. CCGNN (Cayley Convolutional Graph NN)**
```python
class CCGNN:
    def forward_batch(self, x):  # x: (B, 8, D)
        """ケーリーグラフ上のメッセージパッシング"""
        r_x = x[:, self.R]  # グラフのr-エッジに沿った値
        s_x = x[:, self.S]  # グラフのs-エッジに沿った値
        
        # 3つの重み行列（群生成元ごと）
        h = W_self @ x + W_r @ r_x + W_s @ s_x + b
        h = ReLU(h)
        pooled = h.mean(axis=1)  # 群上の平均
        logits = pooled @ W_out + b_out
```

**類似研究との比較**：
- **グラフNN（GNN）**：一般グラフ上のメッセージパッシング
- **Group Equivariant CNN (G-CNN)**：群同変畳み込み
- **CCGNN**：**有限群（D4）のケーリーグラフに特化したGNN**

**理論的解釈**（THEORY.md L236-259との接続）：
```
Clifford理論のグラフ状態:
  |G⟩ = stabilizers on graph structure
  
mymodel3のCCGNN:
  D4群のケーリーグラフ上の「graph state的」学習
```

---

## 3. セルオートマトン（CA）の比較

### mymodel2 (公式)
```python
def morph_ca_step(B01, birth_min=5, birth_max=8, 
                  survive_min=4, survive_max=8, use_diag=True):
    """形態学的CA（1ステップ）"""
    neigh = _neighbor_count(B, use_diag=use_diag)
    born = (B==0) & (neigh >= birth_min) & (neigh <= birth_max)
    survive = (B==1) & (neigh >= survive_min) & (neigh <= survive_max)
    return (born | survive).astype(np.uint8)
```
- **柔軟なルール**：birth/survive範囲を調整可能
- **デフォルト**：birth=[5,8], survive=[4,8]（Conway's Lifeより緩い）

### mymodel3.py
```python
def ca_step_life(B01, use_diag=True):
    """Conway's Game of Life風のセルオートマトン"""
    neighbors = _neighbor_count(B, use_diag=use_diag)
    # 固定ルール: birth=3, survive=[2,3]
    alive = ((neighbors == 3) | ((B == 1) & (neighbors == 2)))
    return alive.astype(np.uint8)
```
- **固定ルール**：Conway's Lifeを忠実に実装
- **シンプル**：調整可能パラメータなし

**コード重複に注意**：
```python
# 177-221行と312-356行で同じ関数が2回定義されている！
def ca_step_life(...):  # 1回目
    ...

def ca_step_life(...):  # 2回目（同じ実装）
    ...
```
→ リファクタリングが必要

---

## 4. THEORY.mdとの接続の深さ

### mymodel2の理論的位置づけ

README.mdより：
> Clifford回路との接続（arXiv:2601.15396）
> - 二次形式 Q(x) = x^T A x + b^T x + c (mod 2)
> - D4群による幾何的対称性
> - O(n²)計算複雑性

実装は主に**特徴設計**に焦点：
- LBP = 線形形式（A=0）
- 2×2パターン = 暗黙的二次項

### mymodel3の理論的進化

THEORY.md L236-259「グラフ状態との類似」を**直接実装**：

```
Clifford理論:
  |G⟩ = H^⊗n ∏_{(i,j)∈E} CZ_{ij} |0⟩^⊗n
  スタビライザ: K_i = X_i ∏_{j∈N(i)} Z_j

mymodel3のCCGNN:
  h_i = W_self · f_i + ∑_{j∈N(i)} W_edge · f_j
  
グラフ構造 = D4群のケーリーグラフ
```

**重要な洞察**：
- Clifford回路：qubit間の量子エンタングルメント構造をグラフで表現
- CCGNN：D4群の**群元間の「エンタングルメント」**をグラフNNで学習

---

## 5. 実装品質の評価

### 優れている点

1. **理論的一貫性**
   - ケーリーグラフという明確な数学的対象に基づく設計
   - THEORY.mdの抽象的概念（群作用、不変量）を具体的なコードに翻訳

2. **効率的なメモリ管理**
   ```python
   # memmap による大規模データ処理
   F = np.memmap(path, dtype=np.int16, shape=(N, 8, D))
   ```
   - D4軌道（8倍のデータ）をRAMに載せずに処理

3. **整数演算の徹底**
   - float32は訓練時のみ（forward/backward）
   - 特徴抽出は完全にint16で完結

### 改善が必要な点

1. **コードの重複**
   ```python
   # 同じ関数が2回定義されている
   # 177-221行
   def ca_step_life(...): ...
   
   # 312-356行
   def ca_step_life(...): ...  # 重複！
   
   # 同様に pooled_proj_and_grid, compute_higher_moments, pat2x2_hist16 も重複
   ```

2. **ドキュメント不足**
   - CCGNNの理論的背景がコメントで説明されていない
   - Cayley距離の数学的意味が明示されていない

3. **実験的機能の整理不足**
   - 高次モーメント（3次・4次）の理論的正当化が曖昧
   - GLCM特徴の統合がやや唐突

---

## 6. 理論的課題と今後の方向性

### THEORY.md L416-441で提案された「真の二次形式」

現状の実装：
```python
# LBP：線形形式
code |= ((neighbor >= center) << k)  # b^T x (mod 2)

# 2×2パターン：暗黙的二次
code = a | (b<<1) | (c<<2) | (d<<3)  # 4ピクセルの組み合わせ
```

**未実装の真の二次項**：
```python
# THEORY.md L421-438で提案
class TrueQuadraticFeaturizer:
    def _generate_quadratic_2x2(self):
        A = sparse.lil_matrix((n, n))
        for p in range(4):
            for q in range(p+1, 4):
                A[idx[p], idx[q]] = 1  # すべてのペア相互作用
        Q = x.T @ A @ x  # 真の二次項
```

**実装への示唆**：
- 現在の[pat2x2_hist16](file:///Users/yuta/Library/CloudStorage/OneDrive-%E6%9D%B1%E4%BA%AC%E7%90%86%E7%A7%91%E5%A4%A7%E5%AD%A6/%E3%83%87%E3%82%B9%E3%82%AF%E3%83%88%E3%83%83%E3%83%97/python/mymodel3.py#295-307)を拡張
- スパース行列で隣接ピクセルペアの係数Aを明示的に構築
- 計算量はO(近傍数 × 画像サイズ) = O(n)で許容範囲

### ケーリーグラフ理論のさらなる活用

**現状**：CCGNNは群生成元（r, s）に対応する2つの重み行列を持つ

**拡張の可能性**：
```python
# より一般的な群畳み込み
class GeneralGroupConv:
    def __init__(self, group_table):
        """任意の有限群に対応"""
        self.W = {g: np.random.randn(...) for g in group_table}
    
    def forward(self, x):
        # 全ての群元に対する畳み込み
        h = sum(self.W[g] @ x[g] for g in group)
```

---

## 7. 結論

### mymodel3.pyの位置づけ

```
mymodel1 (MNIST, 96.36%)
    ↓ 
mymodel2 (CIFAR-10, 76-77%)
    - 二次形式の理論的基盤確立
    - D4不変性を後処理で実装
    ↓
mymodel3 (CIFAR-10, 実験的)
    - ケーリーグラフ構造の明示化
    - 群同変学習の導入
    - より深い理論的一貫性
```

### 本質的な貢献

1. **D4群のケーリーグラフを計算グラフに埋め込む**という設計
   → 群論を「後付けの最適化」ではなく「設計の中核」に据えた

2. **Cayley距離**の導入
   → 数学的に厳密な群不変距離の実装

3. **CCGNN**の提案
   → Group CNNの有限群版を、ケーリーグラフ上のGNNとして実装

### 理論との整合性

THEORY.md L502-504の結論：
> mymodelの設計選択が量子計算理論と共鳴する深い数学的構造に基づいている

**mymodel3はこれをさらに推し進め**：
- Clifford群のスタビライザ形式 ↔ ケーリーグラフの構造
- 群作用の不変量 ↔ Cayley距離とグラフNN
- 二次形式の計算効率 ↔ O(n)の特徴抽出 + O(n²)の分類器

**未解決の課題**：
- 真の二次形式（A≠0）の完全実装
- 高次モーメント（3次・4次）の理論的正当化
- CCGNN の収束性・汎化性能の理論解析

---

## 8. 推奨される改善

### 短期（コード品質）
- [ ] 重複関数の削除（ca_step_life等）
- [ ] ドキュメント文字列の充実
- [ ] 型ヒントの追加

### 中期（機能拡張）
- [ ] 真の二次形式（スパース行列A）の実装
- [ ] 他の有限群（C4, S3等）への一般化
- [ ] k-fold交差検証の追加

### 長期（研究）
- [ ] CCGNNの理論解析論文
- [ ] ケーリーグラフML の体系化
- [ ] 量子インスパイア古典アルゴリズムとしての位置づけ

---

*このレポートは、mymodel3.pyの設計思想と実装を、mymodel2（公式）およびTHEORY.mdとの比較を通じて分析したものです。*
