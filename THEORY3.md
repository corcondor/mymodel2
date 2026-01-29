THEORY3.md — Quantum-Inspired, Integer-First ML Stack for mymodel3
==================================================================

This note documents a practical, **quantum‑inspired → discrete → integer → tree‑based**
pipeline aligned with the following architecture:

Author: corcondor

```
mymodel3（量子インスパイア）
├─ 理論層（量子情報）
│   ├─ Clifford回路の数理
│   ├─ スタビライザ形式
│   └─ 群論（D4, ケーリーグラフ）
│
├─ 設計層（離散数学）
│   ├─ 二次形式 Z₂ 辞書
│   ├─ D4不変/同変特徴
│   └─ ペア相互作用
│
├─ 実装層（整数演算）
│   ├─ QuadraticFormZ2（整数のみ）
│   ├─ D4Symmetry（整数インデックス）
│   └─ ビット演算（XOR, AND）
│
└─ 学習層（木構造）
    └─ LightGBM（離散特徴最適化）
```

---

1. 理論層（量子情報）
--------------------

**Clifford / Stabilizer** のポイントは「古典計算機で効率シミュレーション可能」なこと。
これは“量子インスパイアだが計算は軽い”という設計思想の土台になる。

- Aaronson–Gottesman はスタビライザ回路の高速シミュレーションを示し、
  **ビット演算主体での効率化**が可能であることを明確化した。  
  参照: https://doi.org/10.1103/PhysRevA.70.052328

- Anders–Briegel は**グラフ状態表現**でさらに高速化できることを示し、
  「群構造／グラフ構造」の視点が実装に直結することを後押しした。  
  参照: https://doi.org/10.1103/PhysRevA.73.022334

- Gross の離散Wigner形式は、**スタビライザ状態 ↔ 離散確率的表現**の橋渡しを提供。
  “量子らしさ（干渉）を離散特徴に落とす”方向の根拠になる。  
  参照: https://doi.org/10.1063/1.2393152

**結論**  
Clifford/スタビライザは、理論的に“量子の骨格”を保持しつつ、
**古典計算で軽量・整数向き**に落とし込める。

---

2. 設計層（離散数学）
--------------------

### (A) 二次形式 Z₂ 辞書
量子安定子の「パリティ構造」をヒントに、**二次形式 (xᵀAx)** を Z₂/整数で設計する。
画像なら「近傍ペアの共起（AND）」や「排他的パリティ（XOR）」が対応する。

**実装的な解釈**
- `A` は 8近傍の**ペア相互作用**行列（疎行列）
- `x` は二値マスク（0/1）
- `xᵀAx` は “局所的な相互作用の総量” で、スタビライザの相関構造に近い

### (B) D4 不変/同変特徴
群作用の軌道上で**平均/最小/最大プーリング**を行うと、
**回転・反転に不変な特徴**が得られる。D4は実装が軽く、
整数演算で効率的に回せる。

---

3. 実装層（整数演算）
--------------------

### (A) QuadraticFormZ2（整数のみ）
- 8近傍ペアの **AND** 共起を二次項として集計  
- 必要なら **XOR パリティ**を追加し、スタビライザの「符号」的成分を近似

### (B) D4Symmetry（整数インデックス）
- 画像のD4変換を **配列インデックス操作のみ**で実装  
- 8変換の軌道上でプーリング

### (C) 参考ライブラリ
- **Stim**：超高速スタビライザシミュレーター（実装思想の参考になる）  
  https://github.com/quantumlib/Stim  
  論文: https://doi.org/10.22331/q-2021-07-06-497

- **PyZX**：Clifford回路の簡約/等価変換に強い（設計検証に便利）  
  https://github.com/Quantomatic/pyzx

---

4. 学習層（木構造）
-------------------

**LightGBM** はヒストグラム分割の GBDT で、離散特徴との相性が良い。  
整数特徴をそのまま入力でき、CPUでも高速。  

- LightGBM（公式論文）  
  https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

- LightGBM 公式ドキュメント  
  https://lightgbm.readthedocs.io/en/latest/

---

実装への落とし込み（mymodel3の設計指針）
----------------------------------------

1) **D4 軌道特徴 + 二次形式Z₂**  
   - 画像 → 二値マスク群 → D4 変換 → 軌道プーリング  
   - 近傍AND/XORで “量子風” の相互作用を作る

2) **整数統計の多様化**  
   - 局所パターン (2×2, 3×3)  
   - 二次形式（ペア相互作用）  
   - 必要なら離散Wigner的な “符号” を模した奇偶統計

3) **学習は木構造で固定**  
   - LightGBM を主軸にし、NN/Transferは避ける  
   - CPUで回せるスケールに合わせて特徴設計を調整

---

具体的なアルゴリズム（設計～学習の流れ）
----------------------------------------

**入力**: 画像 `I (32×32×3)`  
**出力**: ラベル予測

1. **前処理（整数化）**
   - `gray = gray_u8(I)` で輝度化（uint8）
   - 閾値群 `T = {t1, t2, ...}` による二値マスク生成

2. **D4軌道生成**
   - `orbit = {g(I) | g ∈ D4}`  
   - D4の8変換を使い、同変/不変の両方に対応

3. **マスク統計（整数）**
   - `pooled_proj_and_grid()` による行/列/グリッド統計
   - `pat2x2_hist16()` による2×2パターン

4. **二次形式Z₂（ペア相互作用）**
   - 近傍ペアを **AND** で集計（8方向）
   - 追加で **XOR** も集計すれば「符号／パリティ」的成分が得られる

5. **特徴ベクトル合成**
   - すべて int16 にクリップして連結
   - 必要に応じて対角スケーリング（整数近似）

6. **学習（LightGBM）**
   - D4軌道平均で不変特徴にし、GBDTで分類

---

疑似コード（整数中心）
----------------------

```
function extract_features(I):
    g = gray_u8(I)                      # uint8
    masks = []
    for t in thresholds:
        B = (g >= t)                    # uint8 mask
        masks.append(B)

    feats = []
    for B in masks:
        cnt, rs, cs, grid = pooled_proj_and_grid(B)
        pat = pat2x2_hist16(B)
        quad_and = quad_pairs_and(B)    # 8方向ペア相互作用
        quad_xor = quad_pairs_xor(B)    # optional
        feats.append(concat(cnt, rs, cs, grid, pat, quad_and, quad_xor))

    return concat_all(feats)            # int16


function classify(I):
    orbit = D4_orbit(I)                 # 8 variants
    F = [extract_features(o) for o in orbit]
    F_pool = mean(F)                    # D4-invariant
    return LightGBM.predict(F_pool)
```

---

実装例（mymodel3の整数二次形式の核）
------------------------------------

```
def quad_pairs_and(B):
    # B: uint8 {0,1}
    feats = []
    feats.append((B[:, :-1] & B[:, 1:]).sum())     # E-W
    feats.append((B[:-1, :] & B[1:, :]).sum())     # N-S
    feats.append((B[:-1, :-1] & B[1:, 1:]).sum())  # SE-NW
    feats.append((B[:-1, 1:] & B[1:, :-1]).sum())  # SW-NE
    # 逆方向は同じ値になるが、対称を明示したい場合は追加
    return np.array(feats, dtype=np.int16)
```

---

参考文献（論文／公式）
----------------------
- Aaronson, S. & Gottesman, D. (2004). Improved Simulation of Stabilizer Circuits.  
  https://doi.org/10.1103/PhysRevA.70.052328

- Anders, S. & Briegel, H. J. (2006). Fast simulation of stabilizer circuits.  
  https://doi.org/10.1103/PhysRevA.73.022334

- Gross, D. (2006). Hudson’s theorem for finite-dimensional quantum systems.  
  https://doi.org/10.1063/1.2393152

- Chen, T., Guestrin, C. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.  
  https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

- LightGBM Documentation  
  https://lightgbm.readthedocs.io/en/latest/

- Stim: a fast stabilizer circuit simulator (Quantum 2021)  
  https://doi.org/10.22331/q-2021-07-06-497

- PyZX (ZX-calculus toolkit)  
  https://github.com/Quantomatic/pyzx
