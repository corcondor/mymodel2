# Feature Grammar V7 (Reward-ready) — README

概要
- プロジェクト: Feature Grammar V7（CIFAR-10 特徴抽出 + LightGBM 用）
- 目的: 1回だけ大きな特徴抽出（約12GB）を行い、以降はキャッシュを読み込んで高速に学習・評価を繰り返す
- 実装言語: Python

主要設計（重要）
- プログラム総数: 1095 自動生成特徴プログラム
- チャンク処理: 100プログラム/チャンク → 全11チャンク
- キャッシュ形式: `.npy`（mmap可能）
- 書き込みフロー（ボトルネック対策）:
  1. 各チャンクはローカル一時領域 `/tmp` に `np.lib.format.open_memmap()` で作成
  2. `flush()` でバッファを同期
  3. 同期後にGoogle Driveのキャッシュディレクトリへ `shutil.move()` で移動
  - 理由: OneDrive はほぼ満杯 → `/tmp` に書き、容量に余裕のある Google Drive に移すのが安全かつ高速
- 読み込みフロー（学習時）:
  - Google Drive 上の `.npy` を `np.load(..., mmap_mode='r')` で順次読み込み
  - mmapからメモリへコピーしている間に OS がネットワークから次チャンクをプリフェッチするため、パイプライン効果が期待できる

設定と既定値
- デフォルト特徴キャッシュディレクトリ（スクリプト内のデフォルト）:
  `/Users/yuta/Library/CloudStorage/GoogleDrive-imtceed@gmail.com/マイドライブ/cifar10_cache`
- チャンクサイズ: `CHUNK_SIZE = 100`
- 並列ワーカー: `--n-jobs`（デフォルト 3 を推奨）

実行例
- フル抽出（nohupでバックグラウンド実行の推奨）:

```bash
cd python
nohup python3 feature_grammar_v7_reward.py --data-dir ../cifar10_data --n-jobs 3 --no-reward-log > fgv7_final.log 2>&1 &
```

- 途中再開・部分読み込みの例（引数で `--feat-cache` を上書き可能）:

```bash
python3 feature_grammar_v7_reward.py --data-dir ../cifar10_data --n-jobs 3 --feat-cache /path/to/your/cache
```

ログとトラブルシュート
- 実行ログ: `fgv7_final.log`（実行開始直後に記録されます）
  - チャンク単位で `Chunk X/11 ...` の出力が出る
- よくある警告:
  - `RuntimeWarning: Precision loss occurred in moment calculation`（skew/kurtosis）
    - 意味: 一部の統計量計算で数値精度の問題が出ているが、処理自体は継続可能
    - 必要なら該当プログラムを無効化して再実行可能
- 失敗時のチェック項目:
  1. `/tmp` に十分な空きがあるか（1チャンクあたり ~1GB が必要）
  2. Google Drive がローカルにマウントされ、書き込み可能か
  3. `fgv7_final.log` の直近出力（エラーとトレースバック）を確認

ファイル配置（実行後に期待されるもの）
- Google Drive キャッシュディレクトリ内:
  - `train_chunk_000.npy` ... `train_chunk_010.npy`
  - `test_chunk_000.npy` ... `test_chunk_010.npy`
  - `metadata.json`（チャンク数、総次元数、サンプル数等）

将来の作業フロー（概要）
1. 全チャンク抽出（現在実行中）
2. Google Drive 上の全チャンク存在を検証
3. `np.load(..., mmap_mode='r')` でチャンクを順次読み込み、`feats_train` / `feats_test` を組み立て
4. LightGBM で訓練・評価（既存コード内で実行）
5. 不要になった `/tmp` の mmap ファイルを削除

GitHubへあげる手順（ローカルで行う）
```bash
cd python
git add README_v7.md feature_grammar_v7_reward.py
git commit -m "Add README for V7 feature extraction and caching strategy"
git push origin main
```
（`git push` のブランチ名は環境に合わせて変更してください）

補足メモ
- `.npy` を使う理由: 圧縮形式の `.npz` は mmap 非対応のため、mmapへの利点を活かすには `.npy` が必須
- 並列化の注意: `joblib` の `n_jobs` は CPU/IO バランスを見て調整してください（現在は `3` を推奨）

ファイル: python/feature_grammar_v7_reward.py
- 既に下記の最適化を適用済み:
  - 書き込み: `/tmp` で `open_memmap` → `flush()` → `shutil.move()` → Google Drive
  - 読み込み: `np.load(..., mmap_mode='r')` によるパイプライン読み込み

---
作成済みファイル: `python/README_v7.md`

必要なら次に `git commit` → `git push` を代行します（認証が必要なので、実行してよければ教えてください）。