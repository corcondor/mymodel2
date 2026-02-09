# mymodel2

このリポジトリは、CIFAR-10用の特徴抽出とLightGBMによる評価を行うための実験用コードを含みます。

- 特に `feature_grammar_v7_reward.py` は大規模な特徴抽出（chunked, mmap を利用）を実行し、一度抽出した特徴をキャッシュして再利用します。
- キャッシュの設計と手順は `python/README_v7.md` にまとめています。

使い方（簡易）:

1. 抽出: `python python/feature_grammar_v7_reward.py --feat-cache <キャッシュ保存先>`
2. 抽出済みキャッシュから学習: スクリプト内オプションを参照してください。

備考:
- キャッシュは `.npy` チャンクとして生成され、ネットワークドライブ上でのmmap問題を避けるため一時的に `/tmp` に書き出してから移動します。
- 詳細な設計意図や運用手順は `python/README_v7.md` を参照してください。

---

