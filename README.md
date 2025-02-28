# 先行研究｜SVMを用いた狭帯域光検出

## 学習
- OutCSV_RGBH.pyにより16種の色特徴量抽出
- train.pyにより学習モデルと特徴量スケーリングモデル作成
- olympus, fujifilm両方のmask画像が必要

## 修士中間発表(mid_presentation)で使用
### データ処理
- outcsv_rgbh_for_mid_presentation.py
  - 動画から色特徴量（RGB, H）を抽出
  - 結果をCSVファイルとして保存
  - ラベルの集計結果も保存

### 学習
- train_for_mid_presentation.py
  - 4分割交差検証用のデータセット分割
  - SVMモデルの学習
  - スケーリングモデルの保存

### テスト
- test_for_mid_presentation.py
  - 学習済みモデルを用いた予測
  - 混同行列の生成
  - 評価指標（適合率・再現率・F1スコア）の算出
  - 予測結果のタイムライン可視化（SVG形式）

### ディレクトリ構造
```
mid_presentation/
├── outcsv_rgbh_for_mid_presentation.py
├── train_for_mid_presentation.py
├── test_for_mid_presentation.py
└── results_visualizer.py
```