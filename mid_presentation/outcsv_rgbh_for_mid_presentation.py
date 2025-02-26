# 色特徴量抽出
# 
# 画像フォルダ内の画像の色特徴量を算出し，csvファイルに出力する
# 
# 入力ディレクトリ構造
# ├── data
# │   ├── video_name_A
# │   │   ├── video_name_A_0.png
# │   │   ├── video_name_A_0.png
# │   │   ...
# │   ├── video_name_B
# │   │   ├── video_name_B_0.png
# │   │   ├── video_name_B_0.png
# │   │   ...
# │   ├── video_name_A.csv
# │   │── video_name_B.csv
#      ...
# 
# 出力ディレクトリ構造
# ├── output_image_folder
# │   ├── video_name_A_rgbh.csv
# │   ├── video_name_B_rgbh.csv
#     ...

from pathlib import Path
import csv
import cv2
import pandas as pd
import numpy as np
import scipy.stats as sstats
import cmath
import logging


# パス
DATA_DIR = "/home/tanaka/demo_data"
# DATA_DIR = "/home/tanaka/debug_data"
OUTPUT_DIR = "demo_data_rgbh_csv"
# OUTPUT_DIR = "debug_data_rgbh_csv"

# resize (deep learning と統一)
IMG_SIZE = 224


# ラベルデータフレームから主クラス、サブクラスの画像枚数を集計する
def count_labels(label_df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Parameters
    ----------
    label_df : pd.DataFrame
        ラベルデータフレーム（ヘッダーなし）
        1列目: 画像名
        2列目以降: ラベル

    Returns
    -------
    tuple[dict, dict]
        (主クラスの集計結果辞書, サブクラスの集計結果辞書)
        辞書の形式: {クラス番号: 画像枚数}
    """
    main_class_counts = {}  # 主クラス（0-5）の集計用
    sub_class_counts = {}   # サブクラス（6-14）の集計用
    
    # 2列目以降のラベルを処理
    for _, row in label_df.iterrows():
        labels = row[1:]  # 2列目以降のラベル
        
        for value in labels:
            if pd.notna(value):  # 欠損値でない場合
                try:
                    value = int(value)
                    if 0 <= value <= 5:
                        main_class_counts[value] = main_class_counts.get(value, 0) + 1
                    elif 6 <= value <= 14:
                        sub_class_counts[value] = sub_class_counts.get(value, 0) + 1
                except ValueError:
                    continue

    # クラス番号をソート
    main_class_counts = dict(sorted(main_class_counts.items()))
    sub_class_counts = dict(sorted(sub_class_counts.items()))
    
    return main_class_counts, sub_class_counts


def save_label_counts(main_counts: dict, sub_counts: dict, out_dir: Path, folder_name: str):
    """集計結果をCSVファイルに保存する"""
    with open(out_dir / f"{folder_name}_label_counts.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['クラス', '画像枚数'])
        writer.writerow(['主クラス'])
        for label, count in sorted(main_counts.items()):
            writer.writerow([label, count])
        writer.writerow(['サブクラス'])
        for label, count in sorted(sub_counts.items()):
            writer.writerow([label, count])


# 角度の平均と標準偏差
def mean_angle(angles):
    a = np.deg2rad(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) , 1)


# 角度の標準偏差
def std_angle(angles):
    a = np.deg2rad(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r = abs(angles_complex.sum()) / angles.size
    std = np.sqrt(-2 * np.log(r))
    return round(np.rad2deg(std), 2)


# HSV色空間の特徴量を算出
def analysis_hsv(image_path):
    img = cv2.imread(str(image_path))
    resize_img = cv2.resize(img, dsize=(IMG_SIZE, IMG_SIZE))

    #RGB
    r,g,b = resize_img[:,:,2], resize_img[:,:,1], resize_img[:,:,0]
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 255])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 255])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 255])

    mean_r = r.mean()
    mean_g = g.mean()
    mean_b = b.mean()

    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)

    # 歪度
    skew_r = sstats.skew(hist_r)
    skew_g = sstats.skew(hist_g)
    skew_b = sstats.skew(hist_b)

    # 尖度
    kurto_r = sstats.kurtosis(hist_r)
    kurto_g = sstats.kurtosis(hist_g)
    kurto_b = sstats.kurtosis(hist_b)

    img_hsv = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)
    h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
    hist_h = cv2.calcHist([h], [0], None, [360], [0, 360])

    # H
    mean_h = mean_angle(h*2)
    std_h = std_angle(h*2)
    # 歪度
    skew_h = sstats.skew(hist_h)
    # 尖度
    kurto_h = sstats.kurtosis(hist_h)

    return mean_r,std_r,skew_r,kurto_r,\
           mean_g,std_g,skew_g,kurto_g,\
           mean_b,std_b,skew_b,kurto_b,\
           mean_h,std_h,skew_h,kurto_h


def write_csv(image_paths: tuple[Path], label_df: pd.DataFrame, folder_name: str, dir_path: Path, logger: logging.Logger):
    """
    画像ファイルのリストから色特徴量を算出し，CSVファイルに出力する

    Parameters
    ----------
    images: 画像ファイルのリスト(image_name.png)
    label_df: ラベルデータフレーム
    folder_name: フォルダ名
    dir_path: 出力先のパス
    """
    # CSVファイルのヘッダーを書き込む
    with open(dir_path / (folder_name + '_rgbh.csv'), 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'mean_r', 'std_r', 'skew_r', 'kurto_r', 
                        'mean_g', 'std_g', 'skew_g', 'kurto_g',
                        'mean_b', 'std_b', 'skew_b', 'kurto_b',
                        'mean_h', 'std_h', 'skew_h', 'kurto_h',
                        'main_label', 'sub_label'])

    # 画像名をキーとしてラベルを検索できるように辞書を作成
    label_dict = {}
    for _, row in label_df.iterrows():
        image_name = row[0] # 1列目の画像名.pngあり
        labels = row[1:]   # 2列目以降のラベル
        label_dict[image_name] = labels

    for image_path in image_paths:
        try:
            mean_r,std_r,skew_r,kurto_r,\
            mean_g,std_g,skew_g,kurto_g,\
            mean_b,std_b,skew_b,kurto_b,\
            mean_h,std_h,skew_h,kurto_h \
            = analysis_hsv(image_path)
            
            # 画像名を取得
            image_name = image_path.stem + image_path.suffix

            # 対応するラベルを取得
            if image_name not in label_dict:
                logger.warning(f"ラベルが見つかりません: {image_name}")
                continue
                
            labels = label_dict[image_name]
            
            # 主クラス（0-5）とサブクラス（6-14）を判別
            main_label = ''
            sub_label = ''
            for value in labels:
                if pd.notna(value):  # 欠損値でない場合
                    try:
                        value = int(value)  # 文字列を整数に変換
                        if 0 <= value <= 5:
                            main_label = value
                        elif 6 <= value <= 14:
                            sub_label = value
                    except ValueError:
                        continue

            with open(dir_path / (folder_name + '_rgbh.csv'), 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([image_name, \
                                 mean_r,std_r,skew_r[0],kurto_r[0],\
                                 mean_g,std_g,skew_g[0],kurto_g[0],\
                                 mean_b,std_b,skew_b[0],kurto_b[0],\
                                 mean_h,std_h,skew_h[0],kurto_h[0],\
                                 main_label, sub_label])
                
        except Exception as e:
            logger.error(f"画像処理中にエラーが発生しました {image_path.stem}: {str(e)}")
            continue


def setup_logger(out_dir: Path):
    # ロギングの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # フォーマッタの作成
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # ファイル出力用のハンドラ
    log_file = out_dir / 'color_feature_extraction.log'
    fh = logging.FileHandler(str(log_file), encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 標準出力へのハンドラ
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main():
    # 入力ディレクトリのパスを取得
    data_dir = Path(DATA_DIR)
    
    # output folderを作成
    out_put_dir = Path(OUTPUT_DIR)
    out_put_dir.mkdir(parents=True, exist_ok=True)

    # ロギングの設定
    logger = setup_logger(out_put_dir)  # out_put_dirを引数として渡す
    logger.info('色特徴量抽出の処理を開始します')
    
    # フォルダ内のすべてのフォルダ名を取得
    folder_names = [name.name for name in data_dir.glob('*') if name.is_dir()]
    
    for folder_name in folder_names:
        logger.info(f"処理中のフォルダ: {folder_name}")
        
        # フォルダ内のファイルパスを取得
        folder_path = Path(DATA_DIR) / folder_name
        label_csv_path = Path(DATA_DIR) / f"{folder_name}.csv"
        
        # labelを読み込む（ヘッダーなしで読み込み）
        label_df = pd.read_csv(label_csv_path, header=None)

        # 入力画像設定
        image_paths = tuple(folder_path.glob('*.png'))
        # ファイル名の数値部分で昇順にソート
        image_paths = sorted(image_paths, key=lambda x: int(x.stem.split('_')[-1]))
        
        # ラベルの集計
        # main_counts, sub_counts = count_labels(label_df)
        # logger.info(f"フォルダ {folder_name} の主クラス集計結果: {main_counts}")
        # logger.info(f"フォルダ {folder_name} のサブクラス集計結果: {sub_counts}")
        # logger.info(f"フォルダ {folder_name} の画像枚数: {len(image_paths)}")
        # logger.info(f"フォルダ {folder_name} のラベル数: {len(label_df)}")
        # save_label_counts(main_counts, sub_counts, out_put_dir, folder_name)

        # 画像の色特徴量を算出し，CSVファイルに出力
        write_csv(image_paths, label_df, folder_name, out_put_dir, logger)

    logger.info('すべての処理が完了しました')


if __name__ == '__main__':
    main()