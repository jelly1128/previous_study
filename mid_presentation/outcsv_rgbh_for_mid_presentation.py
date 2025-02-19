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


# パス
INPUT_DIR = "/home/tanaka/demo_data"
# INPUT_DIR = "/home/tanaka/debug_data"
OUTPUT_DIR = "demo_data_rgbh_csv"
# OUTPUT_DIR = "debug_data_rgbh_csv"

# resize (deep learning と統一)
IMG_SIZE = 224


def mean_angle(angles):
    a = np.deg2rad(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    mean = cmath.phase(angles_complex.sum()) % (2 * np.pi)
    return round(np.rad2deg(mean) , 1)

def std_angle(angles):
    a = np.deg2rad(angles)
    angles_complex = np.frompyfunc(cmath.exp, 1, 1)(a * 1j)
    r = abs(angles_complex.sum()) / angles.size
    std = np.sqrt(-2 * np.log(r))
    return round(np.rad2deg(std), 2)


def analysis_hsv(path):
    img = cv2.imread(str(path))
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

    return mean_r,std_r,skew_r,kurto_r,mean_g,std_g,skew_g,kurto_g,mean_b,std_b,skew_b,kurto_b,mean_h,std_h,skew_h,kurto_h


def write_csv(images: tuple[Path], label_df: pd.DataFrame, name: str, csv_path: Path):
    # CSVファイルのヘッダーを書き込む
    with open(csv_path / (name + '_rgbh.csv'), 'w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'mean_r', 'std_r', 'skew_r', 'kurto_r', 
                        'mean_g', 'std_g', 'skew_g', 'kurto_g',
                        'mean_b', 'std_b', 'skew_b', 'kurto_b',
                        'mean_h', 'std_h', 'skew_h', 'kurto_h',
                        'main_label', 'sub_label'])

    # 画像名をキーとしてラベルを検索できるように辞書を作成
    label_dict = {}
    for _, row in label_df.iterrows():
        # img_name = row[0]  # 1列目の画像名
        # if img_name.endswith('.png'):
        #     img_name = img_name[:-4]  # .pngを除去
        img_name = row[0] # 1列目の画像名.pngあり
        labels = row[1:]   # 2列目以降のラベル
        label_dict[img_name] = labels

    for path in images:
        try:
            mean_r,std_r,skew_r,kurto_r,mean_g,std_g,skew_g,kurto_g,mean_b,std_b,skew_b,kurto_b,mean_h,std_h,skew_h,kurto_h = analysis_hsv(path)
            
            # 画像名を取得（.pngを除く）
            # img_name = path.stem
            img_name = path.stem + path.suffix
            
            # 対応するラベルを取得
            if img_name not in label_dict:
                print(f"Warning: No label found for {img_name}")
                continue
                
            labels = label_dict[img_name]
            
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

            with open(csv_path / (name + '_rgbh.csv'), 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([img_name, mean_r,std_r,skew_r[0],kurto_r[0],mean_g,std_g,skew_g[0],kurto_g[0],
                               mean_b,std_b,skew_b[0],kurto_b[0],mean_h,std_h,skew_h[0],kurto_h[0],
                               main_label, sub_label])
                
        except Exception as e:
            print(f"Error processing {path.stem}: {str(e)}")
            continue


def main():
    # 入力ディレクトリのパスを取得
    input_dir = Path(INPUT_DIR)
    
    # output folderを作成
    out_put_dir = Path(OUTPUT_DIR)
    out_put_dir.mkdir(parents=True, exist_ok=True)
    
    # フォルダ内のすべてのフォルダ名を取得
    folder_names = [name.name for name in input_dir.glob('*') if name.is_dir()]
    
    for folder_name in folder_names:
        print(f"folder_name: {folder_name}")
        
        folder_path = Path(INPUT_DIR) / folder_name
        label_csv_path = Path(INPUT_DIR) / f"{folder_name}.csv"
        
        # labelを読み込む（ヘッダーなしで読み込み）
        label_df = pd.read_csv(label_csv_path, header=None)
        
        # 入力画像設定
        images = tuple(folder_path.glob('*.png'))
        # ファイル名の数値部分で昇順にソート
        images = sorted(images, key=lambda x: int(x.stem.split('_')[-1]))
        
        # 出力ディレクトリの作成
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        
        write_csv(images, label_df, folder_name, output_path)


if __name__ == '__main__':
    main()