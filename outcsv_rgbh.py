# 色特徴量抽出
# 
# 画像フォルダ内の画像の色特徴量を算出し，csvファイルに出力する
# 
# 入力ディレクトリ構造
# ├── input_image_folder
# │   ├── 1
# │   │   ├── fujifilm
# │   │   │   ├── images
# │   │   │   │   ├── data1_a
# │   │   │   │   │   ├── 1.png
# │   │   │   │   │   ├── 2.png
#                     ...
# │   │   │   │   ├── data1_b
# │   │   │   │   │   ├── 1.png
# │   │   │   │   │   ├── 2.png
#                     ...
# │   │   │   ├── labels
# │   │   │   │   ├── data1_a.csv
# │   │   │   │   ├── data1_b.csv
#                 ...
# │   │   ├── olympus
# │   │   │   ├── images
# │   │   │   ├── labels
# │   ├── 2
#     ...
# 
# 出力ディレクトリ構造
# ├── output_image_folder
# │   ├── 1
# │   │   ├── fujifilm
# │   │   │   ├── data_a.csv
# │   │   │   ├── data_b.csv
#                 ...
# │   │   ├── olympus
#         ...
# │   ├── 2
#     ...

import os
import pandas as pd
import pathlib
import csv
import cv2
import numpy as np
import scipy.stats as sstats
import cmath


# パス
INPUT_DIR = "D:\\study\\data"
OUTPUT_DIR = "D:\\previous_research\\new_data_csv"
OLYMPUS_MASK_IMG = 'olympus_mask.png'
FUJIFILM_MASK_IMG = 'fujifilm_mask.png'

# cropping size (deep learning と統一)
OLYMPUS_CROPPPING = [20, 1060, 710, 1890]
FUJIFILM_CROPPPING = [25, 995, 330, 1590]

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


def analysis_hsv(path, system):

    img = cv2.imread(str(path))
    
    # 各撮影systemのマスク画像
    if system == 'olympus':
        mask = cv2.imread(OLYMPUS_MASK_IMG)
    elif system == 'fujifilm':
        mask = cv2.imread(FUJIFILM_MASK_IMG)
    
    img = cv2.bitwise_and(img, mask)
    
    if system == 'olympus':
        crop_img = img[OLYMPUS_CROPPPING[0] : OLYMPUS_CROPPPING[1], OLYMPUS_CROPPPING[2] : OLYMPUS_CROPPPING[3]]
    elif system == 'fujifilm':
        crop_img = img[FUJIFILM_CROPPPING[0] : FUJIFILM_CROPPPING[1], FUJIFILM_CROPPPING[2] : FUJIFILM_CROPPPING[3]]
        
    resize_img = cv2.resize(crop_img, dsize=(IMG_SIZE, IMG_SIZE))

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
    skew_h =sstats.skew(hist_h)
    # 尖度
    kurto_h = sstats.kurtosis(hist_h)

    return mean_r,std_r,skew_r,kurto_r,mean_g,std_g,skew_g,kurto_g,mean_b,std_b,skew_b,kurto_b,mean_h, std_h,skew_h,kurto_h


def write_csv(images, system, label, name, csv_path):
    for path in images:
        mean_r, std_r, skew_r, kurto_r, mean_g, std_g, skew_g, kurto_g, mean_b, std_b, skew_b, kurto_b, mean_h, std_h, skew_h, kurto_h  = analysis_hsv(path, system)

        with open(os.path.join(csv_path, name + '.csv'), 'a',newline="") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(path), mean_r,std_r,skew_r[0],kurto_r[0],mean_g,std_g,skew_g[0],kurto_g[0]
                                ,mean_b,std_b,skew_b[0],kurto_b[0],mean_h, std_h, skew_h[0],kurto_h[0], label])


def main():
    
    # output folderを作成
    if not os.path.exists(os.path.join(OUTPUT_DIR)):
        os.mkdir(os.path.join(OUTPUT_DIR))
    
    for folder_number in range(1, 7):
        
        # output folderを作成
        if not os.path.exists(os.path.join(OUTPUT_DIR, str(folder_number))):
            os.mkdir(os.path.join(OUTPUT_DIR, str(folder_number)))
        
        for system in ['fujifilm', 'olympus']:
            
            # output folderを作成
            if not os.path.exists(os.path.join(OUTPUT_DIR, str(folder_number), system)):
                os.mkdir(os.path.join(OUTPUT_DIR, str(folder_number), system))
            
            # 入力フォルダが存在するディレクトリのパスを指定
            folder_path = os.path.join(INPUT_DIR, str(folder_number), system, "images")
        
            # フォルダ内のすべてのフォルダ名を取得
            folder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
            
            #print("-" * 10 + str(folder_number) + "_" + system + "-" * 10)
            # フォルダ名を指定して解析
            for name in folder_names:
                
                # パス指定
                path = os.path.join(folder_path, name)
                csv_path = os.path.join(OUTPUT_DIR, str(folder_number), system)
                
                # labelを読み込む
                df = pd.read_csv(os.path.join(INPUT_DIR, str(folder_number), system, "labels", name + '.csv'), header=None)
                
                # 一行目の数字(label)を取得
                label = df[0][0]
                print(name, label)
                
                # 入力画像設定
                images = tuple(pathlib.Path(path).glob('*.png'))
                write_csv(images, system, label, name, csv_path)
    
    
if __name__ == '__main__':
    main()