# 特徴量16R,G,B,H
# Hの算出

import os
import pathlib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import cmath
import scipy.stats as sstats
from sklearn import preprocessing as sp
import math

WIDTH = 800

def normalize_img(img):

    height = img.shape[0]
    width = img.shape[1]

    center_h = int(height / 2)
    center_w = int(width / 2)

    for i in range(width):
        blue = img.item(center_h, i, 0)
        green = img.item(center_h, i, 1)
        red = img.item(center_h, i, 2)
        left = 0
        right = width
        if(i==0 and blue > 5 and green >5 and red > 5):
            left = i

            if (blue <= 5 and green <= 5 and red <= 5):
                right = i
                break

        elif (blue <= 5 and green <=5 and red <= 5):
            if (i < center_w):
                left = i

            else:
                right = i
                break

    crop = img[0: height, left: right]
    crop_h = crop.shape[0]
    crop_w = crop.shape[1]
    ratio = WIDTH / crop_w

    resize_img = cv2.resize(crop, (int(crop_w * ratio), int(crop_h * ratio)))

    return resize_img


def analysis_hsv(path):

    img = cv2.imread(str(path))
    normalized = normalize_img(img)
    crop = normalized[50: 600,  100: 700]

    #RGB
    r,g,b = crop[:,:,2],crop[:,:,1],crop[:,:,0]
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

    #hsv = rgb_to_hsv(crop)
    #img_hsv = np.uint8(hsv)

    img_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

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


def write_csv(images, label):
    for path in images:
        mean_r, std_r, skew_r, kurto_r, mean_g, std_g, skew_g, kurto_g, mean_b, std_b, skew_b, kurto_b, mean_h, std_h, skew_h, kurto_h  = analysis_hsv(path)


        with open(os.path.join(OUTPUT_DIR, CSV_NAME + '.csv'), 'a',newline="") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(path), mean_r,std_r,skew_r[0],kurto_r[0],mean_g,std_g,skew_g[0],kurto_g[0]
                                ,mean_b,std_b,skew_b[0],kurto_b[0],mean_h, std_h, skew_h[0],kurto_h[0], label])



def rgb_to_hsv(src, ksize=3):
    # 高さ・幅・チャンネル数を取得
    h, w, c = src.shape

    # 入力画像と同じサイズで出力画像用の配列を生成(中身は空)
    dst = np.empty((h, w, c))

    for y in range(0, h):
        for x in range(0, w):
            # R, G, Bの値を取得して0～1の範囲内にする
            [b, g, r] = src[y][x] / 255.0
            # R, G, Bの値から最大値と最小値を計算
            mx, mn = max(r, g, b), min(r, g, b)
            # 最大値 - 最小値
            diff = mx - mn

            # Hの値を計算
            if mx == mn:
                h = 0
            elif mx == r:
                h = 60 * ((g - b) / diff)
            elif mx == g:
                h = 60 * ((b - r) / diff) + 120
            elif mx == b:
                h = 60 * ((r - g) / diff) + 240
            if h < 0: h = h + 360

            # Sの値を計算
            if mx != 0:
                s = diff / mx
            else:
                s = 0

            # Vの値を計算
            v = mx

            dst[y][x] = [h, s, v]

    return dst

def rgb_to_lab(src):
    img_lab = cv2.cvtColor(src, cv2.COLOR_RGB2Lab)

    img_lab = img_lab.astype(np.float)
    img_lab[:, :, 0] *= 100 / 255
    img_lab[:, :, 1] -= 128
    img_lab[:, :, 2] -= 128
    img_lab = img_lab.astype(np.int)

    return img_lab

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

OUTPUT_DIR = "D:/previous_research/data_csv/2/fujifilm/"
CSV_NAME = "test7_d"


if __name__ == "__main__":
    
    # パス指定
    # input_dir = 'D:/study/data/2/fujifilm/images/test7_d'
    # 入力画像設定
    # img = tuple(pathlib.Path(input_dir).glob('*.png'))
    # write_csv(img)
    # analysis_hsv(img)
    
    # data用（合同中間）
    #
    # D:\\study\\data に含まれるすべての画像の色特徴量を算出
    #
    for system in ['fujifilm', 'olympus']:
        
        for folder_number in range(1, 7):
            
            # フォルダが存在するディレクトリのパスを指定
            folder_path = os.path.join("D:\\study\\data", str(folder_number), system, "images")
        
            # フォルダ内のすべてのフォルダ名を取得
            folder_names = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
            
            #print("-" * 10 + str(folder_number) + "_" + system + "-" * 10)
            # フォルダ名を指定して解析
            for name in folder_names:
                # パス指定
                path = os.path.join(folder_path, name)
                OUTPUT_DIR = os.path.join("D:\\previous_research\\data_csv", str(folder_number), system)
                CSV_NAME = name
                # labelを読み込む
                df = pd.read_csv(os.path.join("D:\\study\\data", str(folder_number), system, "labels", name + '.csv'), header=None)
                # 一行目の数字を取得
                label = df[0][0]
                print(name, label)
                # 入力画像設定
                img = tuple(pathlib.Path(path).glob('*.png'))
                write_csv(img, label)
            
                
            
        
        