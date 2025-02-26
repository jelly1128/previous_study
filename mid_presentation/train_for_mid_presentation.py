# CSVディレクトリ構造
# ├── output_image_folder
# │   ├── video_name_A_rgbh.csv
# │   ├── video_name_B_rgbh.csv
#     ...

from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SPLITS_DICT = {
    'split1': ["20210119093456_000001-001",
                "20210531112330_000005-001",
                "20211223090943_000001-002",
                "20230718-102254-ES06_20230718-102749-es06-hd",
                "20230802-104559-ES09_20230802-105630-es09-hd",
                ],
    'split2': ["20210119093456_000001-002",
                "20210629091641_000001-002",
                "20211223090943_000001-003",
                "20230801-125025-ES06_20230801-125615-es06-hd",
                "20230803-110626-ES06_20230803-111315-es06-hd"
                ],
    'split3': ["20210119093456_000002-001",
                "20210630102301_000001-002",
                "20220322102354_000001-002",
                "20230802-095553-ES09_20230802-101030-es09-hd",
                "20230803-093923-ES09_20230803-094927-es09-hd",
                ],
    'split4': ["20210524100043_000001-001",
                "20210531112330_000001-001",
                "20211021093634_000001-001",
                "20211021093634_000001-003"
                ]
}


# ラベルの分布を確認
def label_distribution(data: np.ndarray):
    """
    ラベルの分布を確認する関数
    
    Args:
        data (np.ndarray): 特徴量とラベルを含む配列
    """
    # main_labelの列番号
    label_column_index = NUM_FEATURES  # main_label が17列目にあると仮定(0始まり)

    # ラベルの分布をdataloader内で確認
    labels = data[:, label_column_index].astype(int)  # main_label
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} samples")
    print(f"Total samples: {len(labels)}")


# splitのデータを読み込む
def dataloader(split_name: str) -> np.ndarray:
    """
    指定されたsplitのデータを読み込む関数
    
    Args:
        split_name (str): 'split1', 'split2', 'split3', 'split4'のいずれか
    
    Returns:
        np.ndarray: 特徴量とラベルを含む配列
    """
    data = None
    
    # 指定されたsplitに含まれる動画名のリストを取得
    video_names = SPLITS_DICT[split_name]
    
    for video_name in video_names:
        # CSVファイルのパスを構築
        csv_path = Path(CSV_DIR) / f"{video_name}_rgbh.csv"
        
        try:
            # CSVファイルを読み込む（ヘッダーあり）
            video_data = pd.read_csv(csv_path)
            
            # 16特徴量（mean_r からkurto_h まで）とラベル（main_label, sub_label）を抽出
            features_labels = video_data.iloc[:, 1:NUM_FEATURES+NUM_LUBEL].values  # filenameを除く全列

            # ラベルの分布をdataloader内で確認
            # label_distribution(features_labels)

            if data is None:
                data = features_labels
            else:
                data = np.concatenate((data, features_labels), axis=0)
                
        except Exception as e:
            print(f"Error loading {csv_path}: {str(e)}")
            continue
    
    return data


# 複数のsplitのデータを読み込む
def load_multiple_splits(split_names):
    """
    複数のsplitのデータを読み込む関数
    
    Args:
        split_names (list): ['split1', 'split2'] などのsplit名のリスト
    
    Returns:
        np.ndarray: 特徴量とラベルを含む配列
    """
    data = None
    
    for split_name in split_names:
        split_data = dataloader(split_name)
        
        if split_data is not None:
            if data is None:
                data = split_data
            else:
                data = np.concatenate((data, split_data), axis=0)
    
    return data


OUTPUT_DIR = "demo_train"
CSV_DIR = "/home/tanaka/mid_presentation/previous_study/demo_data_rgbh_csv"
# TRAIN_SPLITS = ['split1', 'split2']
# TRAIN_SPLITS = ['split2', 'split3']
# TRAIN_SPLITS = ['split3', 'split4']
TRAIN_SPLITS = ['split4', 'split1']
MODEL_NAME = f"{OUTPUT_DIR}_split4_1_svm_model"
NUM_FEATURES = 16  # 特徴量の次元数
NUM_LUBEL = 2  # 主クラスとサブクラス

# メイン関数
def main():
    # output folderを作成
    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 訓練データの読み込み
    train_data = load_multiple_splits(TRAIN_SPLITS)
    
    if train_data is None or len(train_data) == 0:
        print("Error: No training data loaded")
        return
        
    # 目的変数(Y)，説明変数(X)
    train_y = train_data[:, NUM_FEATURES].astype(int)  # main_label
    train_x = train_data[:, 0:NUM_FEATURES]  # 特徴量（16次元）
    
    # ラベルの分布を確認
    # label_distribution(train_data)
    
    # データの標準化処理
    scaler = StandardScaler()
    scaler.fit(train_x)
    x_train_std = scaler.transform(train_x)
        
    # 特徴量スケーリングモデル保存
    with open(Path(OUTPUT_DIR) / (MODEL_NAME + '_scaler.sav'), mode='wb') as feature_scale:
        pickle.dump(scaler, feature_scale)
        
    # モデルの学習
    model = SVC(kernel='linear', gamma='scale')
    model.fit(x_train_std, train_y)
        
    # モデルの保存
    with open(Path(OUTPUT_DIR) / (MODEL_NAME + '.pickle'), mode='wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
        