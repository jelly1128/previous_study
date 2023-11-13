import os
import glob
import numpy as np
import csv
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

MODEL_NAME = "new_svm_model"
OUTPUT_DIR = "D:\\previous_research\\new_data_train"
CSV_DIR = "D:\\previous_research\\new_data_csv"

# train data
TRAIN_DATA = [1, 2, 3, 4]
OUTPUT_FOLDER_NAME = "split1"


def dataloader(data_folds):
    
    data = None
    
    for fold in data_folds:
        
        for system in ['fujifilm', 'olympus']:
            
            # train dataのフォルダのパス指定
            fold_path = os.path.join(CSV_DIR, str(fold), system)
            # フォルダ内のすべてのCSVファイルのパスを取得
            csv_paths = glob.glob(os.path.join(fold_path, '*.csv'))
            
            for csv_path in csv_paths:
                
                # csv内の色特徴量16種を読み込み
                file_data = np.loadtxt(csv_path, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
                
                if data is None:
                    data = file_data
                else:
                    data = np.concatenate((data, file_data), axis=0)
    
    return data
            


# メイン関数
def main():
    
    # output folderを作成
    if not os.path.exists(os.path.join(OUTPUT_DIR)):
        os.mkdir(os.path.join(OUTPUT_DIR))
        
    # output folderを作成
    if not os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER_NAME)):
        os.mkdir(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER_NAME))
    
    # データ読み込み
    train_data = dataloader(TRAIN_DATA)
        
    # 目的変数(Y)，説明変数(X)，画像名
    train_y = train_data[:, 16].astype(int)
    train_x = train_data[:, 0:16]
    
    # データの標準化処理
    scaler = StandardScaler()
    scaler.fit(train_x)
    x_train_std = scaler.transform(train_x)
        
    # 特徴量スケーリングモデル保存
    with open(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER_NAME, MODEL_NAME + OUTPUT_FOLDER_NAME + '.sav'), mode='wb') as feature_scale:
        pickle.dump(scaler, feature_scale)
        
        
    model = SVC(kernel='linear',gamma='scale')
    #model = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
    model.fit(x_train_std, train_y)
        
    # モデルの保存
    with open(os.path.join(OUTPUT_DIR, OUTPUT_FOLDER_NAME, MODEL_NAME + OUTPUT_FOLDER_NAME + '.pickle'), mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
        pickle.dump(model, f)  # オブジェクトをシリアライズ
        
            
if __name__ == '__main__':
    main()
        