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


# test data
MODEL_DIR = "D:\\previous_research\\new_data_test\\split2"
SAV_PICKLE_NAME = "new_svm_model2"
CSV_DIR = "D:\\previous_research\\new_data_csv\\5"


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
    if not os.path.exists(os.path.join(MODEL_DIR)):
        os.mkdir(os.path.join(MODEL_DIR))
    if not os.path.exists(os.path.join(MODEL_DIR, 'test')):
        os.mkdir(os.path.join(MODEL_DIR, 'test'))
    if not os.path.exists(os.path.join(MODEL_DIR, 'test', 'labels')):
        os.mkdir(os.path.join(MODEL_DIR, 'test', 'labels'))
        
    # 学習データ時の標準化処理の読み込み
    with open(os.path.join(MODEL_DIR, SAV_PICKLE_NAME + '.sav'), mode='rb') as feature_scale:
        load_scaler = pickle.load(feature_scale)
    
    # 保存したモデルをロードする
    with open(os.path.join(MODEL_DIR, SAV_PICKLE_NAME + '.pickle'), mode='rb') as f_model:
        load_model = pickle.load(f_model)
    
    # 学習済みモデルによる予測
    # モデルのテストデータに対する精度評価
    # train dataのフォルダのパス指定
    fold_path = os.path.join(CSV_DIR)
    
    total_preds = []
    total_labels = []
    
    for system in ['fujifilm', 'olympus']:
        
        # output folderを作成
        if not os.path.exists(os.path.join(MODEL_DIR, 'test', 'labels', system)):
            os.mkdir(os.path.join(MODEL_DIR, 'test', 'labels', system))
            
        # system内のすべてのCSVファイルのパスを取得
        csv_paths = glob.glob(os.path.join(fold_path, system, '*.csv'))
        
        # system内のラベル
        system_preds = []
        system_labels = []
        
        for csv_path in csv_paths:
            
            # 画像名取得
            test_image = np.loadtxt(csv_path, delimiter=',', dtype=str, usecols=(0))
            
            # csv内の色特徴量16種を読み込み,目的変数(Y)，説明変数(X)に代入
            test_data = np.loadtxt(csv_path, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
            test_y = test_data[:, 16].astype(int)
            test_x = test_data[:, 0:16]
            
            # データの標準化処理
            x_test_std = load_scaler.transform(test_x)
            
            # 予測
            pred_test = load_model.predict(x_test_std)
            
            with open(os.path.join(MODEL_DIR, 'test', 'labels', system, os.path.basename(csv_path)), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['file_name', 'pred_labels', 'true_labels'])  # ヘッダー行を書き込む
                # リストの要素を行ごとに書き込む
                for _, (item1, item2, item3) in enumerate(zip(test_image, pred_test, test_y)):
                    writer.writerow([item1, item2, item3])
            
            system_preds.extend(pred_test)
            system_labels.extend(test_y)
            
        # 混同行列をCSVファイルに保存
        cm = confusion_matrix(system_labels, system_preds)
        with open(os.path.join(MODEL_DIR, 'test', 'labels', system + '_confusion_matrix.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(cm) 
        print(cm)
        
        total_preds.extend(system_preds)
        total_labels.extend(system_labels)
    
    
    # 混同行列をCSVファイルに保存
    cm = confusion_matrix(total_labels, total_preds)
    with open(os.path.join(MODEL_DIR, 'test', 'total_confusion_matrix.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(cm) 
    print(cm)

      
if __name__ == '__main__':
    main()
        