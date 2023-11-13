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
OUTPUT_DIR = "D:\\previous_research\\new_data_test"
CSV_DIR = "D:\\previous_research\\new_data_csv"

# k_folds  ([train], test)
K_FOLDS_SPLIT = [
    ([1, 2, 3, 4], [6]),
    ([1, 2, 3, 6], [5]),
    ([1, 2, 5, 6], [4]),
    ([1, 4, 5, 6], [3]),
    ([3, 4, 5, 6], [2]),
    ([2, 3, 4, 5], [1]),
]


def dataloader(data_folds):
    
    data = None
    
    for fold in data_folds:
        
        for system in ['fujifilm', 'olympus']:
            
            # train or test のフォルダのパス指定
            fold_path = os.path.join(CSV_DIR, str(fold), system)
            # フォルダ内のすべてのCSVファイルのパスを取得
            csv_paths = glob.glob(os.path.join(fold_path, '*.csv'))
            
            for csv_path in csv_paths:
                
                # csv内の色特徴量16種を読み込み
                file_data = np.loadtxt(csv_path, delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
                
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
    
    for split, (train_folds, test_folds) in enumerate(K_FOLDS_SPLIT):
        
        # output folderを作成
        if not os.path.exists(os.path.join(OUTPUT_DIR, 'split' + str(split + 1))):
            os.mkdir(os.path.join(OUTPUT_DIR, 'split' + str(split + 1)))
        
        # データ読み込み
        train_data = dataloader(train_folds)
        
        # 目的変数(Y)、説明変数(X)
        train_y = train_data[:, 16].astype(int)
        train_x = train_data[:, 0:16]

        # データの標準化処理
        scaler = StandardScaler()
        scaler.fit(train_x)
        x_train_std = scaler.transform(train_x)
        
        # 特徴量スケーリングモデル保存
        with open(os.path.join(OUTPUT_DIR, 'split' + str(split + 1), MODEL_NAME + str(split + 1) + '.sav'), mode='wb') as feature_scale:
            pickle.dump(scaler, feature_scale)
        
        
        model = SVC(kernel='linear',gamma='scale')
        #model = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
        model.fit(x_train_std, train_y)
        
        # モデルの保存
        with open(os.path.join(OUTPUT_DIR, 'split' + str(split + 1), MODEL_NAME + str(split + 1) + '.pickle'), mode='wb') as f:  # with構文でファイルパスとバイナリ書き込みモードを設定
            pickle.dump(model, f)  # オブジェクトをシリアライズ
            
        # 学習済みモデルによる予測
        # モデルのテストデータに対する精度評価
        # データ読み込み
        test_data = dataloader(test_folds)
        
        # 目的変数(Y)、説明変数(X)
        test_y = test_data[:, 16].astype(int)
        test_x = test_data[:, 0:16]
        
        # データの標準化処理
        x_test_std = scaler.transform(test_x)
        
        pred_test = model.predict(x_test_std)
        
        accuracy_test = accuracy_score(test_y, pred_test)
        print('テストデータに対する正解率： %.2f' % accuracy_test)
        
        cm = confusion_matrix(test_y, pred_test)
        print(cm)
        
        # 混同行列をCSVファイルに保存
        with open(os.path.join(OUTPUT_DIR, 'split' + str(split + 1), MODEL_NAME + str(split + 1) + 'confusion_matrix.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(cm)
        
            
if __name__ == '__main__':
    main()
        