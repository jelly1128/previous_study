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

MODEL_NAME = "svm_model"

CSV_DIR = "D:\\previous_research\\data_csv"

# k_folds  ([train], test)
K_FOLDS_SPLIT = [
#    ([1, 2, 3, 4], 6),
#    ([1, 2, 3, 6], 5),
#    ([1, 2, 5, 6], 4),
#    ([1, 4, 5, 6], 3),
    ([1, 3, 4, 5, 6], 2),
#    ([2, 3, 4, 5], 1),
]

# メイン関数
if __name__ == '__main__':
    
    for split, (train_indices, test_index) in enumerate(K_FOLDS_SPLIT):
        
        train_data = None
        test_data = None
        
        # 学習データ読み込み
        for train_index in train_indices:
            
            for system in ["fujifilm", "olympus"]:
                # フォルダが存在するディレクトリのパスを指定
                folder_path = os.path.join(CSV_DIR, str(train_index), system)
                # フォルダ内のすべてのCSVファイルのパスを取得
                csv_paths = glob.glob(os.path.join(folder_path, '*.csv'))
                #print(csv_paths)
                
                for csv_path in csv_paths:
                    file_data = np.loadtxt(csv_path, delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
                    if train_data is None:
                        train_data = file_data
                    else:
                        train_data = np.concatenate((train_data, file_data), axis=0)
        
        # テストデータ読み込み    
        for system in ["fujifilm", "olympus"]:
            
            # フォルダが存在するディレクトリのパスを指定
            folder_path = os.path.join(CSV_DIR, str(test_index), system)
            # フォルダ内のすべてのCSVファイルのパスを取得
            csv_paths = glob.glob(os.path.join(folder_path, '*.csv'))
            #print(csv_paths)
            
            for csv_path in csv_paths:
                file_data = np.loadtxt(csv_path, delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17))
                if test_data is None:
                    test_data = file_data
                else:
                    test_data = np.concatenate((test_data, file_data), axis=0)
        
        # 目的変数(Y)、説明変数(X)
        train_y = train_data[:, 16].astype(int)
        train_x = train_data[:, 0:16]
        
        test_y = test_data[:, 16].astype(int)
        test_x = test_data[:, 0:16]

        # データの標準化処理
        sc = StandardScaler()
        sc.fit(train_x)
        sc.fit(test_x)
        # 特徴量スケーリングモデル保存
        with open(MODEL_NAME + str(split + 1) + '.sav', mode='wb') as fo_scale:
            pickle.dump(sc, fo_scale)

        x_train_std = sc.transform(train_x)
        x_test_std = sc.transform(test_x)
        
        model = SVC(kernel='linear',gamma='scale')
        #model = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
        #model.fit(x_selected, y)
        model.fit(x_train_std, train_y)
        
        # モデルの保存
        with open(MODEL_NAME + str(split + 1) + '.pickle', mode='wb') as fo_model:  # with構文でファイルパスとバイナリ書き込みモードを設定
            pickle.dump(model, fo_model)  # オブジェクトをシリアライズ
            
        # 学習済みモデルによる予測
        # モデルの精度評価
        # テストデータに対する精度
        #pred_train = model.predict(x_selected)
        pred_test = model.predict(x_test_std)
        
        accuracy_test = accuracy_score(test_y, pred_test)
        print('テストデータに対する正解率： %.2f' % accuracy_test)
        
        cm = confusion_matrix(test_y, pred_test)
        print(cm)
        
        # 混同行列をCSVファイルに保存
        csv_file = MODEL_NAME + str(split + 1) + 'confusion_matrix.csv'
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(cm)
        