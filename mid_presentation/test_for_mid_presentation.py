from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

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


def test(output_dir: str, csv_dir: str, model_dir: str, model_name: str, test_split: list[str]) -> np.ndarray:
    # 学習済みモデルの読み込み
    try:
        # スケーラーの読み込み
        scaler_path = Path(model_dir) / f"{model_name}_scaler.sav"
        with open(scaler_path, mode='rb') as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from: {scaler_path}")
        
        # モデルの読み込み
        model_path = Path(model_dir) / f"{model_name}.pickle"
        with open(model_path, mode='rb') as f:
            model = pickle.load(f)
        print(f"Loaded model from: {model_path}")
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # 全体の結果
    over_all_cm = np.zeros((6, 6), dtype=int)

    for video_name in test_split:
        # 保存先のディレクトリを作成
        video_name_dir = Path(output_dir, video_name)
        video_name_dir.mkdir(parents=True, exist_ok=True)

        csv_path = Path(csv_dir) / f"{video_name}_rgbh.csv"
        print(f"csv_path: {csv_path}")
        print(f"video_name: {video_name}")
        # テストデータの読み込み
        try: 
            video_data = pd.read_csv(csv_path)

            image_names = video_data.iloc[:, 0].values
            features_labels = video_data.iloc[:, 1:18].values

            test_y = features_labels[:, 16].astype(int)  # main_label
            test_x = features_labels[:, 0:16]  # 特徴量（16次元）

            # テストデータのラベル分布を確認
            unique_labels, counts = np.unique(test_y, return_counts=True)
            print("\nLabel distribution in test data:")
            for label, count in zip(unique_labels, counts):
                print(f"Label {label}: {count} samples")
            print(f"Total samples: {len(test_y)}")


        except Exception as e:
            print(f"Error loading {csv_path}: {str(e)}")

        # テストデータの標準化
        x_test_std = scaler.transform(test_x)

        # 予測
        y_pred = model.predict(x_test_std)

        # 性能評価
        accuracy = accuracy_score(test_y, y_pred)
        cm = confusion_matrix(test_y, y_pred, labels=range(6))
        # 各クラスの適合率と再現率を算出
        precisions = cm.diagonal() / cm.sum(axis=0).clip(min=1e-10)  # 0除算を防ぐ
        recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1e-10)  # 0除算を防ぐ

        # print(f"\nAccuracy: {accuracy:.4f}")
        # print("\nConfusion Matrix:")
        # print(cm)
        # print("\nClassification Report:")
        # print(precisions)
        # print(recalls)

        # 全体の混同行列に加算
        over_all_cm += cm

        # 評価結果を保存
        cm_csv_path = video_name_dir / f"{video_name}_confusion_matrix.csv"
        cm_df = pd.DataFrame(cm)
        # インデックスとカラム名を追加（0-5のラベル）
        cm_df.rename_axis('True_Label', axis=0, inplace=True)
        cm_df.rename_axis('Predicted_Label', axis=1, inplace=True)
        cm_df.index = range(6)  # 行のラベル
        cm_df.columns = range(6)  # 列のラベル
        cm_df.to_csv(cm_csv_path)

        report_df = pd.DataFrame({
            'Precision': precisions,
            'Recall': recalls
        })

        report_csv_path = video_name_dir / f"{video_name}_classification_report.csv"
        report_df.to_csv(report_csv_path, index=False)

        # 予測結果をCSVファイルに保存
        results_df = pd.DataFrame({
            'Image_Name': image_names,
            'True_Label': test_y,
            'Predicted_Label': y_pred
        })

        results_csv_path = video_name_dir / f"{video_name}_results.csv"
        results_df.to_csv(results_csv_path, index=False)

        # import sys
        # sys.exit()

    # 全体の性能評価
    over_all_precisions = over_all_cm.diagonal() / over_all_cm.sum(axis=0).clip(min=1e-10)  # 0除算を防ぐ
    over_all_recalls = over_all_cm.diagonal() / over_all_cm.sum(axis=1).clip(min=1e-10)  # 0除算を防ぐ

    over_all_cm_csv_path = Path(output_dir) / "over_all_confusion_matrix.csv"
    over_all_cm_df = pd.DataFrame(over_all_cm) # 混同行列
    over_all_cm_df.rename_axis('True_Label', axis=0, inplace=True)
    over_all_cm_df.rename_axis('Predicted_Label', axis=1, inplace=True)
    over_all_cm_df.index = range(6)  # 行のラベル
    over_all_cm_df.columns = range(6)  # 列のラベル
    over_all_cm_df.to_csv(over_all_cm_csv_path)

    over_all_report_df = pd.DataFrame({
        'Precision': over_all_precisions,
        'Recall': over_all_recalls
    })

    over_all_report_csv_path = Path(output_dir) / "over_all_classification_report.csv"
    over_all_report_df.to_csv(over_all_report_csv_path, index=False)
    
    return over_all_cm

def cross_test(output_dir: str, csv_dir: str, model_dir: str, model_names: list[str]):
    # 交差検証の結果を保存するためのリスト
    all_fold_cm = np.zeros((6, 6), dtype=int)
    
    # 交差検証
    folds = range(1, 5)
    for fold in folds:
        print(f"\nFold {fold}:")
        
        # テストデータの読み込み
        test_split = SPLITS_DICT[f"split{fold}"]
        model_name = model_names[fold - 1]
        fold_cm = test(output_dir, csv_dir, model_dir, model_name, test_split)

        # import sys
        # sys.exit()
        
        # 交差検証の結果を保存
        all_fold_cm += fold_cm
    
    # 適合率と再現率を計算
    all_fold_precisions = all_fold_cm.diagonal() / all_fold_cm.sum(axis=0).clip(min=1e-10)  # 0除算を防ぐ
    all_fold_recalls = all_fold_cm.diagonal() / all_fold_cm.sum(axis=1).clip(min=1e-10)  # 0除算を防ぐ
    
    # 交差検証の結果を保存
    all_fold_cm_csv_path = Path(output_dir) / "all_fold_confusion_matrix.csv"
    all_fold_cm_df = pd.DataFrame(all_fold_cm)
    all_fold_cm_df.to_csv(all_fold_cm_csv_path)
    all_fold_cm_df.rename_axis('True_Label', axis=0, inplace=True)
    all_fold_cm_df.rename_axis('Predicted_Label', axis=1, inplace=True)
    all_fold_cm_df.index = range(6)  # 行のラベル
    all_fold_cm_df.columns = range(6)  # 列のラベル
    
    # 交差検証の結果を保存
    all_fold_report_df = pd.DataFrame({
        'Precision': all_fold_precisions,
        'Recall': all_fold_recalls
    })
    all_fold_report_csv_path = Path(output_dir) / "all_fold_classification_report.csv"
    all_fold_report_df.to_csv(all_fold_report_csv_path, index=False)
    
    return all_fold_cm
        

def main():
    output_dir = "demo_cross_test"  # 結果を保存するディレクトリ
    csv_dir = "/home/tanaka/mid_presentation/previous/study/demo_data_rgbh_csv"
    model_dir = "/home/tanaka/mid_presentation/previous_study/demo_train"
    # 学習済みモデル
    model_names = ["demo_train_split4_1_svm_model", "demo_train_split3_4_svm_model", "demo_train_split2_3_svm_model", "demo_train_split1_2_svm_model"]

    # output folderを作成
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # テスト
    # test(output_dir, csv_dir, model_dir, model_names[0], SPLITS_DICT['split1'])
    # 交差検証
    cross_test(output_dir, csv_dir, model_dir, model_names)


if __name__ == '__main__':
    main()