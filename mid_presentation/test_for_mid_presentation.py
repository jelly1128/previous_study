from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
from results_visualizer import visualize_label_timeline


# 混同行列から評価指標（適合率・再現率・F1スコアを算出）
def calculate_metrics(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    混同行列から適合率・再現率・F1スコアを算出する関数

    Args:
        cm (np.ndarray): 混同行列
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 適合率・再現率・F1スコア
    """
    # 適合率
    precisions = cm.diagonal() / cm.sum(axis=0).clip(min=1e-10)  # 0除算を防ぐ
    # 再現率
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1e-10)  # 0除算を防ぐ
    # F1スコア
    f1_scores = 2 * precisions * recalls / (precisions + recalls).clip(min=1e-10)  # 0除算を防ぐ

    return precisions, recalls, f1_scores


# 予測結果をcsvファイルに保存
def save_results_to_csv(results: pd.DataFrame, output_dir: Path, video_name: str) -> None:
    """
    予測結果をcsvファイルに保存する関数

    Args:
        results (pd.DataFrame): 予測結果
        output_dir (Path): 保存先ディレクトリ
        video_name (str): 動画名
    """
    # 保存先のディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 予測結果を保存
    results_csv_path = output_dir / f"{video_name}_results.csv"
    results.to_csv(results_csv_path, index=False)
    print(f"Saved results to: {results_csv_path}")


# 混同行列をcsvファイルに保存
def save_confusion_matrix_to_csv(cm: np.ndarray, output_dir: Path, video_name: str) -> None:
    """
    混同行列をcsvファイルに保存する関数

    Args:
        cm (np.ndarray): 混同行列
        output_dir (Path): 保存先ディレクトリ
        video_name (str): 動画名
    """
    # 保存先のディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 混同行列を保存
    cm_csv_path = output_dir / f"{video_name}_confusion_matrix.csv"
    cm_df = pd.DataFrame(cm)
    # インデックスとカラム名を追加（0-5のラベル）
    cm_df.rename_axis('True_Label', axis=0, inplace=True)
    cm_df.rename_axis('Predicted_Label', axis=1, inplace=True)
    cm_df.index = range(NUM_CLASSES)  # 行のラベル
    cm_df.columns = range(NUM_CLASSES)  # 列のラベル
    cm_df.to_csv(cm_csv_path)
    print(f"Saved confusion matrix to: {cm_csv_path}")


# 評価結果をcsvファイルに保存
def save_metrics_to_csv(precision: np.ndarray, recall: np.ndarray, f1_score: np.ndarray, output_dir: Path, video_name: str) -> None:
    """
    評価結果をcsvファイルに保存する関数

    Args:
        metrics (pd.DataFrame): 評価結果
        output_dir (Path): 保存先ディレクトリ
        video_name (str): 動画名
    """
    # 保存先のディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 評価結果を保存
    metrics_csv_path = output_dir / f"{video_name}_classification_report.csv"
    metrics_df = pd.DataFrame({
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1_score
    })
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics to: {metrics_csv_path}")


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
    over_all_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

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
            features_labels = video_data.iloc[:, 1:NUM_FEATURES+NUM_LABELS].values

            test_x = features_labels[:, 0:NUM_FEATURES]  # 特徴量（16次元）
            test_y = features_labels[:, NUM_FEATURES].astype(int)  # main_label

            # テストデータのラベル分布を確認
            # unique_labels, counts = np.unique(test_y, return_counts=True)
            # print("\nLabel distribution in test data:")
            # for label, count in zip(unique_labels, counts):
            #     print(f"Label {label}: {count} samples")
            # print(f"Total samples: {len(test_y)}")


        except Exception as e:
            print(f"Error loading {csv_path}: {str(e)}")

        # テストデータの標準化
        x_test_std = scaler.transform(test_x)

        # 予測
        y_pred = model.predict(x_test_std)

        # 混同行列を計算し，CSVファイルに保存
        cm = confusion_matrix(test_y, y_pred, labels=range(NUM_CLASSES))
        save_confusion_matrix_to_csv(cm, video_name_dir, video_name)

        # 各クラスの適合率・再現率・F1スコアを算出し，CSVファイルに保存
        precisions, recalls, f1_scores = calculate_metrics(cm)
        save_metrics_to_csv(precisions, recalls, f1_scores, video_name_dir, video_name)

        # 全体の混同行列に加算
        over_all_cm += cm

        # 予測結果をCSVファイルに保存
        results_df = pd.DataFrame({
            'True_Label': test_y,
            'Predicted_Label': y_pred
        })
        save_results_to_csv(results_df, video_name_dir, video_name)

        # 予測結果の可視化
        visualize_label_timeline(results_df['Predicted_Label'], video_name_dir, video_name, "predicted")
        # 正解ラベルの可視化
        visualize_label_timeline(results_df['True_Label'], video_name_dir, video_name, "ground_truth")

        # import sys
        # sys.exit()
    
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

        # 混同行列をcsvファイルに保存
        save_confusion_matrix_to_csv(fold_cm, Path(output_dir), f"fold{fold}")

        # 各クラスの適合率・再現率・F1スコアを算出し，CSVファイルに保存
        fold_precisions, fold_recalls, fold_f1_scores = calculate_metrics(fold_cm)
        save_metrics_to_csv(fold_precisions, fold_recalls, fold_f1_scores, Path(output_dir), f"fold{fold}")
        
        # 全体の混同行列に加算
        all_fold_cm += fold_cm
    
    # 全foldの混同行列をcsvファイルに保存
    save_confusion_matrix_to_csv(all_fold_cm, Path(output_dir), "all_fold")

    # 全クラスの適合率・再現率・F1スコアを算出し，CSVファイルに保存
    all_fold_precisions, all_fold_recalls, all_fold_f1_scores = calculate_metrics(all_fold_cm)
    save_metrics_to_csv(all_fold_precisions, all_fold_recalls, all_fold_f1_scores, Path(output_dir), "all_fold")
        

NUM_CLASSES = 6
NUM_FEATURES = 16
NUM_LABELS = 2
OUTPUT_DIR = "demo_cross_test"
CSV_DIR = "/home/tanaka/mid_presentation/previous_study/demo_data_rgbh_csv"
MODEL_DIR = "/home/tanaka/mid_presentation/previous_study/demo_train"
MODEL_NAMES = ["demo_train_split2_3_svm_model",  "demo_train_split3_4_svm_model", "demo_train_split4_1_svm_model", "demo_train_split1_2_svm_model"]
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

def main():
    # output folderを作成
    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # テスト
    # test(output_dir, csv_dir, model_dir, model_names[0], SPLITS_DICT['split1'])
    # 交差検証
    cross_test(OUTPUT_DIR, CSV_DIR, MODEL_DIR, MODEL_NAMES)


if __name__ == '__main__':
    main()