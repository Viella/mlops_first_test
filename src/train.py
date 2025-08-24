import pandas as pd
import numpy as np
import joblib
import argparse
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

def load_data(data_path: str) -> pd.DataFrame:
    """データを読み込む"""
    data = pd.read_csv(data_path)
    print(f"データを読み込みました: {data_path}")
    print(f"データ形状: {data.shape}")
    return data

def prepare_features_target(data: pd.DataFrame):
    """特徴量と目的変数を分離"""
    feature_columns = ['age', 'experience_years', 'skill_score', 'interview_score', 'education_level']
    X = data[feature_columns]
    y = data['hired']
    return X, y

def train_model(X_train, y_train, n_estimators: int, max_depth: int, random_state: int = 42):
    """RandomForestモデルを学習"""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    print("モデル学習中...")
    model.fit(X_train, y_train)
    print("学習完了！")
    
    return model

def evaluate_model(model, X_test, y_test):
    """モデルを評価"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== モデル評価 ===")
    print(f"テストデータ精度: {accuracy:.4f}")
    print(f"\n分類レポート:")
    print(classification_report(y_test, y_pred))
    print(f"\n混同行列:")
    print(confusion_matrix(y_test, y_pred))
    
    # 特徴量重要度
    feature_names = ['age', 'experience_years', 'skill_score', 'interview_score', 'education_level']
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, importances))
    
    print(f"\n特徴量重要度:")
    for feature, importance in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'feature_importance': feature_importance_dict
    }

def main():
    parser = argparse.ArgumentParser(description='採用予測モデルを学習します')
    parser.add_argument('--data', type=str, default='data/hiring_data.csv', help='学習データのパス')
    parser.add_argument('--test_size', type=float, default=0.2, help='テストデータの割合')
    parser.add_argument('--seed', type=int, default=42, help='ランダムシード')
    parser.add_argument('--n_estimators', type=int, default=100, help='決定木の数')
    parser.add_argument('--max_depth', type=int, default=10, help='木の最大深さ')

    args = parser.parse_args()

    mlflow.set_tracking_uri("http://localhost:9800")
    mlflow.start_run()

    # Log parameters
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # データ読み込み
    data = load_data(args.data)

    # 特徴量と目的変数の分離
    X, y = prepare_features_target(data)

    # 学習・テストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print(f"\n学習データ: {X_train.shape[0]} サンプル")
    print(f"テストデータ: {X_test.shape[0]} サンプル")

    # モデル学習
    model = train_model(X_train, y_train, args.n_estimators, args.max_depth, args.seed)

    # モデル評価
    metrics = evaluate_model(model, X_test, y_test)

    # Log metrics
    mlflow.log_metric("accuracy", metrics['accuracy'])

    # Log feature importances as parameters
    for feature, importance in metrics['feature_importance'].items():
        mlflow.log_param(f"feature_importance_{feature}", importance)
    
    # モデルをローカルに保存し、アーティファクトとして記録
    model_path = "hiring_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, "model")
    
    print("\nMLflow Run ID:", mlflow.active_run().info.run_id)
    mlflow.end_run()

    print(f"\n学習完了！")

if __name__ == "__main__":
    main() 