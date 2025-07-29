import pandas as pd
import numpy as np
import joblib
import argparse
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

def train_model(X_train, y_train, random_state: int = 42):
    """RandomForestモデルを学習"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
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
    feature_importance = dict(zip(feature_names, importances))
    
    print(f"\n特徴量重要度:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {importance:.4f}")
    
    return {
        'accuracy': accuracy,
        'feature_importance': feature_importance
    }

def save_model_and_metrics(model, metrics: dict, model_path: str, metrics_path: str):
    """モデルと評価指標を保存"""
    # モデル保存
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"モデルを保存しました: {model_path}")
    
    # 評価指標保存
    metrics_path = Path(metrics_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"評価指標を保存しました: {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description='採用予測モデルを学習します')
    parser.add_argument('--data', type=str, default='data/hiring_data.csv', help='学習データのパス')
    parser.add_argument('--model_output', type=str, default='models/hiring_model.pkl', help='モデル出力パス')
    parser.add_argument('--metrics_output', type=str, default='models/metrics.json', help='評価指標出力パス')
    parser.add_argument('--test_size', type=float, default=0.2, help='テストデータの割合')
    parser.add_argument('--seed', type=int, default=42, help='ランダムシード')
    
    args = parser.parse_args()
    
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
    model = train_model(X_train, y_train, args.seed)
    
    # モデル評価
    metrics = evaluate_model(model, X_test, y_test)
    
    # モデルと評価指標を保存
    save_model_and_metrics(model, metrics, args.model_output, args.metrics_output)
    
    print(f"\n学習完了！")

if __name__ == "__main__":
    main() 