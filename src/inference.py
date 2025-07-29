import pandas as pd
import joblib
import argparse
import json
from pathlib import Path
from typing import Union, List, Dict

class HiringPredictor:
    """採用予測器クラス"""
    
    def __init__(self, model_path: str):
        """モデルを読み込んで初期化"""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
            
        self.model = joblib.load(self.model_path)
        self.feature_columns = ['age', 'experience_years', 'skill_score', 'interview_score', 'education_level']
        print(f"モデルを読み込みました: {model_path}")
    
    def predict_single(self, age: int, experience_years: int, skill_score: int, 
                      interview_score: int, education_level: int) -> Dict:
        """単一サンプルの予測"""
        # 入力データの準備
        input_data = pd.DataFrame({
            'age': [age],
            'experience_years': [experience_years],
            'skill_score': [skill_score],
            'interview_score': [interview_score],
            'education_level': [education_level]
        })
        
        # 予測実行
        prediction = self.model.predict(input_data)[0]
        probability = self.model.predict_proba(input_data)[0]
        
        return {
            'prediction': int(prediction),
            'probability_not_hired': float(probability[0]),
            'probability_hired': float(probability[1]),
            'input': {
                'age': age,
                'experience_years': experience_years,
                'skill_score': skill_score,
                'interview_score': interview_score,
                'education_level': education_level
            }
        }
    
    def predict_batch(self, data_path: str) -> List[Dict]:
        """バッチ予測（CSVファイル）"""
        # データ読み込み
        data = pd.read_csv(data_path)
        print(f"予測対象データを読み込みました: {data_path}")
        print(f"データ形状: {data.shape}")
        
        # 必要な特徴量が存在するかチェック
        missing_columns = set(self.feature_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"必要な特徴量が不足しています: {missing_columns}")
        
        # 特徴量を抽出
        X = data[self.feature_columns]
        
        # 予測実行
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # 結果をまとめる
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'index': i,
                'prediction': int(pred),
                'probability_not_hired': float(prob[0]),
                'probability_hired': float(prob[1]),
                'input': data.iloc[i][self.feature_columns].to_dict()
            }
            results.append(result)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='採用予測を実行します')
    parser.add_argument('--model', type=str, default='models/hiring_model.pkl', help='モデルファイルのパス')
    
    # 予測モード
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single', 
                       help='予測モード (single: 単一予測, batch: バッチ予測)')
    
    # 単一予測用パラメータ
    parser.add_argument('--age', type=int, help='年齢')
    parser.add_argument('--experience', type=int, help='経験年数')
    parser.add_argument('--skill', type=int, help='スキルスコア (1-10)')
    parser.add_argument('--interview', type=int, help='面接スコア (1-10)')
    parser.add_argument('--education', type=int, help='学歴レベル (1-5)')
    
    # バッチ予測用パラメータ
    parser.add_argument('--input', type=str, help='入力CSVファイルのパス')
    parser.add_argument('--output', type=str, help='出力JSONファイルのパス')
    
    args = parser.parse_args()
    
    # 予測器の初期化
    try:
        predictor = HiringPredictor(args.model)
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return
    
    if args.mode == 'single':
        # 単一予測
        if not all([args.age, args.experience, args.skill, args.interview, args.education]):
            print("単一予測には以下のパラメータが必要です:")
            print("  --age, --experience, --skill, --interview, --education")
            return
        
        result = predictor.predict_single(
            args.age, args.experience, args.skill, args.interview, args.education
        )
        
        print(f"\n=== 予測結果 ===")
        print(f"採用予測: {'採用' if result['prediction'] == 1 else '不採用'}")
        print(f"採用確率: {result['probability_hired']:.4f}")
        print(f"不採用確率: {result['probability_not_hired']:.4f}")
        print(f"\n入力データ:")
        for key, value in result['input'].items():
            print(f"  {key}: {value}")
    
    elif args.mode == 'batch':
        # バッチ予測
        if not args.input:
            print("バッチ予測には --input パラメータが必要です")
            return
        
        try:
            results = predictor.predict_batch(args.input)
            
            print(f"\n=== バッチ予測結果 ===")
            print(f"総予測数: {len(results)}")
            
            hired_count = sum(1 for r in results if r['prediction'] == 1)
            print(f"採用予測数: {hired_count}")
            print(f"採用率: {hired_count / len(results):.2%}")
            
            # 結果を保存
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\n結果を保存しました: {output_path}")
            else:
                # 最初の5件を表示
                print(f"\n予測結果（最初の5件）:")
                for result in results[:5]:
                    pred_text = '採用' if result['prediction'] == 1 else '不採用'
                    print(f"  Index {result['index']}: {pred_text} (確率: {result['probability_hired']:.4f})")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"エラー: {e}")

if __name__ == "__main__":
    main() 