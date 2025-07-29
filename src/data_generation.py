import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def generate_hiring_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    採用データを生成する関数
    
    特徴量:
    - age: 年齢 (22-65)  
    - experience_years: 経験年数 (0-30)
    - skill_score: スキルスコア (1-10)
    - interview_score: 面接スコア (1-10)
    - education_level: 学歴レベル (1-5, 5が最高)
    
    目的変数:
    - hired: 採用フラグ (0 or 1)
    """
    np.random.seed(random_state)
    
    # 特徴量生成
    age = np.random.randint(22, 66, n_samples)
    experience_years = np.random.randint(0, 31, n_samples)
    skill_score = np.random.randint(1, 11, n_samples)
    interview_score = np.random.randint(1, 11, n_samples)
    education_level = np.random.randint(1, 6, n_samples)
    
    # 採用確率の計算（シンプルなルールベース）
    # スキルスコア、面接スコア、学歴が高いほど採用されやすい
    # 経験年数も考慮（ただし30年以上は逆に不利）
    base_prob = 0.3
    skill_boost = (skill_score - 5) * 0.05
    interview_boost = (interview_score - 5) * 0.05
    education_boost = (education_level - 3) * 0.03
    experience_boost = np.minimum(experience_years * 0.01, 0.15)
    
    hire_prob = base_prob + skill_boost + interview_boost + education_boost + experience_boost
    hire_prob = np.clip(hire_prob, 0.05, 0.95)  # 確率を0.05-0.95に制限
    
    # 採用フラグの決定
    hired = np.random.binomial(1, hire_prob, n_samples)
    
    # DataFrameの作成
    data = pd.DataFrame({
        'age': age,
        'experience_years': experience_years,
        'skill_score': skill_score,
        'interview_score': interview_score,
        'education_level': education_level,
        'hired': hired
    })
    
    return data

def main():
    parser = argparse.ArgumentParser(description='採用データを生成します')
    parser.add_argument('--samples', type=int, default=1000, help='生成するサンプル数')
    parser.add_argument('--output', type=str, default='data/hiring_data.csv', help='出力ファイルパス')
    parser.add_argument('--seed', type=int, default=42, help='ランダムシード')
    
    args = parser.parse_args()
    
    # データ生成
    print(f"採用データを生成中... (サンプル数: {args.samples})")
    data = generate_hiring_data(n_samples=args.samples, random_state=args.seed)
    
    # 出力ディレクトリの作成
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV出力
    data.to_csv(output_path, index=False)
    print(f"データを保存しました: {output_path}")
    
    # データの統計情報を表示
    print("\n=== データ統計 ===")
    print(f"総サンプル数: {len(data)}")
    print(f"採用率: {data['hired'].mean():.2%}")
    print("\n特徴量の統計:")
    print(data.describe())

if __name__ == "__main__":
    main() 