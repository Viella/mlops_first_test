# 実装メモ

## 🔧 技術的判断と実装詳細

### データ生成の実装判断

#### なぜ合成データを生成？
- **理由**: 実際の採用データは機密性が高く、学習目的では利用困難
- **メリット**: 
  - データサイズ調整可能
  - バイアス制御可能
  - 再現性確保
- **実装ポイント**: 
  ```python
  # 現実的な採用確率を模擬
  base_prob = 0.3  # ベース採用率30%
  # 各特徴量の寄与度を調整
  skill_boost = (skill_score - 5) * 0.05  # スキルが5以上で有利
  ```

#### 特徴量設計の考慮点
- **年齢範囲**: 22-65歳（新卒〜定年前）
- **経験年数**: 0-30年（現実的な範囲）
- **スコア系**: 1-10スケール（直感的）
- **学歴レベル**: 1-5段階（高校〜大学院相当）

### モデル学習の実装判断

#### Random Forest選択理由
```python
RandomForestClassifier(
    n_estimators=100,      # 安定性重視
    max_depth=10,          # 過学習防止
    random_state=42,       # 再現性確保
    n_jobs=-1             # 高速化
)
```

**他候補との比較**:
- ❌ Deep Learning: オーバーキル、解釈困難
- ❌ SVM: パラメータチューニング複雑
- ❌ Logistic Regression: 非線形関係を捉えられない
- ✅ Random Forest: バランスが良い

#### 評価指標の選定
```python
# 分類レポート - 総合的な性能把握
classification_report(y_test, y_pred)

# 特徴量重要度 - ビジネス洞察
feature_importances_ 

# 混同行列 - 誤分類パターン把握
confusion_matrix(y_test, y_pred)
```

### Docker実装の工夫

#### 学習用Dockerfileの最適化
```dockerfile
# ベースイメージ: python:3.9-slim（軽量性とライブラリ互換性）
FROM python:3.9-slim

# システムパッケージ最小化
RUN apt-get update && apt-get install -y \
    gcc \  # scikit-learn コンパイル用
    && rm -rf /var/lib/apt/lists/*  # イメージサイズ削減
```

**判断理由**:
- `python:3.9-slim`: 安定版 + 軽量
- `gcc`のみインストール: scikit-learn要求最小限
- キャッシュクリア: イメージサイズ最適化

#### 推論用Dockerfileの軽量化
```dockerfile
# 推論に必要最小限のパッケージのみ
RUN pip install --no-cache-dir \
    pandas>=1.5.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.1.0 \
    joblib>=1.2.0
```

**最適化ポイント**:
- `--no-cache-dir`: キャッシュ削除でサイズ削減
- 学習用ライブラリ除外（matplotlib等）
- バージョン固定で安定性確保

### CI/CD実装の工夫

#### GitHub Actions設計方針
```yaml
# 並列実行でパイプライン高速化
jobs:
  training:     # 学習ジョブ
  inference:    # 推論ジョブ（training依存）
  docker-test:  # Docker検証（mainのみ）
  model-validation:  # 性能チェック（training依存）
```

#### アーティファクト管理
```yaml
# モデルと学習データを30日間保持
uses: actions/upload-artifact@v3
with:
  name: trained-model
  retention-days: 30
```

**判断理由**:
- 30日保持: 実験サイクルに十分
- モデル+データセット: 推論ジョブで必要
- 自動クリーンアップ: ストレージ節約

### 推論実装の工夫

#### クラス設計の採用
```python
class HiringPredictor:
    def __init__(self, model_path: str):
        # モデル読み込みとバリデーション
        
    def predict_single(self, ...):
        # 単一予測
        
    def predict_batch(self, data_path: str):
        # バッチ予測
```

**メリット**:
- モデル読み込み1回のみ
- 予測メソッド統一化
- エラーハンドリング一元化

#### エラーハンドリング戦略
```python
# モデルファイル存在チェック
if not self.model_path.exists():
    raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

# 特徴量不足チェック  
missing_columns = set(self.feature_columns) - set(data.columns)
if missing_columns:
    raise ValueError(f"必要な特徴量が不足しています: {missing_columns}")
```

## 🐛 実装時のハマりポイント

### 1. Docker コンテキスト問題
**問題**: COPY時にファイルが見つからない
```dockerfile
# ❌ ダメな例
COPY src/ ./src/
COPY models/ ./models/  # buildx時にmodelsが存在しない
```

**解決策**:
```dockerfile  
# ✅ 良い例
RUN mkdir -p models  # 事前にディレクトリ作成
```

### 2. Python依存関係の競合
**問題**: scikit-learn version不整合
**解決策**: pyproject.tomlでバージョン固定
```toml
dependencies = [
    "scikit-learn>=1.1.0",  # 下限指定で互換性確保
]
```

### 3. GitHub Actions Artifact渡し
**問題**: ジョブ間でのファイル受け渡し
**解決策**: 
```yaml
# uploadとdownloadの名前統一
- uses: actions/upload-artifact@v3
  with:
    name: trained-model  # 統一名
    
- uses: actions/download-artifact@v3  
  with:
    name: trained-model  # 同じ名前で取得
```

### 4. パスの扱い
**問題**: 相対パス vs 絶対パス
**解決策**: `pathlib.Path`の統一使用
```python
from pathlib import Path

output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
```

## 💡 ベストプラクティス

### 1. 設定の外部化
```python
# ❌ ハードコーディング
model = RandomForestClassifier(n_estimators=100, max_depth=10)

# ✅ 引数化
parser.add_argument('--n_estimators', type=int, default=100)
```

### 2. ログ出力の充実
```python
print(f"データを読み込みました: {data_path}")
print(f"データ形状: {data.shape}")
print(f"学習データ: {X_train.shape[0]} サンプル")
```

### 3. バリデーションの徹底
```python  
# データ形状チェック
assert len(data) > 0, "データが空です"

# 特徴量存在チェック
required_cols = ['age', 'experience_years', ...]
assert all(col in data.columns for col in required_cols)
```

### 4. 再現性の確保
```python
# 全ての乱数シードを統一
np.random.seed(random_state)
model = RandomForestClassifier(random_state=random_state)
train_test_split(..., random_state=random_state)
```

### 5. リソース最適化
```python
# 並列処理活用
RandomForestClassifier(n_jobs=-1)

# メモリ効率化
data.to_csv(output_path, index=False)  # インデックス保存不要
```

## 🔄 デバッグ・テスト戦略

### ローカルテスト手順
```bash
# 1. データ生成テスト
python src/data_generation.py --samples 100

# 2. 学習テスト  
python src/train.py

# 3. 推論テスト
python src/inference.py --mode single --age 30 --experience 5 --skill 8 --interview 7 --education 4
```

### Docker動作確認
```bash
# イメージビルド確認
docker build -f docker/training.Dockerfile -t test-training .

# 実行テスト
docker run test-training
```

### CI/CD動作確認
- Push前にローカルで全工程確認
- PRでの動作確認
- mainブランチでDocker Test実行

## 📊 パフォーマンス考慮点

### データサイズ
- **開発時**: 500-1000サンプル（高速反復）
- **本格運用**: 10000+サンプル（性能重視）

### モデル選択
- **プロトタイプ**: Random Forest（安定）
- **本格運用**: XGBoost, LightGBM（性能重視）

### インフラ
- **学習**: CPU並列（n_jobs=-1）
- **推論**: メモリ最適化（軽量モデル）

---

**更新履歴**:
- 2024/07/29: 初版作成
- 実装完了後の振り返りメモ 