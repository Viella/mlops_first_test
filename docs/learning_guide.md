# MLOps学習ガイド

## 🎯 このプロジェクトで学べるMLOpsコンセプト

### レベル1: 基礎概念の理解 🌱

#### 1. MLパイプライン
**学習内容**: ML workflowの基本構造
```
データ生成 → 前処理 → 学習 → 評価 → 推論
```

**実践ポイント**:
- 各ステップの役割を理解
- データの流れを追跡
- 各段階での出力を確認

**確認方法**:
```bash
# パイプライン全体を実行
python src/data_generation.py
python src/train.py  
python src/inference.py --mode single --age 30 --experience 5 --skill 8 --interview 7 --education 4
```

#### 2. モデルライフサイクル
**学習内容**: モデルの誕生から廃棄まで
```
開発 → 学習 → 検証 → デプロイ → 監視 → 更新/廃棄
```

**実践ポイント**:
- `models/hiring_model.pkl` - モデルの永続化
- `models/metrics.json` - 性能の記録
- バージョン管理の重要性

#### 3. 再現性 (Reproducibility)
**学習内容**: 同じ結果を何度でも得られる仕組み

**実装例**:
```python
# 全ての乱数要素を制御
np.random.seed(42)
model = RandomForestClassifier(random_state=42)
train_test_split(..., random_state=42)
```

**確認方法**:
```bash
# 同じseedで2回実行 → 同じ結果が得られることを確認
python src/train.py --seed 42
python src/train.py --seed 42
```

### レベル2: 環境管理とコンテナ化 🐳

#### 1. 環境分離 (Environment Isolation)
**学習内容**: 学習環境と推論環境の分離

**なぜ分離？**:
- 学習: 重い依存関係、一時的実行
- 推論: 軽量、常時稼働

**実践ポイント**:
```dockerfile
# 学習用: フル機能
FROM python:3.9-slim
RUN pip install pandas numpy scikit-learn matplotlib seaborn

# 推論用: 最小構成  
FROM python:3.9-slim
RUN pip install pandas numpy scikit-learn joblib
```

#### 2. インフラストラクチャ as Code
**学習内容**: インフラ設定のコード化

**実装例**: `docker-compose.yml`
```yaml
services:
  training:    # 学習環境定義
    build:
      dockerfile: docker/training.Dockerfile
      
  inference:   # 推論環境定義  
    build:
      dockerfile: docker/inference.Dockerfile
    depends_on:
      - training
```

**実践演習**:
```bash
# 環境の一括構築・実行
docker-compose up

# 個別サービス実行
docker-compose up training
docker-compose up inference
```

### レベル3: CI/CD自動化 🔄

#### 1. 継続的インテグレーション (CI)
**学習内容**: コード変更時の自動テスト

**実装構造**:
```yaml
# .github/workflows/mlops.yml
jobs:
  training:      # 学習の自動実行
  inference:     # 推論テストの自動実行  
  model-validation: # 性能チェックの自動実行
```

**学習効果**:
- コード品質の自動保証
- 人的ミスの削減
- 開発効率向上

#### 2. 継続的デプロイメント (CD)  
**学習内容**: モデル更新の自動化

**実装例**:
```yaml
- name: Upload model artifacts
  uses: actions/upload-artifact@v3
  with:
    name: trained-model
    retention-days: 30
```

#### 3. モデル品質管理
**学習内容**: 性能劣化の自動検出

**実装例**:
```python
# 最小精度チェック（50%以上）
if accuracy < 0.5:
    print('❌ Model accuracy is below threshold')
    exit(1)
else:
    print('✅ Model accuracy is acceptable')
```

### レベル4: 実践的MLOps 🚀

#### 1. アーティファクト管理
**学習内容**: モデル・データの版数管理

**現在の実装**:
- `models/hiring_model.pkl` - モデルファイル
- `models/metrics.json` - 評価指標
- `data/hiring_data.csv` - 学習データ

**発展課題**:
- MLflow導入
- DVC (Data Version Control)
- モデルレジストリ

#### 2. 監視・モニタリング
**学習内容**: 本番モデルの性能追跡

**基礎実装**:
```python
# 推論時の確信度記録
result = {
    'prediction': int(prediction),
    'probability_hired': float(probability[1]),
    'timestamp': datetime.now()
}
```

**発展課題**:
- データドリフト検出
- モデルドリフト検出  
- アラート機能

#### 3. 実験管理
**学習内容**: 複数モデルの比較・管理

**現在のパラメータ化**:
```python
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--test_size', type=float, default=0.2)
```

**発展課題**:
- ハイパーパラメータ最適化
- A/Bテスト
- 実験結果の自動比較

## 📚 学習順序の推奨

### ステップ1: ローカル環境での理解 (1-2時間)
```bash
# 基本動作確認
python src/data_generation.py --samples 100
python src/train.py
python src/inference.py --mode single --age 30 --experience 5 --skill 8 --interview 7 --education 4

# パラメータ変更実験
python src/data_generation.py --samples 1000 --seed 100
python src/train.py --test_size 0.3
```

**理解目標**:
- MLパイプラインの流れ
- 各スクリプトの役割
- パラメータの影響

### ステップ2: Docker化の体験 (1-2時間)
```bash
# Docker環境での実行
docker-compose up training
docker-compose up inference

# インタラクティブ実行
docker-compose exec inference python src/inference.py --mode single --age 25 --experience 3 --skill 9 --interview 8 --education 5
```

**理解目標**:
- コンテナ化の利点
- 環境分離の重要性
- インフラの抽象化

### ステップ3: CI/CD自動化の理解 (2-3時間)  
```bash
# GitHubにpush
git add .
git commit -m "MLOps pipeline setup"
git push origin main
```

**GitHub Actionsで確認**:
- Actions タブでワークフロー実行
- 各ジョブの実行ログ確認
- Artifactsの確認

**理解目標**:
- 自動化の価値
- 品質管理の自動化
- チーム開発での利点

### ステップ4: 拡張実装 (3-5時間)
**選択課題**:
1. **新しい特徴量追加**
   - 職種、地域、企業規模等
   - データ生成ロジックの拡張

2. **モデル改善**
   - XGBoost, LightGBM導入
   - ハイパーパラメータ最適化

3. **API化**
   - FastAPIでの推論エンドポイント
   - Dockerでのサービス化

4. **監視機能**
   - ログ出力の充実
   - 性能指標の可視化

## 🧠 重要概念の深掘り

### 1. Infrastructure as Code (IaC)
**なぜ重要？**
- 環境の一貫性確保
- 手動作業削減
- スケーラビリティ

**実践例**:
- `docker-compose.yml` - コンテナ環境定義
- `.github/workflows/` - CI/CD定義
- `pyproject.toml` - 依存関係定義

### 2. モデルドリフト
**概念**: 時間経過によるモデル性能劣化

**検出方法**:
```python
# 基本的な性能監視
current_accuracy = evaluate_model(model, new_data)
if current_accuracy < baseline_accuracy * 0.95:  # 5%劣化で警告
    alert("Model performance degraded")
```

### 3. Feature Store
**概念**: 特徴量の中央管理システム

**簡易実装例**:
```python
# 特徴量定義の標準化
FEATURE_SCHEMA = {
    'age': {'type': 'int', 'range': (22, 65)},
    'experience_years': {'type': 'int', 'range': (0, 30)},
    'skill_score': {'type': 'int', 'range': (1, 10)},
}
```

## 🎓 学習評価チェックリスト

### 基礎レベル ✅
- [ ] MLパイプラインの全ステップを説明できる
- [ ] モデルの学習・推論を実行できる
- [ ] 評価指標を解釈できる
- [ ] 再現性の重要性を理解している

### 中級レベル ✅  
- [ ] Docker環境でパイプラインを実行できる
- [ ] 学習・推論環境の分離意義を説明できる
- [ ] CI/CDパイプラインを構築・実行できる
- [ ] アーティファクト管理を実践できる

### 上級レベル ✅
- [ ] 新しい特徴量・モデルを追加できる
- [ ] 性能監視システムを設計できる
- [ ] スケーラブルなMLOpsアーキテクチャを提案できる
- [ ] チーム開発でのMLOps導入を推進できる

## 🔧 トラブルシューティング学習

### よくある課題と解決法

#### 1. 「モデルファイルが見つからない」
```python
FileNotFoundError: モデルファイルが見つかりません: models/hiring_model.pkl
```
**解決**: 先に学習を実行
```bash
python src/train.py
```

#### 2. 「Docker buildに失敗」
```
ERROR: failed to solve: failed to read dockerfile
```
**解決**: Docker contextの確認
```bash
# プロジェクトルートで実行
docker build -f docker/training.Dockerfile -t mlops-training .
```

#### 3. 「GitHub Actions失敗」
```
Error: No such file or directory
```
**解決**: ファイルパスとcommit状況の確認

これらの体験を通じて、MLOpsの実践的スキルを習得できます！

---

**学習目標**: 実際の業務でMLOpsを推進できるレベルの理解と実装力の獲得 