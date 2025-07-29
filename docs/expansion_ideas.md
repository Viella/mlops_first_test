# 拡張アイデア集

## 🚀 次のステップ：機能拡張案

### Phase 1: 基本機能の強化 (初級〜中級)

#### 1.1 データセット多様化
**概要**: より現実的で複雑なデータセットの生成

**実装案**:
```python
# 新しい特徴量
features_v2 = {
    'job_category': 'categorical',  # 職種カテゴリ
    'location': 'categorical',      # 勤務地  
    'company_size': 'categorical',  # 企業規模
    'salary_expectation': 'float',  # 希望年収
    'availability': 'datetime',     # 入社可能日
    'language_skills': 'list',      # 語学スキル
    'certifications': 'int',        # 資格数
    'previous_companies': 'int'     # 転職回数
}

# 時系列データ対応
def generate_time_series_data(start_date, end_date, samples_per_day):
    """期間内で時系列採用データを生成"""
    pass
```

**学習効果**:
- カテゴリカル変数の前処理
- 時系列データ処理
- 特徴量エンジニアリング

#### 1.2 モデル選択の自動化
**概要**: 複数アルゴリズムの自動比較・選択

**実装案**:
```python
# models/model_comparison.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class AutoModelSelector:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(),
            'gradient_boosting': GradientBoostingClassifier(),
            'svm': SVC(probability=True),
            'logistic_regression': LogisticRegression()
        }
    
    def auto_select(self, X_train, y_train, X_val, y_val):
        """複数モデルを学習して最適モデルを選択"""
        best_model = None
        best_score = 0
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            if score > best_score:
                best_score = score
                best_model = (name, model)
        
        return best_model
```

#### 1.3 ハイパーパラメータ最適化
**概要**: Optuna, Hyperoptによる自動最適化

**実装案**:
```python
# src/hyperparameter_tuning.py
import optuna

def objective(trial):
    # パラメータサンプリング
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    # モデル学習・評価
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    # クロスバリデーション
    scores = cross_val_score(model, X_train, y_train, cv=5)
    return scores.mean()

# 最適化実行
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Phase 2: API化とマイクロサービス (中級〜上級)

#### 2.1 推論API開発
**概要**: FastAPIによるREST API化

**実装案**:
```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(title="Hiring Prediction API")

class PredictionRequest(BaseModel):
    age: int
    experience_years: int
    skill_score: int
    interview_score: int
    education_level: int

class PredictionResponse(BaseModel):
    prediction: int
    probability_hired: float
    confidence: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        model = joblib.load("models/hiring_model.pkl")
        # 予測実行
        result = predictor.predict_single(**request.dict())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Docker化**:
```dockerfile
# docker/api.Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src/api/ ./api/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2.2 フロントエンド開発
**概要**: Streamlit/Gradioによる簡易UI

**実装案**:
```python
# src/frontend/app.py
import streamlit as st
import requests
import json

st.title("採用予測システム")

# 入力フォーム
age = st.slider("年齢", 22, 65, 30)
experience = st.slider("経験年数", 0, 30, 5)
skill = st.slider("スキルスコア", 1, 10, 7)
interview = st.slider("面接スコア", 1, 10, 8)
education = st.selectbox("学歴レベル", [1, 2, 3, 4, 5])

if st.button("予測実行"):
    # API呼び出し
    response = requests.post("http://api:8000/predict", json={
        "age": age,
        "experience_years": experience,
        "skill_score": skill,
        "interview_score": interview,
        "education_level": education
    })
    
    if response.status_code == 200:
        result = response.json()
        st.success(f"採用予測: {'採用' if result['prediction'] == 1 else '不採用'}")
        st.info(f"採用確率: {result['probability_hired']:.2%}")
    else:
        st.error("予測に失敗しました")
```

### Phase 3: 実験管理・監視 (上級)

#### 3.1 MLflow導入
**概要**: 実験管理・モデル管理の高度化

**実装案**:
```python
# src/train_with_mlflow.py
import mlflow
import mlflow.sklearn

# 実験追跡開始
mlflow.set_experiment("hiring_prediction")

with mlflow.start_run():
    # パラメータログ
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # モデル学習
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # 評価指標ログ
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    # モデル保存
    mlflow.sklearn.log_model(model, "model")
    
    # アーティファクトログ
    mlflow.log_artifact("models/metrics.json")
```

**Docker Compose拡張**:
```yaml
# docker-compose.yml に追加
services:
  mlflow:
    image: python:3.9-slim
    command: |
      bash -c "
        pip install mlflow &&
        mlflow server --host 0.0.0.0 --port 5000
      "
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
```

#### 3.2 モデル監視システム
**概要**: データドリフト・モデルドリフト検出

**実装案**:
```python
# src/monitoring/drift_detector.py
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

class DriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        
    def detect_data_drift(self, current_data):
        """データドリフトを検出"""
        report = Report(metrics=[DataDriftPreset()])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=ColumnMapping()
        )
        
        return report
        
    def detect_model_drift(self, model, current_data):
        """モデルドリフトを検出"""
        current_accuracy = model.score(current_data['features'], current_data['target'])
        baseline_accuracy = self.baseline_accuracy
        
        drift_ratio = current_accuracy / baseline_accuracy
        
        if drift_ratio < 0.95:  # 5%以上の性能劣化
            return {
                'drift_detected': True,
                'current_accuracy': current_accuracy,
                'baseline_accuracy': baseline_accuracy,
                'drift_ratio': drift_ratio
            }
        
        return {'drift_detected': False}
```

#### 3.3 A/Bテスト基盤
**概要**: 複数モデルの本番比較

**実装案**:
```python
# src/ab_testing/experiment.py
import random
from typing import Dict, Any

class ABTestManager:
    def __init__(self, models: Dict[str, Any], traffic_split: Dict[str, float]):
        self.models = models
        self.traffic_split = traffic_split
        
    def route_request(self, request_id: str):
        """リクエストをモデルにルーティング"""
        # ハッシュベースの分散
        hash_value = hash(request_id) % 100
        
        cumulative = 0
        for model_name, ratio in self.traffic_split.items():
            cumulative += ratio * 100
            if hash_value < cumulative:
                return model_name
                
        return list(self.models.keys())[0]  # フォールバック
        
    async def predict_with_routing(self, request_id: str, features: Dict):
        """A/Bテスト対応予測"""
        model_name = self.route_request(request_id)
        model = self.models[model_name]
        
        # 予測実行
        prediction = model.predict([list(features.values())])[0]
        
        # ログ記録
        self.log_experiment(request_id, model_name, features, prediction)
        
        return {
            'prediction': prediction,
            'model_used': model_name
        }
```

### Phase 4: 本格運用対応 (上級〜エキスパート)

#### 4.1 Kubernetes対応
**概要**: スケーラブルなデプロイメント

**実装案**:
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hiring-prediction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hiring-prediction-api
  template:
    metadata:
      labels:
        app: hiring-prediction-api
    spec:
      containers:
      - name: api
        image: hiring-prediction:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/hiring_model.pkl"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: hiring-prediction-service
spec:
  selector:
    app: hiring-prediction-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### 4.2 データパイプライン自動化
**概要**: Apache Airflow, Prefect等の導入

**実装案**:
```python
# dags/hiring_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def extract_data():
    """データ抽出"""
    pass

def transform_data():
    """データ変換"""
    pass

def train_model():
    """モデル学習"""
    pass

def deploy_model():
    """モデルデプロイ"""
    pass

# DAG定義
dag = DAG(
    'hiring_model_pipeline',
    default_args={
        'owner': 'mlops-team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    },
    description='採用予測モデルパイプライン',
    schedule_interval='@daily',
    catchup=False
)

# タスク定義
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# 依存関係
extract_task >> transform_task >> train_task >> deploy_task
```

#### 4.3 セキュリティ強化
**概要**: 認証・認可、データ暗号化

**実装案**:
```python
# src/api/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

# API保護
@app.post("/predict")
async def predict(
    request: PredictionRequest, 
    current_user: str = Depends(verify_token)
):
    # 予測処理
    pass
```

## 🎯 学習段階別推奨拡張

### 初級者向け (基本理解後)
1. **特徴量追加** - 新しいカラムでの実験
2. **パラメータ調整** - 手動でのハイパーパラメータ変更
3. **評価指標追加** - F1-score, AUC等の実装
4. **可視化強化** - matplotlib, seabornでのグラフ作成

### 中級者向け (Docker理解後)
1. **API化** - FastAPIでの推論エンドポイント
2. **フロントエンド** - Streamlitでの簡易UI
3. **実験管理** - MLflowの基本導入  
4. **監視基盤** - ログ収集・可視化

### 上級者向け (CI/CD理解後)
1. **マイクロサービス化** - 完全なサービス分離
2. **Kubernetes** - スケーラブルなデプロイ
3. **データパイプライン** - Airflowでの自動化
4. **高度な監視** - ドリフト検出・A/Bテスト

## 💡 業界別応用アイデア

### 1. 金融業界
**応用**: 与信審査、不正検知
**拡張ポイント**:
- 時系列データ対応
- リスク管理機能
- 規制対応ログ

### 2. 製造業
**応用**: 品質管理、予防保全
**拡張ポイント**:  
- IoTデータ統合
- リアルタイム監視
- 異常検知機能

### 3. 小売業
**応用**: 需要予測、レコメンド
**拡張ポイント**:
- 季節性対応  
- 在庫最適化
- パーソナライゼーション

### 4. ヘルスケア
**応用**: 診断支援、創薬
**拡張ポイント**:
- プライバシー保護
- FDA承認プロセス
- 医師向けUI

## 🔄 継続的改善の仕組み

### 1. 定期的なモデル再学習
```python
# scripts/retrain_schedule.py
import schedule
import time

def retrain_job():
    """定期再学習ジョブ"""
    print("Starting model retraining...")
    # データ取得 → 学習 → 評価 → デプロイ
    pass

# 毎週日曜日に実行
schedule.every().sunday.at("02:00").do(retrain_job)

while True:
    schedule.run_pending()
    time.sleep(3600)  # 1時間待機
```

### 2. フィードバックループ
```python
# src/feedback/collector.py
class FeedbackCollector:
    def collect_prediction_feedback(self, prediction_id, actual_result):
        """予測結果のフィードバック収集"""
        feedback = {
            'prediction_id': prediction_id,
            'predicted': self.get_prediction(prediction_id),
            'actual': actual_result,
            'timestamp': datetime.now(),
            'accuracy': predicted == actual_result
        }
        
        self.store_feedback(feedback)
        return feedback
```

### 3. 性能分析ダッシュボード
```python
# src/dashboard/metrics.py  
import plotly.dash as dash
import plotly.graph_objs as go

def create_performance_dashboard():
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(id='accuracy-trend'),
        dcc.Graph(id='prediction-distribution'),
        dcc.Graph(id='feature-importance')
    ])
    
    return app
```

これらの拡張により、エンタープライズレベルのMLOpsシステムへと発展させることができます！

---

**重要**: 段階的に実装し、各ステップで十分に理解を深めることが成功の鍵です。 