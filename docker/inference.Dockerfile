FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# 推論に必要な最小限のパッケージをインストール
RUN pip install --no-cache-dir \
    pandas>=1.5.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.1.0 \
    joblib>=1.2.0

# 推論スクリプトとモデルをコピー
COPY src/inference.py ./src/
COPY models/ ./models/

# ヘルスチェック用のエンドポイント（オプション）
EXPOSE 8000

# デフォルトコマンド（推論スクリプトの実行）
CMD ["python", "src/inference.py", "--help"] 