FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# システムパッケージの更新とクリーンアップ
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Pythonの依存関係をコピーしてインストール
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# ソースコードをコピー
COPY src/ ./src/
COPY data/ ./data/

# モデル保存用ディレクトリを作成
RUN mkdir -p models

# デフォルトコマンド（データ生成 → 学習）
CMD ["sh", "-c", "python src/data_generation.py && python src/train.py"] 