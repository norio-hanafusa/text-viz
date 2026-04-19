FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      fonts-noto-cjk \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# PyTorch を CPU 専用 index から先に入れて CUDA ライブラリ (cudnn/nccl/cusparselt 等、~1.5GB) を回避。
# そのあと残りの依存を PyPI から入れると sentence-transformers は既存 torch を再利用する。
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir -r requirements.txt \
 && python -c "import streamlit; print('streamlit', streamlit.__version__)" \
 && python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())" \
 && python -m nltk.downloader -d ${NLTK_DATA} stopwords vader_lexicon punkt \
 && (python -m spacy download en_core_web_sm || echo "spacy en_core_web_sm のダウンロード失敗 (ルールベース NER にフォールバック)")

# アプリ本体 (jp_nlp_toolkit/ パッケージも同梱されている)
COPY . .

EXPOSE 8501

# python -m streamlit なら PATH に streamlit が無くても起動可能
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
