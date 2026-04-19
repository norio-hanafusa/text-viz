FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# pip install → 直後に streamlit の存在確認 (サイレント失敗検知)
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import streamlit; print('streamlit', streamlit.__version__)" \
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
