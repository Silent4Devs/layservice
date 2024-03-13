FROM langchain/langchain AS base

# Install Tesseract OCR and language packs
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-spa \
        build-essential \
        curl \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt
# RUN pip install --upgrade -r /app/requirements.txt

COPY . /app/

WORKDIR /app

# COPY bot.py .
# COPY utils.py .
# COPY chains.py .

EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health
# CMD ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0"]