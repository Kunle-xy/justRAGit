FROM python:3.10-slim

WORKDIR /app

COPY . /app
COPY .env ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "rag_demo.py"]
