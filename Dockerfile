FROM python:3.11.0-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt && pip cache purge

# Streamlit
#CMD ["streamlit", "run", "streamlit_demo.py"]
#EXPOSE 8501

# Flask
CMD ["python", "flask_demo.py"]
EXPOSE 5601
