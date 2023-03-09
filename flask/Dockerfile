FROM python:3.11.0-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt && pip cache purge

# Flask
CMD ["sh", "launch_app.sh"]
EXPOSE 5601
