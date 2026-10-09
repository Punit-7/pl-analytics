# The API image: Python, the pinned packages, the model code and the model files.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# Packages first. This layer is rebuilt only when requirements.txt changes.
COPY serving/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# The code, then the model files, which change every week.
COPY modelling/__init__.py modelling/
COPY modelling/match/__init__.py modelling/match/dixon_coles.py modelling/match/
COPY modelling/xg/__init__.py modelling/xg/features.py modelling/xg/
COPY serving/__init__.py serving/model_io.py serving/
COPY serving/api serving/api
COPY serving/models serving/models
COPY serving/predictions/latest.json serving/predictions/

# Do not run the server as the root user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn serving.api.main:app --host 0.0.0.0 --port ${PORT}"]
