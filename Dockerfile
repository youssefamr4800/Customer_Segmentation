# syntax=docker/dockerfile:1

# ---------------------------------------------------------------
# Stage 1 - train: run the notebook headlessly to produce
#   kmeans_model.pkl and scaler.pkl.
# ---------------------------------------------------------------
FROM python:3.12-slim AS train

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /build

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install nbconvert ipykernel nbformat

COPY customer_segmentation.csv Analysis_Model.ipynb ./

# Execute every cell; the last cells call joblib.dump(...) into /build.
# The build fails loudly if either model file is missing afterwards.
RUN jupyter nbconvert --to notebook --execute Analysis_Model.ipynb \
        --output /tmp/executed.ipynb \
        --ExecutePreprocessor.timeout=900 \
    && test -f kmeans_model.pkl \
    && test -f scaler.pkl

# ---------------------------------------------------------------
# Stage 2 - runtime: slim image with only what the app needs.
# ---------------------------------------------------------------
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser
COPY --from=train --chown=appuser:appuser /build/kmeans_model.pkl /build/scaler.pkl ./
COPY --chown=appuser:appuser segmentation.py customer_segmentation.csv ./
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "segmentation.py"]