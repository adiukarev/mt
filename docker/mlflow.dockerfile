FROM ghcr.io/mlflow/mlflow

USER root

WORKDIR /opt/mt

RUN python -m pip install --no-cache-dir psycopg2-binary \
	&& mkdir -p /opt/mt/mlruns
