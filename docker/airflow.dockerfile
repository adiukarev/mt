FROM apache/airflow:3.1.8-python3.13

USER root

WORKDIR /opt/mt

RUN apt-get update \
	&& apt-get install -y --no-install-recommends libgomp1 \
	&& rm -rf /var/lib/apt/lists/* \
	&& mkdir -p /opt/airflow/logs /opt/mt/artifacts /opt/mt/mlruns

COPY pyproject.toml README.md ./
COPY src ./src
COPY dags ./dags
COPY manifests ./manifests

RUN chown -R airflow:0 /opt/airflow /opt/mt

USER airflow

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -e .
