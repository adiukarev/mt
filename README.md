## Инициализация

Обязателен `Python 3.13+`

```bash
python3.13 --version
```

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

```bash
mt --help
```

## Команды

### Аудит данных

```bash
mt audit --manifest manifests/audit_category.yaml
mt audit --manifest manifests/audit_sku.yaml
```

Результат:

```bash
artifacts/audit_category
artifacts/audit_sku
```

### Запуск эксперимента

```bash
mt run-experiment --manifest manifests/experiment_smoke.yaml
mt run-experiment --manifest manifests/experiment_category_reduced.yaml
mt run-experiment --manifest manifests/experiment_category.yaml
mt run-experiment --manifest manifests/experiment_sku_reduced.yaml
mt run-experiment --manifest manifests/experiment_sku.yaml
```

Результат:

```bash
artifacts/experiment_smoke
artifacts/experiment_category_reduced
artifacts/experiment_category
artifacts/experiment_sku_reduced
artifacts/experiment_sku
```

### Генерация synthetic dataset

```bash
mt generate-synthetic --manifest manifests/synthetic.yaml
```

Результат:

```bash
artifacts/synthetic
```

### Прогноз на synthetic dataset

```bash
mt predict --manifest manifests/predict_synthetic.yaml
```

Результат:

```bash
artifacts/predict_synthetic
```
