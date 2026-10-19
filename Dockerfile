FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir poetry==2.4.1
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction
COPY models.py runner.py app.py provider.py benchmark_models.py dataset.py benchmark.py metrics.py questions.py ./
COPY pages ./pages
COPY examples ./examples
COPY configs ./configs
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
