FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gcc libpq-dev build-essential

ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.local/bin/:$PATH"

COPY . /app

WORKDIR /app
RUN uv sync --locked

ENV PYTHONUNBUFFERED 1

RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu && \
  uv pip install sentence-transformers && \
  uv pip install --system --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5255

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5255"]
