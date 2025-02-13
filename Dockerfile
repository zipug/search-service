FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN python -m venv /opt/venv && \
  /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .

FROM python:3.12-slim AS production
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /app /app

EXPOSE 5255

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5255"]
