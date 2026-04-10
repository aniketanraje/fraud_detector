# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Aniket Bhosale <aniketbhosale2808@gmail.com>"
LABEL description="Credit Card Fraud Detection — Production ML System"

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/     ./src/
COPY app/     ./app/
COPY configs/ ./configs/
COPY models/ ./models/
COPY requirements.txt .

# Create runtime directories
RUN mkdir -p /app/data/raw /app/data/processed /app/logs /app/models

# Non-root user
RUN addgroup --system fraud && adduser --system --ingroup fraud fraud
RUN chown -R fraud:fraud /app
USER fraud

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Port
EXPOSE 8501

# Healthcheck (safe)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app/main_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]