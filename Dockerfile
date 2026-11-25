FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/opt/fastembed \
    LLM_PROVIDER=groq \
    LLM_MODEL=openai/gpt-oss-120b \
    CHUNKING_STRATEGY=recursive CHUNK_SIZE=800 CHUNK_OVERLAP=100 \
    EMBEDDING_MODEL=fastembed:BAAI/bge-small-en-v1.5 \
    TOP_K=4 RETRIEVAL_MODE=hybrid ROUTER_LLM_PICKER=true \
    LANGSMITH_TRACING=true LANGSMITH_PROJECT=rag-chatbot-spytro \
    PORT=7860

RUN pip install --no-cache-dir uv
WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .

RUN uv run --no-sync python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

RUN useradd -m -u 1000 user && chown -R user /app /opt/fastembed
USER user

EXPOSE 7860
CMD ["sh", "-c", "uv run --no-sync python -m app.ingest && exec uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
