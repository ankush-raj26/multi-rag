# Hugging Face Space Dockerfile: clones the PUBLIC GitHub repo and builds it.
# Paste this as "Dockerfile" in a Docker Space. Space secrets needed (Settings -> Variables and secrets):
#   GROQ_API_KEY, GOOGLE_API_KEY, LANGSMITH_API_KEY
# To deploy new commits: Space Settings -> "Factory rebuild".
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy FASTEMBED_CACHE_PATH=/opt/fastembed \
    LLM_PROVIDER=groq LLM_MODEL=openai/gpt-oss-120b \
    CHUNKING_STRATEGY=recursive CHUNK_SIZE=800 CHUNK_OVERLAP=100 \
    EMBEDDING_MODEL=fastembed:BAAI/bge-small-en-v1.5 TOP_K=4 RETRIEVAL_MODE=hybrid ROUTER_LLM_PICKER=true \
    LANGSMITH_TRACING=true LANGSMITH_PROJECT=rag-chatbot-spytro PORT=7860
RUN pip install --no-cache-dir uv
WORKDIR /app
RUN git clone --depth 1 https://github.com/spytro26/rag-chatbot.git . && rm -rf .git
RUN uv sync --frozen --no-dev --no-install-project
RUN uv run --no-sync python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
RUN useradd -m -u 1000 user && chown -R user /app /opt/fastembed
USER user
EXPOSE 7860
CMD ["sh", "-c", "uv run --no-sync python -m app.ingest && exec uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips='*'"]
