"""MILESTONE 1 — The LLM factory.

`get_llm()` returns a LangChain chat model chosen at runtime by the values in
`.env`.  Nothing here is provider-specific until the function runs, so adding a
new provider is just another branch + one `uv add`.

Supported providers (set LLM_PROVIDER in .env):
  groq      -> ChatGroq               (needs GROQ_API_KEY)      -- fast, free tier
  google    -> ChatGoogleGenerativeAI (needs GOOGLE_API_KEY)    -- Gemini
  anthropic -> ChatAnthropic          (needs ANTHROPIC_API_KEY)
  openai    -> ChatOpenAI             (needs OPENAI_API_KEY)

Keys are NEVER hard-coded: each provider package reads its key from the
environment, which `app.config` has already loaded from `.env`.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.config import get_settings

_DEFAULT_MODEL = {
    "groq": "openai/gpt-oss-120b",
    "google": "gemini-3.8-flash",
    "gemini": "gemini-3.8-flash",
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}


def get_llm() -> BaseChatModel:
    """Build the chat model described by `.env` and return it ready to use.

    OUT: a BaseChatModel that supports `.invoke(messages)` and `.bind_tools(tools)`.
    """
    settings = get_settings()
    provider = settings.llm_provider.strip().lower()
    model = settings.llm_model.strip() or _DEFAULT_MODEL.get(provider, "")

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=0, timeout=60, max_retries=2)

    if provider in ("google", "gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=0, timeout=60, max_retries=2)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0)  # type: ignore[call-arg]

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use groq, google, anthropic or openai)")
