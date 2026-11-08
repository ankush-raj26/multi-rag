"""Check that every external piece of the RAG chatbot really works, and fix .env if it can.

Run from the project root:

    uv run python scripts/check_setup.py          # check only, changes nothing
    uv run python scripts/check_setup.py --fix    # also repair .env (keys in wrong slots, dead model names)

What it checks, in order:
  1. Key formats    Groq keys start with "gsk_", Google keys with "AIza" or "AQ." (new auth keys).
                    If the two are in each other's slots, --fix swaps them.
  2. Groq           lists the models YOUR key can use, picks a current one, makes a real call
                    and a tool call (the chatbot's calculator needs tool calling).
  3. Gemini         same for Google.
  4. LangSmith      authenticates the key (tries the US endpoint, then EU).
  5. The app        indexes data/documents, then runs real chat turns through the full graph:
                    "hi" (should skip RAG) and a leave-policy question (should retrieve).

It never prints your keys, only their first few characters.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning)  # noisy third-party docstrings

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
sys.path.insert(0, str(ROOT))

# Newest first. The checker uses the first one your key can actually see.
GROQ_PREFERRED = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "moonshotai/kimi-k2-instruct"]
GOOGLE_PREFERRED = [
    "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash",
]

OK, BAD, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []
CALL_TIMEOUT = 30   # seconds for one HTTP call to a provider
STEP_LIMIT = 90     # hard ceiling for a whole provider check, so nothing can hang the script


def step(msg: str) -> None:
    """Show what is happening right now (so a slow call never looks like a freeze)."""
    print(f"    … {msg}", flush=True)


def with_timeout(fn, seconds: int):  # type: ignore[no-untyped-def]
    """Run fn() in a daemon thread; raise TimeoutError if it takes longer than `seconds`.

    Provider SDKs retry silently on their own, which is how a check can sit for
    minutes printing nothing. A daemon thread is abandoned on timeout instead of
    being waited for, so the script always moves on.
    """
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"no response after {seconds}s (the call hung)")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def report(name: str, status: str, detail: str) -> None:
    results.append((name, status, detail))
    print(f"  [{status}] {name}: {detail}")


def mask(key: str) -> str:
    return f"{key[:6]}… ({len(key)} chars)" if key else "(empty)"


def kind_of(key: str) -> str:
    """Guess which service a key belongs to from its prefix."""
    if key.startswith("gsk_"):
        return "groq"
    if key.startswith(("AIza", "AQ.")):
        return "google"
    if key.startswith("lsv2_"):
        return "langsmith"
    return "unknown"


def short_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return re.sub(r"\s+", " ", text)[:220]


def classify(exc: Exception) -> str:
    """Turn an exception into advice: bad key, wrong model, no network, or quota."""
    t = f"{type(exc).__name__} {exc}".lower()
    # network problems first: a blocked proxy also says "403", which is NOT a bad key
    if "hung" in t:
        return "timed out: the service did not answer (network, VPN, or a key the service silently rejects)"
    if any(s in t for s in ("proxyerror", "connecterror", "connectionerror", "apiconnectionerror", "connection refused",
                            "timed out", "timeout", "name resolution", "nodename", "ssl", "network is unreachable")):
        return "could not reach the service (network / VPN / firewall / proxy)"
    auth = ("401", "unauthenticated", "invalid api key", "invalid_api_key", "api key not valid",
            "permission_denied", "403")
    if any(s in t for s in auth):
        return "the key was rejected"
    if any(s in t for s in ("404", "not found", "model_not_found", "decommissioned", "does not exist")):
        return "that model name no longer exists"
    if any(s in t for s in ("429", "quota", "rate limit", "resource_exhausted")):
        return "the key works but is out of quota / rate-limited"
    return "unexpected error"


# --------------------------------------------------------------------------- .env helpers


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*)$", line)
            if m:
                values[m.group(1)] = m.group(2).strip()
    return values


def write_env(updates: dict[str, str]) -> None:
    """Change only the given keys in .env, keeping every other line and comment."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    seen: set[str] = set()
    out = []
    for line in lines:
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=", line)
        if m and m.group(1) in updates:
            out.append(f"{m.group(1)}={updates[m.group(1)]}")
            seen.add(m.group(1))
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in updates.items() if k not in seen]
    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- provider checks


def check_groq(key: str) -> str | None:
    """Return a working Groq model id, or None."""
    if not key:
        report("Groq", WARN, "GROQ_API_KEY is empty, skipped")
        return None
    import httpx

    try:
        step("Groq: listing the models your key can use")
        r = httpx.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
        r.raise_for_status()
        available = {m["id"] for m in r.json().get("data", [])}
    except Exception as exc:  # noqa: BLE001 - we report every failure the same way
        report("Groq", BAD, f"{classify(exc)} ({short_error(exc)}) key={mask(key)}")
        return None
    model = next((m for m in GROQ_PREFERRED if m in available), None)
    if model is None:
        report("Groq", BAD, f"key works but none of {GROQ_PREFERRED} is available; available: {sorted(available)[:8]}")
        return None
    return _chat_check("Groq", model, lambda: _groq_llm(key, model))


def _groq_llm(key: str, model: str):  # type: ignore[no-untyped-def]
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=0, api_key=key, timeout=CALL_TIMEOUT, max_retries=1)  # type: ignore[arg-type]


def check_google(key: str) -> str | None:
    """Return a working Gemini model id, or None."""
    if not key:
        report("Gemini", WARN, "GOOGLE_API_KEY is empty, skipped")
        return None
    available: set[str] = set()
    try:
        from google import genai
        from google.genai import types

        step("Gemini: listing the models your key can use")
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=CALL_TIMEOUT * 1000))

        def list_models() -> set[str]:
            names: set[str] = set()
            for m in client.models.list(config={"page_size": 100}):
                name = (m.name or "").removeprefix("models/")
                actions = getattr(m, "supported_actions", None) or []
                if not actions or "generateContent" in actions:
                    names.add(name)
            return names

        available = with_timeout(list_models, CALL_TIMEOUT + 15)
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if key.startswith("AQ.") and classify(exc).startswith("the key was rejected"):
            hint = (" | 'AQ.' keys are Google's new auth keys; if this keeps failing, create the key in a"
                    " project where the Gemini API is enabled, or restrict it to the Generative Language API")
        report("Gemini", BAD, f"{classify(exc)} ({short_error(exc)}) key={mask(key)}{hint}")
        return None
    model = next((m for m in GOOGLE_PREFERRED if m in available), None)
    if model is None:
        flash = sorted(m for m in available if "flash" in m)[:6]
        report("Gemini", BAD, f"key works but none of the preferred models is listed; flash models: {flash}")
        return None
    return _chat_check("Gemini", model, lambda: _google_llm(key, model))


def _google_llm(key: str, model: str):  # type: ignore[no-untyped-def]
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(  # type: ignore[call-arg]
        model=model, temperature=0, google_api_key=key, timeout=CALL_TIMEOUT, max_retries=1
    )


def _chat_check(name: str, model: str, build) -> str | None:  # type: ignore[no-untyped-def]
    """One plain call and one tool call against `model`."""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression."""
        return "0"

    try:
        llm = build()
        step(f"{name}: asking {model} to reply 'OK'")
        text = str(with_timeout(lambda: llm.invoke("Reply with exactly the word OK.").content, STEP_LIMIT)).strip()
        step(f"{name}: testing tool calling (the chatbot's calculator needs it)")
        call = with_timeout(
            lambda: llm.bind_tools([calculator]).invoke("Use the calculator tool to compute 6*7."), STEP_LIMIT
        )
        tools_ok = bool(getattr(call, "tool_calls", None))
    except Exception as exc:  # noqa: BLE001
        report(name, BAD, f"{model}: {classify(exc)} ({short_error(exc)})")
        return None
    report(name, OK if tools_ok else WARN,
           f"{model} answered {text[:20]!r}; tool calling {'works' if tools_ok else 'did NOT trigger'}")
    return model


def check_langsmith(key: str, endpoint: str) -> str | None:
    """Return the endpoint the key authenticates against, or None."""
    if not key:
        report("LangSmith", WARN, "LANGSMITH_API_KEY is empty, tracing stays off")
        return None
    from langsmith import Client

    endpoints = [endpoint] if endpoint else ["https://api.smith.langchain.com", "https://eu.api.smith.langchain.com"]
    last = None
    for ep in endpoints:
        try:
            step(f"LangSmith: checking the key at {ep}")
            client = Client(api_key=key, api_url=ep, timeout_ms=CALL_TIMEOUT * 1000)
            with_timeout(lambda c=client: next(iter(c.list_projects(limit=1)), None), CALL_TIMEOUT + 15)
            report("LangSmith", OK, f"key accepted at {ep}")
            return ep
        except Exception as exc:  # noqa: BLE001
            last = exc
    why = f"{classify(last)} ({short_error(last)})" if last else "error"
    report("LangSmith", BAD, f"{why} key={mask(key)}")
    return None


# --------------------------------------------------------------------------- full app check


def check_app() -> None:
    """Index the documents, then run real chat turns through the whole LangGraph."""
    from app.config import get_settings

    get_settings.cache_clear()
    from app.deps import build_components
    from app.ingest.pipeline import ingest_directory

    try:
        step("loading the embedding model (the first run downloads ~130 MB, this can take a minute)")
        comp = build_components()
    except Exception as exc:  # noqa: BLE001
        report("App", BAD, f"could not build components ({short_error(exc)})")
        return
    if comp.missing:
        report("App", BAD, f"modules failed to load: {comp.missing}")
        return
    assert comp.chunker and comp.embedder and comp.store
    step("indexing data/documents")
    n = ingest_directory(get_settings().documents_dir, chunker=comp.chunker, embedder=comp.embedder, store=comp.store)
    report("Index", OK if n else BAD, f"{n} chunks from data/documents ({type(comp.embedder).__name__})")

    for q, expect_rag in (("hi", False), ("How many annual leave days do I get?", True)):
        try:
            step(f"real chat turn through the whole graph: {q!r}")
            r = with_timeout(lambda q=q: comp.engine.invoke(message=q, history=[], session_id="setup-check"), 180)
        except Exception as exc:  # noqa: BLE001
            report(f"Chat {q!r}", BAD, f"{classify(exc)} ({short_error(exc)})")
            continue
        route = r.route.route if r.route else "?"
        layer = r.route.layer if r.route else "?"
        rag_ok = (route != "none") == expect_rag
        status = OK if r.answer and rag_ok else WARN
        report(f"Chat {q!r}", status, f"route={route} via {layer}, {len(r.sources)} sources, answer={r.answer[:70]!r}")


# --------------------------------------------------------------------------- main


def fix_key_slots(groq_key: str, google_key: str, updates: dict[str, str]) -> tuple[str, str]:
    """Swap or move keys that sit in the wrong variable. Returns the corrected pair."""
    print("\n1. Key formats")
    print(f"  GROQ_API_KEY   = {mask(groq_key)} looks like: {kind_of(groq_key)}")
    print(f"  GOOGLE_API_KEY = {mask(google_key)} looks like: {kind_of(google_key)}")
    if groq_key and groq_key == google_key:
        report("Keys", WARN, f"both variables hold the SAME key (a {kind_of(groq_key)} key), so swapping cannot help")
    if kind_of(groq_key) == "google" and kind_of(google_key) == "groq":
        report("Keys", WARN, "the two keys are in each other's slots")
        groq_key, google_key = google_key, groq_key
        updates.update(GROQ_API_KEY=groq_key, GOOGLE_API_KEY=google_key)
    elif kind_of(groq_key) == "google" and not google_key:
        report("Keys", WARN, "a Google key is sitting in GROQ_API_KEY; moving it to GOOGLE_API_KEY")
        google_key = groq_key
        updates.update(GOOGLE_API_KEY=google_key)
    return groq_key, google_key


def check_providers(groq_key: str, google_key: str) -> dict[str, str | None]:
    print("\n2. Groq")
    groq_model = None
    if kind_of(groq_key) in ("groq", "unknown"):
        groq_model = check_groq(groq_key)
    else:
        report("Groq", BAD, f"GROQ_API_KEY is a {kind_of(groq_key)} key, not a Groq key (Groq keys start with gsk_)."
               " Create one at console.groq.com/keys")
    print("\n3. Gemini")
    return {"groq": groq_model, "google": check_google(google_key)}


def choose_provider(configured: str, working: dict[str, str | None]) -> str:
    """Keep the configured provider if it works, otherwise use the one that does."""
    provider = "google" if configured == "gemini" else configured
    if not working.get(provider):
        provider = next((p for p, m in working.items() if m), provider)
    return provider


def apply_updates(env: dict[str, str], updates: dict[str, str], fix: bool) -> None:
    changed = {k: v for k, v in updates.items() if env.get(k) != v}
    if not changed:
        return
    shown = {k: (mask(v) if k.endswith("_KEY") else v) for k, v in changed.items()}
    if fix:
        write_env(changed)
        print(f"\n  .env updated: {shown}")
    else:
        print(f"\n  .env would change: {shown}  (run again with --fix to apply)")


def run_app_check(env: dict[str, str], provider: str, model: str, key: str, ls_endpoint: str | None) -> None:
    print("\n5. The app (real chat turns)")
    # make this process use the chosen provider even without --fix
    os.environ.update(LLM_PROVIDER=provider, LLM_MODEL=model)
    os.environ["GROQ_API_KEY" if provider == "groq" else "GOOGLE_API_KEY"] = key
    if ls_endpoint:
        os.environ.update(LANGSMITH_TRACING="true", LANGSMITH_API_KEY=env["LANGSMITH_API_KEY"],
                          LANGSMITH_ENDPOINT=ls_endpoint)
    from app.observability import configure_tracing

    configure_tracing()
    check_app()


def main() -> int:
    fix = "--fix" in sys.argv
    env = read_env()
    updates: dict[str, str] = {}
    groq_key, google_key = fix_key_slots(env.get("GROQ_API_KEY", ""), env.get("GOOGLE_API_KEY", ""), updates)
    working = check_providers(groq_key, google_key)
    print("\n4. LangSmith")
    ls_endpoint = check_langsmith(env.get("LANGSMITH_API_KEY", ""), env.get("LANGSMITH_ENDPOINT", ""))

    provider = choose_provider(env.get("LLM_PROVIDER", "groq").lower(), working)
    model = working.get(provider)
    if model:
        updates.update(LLM_PROVIDER=provider, LLM_MODEL=model)
        print(f"\n  -> chatbot will use {provider} / {model}")
    else:
        print("\n  -> no working LLM provider: the chatbot cannot answer until one key works")
    if ls_endpoint:
        updates["LANGSMITH_TRACING"] = "true"
        if "eu." in ls_endpoint:
            updates["LANGSMITH_ENDPOINT"] = ls_endpoint
    apply_updates(env, updates, fix)

    if model:
        run_app_check(env, provider, model, groq_key if provider == "groq" else google_key, ls_endpoint)
    else:
        print("\n5. The app (real chat turns)")
        report("App", WARN, "skipped, no working LLM")

    fails = [r for r in results if r[1] == BAD]
    passes, warns = sum(r[1] == OK for r in results), sum(r[1] == WARN for r in results)
    print(f"\nSummary: {passes} pass, {warns} warn, {len(fails)} fail")
    if ls_endpoint:
        print(f"Traces: https://smith.langchain.com  -> project {env.get('LANGSMITH_PROJECT', 'rag-chatbot')}")
    # Only block start-up when the chatbot itself cannot work. A failed backup provider
    # (or LangSmith) is reported above but does not stop the app.
    blocking = [r for r in fails if r[0] in ("App", "Index") or r[0].startswith("Chat")]
    if not model:
        print("Blocking: no LLM provider works, so the chatbot cannot answer.")
        return 1
    if blocking:
        print("Blocking: the chatbot failed its real chat turns (see above).")
        return 1
    if fails:
        print(f"Not blocking: the chatbot works with {provider}; the failures above are optional extras.")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # exit now: don't wait on abandoned (hung) SDK threads or trace uploads
