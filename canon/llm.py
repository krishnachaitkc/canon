import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import json_repair
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from .config import BASE_URLS, KEY_ENVS, ROUTES, THINK_STAGES, parse_route

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "llm"
FAKE_DIR = PROJECT_ROOT / "samples" / "recorded"
JSON_RULE = '\n\nReturn strict, valid JSON. Escape any double quote inside a string value as \\".'
BACKOFFS = (10, 25, 40, 55, 70)
MAX_INVALID = 3
MAX_RATE_LIMIT = 8
READ_TIMEOUT = 300

RUN_DIR: Path | None = None
_LOG = logging.getLogger(__name__)
_PROGRESS = None

load_dotenv(PROJECT_ROOT / ".env", override=False)

_STATS: dict[str, dict] = {}
_SWITCHES: list[str] = []
_LOCK = threading.Lock()
_PACE = {
    "deepseek": [0.0, 0.0],
    "openai": [0.0, 0.0],
    "anthropic": [0.0, 0.0],
    "gemini": [1.0, 0.0],
    "fake": [0.0, 0.0],
}


class LLMFailed(RuntimeError):
    """A model call Canon cannot continue with.

    `kind` lets the app explain the failure in plain words; `detail` keeps the provider's
    sanitized reply for the technical-details panel.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "error",
        hint: str = "",
        detail: str = "",
        status: int | None = None,
        route: str | None = None,
    ):
        super().__init__(message)
        self.kind = kind
        self.hint = hint
        self.detail = detail
        self.status = status
        self.route = route


class QuotaExhausted(RuntimeError):
    def __init__(
        self, message: str, *, reasons: list[str] | None = None, hint: str = ""
    ):
        super().__init__(message)
        self.reasons = list(reasons or [])
        self.hint = hint


class _NextModel(Exception):
    pass


def set_run_dir(path: str | Path | None) -> None:
    global RUN_DIR
    RUN_DIR = Path(path) if path else None
    if RUN_DIR is not None:
        raw = RUN_DIR / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        (raw / "llm_responses.jsonl").write_text("", encoding="utf-8")


def set_progress(callback) -> None:
    """Register a callback(stage, done, total) that runs after each model answer lands.

    The app passes one so a long run shows its place instead of a silent spinner. The CLI
    leaves it unset. It is never called from inside a lock, so a slow callback cannot
    stall a worker.
    """
    global _PROGRESS
    _PROGRESS = callback


def _report(stage: str, done: int, total: int) -> None:
    callback = _PROGRESS
    if callback is None:
        return
    try:
        callback(stage, done, total)
    except Exception:
        _LOG.debug("progress callback failed", exc_info=True)


def report_phase(name: str) -> None:
    """Tell the registered callback that a code-only phase started (no call count)."""
    _report(name, 0, 0)


def reset_stats() -> None:
    with _LOCK:
        _STATS.clear()
        _SWITCHES.clear()


def get_stats() -> dict[str, dict]:
    with _LOCK:
        return json.loads(json.dumps(_STATS))


def get_switches() -> list[str]:
    with _LOCK:
        return list(_SWITCHES)


def provider_gap(provider: str) -> float:
    with _LOCK:
        return _PACE.get(provider, [0.0, 0.0])[0]


def _bump(stage: str, **fields) -> None:
    with _LOCK:
        stats = _STATS.setdefault(
            stage,
            {
                "calls": 0,
                "cached": 0,
                "input_tokens": 0,
                "cache_hit_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "retries": 0,
                "rate_limited": 0,
                "seconds": 0.0,
                "calls_by_model": {},
                "tokens_by_model": {},
            },
        )
        for key, value in fields.items():
            if key == "model":
                models = stats["calls_by_model"]
                models[value] = models.get(value, 0) + 1
            elif isinstance(stats.get(key), float):
                stats[key] = round(stats[key] + value, 3)
            else:
                stats[key] = stats.get(key, 0) + value


def _bump_tokens(stage: str, route: str, tokens: dict) -> None:
    with _LOCK:
        stats = _STATS.setdefault(
            stage,
            {
                "calls": 0,
                "cached": 0,
                "input_tokens": 0,
                "cache_hit_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "retries": 0,
                "rate_limited": 0,
                "seconds": 0.0,
                "calls_by_model": {},
                "tokens_by_model": {},
            },
        )
        entry = stats["tokens_by_model"].setdefault(
            route, {"input": 0, "output": 0, "cache_hit": 0, "reasoning": 0}
        )
        for name in ("input", "output", "cache_hit", "reasoning"):
            entry[name] += int(tokens.get(name, 0))


def _log_switch(stage: str, route: str, reason: str) -> None:
    message = f"model switch: {stage} {route}: {reason}"
    with _LOCK:
        _SWITCHES.append(message)
    _LOG.warning("[model switch in %s] %s", stage, reason)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _pace(provider: str) -> None:
    with _LOCK:
        gap, last = _PACE.get(provider, [0.0, 0.0])
        start_at = max(time.time(), last + gap)
        _PACE[provider] = [gap, start_at]
    wait = start_at - time.time()
    if wait > 0:
        _sleep(wait)


def _raise_gap(provider: str) -> None:
    with _LOCK:
        gap, last = _PACE.get(provider, [0.0, 0.0])
        _PACE[provider] = [max(gap, 4.5), last]


def _sanitize(text: str | None, key: str | None) -> str:
    value = text or ""
    if key:
        value = value.replace(key, "***")
    return value[:500]


PROVIDER_NAMES = {
    "deepseek": "DeepSeek",
    "openai": "OpenAI",
    "gemini": "Gemini",
    "anthropic": "Anthropic",
    "fake": "the recorded sample",
}

KEY_HINTS = {
    "deepseek": "DeepSeek keys normally start with sk-. Copy one from platform.deepseek.com.",
    "openai": "OpenAI keys normally start with sk-. Copy one from platform.openai.com.",
    "gemini": "Google AI Studio keys normally start with AIza.",
    "anthropic": "Copy a key from console.anthropic.com.",
}

KEY_SETTINGS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _provider_name(provider: str) -> str:
    return PROVIDER_NAMES.get(provider, provider)


def _one_line(text: str | None, limit: int = 500) -> str:
    value = " ".join((text or "").split())
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "\u2026"


def _http_failed(route: str, provider: str, status: int, message: str) -> LLMFailed:
    """Turn an HTTP failure into a short cause, a kind and a useful next step."""
    name = _provider_name(provider)
    lowered = message.lower()
    detail = _one_line(message)
    if status in (401, 403) or (
        status in (400, 422)
        and ("api key" in lowered or "authentication" in lowered or "unauthorized" in lowered)
    ):
        return LLMFailed(
            f"{name} rejected the API key (HTTP {status}).",
            kind="auth",
            status=status,
            route=route,
            detail=detail,
            hint=(
                f"Provide a valid {name} key in the key field, or set "
                f"{KEY_SETTINGS.get(provider, 'the server key')}. "
                f"{KEY_HINTS.get(provider, '')}"
            ).strip(),
        )
    if status == 402:
        return LLMFailed(
            f"The {name} account has no credit left (HTTP 402).",
            kind="balance",
            status=status,
            route=route,
            detail=detail,
            hint="Add credit with the provider, switch to Gemini free, or run Code only.",
        )
    model_missing = status == 404 or (
        status in (400, 422)
        and "model" in lowered
        and any(word in lowered for word in ("not found", "does not exist", "unknown", "invalid"))
    )
    if model_missing:
        model = route.split("/", 1)[-1]
        return LLMFailed(
            f'{name} does not offer the model "{model}" (HTTP {status}).',
            kind="model",
            status=status,
            route=route,
            detail=detail,
            hint=(
                "Check the model name and, for Custom, the base URL. Or choose Gemini "
                "free or DeepSeek."
            ),
        )
    if status in (400, 422):
        return LLMFailed(
            f"{name} rejected the request (HTTP {status}).",
            kind="request",
            status=status,
            route=route,
            detail=detail,
            hint=(
                "If you are using Custom, check the base URL and model name. Otherwise "
                "try another model or run Code only."
            ),
        )
    return LLMFailed(
        f"{name} refused the request (HTTP {status}).",
        kind="http",
        status=status,
        route=route,
        detail=detail,
        hint="Try again, switch to another model, or run Code only.",
    )


def get_key(provider: str, key_env: str | None = None) -> str:
    name = _provider_name(provider)
    if provider == "gemini":
        env_names = [key_env] if key_env else ["GEMINI_API_KEY", "LLM_API_KEY"]
        value = next((os.getenv(env_name) for env_name in env_names if os.getenv(env_name)), "")
        if not value:
            raise LLMFailed(
                "No API key was provided for Gemini.",
                kind="config",
                detail="GEMINI_API_KEY and LLM_API_KEY are both unset.",
                hint=(
                    "Provide a Gemini key in the key field, set GEMINI_API_KEY on the "
                    "server, or choose Code only."
                ),
            )
        return value
    env_name = key_env or KEY_ENVS.get(provider)
    if not env_name:
        raise LLMFailed(
            f"Canon does not know how to authenticate with {name}.",
            kind="config",
            detail=f"No key environment variable is registered for provider {provider}.",
            hint="Choose Gemini free, DeepSeek, or Custom.",
        )
    value = os.getenv(env_name)
    if not value:
        raise LLMFailed(
            f"No API key was provided for {name}.",
            kind="config",
            detail=f"{env_name} is not set.",
            hint=(
                f"Provide your {name} key in the key field, set {env_name} on the server, "
                "or choose Code only."
            ),
        )
    return value


def _cache_key(route: str, think: bool, prompt: str) -> str:
    flag = "thinking" if think else "plain"
    return hashlib.sha256(f"{route}\n{flag}\n{prompt}".encode()).hexdigest()


def _strip_fences(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[A-Za-z]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_jsonl(record: dict) -> None:
    if RUN_DIR is None:
        return
    raw_dir = Path(RUN_DIR) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with open(raw_dir / "llm_responses.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


def _build_request(
    provider: str,
    name: str,
    text: str,
    think: bool,
    max_tokens: int,
    key: str,
    base_url: str | None,
    extra: dict,
    drop_format: bool,
):
    extra = dict(extra or {})
    if provider == "deepseek":
        base = (base_url or BASE_URLS["deepseek"]).rstrip("/")
        url = base + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        body = {
            "model": name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled" if think else "disabled"},
        }
        if not drop_format:
            body["response_format"] = {"type": "json_object"}
    elif provider == "openai":
        base = (base_url or BASE_URLS["openai"]).rstrip("/")
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        body = {
            "model": name,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens,
        }
        if name.startswith("gpt-6"):
            body["reasoning_effort"] = "medium" if think else "none"
        if not drop_format:
            body["response_format"] = {"type": "json_object"}
    elif provider == "gemini":
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{name}:generateContent"
        )
        headers = {"Content-Type": "application/json", "x-goog-api-key": key}
        body = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.2,
            },
        }
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
        }
    else:
        raise LLMFailed(f"Unknown provider {provider}")
    body.update(extra)
    return url, headers, body


def _parse_response(provider: str, response) -> tuple[str, str, dict]:
    data = response.json()
    if provider in ("deepseek", "openai"):
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason") or ""
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        tokens = {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "cache_hit": usage.get("prompt_cache_hit_tokens", 0),
            "reasoning": details.get("reasoning_tokens", 0),
        }
    elif provider == "gemini":
        candidate = data["candidates"][0]
        content = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []))
        finish = "length" if candidate.get("finishReason") == "MAX_TOKENS" else candidate.get("finishReason", "")
        usage = data.get("usageMetadata") or {}
        tokens = {
            "input": usage.get("promptTokenCount", 0),
            "output": usage.get("candidatesTokenCount", 0),
            "cache_hit": usage.get("cachedContentTokenCount", 0),
            "reasoning": 0,
        }
    elif provider == "anthropic":
        content = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        finish = "length" if data.get("stop_reason") == "max_tokens" else data.get("stop_reason", "")
        usage = data.get("usage") or {}
        tokens = {
            "input": usage.get("input_tokens", 0),
            "output": usage.get("output_tokens", 0),
            "cache_hit": 0,
            "reasoning": 0,
        }
    else:
        raise LLMFailed(f"Unknown provider {provider}")
    if finish == "MAX_TOKENS":
        finish = "length"
    return content, finish, tokens


def _rate_info(response) -> tuple[bool, float]:
    delay = None
    daily = False
    body_text = ""
    payload = {}
    try:
        body_text = response.text or ""
    except Exception:
        pass
    try:
        payload = response.json() or {}
    except Exception:
        payload = {}

    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    for detail in error.get("details", []) or []:
        for violation in detail.get("violations", []) or []:
            quota_id = str(violation.get("quotaId", "")).lower()
            if "perday" in quota_id:
                daily = True
        retry_delay = detail.get("retryDelay")
        if retry_delay and delay is None:
            try:
                delay = float(str(retry_delay).rstrip("s"))
            except ValueError:
                pass

    message = (body_text + " " + json.dumps(error)).lower()
    if "perday" in message or "per day" in message or "daily" in message or "insufficient quota" in message:
        daily = True

    headers = getattr(response, "headers", {}) or {}
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if delay is None and retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            pass
    return daily, delay if delay is not None else 60.0


def _call_model(
    stage: str,
    route: str,
    text: str,
    schema,
    base_url: str | None,
    api_key_env: str | None,
    extra: dict,
    use_cache: bool,
):
    provider, name = parse_route(route)
    if provider == "fake":
        return _fake(stage, route, text, schema)

    think = stage in THINK_STAGES
    cache_path = CACHE_DIR / f"{_cache_key(route, think, text)}.json"
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cached = schema.model_validate(json.load(fh))
            _bump(stage, cached=1)
            return cached
        except Exception:
            pass

    key = get_key(provider, api_key_env)

    # A reasoning judge on a 12-spelling group can think past 20,000 tokens (seen on the pack),
    # so it starts at 32,000 and doubles once.
    max_tokens = 32000 if think else 8000
    current_text = text
    transport_attempt = 0
    invalid_attempt = 0
    rate_attempt = 0
    length_retried = False
    drop_format = False
    format_retry_used = False
    http_attempts = 0
    started = time.time()
    last_tokens = {"input": 0, "output": 0, "cache_hit": 0, "reasoning": 0}
    last_content = ""

    while True:
        url, headers, body = _build_request(
            provider,
            name,
            current_text,
            think,
            max_tokens,
            key,
            base_url,
            extra,
            drop_format,
        )
        _pace(provider)
        http_attempts += 1
        call_started = time.time()
        try:
            response = requests.post(url, headers=headers, json=body, timeout=READ_TIMEOUT)
        except requests.RequestException as exc:
            _bump(stage, retries=1)
            if transport_attempt >= len(BACKOFFS):
                reason = _one_line(f"{type(exc).__name__}: {_sanitize(str(exc), key)}", 160)
                raise _NextModel(
                    f"could not reach the provider after {transport_attempt + 1} tries ({reason})"
                )
            _sleep(BACKOFFS[transport_attempt])
            transport_attempt += 1
            continue
        _bump(stage, seconds=time.time() - call_started)
        status = response.status_code
        if status == 200:
            try:
                content, finish, tokens = _parse_response(provider, response)
            except Exception as exc:
                invalid_attempt += 1
                if invalid_attempt > MAX_INVALID:
                    raise LLMFailed(
                        f"{_provider_name(provider)} returned a response Canon could not read.",
                        kind="response",
                        route=route,
                        detail=f"{type(exc).__name__}: {_one_line(str(exc))}",
                        hint="Run again. If it repeats, switch models or run Code only.",
                    )
                current_text = text + f"\n\nYour previous reply could not be read ({type(exc).__name__}). Return strict JSON."
                _bump(stage, retries=1)
                continue

            last_tokens = tokens
            last_content = content
            _bump(
                stage,
                input_tokens=tokens["input"],
                output_tokens=tokens["output"],
                cache_hit_tokens=tokens["cache_hit"],
                reasoning_tokens=tokens["reasoning"],
            )
            _bump_tokens(stage, route, tokens)

            if finish == "length":
                if length_retried:
                    raise LLMFailed(
                        f"{_provider_name(provider)} ran out of room before it finished its answer.",
                        kind="response",
                        route=route,
                        detail=f"{route} hit the token limit twice (max_tokens={max_tokens}).",
                        hint="Run again, or switch to a model with a larger output limit.",
                    )
                max_tokens *= 2
                length_retried = True
                _bump(stage, retries=1)
                continue

            if not content or not content.strip():
                invalid_attempt += 1
                if invalid_attempt > MAX_INVALID:
                    raise LLMFailed(
                        f"{_provider_name(provider)} returned an empty answer.",
                        kind="response",
                        route=route,
                        detail=f"{route} returned blank content {invalid_attempt} times.",
                        hint="Run again. If it repeats, switch models or run Code only.",
                    )
                current_text = text + "\n\nYour previous reply was empty. Return strict JSON."
                _bump(stage, retries=1)
                continue

            raw = _strip_fences(content)
            parse_error = None
            try:
                parsed = json.loads(raw)
            except Exception as exc:
                parse_error = exc
                try:
                    parsed = json_repair.loads(raw)
                except Exception:
                    parsed = None

            if parsed is None:
                invalid_attempt += 1
                if invalid_attempt > MAX_INVALID:
                    raise LLMFailed(
                        f"{_provider_name(provider)} did not return usable JSON.",
                        kind="response",
                        route=route,
                        detail=(
                            f"{route} returned invalid JSON {invalid_attempt} times: "
                            f"{_one_line(str(parse_error), 240)}"
                        ),
                        hint="Run again. If it repeats, switch models or run Code only.",
                    )
                current_text = (
                    text
                    + f"\n\nYour previous reply was invalid JSON ({parse_error}). Return strict JSON."
                )
                _bump(stage, retries=1)
                continue

            if isinstance(parsed, list) and "results" in getattr(schema, "model_fields", {}):
                parsed = {"results": parsed}

            try:
                result = schema.model_validate(parsed)
            except ValidationError as exc:
                invalid_attempt += 1
                if invalid_attempt > MAX_INVALID:
                    raise LLMFailed(
                        f"{_provider_name(provider)}'s answer did not match Canon's expected shape.",
                        kind="response",
                        route=route,
                        detail=f"{route} reply failed validation: {_one_line(str(exc))}",
                        hint="Run again. If it repeats, switch models or run Code only.",
                    )
                current_text = (
                    text
                    + f"\n\nYour previous reply failed validation: {exc}. Fix it and return strict JSON."
                )
                _bump(stage, retries=1)
                continue

            if use_cache:
                _write_cache(cache_path, result.model_dump())
            _bump(stage, calls=1, model=route)
            _write_jsonl(
                {
                    "stage": stage,
                    "model": route,
                    "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "response": content,
                    "tokens": last_tokens,
                    "latency": round(time.time() - started, 3),
                    "attempts": http_attempts,
                    "cached": False,
                }
            )
            return result

        if status == 429:
            _bump(stage, rate_limited=1, retries=1)
            daily, delay = _rate_info(response)
            if daily:
                raise _NextModel(f"daily quota reached ({_sanitize(response.text, key)})")
            _raise_gap(provider)
            _sleep(delay)
            rate_attempt += 1
            if rate_attempt > MAX_RATE_LIMIT:
                raise _NextModel("rate limited too many times")
            continue

        if status == 402:
            raise _NextModel("insufficient balance")

        if status in (500, 502, 503, 504):
            _bump(stage, retries=1)
            if transport_attempt >= len(BACKOFFS):
                raise _NextModel(f"server error HTTP {status}")
            _sleep(BACKOFFS[transport_attempt])
            transport_attempt += 1
            continue

        message = _sanitize(response.text, key)
        if (
            provider == "openai"
            and status in (400, 422)
            and not format_retry_used
            and "response_format" in message.lower()
        ):
            drop_format = True
            format_retry_used = True
            _bump(stage, retries=1)
            continue
        raise _http_failed(route, provider, status, message)


def _fake(stage: str, route: str, text: str, schema):
    digest = hashlib.sha256(text.encode()).hexdigest()
    for candidate in (FAKE_DIR / f"{digest}.json", FAKE_DIR / f"{stage}.json"):
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, list) and "results" in getattr(schema, "model_fields", {}):
                payload = {"results": payload}
            _bump(stage, calls=1, model=route)
            return schema.model_validate(payload)
    raise LLMFailed(
        f"No recorded answer for {route}.",
        kind="config",
        route=route,
        detail=f"Looked for {FAKE_DIR} in the samples.",
        hint="Use a real model or add the recorded answer.",
    )


def generate(
    stage: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    models: list[str] | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    extra: dict | None = None,
    no_fallback: bool = False,
    use_cache: bool = True,
):
    text = prompt + JSON_RULE
    chain = list(models) if models else list(ROUTES.get(stage, []))
    if no_fallback:
        chain = chain[:1]
    if not chain:
        raise LLMFailed(
            f"No model is configured for the {stage} step.",
            kind="config",
            detail=f"No route is registered for stage {stage}.",
            hint="Choose Gemini free, DeepSeek, or Custom.",
        )

    failures: list[str] = []
    for route in chain:
        try:
            return _call_model(
                stage, route, text, schema, base_url, api_key_env, extra or {}, use_cache
            )
        except _NextModel as exc:
            reason = str(exc)
            failures.append(f"{route}: {reason}")
            _log_switch(stage, route, reason)
            continue
    detail = "; ".join(failures) or "no model route answered"
    raise QuotaExhausted(
        f"No model could answer the {stage} step. {detail}",
        reasons=failures,
        hint="Wait if a quota ran out, choose another model, or run Code only.",
    )


def run_many(stage: str, items: list, fn, workers: int) -> dict:
    results = {}
    total, done = len(items), 0
    _report(stage, 0, total)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        try:
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                _report(stage, done, total)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results
