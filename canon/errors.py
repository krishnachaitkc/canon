"""Turn an exception into a short explanation a person can act on.

The app renders these fields as a card; the CLI prints the same words. The raw provider
reply stays in `detail`, never in the headline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import llm


@dataclass(frozen=True)
class ErrorInfo:
    title: str
    cause: str
    hints: tuple[str, ...] = ()
    detail: str = ""

    def as_dict(self) -> dict:
        data = asdict(self)
        data["hints"] = list(self.hints)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ErrorInfo":
        return cls(
            title=str(data.get("title") or "Canon stopped"),
            cause=str(data.get("cause") or ""),
            hints=tuple(str(hint) for hint in data.get("hints") or ()),
            detail=str(data.get("detail") or ""),
        )


def _hints(*values: str) -> tuple[str, ...]:
    seen, hints = set(), []
    for value in values:
        text = (value or "").strip()
        if text and text not in seen:
            seen.add(text)
            hints.append(text)
    return tuple(hints)


CODE_ONLY = "Run Code only (no model) to check every script without sending anything."
TRY_AGAIN = "Run Canon again. Finished model calls are cached, so a retry skips them."


def describe(exc: BaseException) -> ErrorInfo:
    if isinstance(exc, llm.LLMFailed):
        return _llm_failed(exc)
    if isinstance(exc, llm.QuotaExhausted):
        detail = "\n".join(exc.reasons) or str(exc)
        lowered = detail.lower()
        if "balance" in lowered or "402" in lowered:
            title = "The model account is out of credit"
            hints = ("Add credit with the provider, switch to Gemini free, or run Code only.",)
        elif "quota" in lowered or "429" in lowered or "rate limit" in lowered:
            title = "The model quota is used up"
            hints = ("Wait for the quota reset, switch models, or run Code only.",)
        elif "could not reach" in lowered or "connection" in lowered or "timed out" in lowered:
            title = "Canon could not reach the model provider"
            hints = ("Check the connection and the base URL, then run again.",)
        else:
            title = "No model could answer"
            hints = (exc.hint or "Switch to another model or run Code only.",)
        return ErrorInfo(
            title,
            "Every model Canon tried in this run failed.",
            _hints(*hints, CODE_ONLY),
            detail,
        )
    if isinstance(exc, ValueError):
        return ErrorInfo(
            "Canon could not start",
            str(exc),
            _hints(
                "For Custom, use provider/name, for example openai/your-model.",
                "Check the model choice and try again.",
            ),
            "",
        )
    if isinstance(exc, FileNotFoundError):
        return ErrorInfo(
            "A file Canon needs is missing",
            str(exc),
            ("Check the upload and run Canon again.",),
            "",
        )
    return ErrorInfo(
        "Canon stopped before it finished",
        f"{type(exc).__name__}: {exc}",
        _hints(TRY_AGAIN, CODE_ONLY),
        "",
    )


def _llm_failed(exc: llm.LLMFailed) -> ErrorInfo:
    detail = exc.detail or str(exc)
    if exc.kind == "auth":
        return ErrorInfo(
            "The model provider rejected the API key",
            str(exc),
            _hints(
                exc.hint,
                "If the key belongs to another provider, choose Custom and enter that "
                "provider's base URL.",
                CODE_ONLY,
            ),
            detail,
        )
    if exc.kind == "config":
        return ErrorInfo(
            "Canon needs an API key",
            str(exc),
            _hints(exc.hint, CODE_ONLY),
            detail,
        )
    if exc.kind == "balance":
        return ErrorInfo(
            "The model account is out of credit",
            str(exc),
            _hints(
                exc.hint,
                "Switch to Gemini free or run Code only.",
            ),
            detail,
        )
    if exc.kind == "model":
        return ErrorInfo(
            "That model is not available",
            str(exc),
            _hints(
                exc.hint,
                "Pick Gemini free or DeepSeek, or choose Custom with a provider/name.",
            ),
            detail,
        )
    if exc.kind == "request":
        return ErrorInfo(
            "The model provider rejected the request",
            str(exc),
            _hints(exc.hint, "Try another model or run Code only."),
            detail,
        )
    if exc.kind == "response":
        return ErrorInfo(
            "The model's answer could not be used",
            str(exc),
            _hints(exc.hint, TRY_AGAIN, CODE_ONLY),
            detail,
        )
    if exc.kind == "http":
        return ErrorInfo(
            "The model provider refused the request",
            str(exc),
            _hints(exc.hint, "Try again in a moment, switch models, or run Code only."),
            detail,
        )
    return ErrorInfo(
        "Canon could not finish the model run",
        str(exc),
        _hints(exc.hint, TRY_AGAIN, CODE_ONLY),
        detail,
    )
