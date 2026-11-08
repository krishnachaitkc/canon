from dataclasses import dataclass, field
from datetime import datetime, timezone

PRIMARY = "deepseek/deepseek-flash"
FREE = ["gemini/gemini-3.5-flash-lite", "gemini/gemini-3.1-flash-lite"]
# Measured on the pack: with reasoning, the hard name group went from 0 to 10 of 10 (with the
# line before and after each spoken mention). The speakers judge does not reason: on the
# 64-label prompt it spent 60,000 reasoning tokens over two tries without answering, and
# without reasoning it got the main characters right. The plain-prompt baseline does not think.
THINK_STAGES: set[str] = {"names", "spelling"}
STAGES = ["names", "speakers", "spelling", "baseline"]
ROUTES = {stage: [PRIMARY, *FREE] for stage in STAGES}
WORKERS = {"deepseek": 8, "openai": 8, "anthropic": 4, "gemini": 1, "fake": 1}

PROVIDERS = {"deepseek", "openai", "gemini", "anthropic", "fake"}

BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "openai": "https://api.openai.com/v1",
}

KEY_ENVS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def parse_route(spec: str) -> tuple[str, str]:
    if "/" not in spec:
        raise ValueError(f"Model must be provider/name, got {spec!r}")
    provider, name = spec.split("/", 1)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}")
    if not name:
        raise ValueError(f"Model name is missing in {spec!r}")
    return provider, name


def build_routes(
    model: str | None = None,
    free: bool = False,
    model_for: list[str] | None = None,
) -> dict[str, list[str]]:
    overrides = {}
    for item in model_for or []:
        if "=" not in item:
            raise ValueError(f"--model-for must be stage=provider/name, got {item!r}")
        stage, route = item.split("=", 1)
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}")
        parse_route(route)
        overrides[stage] = route
    if model:
        parse_route(model)

    routes = {}
    for stage in STAGES:
        if stage in overrides:
            routes[stage] = [overrides[stage]]
        elif model:
            routes[stage] = [model]
        elif free:
            routes[stage] = list(FREE)
        else:
            routes[stage] = list(ROUTES[stage])
    return routes


def workers_for(model: str) -> int:
    provider, _ = parse_route(model)
    return WORKERS[provider]


def workers_for_options(options: "ModelOptions", stage: str) -> int:
    if options.workers:
        return options.workers
    routes = options.routes.get(stage)
    if not routes:
        return 1
    return workers_for(routes[0])


@dataclass
class ModelOptions:
    routes: dict[str, list[str]]
    base_url: str | None = None
    api_key_env: str | None = None
    extra: dict = field(default_factory=dict)
    no_fallback: bool = False
    workers: int | None = None


# USD per million tokens, from the provider pricing pages read 2026-11-07.
# DeepSeek peak is 01:00-04:00 and 06:00-10:00 UTC, Mon-Fri.
PRICES_AS_OF = "2026-11-07"
PRICES = {
    "deepseek/deepseek-flash": {
        "input": 0.15,
        "cached": 0.003,
        "output": 0.60,
        "peak_input": 0.30,
        "peak_cached": 0.006,
        "peak_output": 1.20,
    },
    "openai/gpt-6-luna": {
        "input": 0.10,
        "cached": 0.01,
        "output": 0.50,
    },
    "gemini/gemini-3.5-flash-lite": {"input": 0.0, "cached": 0.0, "output": 0.0},
    "gemini/gemini-3.1-flash-lite": {"input": 0.0, "cached": 0.0, "output": 0.0},
}


def is_peak(now: datetime | None = None) -> bool:
    current = now or datetime.now(timezone.utc)
    if current.weekday() >= 5:
        return False
    return 1 <= current.hour < 4 or 6 <= current.hour < 10


def price_for(route: str, peak: bool) -> dict[str, float] | None:
    if route.startswith("fake/"):
        return {"input": 0.0, "cached": 0.0, "output": 0.0}
    prices = PRICES.get(route)
    if prices is None:
        return None
    if peak and "peak_input" in prices:
        return {
            "input": prices["peak_input"],
            "cached": prices["peak_cached"],
            "output": prices["peak_output"],
        }
    return {
        "input": prices["input"],
        "cached": prices["cached"],
        "output": prices["output"],
    }
