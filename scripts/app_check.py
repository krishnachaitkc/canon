"""Drive app.py with Streamlit's AppTest on the bundled sample. Print numbers only.

    .venv\\Scripts\\python scripts\\app_check.py --mode code
    .venv\\Scripts\\python scripts\\app_check.py --mode gemini

A model failure (quota, network, missing key) is a failure with its error, never a skip.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from canon import llm
from canon.issues import build_cards, check_places, metrics, ready_places

APP = ROOT / "app.py"
MODELS = {"code": "No AI", "gemini": "Gemini"}
llm.CACHE_DIR = Path(tempfile.mkdtemp(prefix="canon_appcheck_cache_"))


def _text(element) -> str:
    for name in ("value", "body", "message"):
        value = getattr(element, name, None)
        if value:
            return str(value)
    return str(element)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Drive the Canon app on the sample show.")
    parser.add_argument("--mode", choices=sorted(MODELS), required=True)
    args = parser.parse_args(argv)
    print(f"mode: {args.mode}")

    at = AppTest.from_file(str(APP), default_timeout=600).run()
    at.radio(key="input_mode").set_value("Sample show")
    at.radio(key="model_choice").set_value(MODELS[args.mode])
    at.button(key="run_button").click().run(timeout=600)

    state = at.session_state.get("canon_run")
    error = at.session_state.get("canon_error") or {}
    if isinstance(error, dict):
        error = ": ".join(
            part for part in (error.get("title"), error.get("cause")) if part
        )
    else:
        error = str(error)
    if at.error:
        error = error or _text(at.error[0])
    if at.exception:
        error = error or _text(at.exception[0])
    if not state or not state.get("manifest", {}).get("complete") or error:
        print("status: FAILED")
        print(f"error: {error or 'the app did not complete the run'}")
        return 1

    result = state["result"]
    manifest = state["manifest"]
    counts = manifest.get("counts", {})
    llm = manifest.get("llm") or {}
    calls = sum(int(stage.get("calls", 0)) for stage in llm.values())
    cached = sum(int(stage.get("cached", 0)) for stage in llm.values())
    tokens_in = sum(int(stage.get("input_tokens", 0)) for stage in llm.values())
    tokens_out = sum(int(stage.get("output_tokens", 0)) for stage in llm.values())
    cards = build_cards(result)
    card_counts = metrics(cards)
    print("status: OK")
    print(f"episodes: {counts.get('episodes', 0)}")
    print(f"findings: {len(result['findings'])}")
    print(f"checks: {sum(1 for finding in result['findings'] if finding.source == 'check')}")
    print(f"issues: {card_counts['issues']}")
    print(f"heard: {card_counts['heard']}")
    print(f"voice: {card_counts['voice']}")
    print(f"seen: {card_counts['seen']}")
    print(f"ready_cards: {sum(1 for card in cards if card.lane == 'ready')}")
    print(f"check_cards: {sum(1 for card in cards if card.lane == 'check')}")
    print(f"hand_cards: {sum(1 for card in cards if card.lane == 'hand')}")
    print(f"ready_places: {len(ready_places(cards))}")
    print(f"check_places: {len(check_places(cards))}")
    print(f"cards_shown: {sum(1 for button in at.button if button.label == 'Details')}")
    print(f"metrics: {len(at.metric)}")
    print(f"calls: {calls}")
    print(f"cached: {cached}")
    print(f"tokens_in: {tokens_in}")
    print(f"tokens_out: {tokens_out}")
    print(f"cost_usd: {manifest.get('cost_usd', 0.0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
