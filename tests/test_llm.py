import json
import logging
import time

import pytest
import requests
from pydantic import BaseModel

from canon import llm

MODEL = "deepseek/deepseek-flash"


class Out(BaseModel):
    ok: bool = True
    value: str = ""


class ListOut(BaseModel):
    results: list[str]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


def openai_response(content="", finish="stop", usage=None):
    return FakeResponse(
        payload={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )


def mock_post(monkeypatch, responses, calls=None):
    queue = list(responses)

    def fake_post(url, headers=None, json=None, timeout=None):
        if calls is not None:
            calls.append({"url": url, "headers": headers or {}, "json": json, "timeout": timeout})
        if not queue:
            raise AssertionError("unexpected extra HTTP call")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm.requests, "post", fake_post)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "RUN_DIR", None)
    monkeypatch.setattr(llm, "_sleep", lambda seconds: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    with llm._LOCK:
        for provider, gap in (
            ("deepseek", 0.0),
            ("openai", 0.0),
            ("anthropic", 0.0),
            ("gemini", 1.0),
            ("fake", 0.0),
        ):
            llm._PACE[provider] = [gap, 0.0]
    llm.reset_stats()
    llm.set_progress(None)
    yield


def test_503_then_200_succeeds(monkeypatch):
    calls = []
    mock_post(
        monkeypatch,
        [FakeResponse(503, text="busy"), openai_response('{"ok": true}')],
        calls,
    )

    result = llm.generate("names", "prompt", Out, models=[MODEL])

    assert result.ok is True
    assert len(calls) == 2
    assert llm.get_stats()["names"]["retries"] == 1


def test_per_minute_429_then_200_raises_the_gap(monkeypatch):
    limited = FakeResponse(
        429,
        payload={"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerMinute"}]}]}},
    )
    mock_post(monkeypatch, [limited, openai_response('{"ok": true}')])

    result = llm.generate("names", "prompt", Out, models=[MODEL])

    assert result.ok is True
    assert llm.provider_gap("deepseek") == 4.5
    assert llm.get_stats()["names"]["rate_limited"] == 1


def test_per_day_429_switches_model_and_logs(monkeypatch, caplog):
    limited = FakeResponse(
        429,
        payload={"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDay"}]}]}},
    )
    calls = []
    mock_post(monkeypatch, [limited, openai_response('{"ok": true}')], calls)

    with caplog.at_level(logging.WARNING):
        result = llm.generate(
            "names",
            "prompt",
            Out,
            models=["deepseek/deepseek-flash", "deepseek/deepseek-reasoner"],
        )

    assert result.ok is True
    assert calls[1]["json"]["model"] == "deepseek-reasoner"
    assert any("model switch" in entry for entry in llm.get_switches())
    assert "model switch" in caplog.text


def test_every_model_out_of_quota_raises(monkeypatch):
    limited = FakeResponse(
        429,
        payload={"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDay"}]}]}},
    )
    mock_post(monkeypatch, [limited, limited])

    with pytest.raises(llm.QuotaExhausted) as error:
        llm.generate(
            "names",
            "prompt",
            Out,
            models=["deepseek/deepseek-flash", "deepseek/deepseek-reasoner"],
        )

    assert "deepseek/deepseek-flash" in str(error.value)
    assert "deepseek/deepseek-reasoner" in str(error.value)
    assert len(error.value.reasons) == 2


def test_402_moves_to_the_next_model(monkeypatch):
    calls = []
    mock_post(
        monkeypatch,
        [FakeResponse(402, text="insufficient balance"), openai_response('{"ok": true}')],
        calls,
    )

    result = llm.generate(
        "names",
        "prompt",
        Out,
        models=["deepseek/deepseek-flash", "deepseek/deepseek-reasoner"],
    )

    assert result.ok is True
    assert calls[1]["json"]["model"] == "deepseek-reasoner"


def test_connection_reset_then_200_succeeds(monkeypatch):
    mock_post(
        monkeypatch,
        [requests.exceptions.ConnectionError("reset"), openai_response('{"ok": true}')],
    )

    result = llm.generate("names", "prompt", Out, models=[MODEL])

    assert result.ok is True


def test_malformed_json_is_repaired_without_a_retry(monkeypatch):
    calls = []
    mock_post(monkeypatch, [openai_response('{"ok": true,}')], calls)

    result = llm.generate("names", "prompt", Out, models=[MODEL])

    assert result.ok is True
    assert len(calls) == 1


def test_irreparable_json_then_valid_succeeds(monkeypatch):
    def unrecoverable(_text):
        raise ValueError("cannot repair")

    monkeypatch.setattr(llm.json_repair, "loads", unrecoverable)
    calls = []
    mock_post(
        monkeypatch,
        [openai_response("this is not json {{{"), openai_response('{"ok": true}')],
        calls,
    )

    result = llm.generate("names", "prompt", Out, models=[MODEL])

    assert result.ok is True
    assert len(calls) == 2
    assert "invalid JSON" in calls[1]["json"]["messages"][0]["content"]


def test_irreparable_json_four_times_raises(monkeypatch):
    def unrecoverable(_text):
        raise ValueError("cannot repair")

    monkeypatch.setattr(llm.json_repair, "loads", unrecoverable)
    calls = []
    mock_post(monkeypatch, [openai_response("nope {{{") for _ in range(4)], calls)

    with pytest.raises(llm.LLMFailed):
        llm.generate("names", "prompt", Out, models=[MODEL])

    assert len(calls) == 4


def test_blank_content_retries(monkeypatch):
    calls = []
    mock_post(
        monkeypatch,
        [openai_response(""), openai_response('{"ok": true}')],
        calls,
    )

    result = llm.generate("names", "prompt", Out, models=[MODEL])

    assert result.ok is True
    assert len(calls) == 2


def test_length_finish_retries_with_double_max_tokens(monkeypatch):
    calls = []
    mock_post(
        monkeypatch,
        [openai_response("", finish="length"), openai_response('{"ok": true}')],
        calls,
    )

    result = llm.generate("baseline", "prompt", Out, models=[MODEL])

    assert result.ok is True
    assert calls[0]["json"]["max_tokens"] == 8000
    assert calls[1]["json"]["max_tokens"] == 16000


def test_bare_list_is_wrapped_when_schema_expects_results(monkeypatch):
    mock_post(monkeypatch, [openai_response('["a", "b"]')])

    result = llm.generate("names", "prompt", ListOut, models=[MODEL])

    assert result.results == ["a", "b"]


def test_second_identical_call_is_a_cache_hit(monkeypatch):
    calls = []
    mock_post(monkeypatch, [openai_response('{"ok": true}')], calls)

    llm.generate("names", "prompt", Out, models=[MODEL])
    second = llm.generate("names", "prompt", Out, models=[MODEL])

    assert second.ok is True
    assert len(calls) == 1
    assert llm.get_stats()["names"]["cached"] == 1


def test_one_failed_call_does_not_block_a_later_call(monkeypatch):
    limited = FakeResponse(
        429,
        payload={"error": {"details": [{"violations": [{"quotaId": "GenerateRequestsPerDay"}]}]}},
    )
    mock_post(monkeypatch, [limited])

    with pytest.raises(llm.QuotaExhausted):
        llm.generate("names", "first prompt", Out, models=[MODEL])

    mock_post(monkeypatch, [openai_response('{"ok": true}')])
    result = llm.generate("names", "second prompt", Out, models=[MODEL])

    assert result.ok is True


def test_only_the_name_and_spelling_judges_reason(monkeypatch):
    stages = ["names", "speakers", "spelling", "baseline"]
    calls = []
    mock_post(monkeypatch, [openai_response('{"ok": true}') for _ in stages], calls)

    for stage in stages:
        llm.generate(stage, f"{stage} prompt", Out, models=[MODEL])

    thinking = {stage: call["json"]["thinking"]["type"] for stage, call in zip(stages, calls)}
    assert thinking == {"names": "enabled", "speakers": "disabled", "spelling": "enabled", "baseline": "disabled"}
    assert [call["json"]["max_tokens"] for call in calls] == [32000, 8000, 32000, 8000]


def test_run_many_returns_results_keyed_by_item():
    def work(item):
        time.sleep(0.01 * (5 - item))
        return item * 2

    results = llm.run_many("names", [1, 2, 3, 4], work, workers=4)

    assert results == {1: 2, 2: 4, 3: 6, 4: 8}


def test_run_many_reports_progress_as_each_item_finishes():
    seen = []

    def work(item):
        return item

    llm.set_progress(lambda stage, done, total: seen.append((stage, done, total)))
    llm.run_many("names", ["a", "b", "c"], work, workers=1)

    assert seen[0] == ("names", 0, 3)
    assert seen[-1] == ("names", 3, 3)
    assert [done for _, done, _ in seen[1:]] == [1, 2, 3]
    assert {total for _, _, total in seen} == {3}


def test_a_phase_reports_without_a_call_count():
    seen = []
    llm.set_progress(lambda stage, done, total: seen.append((stage, done, total)))

    llm.report_phase("Reading the scripts")

    assert seen == [("Reading the scripts", 0, 0)]


def test_a_failing_progress_callback_never_breaks_a_run():
    def explode(stage, done, total):
        raise RuntimeError("the app is gone")

    llm.set_progress(explode)

    assert llm.run_many("names", [1, 2], lambda item: item, workers=1) == {1: 1, 2: 2}


def test_setting_progress_to_none_silences_it():
    seen = []
    llm.set_progress(lambda stage, done, total: seen.append(stage))
    llm.set_progress(None)

    llm.run_many("names", [1], lambda item: item, workers=1)

    assert seen == []


def test_api_key_never_appears_in_the_error(monkeypatch, caplog):
    secret = "super-secret-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    mock_post(monkeypatch, [FakeResponse(400, text=f"bad key {secret}")])

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(llm.LLMFailed) as error:
            llm.generate("names", "prompt", Out, models=[MODEL])

    assert secret not in str(error.value)
    assert secret not in error.value.detail
    assert secret not in caplog.text
    assert "***" in error.value.detail


def test_a_bad_key_is_reported_as_a_key_problem(monkeypatch):
    reply = {
        "error": {
            "message": "Authentication Fails, Your api key: ****aaff is invalid",
            "type": "authentication_error",
        }
    }
    mock_post(monkeypatch, [FakeResponse(401, payload=reply)])

    with pytest.raises(llm.LLMFailed) as error:
        llm.generate("names", "prompt", Out, models=[MODEL])

    assert error.value.kind == "auth"
    assert error.value.status == 401
    assert "rejected the API key" in str(error.value)
    assert "HTTP 401" in str(error.value)
    assert "sk-" in error.value.hint
    assert "Authentication Fails" in error.value.detail


def test_a_missing_key_says_which_setting_to_change(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")

    with pytest.raises(llm.LLMFailed) as error:
        llm.generate("names", "prompt", Out, models=[MODEL])

    assert error.value.kind == "config"
    assert "DeepSeek" in str(error.value)
    assert "DEEPSEEK_API_KEY" in error.value.detail
    assert "Code only" in error.value.hint


def test_a_missing_model_names_the_model(monkeypatch):
    mock_post(monkeypatch, [FakeResponse(404, text='{"error":{"message":"model not found"}}')])

    with pytest.raises(llm.LLMFailed) as error:
        llm.generate("names", "prompt", Out, models=["deepseek/no-such-model"])

    assert error.value.kind == "model"
    assert "no-such-model" in str(error.value)
