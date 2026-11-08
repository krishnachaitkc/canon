from canon import llm
from canon.errors import ErrorInfo, describe


def test_a_bad_key_becomes_a_short_card_with_next_steps():
    info = describe(
        llm.LLMFailed(
            "DeepSeek rejected the API key (HTTP 401).",
            kind="auth",
            hint="Paste a valid DeepSeek key in the sidebar.",
            detail="Authentication Fails, Your api key: ****aaff is invalid",
        )
    )

    assert info.title == "The model provider rejected the API key"
    assert info.cause == "DeepSeek rejected the API key (HTTP 401)."
    assert any("Code only" in hint for hint in info.hints)
    assert any("Custom" in hint for hint in info.hints)
    assert "Authentication Fails" in info.detail


def test_a_quota_error_lists_each_model_reason():
    info = describe(
        llm.QuotaExhausted(
            "No model could answer the names step.",
            reasons=["a: daily quota reached", "b: rate limited too many times"],
            hint="Wait for the quota reset.",
        )
    )

    assert info.title == "The model quota is used up"
    assert "a: daily quota reached" in info.detail
    assert "b: rate limited too many times" in info.detail
    assert any("Code only" in hint for hint in info.hints)


def test_an_out_of_credit_error_says_so():
    info = describe(
        llm.QuotaExhausted(
            "No model could answer the names step.",
            reasons=["deepseek/deepseek-flash: insufficient balance"],
        )
    )

    assert info.title == "The model account is out of credit"
    assert any("Add credit" in hint for hint in info.hints)


def test_error_info_round_trips_for_session_state():
    info = ErrorInfo("A title", "A cause", ("One hint",), "raw detail")

    assert ErrorInfo.from_dict(info.as_dict()) == info
