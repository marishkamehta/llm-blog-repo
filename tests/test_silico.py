from unittest.mock import Mock, patch

from silico import ChatModelConfig, LLM, MockLLM, make_llm


def test_mock_needs_no_registry():
    assert make_llm("mock").respond("hello") == "1"


def test_mock_uses_latest_user_turn_from_trajectory():
    llm = MockLLM(reply=lambda turn: f"latest: {turn}")
    result = llm.respond_with_metadata(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ],
        system="reply briefly",
        config=ChatModelConfig(seed=1),
    )

    assert result.text == "latest: second"
    assert result.request["messages"][0] == {
        "role": "system",
        "content": "reply briefly",
    }
    assert result.request["generation"]["seed"] == 1


def test_client_captures_metadata_and_redacts_key():
    response = Mock(status_code=200)
    response.json.return_value = {
        "id": "chatcmpl-test",
        "model": "served-model",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 3},
    }
    response.headers = {
        "x-request-id": "request-123",
        "x-ratelimit-remaining-requests": "9",
        "set-cookie": "secret-cookie",
    }
    response.raise_for_status = Mock()

    client = LLM(
        base_url="http://example.test/v1",
        model="served-model",
        api_key="secret",
    )

    with patch("silico.llm.requests.post", return_value=response) as post:
        result = client.respond_with_metadata("hi", config=ChatModelConfig(seed=1))

    assert result.text == "hello"
    assert result.request["request_headers"]["Authorization"] == "[REDACTED]"
    assert result.response["id"] == "chatcmpl-test"
    assert result.response["rate_limit_headers"] == {
        "x-ratelimit-remaining-requests": "9"
    }
    assert result.response["headers"]["set-cookie"] == "[REDACTED]"
    assert post.call_args.kwargs["json"]["seed"] == 1
