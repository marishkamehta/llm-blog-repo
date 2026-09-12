from unittest.mock import Mock, patch

from llm_behavior_pipeline import GenerationConfig, ModelClient, make_llm


def test_mock_needs_no_registry():
    assert make_llm("mock").respond("hello") == "MOCK_RESPONSE"


def test_client_captures_metadata_and_redacts_key():
    response = Mock(status_code=200)
    response.json.return_value = {
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
    client = ModelClient(base_url="http://example.test/v1", model="served-model", api_key="secret")
    with patch("llm_behavior_pipeline.client.requests.post", return_value=response) as post:
        result = client.respond_with_metadata("hi", config=GenerationConfig(seed=1))
    assert result.text == "hello"
    assert result.request["headers"]["Authorization"] == "[REDACTED]"
    assert result.response["request_id"] == "request-123"
    assert result.response["rate_limit_headers"] == {
        "x-ratelimit-remaining-requests": "9"
    }
    assert result.response["headers"]["set-cookie"] == "[REDACTED]"
    assert post.call_args.kwargs["json"]["seed"] == 1
