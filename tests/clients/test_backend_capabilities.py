"""Capability planning follows the native route, including per-call overrides."""

import asyncio

import pytest

import AlinGo-Prompt
from AlinGo-Prompt.clients import capabilities as hints
from AlinGo-Prompt.clients.engines import AsyncLM15Engine, LM15Engine
from AlinGo-Prompt.lm15 import Message, Response, Usage


@pytest.fixture
def metadata(monkeypatch):
    monkeypatch.setattr(hints, "model_info", lambda *args, **kwargs: {
        "supports_function_calling": True, "supports_reasoning": True, "supports_response_schema": True,
    })


@pytest.mark.parametrize("engine", ["auto", "lm15"])
def test_native_capabilities_follow_snapshot(engine, metadata):
    lm = AlinGo-Prompt.LM("openai/gpt-4o", engine=engine, api_key="fake", api_base="https://example.invalid")
    assert lm.supports_function_calling
    assert lm.supports_reasoning
    assert lm.supports_response_schema
    assert "response_format" in lm.supported_params
    lm.close()


def test_copy_recomputes_capabilities_without_mutating_source(monkeypatch):
    monkeypatch.setattr(hints, "model_info", lambda provider, model, **kwargs: {
        "supports_response_schema": model == "gpt-4o",
    })
    lm = AlinGo-Prompt.LM("openai/gpt-4o")
    assert lm.supports_response_schema
    assert not lm.copy(model="openai/unlisted").supports_response_schema
    assert lm.supports_response_schema


def test_configuration_refusal_matches_execution(metadata):
    lm = AlinGo-Prompt.LM("openai/gpt-4o", engine="lm15", headers={"x-test": "yes"})
    with pytest.raises(AlinGo-Prompt.LMUnsupportedFeatureError):
        lm("hello")
    with pytest.raises(AlinGo-Prompt.LMUnsupportedFeatureError):
        _ = lm.supports_response_schema


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_adapter_rejects_unsupported_call_time_client_without_leaking_override(asynchronous, metadata):
    lm = AlinGo-Prompt.LM("openai/gpt-4o", engine="lm15", cache=False)
    adapter = AlinGo-Prompt.JSONAdapter()
    args = (lm, {"custom_llm_provider": "alternate"}, AlinGo-Prompt.Signature("question -> answer"), [], {"question": "hi"})
    with pytest.raises(AlinGo-Prompt.LMUnsupportedFeatureError):
        await adapter.acall(*args) if asynchronous else adapter(*args)
    assert "custom_llm_provider" not in lm.kwargs
    assert lm.supports_response_schema


def test_chat_adapter_does_not_enable_unsupported_tools(monkeypatch):
    monkeypatch.setattr(hints, "model_info", lambda *args, **kwargs: {"supports_function_calling": False})
    requests = []

    def complete(self, request):
        requests.append(request)
        return Response(None, request.model, Message.assistant('[[ ## calls ## ]]\n{"tool_calls": []}'), "stop", Usage())

    monkeypatch.setattr(LM15Engine, "complete", complete)
    signature = AlinGo-Prompt.Signature({
        "tools": (list[AlinGo-Prompt.Tool], AlinGo-Prompt.InputField()), "calls": (AlinGo-Prompt.ToolCalls, AlinGo-Prompt.OutputField()),
    })
    result = AlinGo-Prompt.ChatAdapter(use_native_function_calling=True, use_json_adapter_fallback=False)(
        AlinGo-Prompt.LM("openai/gpt-4o", cache=False), {}, signature, [], {"tools": [AlinGo-Prompt.Tool(lambda query: query)]},
    )
    assert result[0]["calls"].tool_calls == []
    assert requests[0].tools == ()
    assert requests[0].config.tool_choice is None


@pytest.mark.asyncio
async def test_concurrent_planning_scopes_are_isolated(metadata, monkeypatch):
    async def complete(self, request):
        await asyncio.sleep(0)
        assert request.config.response_format is not None
        return Response(None, request.model, Message.assistant('{"answer":"native"}'), "stop", Usage())

    monkeypatch.setattr(AsyncLM15Engine, "complete", complete)
    lm = AlinGo-Prompt.LM("openai/gpt-4o", engine="lm15", cache=False)
    adapter = AlinGo-Prompt.JSONAdapter()
    signature = AlinGo-Prompt.Signature("question -> answer")
    try:
        results = await asyncio.gather(
            adapter.acall(lm, {}, signature, [], {"question": "a"}),
            adapter.acall(lm, {"headers": {"x-test": "no"}}, signature, [], {"question": "b"}),
            return_exceptions=True,
        )
        assert results[0] == [{"answer": "native"}]
        assert isinstance(results[1], AlinGo-Prompt.LMUnsupportedFeatureError)
        assert lm.supports_response_schema
    finally:
        await lm.aclose()
