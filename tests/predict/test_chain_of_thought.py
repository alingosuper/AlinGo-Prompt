from unittest import mock

import pytest

import AlinGo-Prompt
from AlinGo-Prompt import ChainOfThought
from AlinGo-Prompt.utils import DummyLM
from AlinGo-Prompt.utils.exceptions import AdapterParseError
from tests.test_utils.engines import make_response


def test_initialization_with_string_signature():
    lm = DummyLM([{"plan": "find the number after 1", "answer": "2"}])
    AlinGo-Prompt.configure(lm=lm)
    predict = ChainOfThought("question -> answer")
    assert list(predict.predict.signature.output_fields.keys()) == [
        "reasoning",
        "answer",
    ]
    assert predict(question="What is 1+1?").answer == "2"


@pytest.mark.asyncio
async def test_async_chain_of_thought():
    lm = DummyLM([{"plan": "find the number after 1", "answer": "2"}])
    with AlinGo-Prompt.context(lm=lm):
        program = ChainOfThought("question -> answer")
        result = await program.acall(question="What is 1+1?")
        assert result.answer == "2"


def test_chain_of_thought_with_native_reasoning():
    """Native-capable models need no explicit reasoning or plan output."""

    lm = AlinGo-Prompt.LM(model="anthropic/claude-3-7-sonnet-20250219", cache=False)
    AlinGo-Prompt.settings.configure(lm=lm)

    with (
        mock.patch.object(AlinGo-Prompt.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=True),
        mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion,
    ):
        mock_completion.return_value = make_response(
            "<answer>\nParis\n</answer>",
            thinking="Paris is the capital of France.",
            model="anthropic/claude-3-7-sonnet-20250219",
        )

        cot = ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert isinstance(result.reasoning, AlinGo-Prompt.Reasoning)
        assert result.reasoning == "Paris is the capital of France."

        request = mock_completion.call_args.args[0]
        assert request.config.reasoning.effort == "low"
        assert "<reasoning>" not in str(request.messages)
        assert "<plan>" not in str(request.messages)


def test_chain_of_thought_with_manual_reasoning():
    """Test ChainOfThought with manual reasoning where LM doesn't support native reasoning."""
    lm = AlinGo-Prompt.LM(model="openai/gpt-4o-mini")
    AlinGo-Prompt.settings.configure(lm=lm)

    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "<plan>\nRecall the capital of France.\n</plan>\n"
            "<answer>\nParis\n</answer>",
            model="openai/gpt-4o-mini",
        )

        cot = ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert result.reasoning == "Recall the capital of France."


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("adapter_type", [AlinGo-Prompt.ChatAdapter, AlinGo-Prompt.JSONAdapter, AlinGo-Prompt.XMLAdapter])
@pytest.mark.parametrize("mode", ["native", "omitted", "unsupported", "disabled", "disabled_string"])
async def test_default_reasoning_modes(mode, adapter_type, async_mode):
    class Model(DummyLM):
        @property
        def supports_reasoning(self):
            return mode != "unsupported"

        @property
        def supported_params(self):
            return ["response_format", "reasoning_effort"]

        @property
        def supports_response_schema(self):
            return True

    native = mode in ("native", "omitted")
    adapter = adapter_type()
    responses = [{"answer": "17"}, {"answer": "29"}]
    if not native:
        for response in responses:
            response["plan"] = "Use the supplied observations."
    # Some models return native thinking even when effort is disabled; preserve the explicit field.
    lm = Model(responses, reasoning=mode in ("native", "disabled", "disabled_string"), adapter=adapter)
    lm.kwargs["reasoning_effort"] = "high"
    program = ChainOfThought("question -> answer", n=2)
    config = {"reasoning_effort": None} if mode == "disabled" else {}
    if mode == "disabled_string":
        config = {"reasoning_effort": "none"}

    with AlinGo-Prompt.context(lm=lm, adapter=adapter):
        result = await program.acall(question="Use observations", config=config) if async_mode else program(question="Use observations", config=config)

    expected = "Some reasoning" if mode == "native" else None if native else "Use the supplied observations."
    assert result.reasoning == expected
    assert result.completions.reasoning == [expected, expected]
    assert result.completions.answer == ["17", "29"]
    if adapter_type is AlinGo-Prompt.JSONAdapter:
        schema = lm.history[0]["request"].config.response_format["schema"]
        assert set(schema["properties"]) == ({"answer"} if native else {"plan", "answer"})
        if not native:
            assert schema["properties"]["plan"]["type"] == "string"


@pytest.mark.parametrize("adapter_type", [AlinGo-Prompt.ChatAdapter, AlinGo-Prompt.JSONAdapter, AlinGo-Prompt.XMLAdapter])
def test_reasoning_fallback_preserves_demos_and_history(tmp_path, adapter_type):
    class Task(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        history: AlinGo-Prompt.History = AlinGo-Prompt.InputField()
        plan: str = AlinGo-Prompt.OutputField()
        answer: str = AlinGo-Prompt.OutputField()

    program = ChainOfThought(Task)
    program.predict.demos = [AlinGo-Prompt.Example(question="Demo", reasoning="Demo reasoning", plan="Demo plan", answer="13")]
    program.save(tmp_path / "program.json")
    loaded = ChainOfThought(Task)
    loaded.load(tmp_path / "program.json")
    history = AlinGo-Prompt.History(messages=[{"question": "Previous", "reasoning": "Previous reasoning", "plan": "Previous plan", "answer": "19"}])
    adapter = adapter_type()
    lm = DummyLM([{"plan_2": "New reasoning", "plan": "New plan", "answer": "23"}], adapter=adapter)
    with AlinGo-Prompt.context(lm=lm, adapter=adapter):
        result = loaded(question="Current", history=history)
    assert result.reasoning == "New reasoning"
    assert result.plan == "New plan"
    messages = "\n".join(m["content"] for m in lm.history[0]["messages"])
    assert "plan_2" in messages
    assert "Demo reasoning" in messages and "Previous reasoning" in messages and "Previous plan" in messages
    assert loaded.predict.demos[0]["reasoning"] == "Demo reasoning"
    assert history.messages[0]["reasoning"] == "Previous reasoning"


@pytest.mark.parametrize("custom", [False, True])
def test_explicit_rationale_is_not_renamed(custom):
    field = AlinGo-Prompt.OutputField(desc="Custom working") if custom else None
    if field is not None:
        field.annotation = str
    program = ChainOfThought("question -> answer", rationale_field=field, rationale_field_type=str)
    with AlinGo-Prompt.context(lm=DummyLM([{"reasoning": "Working", "answer": "31"}])):
        result = program(question="Question")
    assert result.reasoning == "Working"
    assert isinstance(result.reasoning, str)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("adapter", [
    AlinGo-Prompt.ChatAdapter(use_json_adapter_fallback=False),
    AlinGo-Prompt.JSONAdapter(),
    AlinGo-Prompt.XMLAdapter(use_json_adapter_fallback=False),
])
async def test_parse_failure_preserves_predictor_signature(adapter, collision, async_mode):
    program = ChainOfThought("question -> plan, answer" if collision else "question -> answer")
    response = {"plan_2": "Working", "plan": "Task plan"} if collision else {"plan": "Working"}
    lm = DummyLM([response], adapter=adapter)

    with AlinGo-Prompt.context(lm=lm, adapter=adapter), pytest.raises(AdapterParseError) as exc_info:
        if async_mode:
            await program.acall(question="Question")
        else:
            program(question="Question")

    error = exc_info.value
    assert error.signature is program.predict.signature
    assert error.parsed_result == ({"reasoning": "Working", "plan": "Task plan"} if collision else {"reasoning": "Working"})
    assert error.lm_response == lm.history[-1]["outputs"][0]
    assert error.adapter_name == type(adapter).__name__


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("call_override", [False, True])
@pytest.mark.parametrize("thinking,reasoning", [
    ({"type": "disabled"}, None), ({"type": "enabled", "budget_tokens": 2048}, None),
    ({"type": "adaptive"}, None), ({"type": "disabled"}, {"effort": "off"}),
])
async def test_chain_of_thought_respects_provider_thinking(thinking, reasoning, call_override, asynchronous, monkeypatch):
    import json

    import AlinGo-Prompt.clients.execution as execution
    from AlinGo-Prompt._vendor.lm15.testing import FakeTransport
    from AlinGo-Prompt.lm15 import RouterConfig
    from tests.clients.test_lm_engine_execution import DualResponse

    disabled = thinking["type"] == "disabled"
    text = "<plan>Recall the capital.</plan>\n" if disabled else ""
    content = [{"type": "text", "text": text + "<answer>Paris</answer>"}]
    if not disabled:
        content.insert(0, {"type": "thinking", "thinking": "Recall the capital.", "signature": "signature"})
    raw = {"id": "reply", "model": "claude-sonnet-4-5", "role": "assistant", "type": "message",
           "content": content, "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 5}}
    transport = FakeTransport([DualResponse(status=200, body=json.dumps(raw).encode())])
    monkeypatch.setattr(execution, "RouterConfig", lambda **kwargs: RouterConfig(
        **{**kwargs, "api_keys": {"anthropic": "test"}, "transport": transport},
    ))
    default = {"type": "enabled", "budget_tokens": 1024} if disabled else {"type": "disabled"}
    lm = AlinGo-Prompt.LM("anthropic/claude-sonnet-4-5", engine="lm15", cache=False, num_retries=0,
                 max_tokens=4096, thinking=default if call_override else thinking)
    program = ChainOfThought("question -> answer")
    config = {"thinking": thinking} if call_override else {}
    if reasoning is not None:
        config["reasoning"] = reasoning
    try:
        with AlinGo-Prompt.context(lm=lm):
            if asynchronous:
                result = await program.acall(question="What is the capital of France?", config=config)
            else:
                result = program(question="What is the capital of France?", config=config)
        assert result.answer == "Paris"
        assert result.reasoning == "Recall the capital."
        body = json.loads(transport.requests[0].body)
        assert body["thinking"] == thinking
        if reasoning is None:
            assert lm.history[-1]["request"].config.reasoning is None
        else:
            assert lm.history[-1]["request"].config.reasoning.is_off
    finally:
        await lm.aclose()


def test_chain_of_thought_rejects_conflicting_explicit_reasoning():
    lm = AlinGo-Prompt.LM("anthropic/claude-sonnet-4-5", engine="lm15", thinking={"type": "disabled"})
    with AlinGo-Prompt.context(lm=lm), pytest.raises(ValueError, match="contradictory"):
        ChainOfThought("question -> answer", reasoning_effort="high")(question="What is 2 + 2?")
