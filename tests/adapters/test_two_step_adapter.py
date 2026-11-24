import pytest

import AlinGo-Prompt
from AlinGo-Prompt.lm15 import Message
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.test_utils.engines import recording_lm


def test_two_step_adapter_deprecation_warning():
    extraction_lm = AlinGo-Prompt.utils.DummyLM([{"answer": "x"}])
    with pytest.warns(
        DeprecationWarning,
        match=r"AlinGo-Prompt\.TwoStepAdapter is deprecated in DSPy 3\.5 and will be removed in DSPy 3\.6\. "
        r"Use AlinGo-Prompt\.JSONAdapter\(\) for models supporting structured outputs, or AlinGo-Prompt\.XMLAdapter\(\) otherwise\.",
    ) as warnings:
        adapter = AlinGo-Prompt.TwoStepAdapter(extraction_lm)

    assert len(warnings) == 1
    assert warnings[0].filename == __file__
    assert adapter.extraction_model is extraction_lm


def test_two_step_adapter_format_exact_messages_for_simple_signature_with_demo():
    class QA(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.TwoStepAdapter(AlinGo-Prompt.utils.DummyLM([{"answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(adapter, QA, [{"question": "Q1", "answer": "A1"}], {"question": "Q2"})

    expected_messages = [{"role": "system",
      "content": "You are a helpful assistant that can solve tasks based on user input.\n"
                 "As input, you will be provided with:\n"
                 "1. `question` (str):\n"
                 "Your outputs must contain:\n"
                 "1. `answer` (str):\n"
                 "You should lay out your outputs in detail so that your answer can be understood by "
                 "another agent\n"
                 "Specific instructions: Given the fields `question`, produce the fields `answer`."},
     {"role": "user", "content": "question: Q1"},
     {"role": "assistant", "content": "answer: A1"},
     {"role": "user", "content": "question: Q2"}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_format_conversation_history():
    class QA(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.TwoStepAdapter(AlinGo-Prompt.utils.DummyLM([{"answer": "x"}]))
    history = AlinGo-Prompt.History(messages=[{"question": "Q1", "answer": "A1"}, {"answer": "A2"}])

    messages = adapter.format_conversation_history(QA, "history", {"question": "Q3", "history": history})

    assert messages == [
        Message.user("question: Q1"),
        Message.assistant("answer: A1"),
        Message.assistant("answer: A2"),
    ]


def test_two_step_adapter_format_exact_messages_with_typed_outputs():
    class TypedSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        count: int = AlinGo-Prompt.OutputField()
        answer: str = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.TwoStepAdapter(AlinGo-Prompt.utils.DummyLM([{"count": 1, "answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(adapter, TypedSignature, [], {"question": "Q"})

    expected_messages = [{"role": "system",
      "content": "You are a helpful assistant that can solve tasks based on user input.\n"
                 "As input, you will be provided with:\n"
                 "1. `question` (str):\n"
                 "Your outputs must contain:\n"
                 "1. `count` (int): \n"
                 "2. `answer` (str):\n"
                 "You should lay out your outputs in detail so that your answer can be understood by "
                 "another agent\n"
                 "Specific instructions: Given the fields `question`, produce the fields `count`, "
                 "`answer`."},
     {"role": "user", "content": "question: Q"}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def _assert_two_step_requests(main_lm, extraction_lm, n=1):
    # main LM call
    assert len(main_lm.engine.requests) == n
    main_request = main_lm.engine.requests[0]
    assert len(main_request.messages) == 1
    assert "1. `question` (str)" in main_request.system
    assert "1. `solution` (str)" in main_request.system
    assert "2. `answer` (float)" in main_request.system
    assert main_request.messages[0].role == "user"
    assert "question:" in main_request.messages[0].text.lower()
    assert "What is 5 + 7?" in main_request.messages[0].text

    # extraction LM call
    assert len(extraction_lm.engine.requests) == n
    extraction_request = extraction_lm.engine.requests[0]
    assert len(extraction_request.messages) == 1
    assert "`text` (str)" in extraction_request.system
    assert "`solution` (str)" in extraction_request.system
    assert "`answer` (float)" in extraction_request.system
    assert extraction_request.messages[0].role == "user"
    assert "text from main LM" in extraction_request.messages[0].text


EXTRACTED = """
[[ ## solution ## ]] result
[[ ## answer ## ]] 12
[[ ## completed ## ]]
"""


def test_two_step_adapter_call():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField(desc="The math question to solve")
        solution: str = AlinGo-Prompt.OutputField(desc="Step by step solution")
        answer: float = AlinGo-Prompt.OutputField(desc="The final numerical answer")

    program = AlinGo-Prompt.Predict(TestSignature)
    main_lm = recording_lm(["text from main LM"], model="openai/gpt-4o-mini", temperature=1.0)
    extraction_lm = recording_lm([EXTRACTED], model="openai/gpt-4o", temperature=1.0)

    AlinGo-Prompt.configure(lm=main_lm, adapter=AlinGo-Prompt.TwoStepAdapter(extraction_model=extraction_lm))

    result = program(question="What is 5 + 7?")

    assert result.answer == 12
    _assert_two_step_requests(main_lm, extraction_lm)


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [1, 2])
async def test_two_step_adapter_async_call(n):
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField(desc="The math question to solve")
        solution: str = AlinGo-Prompt.OutputField(desc="Step by step solution")
        answer: float = AlinGo-Prompt.OutputField(desc="The final numerical answer")

    program = AlinGo-Prompt.Predict(TestSignature)
    main_lm = recording_lm(["text from main LM", "second candidate"], model="openai/gpt-4o-mini", temperature=1.0, n=n)
    extraction_lm = recording_lm([EXTRACTED, EXTRACTED.replace("12", "13")], model="openai/gpt-4o", temperature=1.0)

    with AlinGo-Prompt.context(lm=main_lm, adapter=AlinGo-Prompt.TwoStepAdapter(extraction_model=extraction_lm)):
        result = await program.acall(question="What is 5 + 7?")

    assert result.completions.answer == [12, 13][:n]
    assert len(main_lm.history) == len(extraction_lm.history) == n
    _assert_two_step_requests(main_lm, extraction_lm, n)


def test_two_step_adapter_parse():
    class ComplexSignature(AlinGo-Prompt.Signature):
        input_text: str = AlinGo-Prompt.InputField()
        tags: list[str] = AlinGo-Prompt.OutputField(desc="List of relevant tags")
        confidence: float = AlinGo-Prompt.OutputField(desc="Confidence score")

    first_response = "main LM response"

    # The chat-format parse fails on this JSON reply; the JSON fallback reads it.
    reply = """
        {
            "tags": ["AI", "deep learning", "neural networks"],
            "confidence": 0.87
        }
    """
    extraction_lm = recording_lm([reply, reply], model="openai/gpt-4o", temperature=1.0)
    adapter = AlinGo-Prompt.TwoStepAdapter(extraction_lm)
    AlinGo-Prompt.configure(adapter=adapter, lm=extraction_lm)

    result = adapter.parse(ComplexSignature, first_response)

    assert result["tags"] == ["AI", "deep learning", "neural networks"]
    assert result["confidence"] == 0.87


def test_two_step_adapter_parse_errors():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()

    first_response = "main LM response"
    # The extraction LM never answers in the chat format, so the chat parse and
    # its JSON fallback both fail.
    extraction_lm = recording_lm(["invalid response", "invalid response"], model="openai/gpt-4o", temperature=1.0)

    adapter = AlinGo-Prompt.TwoStepAdapter(extraction_lm)

    with pytest.raises(AlinGo-Prompt.AdapterParseError, match="Failed to parse response"):
        adapter.parse(TestSignature, first_response)
