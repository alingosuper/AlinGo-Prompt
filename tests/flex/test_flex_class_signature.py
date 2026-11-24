"""Class-based signatures through AlinGo-Prompt.Flex: the Flex's own signature, and ones defined by sandboxed code."""

import shutil
from decimal import Decimal
from typing import Literal

import pydantic
import pytest

import AlinGo-Prompt
from AlinGo-Prompt.primitives.code_interpreter import CodeExecutionError
from AlinGo-Prompt.utils.dummies import DummyLM

INTERPRETERS = [
    pytest.param(
        AlinGo-Prompt.PythonInterpreter,
        marks=pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed"),
        id="deno",
    ),
    pytest.param(AlinGo-Prompt.LocalInterpreter, id="local"),
]


class Verdict(pydantic.BaseModel):
    label: str
    score: float


class Grade(AlinGo-Prompt.Signature):
    """Grade the essay against the rubric."""

    essay: str = AlinGo-Prompt.InputField(desc="the student's essay, verbatim")
    tone: Literal["formal", "casual"] = AlinGo-Prompt.OutputField(desc="register of the essay")
    points: int = AlinGo-Prompt.OutputField(desc="total points awarded", ge=0, le=100)
    verdict: Verdict = AlinGo-Prompt.OutputField(desc="structured verdict")


ANSWER = {"tone": "formal", "points": 87, "verdict": '{"label": "pass", "score": 0.87}'}

# Sandboxed code with its own class-based signature: inherited fields and instructions, a typing annotation,
# a quoted custom type.
SANDBOX_SIGNATURE_SRC = '''
from typing import Literal


class EssayInput(AlinGo-Prompt.Signature):
    """Grade the essay against the rubric."""

    essay: str = AlinGo-Prompt.InputField(desc="the student's essay, verbatim")


class SubGrade(EssayInput):
    tone: Literal["formal", "casual"] = AlinGo-Prompt.OutputField(desc="register of the essay")
    points: int = AlinGo-Prompt.OutputField(desc="total points awarded", ge=0, le=100)
    verdict: "Verdict" = AlinGo-Prompt.OutputField(desc="structured verdict")


class GradeModule(AlinGo-Prompt.Module):
    def __init__(self):
        super().__init__()
        self.grade = AlinGo-Prompt.Predict(SubGrade)

    def forward(self, essay):
        r = self.grade(essay=essay)
        return AlinGo-Prompt.Prediction(tone=r.tone, points=r.points, verdict=r.verdict)
'''


def assert_same_fields(actual, expected):
    assert actual.instructions == expected.instructions
    assert list(actual.input_fields) == list(expected.input_fields)
    assert list(actual.output_fields) == list(expected.output_fields)
    for name, field in expected.fields.items():
        assert actual.fields[name].annotation == field.annotation
        assert actual.fields[name].metadata == field.metadata
        assert actual.fields[name].json_schema_extra == field.json_schema_extra


def assert_graded(lm, prediction):
    assert prediction.tone == "formal"
    assert prediction.points == 87
    assert prediction.verdict == Verdict(label="pass", score=0.87)
    prompt = lm.history[-1]["messages"][0]["content"]
    assert "the student's essay, verbatim" in prompt
    assert "greater than or equal to: 0" in prompt


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_flex_class_signature_keeps_field_metadata(factory):
    flex = AlinGo-Prompt.Flex(Grade, interpreter_factory=factory)
    lm = DummyLM([ANSWER])
    with AlinGo-Prompt.context(lm=lm, trace=[]):
        prediction = flex(essay="An essay.")
        [(predictor, _, _)] = AlinGo-Prompt.settings.trace

    assert_graded(lm, prediction)
    assert_same_fields(predictor.signature, Grade)


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_sandboxed_class_signature(factory):
    flex = AlinGo-Prompt.Flex(Grade, interpreter_factory=factory)
    flex.load_state({"module_src": SANDBOX_SIGNATURE_SRC})
    lm = DummyLM([ANSWER])
    with AlinGo-Prompt.context(lm=lm, trace=[]):
        prediction = flex(essay="An essay.")
        [(predictor, _, _)] = AlinGo-Prompt.settings.trace

    assert_graded(lm, prediction)
    assert_same_fields(predictor.signature, Grade)


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_sandboxed_class_signature_requires_input_or_output_field(factory):
    flex = AlinGo-Prompt.Flex(Grade, interpreter_factory=factory)
    flex.load_state({"module_src": SANDBOX_SIGNATURE_SRC.replace(' = AlinGo-Prompt.InputField(desc="the student\'s essay, verbatim")', "")})

    with AlinGo-Prompt.context(lm=DummyLM([ANSWER])):
        with pytest.raises(CodeExecutionError, match=r"essay.*must be declared with InputField or OutputField"):
            flex(essay="An essay.")


@pytest.mark.parametrize("factory", INTERPRETERS)
@pytest.mark.parametrize("note_fields", [{}, {"note": None}, {"note": "jsnull"}, {"note": ""}],
                         ids=["omitted", "null", "literal-jsnull", "empty-string"])
def test_sandbox_prediction_preserves_null_and_string_outputs(factory, note_fields):
    source = """
class Task(AlinGo-Prompt.Signature):
    question: str = AlinGo-Prompt.InputField()
    answer: str = AlinGo-Prompt.OutputField()
    note: str | None = AlinGo-Prompt.OutputField(default=None)
    nested: dict[str, list[str | None]] = AlinGo-Prompt.OutputField()

class Module(AlinGo-Prompt.Module):
    def forward(self, question):
        result = AlinGo-Prompt.Predict(Task)(question=question)
        return AlinGo-Prompt.Prediction(note=result.note, nested=result.nested)
"""
    nested = {"values": [None, "jsnull", "", "null", "None"]}
    program = AlinGo-Prompt.Flex("question -> note: str | None, nested: dict", interpreter_factory=factory)
    program.load_state({"module_src": source})
    with AlinGo-Prompt.context(lm=DummyLM([{"answer": "ok", "nested": nested, **note_fields}]),
                      adapter=AlinGo-Prompt.JSONAdapter()):
        result = program(question="preserve nulls and literal strings")
    assert result.note == note_fields.get("note")
    assert result.nested == nested


@pytest.mark.parametrize("factory", INTERPRETERS)
def test_flex_baseline_skips_constraints_that_are_not_plain_numbers(factory):
    class Pay(AlinGo-Prompt.Signature):
        """Extract the amount."""

        text: str = AlinGo-Prompt.InputField(desc="invoice text")
        amount: Decimal = AlinGo-Prompt.OutputField(gt=Decimal("0"), le=1000)

    flex = AlinGo-Prompt.Flex(Pay, interpreter_factory=factory)
    assert "amount: Decimal = AlinGo-Prompt.OutputField(le=1000)" in flex.module_src
    with AlinGo-Prompt.context(lm=DummyLM([{"amount": "12.50"}])):
        assert flex(text="Total: 12.50").amount == Decimal("12.50")


def test_flex_baseline_keeps_string_signature_without_field_metadata():
    class Answer(AlinGo-Prompt.Signature):
        """Answer the question."""

        question: str = AlinGo-Prompt.InputField()
        answer: int = AlinGo-Prompt.OutputField()

    flex = AlinGo-Prompt.Flex(Answer, interpreter_factory=AlinGo-Prompt.LocalInterpreter)
    assert "AlinGo-Prompt.Predict(AlinGo-Prompt.Signature('question: str -> answer: int', 'Answer the question.'))" in flex.module_src
    assert "class Answer(AlinGo-Prompt.Signature)" not in flex.module_src


@pytest.mark.parametrize("factory", INTERPRETERS)
@pytest.mark.parametrize("adapter_type", [AlinGo-Prompt.ChatAdapter, AlinGo-Prompt.XMLAdapter, AlinGo-Prompt.JSONAdapter])
@pytest.mark.parametrize("provided", [False, True])
def test_loaded_sandbox_signature_preserves_output_defaults(factory, adapter_type, provided):
    source = """
class Defaults(AlinGo-Prompt.Signature):
    question: str = AlinGo-Prompt.InputField()
    answer: str = AlinGo-Prompt.OutputField()
    note: str | None = AlinGo-Prompt.OutputField(default="not provided")
    label: str = AlinGo-Prompt.OutputField(default="unclassified")
    count: int = AlinGo-Prompt.OutputField(default=0)
    enabled: bool = AlinGo-Prompt.OutputField(default=False)
    tags: list[str] = AlinGo-Prompt.OutputField(default=[])
    pair: tuple[str, int] = AlinGo-Prompt.OutputField(default=("fallback", 7))

class DefaultModule(AlinGo-Prompt.Module):
    def forward(self, question):
        return AlinGo-Prompt.Prediction(result=dict(AlinGo-Prompt.Predict(Defaults)(question=question)))
"""
    defaults = {"note": "not provided", "label": "unclassified", "count": 0, "enabled": False, "tags": [], "pair": ("fallback", 7)}
    supplied = {"note": "observed", "label": "classified", "count": 13, "enabled": True, "tags": ["seen"], "pair": ("chosen", 19)}
    response = {"answer": "ok", **(supplied if provided else {})}
    adapter = adapter_type() if adapter_type is AlinGo-Prompt.JSONAdapter else adapter_type(use_json_adapter_fallback=False)
    flex = AlinGo-Prompt.Flex("question -> result: dict", interpreter_factory=factory)
    flex.load_state({"module_src": source})
    with AlinGo-Prompt.context(lm=DummyLM([response]), adapter=adapter, trace=[]):
        result = flex(question="a response with optional fields")
        [(predictor, _, _)] = AlinGo-Prompt.settings.trace

    assert result.result == {"answer": "ok", **(supplied if provided else defaults)}
    # The traced predictor must retain the same defaults when used again on the host.
    with AlinGo-Prompt.context(lm=DummyLM([{"answer": "again"}]), adapter=adapter):
        assert dict(predictor(question="another response")) == {"answer": "again", **defaults}
