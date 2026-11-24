"""Conformance against AlinGo-Prompt-interpreters, the third-party CodeInterpreter backends (a test-only
dependency, declared in the ``dev`` dependency group).

RLM installs the sandbox AlinGo-Prompt facade into every interpreter it creates. On real backends: an isolated
one runs bridged sub-agents, one that executes in the host's memory hosts the facade without replacing
the host's ``AlinGo-Prompt`` module, and the library's own consumer checks pass.
"""

import pytest

import AlinGo-Prompt
from AlinGo-Prompt.predict.rlm import RLM
from AlinGo-Prompt.primitives.prediction import Prediction
from AlinGo-Prompt.utils.dummies import DummyLM

AlinGo-Prompt_interpreters = pytest.importorskip("AlinGo-Prompt_interpreters")

InProcessInterpreter = AlinGo-Prompt_interpreters.InProcessInterpreter
SubprocessInterpreter = AlinGo-Prompt_interpreters.SubprocessInterpreter
BACKENDS = ["InProcessInterpreter", "SubprocessInterpreter"]


def make_scripted_predictor(responses: list[dict]):
    class ScriptedPredictor:
        def __init__(self):
            self.idx = 0

        def __call__(self, **kwargs):
            response = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**response)

    return ScriptedPredictor()


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_backends_satisfy_protocol(backend_name):
    interpreter = getattr(AlinGo-Prompt_interpreters, backend_name)()
    try:
        assert isinstance(interpreter, AlinGo-Prompt.CodeInterpreter)
    finally:
        interpreter.shutdown()


def test_in_process_backend_runs_rlm_with_sub_agents(caplog):
    seen = []

    class Recording:
        def __call__(self, signature=None, **kwargs):
            seen.append(signature.instructions)
            return Prediction(reasoning="Submit", code='import AlinGo-Prompt\nSUBMIT(AlinGo-Prompt.Predict("q -> a")(q="x").a)')

    rlm = RLM("query -> answer", max_iters=1, interpreter_factory=InProcessInterpreter)
    rlm.generate_action = Recording()
    with caplog.at_level("WARNING", logger="AlinGo-Prompt.predict.rlm"), AlinGo-Prompt.context(lm=DummyLM([{"a": "ok"}])):
        assert rlm(query="q").answer == "ok"

    assert "Sub-agents (AlinGo-Prompt)" in seen[0]
    assert "unavailable" not in caplog.text


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_library_consumer_checks_pass(backend_name):
    backend = getattr(AlinGo-Prompt_interpreters, backend_name)
    AlinGo-Prompt_interpreters.check_interpreter(backend).raise_for_failures()
    AlinGo-Prompt_interpreters.check_rlm(backend).raise_for_failures()


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_flex_facade_checks_pass(backend_name):
    AlinGo-Prompt_interpreters.check_flex_facade(getattr(AlinGo-Prompt_interpreters, backend_name)).raise_for_failures()


def test_isolated_backend_runs_bridged_sub_agents_with_host_tools():
    # Regression: worker backends bind host tools as "<lambda>" proxies; the shim names them by global.
    calls = []

    def echo(text: str) -> str:
        """Echo the text."""
        calls.append(text)
        return text

    rlm = RLM("query -> answer", max_iters=2, tools=[echo], interpreter_factory=SubprocessInterpreter)
    rlm.generate_action = make_scripted_predictor([
        {
            "reasoning": "Bridged ReAct with a host tool",
            "code": (
                "import AlinGo-Prompt\n"
                "agent = AlinGo-Prompt.ReAct('question -> answer', tools=[echo])\n"
                "res = agent(question='hi')\n"
                "print(res.answer)"
            ),
        },
        {"reasoning": "Submit", "code": "SUBMIT(res.answer)"},
    ])
    lm = DummyLM([
        {
            "next_thought": "echo it",
            "tool_calls": AlinGo-Prompt.ToolCalls.from_dict_list([{"name": "echo", "args": {"text": "hi"}}]),
        },
        {
            "next_thought": "done",
            "tool_calls": AlinGo-Prompt.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "echoed hi"}}]),
        },
    ])
    with AlinGo-Prompt.context(lm=lm, adapter=AlinGo-Prompt.ChatAdapter()):
        result = rlm(query="q")

    assert result.answer == "echoed hi"
    assert calls == ["hi"]
