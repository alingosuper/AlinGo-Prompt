"""AlinGo-Prompt.Flex accepts tools, hands them to the underlying RLM baseline, exposes them to the
GEPA code proposer, and round-trips them through save/load.

Tools are callables/AlinGo-Prompt.Tool objects — they are NOT serialized. Like the signature, they are
re-provided at construction; the saved ``module_src`` references them by name and resolves
against the reconstructed Flex's context at bind time.

Flex runs its code through an interpreter (no in-process mode), so tests that only inspect
``module_src`` / context use a MockInterpreter (Deno-free), while tests that bind-and-run
generated code are marked ``deno_required``.
"""

from __future__ import annotations

import re
import shutil
import textwrap

import pytest

import AlinGo-Prompt
from AlinGo-Prompt.predict.flex import Flex
from AlinGo-Prompt.predict.flex.primitives_doc import PRIMITIVES_CATALOG
from AlinGo-Prompt.primitives.code_interpreter import CodeInterpreterError
from AlinGo-Prompt.teleprompt.gepa.gepa_flex_utils import CodeProposalSignature
from AlinGo-Prompt.teleprompt.gepa.gepa_utils import DspyAdapter
from AlinGo-Prompt.utils.dummies import DummyLM
from tests.mock_interpreter import MockInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")


def shout(text: str) -> str:
    """Return the text in uppercase."""
    return text.upper()


class Echo(AlinGo-Prompt.Signature):
    """Echo the question as the answer."""

    q: str = AlinGo-Prompt.InputField()
    a: str = AlinGo-Prompt.OutputField()


# A tool-using module with NO predictors: forward calls the in-scope `shout` tool directly,
# so it runs LM-free (stands in for code GEPA might produce that wires a tool in).
TOOL_MODULE = textwrap.dedent("""
    class EchoModule(AlinGo-Prompt.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q):
            return AlinGo-Prompt.Prediction(a=shout(q))
""").strip()


def test_baseline_is_rlm_with_tools() -> None:
    # With tools, the baseline delegates to AlinGo-Prompt.RLM (which can call them via its REPL), not Predict,
    # and the tools are wired into its constructor by name.
    src = Flex(Echo, tools=[shout], interpreter_factory=lambda: MockInterpreter()).module_src
    assert "AlinGo-Prompt.RLM(" in src
    assert "tools=[shout]" in src
    assert "AlinGo-Prompt.Predict(" not in src


def test_reserved_bridge_tool_names_are_rejected() -> None:
    # The sandbox bridge registers its own host tool (__AlinGo-Prompt_lm__) alongside user tools, and owns
    # `AlinGo-Prompt`/`_AlinGo-Prompt*` in the sandbox namespace; a user tool with a reserved name would silently
    # replace or be shadowed by them.
    for reserved in ("__AlinGo-Prompt_lm__", "_AlinGo-Prompt_host", "AlinGo-Prompt"):
        with pytest.raises(ValueError, match="reserved"):
            Flex(Echo, tools=[AlinGo-Prompt.Tool(shout, name=reserved)], interpreter_factory=lambda: MockInterpreter())


def test_baseline_is_predict_without_tools() -> None:
    # Without tools, the baseline is a single AlinGo-Prompt.Predict (no RLM, no tools arg).
    src = Flex(Echo, interpreter_factory=lambda: MockInterpreter()).module_src
    assert "AlinGo-Prompt.Predict(" in src
    assert "AlinGo-Prompt.RLM(" not in src
    assert "tools=[" not in src


@deno_required
def test_tool_is_in_scope_for_bound_code() -> None:
    program = Flex(Echo, tools=[shout], interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    program._bind_code(TOOL_MODULE)
    # LM-free: forward only calls the tool (bridged from the sandbox to the host).
    assert program(q="hello").a == "HELLO"


@deno_required
def test_save_load_roundtrips_tool_using_code(tmp_path) -> None:
    path = tmp_path / "program.json"
    program = Flex(Echo, tools=[shout], interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    program._bind_code(TOOL_MODULE)
    program.save(path)

    # Reconstruct with the SAME tools (the architecture is re-provided in code, like the
    # signature), then load the saved code — the tool reference resolves and runs.
    reloaded = Flex(Echo, tools=[shout], interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    reloaded.load(path)
    assert "shout(q)" in reloaded.module_src
    assert reloaded(q="hi").a == "HI"


@deno_required
def test_load_without_tools_cannot_resolve_them(tmp_path) -> None:
    # The baseline RLM is built with tools=[shout] in __init__; reconstructing without the
    # tool can't resolve the name in the sandbox. Loading succeeds (binding is a host-side
    # syntax check) — the failure surfaces when the code first runs (tools must be re-provided).
    path = tmp_path / "program.json"
    program = Flex(Echo, tools=[shout], interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    program.save(path)

    reloaded = Flex(Echo, interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())  # tools NOT re-provided
    reloaded.load(path)
    with pytest.raises(CodeInterpreterError):
        reloaded(q="hi")


# The primitives catalog offers the proposer `AlinGo-Prompt.Tool(func)`; wrapping a provided tool again must
# keep it the host tool.
WRAPPED_TOOL_MODULE = textwrap.dedent("""
    class EchoModule(AlinGo-Prompt.Module):
        def __init__(self):
            super().__init__()
            self.react = AlinGo-Prompt.ReAct("q -> a", tools=[AlinGo-Prompt.Tool(shout)])

        def forward(self, **inputs):
            return AlinGo-Prompt.Prediction(a=shout(inputs["q"]))
""").strip()


@deno_required
def test_AlinGo-Prompt_tool_wrapper_is_available_in_the_sandbox() -> None:
    program = Flex(Echo, tools=[shout], interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    program._bind_code(WRAPPED_TOOL_MODULE)
    assert program(q="hi").a == "HI"


def test_proposer_prompts_only_advertise_names_AlinGo-Prompt_defines() -> None:
    """Every `AlinGo-Prompt.X(...)` the proposer is told to write must exist: the sandbox runs the real AlinGo-Prompt, and a
    name advertised but not defined becomes an AttributeError at bind time, scored as a failed candidate."""
    prompts = PRIMITIVES_CATALOG + (CodeProposalSignature.__doc__ or "")
    advertised = set(re.findall(r"AlinGo-Prompt\.([A-Z]\w+)\(", prompts))
    assert advertised, "expected the prompts to name some primitives"
    missing = sorted(name for name in advertised if not hasattr(AlinGo-Prompt, name))
    assert not missing, f"advertised to the proposer but missing from AlinGo-Prompt: {missing}"


def test_gepa_proposer_is_told_about_tools() -> None:
    """GEPA can optimize tool usage because the proposer is shown the available tools."""

    def _metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0

    student = Flex(Echo, tools=[shout], interpreter_factory=lambda: MockInterpreter())
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={})
    assert "shout" in adapter._flex_context_blurbs["self"]


# A module that AUTHORS its own helper (defined inside forward) and calls it directly. The helper
# lives in the source, so it persists with the code itself.
SELF_AUTHORED_TOOL_MODULE = textwrap.dedent("""
    class EchoModule(AlinGo-Prompt.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **inputs):
            def upcase(text: str) -> str:
                "Uppercase the text."
                return text.upper()
            return AlinGo-Prompt.Prediction(a=upcase(inputs["q"]))
""").strip()


@deno_required
def test_self_authored_tool_persists_with_the_code(tmp_path) -> None:
    # Helpers GEPA writes itself live in module_src, so they persist exactly like the code (and like
    # instructions): reconstructing with ONLY the signature — no tools re-provided — rebuilds the
    # helper from the saved source and runs it.
    path = tmp_path / "program.json"
    flex = Flex(Echo, interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())  # no user-provided tools
    flex._bind_code(SELF_AUTHORED_TOOL_MODULE)
    assert "def upcase" in flex.module_src
    flex.save(path)

    reloaded = Flex(Echo, interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())  # still no tools
    reloaded.load(path)
    assert reloaded.module_src == SELF_AUTHORED_TOOL_MODULE
    assert reloaded(q="hi").a == "HI"


# Instructions baked into the generated code, on an inner predictor.
INSTR_MODULE = textwrap.dedent("""
    class EchoModule(AlinGo-Prompt.Module):
        def __init__(self):
            super().__init__()
            self.p = AlinGo-Prompt.Predict(AlinGo-Prompt.Signature("q -> a", "Always answer in uppercase."))

        def forward(self, q):
            return AlinGo-Prompt.Prediction(a=self.p(q=q).a)
""").strip()


@deno_required
def test_inner_predictor_instructions_persist(tmp_path) -> None:
    # The instructions GEPA writes into the code persist through save/load via module_src alone:
    # each forward rebuilds the predictor from the source, instructions included — observable in
    # the prompt the LM actually receives.
    path = tmp_path / "program.json"
    lm = DummyLM([{"a": "HI"}] * 4)
    AlinGo-Prompt.configure(lm=lm)
    flex = Flex(Echo, interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    flex._bind_code(INSTR_MODULE)
    flex(q="hi")
    assert "Always answer in uppercase." in str(lm.history[-1]["messages"])
    flex.save(path)

    reloaded = Flex(Echo, interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    reloaded.load(path)
    reloaded(q="hi")
    assert "Always answer in uppercase." in str(lm.history[-1]["messages"])


def double(n: int) -> str:
    """Double the number."""
    return str(n * 2)


@deno_required
def test_direct_tool_calls_go_through_the_tool_wrapper() -> None:
    # Tools keep their AlinGo-Prompt.Tool wrapper when the sandbox calls them directly, so DSPy's argument
    # validation applies: n=3 works, n="3" is rejected loudly — the raw underlying function would
    # have silently returned "33".
    def src(arg: str) -> str:
        return textwrap.dedent(
            f"""
            class EchoModule(AlinGo-Prompt.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, q):
                    return AlinGo-Prompt.Prediction(a=double(n={arg}))
            """
        ).strip()

    program = Flex(Echo, tools=[AlinGo-Prompt.Tool(double)], interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    program._bind_code(src("3"))
    assert program(q="x").a == "6"
    program._bind_code(src('"3"'))
    with pytest.raises(CodeInterpreterError, match="not of type 'integer'"):
        program(q="x")


@deno_required
def test_direct_tool_calls_receive_restored_custom_type_inputs() -> None:
    """A custom-type input (AlinGo-Prompt.Image) crosses into the sandbox as its fields and is an Image there; a
    direct tool call must hand the host tool the original object."""
    seen: dict = {}

    def describe(img) -> str:
        """Describe an image."""
        seen["img"] = img
        return "a cat"

    class Caption(AlinGo-Prompt.Signature):
        image: AlinGo-Prompt.Image = AlinGo-Prompt.InputField()
        a: str = AlinGo-Prompt.OutputField()

    src = textwrap.dedent(
        """
        class CaptionModule(AlinGo-Prompt.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                return AlinGo-Prompt.Prediction(a=describe(img=inputs["image"]))
        """
    ).strip()

    program = Flex(Caption, tools=[describe], interpreter_factory=lambda: AlinGo-Prompt.PythonInterpreter())
    program._bind_code(src)
    image = AlinGo-Prompt.Image(url="https://example.com/x.png")
    assert program(image=image).a == "a cat"
    assert seen["img"] is image  # the original object, not the serialized tag string
