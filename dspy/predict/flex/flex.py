from __future__ import annotations

from typing import Any, Callable

from AlinGo-Prompt.clients.base_lm import BaseLM
from AlinGo-Prompt.dsp.utils.settings import settings
from AlinGo-Prompt.predict.flex.bridge import BridgeRuntime
from AlinGo-Prompt.predict.flex.ctx import FlexContext
from AlinGo-Prompt.predict.parameter import Parameter
from AlinGo-Prompt.predict.predict import _sanitize_lm_state
from AlinGo-Prompt.primitives.code_interpreter import CodeInterpreter, _validate_interpreter_factory
from AlinGo-Prompt.primitives.module import Module
from AlinGo-Prompt.primitives.python_interpreter import PythonInterpreter
from AlinGo-Prompt.signatures.signature import Signature, ensure_signature


class Flex(Module, Parameter):
    """A module whose implementation is optimizable code, not just a prompt.

    Construct it like any module (``AlinGo-Prompt.Flex(MySignature)``). It starts as a baseline that delegates
    to a single ``AlinGo-Prompt.Predict`` over the signature — or ``AlinGo-Prompt.RLM`` when ``tools`` are given, so the
    baseline can call them. ``AlinGo-Prompt.GEPA`` recognizes ``Flex`` instances by type and rewrites their
    source — a single ``AlinGo-Prompt.Module`` subclass, exposed as ``module_src`` — into decomposed predictors
    plus plain Python instead of only tuning instructions.

    The optimizer-authored code runs inside an interpreter. ``Flex`` never
    runs it in the host Python process. ``interpreter_factory`` defaults to
    ``AlinGo-Prompt.PythonInterpreter`` (Deno/Pyodide) and must be a zero-argument callable returning a new
    ``CodeInterpreter``. ``AlinGo-Prompt.configure(interpreter_factory=...)`` replaces that default.
    The source imports the real AlinGo-Prompt in the interpreter; a custom interpreter therefore defines the
    Python and standard-library subset available to it, and must let it import the host's AlinGo-Prompt.
    The optimizer-authored code runs isolated; only provided-tool calls and LM calls go back to the
    host, which makes the real LM calls.

    Args:
        signature: A ``AlinGo-Prompt.Signature`` class or string declaring inputs/outputs.
        tools: ``AlinGo-Prompt.Tool`` instances or named callables.
        interpreter_factory: Zero-argument callable returning a fresh ``CodeInterpreter`` for each
            interpreter session. Defaults to ``AlinGo-Prompt.PythonInterpreter`` (sandbox, requires Deno).
            ``AlinGo-Prompt.configure(interpreter_factory=...)`` replaces the default.
        max_predictor_calls: Maximum number of LM calls the optimizer-authored code can make per
            ``forward``. ``None`` removes the limit.
    """

    def __init__(
        self,
        signature: Any,
        *,
        tools: list[Any] | None = None,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
        max_predictor_calls: int | None = 100,
    ):
        super().__init__()

        self._signature_cls = ensure_signature(signature)
        self._name = getattr(self._signature_cls, "__name__", None) or "Flex"
        self._flex_ctx = FlexContext(signature_cls=self._signature_cls, tools=list(tools or []))

        self._module_src: str | None = None
        self.lm = None

        _validate_interpreter_factory(interpreter_factory)
        self._interpreter_factory = interpreter_factory
        self._max_predictor_calls = max_predictor_calls

        self._rebuild_bridge()
        self._bind_code(self._baseline_src())

    def _rebuild_bridge(self) -> None:
        self._bridge: BridgeRuntime = BridgeRuntime(self, self._interpreter_factory, self._max_predictor_calls)

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("_bridge", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self._rebuild_bridge()
        if self._module_src is not None:
            self._bridge.bind(self._module_src)

    @property
    def signature(self) -> type[Signature]:
        return self._signature_cls

    @property
    def module_src(self) -> str | None:
        return self._module_src

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        return {
            "module_src": self._module_src,
            "lm": self.lm.dump_state() if self.lm else None,
        }

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        module_src = state.get("module_src")
        if module_src:
            self._bind_code(module_src)
        lm_state = state.get("lm")
        if lm_state:
            sanitized = _sanitize_lm_state(lm_state, allow_unsafe_lm_state)
            self.lm = (
                BaseLM.load_state(sanitized, allow_custom_lm_class=allow_unsafe_lm_state) if sanitized else None
            )
        else:
            self.lm = None

    def reset(self) -> None:
        """Clear the LM; ``module_src`` is kept, and predictors are rebuilt from it each forward."""
        self.lm = None

    def named_predictors(self):
        """A Flex's update unit is only its ``module_src``."""
        return []

    def set_lm(self, lm) -> None:
        self.lm = lm

    def get_lm(self):
        return self.lm

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name})"

    def _baseline_src(self) -> str:
        """Starting source: one predictor over the whole signature, wrapped in a ``AlinGo-Prompt.Module``.

        Uses ``AlinGo-Prompt.RLM`` when tools are provided; otherwise a single ``AlinGo-Prompt.Predict``.
        """
        cls: type[Signature] = self._signature_cls
        returns = ", ".join(f"{name}=result[{name!r}]" for name in cls.output_fields)
        preamble = ""
        if any(self._flex_ctx.field_kwargs().values()):
            # A class-based signature keeps the field descriptions and constraints that the string form drops.
            sig_arg = self._class_name().removesuffix("Module")
            preamble = self._flex_ctx.render_signature_class(sig_arg) + "\n\n\n"
        else:
            sig_str = self._flex_ctx.render_signature_string()
            instructions = (getattr(cls, "instructions", "") or "").strip()
            if custom := self._flex_ctx.custom_types():
                # Caller-frame discovery does not find types nested inside list/dict annotations.
                bindings = ", ".join(f"{name!r}: {name}" for name in custom)
                sig_arg = f"AlinGo-Prompt.Signature({sig_str!r}, {instructions!r}, custom_types={{{bindings}}})"
            else:
                sig_arg = f"AlinGo-Prompt.Signature({sig_str!r}, {instructions!r})" if instructions else repr(sig_str)
        tool_names = list(self._flex_ctx.context_names())
        if tool_names:
            attr, ctor = "rlm", f"AlinGo-Prompt.RLM({sig_arg}, tools=[{', '.join(tool_names)}])"
        else:
            attr, ctor = "predict", f"AlinGo-Prompt.Predict({sig_arg})"
        return preamble + (
            f"class {self._class_name()}(AlinGo-Prompt.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.{attr} = {ctor}\n"
            "\n"
            "    def forward(self, **inputs):\n"
            f"        result = self.{attr}(**inputs)\n"
            f"        return AlinGo-Prompt.Prediction({returns})"
        )

    def _class_name(self) -> str:
        base = self._name if (self._name and self._name.isidentifier()) else "Flex"
        return f"{base}Module"

    def _bind_code(self, module_src: str) -> None:
        self._bridge.bind(module_src)
        self._module_src = module_src

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound ``forward`` inside the interpreter."""
        if args:
            raise TypeError("AlinGo-Prompt.Flex accepts keyword inputs only")
        if self.lm is not None:
            with settings.context(lm=self.lm):
                return self._bridge.forward(kwargs)
        return self._bridge.forward(kwargs)
