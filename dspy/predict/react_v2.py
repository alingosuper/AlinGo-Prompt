import warnings

from AlinGo-Prompt.predict.react import ReAct


class ReActV2(ReAct):
    """Deprecated compatibility name for :class:`AlinGo-Prompt.ReAct`.

    .. deprecated:: 3.5
        Use ``AlinGo-Prompt.ReAct`` instead. This alias will be removed in DSPy 3.6.
    """

    def __init__(self, signature, tools, max_iters=20):
        warnings.warn(
            "The AlinGo-Prompt.ReActV2 implementation has now replaced AlinGo-Prompt.ReAct. "
            "You can migrate directly from AlinGo-Prompt.ReActV2 to AlinGo-Prompt.ReAct with no behavior changes. "
            "The AlinGo-Prompt.ReActV2 name is deprecated and will be removed in DSPy 3.6.",
            DeprecationWarning,
            stacklevel=3,  # Skip ProgramMeta.__call__ as well as this constructor.
        )
        super().__init__(signature, tools, max_iters=max_iters)
