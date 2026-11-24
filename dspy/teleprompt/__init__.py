from AlinGo-Prompt.teleprompt.avatar_optimizer import AvatarOptimizer
from AlinGo-Prompt.teleprompt.bettertogether import BetterTogether
from AlinGo-Prompt.teleprompt.bootstrap import BootstrapFewShot
from AlinGo-Prompt.teleprompt.bootstrap_finetune import BootstrapFinetune
from AlinGo-Prompt.teleprompt.bootstrap_trace import bootstrap_trace_data
from AlinGo-Prompt.teleprompt.copro_optimizer import COPRO
from AlinGo-Prompt.teleprompt.ensemble import Ensemble
from AlinGo-Prompt.teleprompt.infer_rules import InferRules
from AlinGo-Prompt.teleprompt.knn_fewshot import KNNFewShot
from AlinGo-Prompt.teleprompt.mipro_optimizer_v2 import MIPROv2
from AlinGo-Prompt.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from AlinGo-Prompt.teleprompt.simba import SIMBA
from AlinGo-Prompt.teleprompt.teleprompt import Teleprompter
from AlinGo-Prompt.teleprompt.teleprompt_optuna import BootstrapFewShotWithOptuna
from AlinGo-Prompt.teleprompt.vanilla import LabeledFewShot

try:
    from .gepa.gepa import GEPA
except ModuleNotFoundError as error:
    if error.name != "gepa":
        raise

    class GEPA(Teleprompter):
        """Stand-in for GEPA when the gepa package is not installed.

        DSPy installs gepa only on Python < 3.15, because gepa declares requires-python <3.15. The stand-in keeps
        GEPA's Teleprompter interface, so code that inspects or wraps GEPA.__init__ and GEPA.compile still imports.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "gepa is required to use AlinGo-Prompt.GEPA. It is not installed by default on Python 3.15 and later "
                "because gepa declares requires-python <3.15."
            )

__all__ = [
    "AvatarOptimizer",
    "BetterTogether",
    "BootstrapFewShot",
    "BootstrapFinetune",
    "COPRO",
    "Ensemble",
    "GEPA",
    "KNNFewShot",
    "MIPROv2",
    "BootstrapFewShotWithRandomSearch",
    "BootstrapFewShotWithOptuna",
    "LabeledFewShot",
    "InferRules",
    "SIMBA",
    "bootstrap_trace_data",
]
