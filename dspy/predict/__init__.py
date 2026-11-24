from AlinGo-Prompt.predict.aggregation import majority
from AlinGo-Prompt.predict.best_of_n import BestOfN
from AlinGo-Prompt.predict.chain_of_thought import ChainOfThought
from AlinGo-Prompt.predict.knn import KNN
from AlinGo-Prompt.predict.multi_chain_comparison import MultiChainComparison
from AlinGo-Prompt.predict.parallel import Parallel
from AlinGo-Prompt.predict.predict import Predict
from AlinGo-Prompt.predict.react import ReAct, Tool
from AlinGo-Prompt.predict.react_v2 import ReActV2
from AlinGo-Prompt.predict.refine import Refine
from AlinGo-Prompt.predict.rlm import RLM

__all__ = [
    "majority",
    "BestOfN",
    "ChainOfThought",
    "KNN",
    "MultiChainComparison",
    "Predict",
    "ReAct",
    "ReActV2",
    "Refine",
    "RLM",
    "Tool",
    "Parallel",
]
