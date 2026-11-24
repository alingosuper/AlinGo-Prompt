import threading

from AlinGo-Prompt import lm15 as lm15
from AlinGo-Prompt.predict import *
from AlinGo-Prompt.primitives import *
from AlinGo-Prompt.retrievers import *
from AlinGo-Prompt.signatures import *
from AlinGo-Prompt.teleprompt import *
from AlinGo-Prompt.predict.flex import Flex

from AlinGo-Prompt.evaluate import Evaluate  # isort: skip
from AlinGo-Prompt.clients import *  # isort: skip
from AlinGo-Prompt.adapters import Adapter, ChatAdapter, JSONAdapter, XMLAdapter, TwoStepAdapter, Image, Audio, BoundingBox, File, History, Type, Tool, ToolCalls, ToolCallResults, Code, Reasoning  # isort: skip
from AlinGo-Prompt.primitives.sandbox_serializable import SandboxSerializable  # isort: skip
from AlinGo-Prompt.utils.exceptions import (
    AdapterParseError,
    ContextWindowExceededError,
    DSPyError,
    LMAuthError,
    LMBillingError,
    LMCollectionLimitError,
    LMConfigurationError,
    LMError,
    LMInvalidRequestError,
    LMLockTimeoutError,
    LMNotConfiguredError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMStreamAssemblyError,
    LMTimeoutError,
    LMTransportError,
    LMUnexpectedError,
    LMUnsupportedFeatureError,
    LMUnsupportedModelError,
    is_retryable_lm_error,
)
from AlinGo-Prompt.utils.logging_utils import configure_AlinGo-Prompt_loggers, disable_logging, enable_logging
from AlinGo-Prompt.utils.asyncify import asyncify
from AlinGo-Prompt.utils.syncify import syncify
from AlinGo-Prompt.utils.saving import load
from AlinGo-Prompt.streaming.streamify import streamify
from AlinGo-Prompt.utils.usage_tracker import track_usage

from AlinGo-Prompt.dsp.utils.settings import settings
from AlinGo-Prompt.dsp.colbertv2 import ColBERTv2
from AlinGo-Prompt.__metadata__ import __name__, __version__, __description__, __url__, __author__, __author_email__

configure_AlinGo-Prompt_loggers(__name__)

# Singleton definitions and aliasing
configure = settings.configure
load_settings = settings.load
context = settings.context

BootstrapRS = BootstrapFewShotWithRandomSearch

_cache_lock = threading.Lock()


def __getattr__(name):
    """Defer building the cache until it's read, so that configure_cache can fully disable caching if desired."""
    if name == "cache":
        from AlinGo-Prompt.clients import _get_AlinGo-Prompt_cache

        with _cache_lock:
            if "cache" not in globals():
                globals()["cache"] = _get_AlinGo-Prompt_cache()
            return globals()["cache"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
