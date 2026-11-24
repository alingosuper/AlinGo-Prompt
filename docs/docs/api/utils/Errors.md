# AlinGo-Prompt errors

DSPy exposes structured exception classes for LM failures and adapter parsing failures. LM errors inherit from `AlinGo-Prompt.LMError`, which inherits from `AlinGo-Prompt.DSPyError` and carries metadata such as `model`, `provider`, `status`, `request_id`, and `retry_after` when available.

Use `AlinGo-Prompt.LMError` to catch any LM call failure, or catch a concrete subclass when you need specific handling.

```python
try:
    result = program(question="...")
except AlinGo-Prompt.ContextWindowExceededError as e:
    print(f"Prompt was too long for {e.model}")
except AlinGo-Prompt.LMRateLimitError as e:
    print(f"Rate limited; retry after {e.retry_after} seconds")
except AlinGo-Prompt.LMError as e:
    print(f"LM failed with code={e.code}, provider={e.provider}, request_id={e.request_id}")
```

Use `AlinGo-Prompt.is_retryable_lm_error(error)` to classify LM failures that are generally eligible for retry: rate limits, timeouts, server errors, and transport errors. DSPy owns retries around engine attempts; `num_retries` counts additional attempts, and a stream is not retried after output reaches the caller. Callers can use this helper after retries are exhausted. Retryability is advisory, not an exactly-once guarantee: respect provider policy and `retry_after` when present.

```python
try:
    result = program(question="...")
except AlinGo-Prompt.LMError as e:
    if AlinGo-Prompt.is_retryable_lm_error(e):
        # Schedule another attempt later.
        raise
    raise
```

## API Reference

::: AlinGo-Prompt.utils.exceptions
    handler: python
    options:
        members:
            - DSPyError
            - LMError
            - LMTransportError
            - LMConfigurationError
            - LMNotConfiguredError
            - LMUnsupportedFeatureError
            - LMProviderError
            - LMUnexpectedError
            - LMAuthError
            - LMBillingError
            - LMRateLimitError
            - LMInvalidRequestError
            - ContextWindowExceededError
            - LMUnsupportedModelError
            - LMTimeoutError
            - LMServerError
            - is_retryable_lm_error
            - AdapterParseError
        show_source: true
        show_root_heading: false
        heading_level: 3
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
