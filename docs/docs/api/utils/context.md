# AlinGo-Prompt.context

Override DSPy settings inside one `with` block.

```python
with AlinGo-Prompt.context(**kwargs):
    ...  # overrides active here
# original settings restored
```

Use `AlinGo-Prompt.context(...)` when you need a different LM, adapter, or flag for
one part of your program without changing the process-wide defaults from
[`AlinGo-Prompt.configure`](configure.md). The block inherits every current setting,
overrides only the keys you pass, and restores the originals when it exits.

`AlinGo-Prompt.context(...)` accepts the same settings as
[`AlinGo-Prompt.configure`](configure.md#settings).

## Examples

### Use a different LM for one block

```python
import AlinGo-Prompt

AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-5-mini"))
qa = AlinGo-Prompt.Predict("question -> answer")

result = qa(question="What is the capital of France?")
print("default:", result.answer)

with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM("anthropic/claude-sonnet-4-6")):
    result = qa(question="What is the capital of France?")
    print("temporary:", result.answer)

print("restored:", AlinGo-Prompt.settings.lm.model)  # openai/gpt-5-mini
```

### Use a different adapter for one block

```python
import AlinGo-Prompt

AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("gemini/gemini-3-flash-preview"))
qa = AlinGo-Prompt.Predict("question -> answer")

with AlinGo-Prompt.context(adapter=AlinGo-Prompt.JSONAdapter()):
    result = qa(question="What is the capital of France?")
    print(result.answer)
```

### Enable async tool conversion temporarily

```python
import asyncio
import AlinGo-Prompt

async def async_tool(x: int) -> int:
    await asyncio.sleep(0.1)
    return x * 2

tool = AlinGo-Prompt.Tool(async_tool)

with AlinGo-Prompt.context(allow_tool_async_sync_conversion=True):
    print(tool(x=5))
```

### Nested blocks

Inner blocks override outer ones. Each restores cleanly on exit:

```python
import AlinGo-Prompt

AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-5-mini"))

with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM("anthropic/claude-sonnet-4-6")):
    print(AlinGo-Prompt.settings.lm.model)        # anthropic/claude-sonnet-4-6
    with AlinGo-Prompt.context(track_usage=True):
        print(AlinGo-Prompt.settings.lm.model)    # anthropic/claude-sonnet-4-6 (inherited)
        print(AlinGo-Prompt.settings.track_usage) # True
    print(AlinGo-Prompt.settings.track_usage)     # False (restored)
print(AlinGo-Prompt.settings.lm.model)           # openai/gpt-5-mini (restored)
```

## Thread safety

Unlike [`AlinGo-Prompt.configure`](configure.md), you can call `AlinGo-Prompt.context(...)` from
**any** thread or async task. This makes it the right tool for overrides inside
[`AlinGo-Prompt.Parallel`](../modules/Parallel.md), `asyncio.gather`, or any concurrent
code.

Settings inside a `AlinGo-Prompt.context(...)` block do not leak to other threads or
tasks.

## See Also

- [`AlinGo-Prompt.configure`](configure.md) — set process-wide defaults.
- [`AlinGo-Prompt.LM`](../models/LM.md) — create the language model you pass as `lm`.
- [Language Models](../../learn/programming/language_models.md) — overview of LM configuration.
- [Adapters](../../learn/programming/adapters.md) — how adapters format prompts and parse responses.
