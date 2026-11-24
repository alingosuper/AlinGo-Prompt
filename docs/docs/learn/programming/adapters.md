# Understanding DSPy Adapters

## What are Adapters?

Adapters are the bridge between `AlinGo-Prompt.Predict` and the actual Language Model (LM). When you call a DSPy module, the
adapter takes your signature, user inputs, and other attributes like `demos` (few-shot examples) and converts them
into multi-turn messages that get sent to the LM.

The adapter system is responsible for:

- Translating DSPy signatures into system messages that define the task and request/response structure.
- Formatting input data according to the request structure outlined in DSPy signatures.
- Parsing LM responses back into structured DSPy outputs, such as `AlinGo-Prompt.Prediction` instances.
- Managing conversation history and function calls.
- Converting pre-built DSPy types into LM prompt messages, e.g., `AlinGo-Prompt.Tool`, `AlinGo-Prompt.Image`, etc.

## Configure Adapters

You can use `AlinGo-Prompt.configure(adapter=...)` to choose the adapter for the entire Python process, or
`with AlinGo-Prompt.context(adapter=...):` to only affect a certain namespace.

If no adapter is specified, each `AlinGo-Prompt.Predict.__call__` uses `AlinGo-Prompt.XMLAdapter(use_native_function_calling=True)`. It uses native tool calling when the LM supports it, and text otherwise. The two code snippets below are equivalent:

```python
import AlinGo-Prompt

AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"))

predict = AlinGo-Prompt.Predict("question -> answer")
result = predict(question="What is the capital of France?")
```

```python
import AlinGo-Prompt

AlinGo-Prompt.configure(
    lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"),
    adapter=AlinGo-Prompt.XMLAdapter(use_native_function_calling=True),  # The implicit default
)

predict = AlinGo-Prompt.Predict("question -> answer")
result = predict(question="What is the capital of France?")
```

## Where Adapters Fit in the System

The flow works as follows:

1. The user calls their DSPy agent, typically a `AlinGo-Prompt.Module` with inputs.
2. The inner `AlinGo-Prompt.Predict` is invoked to obtain the LM response.
3. `AlinGo-Prompt.Predict` calls **Adapter.format()**, which converts its signature, inputs, and demos into an lm15 `Prompt`. The adapter builds a canonical `Request` for `AlinGo-Prompt.LM`, which prefers native engines and can use optional LiteLLM compatibility. Both use the same `Request`/`Response` contract.
4. The LM receives the messages and generates a response.
5. **Adapter.parse()** converts the LM response into structured DSPy outputs, as specified in the signature.
6. The caller of `AlinGo-Prompt.Predict` receives the parsed outputs.

You can explicitly call `Adapter.format()` to view the messages sent to the LM. This example chooses `ChatAdapter` to show its marker format.

```python
# Simplified flow example
signature = AlinGo-Prompt.Signature("question -> answer")
inputs = {"question": "What is 2+2?"}
demos = [{"question": "What is 1+1?", "answer": "2"}]

adapter = AlinGo-Prompt.ChatAdapter()
print(adapter.format(signature, demos, inputs))
```

The output should resemble:

```
{'role': 'system', 'content': 'Your input fields are:\n1. `question` (str):\nYour output fields are:\n1. `answer` (str):\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## question ## ]]\n{question}\n\n[[ ## answer ## ]]\n{answer}\n\n[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        Given the fields `question`, produce the fields `answer`.'}
{'role': 'user', 'content': '[[ ## question ## ]]\nWhat is 1+1?'}
{'role': 'assistant', 'content': '[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n'}
{'role': 'user', 'content': '[[ ## question ## ]]\nWhat is 2+2?\n\nRespond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.'}
```

You can also only fetch the system message by calling `adapter.format_system_message(signature)`.

```python
import AlinGo-Prompt

signature = AlinGo-Prompt.Signature("question -> answer")
system_message = AlinGo-Prompt.ChatAdapter().format_system_message(signature)
print(system_message)
```

The output should resemble:

```
Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## question ## ]]
{question}
[[ ## answer ## ]]
{answer}
[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Given the fields `question`, produce the fields `answer`.
```

## Types of Adapters

DSPy offers several adapter types, each tailored for specific use cases:

### XMLAdapter

**XMLAdapter** is the implicit default, with native tool calling enabled. It delimits fields with `<field_name>...</field_name>` tags and uses nested tags for objects and `<item>` for lists. DSPy parses and validates the response; XML formatting does not provide provider-enforced structured output.

For example, a text response to `question -> answer` looks like:

```xml
<answer>
Paris
</answer>
```

### ChatAdapter

**ChatAdapter** is an explicit alternative that uses a field-based format with special markers.

#### Format Structure

ChatAdapter uses `[[ ## field_name ## ]]` markers to delineate fields. For fields of non-primitive Python types, it includes the JSON schema of the type. Below, we use `AlinGo-Prompt.inspect_history()` to display the formatted messages by `AlinGo-Prompt.ChatAdapter` clearly.

```python
import AlinGo-Prompt
import pydantic

AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"), adapter=AlinGo-Prompt.ChatAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(AlinGo-Prompt.Signature):
    """Get news about the given science field"""

    science_field: str = AlinGo-Prompt.InputField()
    year: int = AlinGo-Prompt.InputField()
    num_of_outputs: int = AlinGo-Prompt.InputField()
    news: list[ScienceNews] = AlinGo-Prompt.OutputField(desc="science news")


predict = AlinGo-Prompt.Predict(NewsQA)
predict(science_field="Computer Theory", year=2022, num_of_outputs=1)
AlinGo-Prompt.inspect_history()
```

The output looks like:

```
[2025-08-15T22:24:29.378666]

System message:

Your input fields are:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
Your output fields are:
1. `news` (list[ScienceNews]): science news
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

[[ ## news ## ]]
{news}        # note: the value you produce must adhere to the JSON schema: {"type": "array", "$defs": {"ScienceNews": {"type": "object", "properties": {"scientists_involved": {"type": "array", "items": {"type": "string"}, "title": "Scientists Involved"}, "text": {"type": "string", "title": "Text"}}, "required": ["text", "scientists_involved"], "title": "ScienceNews"}}, "items": {"$ref": "#/$defs/ScienceNews"}}

[[ ## completed ## ]]
In adhering to this structure, your objective is:
        Get news about the given science field


User message:

[[ ## science_field ## ]]
Computer Theory

[[ ## year ## ]]
2022

[[ ## num_of_outputs ## ]]
1

Respond with the corresponding output fields, starting with the field `[[ ## news ## ]]` (must be formatted as a valid Python list[ScienceNews]), and then ending with the marker for `[[ ## completed ## ]]`.


Response:

[[ ## news ## ]]
[
    {
        "scientists_involved": ["John Doe", "Jane Smith"],
        "text": "In 2022, researchers made significant advancements in quantum computing algorithms, demonstrating their potential to solve complex problems faster than classical computers. This breakthrough could revolutionize fields such as cryptography and optimization."
    }
]

[[ ## completed ## ]]
```

!!! info "Practice: locate Signature information in the printed LM history"

    Try adjusting the signature, and observe how the changes are reflected in the printed LM message.


Each field is preceded by a marker `[[ ## field_name ## ]]`. If an output field has non-primitive types, the instruction includes the type's JSON schema, and the output is formatted accordingly. Because the output field is structured as defined by ChatAdapter, it can be automatically parsed into structured data.

#### When to Use ChatAdapter

`ChatAdapter` offers the following advantages:

- **Universal compatibility**: Works with all language models, though smaller models may generate responses that do not match the required format.
- **Fallback protection**: If `ChatAdapter` fails, it automatically retries with `JSONAdapter`.

In general, `ChatAdapter` is a reliable choice if you don't have specific requirements.

#### When Not to Use ChatAdapter

Avoid using `ChatAdapter` if you are:

- **Latency sensitive**: `ChatAdapter` includes more boilerplate output tokens compared to other adapters, so if you're building a system sensitive to latency, consider using a different adapter.

### JSONAdapter

**JSONAdapter** prompts the LM to return JSON data containing all output fields as specified in the signature. It is effective for models that support structured output via the `response_format` parameter, leveraging native JSON generation capabilities for more reliable parsing.

#### Format Structure

The input part of the prompt formatted by `JSONAdapter` is similar to `ChatAdapter`, but the output part differs, as shown below:

```python
import AlinGo-Prompt
import pydantic

AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"), adapter=AlinGo-Prompt.JSONAdapter())


class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]


class NewsQA(AlinGo-Prompt.Signature):
    """Get news about the given science field"""

    science_field: str = AlinGo-Prompt.InputField()
    year: int = AlinGo-Prompt.InputField()
    num_of_outputs: int = AlinGo-Prompt.InputField()
    news: list[ScienceNews] = AlinGo-Prompt.OutputField(desc="science news")


predict = AlinGo-Prompt.Predict(NewsQA)
predict(science_field="Computer Theory", year=2022, num_of_outputs=1)
AlinGo-Prompt.inspect_history()
```

```
System message:

Your input fields are:
1. `science_field` (str):
2. `year` (int):
3. `num_of_outputs` (int):
Your output fields are:
1. `news` (list[ScienceNews]): science news
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## science_field ## ]]
{science_field}

[[ ## year ## ]]
{year}

[[ ## num_of_outputs ## ]]
{num_of_outputs}

Outputs will be a JSON object with the following fields.

{
  "news": "{news}        # note: the value you produce must adhere to the JSON schema: {\"type\": \"array\", \"$defs\": {\"ScienceNews\": {\"type\": \"object\", \"properties\": {\"scientists_involved\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"title\": \"Scientists Involved\"}, \"text\": {\"type\": \"string\", \"title\": \"Text\"}}, \"required\": [\"text\", \"scientists_involved\"], \"title\": \"ScienceNews\"}}, \"items\": {\"$ref\": \"#/$defs/ScienceNews\"}}"
}
In adhering to this structure, your objective is:
        Get news about the given science field


User message:

[[ ## science_field ## ]]
Computer Theory

[[ ## year ## ]]
2022

[[ ## num_of_outputs ## ]]
1

Respond with a JSON object in the following order of fields: `news` (must be formatted as a valid Python list[ScienceNews]).


Response:

{
  "news": [
    {
      "text": "In 2022, researchers made significant advancements in quantum computing algorithms, demonstrating that quantum systems can outperform classical computers in specific tasks. This breakthrough could revolutionize fields such as cryptography and complex system simulations.",
      "scientists_involved": [
        "Dr. Alice Smith",
        "Dr. Bob Johnson",
        "Dr. Carol Lee"
      ]
    }
  ]
}
```

#### When to Use JSONAdapter

`JSONAdapter` is good at:

- **Structured output support**: When the model supports the `response_format` parameter.
- **Low latency**: Minimal boilerplate in the LM response results in faster responses.

#### When Not to Use JSONAdapter

Avoid using `JSONAdapter` if you are:

- Using a model that does not natively support structured output, such as a small open-source model hosted on Ollama.

## Summary

Adapters are a crucial component of DSPy that bridge the gap between structured DSPy signatures and language model APIs.
Understanding when and how to use different adapters will help you build more reliable and efficient DSPy programs.
