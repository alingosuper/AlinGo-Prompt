---
sidebar_position: 2
---

# Signatures

When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

**A signature is a declarative specification of input/output behavior of a DSPy module.** Signatures allow you to tell the LM _what_ it needs to do, rather than specify _how_ we should ask the LM to do it.

You're probably familiar with function signatures, which specify the input and output arguments and their types. DSPy signatures are similar, but with a couple of differences. While typical function signatures just _describe_ things, DSPy Signatures _declare and initialize the behavior_ of modules. Moreover, the field names matter in DSPy Signatures. You express semantic roles in plain English: a `question` is different from an `answer`, a `sql_query` is different from `python_code`.

## Why should I use a DSPy Signature?

For modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes). Most people coerce LMs to do tasks by hacking long, brittle prompts. Or by collecting/generating data for fine-tuning. Writing signatures is far more modular, adaptive, and reproducible than hacking at prompts or finetunes. The DSPy compiler will figure out how to build a highly-optimized prompt for your LM (or finetune your small LM) for your signature, on your data, and within your pipeline. In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things and tune the metrics directly.

## **Inline** DSPy Signatures

Signatures can be defined as a short string, with argument names and optional types that define semantic roles for inputs/outputs.

1. Question Answering: `"question -> answer"`, which is equivalent to `"question: str -> answer: str"` as the default type is always `str`

2. Sentiment Classification: `"sentence -> sentiment: bool"`, e.g. `True` if positive

3. Summarization: `"document -> summary"`

Your signatures can also have multiple input/output fields with types:

4. Retrieval-Augmented Question Answering: `"context: list[str], question: str -> answer: str"`

5. Multiple-Choice Question Answering with Reasoning: `"question, choices: list[str] -> reasoning: str, selection: int"`

**Tip:** For fields, any valid variable names work! Field names should be semantically meaningful, but start simple and don't prematurely optimize keywords! Leave that kind of hacking to the DSPy compiler. For example, for summarization, it's probably fine to say `"document -> summary"`, `"text -> gist"`, or `"long_context -> tldr"`.

You can also add instructions to your inline signature, which can use variables at runtime. Use the `instructions` keyword argument to add instructions to your signature.

```python
toxicity = AlinGo-Prompt.Predict(
    AlinGo-Prompt.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
    )
)
comment = "you are beautiful."
toxicity(comment=comment).toxic
```

**Output:**
```text
False
```


### Example A: Sentiment Classification

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

classify = AlinGo-Prompt.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
classify(sentence=sentence).sentiment
```
**Output:**
```text
True
```

### Example B: Summarization

```python
# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = AlinGo-Prompt.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)
```
**Possible Output:**
```text
The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
```

Many DSPy modules (except `AlinGo-Prompt.Predict`) return auxiliary information by expanding your signature under the hood.

For example, `AlinGo-Prompt.ChainOfThought` also adds a `reasoning` field that includes the LM's reasoning before it generates the output `summary`.

```python
print("Reasoning:", response.reasoning)
```
**Possible Output:**
```text
Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
```

## **Class-based** DSPy Signatures

For some advanced tasks, you need more verbose signatures. This is typically to:

1. Clarify something about the nature of the task (expressed below as a `docstring`).

2. Supply hints on the nature of an input field, expressed as a `desc` keyword argument for `AlinGo-Prompt.InputField`.

3. Supply constraints on an output field, expressed as a `desc` keyword argument for `AlinGo-Prompt.OutputField`.

### Example C: Classification

```python
from typing import Literal

class Emotion(AlinGo-Prompt.Signature):
    """Classify emotion."""
    
    sentence: str = AlinGo-Prompt.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = AlinGo-Prompt.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = AlinGo-Prompt.Predict(Emotion)
classify(sentence=sentence)
```
**Possible Output:**
```text
Prediction(
    sentiment='fear'
)
```

**Tip:** There's nothing wrong with specifying your requests to the LM more clearly. Class-based Signatures help you with that. However, don't prematurely tune the keywords of your signature by hand. The DSPy optimizers will likely do a better job (and will transfer better across LMs).

### Example D: A metric that evaluates faithfulness to citations

```python
class CheckCitationFaithfulness(AlinGo-Prompt.Signature):
    """Verify that the text is based on the provided context."""

    context: str = AlinGo-Prompt.InputField(desc="facts here are assumed to be true")
    text: str = AlinGo-Prompt.InputField()
    faithfulness: bool = AlinGo-Prompt.OutputField()
    evidence: dict[str, list[str]] = AlinGo-Prompt.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = AlinGo-Prompt.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
**Possible Output:**
```text
Prediction(
    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.",
    faithfulness=False,
    evidence={'goal_count': ["scored twice for the U's"]}
)
```

### Example E: Multi-modal image classification

```python
class DogPictureSignature(AlinGo-Prompt.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: AlinGo-Prompt.Image = AlinGo-Prompt.InputField(desc="An image of a dog")
    answer: str = AlinGo-Prompt.OutputField(desc="The dog breed of the dog in the image")

image_url = "https://picsum.photos/id/237/200/300"
classify = AlinGo-Prompt.Predict(DogPictureSignature)
classify(image_1=AlinGo-Prompt.Image(image_url))
```

**Possible Output:**

```text
Prediction(
    answer='Labrador Retriever'
)
```

## Type Resolution in Signatures

DSPy signatures support various annotation types:

1. **Basic types** like `str`, `int`, `bool`
2. **Typing module types** like `list[str]`, `dict[str, int]`, `Optional[float]`. `Union[str, int]`
3. **Custom types** defined in your code
4. **Dot notation** for nested types with proper configuration
5. **Special data types** like `AlinGo-Prompt.Image, AlinGo-Prompt.History`

### Working with Custom Types

```python
# Simple custom type
class QueryResult(pydantic.BaseModel):
    text: str
    score: float

signature = AlinGo-Prompt.Signature("query: str -> result: QueryResult")

class MyContainer:
    class Query(pydantic.BaseModel):
        text: str
    class Score(pydantic.BaseModel):
        score: float

signature = AlinGo-Prompt.Signature("query: MyContainer.Query -> score: MyContainer.Score")
```

### Type Checking for Input Fields

DSPy automatically validates that the values you pass to input fields match the types specified in your signature. This works for both inline and class-based signatures.
When there's a type mismatch, DSPy will log a warning.

**Example: Type Mismatch Warning**

```python
# Define a signature expecting an integer as input
class MathSignature(AlinGo-Prompt.Signature):
    """Perform a mathematical operation."""
    number: int = AlinGo-Prompt.InputField()
    result: str = AlinGo-Prompt.OutputField()

predictor = AlinGo-Prompt.Predict(MathSignature)

# This will trigger a warning because we're passing a string instead of an int
predictor(number="42")  # Warning: Type mismatch for field 'number': expected int, but provided value is incompatible
```
**Disabling Type Checking**

If you want to disable type mismatch warnings, you can turn off this feature:

```python
# Disable type mismatch warnings globally
AlinGo-Prompt.configure(warn_on_type_mismatch=False)

predictor = AlinGo-Prompt.Predict("number: int -> result: str")
predictor(number="42")  # No warning
```

### Optional Output Fields

Output fields are required by default: if the LM response does not contain a declared output field, the adapter raises an `AdapterParseError`. An output field becomes optional when it declares a default value, a `default_factory`, or an annotation that allows `None`. When the LM omits an optional field, the parsed prediction falls back to the default, the factory result, or `None`, in that order of precedence.

```python
class Extract(AlinGo-Prompt.Signature):
    """Extract structured information from text."""
    text: str = AlinGo-Prompt.InputField()
    title: str = AlinGo-Prompt.OutputField()                                # required: missing -> AdapterParseError
    note: str | None = AlinGo-Prompt.OutputField(default="No note")         # missing -> "No note"
    tags: list[str] = AlinGo-Prompt.OutputField(default_factory=list)      # missing -> []
    subtitle: str | None = AlinGo-Prompt.OutputField()                      # missing -> None
```

As for input fields: a missing input with a default is filled into the prompt, an input whose annotation allows `None` is omitted from the prompt, and a missing required input logs a warning.

## Using signatures to build modules & compiling them

While signatures are convenient for prototyping with structured inputs/outputs, that's not the only reason to use them!

You should compose multiple signatures into bigger [DSPy modules](modules.md) and [compile these modules into optimized prompts](../optimization/optimizers.md) and finetunes.
