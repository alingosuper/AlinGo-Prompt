import enum
from typing import Literal
from unittest import mock

import pydantic
import pytest

import AlinGo-Prompt
from AlinGo-Prompt.clients.lm15_boundary import request_kwargs
from AlinGo-Prompt.lm15 import ImagePart, Usage
from AlinGo-Prompt.primitives.repl_types import REPLEntry
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.test_utils.engines import make_response, recording_lm


def test_json_adapter_format_exact_messages_for_simple_signature():
    class StringSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(),
        StringSignature,
        demos=[],
        inputs={"question": "What is the capital of France?"},
    )

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.

{
  "answer": "{answer}"
}
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`.""",
        },
        {
            "role": "user",
            "content": """[[ ## question ## ]]
What is the capital of France?

Respond with a JSON object in the following order of fields: `answer`.""",
        },
    ]


def test_json_adapter_format_exact_messages_with_demo_and_typed_output():
    class MultiAnswer(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()
        confidence: float = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(),
        MultiAnswer,
        demos=[{"question": "Q1", "answer": "A1", "confidence": 0.9}],
        inputs={"question": "Q2"},
    )

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answer` (str):\x20
2. `confidence` (float):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.

{
  "answer": "{answer}",
  "confidence": "{confidence}        # note: the value you produce must be a single float value"
}
In adhering to this structure, your objective is:\x20
        Given the fields `question`, produce the fields `answer`, `confidence`.""",
        },
        {"role": "user", "content": """[[ ## question ## ]]
Q1"""},
        {
            "role": "assistant",
            "content": """{
  "answer": "A1",
  "confidence": 0.9
}""",
        },
        {
            "role": "user",
            "content": """[[ ## question ## ]]
Q2

Respond with a JSON object in the following order of fields: `answer`, then `confidence` (must be formatted as a valid Python float).""",
        },
    ]


def test_json_adapter_format_exact_messages_with_described_and_bool_outputs():
    class TestSignature(AlinGo-Prompt.Signature):
        input1: str = AlinGo-Prompt.InputField()
        output1: str = AlinGo-Prompt.OutputField(desc="String output field")
        output2: bool = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(), TestSignature, [], {"input1": "Test input"})

    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

    assert messages == [
        {
            "role": "system",
            "content": """Your input fields are:
1. `input1` (str):
Your output fields are:
1. `output1` (str): String output field
2. `output2` (bool):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## input1 ## ]]
{input1}

Outputs will be a JSON object with the following fields.

{
  "output1": "{output1}",
  "output2": "{output2}        # note: the value you produce must be True or False"
}
In adhering to this structure, your objective is:\x20
        Given the fields `input1`, produce the fields `output1`, `output2`.""",
        },
        {
            "role": "user",
            "content": """[[ ## input1 ## ]]
Test input

Respond with a JSON object in the following order of fields: `output1`, then `output2` (must be formatted as a valid Python bool).""",
        },
    ]


def test_json_adapter_format_exact_messages_with_history_demo_pydantic_tools_and_image():
    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
        return query

    class Location(pydantic.BaseModel):
        city: str
        country: str

    class Profile(pydantic.BaseModel):
        name: str
        location: Location
        interests: list[str]

    class AnswerCard(pydantic.BaseModel):
        answer: str
        sources: list[str]

    class RichRenderingSignature(AlinGo-Prompt.Signature):
        """Answer using all supplied context."""

        history: AlinGo-Prompt.History = AlinGo-Prompt.InputField()
        image: AlinGo-Prompt.Image = AlinGo-Prompt.InputField()
        tools: list[AlinGo-Prompt.Tool] = AlinGo-Prompt.InputField()
        profile: Profile = AlinGo-Prompt.InputField()
        question: str = AlinGo-Prompt.InputField()
        answer: AnswerCard = AlinGo-Prompt.OutputField()

    tool = AlinGo-Prompt.Tool(search)
    demo_profile = Profile(
        name="Ada",
        location=Location(city="London", country="UK"),
        interests=["math", "machines"],
    )
    current_profile = Profile(
        name="Grace",
        location=Location(city="Arlington", country="USA"),
        interests=["compilers", "navy"],
    )
    history = AlinGo-Prompt.History(
        messages=[
            {
                "profile": demo_profile,
                "question": "Who is Ada?",
                "answer": AnswerCard(answer="Ada is a mathematician.", sources=["memory"]),
            }
        ]
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(),
        RichRenderingSignature,
        demos=[
            {
                "image": AlinGo-Prompt.Image("https://example.com/demo.png"),
                "tools": [tool],
                "profile": demo_profile,
                "question": "What should we mention?",
                "answer": AnswerCard(answer="Mention analytical engines.", sources=["demo"]),
            }
        ],
        inputs={
            "history": history,
            "image": AlinGo-Prompt.Image("https://example.com/current.png"),
            "tools": [tool],
            "profile": current_profile,
            "question": "What should the answer include?",
        },
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `history` (History): \n'
                 '2. `image` (Image): \n'
                 '3. `tools` (list[Tool]): \n'
                 '4. `profile` (Profile): \n'
                 '5. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `answer` (AnswerCard):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## history ## ]]\n'
                 '{history}\n'
                 '\n'
                 '[[ ## image ## ]]\n'
                 '{image}\n'
                 '\n'
                 '[[ ## tools ## ]]\n'
                 '{tools}\n'
                 '\n'
                 '[[ ## profile ## ]]\n'
                 '{profile}\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "answer": "{answer}        # note: the value you produce must adhere to the JSON '
                 'schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"answer\\": {\\"type\\": '
                 '\\"string\\", \\"title\\": \\"Answer\\"}, \\"sources\\": {\\"type\\": \\"array\\", '
                 '\\"items\\": {\\"type\\": \\"string\\"}, \\"title\\": \\"Sources\\"}}, '
                 '\\"required\\": [\\"answer\\", \\"sources\\"], \\"title\\": \\"AnswerCard\\"}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Answer using all supplied context.'},
     {"role": "user",
      "content": [{"type": "text",
                   "text": "This is an example of the task, though some input or output fields are not "
                           "supplied.\n"
                           "\n"
                           "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Ada", "location": {"city": "London", "country": "UK"}, '
                           '"interests": ["math", "machines"]}\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should we mention?'}]},
     {"role": "assistant",
      "content": '{\n'
                 '  "answer": {\n'
                 '    "answer": "Mention analytical engines.",\n'
                 '    "sources": [\n'
                 '      "demo"\n'
                 '    ]\n'
                 '  }\n'
                 '}'},
     {"role": "user",
      "content": '[[ ## profile ## ]]\n'
                 '{"name": "Ada", "location": {"city": "London", "country": "UK"}, "interests": '
                 '["math", "machines"]}\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 'Who is Ada?\n\n'
                 'Respond with a JSON object in the following order of fields: `answer` '
                 '(must be formatted as a valid Python AnswerCard).'},
     {"role": "assistant",
      "content": '{\n'
                 '  "answer": {\n'
                 '    "answer": "Ada is a mathematician.",\n'
                 '    "sources": [\n'
                 '      "memory"\n'
                 '    ]\n'
                 '  }\n'
                 '}'},
     {"role": "user",
      "content": [{"type": "text", "text": "[[ ## image ## ]]\n"},
                  {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                  {"type": "text",
                   "text": '\n'
                           '\n'
                           '[[ ## tools ## ]]\n'
                           '["search, whose description is <desc>Search for documents.</desc>. It '
                           "takes arguments {'query': {'type': 'string'}, 'k': {'type': 'integer', "
                           '\'default\': 3}}."]\n'
                           '\n'
                           '[[ ## profile ## ]]\n'
                           '{"name": "Grace", "location": {"city": "Arlington", "country": "USA"}, '
                           '"interests": ["compilers", "navy"]}\n'
                           '\n'
                           '[[ ## question ## ]]\n'
                           'What should the answer include?\n'
                           '\n'
                           'Respond with a JSON object in the following order of fields: `answer` '
                           '(must be formatted as a valid Python AnswerCard).'}]}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs

def test_json_adapter_format_exact_messages_with_int_and_mapping_outputs():
    class IntDictSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        count: int = AlinGo-Prompt.OutputField()
        metadata: dict[str, int] = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(), IntDictSignature, [], {"question": "Count things"})

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `count` (int): \n'
                 '2. `metadata` (dict[str, int]):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "count": "{count}        # note: the value you produce must be a single int '
                 'value",\n'
                 '  "metadata": "{metadata}        # note: the value you produce must adhere to the '
                 'JSON schema: {\\"type\\": \\"object\\", \\"additionalProperties\\": {\\"type\\": '
                 '\\"integer\\"}}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `count`, `metadata`.'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Count things\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `count` (must be "
                 "formatted as a valid Python int), then `metadata` (must be formatted as a valid "
                 "Python dict[str, int])."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_literal_and_enum_outputs():
    class Label(enum.Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"

    class LiteralEnumSignature(AlinGo-Prompt.Signature):
        text: str = AlinGo-Prompt.InputField()
        decision: Literal["accept", "reject"] = AlinGo-Prompt.OutputField()
        label: Label = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(), LiteralEnumSignature, [], {"text": "Looks good"})

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `text` (str):\n'
                 'Your output fields are:\n'
                 "1. `decision` (Literal['accept', 'reject']): \n"
                 '2. `label` (Label):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## text ## ]]\n'
                 '{text}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "decision": "{decision}        # note: the value you produce must exactly match '
                 '(no extra characters) one of: accept; reject",\n'
                 '  "label": "{label}        # note: the value you produce must be one of: positive; '
                 'negative"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `text`, produce the fields `decision`, `label`.'},
     {"role": "user",
      "content": "[[ ## text ## ]]\n"
                 "Looks good\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `decision` (must be "
                 "formatted as a valid Python Literal['accept', 'reject']), then `label` (must be "
                 "formatted as a valid Python Label)."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_nested_pydantic_output():
    class JsonNestedAddress(pydantic.BaseModel):
        city: str
        country: str

    class JsonNestedSummary(pydantic.BaseModel):
        title: str
        address: JsonNestedAddress
        scores: list[float]

    class PydanticSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        summary: JsonNestedSummary = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(), PydanticSignature, [], {"question": "Summarize"})

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `summary` (JsonNestedSummary):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "summary": "{summary}        # note: the value you produce must adhere to the JSON '
                 'schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"JsonNestedAddress\\": '
                 '{\\"type\\": \\"object\\", \\"properties\\": {\\"city\\": {\\"type\\": \\"string\\", '
                 '\\"title\\": \\"City\\"}, \\"country\\": {\\"type\\": \\"string\\", \\"title\\": '
                 '\\"Country\\"}}, \\"required\\": [\\"city\\", \\"country\\"], \\"title\\": '
                 '\\"JsonNestedAddress\\"}}, \\"properties\\": {\\"address\\": {\\"$ref\\": '
                 '\\"#/$defs/JsonNestedAddress\\"}, \\"scores\\": {\\"type\\": \\"array\\", '
                 '\\"items\\": {\\"type\\": \\"number\\"}, \\"title\\": \\"Scores\\"}, \\"title\\": '
                 '{\\"type\\": \\"string\\", \\"title\\": \\"Title\\"}}, \\"required\\": [\\"title\\", '
                 '\\"address\\", \\"scores\\"], \\"title\\": \\"JsonNestedSummary\\"}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `summary`.'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Summarize\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `summary` (must be "
                 "formatted as a valid Python JsonNestedSummary)."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_with_incomplete_demo():
    class IncompleteDemoSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        context: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()
        score: float = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(),
        IncompleteDemoSignature,
        [{"question": "Q1", "answer": "A1"}],
        {"question": "Q2", "context": "C2"},
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str): \n'
                 '2. `context` (str):\n'
                 'Your output fields are:\n'
                 '1. `answer` (str): \n'
                 '2. `score` (float):\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 '[[ ## context ## ]]\n'
                 '{context}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "answer": "{answer}",\n'
                 '  "score": "{score}        # note: the value you produce must be a single float '
                 'value"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, `context`, produce the fields `answer`, '
                 '`score`.'},
     {"role": "user",
      "content": "This is an example of the task, though some input or output fields are not "
                 "supplied.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "Q1"},
     {"role": "assistant",
      "content": '{\n  "answer": "A1",\n  "score": "Not supplied for this particular example. "\n}'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 "[[ ## context ## ]]\n"
                 "C2\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `answer`, then `score` "
                 "(must be formatted as a valid Python float)."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_messages_and_lm_kwargs_with_native_tool_calling():
    class FunctionCallingLM(AlinGo-Prompt.utils.DummyLM):
        @property
        def supports_function_calling(self):
            return True

    def search(query: str, k: int = 3) -> str:
        """Search for documents."""
        return query

    class NativeToolSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        tools: list[AlinGo-Prompt.Tool] = AlinGo-Prompt.InputField()
        tool_calls: AlinGo-Prompt.ToolCalls = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        AlinGo-Prompt.JSONAdapter(use_native_function_calling=True),
        NativeToolSignature,
        [],
        {"question": "Q?", "tools": [AlinGo-Prompt.Tool(search)]},
        lm=FunctionCallingLM([{}]),
    )

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "Inputs will have the following structure:\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "Outputs will be a JSON object with the following fields.\n"
                 "\n"
                 "{}\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, `tools`, produce the fields `tool_calls`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q?\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: ."}]
    assert messages == expected_messages
    expected_lm_kwargs = {"tools": [{"type": "function",
                "function": {"name": "search",
                             "description": "Search for documents.",
                             "parameters": {"type": "object",
                                            "properties": {"query": {"type": "string"},
                                                           "k": {"type": "integer", "default": 3}},
                                            "required": ["query"]}}}]}
    assert lm_kwargs == expected_lm_kwargs

def test_json_adapter_format_messages_and_lm_kwargs_with_repl_history_non_reasoning_lm():
    class NonReasoningLM(AlinGo-Prompt.utils.DummyLM):
        @property
        def supports_reasoning(self):
            return False

    class REPLHistorySignature(AlinGo-Prompt.Signature):
        history: AlinGo-Prompt.History = AlinGo-Prompt.InputField()
        variables_info: str = AlinGo-Prompt.InputField()
        reasoning: AlinGo-Prompt.Reasoning = AlinGo-Prompt.OutputField()
        code: str = AlinGo-Prompt.OutputField()

    history = AlinGo-Prompt.History(
        messages=[
            {
                "variables_info": "INFO_1",
                "repl_entry": REPLEntry(
                    reasoning="First explore the data",
                    code="x = 10\nprint(f'x = {x}')",
                    output="x = 20",
                    max_output_chars=100,
                ),
            }
        ]
    )

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        AlinGo-Prompt.JSONAdapter(),
        REPLHistorySignature,
        [],
        {"variables_info": "INFO_2", "history": history},
        lm=NonReasoningLM([{}]),
    )

    assert messages[1]["role"] == "user"
    assert "INFO_1" in messages[1]["content"]
    assert messages[2] == {
        "role": "assistant",
        "content": '{\n  "reasoning": "First explore the data",\n  "code": "x = 10\\nprint(f\'x = {x}\')"\n}',
    }
    assert messages[3] == {
        "role": "user",
        "content": "[[ ## repl_output ## ]]\nx = 20",
    }
    assert messages[4]["role"] == "user"
    assert "INFO_2" in messages[4]["content"]
    assert "history" not in messages[4]["content"]
    assert "reasoning_effort" not in lm_kwargs


def test_json_adapter_format_messages_and_lm_kwargs_with_repl_history_reasoning_lm():
    class ReasoningLM(AlinGo-Prompt.utils.DummyLM):
        @property
        def supports_reasoning(self):
            return True

    class REPLHistorySignature(AlinGo-Prompt.Signature):
        history: AlinGo-Prompt.History = AlinGo-Prompt.InputField()
        variables_info: str = AlinGo-Prompt.InputField()
        reasoning: AlinGo-Prompt.Reasoning = AlinGo-Prompt.OutputField()
        code: str = AlinGo-Prompt.OutputField()

    history = AlinGo-Prompt.History(
        messages=[
            {
                "variables_info": "INFO_1",
                "repl_entry": REPLEntry(
                    reasoning="First explore the data",
                    code="x = 10\nprint(f'x = {x}')",
                    output="x = 20",
                    max_output_chars=100,
                ),
            }
        ]
    )

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        AlinGo-Prompt.JSONAdapter(),
        REPLHistorySignature,
        [],
        {"variables_info": "INFO_2", "history": history},
        lm=ReasoningLM([{}]),
    )

    assert messages[1]["role"] == "user"
    assert "INFO_1" in messages[1]["content"]
    assert messages[2] == {
        "role": "assistant",
        "content": '{\n  "code": "x = 10\\nprint(f\'x = {x}\')"\n}',
    }
    assert messages[3] == {
        "role": "user",
        "content": "[[ ## repl_output ## ]]\nx = 20",
    }
    assert messages[4]["role"] == "user"
    assert "INFO_2" in messages[4]["content"]
    assert "history" not in messages[4]["content"]
    assert "reasoning_effort" in lm_kwargs


def test_json_adapter_format_messages_and_lm_kwargs_with_repl_history_and_truncation():
    class ReasoningLM(AlinGo-Prompt.utils.DummyLM):
        @property
        def supports_reasoning(self):
            return True

    class REPLHistorySignature(AlinGo-Prompt.Signature):
        history: AlinGo-Prompt.History = AlinGo-Prompt.InputField()
        variables_info: str = AlinGo-Prompt.InputField()
        reasoning: AlinGo-Prompt.Reasoning = AlinGo-Prompt.OutputField()
        code: str = AlinGo-Prompt.OutputField()

    history = AlinGo-Prompt.History(
        messages=[
            {
                "variables_info": "INFO_1",
                "repl_entry": REPLEntry(
                    reasoning="First explore the data",
                    code="x = 10\nprint(f'x = {x}')",
                    output="x = 20",
                    max_output_chars=4,
                ),
            }
        ]
    )

    messages, _ = format_messages_and_lm_kwargs(
        AlinGo-Prompt.JSONAdapter(),
        REPLHistorySignature,
        [],
        {"variables_info": "INFO_2", "history": history},
        lm=ReasoningLM([{}]),
    )

    assert messages[1]["role"] == "user"
    assert "INFO_1" in messages[1]["content"]
    assert messages[2] == {
        "role": "assistant",
        "content": '{\n  "code": "x = 10\\nprint(f\'x = {x}\')"\n}',
    }
    assert messages[3] == {
        "role": "user",
        "content": "[[ ## repl_output ## ]]\nx \n\n... (2 characters omitted) ...\n\n20",
    }
    assert messages[4]["role"] == "user"
    assert "INFO_2" in messages[4]["content"]
    assert "history" not in messages[4]["content"]


def test_json_adapter_format_exact_messages_with_tool_calls_output_demo():
    class ToolCallsSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        tool_calls: AlinGo-Prompt.ToolCalls = AlinGo-Prompt.OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(AlinGo-Prompt.JSONAdapter(use_native_function_calling=False),
        ToolCallsSignature,
        [{"question": "Q1", "tool_calls": AlinGo-Prompt.ToolCalls.from_dict_list([{"name": "search", "args": {"query": "cats"}}])}],
        {"question": "Q2"},
    )

    expected_messages = [{"role": "system",
      "content": 'Your input fields are:\n'
                 '1. `question` (str):\n'
                 'Your output fields are:\n'
                 '1. `tool_calls` (ToolCalls): \n'
                 '    Type description of ToolCalls: Tool calls must be a JSON object with `tool_calls`, '
                 'a list of calls. Each call must include `name` and `args`. Example: {"tool_calls": '
                 '[{"name": "search", "args": {"query": "cats"}}]}\n'
                 'All interactions will be structured in the following way, with the appropriate '
                 'values filled in.\n'
                 '\n'
                 'Inputs will have the following structure:\n'
                 '\n'
                 '[[ ## question ## ]]\n'
                 '{question}\n'
                 '\n'
                 'Outputs will be a JSON object with the following fields.\n'
                 '\n'
                 '{\n'
                 '  "tool_calls": "{tool_calls}        # note: the value you produce must adhere to '
                 'the JSON schema: {\\"type\\": \\"object\\", \\"$defs\\": {\\"ToolCall\\": '
                 '{\\"type\\": \\"object\\", \\"properties\\": {\\"args\\": {\\"type\\": \\"object\\", '
                 '\\"additionalProperties\\": true, \\"title\\": \\"Args\\"}, \\"name\\": {\\"type\\": '
                 '\\"string\\", \\"title\\": \\"Name\\"}}, \\"required\\": [\\"name\\", \\"args\\"], '
                 '\\"title\\": \\"ToolCall\\"}}, \\"properties\\": {\\"tool_calls\\": {\\"type\\": '
                 '\\"array\\", \\"items\\": {\\"$ref\\": \\"#/$defs/ToolCall\\"}, \\"title\\": \\"Tool '
                 'Calls\\"}}, \\"required\\": [\\"tool_calls\\"], \\"title\\": \\"ToolCalls\\"}"\n'
                 '}\n'
                 'In adhering to this structure, your objective is: \n'
                 '        Given the fields `question`, produce the fields `tool_calls`.'},
     {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
     {"role": "assistant",
      "content": '{\n'
                 '  "tool_calls": {\n'
                 '    "tool_calls": [\n'
                 '      {\n'
                 '        "name": "search",\n'
                 '        "args": {\n'
                 '          "query": "cats"\n'
                 '        }\n'
                 '      }\n'
                 '    ]\n'
                 '  }\n'
                 '}'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 'Respond with a JSON object in the following order of fields: `tool_calls` (must be '
                 'a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).'}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_json_adapter_format_exact_non_native_tool_result_history_field():
    def search(query: str) -> str:
        return query

    class ToolHistorySignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        history: AlinGo-Prompt.History = AlinGo-Prompt.InputField()
        tools: list[AlinGo-Prompt.Tool] = AlinGo-Prompt.InputField()
        next_thought: str = AlinGo-Prompt.OutputField()
        tool_calls: AlinGo-Prompt.ToolCalls = AlinGo-Prompt.OutputField()

    tool_call = AlinGo-Prompt.ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    tool_call_results = AlinGo-Prompt.ToolCallResults.from_tool_calls_and_values([tool_call], ["cat"])

    messages, _lm_kwargs = format_messages_and_lm_kwargs(
        AlinGo-Prompt.JSONAdapter(use_native_function_calling=False),
        ToolHistorySignature,
        [],
        {
            "question": "Q2",
            "history": AlinGo-Prompt.History(
                messages=[
                    {
                        "question": "Q1",
                        "next_thought": "I should search.",
                        "tool_calls": AlinGo-Prompt.ToolCalls(tool_calls=[tool_call], tool_call_results=tool_call_results),
                    }
                ]
            ),
            "tools": [AlinGo-Prompt.Tool(search)],
        },
    )

    assert messages[3]["content"] == (
        "[[ ## tool_call_results ## ]]\n"
        '{"tool_call_results": [{"call_id": "call_1", "name": "search", "value": "cat", "is_error": false}]}'
    )
    assert messages[4]["content"] == (
        "[[ ## question ## ]]\n"
        "Q2\n"
        "\n"
        "[[ ## tools ## ]]\n"
        '["search. It takes arguments {\'query\': {\'type\': \'string\'}}."]\n'
        "\n"
        "Respond with a JSON object in the following order of fields: `next_thought`, then "
        '`tool_calls` (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]}).'
    )


def test_json_adapter_passes_structured_output_when_supported_by_model():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1", ge=0, le=10)
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(AlinGo-Prompt.Signature):
        input1: str = AlinGo-Prompt.InputField()
        output1: str = AlinGo-Prompt.OutputField()  # Description intentionally left blank
        output2: bool = AlinGo-Prompt.OutputField(desc="Boolean output field")
        output3: OutputField3 = AlinGo-Prompt.OutputField(desc="Nested output field")
        output4_unannotated = AlinGo-Prompt.OutputField(desc="Unannotated output field")

    program = AlinGo-Prompt.Predict(TestSignature)

    # Configure DSPy to use an OpenAI LM that supports structured outputs
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM(model="openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter())
    reply = make_response('{"output1": "a", "output2": true, "output3": {"subfield1": 1, "subfield2": 2.0}, "output4_unannotated": "x"}')
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete", return_value=reply) as mock_completion:
        program(input1="Test input")

    def clean_schema_extra(field_name, field_info):
        attrs = dict(field_info.__repr_args__())
        if "json_schema_extra" in attrs:
            attrs["json_schema_extra"] = {
                k: v
                for k, v in attrs["json_schema_extra"].items()
                if k != "__AlinGo-Prompt_field_type" and not (k == "desc" and v == f"${{{field_name}}}")
            }
        return attrs

    mock_completion.assert_called_once()
    call_kwargs = request_kwargs(mock_completion.call_args.args[0], "chat")
    response_format = call_kwargs.get("response_format")
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    schema = response_format["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"].keys() == {"output1", "output2", "output3", "output4_unannotated"}
    assert schema["schema"]["additionalProperties"] is False


def test_json_adapter_not_using_structured_outputs_when_not_supported_by_model():
    class TestSignature(AlinGo-Prompt.Signature):
        input1: str = AlinGo-Prompt.InputField()
        output1: str = AlinGo-Prompt.OutputField()
        output2: bool = AlinGo-Prompt.OutputField()

    program = AlinGo-Prompt.Predict(TestSignature)

    # Configure DSPy to use a model from a fake provider that doesn't support structured outputs
    lm = recording_lm(["{'output1': 'Test output', 'output2': True}"], model="fakeprovider/fakemodel")
    lm.engine.supports_response_schema = False
    lm.engine.supported_params = set()
    AlinGo-Prompt.configure(lm=lm, adapter=AlinGo-Prompt.JSONAdapter())
    program(input1="Test input")

    assert len(lm.engine.requests) == 1
    assert lm.engine.requests[0].config.response_format is None



def test_json_adapter_with_structured_outputs_does_not_mutate_original_signature():
    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1")
        subfield2: float = pydantic.Field(description="Float subfield 2")

    class TestSignature(AlinGo-Prompt.Signature):
        input1: str = AlinGo-Prompt.InputField()
        output1: str = AlinGo-Prompt.OutputField()  # Description intentionally left blank
        output2: bool = AlinGo-Prompt.OutputField(desc="Boolean output field")
        output3: OutputField3 = AlinGo-Prompt.OutputField(desc="Nested output field")
        output4_unannotated = AlinGo-Prompt.OutputField(desc="Unannotated output field")

    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM(model="openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter())
    program = AlinGo-Prompt.Predict(TestSignature)
    reply = make_response('{"output1": "a", "output2": true, "output3": {"subfield1": 1, "subfield2": 2.0}, "output4_unannotated": "x"}')
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete", return_value=reply):
        program(input1="Test input")

    assert program.signature.output_fields == TestSignature.output_fields


def test_json_adapter_sync_call():
    signature = AlinGo-Prompt.make_signature("question->answer")
    adapter = AlinGo-Prompt.JSONAdapter()
    lm = AlinGo-Prompt.utils.DummyLM([{"answer": "Paris"}], adapter=adapter)
    with AlinGo-Prompt.context(adapter=adapter):
        result = adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


@pytest.mark.asyncio
async def test_json_adapter_async_call():
    signature = AlinGo-Prompt.make_signature("question->answer")
    adapter = AlinGo-Prompt.JSONAdapter()
    lm = AlinGo-Prompt.utils.DummyLM([{"answer": "Paris"}], adapter=adapter)
    with AlinGo-Prompt.context(adapter=adapter):
        result = await adapter.acall(lm, {}, signature, [], {"question": "What is the capital of France?"})
    assert result == [{"answer": "Paris"}]


def test_json_adapter_on_pydantic_model():
    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    class TestSignature(AlinGo-Prompt.Signature):
        user: User = AlinGo-Prompt.InputField(desc="The user who asks the question")
        question: str = AlinGo-Prompt.InputField(desc="Question the user asks")
        answer: Answer = AlinGo-Prompt.OutputField(desc="Answer to this question")

    program = AlinGo-Prompt.Predict(TestSignature)

    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM(model="openai/gpt-4o", cache=False), adapter=AlinGo-Prompt.JSONAdapter())

    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}",
            model="openai/gpt-4o",
        )
        result = program(
            user={"id": 5, "name": "name_test", "email": "email_test"}, question="What is the capital of France?"
        )

        # Check that the engine was called exactly once
        mock_completion.assert_called_once()

        call_kwargs = request_kwargs(mock_completion.call_args.args[0], "chat")
        # Assert that there are exactly 2 messages (system + user)
        assert len(call_kwargs["messages"]) == 2

        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None

        # Assert that system prompt includes correct input field descriptions
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content

        # Assert that system prompt includes correct output field description
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content

        # Assert that system prompt includes input formatting structure
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content

        # Assert that system prompt includes output formatting structure
        expected_output_structure = (
            "Outputs will be a JSON object with the following fields.\n\n{\n  "
            '"answer": "{answer}        # note: the value you produce must adhere to the JSON schema: '
            '{\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": '
            '\\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": '
            '[\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        )
        assert expected_output_structure in content

        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None

        # Assert that the user input data is formatted correctly
        expected_input_data = (
            '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\n'
            "What is the capital of France?\n\n"
        )
        assert expected_input_data in user_message_content

        # Assert that the adapter output has expected fields and values
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_parse_raise_error_on_mismatch_fields():
    signature = AlinGo-Prompt.make_signature("question->answer")
    adapter = AlinGo-Prompt.JSONAdapter()
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "{'answer1': 'Paris'}",
            model="openai/gpt-4o",
        )
        lm = AlinGo-Prompt.LM(model="openai/gpt-4o-mini")
        with pytest.raises(AlinGo-Prompt.utils.exceptions.AdapterParseError) as e:
            adapter(lm, {}, signature, [], {"question": "What is the capital of France?"})

    assert e.value.adapter_name == "JSONAdapter"
    assert e.value.signature == signature
    assert e.value.lm_response == "{'answer1': 'Paris'}"
    assert e.value.parsed_result == {}

    assert str(e.value) == (
        "Adapter JSONAdapter failed to parse the LM response. \n\n"
        "LM Response: {'answer1': 'Paris'} \n\n"
        "Expected to find output fields in the LM response: [answer] \n\n"
        "Actual output fields parsed from the LM response: [] \n\n"
    )


def test_json_adapter_formats_image():
    # Test basic image formatting
    image = AlinGo-Prompt.Image(url="https://example.com/image.jpg")

    class MySignature(AlinGo-Prompt.Signature):
        image: AlinGo-Prompt.Image = AlinGo-Prompt.InputField()
        text: str = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.JSONAdapter()
    prompt = adapter.format(MySignature, [], {"image": image})

    assert prompt.system
    assert len(prompt.messages) == 1
    parts = prompt.messages[0].parts

    # The message should have 3 parts: text, image, text
    assert len(parts) == 3
    assert parts[0].type == "text"
    assert parts[2].type == "text"

    # Assert that the image is formatted correctly
    assert parts[1] == ImagePart(url="https://example.com/image.jpg", media_type="image/jpeg")


def test_json_adapter_formats_image_with_few_shot_examples():
    class MySignature(AlinGo-Prompt.Signature):
        image: AlinGo-Prompt.Image = AlinGo-Prompt.InputField()
        text: str = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.JSONAdapter()

    demos = [
        AlinGo-Prompt.Example(
            image=AlinGo-Prompt.Image(url="https://example.com/image1.jpg"),
            text="This is a test image",
        ),
        AlinGo-Prompt.Example(
            image=AlinGo-Prompt.Image(url="https://example.com/image2.jpg"),
            text="This is another test image",
        ),
    ]
    messages = adapter.format(MySignature, demos, {"image": AlinGo-Prompt.Image(url="https://example.com/image3.jpg")}).messages

    # 2 few shot examples (1 user and assistant message for each example), 1 user message
    assert len(messages) == 5

    assert ImagePart(url="https://example.com/image1.jpg", media_type="image/jpeg") in messages[0].parts
    assert ImagePart(url="https://example.com/image2.jpg", media_type="image/jpeg") in messages[2].parts
    assert ImagePart(url="https://example.com/image3.jpg", media_type="image/jpeg") in messages[4].parts


def test_json_adapter_formats_image_with_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[AlinGo-Prompt.Image]
        tag: list[str]

    class MySignature(AlinGo-Prompt.Signature):
        image: ImageWrapper = AlinGo-Prompt.InputField()
        text: str = AlinGo-Prompt.OutputField()

    image1 = AlinGo-Prompt.Image(url="https://example.com/image1.jpg")
    image2 = AlinGo-Prompt.Image(url="https://example.com/image2.jpg")
    image3 = AlinGo-Prompt.Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])

    adapter = AlinGo-Prompt.JSONAdapter()
    parts = adapter.format(MySignature, [], {"image": image_wrapper}).messages[0].parts

    assert ImagePart(url="https://example.com/image1.jpg", media_type="image/jpeg") in parts
    assert ImagePart(url="https://example.com/image2.jpg", media_type="image/jpeg") in parts
    assert ImagePart(url="https://example.com/image3.jpg", media_type="image/jpeg") in parts


def test_json_adapter_formats_with_nested_documents():
    class DocumentWrapper(pydantic.BaseModel):
        documents: list[AlinGo-Prompt.experimental.Document]

    class MySignature(AlinGo-Prompt.Signature):
        document: DocumentWrapper = AlinGo-Prompt.InputField()
        text: str = AlinGo-Prompt.OutputField()

    doc1 = AlinGo-Prompt.experimental.Document(data="Hello, world!")
    doc2 = AlinGo-Prompt.experimental.Document(data="Hello, world 2!")

    document_wrapper = DocumentWrapper(documents=[doc1, doc2])

    adapter = AlinGo-Prompt.JSONAdapter()
    message = adapter.format(MySignature, [], {"document": document_wrapper}).messages[0]

    # Plain-text documents are read by the model as text, inside the user message.
    assert all(part.type == "text" for part in message.parts)
    assert "Hello, world!" in message.text
    assert "Hello, world 2!" in message.text


def test_json_adapter_formats_image_with_few_shot_examples_with_nested_images():
    class ImageWrapper(pydantic.BaseModel):
        images: list[AlinGo-Prompt.Image]
        tag: list[str]

    class MySignature(AlinGo-Prompt.Signature):
        image: ImageWrapper = AlinGo-Prompt.InputField()
        text: str = AlinGo-Prompt.OutputField()

    image1 = AlinGo-Prompt.Image(url="https://example.com/image1.jpg")
    image2 = AlinGo-Prompt.Image(url="https://example.com/image2.jpg")
    image3 = AlinGo-Prompt.Image(url="https://example.com/image3.jpg")

    image_wrapper = ImageWrapper(images=[image1, image2, image3], tag=["test", "example"])
    demos = [
        AlinGo-Prompt.Example(
            image=image_wrapper,
            text="This is a test image",
        ),
    ]

    image_wrapper_2 = ImageWrapper(images=[AlinGo-Prompt.Image(url="https://example.com/image4.jpg")], tag=["test", "example"])
    adapter = AlinGo-Prompt.JSONAdapter()
    messages = adapter.format(MySignature, demos, {"image": image_wrapper_2}).messages

    assert len(messages) == 3

    # Image information in the few-shot example's user message
    assert ImagePart(url="https://example.com/image1.jpg", media_type="image/jpeg") in messages[0].parts
    assert ImagePart(url="https://example.com/image2.jpg", media_type="image/jpeg") in messages[0].parts
    assert ImagePart(url="https://example.com/image3.jpg", media_type="image/jpeg") in messages[0].parts

    # The query image is formatted in the last user message
    assert ImagePart(url="https://example.com/image4.jpg", media_type="image/jpeg") in messages[-1].parts


def test_json_adapter_with_tool():
    class MySignature(AlinGo-Prompt.Signature):
        """Answer question with the help of the tools"""

        question: str = AlinGo-Prompt.InputField()
        tools: list[AlinGo-Prompt.Tool] = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()
        tool_calls: AlinGo-Prompt.ToolCalls = AlinGo-Prompt.OutputField()

    def get_weather(city: str) -> str:
        """Get the weather for a city"""
        return f"The weather in {city} is sunny"

    def get_population(country: str, year: int) -> str:
        """Get the population for a country"""
        return f"The population of {country} in {year} is 1000000"

    tools = [AlinGo-Prompt.Tool(get_weather), AlinGo-Prompt.Tool(get_population)]

    adapter = AlinGo-Prompt.JSONAdapter()
    prompt = adapter.format(MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

    assert len(prompt.messages) == 1
    user_text = prompt.messages[0].text

    # The output field type description should be included in the system message even if the output field is nested
    assert AlinGo-Prompt.ToolCalls.description() in prompt.system

    # The user message should include the question and the tools
    assert "What is the weather in Tokyo?" in user_text
    assert "get_weather" in user_text
    assert "get_population" in user_text

    # Tool arguments format should be included in the user message
    assert "{'city': {'type': 'string'}}" in user_text
    assert "{'country': {'type': 'string'}, 'year': {'type': 'integer'}}" in user_text

    reply = make_response('{"answer": "sunny"}')
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete", return_value=reply) as mock_completion:
        lm = AlinGo-Prompt.LM(model="openai/gpt-4o-mini")
        adapter(lm, {}, MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

    mock_completion.assert_called_once()
    call_kwargs = request_kwargs(mock_completion.call_args.args[0], "chat")

    # Assert tool calls are included in the `tools` arg
    assert len(call_kwargs["tools"]) > 0
    assert call_kwargs["tools"][0] == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                    },
                },
                "required": ["city"],
            },
        },
    }
    assert call_kwargs["tools"][1] == {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population for a country",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                    },
                    "year": {
                        "type": "integer",
                    },
                },
                "required": ["country", "year"],
            },
        },
    }


def test_json_adapter_with_code():
    # Test with code as input field
    class CodeAnalysis(AlinGo-Prompt.Signature):
        """Analyze the time complexity of the code"""

        code: AlinGo-Prompt.Code = AlinGo-Prompt.InputField()
        result: str = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.JSONAdapter()
    prompt = adapter.format(CodeAnalysis, [], {"code": "print('Hello, world!')"})

    assert len(prompt.messages) == 1

    # The output field type description should be included in the system message even if the output field is nested
    assert AlinGo-Prompt.Code.description() in prompt.system

    # The user message should include the question and the tools
    assert "print('Hello, world!')" in prompt.messages[0].text

    # Test with code as output field
    class CodeGeneration(AlinGo-Prompt.Signature):
        """Generate code to answer the question"""

        question: str = AlinGo-Prompt.InputField()
        code: AlinGo-Prompt.Code = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.JSONAdapter()
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "{'code': 'print(\"Hello, world!\")'}",
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False),
            {},
            CodeGeneration,
            [],
            {"question": "Write a python program to print 'Hello, world!'"},
        )
        assert result[0]["code"].code == 'print("Hello, world!")'


def test_json_adapter_formats_conversation_history():
    class MySignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        history: AlinGo-Prompt.History = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()

    history = AlinGo-Prompt.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )

    adapter = AlinGo-Prompt.JSONAdapter()
    messages = adapter.format(MySignature, [], {"question": "What is the capital of France?", "history": history}).messages

    assert len(messages) == 5
    reminder = "\n\nRespond with a JSON object in the following order of fields: `answer`."
    assert messages[0].text == "[[ ## question ## ]]\nWhat is the capital of France?" + reminder
    assert messages[1].text == '{\n  "answer": "Paris"\n}'
    assert messages[2].text == "[[ ## question ## ]]\nWhat is the capital of Germany?" + reminder
    assert messages[3].text == '{\n  "answer": "Berlin"\n}'


@pytest.mark.asyncio
async def test_json_adapter_on_pydantic_model_async():
    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    class TestSignature(AlinGo-Prompt.Signature):
        user: User = AlinGo-Prompt.InputField(desc="The user who asks the question")
        question: str = AlinGo-Prompt.InputField(desc="Question the user asks")
        answer: Answer = AlinGo-Prompt.OutputField(desc="Answer to this question")

    program = AlinGo-Prompt.Predict(TestSignature)

    with mock.patch("AlinGo-Prompt.clients.engines.AsyncLM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}",
            model="openai/gpt-4o",
        )

        with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM(model="openai/gpt-4o", cache=False), adapter=AlinGo-Prompt.JSONAdapter()):
            result = await program.acall(
                user={"id": 5, "name": "name_test", "email": "email_test"}, question="What is the capital of France?"
            )

        # Check that the engine was called exactly once
        mock_completion.assert_called_once()

        call_kwargs = request_kwargs(mock_completion.call_args.args[0], "chat")
        # Assert that there are exactly 2 messages (system + user)
        assert len(call_kwargs["messages"]) == 2

        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None

        # Assert that system prompt includes correct input field descriptions
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content

        # Assert that system prompt includes correct output field description
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content

        # Assert that system prompt includes input formatting structure
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content

        # Assert that system prompt includes output formatting structure
        expected_output_structure = (
            "Outputs will be a JSON object with the following fields.\n\n{\n  "
            '"answer": "{answer}        # note: the value you produce must adhere to the JSON schema: '
            '{\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": '
            '\\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": '
            '[\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        )
        assert expected_output_structure in content

        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None

        # Assert that the user input data is formatted correctly
        expected_input_data = (
            '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\n'
            "What is the capital of France?\n\n"
        )
        assert expected_input_data in user_message_content

        # Assert that the adapter output has expected fields and values
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField(desc="String output field")

    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False), adapter=AlinGo-Prompt.JSONAdapter())
    program = AlinGo-Prompt.Predict(TestSignature)

    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.side_effect = RuntimeError("Structured output failed!")

        with pytest.raises(AlinGo-Prompt.LMUnexpectedError, match="Structured output failed"):
            program(question="Dummy question!")

        assert mock_completion.call_count == 1
        first_call_kwargs = request_kwargs(mock_completion.call_args_list[0].args[0], "chat")
        assert first_call_kwargs.get("response_format")["type"] == "json_schema"


def test_json_adapter_json_mode_no_structured_outputs():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField(desc="String output field")

    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM(model="openai/gpt-4o", cache=False), adapter=AlinGo-Prompt.JSONAdapter())
    program = AlinGo-Prompt.Predict(TestSignature)

    with (
        mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion,
        mock.patch.object(AlinGo-Prompt.LM, "supported_params", new_callable=mock.PropertyMock) as mock_supported_params,
        mock.patch.object(AlinGo-Prompt.LM, "supports_response_schema", new_callable=mock.PropertyMock) as mock_supports_response_schema,
    ):
        # Call a model that allows json but not structured outputs
        mock_completion.return_value = make_response("{'answer': 'Test output'}")
        mock_supported_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False

        result = program(question="Dummy question!")

        assert mock_completion.call_count == 1
        assert result.answer == "Test output"

        call_kwargs = request_kwargs(mock_completion.call_args_list[0].args[0], "chat")
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_json_mode_no_structured_outputs_async():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField(desc="String output field")

    program = AlinGo-Prompt.Predict(TestSignature)

    with (
        mock.patch("AlinGo-Prompt.clients.engines.AsyncLM15Engine.complete") as mock_acompletion,
        mock.patch.object(AlinGo-Prompt.LM, "supported_params", new_callable=mock.PropertyMock) as mock_supported_params,
        mock.patch.object(AlinGo-Prompt.LM, "supports_response_schema", new_callable=mock.PropertyMock) as mock_supports_response_schema,
    ):
        # Call a model that allows json but not structured outputs
        mock_acompletion.return_value = make_response("{'answer': 'Test output'}")
        mock_supported_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False

        with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM(model="openai/gpt-4o", cache=False), adapter=AlinGo-Prompt.JSONAdapter()):
            result = await program.acall(question="Dummy question!")

        assert mock_acompletion.call_count == 1
        assert result.answer == "Test output"

        call_kwargs = request_kwargs(mock_acompletion.call_args_list[0].args[0], "chat")
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error_async():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField(desc="String output field")

    program = AlinGo-Prompt.Predict(TestSignature)

    with mock.patch("AlinGo-Prompt.clients.engines.AsyncLM15Engine.complete") as mock_acompletion:
        mock_acompletion.side_effect = RuntimeError("Structured output failed!")

        with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False), adapter=AlinGo-Prompt.JSONAdapter()):
            with pytest.raises(AlinGo-Prompt.LMUnexpectedError, match="Structured output failed"):
                await program.acall(question="Dummy question!")

        assert mock_acompletion.call_count == 1
        first_call_kwargs = request_kwargs(mock_acompletion.call_args_list[0].args[0], "chat")
        assert first_call_kwargs.get("response_format")["type"] == "json_schema"


def test_error_message_on_json_adapter_failure():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField(desc="String output field")

    program = AlinGo-Prompt.Predict(TestSignature)

    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False), adapter=AlinGo-Prompt.JSONAdapter())

    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.side_effect = RuntimeError("RuntimeError!")

        with pytest.raises(AlinGo-Prompt.LMUnexpectedError) as error:
            program(question="Dummy question!")

        assert "RuntimeError!" in str(error.value)

        mock_completion.side_effect = ValueError("ValueError!")
        with pytest.raises(AlinGo-Prompt.LMUnexpectedError) as error:
            program(question="Dummy question!")

        assert "ValueError!" in str(error.value)


@pytest.mark.asyncio
async def test_error_message_on_json_adapter_failure_async():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField(desc="String output field")

    program = AlinGo-Prompt.Predict(TestSignature)

    with mock.patch("AlinGo-Prompt.clients.engines.AsyncLM15Engine.complete") as mock_acompletion:
        with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False), adapter=AlinGo-Prompt.JSONAdapter()):
            mock_acompletion.side_effect = RuntimeError("RuntimeError!")
            with pytest.raises(AlinGo-Prompt.LMUnexpectedError) as error:
                await program.acall(question="Dummy question!")

            assert "RuntimeError!" in str(error.value)

            mock_acompletion.side_effect = ValueError("ValueError!")
            with pytest.raises(AlinGo-Prompt.LMUnexpectedError) as error:
                await program.acall(question="Dummy question!")

            assert "ValueError!" in str(error.value)


def test_json_adapter_toolcalls_native_function_calling():
    class MySignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        tools: list[AlinGo-Prompt.Tool] = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()
        tool_calls: AlinGo-Prompt.ToolCalls = AlinGo-Prompt.OutputField()

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [AlinGo-Prompt.Tool(get_weather)]

    adapter = AlinGo-Prompt.JSONAdapter(use_native_function_calling=True)

    # Case 1: Tool calls are present in the response, while content is None.
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            tool_calls=[("call_pQm8ajtSMxgA0nrzK2ivFmxG", "get_weather", {"city": "Paris"})],
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the weather in Paris?", "tools": tools},
        )

        assert result[0]["tool_calls"] == AlinGo-Prompt.ToolCalls(
            tool_calls=[
                AlinGo-Prompt.ToolCalls.ToolCall(
                    id="call_pQm8ajtSMxgA0nrzK2ivFmxG",
                    name="get_weather",
                    args={"city": "Paris"},
                )
            ]
        )
        # `answer` is not present, so we set it to None
        assert result[0]["answer"] is None

    # Case 2: Tool calls are not present in the response, while content is present.
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "{'answer': 'Paris'}",
            model="openai/gpt-4o-mini",
        )
        result = adapter(
            AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the weather in Paris?", "tools": tools},
        )
        assert result[0]["answer"] == "Paris"
        assert result[0]["tool_calls"] is None


def test_json_adapter_toolcalls_no_native_function_calling():
    class MySignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        tools: list[AlinGo-Prompt.Tool] = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()
        tool_calls: AlinGo-Prompt.ToolCalls = AlinGo-Prompt.OutputField()

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [AlinGo-Prompt.Tool(get_weather)]

    # Patch _get_structured_outputs_response_format to track calls
    with mock.patch("AlinGo-Prompt.adapters.json_adapter._get_structured_outputs_response_format") as mock_structured:
        # Patch the engine to return a dummy response
        with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
            mock_completion.return_value = make_response(
                "{'answer': 'sunny', 'tool_calls': {'tool_calls': []}}",
                model="openai/gpt-4o-mini",
            )
            adapter = AlinGo-Prompt.JSONAdapter(use_native_function_calling=False)
            lm = AlinGo-Prompt.LM(model="openai/gpt-4o-mini", cache=False)
            adapter(lm, {}, MySignature, [], {"question": "What is the weather in Tokyo?", "tools": tools})

        # _get_structured_outputs_response_format is not called because without using native function calling,
        # JSONAdapter falls back to json mode for stable quality.
        mock_structured.assert_not_called()
        mock_completion.assert_called_once()
        call_kwargs = request_kwargs(mock_completion.call_args.args[0], "chat")
        assert call_kwargs["response_format"] == {"type": "json_object"}


@mock.patch.object(AlinGo-Prompt.LM, "supports_reasoning", new_callable=mock.PropertyMock, return_value=True)
def test_json_adapter_native_reasoning(_supports_reasoning):
    class MySignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        reasoning: AlinGo-Prompt.Reasoning = AlinGo-Prompt.OutputField()
        answer: str = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.JSONAdapter()

    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "{'answer': 'Paris'}",
            thinking="Step-by-step thinking about the capital of France",
            model="anthropic/claude-3-7-sonnet-20250219",
        )
        modified_signature = adapter._call_preprocess(
            AlinGo-Prompt.LM(model="anthropic/claude-3-7-sonnet-20250219", reasoning_effort="low", cache=False),
            {},
            MySignature,
            {"question": "What is the capital of France?"},
        )
        assert "reasoning" not in modified_signature.output_fields

        result = adapter(
            AlinGo-Prompt.LM(model="anthropic/claude-3-7-sonnet-20250219", reasoning_effort="low", cache=False),
            {},
            MySignature,
            [],
            {"question": "What is the capital of France?"},
        )
        assert result[0]["reasoning"] == AlinGo-Prompt.Reasoning(content="Step-by-step thinking about the capital of France")


def test_json_adapter_with_responses_api():
    class TestSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()

    api_response = make_response(
        '{"answer": "Washington, D.C."}',
        model="openai/gpt-4o",
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )

    lm = AlinGo-Prompt.LM(model="openai/gpt-4o", model_type="responses", cache=False)
    AlinGo-Prompt.configure(lm=lm, adapter=AlinGo-Prompt.JSONAdapter())

    program = AlinGo-Prompt.Predict(TestSignature)
    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete", return_value=api_response) as mock_responses:
        result = program(question="What is the capital of the USA?")

    assert result.answer == "Washington, D.C."
    mock_responses.assert_called_once()
    # Verify the canonical adapter request renders to Responses text.format.
    call_kwargs = request_kwargs(mock_responses.call_args.args[0], "responses")
    assert "response_format" not in call_kwargs
    assert "text" in call_kwargs

    assert isinstance(call_kwargs["text"]["format"], dict)
    assert isinstance(call_kwargs["text"]["format"]["name"], str)
    assert call_kwargs["text"]["format"]["type"] == "json_schema"
    assert isinstance(call_kwargs["text"]["format"]["schema"], dict)


def test_format_system_message():
    class MySignature(AlinGo-Prompt.Signature):
        """Answer the question with multiple answers and scores"""

        question: str = AlinGo-Prompt.InputField()
        answers: list[str] = AlinGo-Prompt.OutputField()
        scores: list[float] = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.JSONAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = """Your input fields are:
1. `question` (str):
Your output fields are:
1. `answers` (list[str]):\x20
2. `scores` (list[float]):
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

[[ ## question ## ]]
{question}

Outputs will be a JSON object with the following fields.

{
  "answers": "{answers}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"string\\"}}",
  "scores": "{scores}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"number\\"}}"
}
In adhering to this structure, your objective is:\x20
        Answer the question with multiple answers and scores"""
    assert system_message == expected_system_message


def test_missing_optional_output_fields_fall_back_to_defaults():
    from AlinGo-Prompt.utils.exceptions import AdapterParseError

    class OptionalOutputSignature(AlinGo-Prompt.Signature):
        question: str = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()
        note: str | None = AlinGo-Prompt.OutputField(default="No note")
        maybe: str | None = AlinGo-Prompt.OutputField()

    adapter = AlinGo-Prompt.JSONAdapter()
    parsed = adapter.parse(OptionalOutputSignature, '{"answer": "42"}')
    assert parsed == {"answer": "42", "note": "No note", "maybe": None}

    with pytest.raises(AdapterParseError):
        adapter.parse(OptionalOutputSignature, '{"note": "present"}')
