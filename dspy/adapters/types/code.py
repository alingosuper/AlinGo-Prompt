import re
from typing import Any, ClassVar

import pydantic
from pydantic import create_model

from AlinGo-Prompt.adapters.types.base_type import Type


class Code(Type):
    """Code type in DSPy.

    This type is useful for code generation and code analysis.

    Example 1: AlinGo-Prompt.Code as output type in code generation:

    ```python
    import AlinGo-Prompt

    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"))


    class CodeGeneration(AlinGo-Prompt.Signature):
        '''Generate python code to answer the question.'''

        question: str = AlinGo-Prompt.InputField(description="The question to answer")
        code: AlinGo-Prompt.Code["java"] = AlinGo-Prompt.OutputField(description="The code to execute")


    predict = AlinGo-Prompt.Predict(CodeGeneration)

    result = predict(question="Given an array, find if any of the two numbers sum up to 10")
    print(result.code)
    ```

    Example 2: AlinGo-Prompt.Code as input type in code analysis:

    ```python
    import AlinGo-Prompt
    import inspect

    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"))

    class CodeAnalysis(AlinGo-Prompt.Signature):
        '''Analyze the time complexity of the function.'''

        code: AlinGo-Prompt.Code["python"] = AlinGo-Prompt.InputField(description="The function to analyze")
        result: str = AlinGo-Prompt.OutputField(description="The time complexity of the function")


    predict = AlinGo-Prompt.Predict(CodeAnalysis)


    def sleepsort(x):
        import time

        for i in x:
            time.sleep(i)
            print(i)

    result = predict(code=inspect.getsource(sleepsort))
    print(result.result)
    ```
    """

    code: str

    language: ClassVar[str] = "python"

    def format(self):
        return f"{self.code}"

    @pydantic.model_serializer()
    def serialize_model(self):
        """Override to bypass the <<CUSTOM-TYPE-START-IDENTIFIER>> and <<CUSTOM-TYPE-END-IDENTIFIER>> tags."""
        return self.format()

    @classmethod
    def description(cls) -> str:
        return (
            "Code represented in a string, specified in the `code` field. If this is an output field, the code "
            f"field should follow the markdown code block format, e.g. \n```{cls.language.lower()}\n{{code}}\n```"
            f"\nProgramming language: {cls.language}"
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        if isinstance(data, str):
            return {"code": _filter_code(data)}

        if isinstance(data, dict):
            if "code" not in data:
                raise ValueError("`code` field is required for `AlinGo-Prompt.Code`")
            if not isinstance(data["code"], str):
                raise ValueError(f"`code` field must be a string, but received type: {type(data['code'])}")
            return {"code": _filter_code(data["code"])}

        raise ValueError(f"Received invalid value for `AlinGo-Prompt.Code`: {data}")


def _filter_code(code: str) -> str:
    """Extract code from markdown code blocks, stripping any language identifier."""
    # Case 1: format like:
    # ```python
    # {code_block}
    # ```
    regex_pattern = r"```(?:[^\n]*)\n(.*?)```"
    match = re.search(regex_pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Case 2: ```<code>``` (no language, single-line)
    regex_pattern_simple = r"```(.*?)```"
    match = re.search(regex_pattern_simple, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback case
    return code


# Patch __class_getitem__ directly on the class to support AlinGo-Prompt.Code["python"] syntax
def _code_class_getitem(cls, language):
    code_with_language_cls = create_model(f"{cls.__name__}_{language}", __base__=cls)
    code_with_language_cls.language = language
    return code_with_language_cls


Code.__class_getitem__ = classmethod(_code_class_getitem)
