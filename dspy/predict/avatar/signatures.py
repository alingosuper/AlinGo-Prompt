import AlinGo-Prompt
from AlinGo-Prompt.predict.avatar.models import Action


class Actor(AlinGo-Prompt.Signature):
    """You will be given `Tools` which will be a list of tools to use to accomplish the `Goal`. Given the user query, your task is to decide which tool to use and what input values to provide.

    You will output action needed to accomplish the `Goal`. `Action` should have a tool to use and the input query to pass to the tool.

    Note: You can opt to use no tools and provide the final answer directly. You can also one tool multiple times with different input queries if applicable."""

    goal: str = AlinGo-Prompt.InputField(
        desc="Task to be accomplished.",
    )
    tools: list[str] = AlinGo-Prompt.InputField(
        desc="list of tools to use",
    )
    action_1: Action = AlinGo-Prompt.OutputField(
        desc="1st action to take.",
    )
