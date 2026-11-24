# FILEPATH: /Users/ahle/repos/AlinGo-Prompt/tests/evaluate/test_metrics.py

import AlinGo-Prompt
from AlinGo-Prompt.evaluate.metrics import answer_exact_match
from AlinGo-Prompt.predict import Predict


def test_answer_exact_match_string():
    example = AlinGo-Prompt.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_list():
    example = AlinGo-Prompt.Example(
        question="What is 1+1?",
        answer=["2", "two"],
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_no_match():
    example = AlinGo-Prompt.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)
