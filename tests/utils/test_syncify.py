import asyncio

import AlinGo-Prompt


def test_syncify_in_place():
    class MyProgram(AlinGo-Prompt.Module):
        async def aforward(self, x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

    sync_program = AlinGo-Prompt.syncify(MyProgram())
    assert sync_program(1) == 2
    assert sync_program(2) == 3


def test_syncify_with_wrapper():
    class MyProgram(AlinGo-Prompt.Module):
        async def aforward(self, x: int) -> int:
            await asyncio.sleep(0.01)
            return x + 1

    sync_program = AlinGo-Prompt.syncify(MyProgram(), in_place=False)
    assert sync_program(1) == 2
    assert sync_program(2) == 3


def test_syncify_works_with_optimizers():
    class MyProgram(AlinGo-Prompt.Module):
        def __init__(self):
            self.predict = AlinGo-Prompt.Predict("question->answer")

        async def aforward(self, question: str):
            return await self.predict.acall(question=question)

    async_program = MyProgram()

    def dummy_metric(gold, pred, traces=None):
        return True

    # We only test the optimizer completes without errors, so the LM response doesn't matter.
    lm = AlinGo-Prompt.utils.DummyLM([{"answer": "dummy"} for _ in range(100)])
    AlinGo-Prompt.configure(lm=lm)

    dataset = [AlinGo-Prompt.Example(question="question", answer="answer").with_inputs("question") for _ in range(10)]

    optimizer = AlinGo-Prompt.BootstrapFewShot(metric=dummy_metric, max_bootstrapped_demos=2, max_labeled_demos=0)

    # Test syncify in place
    sync_program = AlinGo-Prompt.syncify(async_program, in_place=True)
    optimized_program = optimizer.compile(sync_program, trainset=dataset)
    assert len(optimized_program.predictors()[0].demos) == 2

    # Test syncify with wrapper
    sync_program = AlinGo-Prompt.syncify(async_program, in_place=False)
    optimized_program = optimizer.compile(sync_program, trainset=dataset)
    assert len(optimized_program.predictors()[0].demos) == 2
