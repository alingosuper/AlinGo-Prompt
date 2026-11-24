import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest

import AlinGo-Prompt
from tests.test_utils.engines import make_response


def test_basic_AlinGo-Prompt_settings():
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter(), callbacks=[lambda x: x])
    assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o"
    assert isinstance(AlinGo-Prompt.settings.adapter, AlinGo-Prompt.JSONAdapter)
    assert len(AlinGo-Prompt.settings.callbacks) == 1


def test_forbid_configure_call_in_child_thread():
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter(), callbacks=[lambda x: x])

    def worker():
        with pytest.raises(RuntimeError, match="AlinGo-Prompt.settings can only be changed by the thread"):
            AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"), callbacks=[])

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker).result()


def test_AlinGo-Prompt_context():
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter(), callbacks=[lambda x: x])
    with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"), callbacks=[]):
        assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o-mini"
        assert len(AlinGo-Prompt.settings.callbacks) == 0

    assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o"
    assert len(AlinGo-Prompt.settings.callbacks) == 1


def test_AlinGo-Prompt_context_parallel():
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter(), callbacks=[lambda x: x])

    def worker(i):
        with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"), trace=[i], callbacks=[]):
            assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o-mini"
            assert AlinGo-Prompt.settings.trace == [i]
            assert len(AlinGo-Prompt.settings.callbacks) == 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(worker, range(3)))

    assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o"
    assert len(AlinGo-Prompt.settings.callbacks) == 1


def test_AlinGo-Prompt_context_with_AlinGo-Prompt_parallel():
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o", cache=False), adapter=AlinGo-Prompt.ChatAdapter())

    class MyModule(AlinGo-Prompt.Module):
        def __init__(self):
            self.predict = AlinGo-Prompt.Predict("question -> answer")

        def forward(self, question: str) -> str:
            lm = AlinGo-Prompt.LM("openai/gpt-4o-mini", cache=False) if "France" in question else AlinGo-Prompt.settings.lm
            with AlinGo-Prompt.context(lm=lm):
                time.sleep(1)
                assert AlinGo-Prompt.settings.lm.model == lm.model
                return self.predict(question=question)

    with mock.patch("AlinGo-Prompt.clients.engines.LM15Engine.complete") as mock_completion:
        mock_completion.return_value = make_response(
            "[[ ## answer ## ]]\nParis",
            model="openai/gpt-4o-mini",
        )

        module = MyModule()
        parallelizer = AlinGo-Prompt.Parallel()
        input_pairs = [
            (module, {"question": "What is the capital of France?"}),
            (module, {"question": "What is the capital of Germany?"}),
        ]
        parallelizer(input_pairs)

        # Verify mock was called correctly
        assert mock_completion.call_count == 2
        for call_args in mock_completion.call_args_list:
            request = call_args.args[0]
            if "France" in request.messages[-1].text:
                # France question uses gpt-4o-mini
                assert request.model == "openai/gpt-4o-mini"
            else:
                # Germany question uses gpt-4o
                assert request.model == "openai/gpt-4o"

        # The main thread is not affected by the context
        assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_AlinGo-Prompt_context_with_async_task_group():
    class MyModule(AlinGo-Prompt.Module):
        def __init__(self):
            self.predict = AlinGo-Prompt.Predict("question -> answer")

        async def aforward(self, question: str) -> str:
            lm = (
                AlinGo-Prompt.LM("openai/gpt-4o-mini", cache=False)
                if "France" in question
                else AlinGo-Prompt.LM("openai/gpt-4o", cache=False)
            )
            with AlinGo-Prompt.context(lm=lm, trace=[]):
                await asyncio.sleep(1)
                assert AlinGo-Prompt.settings.lm.model == lm.model
                result = await self.predict.acall(question=question)
                assert len(AlinGo-Prompt.settings.trace) == 1
                return result

    module = MyModule()

    with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM("openai/gpt-4.1", cache=False), adapter=AlinGo-Prompt.ChatAdapter()):
        with mock.patch("AlinGo-Prompt.clients.engines.AsyncLM15Engine.complete") as mock_completion:
            mock_completion.return_value = make_response(
                "[[ ## answer ## ]]\nParis",
                model="openai/gpt-4o-mini",
            )

            # Define the coroutines to be run
            coroutines = [
                module.acall(question="What is the capital of France?"),
                module.acall(question="What is the capital of France?"),
                module.acall(question="What is the capital of Germany?"),
                module.acall(question="What is the capital of Germany?"),
            ]

            # Run them concurrently and gather results
            results = await asyncio.gather(*coroutines)

        assert results[0].answer == "Paris"
        assert results[1].answer == "Paris"
        assert results[2].answer == "Paris"
        assert results[3].answer == "Paris"

        # Concurrent calls can arrive in any order. Check the association,
        # not just model counts, so leaked task-local settings still fail.
        assert mock_completion.call_count == 4
        seen = {"France": 0, "Germany": 0}
        for call in mock_completion.call_args_list:
            request = call.args[0]
            content = request.messages[-1].text
            country = "France" if "France" in content else "Germany"
            assert country in content
            expected = "openai/gpt-4o-mini" if country == "France" else "openai/gpt-4o"
            assert request.model == expected
            seen[country] += 1
        assert seen == {"France": 2, "Germany": 2}

        # The main thread is not affected by the context
        assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4.1"
        assert AlinGo-Prompt.settings.trace == []


@pytest.mark.asyncio
async def test_AlinGo-Prompt_configure_allowance_async():
    def bar1():
        # `AlinGo-Prompt.configure` is disallowed in different async tasks from the initial one.
        # In this case, foo1 (async) calls bar1 (sync), and bar1 uses the async task from foo1.
        with pytest.raises(RuntimeError) as e:
            AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"))
        assert "AlinGo-Prompt.configure(...) can only be called from the same async" in str(e.value)

    async def foo1():
        bar1()
        await asyncio.sleep(0.1)

    async def foo2():
        # `AlinGo-Prompt.configure` is disallowed in different async tasks from the initial one.
        with pytest.raises(RuntimeError) as e:
            AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"))
        assert "AlinGo-Prompt.configure(...) can only be called from the same async" in str(e.value)
        await asyncio.sleep(0.1)

    async def foo3():
        # `AlinGo-Prompt.context` is allowed in different async tasks from the initial one.
        with AlinGo-Prompt.context(lm=AlinGo-Prompt.LM("openai/gpt-4o")):
            await asyncio.sleep(0.1)

    async def foo4():
        # foo4 is directly invoked by the entry task, so it has the same async task as the entry task.
        AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"))
        await asyncio.sleep(0.1)

    # `AlinGo-Prompt.configure` is allowed to be called multiple times in the same async task.
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o-mini"))
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"))
    AlinGo-Prompt.configure(adapter=AlinGo-Prompt.JSONAdapter())

    await asyncio.gather(foo1(), foo2(), foo3())

    await foo4()


def test_AlinGo-Prompt_settings_save_load(tmp_path):
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter(), callbacks=[lambda x: x])

    AlinGo-Prompt.settings.save(tmp_path / "settings.pkl")
    AlinGo-Prompt.configure(lm=None, adapter=None, callbacks=None)

    loaded_settings = AlinGo-Prompt.load_settings(tmp_path / "settings.pkl", allow_pickle=True)
    AlinGo-Prompt.configure(**loaded_settings)
    assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o"
    assert isinstance(AlinGo-Prompt.settings.adapter, AlinGo-Prompt.JSONAdapter)
    assert len(AlinGo-Prompt.settings.callbacks) == 1


def test_AlinGo-Prompt_settings_save_exclude_keys(tmp_path):
    AlinGo-Prompt.configure(lm=AlinGo-Prompt.LM("openai/gpt-4o"), adapter=AlinGo-Prompt.JSONAdapter(), track_usage=True)

    AlinGo-Prompt.settings.save(tmp_path / "settings.pkl", exclude_keys=["adapter", "track_usage"])
    AlinGo-Prompt.configure(lm=None, adapter=None, track_usage=False)

    loaded_settings = AlinGo-Prompt.load_settings(tmp_path / "settings.pkl", allow_pickle=True)
    AlinGo-Prompt.configure(**loaded_settings)
    assert AlinGo-Prompt.settings.lm.model == "openai/gpt-4o"
    assert AlinGo-Prompt.settings.adapter is None
    assert not AlinGo-Prompt.settings.track_usage



def test_settings_save_with_extra_modules(tmp_path):
    # Create a temporary Python file with our custom module
    custom_module_path = tmp_path / "custom_module.py"
    with open(custom_module_path, "w") as f:
        f.write(
            """
def callback(x):
    return x + 1
"""
        )

    # Add the tmp_path to Python path so we can import the module
    sys.path.insert(0, str(tmp_path))
    try:
        import custom_module

        AlinGo-Prompt.configure(callbacks=[custom_module.callback])

        settings_path = tmp_path / "settings.pkl"
        sys.path.insert(0, str(tmp_path))

        AlinGo-Prompt.configure(callbacks=[custom_module.callback])
        AlinGo-Prompt.settings.save(settings_path, modules_to_serialize=[custom_module])

        # Remove the custom module again to simulate it not being available at load time
        sys.modules.pop("custom_module", None)
        sys.path.remove(str(tmp_path))
        del custom_module

        AlinGo-Prompt.configure(callbacks=None)

        # Loading should now succeed and preserve the adapter instance
        loaded_settings = AlinGo-Prompt.load_settings(settings_path, allow_pickle=True)
        AlinGo-Prompt.settings.configure(**loaded_settings)

        assert AlinGo-Prompt.settings.callbacks[0](3) == 4

    finally:
        # Only need to clean up sys.path
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
