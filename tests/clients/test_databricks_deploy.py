import pytest

from AlinGo-Prompt.clients import databricks
from AlinGo-Prompt.utils.lazy_import import _MissingModule


def test_deploy_checks_for_the_openai_sdk_before_touching_the_workspace(monkeypatch):
    calls = []
    monkeypatch.setattr(
        databricks, "require", lambda module, **kwargs: _MissingModule(module, "openai is required", {
            "filename": __file__, "lineno": 1, "function": "test", "code_context": None,
        })
    )
    monkeypatch.setattr(databricks, "_get_workspace_client", lambda: calls.append("workspace"))

    with pytest.raises(ImportError, match="openai is required"):
        databricks.DatabricksProvider.deploy_finetuned_model("catalog.schema.model")
    assert calls == []
