import json
from datetime import date
from typing import Literal

import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

import AlinGo-Prompt
from AlinGo-Prompt.adapters.decision import record_evidence
from AlinGo-Prompt.experimental import Score
from AlinGo-Prompt.lm15 import DataPart, ImagePart, TextPart
from tests.test_utils.engines import recording_lm

EDGES = {"top": 0.13, "bottom": 0.81, "left": 0.27, "right": 0.64}


class Locate(AlinGo-Prompt.Signature):
    """Find the target in the photo."""

    photo: AlinGo-Prompt.Image = AlinGo-Prompt.InputField()
    box: AlinGo-Prompt.BoundingBox = AlinGo-Prompt.OutputField(desc="The dog in inputs.photo.")


def test_geometry_round_trip_and_boundaries():
    box = AlinGo-Prompt.BoundingBox(**EDGES)
    assert isinstance(box, AlinGo-Prompt.Type)
    assert box.model_dump() == EDGES
    assert AlinGo-Prompt.BoundingBox.model_validate_json(box.model_dump_json()) == box
    assert json.loads(box.format()) == EDGES
    assert AlinGo-Prompt.BoundingBox(top=0, bottom=1, left=0, right=1).right == 1
    with pytest.raises(ValidationError):
        box.left = 0.9


@pytest.mark.parametrize("change", [
    {"top": -0.01}, {"right": 1.01}, {"left": float("nan")}, {"bottom": float("inf")},
    {"top": True}, {"left": "0.27"}, {"top": 0.81}, {"left": 0.65}, {"confidence": 0.9},
])
def test_invalid_geometry_is_not_silently_repaired(change):
    with pytest.raises(ValidationError):
        AlinGo-Prompt.BoundingBox(**{**EDGES, **change})


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter_class", [AlinGo-Prompt.ChatAdapter, AlinGo-Prompt.JSONAdapter, AlinGo-Prompt.XMLAdapter, AlinGo-Prompt.TwoStepAdapter])
async def test_boxes_are_structured_outputs_inputs_and_saved_demos(adapter_class, asynchronous, tmp_path):
    class Regions(BaseModel):
        boxes: list[AlinGo-Prompt.BoundingBox]

    signature = AlinGo-Prompt.Signature({
        "previous": (AlinGo-Prompt.BoundingBox, AlinGo-Prompt.InputField()),
        "region": (Regions, AlinGo-Prompt.OutputField()),
        "missing": (AlinGo-Prompt.BoundingBox | None, AlinGo-Prompt.OutputField()),
    })
    predict = AlinGo-Prompt.Predict(signature)
    box = AlinGo-Prompt.BoundingBox(**EDGES)
    predict.demos = [AlinGo-Prompt.Example(previous=box, region=Regions(boxes=[box]), missing=None)]
    predict.save(tmp_path / "boxes.json")
    predict.load(tmp_path / "boxes.json")
    assert predict.demos[0]["previous"] == EDGES
    assert predict.demos[0]["region"] == {"boxes": [EDGES]}
    answer = {"region": {"boxes": [EDGES]}, "missing": None}
    if adapter_class is AlinGo-Prompt.TwoStepAdapter:
        # TwoStepAdapter's extraction stage uses ChatAdapter.
        extraction_lm = recording_lm([AlinGo-Prompt.ChatAdapter().format_assistant_message_content(signature, answer)])
        adapter = adapter_class(extraction_model=extraction_lm)
        lm = recording_lm(["The region contains one box: top 0.13, bottom 0.81, left 0.27, right 0.64. "
                           "The missing box is null."])
    else:
        adapter = adapter_class()
        lm = recording_lm([adapter.format_assistant_message_content(signature, answer)])
    with AlinGo-Prompt.context(adapter=adapter):
        result = await predict.acall(previous=box, lm=lm) if asynchronous else predict(previous=box, lm=lm)
    assert result.region.boxes == [box]
    assert result.missing is None
    assert "CUSTOM-TYPE" not in json.dumps(lm.history[-1]["messages"])


@pytest.mark.parametrize("left", ["0.27", "true"])
def test_predict_boxes_from_literal_nested_xml(left, monkeypatch):
    from tests.clients.test_lm_engine_execution import native_transport, wire

    class Regions(BaseModel):
        boxes: list[AlinGo-Prompt.BoundingBox]
        label: str

    signature = AlinGo-Prompt.Signature({"region": (Regions, AlinGo-Prompt.OutputField())})
    completion = (
        f"<region><boxes><item><top>0.13</top><bottom>0.81</bottom><left>{left}</left>"
        "<right>0.64</right></item></boxes><label>0.27</label></region>"
    )
    native_transport(monkeypatch, [wire(completion), wire(completion)])
    lm = AlinGo-Prompt.LM("openai/gpt-4o-mini", model_type="chat", engine="lm15", cache=False)
    try:
        with AlinGo-Prompt.context(adapter=AlinGo-Prompt.XMLAdapter()):
            if left == "true":
                with pytest.raises(AlinGo-Prompt.AdapterParseError):
                    AlinGo-Prompt.Predict(signature)(lm=lm)
            else:
                result = AlinGo-Prompt.Predict(signature)(lm=lm)
                assert result.region.boxes == [AlinGo-Prompt.BoundingBox(**EDGES)]
                assert result.region.label == "0.27"
    finally:
        lm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
async def test_predict_native_boxes_evidence_images_and_cache(asynchronous, mixed, monkeypatch):
    from AlinGo-Prompt import lm15
    from AlinGo-Prompt._vendor.lm15.testing import FakeTransport
    from tests.clients.test_lm_engine_execution import DualResponse

    signature = Locate
    answers = {"box": {"type": "box", "box": EDGES, "confidence": 0.83}}
    if mixed:
        signature = signature.append("second", AlinGo-Prompt.OutputField(desc="The other target."), type_=AlinGo-Prompt.BoundingBox)
        signature = signature.append("visible", AlinGo-Prompt.OutputField(desc="Is a dog visible?"), type_=bool)
        signature = signature.append("kind", AlinGo-Prompt.OutputField(desc="What kind of animal?"), type_=Literal["dog", "cat"])
        signature = signature.append("size", AlinGo-Prompt.OutputField(desc="Size, small to large."), type_=Score["small", "medium", "large"])
        answers.update({
            "second": {"type": "box", "box": {"top": 0.21, "bottom": 0.67, "left": 0.04, "right": 0.19}, "confidence": 0.62},
            "visible": {"type": "noul", "noul": 0.91},
            "kind": {"type": "choice", "choice": "dog", "probabilities": {"dog": 0.8, "cat": 0.2}, "confidence": 0.7},
            "size": {"type": "score", "score": 1.6, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}, "confidence": 0.6},
        })
    wire_response = {"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": 31, "output_tokens": 8}}
    transport = FakeTransport([DualResponse(200, json.dumps(wire_response).encode())])
    monkeypatch.setattr("AlinGo-Prompt.clients.execution.RouterConfig", lambda **kwargs: lm15.RouterConfig(**{**kwargs, "transport": transport}))
    lm = AlinGo-Prompt.LM("typesafe:jev-1.14.0", engine="lm15", api_key="test", num_retries=0)
    image = AlinGo-Prompt.Image(Image.new("RGB", (320, 280), "red"))
    predict = AlinGo-Prompt.Predict(signature)
    predict.demos = [AlinGo-Prompt.Example(photo=image, box=AlinGo-Prompt.BoundingBox(**EDGES))]
    try:
        with AlinGo-Prompt.context(adapter=None), record_evidence() as evidence:
            for _ in range(2):
                result = await predict.acall(photo=image, lm=lm) if asynchronous else predict(photo=image, lm=lm)
                assert result.box.model_dump() == EDGES
                response = lm.history[-1]["response"]
                assert response.provider_data["typesafe"]["answers"] == answers
                assert response.data["box"] == EDGES
                if mixed:
                    assert result.second.model_dump() == answers["second"]["box"]
                    assert result.visible is True and result.kind == "dog" and result.size.level == 2
                    assert result.size.value == pytest.approx(1.6)
                    assert set(response.probabilities) == {"visible", "kind", "size"}
                    assert response.probabilities["visible"]["true"] == 0.91
                else:
                    assert response.probabilities is None
            if mixed:
                predict.fields = {"visible": {"threshold": 0.95}, "kind": {"weights": {"cat": 10}},
                                  "size": {"cuts": [0.4, 1.8]}}
                result = await predict.acall(photo=image, lm=lm) if asynchronous else predict(photo=image, lm=lm)
                assert result.visible is False and result.kind == "cat" and result.size.level == 1
                assert result.box.model_dump() == EDGES
                predict.fields["box"] = {"threshold": 0.9}
                with pytest.raises(ValueError, match="unsupported outputs"):
                    if asynchronous:
                        await predict.acall(photo=image, lm=lm)
                    else:
                        predict(photo=image, lm=lm)
            assert next(answer for _, name, answer in evidence if name == "box") == answers["box"]
        assert len(transport.requests) == 1
        request = transport.requests[0]
        payload = json.loads(request.body)
        assert payload["questions"]["box"] == {"type": "box", "instructions": "The dog in inputs.photo."}
        assert len(payload["assets"]) == 2  # input and demonstration both stay images
        assert all(asset["data_base64"] == image.url.split(",", 1)[1] for asset in payload["assets"].values())
        assert "Find the target" in str(payload["state"]) and "demos" in str(payload["state"])
        assert dict(request.headers)["TypeSafe-Danger-Zone-Images"] == "true"
        assert lm.history[-1]["usage"] == {}
    finally:
        await lm.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("media_location", ["none", "top_level", "nested_input", "nested_demo", "nested_both"])
async def test_native_state_serializes_inputs_and_demos(media_location, asynchronous, monkeypatch):
    from AlinGo-Prompt import lm15
    from AlinGo-Prompt._vendor.lm15.testing import FakeTransport
    from tests.clients.test_lm_engine_execution import DualResponse

    signature = AlinGo-Prompt.Signature({"metadata": (dict, AlinGo-Prompt.InputField()), "box": (AlinGo-Prompt.BoundingBox, AlinGo-Prompt.OutputField())})
    inputs = {"metadata": {"point": (9, 2), "date": date(2026, 10, 19), "tags": {"input"}}}
    demo = {"metadata": {"point": (4, 7), "date": date(2025, 3, 8), "tags": {"demo"}}, "box": AlinGo-Prompt.BoundingBox(**EDGES)}
    image = AlinGo-Prompt.Image(Image.new("RGB", (320, 280), "red"))
    if media_location == "top_level":
        signature = signature.prepend("photo", AlinGo-Prompt.InputField(), type_=AlinGo-Prompt.Image)
        inputs["photo"] = demo["photo"] = image
    if media_location in ("nested_input", "nested_both"):
        inputs["metadata"]["photos"] = [{"image": image}]
    if media_location in ("nested_demo", "nested_both"):
        demo["metadata"]["photos"] = [{"image": image}]
    response = {"model": "jev-1.14.0", "answers": {"box": {"type": "box", "box": EDGES, "confidence": 0.8}}}
    transport = FakeTransport([DualResponse(200, json.dumps(response).encode())])
    monkeypatch.setattr("AlinGo-Prompt.clients.execution.RouterConfig", lambda **kwargs: lm15.RouterConfig(**{**kwargs, "transport": transport}))
    lm = AlinGo-Prompt.LM("typesafe:jev-1.14.0", engine="lm15", api_key="test", cache=False, num_retries=0)
    predict = AlinGo-Prompt.Predict(signature)
    predict.demos = [AlinGo-Prompt.Example(**demo)]
    try:
        result = await predict.acall(lm=lm, **inputs) if asynchronous else predict(lm=lm, **inputs)
        assert result.box.model_dump() == EDGES
        request = lm.history[-1]["request"]
        assert len(request.messages) == 1
        assert request.messages[0].role == "user"
        parts = request.messages[0].parts
        payload = json.loads(transport.requests[0].body)
        if media_location != "none":
            image_count = 2 if media_location in ("top_level", "nested_both") else 1
            assert [part for part in parts if isinstance(part, ImagePart)] == image.format() * image_count
            assert all(isinstance(part, (TextPart, ImagePart)) for part in parts)
            assert "CUSTOM-TYPE" not in "".join(part.text for part in parts if isinstance(part, TextPart))
            assert len(payload["assets"]) == image_count
            assert all(asset["data_base64"] == image.url.split(",", 1)[1] for asset in payload["assets"].values())
        else:
            assert len(parts) == 1 and isinstance(parts[0], DataPart)
            assert parts[0].value == payload["state"]
            assert payload["state"]["inputs"]["metadata"] == {"point": [9, 2], "date": "2026-10-19", "tags": ["input"]}
            assert payload["state"]["demos"][0]["metadata"] == {"point": [4, 7], "date": "2025-03-08", "tags": ["demo"]}
            assert payload["state"]["demos"][0]["box"] == EDGES
    finally:
        await lm.aclose()


@pytest.mark.parametrize("annotation", [list[AlinGo-Prompt.BoundingBox], AlinGo-Prompt.BoundingBox | None, str])
def test_unsupported_native_outputs_are_refused(annotation, monkeypatch):
    from AlinGo-Prompt import lm15
    from AlinGo-Prompt._vendor.lm15.testing import FakeTransport

    transport = FakeTransport([])
    monkeypatch.setattr("AlinGo-Prompt.clients.execution.RouterConfig", lambda **kwargs: lm15.RouterConfig(**{**kwargs, "transport": transport}))
    signature = Locate.with_updated_fields("box", type_=annotation)
    lm = AlinGo-Prompt.LM("typesafe:jev-1.14.0", engine="lm15", api_key="test", cache=False, num_retries=0)
    try:
        with AlinGo-Prompt.context(adapter=None), pytest.raises(ValueError, match="Unsupported System One output"):
            AlinGo-Prompt.Predict(signature)(photo=AlinGo-Prompt.Image(Image.new("RGB", (320, 280))), lm=lm)
        assert not transport.requests
    finally:
        lm.close()


def test_adapter_rejects_invalid_generated_geometry():
    with pytest.raises(AlinGo-Prompt.AdapterParseError, match="top < bottom"):
        AlinGo-Prompt.JSONAdapter().parse(Locate, json.dumps({"box": {**EDGES, "top": 0.9}}))

    from tests.predict.test_decision_parameters import FakeClient

    signature = AlinGo-Prompt.Signature({"box": (AlinGo-Prompt.BoundingBox, AlinGo-Prompt.OutputField())})
    lm = FakeClient(answers={"box": {"box": {**EDGES, "top": 0.9}, "confidence": 0.8}})
    with pytest.raises(AlinGo-Prompt.AdapterParseError, match="top < bottom"):
        AlinGo-Prompt.Predict(signature)(lm=lm)
