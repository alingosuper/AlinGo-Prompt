import pydantic

import AlinGo-Prompt


def test_basic_extract_custom_type_from_annotation():
    class Event(AlinGo-Prompt.Type):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    class ExtractEvent(AlinGo-Prompt.Signature):
        """Extract all events from the email content."""

        email: str = AlinGo-Prompt.InputField()
        event: Event = AlinGo-Prompt.OutputField()

    assert AlinGo-Prompt.Type.extract_custom_type_from_annotation(ExtractEvent.output_fields["event"].annotation) == [Event]

    class ExtractEvents(AlinGo-Prompt.Signature):
        """Extract all events from the email content."""

        email: str = AlinGo-Prompt.InputField()
        events: list[Event] = AlinGo-Prompt.OutputField()

    assert AlinGo-Prompt.Type.extract_custom_type_from_annotation(ExtractEvents.output_fields["events"].annotation) == [Event]


def test_extract_custom_type_from_annotation_with_nested_type():
    class Event(AlinGo-Prompt.Type):
        event_name: str
        start_date_time: str
        end_date_time: str | None
        location: str | None

    class EventIdentifier(AlinGo-Prompt.Type):
        model_config = pydantic.ConfigDict(frozen=True)  # Make it hashable
        event_id: str
        event_name: str

    class ExtractEvents(AlinGo-Prompt.Signature):
        """Extract all events from the email content."""

        email: str = AlinGo-Prompt.InputField()
        events: list[dict[EventIdentifier, Event]] = AlinGo-Prompt.OutputField()

    assert AlinGo-Prompt.Type.extract_custom_type_from_annotation(ExtractEvents.output_fields["events"].annotation) == [
        EventIdentifier,
        Event,
    ]


def test_custom_input_preserves_text_continuation():
    from AlinGo-Prompt.lm15 import ContinuationState, TextPart

    source = TextPart(
        "annotated",
        continuation=(ContinuationState(provider="example", kind="opaque", data={"id": 7}),),
    )

    class Annotated(AlinGo-Prompt.Type):
        def format(self):
            return [source]

    class QA(AlinGo-Prompt.Signature):
        context: Annotated = AlinGo-Prompt.InputField()
        answer: str = AlinGo-Prompt.OutputField()

    lm = AlinGo-Prompt.utils.DummyLM([{"answer": "ok"}])
    with AlinGo-Prompt.context(lm=lm, adapter=AlinGo-Prompt.ChatAdapter()):
        assert AlinGo-Prompt.Predict(QA)(context=Annotated()).answer == "ok"

    assert source in lm.history[0]["request"].messages[-1].parts
