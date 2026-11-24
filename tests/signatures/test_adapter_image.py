import base64
import os
import tempfile
import warnings
from io import BytesIO

import pydantic
import pytest
import requests
from PIL import Image as PILImage

import AlinGo-Prompt
from AlinGo-Prompt.adapters.types.image import encode_image
from AlinGo-Prompt.utils.dummies import DummyLM


@pytest.fixture
def sample_pil_image():
    """Fixture to provide a sample image for testing"""
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    response = requests.get(url)
    response.raise_for_status()
    return PILImage.open(BytesIO(response.content))


@pytest.fixture
def sample_AlinGo-Prompt_image_download():
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    return AlinGo-Prompt.Image.from_url(url)


@pytest.fixture
def sample_url():
    return "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"


@pytest.fixture
def sample_AlinGo-Prompt_image_no_download():
    return AlinGo-Prompt.Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")


def count_messages_with_image_url_pattern(messages):
    """Count lm15 image parts (as recorded in LM history) addressed by a URL or inline data."""
    pattern = {"type": "image", "media_type": lambda x: isinstance(x, str)}

    try:

        def check_pattern(obj, pattern):
            if isinstance(pattern, dict):
                if not isinstance(obj, dict):
                    return False
                return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
            if callable(pattern):
                return pattern(obj)
            return obj == pattern

        def count_patterns(obj, pattern):
            count = 0
            if check_pattern(obj, pattern):
                count += 1
            if isinstance(obj, dict):
                count += sum(count_patterns(v, pattern) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                count += sum(count_patterns(v, pattern) for v in obj)
            return count

        return count_patterns(messages, pattern)
    except Exception:
        return 0


def setup_predictor(signature, expected_output):
    """Helper to set up a predictor with DummyLM"""
    lm = DummyLM([expected_output])
    AlinGo-Prompt.configure(lm=lm)
    return AlinGo-Prompt.Predict(signature), lm


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "probabilistic_classification",
            "signature": "image: AlinGo-Prompt.Image, class_labels: list[str] -> probabilities: dict[str, float]",
            "inputs": {"image": "https://example.com/dog.jpg", "class_labels": ["dog", "cat", "bird"]},
            "key_output": "probabilities",
            "expected": {"probabilities": {"dog": 0.8, "cat": 0.1, "bird": 0.1}},
        },
        {
            "name": "image_to_code",
            "signature": "ui_image: AlinGo-Prompt.Image, target_language: str -> generated_code: str",
            "inputs": {"ui_image": "https://example.com/button.png", "target_language": "HTML"},
            "key_output": "generated_code",
            "expected": {"generated_code": "<button>Click me</button>"},
        },
        {
            "name": "bbox_detection",
            "signature": "image: AlinGo-Prompt.Image -> bboxes: list[Tuple[int, int, int, int]]",
            "inputs": {"image": "https://example.com/image.jpg"},
            "key_output": "bboxes",
            "expected": {"bboxes": [(10, 20, 30, 40), (50, 60, 70, 80)]},
        },
        {
            "name": "multilingual_caption",
            "signature": "image: AlinGo-Prompt.Image, languages: list[str] -> captions: dict[str, str]",
            "inputs": {"image": "https://example.com/dog.jpg", "languages": ["en", "es", "fr"]},
            "key_output": "captions",
            "expected": {
                "captions": {"en": "A golden retriever", "es": "Un golden retriever", "fr": "Un golden retriever"}
            },
        },
    ],
)
def test_basic_image_operations(test_case):
    """Consolidated test for basic image operations"""
    predictor, lm = setup_predictor(test_case["signature"], test_case["expected"])

    # Convert string URLs to AlinGo-Prompt.Image objects
    inputs = {
        k: AlinGo-Prompt.Image(v) if isinstance(v, str) and k in ["image", "ui_image"] else v
        for k, v in test_case["inputs"].items()
    }

    result = predictor(**inputs)

    # Check result based on output field name
    output_field = next(f for f in ["probabilities", "generated_code", "bboxes", "captions"] if hasattr(result, f))
    assert getattr(result, output_field) == test_case["expected"][test_case["key_output"]]
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1


@pytest.mark.parametrize(
    "image_input,description",
    [
        ("pil_image", "PIL Image"),
        ("encoded_pil_image", "encoded PIL image string"),
        ("AlinGo-Prompt_image_download", "AlinGo-Prompt.Image with download=True"),
        ("AlinGo-Prompt_image_no_download", "AlinGo-Prompt.Image without download"),
    ],
)
def test_image_input_formats(
    request, sample_pil_image, sample_AlinGo-Prompt_image_download, sample_AlinGo-Prompt_image_no_download, image_input, description
):
    """Test different input formats for image fields"""
    signature = "image: AlinGo-Prompt.Image, class_labels: list[str] -> probabilities: dict[str, float]"
    expected = {"probabilities": {"dog": 0.8, "cat": 0.1, "bird": 0.1}}
    predictor, lm = setup_predictor(signature, expected)

    input_map = {
        "pil_image": sample_pil_image,
        "encoded_pil_image": encode_image(sample_pil_image),
        "AlinGo-Prompt_image_download": sample_AlinGo-Prompt_image_download,
        "AlinGo-Prompt_image_no_download": sample_AlinGo-Prompt_image_no_download,
    }

    actual_input = input_map[image_input]
    # TODO(isaacbmiller): Support the cases without direct AlinGo-Prompt.Image coercion
    if image_input in ["pil_image", "encoded_pil_image"]:
        pytest.xfail(f"{description} not fully supported without AlinGo-Prompt.Image coercion")

    result = predictor(image=actual_input, class_labels=["dog", "cat", "bird"])
    assert result.probabilities == expected["probabilities"]
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1


def test_predictor_save_load(sample_url, sample_pil_image):
    """Test saving and loading predictors with image fields"""
    signature = "image: AlinGo-Prompt.Image -> caption: str"
    examples = [
        AlinGo-Prompt.Example(image=AlinGo-Prompt.Image(sample_url), caption="Example 1"),
        AlinGo-Prompt.Example(image=sample_pil_image, caption="Example 2"),
    ]

    predictor, lm = setup_predictor(signature, {"caption": "A golden retriever"})
    optimizer = AlinGo-Prompt.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = AlinGo-Prompt.Predict(signature)
        loaded_predictor.load(temp_file.name)

    loaded_predictor(image=AlinGo-Prompt.Image("https://example.com/dog.jpg"))
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 2
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])


def test_save_load_complex_default_types():
    """Test saving and loading predictors with complex default types (lists of images)"""
    examples = [
        AlinGo-Prompt.Example(
            image_list=[
                AlinGo-Prompt.Image("https://example.com/dog.jpg"),
                AlinGo-Prompt.Image("https://example.com/cat.jpg"),
            ],
            caption="Example 1",
        ).with_inputs("image_list"),
    ]

    class ComplexTypeSignature(AlinGo-Prompt.Signature):
        image_list: list[AlinGo-Prompt.Image] = AlinGo-Prompt.InputField(desc="A list of images")
        caption: str = AlinGo-Prompt.OutputField(desc="A caption for the image list")

    predictor, lm = setup_predictor(ComplexTypeSignature, {"caption": "A list of images"})
    optimizer = AlinGo-Prompt.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = AlinGo-Prompt.Predict(ComplexTypeSignature)
        loaded_predictor.load(temp_file.name)

    result = loaded_predictor(**examples[0].inputs())
    assert result.caption == "A list of images"
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])


class BasicImageSignature(AlinGo-Prompt.Signature):
    """Basic signature with a single image input"""

    image: AlinGo-Prompt.Image = AlinGo-Prompt.InputField()
    output: str = AlinGo-Prompt.OutputField()


class ImageListSignature(AlinGo-Prompt.Signature):
    """Signature with a list of images input"""

    image_list: list[AlinGo-Prompt.Image] = AlinGo-Prompt.InputField()
    output: str = AlinGo-Prompt.OutputField()


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "basic_AlinGo-Prompt_signature",
            "signature_class": BasicImageSignature,
            "inputs": {"image": "https://example.com/dog.jpg"},
            "expected": {"output": "A dog photo"},
            "expected_image_urls": 2,
        },
        {
            "name": "list_AlinGo-Prompt_signature",
            "signature_class": ImageListSignature,
            "inputs": {"image_list": ["https://example.com/dog.jpg", "https://example.com/cat.jpg"]},
            "expected": {"output": "Multiple photos"},
            "expected_image_urls": 4,
        },
    ],
)
def test_save_load_complex_types(test_case):
    """Test saving and loading predictors with complex types"""
    signature_cls = test_case["signature_class"]

    # Convert string URLs to AlinGo-Prompt.Image objects in input
    processed_input = {}
    for key, value in test_case["inputs"].items():
        if isinstance(value, str) and "http" in value:
            processed_input[key] = AlinGo-Prompt.Image(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            processed_input[key] = [AlinGo-Prompt.Image(url) for url in value]
        else:
            processed_input[key] = value

    # Create example and predictor
    examples = [AlinGo-Prompt.Example(**processed_input, **test_case["expected"]).with_inputs(*processed_input.keys())]

    predictor, lm = setup_predictor(signature_cls, test_case["expected"])
    optimizer = AlinGo-Prompt.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    # Test save and load
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = AlinGo-Prompt.Predict(signature_cls)
        loaded_predictor.load(temp_file.name)

    # Run prediction
    result = loaded_predictor(**processed_input)

    # Verify output matches expected
    for key, value in test_case["expected"].items():
        assert getattr(result, key) == value

    # Verify correct number of image URLs in messages
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == test_case["expected_image_urls"]
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])


def test_save_load_pydantic_model():
    """Test saving and loading predictors with pydantic models"""

    class ImageModel(pydantic.BaseModel):
        image: AlinGo-Prompt.Image
        image_list: list[AlinGo-Prompt.Image] | None = None
        output: str

    class PydanticSignature(AlinGo-Prompt.Signature):
        model_input: ImageModel = AlinGo-Prompt.InputField()
        output: str = AlinGo-Prompt.OutputField()

    # Create model instance
    model_input = ImageModel(
        image=AlinGo-Prompt.Image("https://example.com/dog.jpg"),
        image_list=[AlinGo-Prompt.Image("https://example.com/cat.jpg")],
        output="Multiple photos",
    )

    # Create example and predictor
    examples = [AlinGo-Prompt.Example(model_input=model_input, output="Multiple photos").with_inputs("model_input")]

    predictor, lm = setup_predictor(PydanticSignature, {"output": "Multiple photos"})
    optimizer = AlinGo-Prompt.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    # Test save and load
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = AlinGo-Prompt.Predict(PydanticSignature)
        loaded_predictor.load(temp_file.name)

    # Run prediction
    result = loaded_predictor(model_input=model_input)

    # Verify output matches expected
    assert result.output == "Multiple photos"
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])


def test_optional_image_field():
    """Test that optional image fields are not required"""

    class OptionalImageSignature(AlinGo-Prompt.Signature):
        image: AlinGo-Prompt.Image | None = AlinGo-Prompt.InputField()
        output: str = AlinGo-Prompt.OutputField()

    predictor, lm = setup_predictor(OptionalImageSignature, {"output": "Hello"})
    result = predictor(image=None)
    assert result.output == "Hello"
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 0


def test_pdf_url_support():
    """Test support for PDF files from URLs"""
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Download the PDF into a AlinGo-Prompt.Image data URI.
    pdf_image = AlinGo-Prompt.Image.from_url(pdf_url)

    # The data URI should contain application/pdf in the MIME type
    assert "data:application/pdf" in pdf_image.url
    assert ";base64," in pdf_image.url

    # Test using it in a predictor
    class PDFSignature(AlinGo-Prompt.Signature):
        document: AlinGo-Prompt.Image = AlinGo-Prompt.InputField(desc="A PDF document")
        summary: str = AlinGo-Prompt.OutputField(desc="A summary of the PDF")

    predictor, lm = setup_predictor(PDFSignature, {"summary": "This is a dummy PDF"})
    result = predictor(document=pdf_image)

    assert result.summary == "This is a dummy PDF"
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1

    # Ensure the URL was properly expanded in messages
    messages_str = str(lm.history[-1]["messages"])
    assert "application/pdf" in messages_str


def test_different_mime_types():
    """Test support for different file types and MIME type detection"""
    # Test with various file types
    file_urls = {
        "pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "image": "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg",
    }

    expected_mime_types = {
        "pdf": "application/pdf",
        "image": "image/jpeg",
    }

    for file_type, url in file_urls.items():
        # Download and encode
        encoded = AlinGo-Prompt.Image.from_url(url).url

        # Check for correct MIME type in the encoded data - using 'in' instead of startswith
        # to account for possible parameters in the MIME type
        assert f"data:{expected_mime_types[file_type]}" in encoded
        assert ";base64," in encoded


def test_mime_type_from_response_headers():
    """Test that MIME types from response headers are correctly used"""
    # This URL returns proper Content-Type header
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Make an actual request to get the content type from headers
    response = requests.get(pdf_url)
    expected_mime_type = response.headers.get("Content-Type", "")

    # Should be application/pdf or similar
    assert "pdf" in expected_mime_type.lower()

    # Encode with download to test MIME type from headers
    encoded = AlinGo-Prompt.Image.from_url(pdf_url).url

    # The encoded data should contain the correct MIME type
    assert "application/pdf" in encoded
    assert ";base64," in encoded


def test_pdf_from_file():
    """Test handling a PDF file from disk"""
    # Download a PDF to a temporary file
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    try:
        # Create a AlinGo-Prompt.Image from the file
        pdf_image = AlinGo-Prompt.Image.from_path(tmp_file_path)

        # The constructor encodes the file into a data URI we can inspect directly
        assert "data:application/pdf" in pdf_image.url
        assert ";base64," in pdf_image.url

        # Test the image in a predictor
        class FilePDFSignature(AlinGo-Prompt.Signature):
            document: AlinGo-Prompt.Image = AlinGo-Prompt.InputField(desc="A PDF document from file")
            summary: str = AlinGo-Prompt.OutputField(desc="A summary of the PDF")

        predictor, lm = setup_predictor(FilePDFSignature, {"summary": "This is a PDF from file"})
        result = predictor(document=pdf_image)

        assert result.summary == "This is a PDF from file"
        assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_file_path)
        except Exception:
            pass


def test_image_repr():
    """Test string representation of Image objects"""
    url_image = AlinGo-Prompt.Image("https://example.com/dog.jpg")
    assert str(url_image) == (
        "<<CUSTOM-TYPE-START-IDENTIFIER>>"
        '[{"type": "image", "media_type": "image/jpeg", "url": "https://example.com/dog.jpg"}]'
        "<<CUSTOM-TYPE-END-IDENTIFIER>>"
    )
    assert repr(url_image) == "Image(url='https://example.com/dog.jpg')"

    sample_pil = PILImage.new("RGB", (60, 30), color="red")
    pil_image = AlinGo-Prompt.Image(sample_pil)
    assert str(pil_image).startswith('<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "image", "media_type": "image/png", "data": "')
    assert str(pil_image).endswith("<<CUSTOM-TYPE-END-IDENTIFIER>>")
    assert '"data": "' in str(pil_image)


def test_image_constructor_supports_positional_source_and_url_keyword():
    source = "https://example.com/dog.jpg"

    assert AlinGo-Prompt.Image(source).url == source
    assert AlinGo-Prompt.Image(url=source).url == source

    with pytest.raises(pydantic.ValidationError, match="source"):
        AlinGo-Prompt.Image(source=source)

    with pytest.raises(TypeError, match="both `source` and `url`"):
        AlinGo-Prompt.Image(source, url=source)


def test_canonical_factories_do_not_warn(tmp_path):
    tmp_file = tmp_path / "test.png"
    tmp_file.write_bytes(b"pngdata")

    with pytest.MonkeyPatch.context() as monkeypatch:
        response = type("Response", (), {"content": b"pngdata", "headers": {"Content-Type": "image/png"}})()
        response.raise_for_status = lambda: None
        monkeypatch.setattr("AlinGo-Prompt.adapters.types.image.requests.get", lambda *args, **kwargs: response)
        AlinGo-Prompt.Image.from_url("https://example.com/dog.jpg")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        AlinGo-Prompt.Image.from_path(str(tmp_file))


@pytest.mark.parametrize("name", ["from_file", "from_PIL"])
def test_removed_image_factory_aliases(name):
    with pytest.raises(AttributeError, match=name):
        getattr(AlinGo-Prompt.Image, name)


def test_pil_image_constructor_preserves_pixels():
    sample_pil = PILImage.new("RGB", (10, 10), color="blue")

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        image = AlinGo-Prompt.Image(sample_pil)

    assert image.url.startswith("data:image/png;base64,")
    decoded = PILImage.open(BytesIO(base64.b64decode(image.url.split(",", 1)[1])))
    assert decoded.size == (10, 10)
    assert decoded.getpixel((0, 0)) == (0, 0, 255)


def test_invalid_string_format():
    """A string that is neither a data URI nor a URL raises."""
    invalid_string = "this_is_not_a_url_or_file"

    # Should raise a ValueError and not pass the string through
    with pytest.raises(ValueError, match="Unrecognized"):
        AlinGo-Prompt.Image(invalid_string)


def test_positional_local_path_hard_breaks(tmp_path, monkeypatch):
    """Image(path) raises pointing at from_path, and never touches the filesystem to decide."""
    tmp_file = tmp_path / "photo.png"
    tmp_file.write_bytes(b"pngdata")

    monkeypatch.setattr("builtins.open", lambda *a, **k: pytest.fail("Image(path) read the filesystem"))

    with pytest.raises(ValueError, match=r"Image\.from_path"):
        AlinGo-Prompt.Image(str(tmp_file))


@pytest.mark.parametrize("source", ["https://example.com/dog.jpg", PILImage.new("RGB", (2, 1), color="red")])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"download": True},
        {"download": False},
        {"download": None},
        {"verify": True},
        {"verify": False},
        {"verify": None},
        {"download": True, "verify": False},
    ],
)
def test_image_constructor_rejects_removed_keywords(source, kwargs):
    with pytest.raises(TypeError, match=r"no longer accepts.*Image\.from_url"):
        AlinGo-Prompt.Image(source, **kwargs)


def test_from_path_missing_file():
    with pytest.raises(ValueError, match="File not found"):
        AlinGo-Prompt.Image.from_path("/nonexistent/image.png")


def test_unidentifiable_bytes_raise_value_error():
    with pytest.raises(ValueError, match="could not be identified as an image"):
        AlinGo-Prompt.Image(b"not an image")


def test_image_to_pil_decodes_embedded_images_only(monkeypatch):
    image = AlinGo-Prompt.Image(PILImage.new("RGB", (6, 4), (255, 0, 0)))
    pil = image.to_pil()
    assert pil.size == (6, 4) and pil.getpixel((0, 0)) == (255, 0, 0)
    assert AlinGo-Prompt.Image(pil).url == image.url  # AlinGo-Prompt.Image(pil_image) goes the other way

    with pytest.raises(ValueError, match="needs an embedded base64 image"):
        AlinGo-Prompt.Image("https://example.com/x.png").to_pil()
    monkeypatch.setattr(AlinGo-Prompt.adapters.types.image, "PIL_AVAILABLE", False)
    with pytest.raises(ImportError, match="Pillow is required"):
        image.to_pil()
