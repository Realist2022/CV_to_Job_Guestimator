"""The gate between a locally stored RedactedCV and a file that ships.

RedactedCV.pii_spans holds detected PII verbatim, so these cover the two
distinct ways the serving copy could carry it out of the repo: the span
inventory itself, and a value the detector recorded but never actually
removed from `text`.
"""

import json

import pytest

from scripts.build_serving_cv import _find_leaks, _mask
from src.schemas.ingestion import WITHHELD_SPAN_TEXT, RedactedCV
from src.schemas.pii import TextSpan

RAW = "Jane Doe\n12 Example Street\nReact developer"
REDACTED = "[PERSON_NAME]\n[STREET_ADDRESS]\nReact developer"
SPANS = [
    TextSpan(kind="person_name", text="Jane Doe"),
    TextSpan(kind="street_address", text="12 Example Street"),
]


def _cv(redacted_text: str = REDACTED, spans: list[TextSpan] | None = None) -> RedactedCV:
    return RedactedCV.from_raw_text(
        raw_text=RAW,
        redacted_text=redacted_text,
        pii_spans=SPANS if spans is None else spans,
        pii_engine="presidio:test",
    )


def test_for_serving_withholds_span_values():
    serving = _cv().for_serving()

    assert [span.text for span in serving.pii_spans] == [WITHHELD_SPAN_TEXT] * 2
    assert "Jane Doe" not in serving.model_dump_json()
    assert "12 Example Street" not in serving.model_dump_json()


def test_for_serving_keeps_the_shape_of_the_redaction():
    # Withheld, not dropped: an empty list would read as "no PII found",
    # which is a different and false claim. Count and kind still hold.
    original = _cv()
    serving = original.for_serving()

    assert len(serving.pii_spans) == len(original.pii_spans)
    assert [span.kind for span in serving.pii_spans] == ["person_name", "street_address"]


def test_for_serving_leaves_everything_else_untouched():
    original = _cv()
    serving = original.for_serving()

    assert serving.cv_id == original.cv_id
    assert serving.text == original.text
    assert serving.pii_engine == original.pii_engine
    assert serving.ingestion_trace_id == original.ingestion_trace_id


def test_leak_check_passes_on_a_fully_redacted_cv():
    original = _cv()
    assert _find_leaks(original.for_serving().model_dump_json(), original) == []


def test_leak_check_catches_a_value_redaction_missed():
    # The span list says the address was detected, but `text` still has it
    # — for_serving() cannot fix that, so the build has to refuse.
    original = _cv(redacted_text="[PERSON_NAME]\n12 Example Street\nReact developer")

    leaks = _find_leaks(original.for_serving().model_dump_json(), original)

    assert len(leaks) == 1
    assert leaks[0].startswith("street_address:")


def test_leak_report_never_prints_the_value_it_found():
    original = _cv(redacted_text=RAW)

    report = " ".join(_find_leaks(original.for_serving().model_dump_json(), original))

    assert "Jane Doe" not in report
    assert "12 Example Street" not in report
    assert "J...(8 chars)" in report


@pytest.mark.parametrize("value", ["Jane Doe", "12 Example Street", "x"])
def test_mask_never_reveals_more_than_the_first_character(value):
    # A one-character value is fully disclosed by its own first character.
    # That is the design, not a gap: a single character is not identifying,
    # and the length is what makes the report useful for tracking the leak
    # down in the source document.
    masked = _mask(value)

    assert masked.startswith(value[:1])
    assert str(len(value)) in masked
    if len(value) > 1:
        assert value not in masked
        assert value[1:] not in masked


def test_serving_file_on_disk_carries_no_span_values():
    """serving/redacted_cv.json is committed and copied into the image, so
    this is the check that matters after any rebuild of it."""
    from scripts.build_serving_cv import OUT_PATH

    if not OUT_PATH.exists():
        pytest.skip("No serving CV built yet (scripts/build_serving_cv.py).")

    payload = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    assert payload["pii_spans"] == [
        {"kind": span["kind"], "text": WITHHELD_SPAN_TEXT} for span in payload["pii_spans"]
    ]
