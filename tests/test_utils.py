import os

import pytest

from src import utils


def test_retry_call_succeeds_after_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    assert utils.retry_call(flaky, retries=3, delay=0) == "ok"
    assert calls["n"] == 3


def test_retry_call_gives_up():
    def always_fails():
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        utils.retry_call(always_fails, retries=2, delay=0)


def test_safe_call_never_raises():
    def boom():
        raise ValueError("bad")

    result, error = utils.safe_call(boom, default="fallback", error_message="Oops.")
    assert result == "fallback"
    assert "Oops." in error and "bad" in error


def test_safe_call_success():
    assert utils.safe_call(lambda x: x * 2, 4) == (8, None)


def test_session_dir_lifecycle():
    folder = utils.create_session_dir()
    path = utils.save_upload(folder, "../../evil name.PDF", b"data")
    assert os.path.dirname(path) == folder          # cannot escape the folder
    assert path.endswith(".pdf")
    utils.cleanup_session_dir(folder)
    assert not os.path.exists(folder)
    utils.cleanup_session_dir(None)                 # must not raise


def test_history_is_trimmed_and_prepended():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(10)]
    assert len(utils.trim_history(history, 4)) == 4
    seen = {}
    wrapped = utils.with_history(lambda p: seen.setdefault("p", p) and "ok", history)
    wrapped("PROMPT")
    assert "m9" in seen["p"] and "PROMPT" in seen["p"]
    assert "m0" not in seen["p"]


def test_with_history_passthrough_when_empty():
    fn = lambda p: p
    assert utils.with_history(fn, []) is fn


def test_embed_in_batches_respects_batch_size():
    sizes = []

    def fake_embed(batch):
        sizes.append(len(batch))
        return [[0.0]] * len(batch)

    out = utils.embed_in_batches(fake_embed, ["t"] * 150, batch_size=64)
    assert len(out) == 150 and sizes == [64, 64, 22]


class FakeStore:
    def __init__(self, results):
        self.results = results

    def search(self, q, top_k=4):
        return self.results

    def is_empty(self):
        return False


def test_thresholded_store_filters_far_chunks():
    store = FakeStore([{"id": "a", "distance": 0.4}, {"id": "b", "distance": 1.9}])
    kept = utils.ThresholdedStore(store, max_distance=1.0).search([0.0])
    assert [r["id"] for r in kept] == ["a"]
    assert utils.ThresholdedStore(store).is_empty() is False  # delegation works


def test_validate_fields_flags_invented_values():
    source = "Invoice INV-2026-0042 dated 2026-03-14. Total Amount Due: 1,200.00 EGP"
    fields = {"invoice_number": "INV-2026-0042", "total": "1200.00", "party": "Made Up Ltd.",
              "missing": None, "dates": ["2026-03-14"]}
    checked = utils.validate_fields(fields, source)
    assert checked["invoice_number"]["verified"]
    assert checked["total"]["verified"]           # comma-insensitive
    assert checked["dates"]["verified"]
    assert not checked["party"]["verified"]
    assert "missing" not in checked
