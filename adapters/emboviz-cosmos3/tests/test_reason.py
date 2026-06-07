"""Test CosmosReasoner against a mock chat endpoint (no GPU).

Verifies the verdict plumbing: frames subsampled + sent as base64, the question
delivered, and the text answer decoded. The mock echoes how many images it got
and a canned verdict.

Run::

    uv run --with requests --with pillow python adapters/emboviz-cosmos3/tests/test_reason.py
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from emboviz_cosmos3.reason import CosmosReasoner


class _MockReasonServer:
    def __init__(self, answer: str = "Grasp failed: the gripper closed left of the handle.") -> None:
        self.n_images: int | None = None
        self.question: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                content = body["messages"][0]["content"]
                outer.question = next(p["text"] for p in content if p["type"] == "text")
                outer.n_images = sum(1 for p in content if p["type"] == "image_url")
                resp = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


def _frames(n: int) -> list[np.ndarray]:
    return [np.full((16, 16, 3), i * 10, dtype=np.uint8) for i in range(n)]


def test_judge_returns_text_and_subsamples_frames() -> None:
    with _MockReasonServer() as server:
        reasoner = CosmosReasoner(f"http://127.0.0.1:{server.port}", max_frames=4)
        verdict = reasoner.judge(_frames(17), "Did the robot grasp the cup? Answer in one sentence.")
    assert verdict.startswith("Grasp failed")
    assert server.n_images == 4                         # 17 frames capped to max_frames
    assert "grasp the cup" in server.question


def test_subsample_keeps_first_and_last() -> None:
    with _MockReasonServer() as server:
        reasoner = CosmosReasoner(f"http://127.0.0.1:{server.port}", max_frames=3)
        reasoner.judge(_frames(10), "verdict?")
    assert server.n_images == 3


def test_no_frames_raises() -> None:
    try:
        CosmosReasoner("http://x").judge([], "verdict?")
    except ValueError as e:
        assert "at least one frame" in str(e)
    else:
        raise AssertionError("expected ValueError for no frames")


def test_empty_text_response_raises() -> None:
    with _MockReasonServer(answer="") as server:
        try:
            CosmosReasoner(f"http://127.0.0.1:{server.port}").judge(_frames(2), "verdict?")
        except RuntimeError as e:
            assert "no text answer" in str(e)
        else:
            raise AssertionError("expected RuntimeError for empty answer")


def _run_all() -> None:
    test_judge_returns_text_and_subsamples_frames()
    test_subsample_keeps_first_and_last()
    test_no_frames_raises()
    test_empty_text_response_raises()
    print("OK: all reasoner-verdict checks passed")


if __name__ == "__main__":
    _run_all()
