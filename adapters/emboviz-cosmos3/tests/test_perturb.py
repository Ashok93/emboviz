"""Test CosmosImageEditor against a mock vLLM-Omni chat endpoint (no GPU).

Proves the edit plumbing: PNG/base64 encode of the input image, the chat-style
payload (instruction text + image data URL + extra_body), and decode of the
edited image out of the OpenAI-style response. The mock inverts the image and
echoes the received instruction, so the test verifies both the round-trip and
that the instruction reached the server.

Run::

    uv run --with requests --with pillow python adapters/emboviz-cosmos3/tests/test_perturb.py
"""

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from emboviz_cosmos3.perturb import CosmosImageEditor


class _MockEditServer:
    def __init__(self) -> None:
        self.last_instruction: str | None = None
        self.last_extra_body: dict | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                content = body["messages"][0]["content"]
                outer.last_instruction = next(p["text"] for p in content if p["type"] == "text")
                outer.last_extra_body = body.get("extra_body")
                data_url = next(p["image_url"]["url"] for p in content if p["type"] == "image_url")

                from PIL import Image
                src = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB")
                edited = Image.fromarray(255 - np.asarray(src, dtype=np.uint8))  # "edit" = invert
                buf = io.BytesIO()
                edited.save(buf, format="PNG")
                out_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

                resp = json.dumps({
                    "choices": [{"message": {"content": [
                        {"type": "image_url", "image_url": {"url": out_url}},
                    ]}}]
                }).encode()
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


def test_edit_round_trips_and_sends_instruction() -> None:
    img = np.random.RandomState(0).randint(0, 256, size=(48, 64, 3), dtype=np.uint8)
    with _MockEditServer() as server:
        editor = CosmosImageEditor(f"http://127.0.0.1:{server.port}", seed=7)
        out = editor.edit(img, "  replace the cup with a rubber duck  ")

    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out, 255 - img)                      # the mock's "edit" decoded correctly
    assert server.last_instruction == "replace the cup with a rubber duck"  # trimmed, delivered
    assert server.last_extra_body["seed"] == 7
    assert server.last_extra_body["height"] == 48 and server.last_extra_body["width"] == 64


def test_empty_instruction_raises() -> None:
    try:
        CosmosImageEditor("http://x").edit(np.zeros((4, 4, 3), np.uint8), "   ")
    except ValueError as e:
        assert "non-empty instruction" in str(e)
    else:
        raise AssertionError("expected ValueError for empty instruction")


def test_bad_image_dtype_raises() -> None:
    try:
        CosmosImageEditor("http://x").edit(np.zeros((4, 4, 3), np.float32), "do it")
    except ValueError as e:
        assert "uint8 RGB" in str(e)
    else:
        raise AssertionError("expected ValueError for non-uint8 image")


def test_malformed_response_raises() -> None:
    img = np.zeros((8, 8, 3), np.uint8)

    class _NoImageServer:
        def __init__(self):
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_):
                    pass

                def do_POST(self):
                    length = int(self.headers["Content-Length"])
                    self.rfile.read(length)
                    resp = json.dumps({"choices": [{"message": {"content": "no image here"}}]}).encode()
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

    with _NoImageServer() as server:
        try:
            CosmosImageEditor(f"http://127.0.0.1:{server.port}").edit(img, "edit it")
        except RuntimeError as e:
            assert "did not contain an edited image" in str(e)
        else:
            raise AssertionError("expected RuntimeError for missing image in response")


def _run_all() -> None:
    test_edit_round_trips_and_sends_instruction()
    test_empty_instruction_raises()
    test_bad_image_dtype_raises()
    test_malformed_response_raises()
    print("OK: all image-edit (perturbation) checks passed")


if __name__ == "__main__":
    _run_all()
