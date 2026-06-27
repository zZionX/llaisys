import argparse
import json
import mimetypes
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from nanonona.engine.engine import llm_engine
from nanonona.utils.config import Config
from nanonona.utils.sample_params import SamplingParams


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"


@dataclass
class ChatSession:
    messages: list[dict[str, str]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class GenerationJob:
    prompt: str
    params: SamplingParams
    event: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: BaseException | None = None


class BatchedLLMWorker:
    def __init__(self, batch_delay_s: float = 0.02, max_batch_size: int | None = None):
        self.engine = llm_engine()
        self.batch_delay_s = batch_delay_s
        self.max_batch_size = max_batch_size or Config.engine.max_num_running_seqs
        self.jobs: queue.Queue[GenerationJob | None] = queue.Queue()
        self.thread = threading.Thread(target=self._loop, name="llm-worker", daemon=True)
        self.thread.start()

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def generate(self, prompt: str, params: SamplingParams) -> dict[str, Any]:
        job = GenerationJob(prompt=prompt, params=params)
        self.jobs.put(job)
        job.event.wait()
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

    def _collect_batch(self, first: GenerationJob) -> list[GenerationJob]:
        batch = [first]
        deadline = time.monotonic() + self.batch_delay_s
        while len(batch) < self.max_batch_size:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout == 0:
                break
            try:
                job = self.jobs.get(timeout=timeout)
            except queue.Empty:
                break
            if job is None:
                self.jobs.put(None)
                break
            batch.append(job)
        return batch

    def _loop(self) -> None:
        while True:
            first = self.jobs.get()
            if first is None:
                return
            batch = self._collect_batch(first)
            try:
                outputs = self.engine.generate(
                    [job.prompt for job in batch],
                    [job.params for job in batch],
                )
                for job, output in zip(batch, outputs):
                    job.result = output
                    job.event.set()
            except BaseException as exc:
                for job in batch:
                    job.error = exc
                    job.event.set()


class ChatState:
    def __init__(self):
        self.sessions: dict[str, ChatSession] = {}
        self.lock = threading.Lock()
        self.worker = BatchedLLMWorker()

    def get_session(self, session_id: str | None) -> tuple[str, ChatSession]:
        if not session_id:
            session_id = uuid.uuid4().hex
        with self.lock:
            session = self.sessions.setdefault(session_id, ChatSession())
        return session_id, session

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.worker.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def fit_messages(self, messages: list[dict[str, str]], max_new_tokens: int) -> list[dict[str, str]]:
        prompt_budget = max(1, Config.engine.max_model_len - max_new_tokens)
        fitted = list(messages)
        while len(fitted) > 1:
            prompt = self.build_prompt(fitted)
            token_count = len(self.worker.tokenizer.encode(prompt, add_special_tokens=False))
            if token_count <= prompt_budget:
                return fitted
            fitted.pop(0)
            while fitted and fitted[0]["role"] != "user":
                fitted.pop(0)
        return fitted

    def clean_assistant_text(self, text: str) -> str:
        text = text.strip()
        for token in (
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
        ):
            text = text.replace(token, "")

        if "</think>" in text:
            final_text = text.rsplit("</think>", 1)[1].strip()
            if final_text:
                text = final_text
        elif "<think>" in text:
            text = text.replace("<think>", "").strip()

        return text.strip()


STATE: ChatState | None = None


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "NanononaChat/0.1"

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.write_json({"ok": True})
            return
        if self.path.startswith("/api/history"):
            session_id = self.query_param("session_id")
            session_id, session = self.state.get_session(session_id)
            with session.lock:
                messages = list(session.messages)
            self.write_json({"session_id": session_id, "messages": messages})
            return
        self.serve_static()

    def do_POST(self) -> None:
        if self.path == "/api/chat":
            self.handle_chat()
            return
        if self.path == "/api/clear":
            self.handle_clear()
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    @property
    def state(self) -> ChatState:
        assert STATE is not None
        return STATE

    def handle_chat(self) -> None:
        payload = self.read_json()
        message = str(payload.get("message", "")).strip()
        if not message:
            self.write_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
            return

        session_id, session = self.state.get_session(payload.get("session_id"))
        temperature = float(payload.get("temperature", 0.6))
        max_tokens = int(payload.get("max_tokens", 512))
        max_tokens = max(1, min(max_tokens, 2048))
        temperature = max(1.0e-5, min(temperature, 2.0))

        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        with session.lock:
            session.messages.append({"role": "user", "content": message})
            session.messages = self.state.fit_messages(session.messages, max_tokens)
            prompt = self.state.build_prompt(session.messages)
            output = self.state.worker.generate(prompt, params)
            assistant_text = output["text"]
            token_ids = output.get("token_ids") or []
            if token_ids:
                assistant_text = self.state.worker.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                ).strip()
            assistant_text = self.state.clean_assistant_text(assistant_text)
            session.messages.append({"role": "assistant", "content": assistant_text})
            messages = list(session.messages)

        self.write_json(
            {
                "session_id": session_id,
                "reply": assistant_text,
                "messages": messages,
                "token_ids": output.get("token_ids", []),
            }
        )

    def handle_clear(self) -> None:
        payload = self.read_json()
        session_id, session = self.state.get_session(payload.get("session_id"))
        with session.lock:
            session.messages.clear()
        self.write_json({"session_id": session_id, "messages": []})

    def serve_static(self) -> None:
        route = self.path.split("?", 1)[0]
        rel_path = "index.html" if route in ("", "/") else route.lstrip("/")
        file_path = (WEB_ROOT / rel_path).resolve()
        if not str(file_path).startswith(str(WEB_ROOT.resolve())) or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        data = self.rfile.read(content_length)
        return json.loads(data.decode("utf-8"))

    def write_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def query_param(self, name: str) -> str | None:
        if "?" not in self.path:
            return None
        query = self.path.split("?", 1)[1]
        for part in query.split("&"):
            key, _, value = part.partition("=")
            if key == name:
                return value or None
        return None

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the nanonona web chat server.")
    parser.add_argument("--host", default=Config.server.host)
    parser.add_argument("--port", default=Config.server.port, type=int)
    args = parser.parse_args()

    global STATE
    STATE = ChatState()

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Chat server running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
