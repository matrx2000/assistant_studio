#!/usr/bin/env python3
"""
PySide6 Chat UI for Ollama (OpenAI-compatible API) with tools + STREAMING
- Safe worker shutdown (abort/wait) to avoid "QThread: Destroyed while thread is still running"
- Live token streaming; text between <thinking>...</thinking> or <think>...</think> is italic
- Tools: add, read_file, write_file, list_models, set_model (optional)
"""

from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List
from urllib import request as _urlreq

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QMessageBox, QCheckBox,
)
from PySide6.QtGui import QTextCursor, QTextCharFormat

from openai import OpenAI

# ----------------------------
# Configuration
# ----------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "devstral:latest")
API_KEY = os.environ.get("OLLAMA_API_KEY", "local")  # value ignored by Ollama; client requires a string

# Active model pointer (updated by set_model tool)
CURRENT_MODEL = MODEL

def _ollama_host_base() -> str:
    u = OLLAMA_BASE_URL.rstrip("/")
    return u[:-3] if u.endswith("/v1") else u

# ----------------------------
# Tools
# ----------------------------
def add(a: float, b: float) -> str:
    return json.dumps({"sum": a + b})

def read_file(path: str, max_bytes: int = 4096) -> str:
    norm = os.path.normpath(path)
    if os.path.isabs(path) or ".." in norm.split(os.sep):
        return json.dumps({"error": "unsafe_path"})
    if not os.path.exists(norm):
        return json.dumps({"error": "not_found"})
    try:
        with open(norm, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(max_bytes)
        return json.dumps({"path": norm, "preview": data})
    except Exception as e:
        return json.dumps({"error": str(e)})

def write_file(path: str, content: str, mode: str = "overwrite") -> str:
    try:
        norm = os.path.normpath(path)
        if os.path.isabs(path) or ".." in norm.split(os.sep):
            return json.dumps({"error": "unsafe_path", "path": path})
        parent = os.path.dirname(norm)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if mode not in ("overwrite", "append"):
            return json.dumps({"error": "bad_mode", "allowed": ["overwrite", "append"]})
        m = "w" if mode == "overwrite" else "a"
        with open(norm, m, encoding="utf-8") as f:
            f.write(content)
        return json.dumps({"status": "ok", "path": norm, "mode": mode, "bytes_written": len(content)})
    except Exception as e:
        return json.dumps({"error": str(e), "path": path})

def list_models() -> str:
    # Try /v1/models via OpenAI client
    try:
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)
        data = client.models.list()
        names = []
        for m in data.data:
            name = getattr(m, "id", None) or getattr(m, "root", None) or getattr(m, "owned_by", None)
            if name:
                names.append({"name": name})
        if names:
            return json.dumps({"source": "v1/models", "models": names})
    except Exception:
        pass
    # Fallback to native /api/tags
    try:
        url = _ollama_host_base().rstrip("/") + "/api/tags"
        req = _urlreq.Request(url, method="GET")
        with _urlreq.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", "replace")
        obj = json.loads(raw)
        models = [{"name": m.get("name"), "size": m.get("size"), "modified_at": m.get("modified_at")}
                  for m in obj.get("models", [])]
        return json.dumps({"source": "api/tags", "models": models})
    except Exception as e:
        return json.dumps({"error": str(e)})

def set_model(model: str, ensure_loaded: bool = True, unload_previous: bool = False) -> str:
    """Switch CURRENT_MODEL; optionally warm it and unload previous (best-effort)."""
    global CURRENT_MODEL
    old = CURRENT_MODEL
    errors: List[str] = []
    CURRENT_MODEL = model
    if ensure_loaded:
        try:
            client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=API_KEY)
            _ = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=4,
            )
        except Exception as e:
            CURRENT_MODEL = old
            return json.dumps({"status": "failed", "error": f"ensure_failed: {e}", "old": old, "new": model})
    if unload_previous and old and old != model:
        host = _ollama_host_base().rstrip("/")
        for path, body in (("/api/unload", {"model": old}),
                           ("/api/stop", {"model": old}),
                           ("/api/stop", {"name": old})):
            try:
                req = _urlreq.Request(
                    host + path,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                _ = _urlreq.urlopen(req, timeout=3).read()
                break
            except Exception as e:
                errors.append(f"{path}:{e}")
    return json.dumps({"status": "ok", "old": old, "new": CURRENT_MODEL, "errors": errors})

FUNCTIONS: Dict[str, Callable[..., str]] = {
    "add": add,
    "read_file": read_file,
    "write_file": write_file,
    "list_models": list_models,
    "set_model": set_model,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "add",
        "description": "Add two numbers and return the sum.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a small UTF-8 text file (relative path, up to ~4KB).",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "max_bytes": {"type": "integer", "default": 4096, "minimum": 1}},
                       "required": ["path"], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write UTF-8 text to a file (relative path only). Choose overwrite or append.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"},
                                      "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"}},
                       "required": ["path", "content"], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "list_models",
        "description": "List available local models from the Ollama server.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "set_model",
        "description": "Switch the active chat model. Optionally ensure it is loaded and unload the previous one.",
        "parameters": {"type": "object",
                       "properties": {"model": {"type": "string"},
                                      "ensure_loaded": {"type": "boolean", "default": True},
                                      "unload_previous": {"type": "boolean", "default": False}},
                       "required": ["model"], "additionalProperties": False},
    }},
]

# ----------------------------
# Data classes
# ----------------------------
@dataclass
class ChatConfig:
    base_url: str = OLLAMA_BASE_URL
    model: str = CURRENT_MODEL
    api_key: str = API_KEY
    use_tools: bool = True

# ----------------------------
# Stream helpers
# ----------------------------
def _extract_stream_piece(chunk) -> str | None:
    try:
        if not chunk.choices:
            return None
        c = chunk.choices[0]
        if getattr(c, "delta", None) and getattr(c.delta, "content", None):
            return c.delta.content
        if getattr(c, "message", None) and getattr(c.message, "content", None):
            return c.message.content
        if getattr(c, "text", None):
            return c.text
    except Exception:
        return None
    return None

# ----------------------------
# Worker Thread (with abort)
# ----------------------------
class ChatWorker(QThread):
    result = Signal(str)
    error = Signal(str)
    tool_debug = Signal(str)
    stream_started = Signal()
    stream_delta = Signal(str)
    stream_finished = Signal()

    def __init__(self, messages: List[dict], cfg: ChatConfig, stream: bool = True):
        super().__init__()
        self.messages = messages[:]  # copy
        self.cfg = cfg
        self.stream = stream
        self._abort = False
        self._active_stream = None  # handle to streaming iterator

    def abort(self):
        self._abort = True
        # best-effort stop of streaming iterator
        try:
            if self._active_stream and hasattr(self._active_stream, "close"):
                self._active_stream.close()
        except Exception:
            pass

    def _stream_final_answer(self, client: OpenAI):
        self.stream_started.emit()
        buf = []
        self._active_stream = client.chat.completions.create(
            model=self.cfg.model, messages=self.messages, stream=True
        )
        try:
            for chunk in self._active_stream:
                if self._abort or self.isInterruptionRequested():
                    break
                piece = _extract_stream_piece(chunk)
                if piece:
                    buf.append(piece)
                    self.stream_delta.emit(piece)
        finally:
            self._active_stream = None
            self.stream_finished.emit()
        return "".join(buf).strip()

    def run(self):
        try:
            client = OpenAI(base_url=self.cfg.base_url, api_key=self.cfg.api_key)

            # First, detect tool calls (non-stream)
            kwargs = {"model": self.cfg.model, "messages": self.messages}
            if self.cfg.use_tools:
                kwargs["tools"] = TOOLS
                kwargs["tool_choice"] = "auto"

            first = client.chat.completions.create(**kwargs)
            if self._abort:  # if aborted during request
                return
            msg = first.choices[0].message

            # Tool calls?
            if getattr(msg, "tool_calls", None):
                self.messages.append({"role": "assistant", "content": None,
                                      "tool_calls": [tc for tc in msg.tool_calls]})
                for tc in msg.tool_calls:
                    if self._abort:
                        return
                    try:
                        name = tc.function.name
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception as e:
                        self.tool_debug.emit(f"Tool args parse error: {e}")
                        name, args = getattr(tc.function, "name", "unknown"), {}

                    fn = FUNCTIONS.get(name)
                    if not fn:
                        content = json.dumps({"error": f"unknown_tool:{name}"})
                    else:
                        try:
                            content = fn(**args)
                        except Exception as e:
                            content = json.dumps({"error": str(e)})

                    self.tool_debug.emit(f"Called tool '{name}' with args={args} -> {content[:200]}...")
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": content})

                # pick up model changes
                self.cfg.model = CURRENT_MODEL

                if self.stream and not self._abort:
                    final_text = self._stream_final_answer(client)
                    if not self._abort:
                        self.result.emit(final_text)
                elif not self._abort:
                    final = client.chat.completions.create(model=self.cfg.model, messages=self.messages)
                    out = (final.choices[0].message.content or "").strip()
                    self.result.emit(out)
                return

            # No tools; stream or not
            if self.stream and not self._abort:
                stream_kwargs = {"model": self.cfg.model, "messages": self.messages, "stream": True}
                if self.cfg.use_tools:
                    stream_kwargs.update({"tools": TOOLS, "tool_choice": "auto"})

                self.stream_started.emit()
                buf = []
                self._active_stream = client.chat.completions.create(**stream_kwargs)
                try:
                    for chunk in self._active_stream:
                        if self._abort or self.isInterruptionRequested():
                            break
                        piece = _extract_stream_piece(chunk)
                        if piece:
                            buf.append(piece)
                            self.stream_delta.emit(piece)
                finally:
                    self._active_stream = None
                    self.stream_finished.emit()
                if not self._abort:
                    self.result.emit("".join(buf).strip())
            elif not self._abort:
                out = (first.choices[0].message.content or "").strip()
                self.result.emit(out)

        except Exception as e:
            if not self._abort:
                self.error.emit(str(e))

# ----------------------------
# UI
# ----------------------------
class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ollama Chat (PySide6 + Tools)")
        self.resize(900, 680)

        self.messages: List[dict] = [
            {"role": "system", "content": "Be brief. If you need data, use the available tools."}
        ]

        # Widgets
        self.history = QTextEdit(); self.history.setReadOnly(True)
        self.input = QLineEdit(); self.input.setPlaceholderText("Type your message and press Enter...")
        self.send_btn = QPushButton("Send")
        self.model_label = QLabel(f"Model: {CURRENT_MODEL}")
        self.base_label = QLabel(f"Base URL: {OLLAMA_BASE_URL}")
        self.tools_checkbox = QCheckBox("Enable tools"); self.tools_checkbox.setChecked(True)
        self.stream_checkbox = QCheckBox("Stream"); self.stream_checkbox.setChecked(True)

        # Layouts
        top = QVBoxLayout(self); top.addWidget(self.history)
        bottom = QHBoxLayout(); bottom.addWidget(self.input, 1); bottom.addWidget(self.send_btn); top.addLayout(bottom)
        meta = QHBoxLayout()
        meta.addWidget(self.model_label); meta.addWidget(self.base_label); meta.addStretch(1)
        meta.addWidget(self.stream_checkbox); meta.addWidget(self.tools_checkbox); top.addLayout(meta)

        # Signals
        self.send_btn.clicked.connect(self.on_send)
        self.input.returnPressed.connect(self.on_send)

        self.append_history("system", "Chat ready. Try:\n"
                                      "  Add 41 and 1 using the 'add' tool.\n"
                                      "  list_models / set_model to switch models.")

        # state
        self._stream_active = False
        self._last_was_streamed = False
        self._thinking_italic = False
        self.worker: ChatWorker | None = None

    def append_history(self, role: str, text: str):
        role_tag = {"user": "You", "assistant": "Assistant", "system": "System", "tool": "Tool"}.get(role, role)
        self.history.append(f"<b>{role_tag}:</b> {self.escape_html(text)}")

    @staticmethod
    def escape_html(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

    # italic handling for <thinking>/<think>
    def _insert_stream_piece(self, piece: str):
        cursor = self.history.textCursor()
        i = 0
        while i < len(piece):
            next_pos, tag = len(piece), None
            for t in ("<thinking>", "</thinking>", "<think>", "</think>"):
                p = piece.find(t, i)
                if p != -1 and p < next_pos:
                    next_pos, tag = p, t
            if next_pos > i:
                chunk = piece[i:next_pos]
                fmt = QTextCharFormat(); fmt.setFontItalic(self._thinking_italic)
                cursor.insertText(chunk, fmt)
            i = next_pos
            if tag is None:
                break
            if tag in ("<thinking>", "<think>"):
                self._thinking_italic = True
            elif tag in ("</thinking>", "</think>"):
                self._thinking_italic = False
            i += len(tag)
        self.history.setTextCursor(cursor)
        self.history.ensureCursorVisible()

    def _refresh_model_label(self):
        self.model_label.setText(f"Model: {CURRENT_MODEL}")

    def on_send(self):
        content = self.input.text().strip()
        if not content:
            return

        # Abort any running worker (prevents overlap + shutdown errors)
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self.worker.wait(2000)

        self.append_history("user", content)
        self.input.clear()
        self.messages.append({"role": "user", "content": content})

        cfg = ChatConfig(base_url=OLLAMA_BASE_URL, model=CURRENT_MODEL,
                         api_key=API_KEY, use_tools=self.tools_checkbox.isChecked())

        self.worker = ChatWorker(self.messages, cfg, stream=self.stream_checkbox.isChecked())
        self.worker.result.connect(self.on_result)
        self.worker.error.connect(self.on_error)
        self.worker.tool_debug.connect(self.on_tool_debug)
        self.worker.stream_started.connect(self.on_stream_started)
        self.worker.stream_delta.connect(self.on_stream_delta)
        self.worker.stream_finished.connect(self.on_stream_finished)
        self.worker.finished.connect(self.on_worker_finished)

        # disable input during request
        self.send_btn.setEnabled(False); self.input.setEnabled(False)

        self.worker.start()

    # streaming handlers
    def on_stream_started(self):
        self._stream_active = True
        self._last_was_streamed = True
        self._thinking_italic = False
        self.history.append("<b>Assistant:</b> ")
        self.history.moveCursor(QTextCursor.End)
        self.history.ensureCursorVisible()

    def on_stream_delta(self, piece: str):
        self._insert_stream_piece(piece)

    def on_stream_finished(self):
        self._stream_active = False
        self.history.append("")  # newline
        self._refresh_model_label()

    def on_result(self, text: str):
        self.messages.append({"role": "assistant", "content": text})
        if self._last_was_streamed:
            self._last_was_streamed = False
            self._refresh_model_label()
            return
        self.append_history("assistant", text)
        self._refresh_model_label()

    def on_worker_finished(self):
        self.send_btn.setEnabled(True); self.input.setEnabled(True)

    def on_error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.send_btn.setEnabled(True); self.input.setEnabled(True)

    def on_tool_debug(self, info: str):
        self.append_history("tool", info)

    # Ensure clean shutdown to avoid "QThread destroyed while running"
    def closeEvent(self, event):
        try:
            if self.worker and self.worker.isRunning():
                self.worker.abort()
                self.worker.wait(2000)  # wait up to 2s
        finally:
            super().closeEvent(event)

# ----------------------------
# Main
# ----------------------------
def main():
    app = QApplication(sys.argv)
    w = ChatWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
