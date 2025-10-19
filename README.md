# Assistant studio - PySide6 + Ollama Chat (Tools, Streaming, Model Switching)

Minimal desktop chat app using **PySide6** that talks to a **local Ollama** server via its **OpenAI‑compatible** API. It demonstrates:

* **Tool/function calling** (the model can call Python functions like `add`, `read_file`, `write_file`, `list_models`, `set_model`).
* **Live streaming** of tokens in the UI (with optional italic styling for `<thinking>` blocks).
* **Runtime model switching** from chat (list available models, switch, optionally unload previous).
* **Safe worker shutdown** to avoid `QThread: Destroyed while thread is still running`.

---

## Features

* Connects to Ollama at `http://127.0.0.1:11434/v1` (override with env vars).
* Default model `devstral:latest` (override with `OLLAMA_MODEL`).
* Non‑blocking UI (worker thread) with safe abort/wait.
* Streaming checkbox: see tokens as they arrive; text inside `<thinking>…</thinking>` (or `<think>…</think>`) renders *italic*.
* Tool calling:

  * `add(a, b)` → returns JSON `{ "sum": a+b }`
  * `read_file(path, max_bytes=4096)` → safely reads a **small** UTF‑8 file (**relative paths only**)
  * `write_file(path, content, mode)` → write text (`overwrite`/`append`, creates file if missing)
  * `list_models()` → list local models from Ollama
  * `set_model(model, ensure_loaded=true, unload_previous=false)` → switch active model; can warm it and unload the old one

---

## Prerequisites

* Python 3.10+
* Ollama running locally and reachable at `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434/v1`).
* A pulled chat model (e.g., `devstral:latest`).

Quick check:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

---

## Install & Run

### 1) Create & activate a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
py -3 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
# If you get a script policy error:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

**Windows (CMD):**

```cmd
py -3 -m venv .venv
.\\.venv\\Scripts\\activate.bat
```

**macOS / Linux (bash/zsh):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

(Optional) Upgrade pip:

```bash
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
python app.py
```

### 4) Deactivate the venv when done

```bash
deactivate
```

To use the app later, re-activate the venv first (PowerShell: `./.venv/Scripts/Activate.ps1`, macOS/Linux: `source .venv/bin/activate`).

### Environment variables (optional)

* `OLLAMA_BASE_URL` (default: `http://127.0.0.1:11434/v1`)
* `OLLAMA_MODEL` (default: `devstral:latest`)
* `OLLAMA_API_KEY` (default: `local`) — Ollama ignores the actual value; client requires a string.

Windows PowerShell example:

```powershell
$env:OLLAMA_BASE_URL = "http://192.168.0.121:11434/v1"
$env:OLLAMA_MODEL    = "devstral:latest"
python app.py
```

Linux/macOS:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
export OLLAMA_MODEL=devstral:latest
python app.py
```

---

## Tools available & how to use them

The model **decides** when to call a tool. To increase reliability, state it explicitly in your prompt (e.g., “use the `read_file` tool…”). All file tools use **relative paths** for safety.

### `add(a:number, b:number) → { sum }`

* Purpose: quick arithmetic via a tool call.
* Example prompt:

  ```text
  Add 41 and 1 using the 'add' tool and report only the result.
  ```

### `read_file(path:string, max_bytes:int=4096) → { path, preview }`

* Purpose: preview *small* text files.
* Safety: blocks absolute paths and parent traversal (`..`).
* Example (project contains `test.txt` with two lines):

  ```text
  Use the 'read_file' tool on 'test.txt' (max_bytes=128) and return both lines exactly.
  ```
* First‑line only variant:

  ```text
  Use the 'read_file' tool on 'test.txt' (max_bytes=128) and return only the first line.
  ```

### `write_file(path:string, content:string, mode:"overwrite"|"append"="overwrite") → { status, path, bytes_written }`

* Purpose: write text files; creates the file if missing.
* Examples:

  * Overwrite with a haiku:

    ```text
    Write a 3‑line haiku about autumn. Then use the 'write_file' tool with path='poems/haiku.txt' and mode='overwrite' to store it. Confirm when saved.
    ```
  * Append another haiku:

    ```text
    Compose a new haiku about dawn. Use the 'write_file' tool with path='poems/haiku.txt' and mode='append' to add it after a blank line. Confirm when appended.
    ```

### `list_models() → { models:[{name,...}], source }`

* Purpose: discover available local models.
* Example prompt:

  ```text
  Call the 'list_models' tool and show just the model names.
  ```

### `set_model(model:string, ensure_loaded:boolean=true, unload_previous:boolean=false) → { old, new }`

* Purpose: switch the active model; optionally warm it and unload previous.
* Examples:

  ```text
  Use the 'set_model' tool with model='qwen3-coder:30b' and ensure_loaded=true. Confirm the active model.
  ```

  ```text
  Call 'set_model' with model='devstral:latest', ensure_loaded=true, unload_previous=true. Then state the active model.
  ```

---

## Streaming

* Toggle **Stream** to see tokens as they arrive.
* Text enclosed by `<thinking>…</thinking>` or `<think>…</think>` is rendered *italic* in the transcript.

---

## How tool calling works (short)

1. The app sends your message + JSON tool schemas to `/v1/chat/completions`.
2. If the model needs data/action, it returns `tool_calls` (function + JSON args).
3. Python executes the function and returns a `role="tool"` message with the result tied to the `tool_call_id`.
4. The model produces the final answer (streamed if enabled).

> Tool calling support depends on the specific model and Ollama version. If tools don’t trigger, strengthen the system instruction or try another model.

---

## Safety notes

* `read_file` & `write_file` accept **relative paths only**. If you need absolute paths, extend with an *allowed‑roots* mechanism.
* Any tool that touches filesystem/shell is sensitive — validate inputs.

---

## Troubleshooting

* **No streaming text**: ensure you’re using a **client** address (e.g., `127.0.0.1`) and your Ollama build streams. Test with `curl -N ... "stream":true`.
* **Double responses**: streaming already rendered; the app suppresses duplicate final print.
* **QThread error**: the app aborts and waits for the worker on close/send; this resolves `QThread: Destroyed while thread is still running`.
* **Windows + WSL/Docker**: if the UI can’t reach Ollama on Windows, set `OLLAMA_BASE_URL` to the Windows host IP or `http://host.docker.internal:11434/v1`.

---

## License

MIT (do whatever you want, no warranty).
