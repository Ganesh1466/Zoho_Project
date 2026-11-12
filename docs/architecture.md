# LocalMind architecture

## Current runnable architecture

```text
Terminal (chat.py)                   Browser (web/index.html)
        |                                      |
        | Python function calls               | HTTP on 127.0.0.1
        |                                      v
        +----------------------------> web_chat.py
                                               |
                                     localmind.LocalMindEngine
                                               |
                              Per-session ConversationHistory + lock
                                               |
                              OpenVINO GenAI LLMPipeline (CPU)
                                               |
                                Local Qwen2.5 3B INT4 OpenVINO IR
```

`LocalMindEngine` constructs the pipeline once per process, reuses it across
requests, and returns the runtime's performance metrics. Each request uses a
fresh OpenVINO `ChatHistory` containing a general-purpose system prompt, a
token-counted context window, and the selected prior turns. The source history
is kept separately per session, with up to eight recent turns and compact
user/assistant excerpts from older turns; the input window is capped at 8,192
tokens, below the model's 32,768-token context limit. EOS handling uses the
model tokenizer's end-of-turn token, and requests that hit their output cap are
flagged as potentially incomplete.

A generation lock serializes access to the shared pipeline. The web server
keeps UUID-keyed in-memory session states with per-session locks and a
32-session cap. `POST /api/reset` clears and releases a session. The browser
keeps displayed conversations in local storage, but server-side context is
process-memory only and is lost when the server restarts. The server remains
bound to loopback.

`download_model.py` is a separate optional utility. The original `test_model.py`
and `test_genai.py` remain unchanged and continue to demonstrate single-prompt
inference.

## Security and deployment boundary

The current interface has no authentication, authorization, audit logging,
durable retention policy, CSRF protection, or shared-host configuration. It is
a local development/demo interface, not a production company service. Browser
transcripts are retained in browser local storage; backend inference context is
retained in server memory for the process lifetime unless reset or the server
stops. When 32 session states are active, new sessions are rejected rather than
silently evicting existing context. Do not expose the service to a company
network without adding and testing appropriate controls.

## C++ prompt-to-response flow

The separate `cpp_migration/main.cpp` command-line application accepts one
prompt and a path to the existing model:

```text
Terminal prompt
    -> C++ main.cpp
    -> OpenVINO GenAI LLMPipeline (model + tokenizer)
    -> Qwen2.5 3B INT4 OpenVINO model on CPU
    -> Generated response in terminal
```

This is a basic one-prompt application, not a replacement for the Python
terminal or browser chat. It compiles against the OpenVINO GenAI C++ SDK, but
model-backed inference still needs a successful run under sufficient free
memory.
