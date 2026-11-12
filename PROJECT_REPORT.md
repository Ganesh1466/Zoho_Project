# LocalMind Project Report

## Executive summary

LocalMind runs the existing Qwen2.5 3B Instruct INT4 model locally through
OpenVINO GenAI on CPU. The Python terminal and browser interfaces remain the
primary supported chat clients. Work in this project improved conversation
context handling, per-session isolation, answer-limit reporting, and browser
Markdown rendering without replacing the model.

The recorded evaluations show that a 128-token output limit can still truncate
answers. Prompt changes helped some responses, but the latest prompt revision
has not had a full evaluation. No controlled before-and-after performance or
RAM comparison is available.

## Existing architecture

```text
Terminal: chat.py ─────────────────────────┐
                                           v
Browser: web/index.html -> web_chat.py -> localmind.py
                                           |
                                 OpenVINO GenAI pipeline
                                           |
                            Qwen2.5 3B INT4 model on CPU
```

- The inference pipeline is loaded once per backend process and reused.
- The web server binds to loopback at `127.0.0.1:8765`.
- Browser conversations are stored in browser local storage; inference context
  is held in server memory and is lost when the server stops.
- The model configuration reports a 32,768-token context. LocalMind limits
  prepared input to 8,192 tokens, includes recent turns and selected older
  excerpts, and uses the tokenizer's end-of-turn token.
- Web sessions use separate UUID-keyed state and per-session locks. The server
  caps active sessions at 32 and exposes a reset operation.
- The browser renders Markdown locally. It waits for the complete JSON response
  rather than streaming partial tokens.

## Root-cause findings

1. **Incomplete answers:** The configured output cap was often reached before
   the model completed its response. This is the clearest cause of the
   previously observed mid-sentence endings.
2. **Ambiguous terms:** The model can select a common but contextually wrong
   expansion when an abbreviation has multiple meanings. Conversation history
   and a cautious system prompt can help, but cannot guarantee correct
   disambiguation.
3. **Knowledge limits:** The model did not reliably know the intended ENTC
   engineering meaning. Prompting alone cannot supply absent or weakly
   represented facts; a vetted local retrieval corpus would be a future option.
4. **Evaluation limits:** Automated case results contain no human rubric scores.
   They are useful for response and token-cap comparisons, not proof of factual
   quality.

## Implemented improvements

- Added bounded, token-counted conversation context with recent turns and
  selected older user/assistant excerpts.
- Preserved separate in-memory context per web session and added explicit
  session reset and capacity handling.
- Added the tokenizer's end-of-turn token to generation configuration and
  surfaced warnings when an answer reaches its output limit.
- Improved the system prompt to use context, avoid unsupported acronym
  expansions, and ask for clarification when intent is unclear.
- Added browser-side rendering for headings, paragraphs, ordered and unordered
  lists, inline code, and fenced code, along with loading, error, and
  truncation states.
- Added focused unit tests and a repeatable multi-domain evaluation runner.
- Kept the existing Qwen INT4 model and OpenVINO CPU inference path.

## Evaluation results

The saved evaluation artifacts used a 128-token output limit. The earlier
candidate run predates the latest conservative prompt wording.

| Evaluation | Cases | Turns | Case-ending answers at token cap | All turns at token cap |
|---|---:|---:|---:|---:|
| Baseline | 16 | 22 | 16 / 16 | 21 / 22 |
| Earlier candidate | 16 | 22 | 7 / 16 | 8 / 22 |

These are output-completeness measurements, not human quality scores. Manual
relevance, correctness, clarity, hallucination, and continuity scores remain
unrated. Under the latest prompt, limited targeted checks were observed: the
assistant answered a cryptography-context DSA question and defined API
appropriately, while it asked for clarification rather than guessing an ENTC
meaning. The latest prompt has not had a full evaluation, as requested testing
was stopped.

Nine focused unit tests passed before the latest prompt wording changes. They
were not rerun afterward. No further test suite or multi-case evaluation was
run after the request to stop testing.

## Performance and memory evidence

The available baseline API profile made four measured requests on CPU, each
with a 16-token output limit: three short prompts and one longer prompt.

- Mean OpenVINO throughput: **5.10 tokens/second**.
- Median OpenVINO throughput: **4.90 tokens/second**.
- Time to first token: **0.41 to 7.87 seconds** across the four requests. The
  long-prompt result is included, so this is not a controlled short-prompt
  range.
- The requested single browser run displayed **4.72 tokens/second** for the
  prompt below. This is one UI-reported observation, not a comparable
  before-and-after benchmark.
- Process peak working set: **3.92 GB**.
- Sampled generation peak working set: **2.07 GB**.
- Sampled process private bytes: **2.63 GB**.

Memory values are process counters, not total laptop RAM usage. The recorded
profile was affected by concurrent system load and is diagnostic rather than a
controlled benchmark. No candidate profile was collected, so these numbers
must not be presented as a before-and-after improvement or as measurements of
the latest prompt revision.

Sources: [baseline API profile](performance_experiments/baseline_api_profile.csv),
[baseline evaluation](performance_experiments/baseline_evaluation.json), and
[earlier candidate evaluation](performance_experiments/candidate_evaluation.json).

## Limitations and remaining work

- The Qwen model can still produce incorrect or incomplete answers; prompting
  cannot guarantee facts absent from the model's knowledge.
- The 128-token cap remains a significant completeness limitation. A higher
  limit should be evaluated deliberately rather than increased blindly.
- The latest prompt has not been evaluated across all cases.
- No current-version performance or memory profile is available.
- No verified end-to-end screenshot set has been saved yet. The requested
  terminal, browser, performance, OpenVINO, and memory screenshots remain
  pending.
- The server is a local demo, not a production or shared-network service. It
  has no authentication or access-control layer.
- No RAG corpus, embedding dependency, fine-tuning, or model replacement was
  introduced.

## Requested single-run evidence

One question was sent in a separate browser session, leaving the existing
Edge conversation and its draft untouched:

> **Question:** What is JavaScript? Answer in 2–3 simple sentences.
>
> **Local model response:** JavaScript is a scripting language that is used
> for adding interactive elements to web pages. It allows developers to
> create dynamic content that can respond to user actions like clicks or
> scrolls.

The browser showed **4.72 tokens/second** and **CPU · Ready**. A browser
screenshot of this result was captured in chat, but it was not saved as a PNG
in the project. The terminal, separate performance, OpenVINO-run, and
RAM/memory screenshot files remain pending. Do not substitute older
evaluation output for screenshots of this new run.
