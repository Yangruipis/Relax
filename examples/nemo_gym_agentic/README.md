# nemo_gym Agentic Example

Reference integration of
[NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym/tree/f82b601a9f5951793226cbe2d77336b677c6311e)
into Relax's agentic-rollout flow. nemo_gym runs as a normal webserver
stack — we spin the stack up per session inside the agent process
Relax launches.

Two environments are wired in today:

- `multi_step` — the `example_multi_step` synonym-extraction task,
  driven by `simple_agent` (Responses-API tool loop).
- `multi_turn_gymnasium` — the `example_multi_turn_gymnasium` task,
  driven by `gymnasium_agent` (reset/step loop).

Adapters are pluggable: each adapter under `app/envs/<name>.py`
declares which nemo_gym yamls to load and how to drive the resulting
servers. Pick one at launch time with
`--agent-env "NEMO_GYM_ADAPTER=<name>"`.

## Architecture

```
        ┌─ per-session agent process (started by Relax via --agent-command) ─┐
        │                                                                    │
        │  ┌──────────────┐   POST /run or          ┌──────────────┐         │
        │  │ thin client  │   POST /v1/responses    │ agent server │         │
        │  │ (agent.py)   │ ──────────────────────► │  (nemo_gym)  │         │
        │  └──────┬───────┘                         └──────┬───────┘         │
        │         │                                        │ /v1/responses   │
        │         │                                        ▼                 │
        │         │                                 ┌──────────────┐         │
        │         │                                 │ vllm_model   │ ─────── │── OpenAI ──► Relax /v1/chat/completions
        │         │                                 │  (nemo_gym)  │         │              ($RELAX_BASE_URL)
        │         │                                 └──────┬───────┘         │
        │         │                                        │ env-specific    │
        │         │                                        │ tool endpoints  │
        │         │                                        ▼                 │
        │         │                                 ┌──────────────┐         │
        │         │                                 │ resources    │         │
        │         │                                 │  server      │         │
        │         │                                 │ (nemo_gym)   │         │
        │         │                                 └──────────────┘         │
        │         │                                                          │
        │         └──► RELAX_OUTPUT_JSON                                     │
        └────────────────────────────────────────────────────────────────────┘
```

For each session the agent process spins up the set of uvicorn
webservers declared by the active adapter's `CONFIG_PATHS` on local
random ports, inside a single asyncio loop. For `multi_step` that is
three servers (agent + vllm_model + resources); other adapters may
declare a different set. `vllm_model` translates Responses ↔ Chat
against `$RELAX_BASE_URL`.

**Why per-session:** Relax authorises chat requests by
`Authorization: Bearer <session_id>`
(`relax/agentic/session/service.py:277`), while nemo_gym's
`vllm_model` is configured with one static `api_key`. A shared
nemo_gym instance would multiplex every session into the same Relax
session record. Per-session uvicorn adds ~1–2s of startup cost but
keeps the wiring straight without patching nemo_gym.

## Files

```
app/
  agent.py                  per-session entry point: walks adapter CONFIG_PATHS,
                            spins servers, hands control to adapter.drive()
  envs/
    multi_step.py           adapter for example_multi_step (Responses tool loop)
    multi_turn_gymnasium.py adapter for example_multi_turn_gymnasium (reset/step)
scripts/
  setup_env.sh              one-shot env install (clone Gym + uv venv + uv sync)
  convert_dataset.py        upstream jsonl → scripts/example.parquet
  run_smoke.sh              OpenAI-endpoint smoke test for multi_step
  run_smoke_gymnasium.py    OpenAI-endpoint smoke test for multi_turn_gymnasium
  run_agent_app_smoke.sh    drives one agent.py session locally (no Relax)
run_agent_app.sh            Relax → agent CLI shim (entrypoint for --agent-command)
run_nemo_gym_agentic_qwen35_9B_colocate.sh    9B, 8 GPUs, colocate (sync)
run_nemo_gym_agentic_qwen35_9B_async.sh       9B, 8 GPUs, fully-async
run_nemo_gym_agentic_qwen3_vl_4B_2xgpu.sh     4B-VL, 2 GPUs, colocate
CONFIG_DRIVEN_PLAN.md       design note: pushing more agent.py wiring into
                            upstream nemo_gym yamls
```

## Requirements

Run **inside every container** that will host the agent process (all
Ray worker nodes for multi-node training):

```bash
bash examples/nemo_gym_agentic/scripts/setup_env.sh
```

Clones NeMo-Gym at the pinned commit to `${GYM_REPO:-/root/repos/Gym}`,
creates `.venv` with `uv venv --python 3.12`, runs `uv sync`, and
verifies `import nemo_gym`. Idempotent — existing checkout/venv are
reused. Override `GYM_REPO` to clone elsewhere, `GYM_COMMIT` to pin a
different revision, or `UV_DEFAULT_INDEX` to use a mirror instead of
the public PyPI index.

`run_agent_app.sh` pins the agent interpreter to
`${GYM_REPO}/.venv/bin/python` and hard-fails if the path is missing —
silently falling back to PATH `python` previously masked worker-side
env gaps as the opaque "managed agent session completed before
producing a chat IR" error from upstream.

## Generate the dataset

```bash
GYM_REPO=/path/to/NeMo-Gym \
    python examples/nemo_gym_agentic/scripts/convert_dataset.py
```

Reads `resources_servers/example_multi_step/data/example.jsonl` from
the checkout and writes 5 rows into `scripts/example.parquet`. Each
row carries the upstream system prompt + user query in `prompt`, and
the verify ground truth (`id`, `expected_synonyms`,
`expected_synonym_values`, `minefield_label`, `minefield_label_value`)
plus `tools` / `parallel_tool_calls` in `metadata`. Pass
`--split train` (or `validation`) for the full set.

## Smoke test against any OpenAI-format endpoint

Create `examples/nemo_gym_agentic/env.sh` (gitignored) with your
endpoint credentials:

```bash
export OPENAI_BASE_URL=<endpoint, e.g. https://.../v1>
export OPENAI_API_KEY=<api key>
export OPENAI_MODEL=<model id served by that endpoint>
```

Then:

```bash
# multi_step (synonym extraction)
bash examples/nemo_gym_agentic/scripts/run_smoke.sh

# multi_turn_gymnasium
python examples/nemo_gym_agentic/scripts/run_smoke_gymnasium.py
```

Runs the example rows through the in-process nemo_gym stack — no Ray,
no Relax controller — and prints per-row reward + mean. Use this to
confirm your endpoint, reasoning-parser settings, and verify wiring
before launching training. `GYM_REPO` only needs to be set when
nemo_gym isn't `pip install`-ed; `run_smoke.sh` prepends it to
`PYTHONPATH`.

## Run a single session locally

```bash
RELAX_BASE_URL=http://<qwen3-endpoint>/v1 \
RELAX_SESSION_ID=local-test \
RELAX_INPUT_JSON=/tmp/in.json \
RELAX_OUTPUT_JSON=/tmp/out.json \
NEMO_GYM_ADAPTER=multi_step \
bash examples/nemo_gym_agentic/run_agent_app.sh
```

`/tmp/in.json` follows the Relax agentic session-input shape — one row
of `scripts/example.parquet`:

```json
{
  "messages": [
    {"role": "system", "content": "# Instructions\nYou are an extraction agent..."},
    {"role": "user",   "content": "I'm very warm"}
  ],
  "metadata": {
    "tools": [
      {"type": "function", "name": "get_synonym_value",      "parameters": {...}},
      {"type": "function", "name": "extract_synonym_values", "parameters": {...}}
    ],
    "parallel_tool_calls": false,
    "id": 0,
    "expected_synonyms": ["Blazing", "Warm"],
    "expected_synonym_values": [711, 407],
    "minefield_label": "Hot",
    "minefield_label_value": 299
  }
}
```

## Launch training

| Launcher                                     | Model       | GPUs | Mode            |
| -------------------------------------------- | ----------- | ---- | --------------- |
| `run_nemo_gym_agentic_qwen35_9B_colocate.sh` | Qwen3.5-9B  | 8    | colocate (sync) |
| `run_nemo_gym_agentic_qwen35_9B_async.sh`    | Qwen3.5-9B  | 8    | fully-async     |
| `run_nemo_gym_agentic_qwen3_vl_4B_2xgpu.sh`  | Qwen3-VL-4B | 2    | colocate        |

```bash
MODEL_DIR=/path/to/models \
SAVE_DIR=/path/to/save \
bash examples/nemo_gym_agentic/run_nemo_gym_agentic_qwen35_9B_colocate.sh
```

`MODEL_DIR/<model>` must contain the HuggingFace checkpoint. Default
adapter is `multi_step`; change `NEMO_GYM_ADAPTER` (and swap
`--prompt-data` to a parquet built for the target env) to train a
different env.

## Data flow

1. Relax writes the session input JSON with `messages` and `metadata`.
2. `run_agent_app.sh` invokes `python -m app.agent` with input/output paths.
3. `agent.py` resolves `NEMO_GYM_ADAPTER` → `app/envs/<name>.py`, then
   for each yaml in the adapter's `CONFIG_PATHS` reserves a free port,
   instantiates the declared server, and runs `setup_webserver()`
   under uvicorn.
4. `agent.py` hands control to `adapter.drive(messages, metadata, agent_port, resources_port)`, which speaks the env-specific
   protocol (Responses tool loop for `multi_step`; reset/step for
   `multi_turn_gymnasium`).
5. The adapter routes model calls through `vllm_model`, which
   translates Responses → Chat and calls Relax's chat-completions
   endpoint with `Authorization: Bearer $RELAX_SESSION_ID`.
6. The adapter returns `{reward, metadata}`; `agent.py` writes it to
   `RELAX_OUTPUT_JSON`.
7. The agent process exits, tearing down every uvicorn server it spun.
8. Relax finalises the session as a training sample with the
   adapter-provided reward.

## Model-family note

`VLLMConverter` parses the tool-call format produced by Qwen3-family
chat templates and reasoning parsers. For other model families, point
the relevant `POLICY_*` env vars at the right setting before launching
the agent (read by `app/agent.py` and applied as nemo_gym
interpolation overrides):

| Env var                                     | nemo_gym key                                |
| ------------------------------------------- | ------------------------------------------- |
| `POLICY_USES_REASONING_PARSER`              | `policy_uses_reasoning_parser`              |
| `POLICY_USES_INTERLEAVED_REASONING`         | `policy_uses_interleaved_reasoning`         |
| `POLICY_IS_RESPONSES_NATIVE`                | `policy_is_responses_native`                |
| `POLICY_REPLACE_DEVELOPER_ROLE_WITH_SYSTEM` | `policy_replace_developer_role_with_system` |
| `POLICY_RETURN_TOKEN_ID_INFORMATION`        | `policy_return_token_id_information`        |

Also make sure the model's chat template emits tool calls in a format
`VLLMConverter.postprocess_chat_response` can read — see the
Troubleshooting note on `tool_calls` parsing below.

## Reward

`example_multi_step` returns `reward = float(extracted == expected)`
from `/verify`, comparing the final `extract_synonym_values` call's
arguments against `expected_synonym_values`. The verify response also
surfaces `set_overlap`, `original_term_minefield_hit`, and
`order_instruction_following_failure` for analysis — currently unused
by Relax, but available if you extend `app/envs/multi_step.py` to
forward them as session metadata.

## Add a new nemo_gym env

Take `app/envs/multi_step.py` as the template; the contract is small.

**1. Decide which upstream servers to spin up.** Pick the
`resources_servers/<env_name>/` directory you want, plus the
responses-API model adapter (almost always `vllm_model`) and — if the
env's agent runs server-side — the matching `responses_api_agents/<agent_name>/`.
For pure client-driven loops you don't need an agent server in
`CONFIG_PATHS`.

**2. Drop a module at `app/envs/<your_env>.py`.** Two public symbols:

```python
CONFIG_PATHS: list[str] = [
    "resources_servers/<env_name>/configs/<env_name>.yaml",
    "responses_api_models/vllm_model/configs/vllm_model.yaml",
    # optional: "responses_api_agents/<agent_name>/configs/<agent_name>.yaml",
]

async def drive(
    messages: list[dict],
    metadata: dict,
    *,
    agent_port: int,           # first responses_api_agents server, if any
    resources_port: int | None, # first resources_servers server, if any
) -> dict[str, Any]:
    ...
    return {
        "reward": float(...),       # required
        "metadata": {...},          # optional; merged into session metadata
    }
```

`agent.py` already loads `CONFIG_PATHS` through nemo_gym's config
parser, reserves free ports, starts every declared server under
uvicorn, then `await`s your `drive()` inside the same event loop.

**3. Pick the right driving style.**

| Style                                      | When                                                                                                                                                                     | Examples                                                                                          |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| **Agent-server-driven** (`multi_step`)     | Upstream ships a SimpleAgent-style server that owns the tool loop.                                                                                                       | POST `/v1/responses` on `agent_port`, then POST `/verify` on `resources_port` for the reward.     |
| **Client-driven** (`multi_turn_gymnasium`) | Upstream ships a stateful env (`/reset`, `/step`). Either drive it yourself, or POST `/run` on the gymnasium agent and let *it* drive (current example does the latter). | POST `/run` on `agent_port`; the agent returns `{response, reward, terminated, ...}` in one shot. |

Reuse `_build_responses_body(messages, metadata)` and
`_BODY_METADATA_KEYS` from `app.agent` so input parsing stays
consistent across adapters (tools coercion, developer-role injection,
etc.).

**4. Dataset.** The parquet schema is fixed: a row needs `prompt`
(string) and `metadata` (dict). `metadata` carries everything env-
specific — tool schema, ground truth, env-init args, anything your
`drive()` reads. Either extend `scripts/convert_dataset.py` for the
new split, or write a new converter under `scripts/`. Verify with
`pyarrow.parquet.read_table(...).to_pylist()[0]` before launching.

**5. Smoke before training.** Mirror the existing smoke scripts:

- `scripts/run_smoke.sh` / `run_smoke_gymnasium.py` — drive the
  in-process nemo_gym stack against an external OpenAI endpoint.
- `scripts/run_agent_app_smoke.sh` — drive one full `agent.py` session
  end-to-end locally without Relax.

Set `AGENT_TRACE=1` to make the adapter dump per-session reward, tool
calls, and the model's text/tool outputs to stdout — drops straight
into `${AGENT_DEBUG_LOG_DIR}/<session_id>.log` when AGENT_DEBUG_LOG_DIR
is also set.

**6. Launch.** Add `--agent-env "NEMO_GYM_ADAPTER=<your_env>"` to a
launcher and swap `--prompt-data` to the new parquet. Nothing else in
the launcher needs to change.

## Troubleshooting

Three failure modes seen during bring-up. The first two share an
upstream error string — distinguish them by reading the per-session
agent log under `${AGENT_DEBUG_LOG_DIR}/agentic_session_*.log`. The
third is silent: the session "succeeds" but produces no learning
signal, so always sanity-check `rollout_result/train/0.jsonl` before
trusting later training curves.

### Worker container missing the nemo_gym venv

**Symptom:** training log ends with
`RuntimeError: Prepare-owned managed agent session completed before producing a chat IR ... agentic resident dataflow loop failed`.

**Real cause:** the per-session log shows
`ModuleNotFoundError: No module named 'nemo_gym'`. On a multi-node Ray
cluster the head container has the venv at `${GYM_REPO}/.venv` but a
worker container doesn't; the agent subprocess crashes at import
time. The Rollout actor's reported `ip=...` will not match the host
where you ran `setup_env.sh`.

**Fix:** run `scripts/setup_env.sh` inside **every** container that
hosts a rollout actor. `run_agent_app.sh` now hard-fails if the venv
is missing, so the failure surfaces at the right layer instead of
collapsing into the dataflow-loop error.

### `RELAX_BASE_URL` trailing slash

**Symptom:** identical
`Prepare-owned managed agent session completed before producing a chat IR`.

**Real cause:** the per-session log shows an aiohttp 404. Relax's
`resolve_chat_api_base_url()` returns a URL ending in `/`, and
nemo_gym's `NeMoGymAsyncOpenAI` builds endpoints by string concat
(`f"{base_url}/chat/completions"`), producing a double slash that
matches no FastAPI route.

**Fix:** `run_agent_app.sh:13` strips the slash via
`export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"`. Any new agent example
built on nemo_gym or any custom aiohttp client needs the same shim.

### Relax chat endpoint does not parse `tool_calls` (open)

**Symptom:** training runs end-to-end without errors but
`rollout_result/train/*.jsonl` shows `reward=0` everywhere,
`agent_turns=1` everywhere, and `status=completed`. Inspecting the
`response` field, the model is clearly emitting well-formed
`<tool_call>...</tool_call>` blocks matching what the prompt asked
for.

**Real cause:** `relax/agentic/session/service.py:1463` returns
`{"role": "assistant", "content": response_text}` with **no
`tool_calls` field populated** (grep `tool_calls=` and
`message["tool_calls"]` across `relax/agentic/` and
`relax/backends/sglang/` returns zero hits). Downstream,
`VLLMConverter.postprocess_assistant_message_dict` in
`Gym/responses_api_models/vllm_model/app.py:871` reads
`message_dict.get("tool_calls", []) or []`, so no `function_call`
items reach the agent, the loop terminates after one model turn, and
`/verify` scores 0 because the terminal tool (e.g.
`extract_synonym_values`) was never called.

**Diagnostic shortcut:** the combination *reward=0 + agent_turns=1 +
status=completed + `<tool_call>...` visible in `response`* is unique
to this bug — distinct from the two errors above, which crash the
session outright.

**Fix direction (not yet implemented):** plumb a tool-call parser
(SGLang ships `hermes`, `qwen2_5`, etc.) into `_chat_completions_impl`
before returning, and populate `message.tool_calls`. Expose the parser
choice via a flag such as `--agentic-tool-call-parser`. Patching
nemo_gym's converter instead would require forking upstream.
