# nemo_gym integration — architecture note

`app/agent.py` is per-session. Relax spawns one subprocess per rollout
session because nemo_gym authenticates by a single `api_key`; sharing
nemo_gym instances across sessions would multiplex them into the same
Relax session record.

## Two layers: generic loader + per-env adapter

`agent.py` itself is env-agnostic. It:

1. Loads an adapter from `app.envs.<NEMO_GYM_ADAPTER>`.
2. Reads `adapter.CONFIG_PATHS` (upstream yamls) and feeds them to
   `nemo_gym.global_config`'s parser, with a small initial dict providing
   `policy_base_url` / `policy_api_key` / `policy_model_name` for
   `${...}` interpolation and a pre-reserved `head_server.port`.
3. Walks the resulting merged dict; for each non-reserved top-level key
   detects the category (`resources_servers` / `responses_api_models` /
   `responses_api_agents`), reserves a socket, dynamically imports
   `<category>/<subfolder>/app.py`, reflects the single
   `Simple{Resources,Agent,Model}*` subclass and its
   `model_fields["config"].annotation` Config class, and instantiates.
4. Brings every webserver up under one `AsyncExitStack` and hands off to
   `adapter.drive(messages, metadata, agent_port, resources_port)`.

Each adapter under `app/envs/` is ~30-100 lines and owns:

- `CONFIG_PATHS` — the upstream yamls this env needs (resources +
  model + optional standalone agent yaml).
- `drive()` — the request/response shape and result formatter for
  that env's agent class.

Today there are two adapters:

| adapter                | agent class      | protocol                                                        |
| ---------------------- | ---------------- | --------------------------------------------------------------- |
| `multi_step`           | `SimpleAgent`    | POST `/v1/responses` on agent + POST `/verify` on resources     |
| `multi_turn_gymnasium` | `GymnasiumAgent` | POST `/run` on agent (env loop is internal, reward in response) |

## Why this shape

Upstream `ng_run` uses `Popen` to launch each server as a separate
process with its own venv. We can't do that in a per-session subprocess
(too slow, too much port plumbing), so we collapse everything into one
Python process and reuse `nemo_gym.global_config`'s parser to merge the
yamls. The parser has two entry points: a fast path that reads the
merged dict directly from `NEMO_GYM_CONFIG_DICT` (skips config_paths
file loading), and the full parser path. We use the parser path so
`config_paths` yamls and `${...}` interpolation both work.

Adapters exist because upstream agent classes have incompatible
request/response shapes (`/v1/responses` + `/verify` vs `/run`; OpenAI
Responses input vs `BaseRunRequest` envelope; reward as a top-level
verify response field vs nested inside `GymnasiumRunResponse`). A
single dispatcher inside `agent.py` was attempted first and discarded —
it would have grown into a registry of every upstream agent class.

## External contract

- `NEMO_GYM_ADAPTER` (required) — adapter module name under `app.envs.`
- `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` (required) —
  per-session credentials; translated internally to
  `policy_base_url` / `policy_api_key` / `policy_model_name`.
- `POLICY_USES_REASONING_PARSER`, `POLICY_USES_INTERLEAVED_REASONING`,
  `POLICY_IS_RESPONSES_NATIVE`, `POLICY_REPLACE_DEVELOPER_ROLE_WITH_SYSTEM`,
  `POLICY_RETURN_TOKEN_ID_INFORMATION` (optional) — override the
  `${policy_*}` interpolation defaults in upstream model yamls (e.g.
  `vllm_model.yaml`).
- `NEMO_GYM_DEBUG_HTTP` (optional) — flips aiohttp request debug logs in
  nemo_gym.
- `AGENT_TRACE` (optional) — emit per-session trace to stdout.

## Per-env contract

To add a new env (say `foo`):

1. Verify upstream provides yamls that, when merged, declare one agent
   and at most one resources_server (multi-agent / multi-resources envs
   need code-side extension).
2. Create `app/envs/foo.py` with `CONFIG_PATHS` and `drive()`. If the
   agent class is `SimpleAgent` or `GymnasiumAgent`, copy the matching
   existing adapter and adjust `CONFIG_PATHS`. If it's a new class,
   inspect the agent's FastAPI routes and reward shape and write
   `drive()` accordingly.
3. Write a dataset converter (or a smoke driver that consumes the
   upstream jsonl directly) that produces `(messages, metadata)` pairs
   where `metadata` carries everything `drive()` needs to forward.
4. Set `NEMO_GYM_ADAPTER=foo` and run.

## Known limits

- One agent + at most one resources_server per session. Envs that need
  multiple of either won't load.
- The agent must run entirely in this Python process. Envs that need
  Docker, a database, or a separate venv (typical of upstream
  `Popen`-only envs) are out of scope.
- No automatic adapter selection from `CONFIG_PATHS` or jsonl content;
  the caller must pick.
