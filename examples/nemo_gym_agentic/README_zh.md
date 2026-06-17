# nemo_gym Agentic 示例

把
[NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym/tree/f82b601a9f5951793226cbe2d77336b677c6311e)
接入 Relax agentic-rollout 流程的参考实现。nemo_gym 本身就是一套常规
webserver 栈，我们的做法是：在 Relax 拉起的 agent 进程里**按 session 起整套
nemo_gym**。

目前接入了两个 env：

- `multi_step` —— `example_multi_step` 同义词抽取任务，由 `simple_agent`
  驱动（Responses-API 工具循环）。
- `multi_turn_gymnasium` —— `example_multi_turn_gymnasium` 任务，由
  `gymnasium_agent` 驱动（reset/step 循环）。

Adapter 是可插拔的：`app/envs/<name>.py` 各自声明要加载的 nemo_gym yaml
以及如何驱动产生的 server。启动时用
`--agent-env "NEMO_GYM_ADAPTER=<name>"` 选一个。

## 架构

```
        ┌─ 每个 session 一个 agent 进程（Relax 通过 --agent-command 启动）─┐
        │                                                                  │
        │  ┌──────────────┐   POST /run 或          ┌──────────────┐       │
        │  │  thin client │   POST /v1/responses    │ agent server │       │
        │  │  (agent.py)  │ ──────────────────────► │  (nemo_gym)  │       │
        │  └──────┬───────┘                         └──────┬───────┘       │
        │         │                                        │ /v1/responses │
        │         │                                        ▼               │
        │         │                                 ┌──────────────┐       │
        │         │                                 │ vllm_model   │ ────  │── OpenAI ──► Relax /v1/chat/completions
        │         │                                 │  (nemo_gym)  │       │              ($RELAX_BASE_URL)
        │         │                                 └──────┬───────┘       │
        │         │                                        │ env 专属工具  │
        │         │                                        ▼               │
        │         │                                 ┌──────────────┐       │
        │         │                                 │  resources   │       │
        │         │                                 │   server     │       │
        │         │                                 │  (nemo_gym)  │       │
        │         │                                 └──────────────┘       │
        │         │                                                        │
        │         └──► RELAX_OUTPUT_JSON                                   │
        └──────────────────────────────────────────────────────────────────┘
```

每个 session，agent 进程在一个 asyncio 事件循环里、按本地随机端口起一组
uvicorn webserver，server 的组合由当前 adapter 的 `CONFIG_PATHS` 决定。
`multi_step` 是三个（agent + vllm_model + resources），其他 adapter 可能不同。
`vllm_model` 负责把 Responses ↔ Chat 转换并把请求打到 `$RELAX_BASE_URL`。

**为什么每 session 一组：** Relax 通过 `Authorization: Bearer <session_id>`
来鉴权 chat 请求（`relax/agentic/session/service.py:277`），而 nemo_gym
`vllm_model` 是配死的单 api_key —— 共享一组 nemo_gym 会把所有 session 的 chat
都串到同一个 Relax session 记录里去。每 session 起一组 uvicorn 多花 ~1–2s
启动，但布线非常清晰，且**不需要 patch nemo_gym**。

## 文件

```
app/
  agent.py                  session 入口：走 adapter CONFIG_PATHS、起 server、
                            交给 adapter.drive() 跑业务
  envs/
    multi_step.py           example_multi_step adapter（Responses 工具循环）
    multi_turn_gymnasium.py example_multi_turn_gymnasium adapter（reset/step）
scripts/
  setup_env.sh              一键装环境（clone Gym + uv venv + uv sync）
  convert_dataset.py        把上游 jsonl 转成 scripts/example.parquet
  run_smoke.sh              对任意 OpenAI 端点跑 multi_step 烟测
  run_smoke_gymnasium.py    对任意 OpenAI 端点跑 multi_turn_gymnasium 烟测
  run_agent_app_smoke.sh    本地跑一次 agent.py session（不走 Relax）
run_agent_app.sh            Relax → agent CLI 的胶水（--agent-command 入口）
run_nemo_gym_agentic_qwen35_9B_colocate.sh    9B、8 卡、colocate（同步）
run_nemo_gym_agentic_qwen35_9B_async.sh       9B、8 卡、fully-async
run_nemo_gym_agentic_qwen3_vl_4B_2xgpu.sh     4B-VL、2 卡、colocate
CONFIG_DRIVEN_PLAN.md       设计 note：把 agent.py 里更多布线下推到 nemo_gym yaml
```

## 环境准备

**所有承载 agent 进程的容器都要跑一次**（多机训练就是每个 Ray worker 容器）：

```bash
bash examples/nemo_gym_agentic/scripts/setup_env.sh
```

脚本会把 NeMo-Gym 按 pin 的 commit clone 到 `${GYM_REPO:-/root/repos/Gym}`、
用 `uv venv --python 3.12` 建 venv、跑 `uv sync`、最后 `import nemo_gym`
自检。可重复执行——已有的 checkout/venv 会被复用。需要时覆盖：`GYM_REPO`
换路径、`GYM_COMMIT` 换 pin、`UV_DEFAULT_INDEX` 换镜像（默认使用公开
PyPI 源）。

`run_agent_app.sh` 把 agent 解释器钉死在 `${GYM_REPO}/.venv/bin/python`，
路径不存在就 hard-fail —— 旧版静默回退到 PATH `python` 会把 worker 端
环境缺失伪装成上游那段晦涩的 "managed agent session completed before
producing a chat IR" 错误。

## 生成数据集

```bash
GYM_REPO=/path/to/NeMo-Gym \
    python examples/nemo_gym_agentic/scripts/convert_dataset.py
```

读取 checkout 的 `resources_servers/example_multi_step/data/example.jsonl`，
写 5 行到 `scripts/example.parquet`。每行：`prompt` 里放上游 system prompt

- user query，`metadata` 里放 verify ground truth（`id`、
  `expected_synonyms`、`expected_synonym_values`、`minefield_label`、
  `minefield_label_value`）以及 `tools` / `parallel_tool_calls`。要全量就
  `--split train`（或 `validation`）。

## 对接任意 OpenAI 端点烟测

先建 `examples/nemo_gym_agentic/env.sh`（已 gitignore），写端点凭证：

```bash
export OPENAI_BASE_URL=<端点，如 https://.../v1>
export OPENAI_API_KEY=<api key>
export OPENAI_MODEL=<端点上跑的 model id>
```

然后：

```bash
# multi_step（同义词抽取）
bash examples/nemo_gym_agentic/scripts/run_smoke.sh

# multi_turn_gymnasium
python examples/nemo_gym_agentic/scripts/run_smoke_gymnasium.py
```

把样例数据走一遍 in-process 的 nemo_gym 栈——不动 Ray、不动 Relax
controller——逐行打印 reward 和均值。用它在正式训练前确认端点、
reasoning-parser 配置、verify 链路是否对。`GYM_REPO` 仅在 nemo_gym 没
`pip install` 时需要设，`run_smoke.sh` 会把它 prepend 到 `PYTHONPATH`。

## 本地跑单个 session

```bash
RELAX_BASE_URL=http://<qwen3-endpoint>/v1 \
RELAX_SESSION_ID=local-test \
RELAX_INPUT_JSON=/tmp/in.json \
RELAX_OUTPUT_JSON=/tmp/out.json \
NEMO_GYM_ADAPTER=multi_step \
bash examples/nemo_gym_agentic/run_agent_app.sh
```

`/tmp/in.json` 的 schema 就是 Relax agentic session input —— 跟
`scripts/example.parquet` 的一行结构一样：

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

## 启动训练

| Launcher                                     | Model       | GPU | 模式             |
| -------------------------------------------- | ----------- | --- | ---------------- |
| `run_nemo_gym_agentic_qwen35_9B_colocate.sh` | Qwen3.5-9B  | 8   | colocate（同步） |
| `run_nemo_gym_agentic_qwen35_9B_async.sh`    | Qwen3.5-9B  | 8   | fully-async      |
| `run_nemo_gym_agentic_qwen3_vl_4B_2xgpu.sh`  | Qwen3-VL-4B | 2   | colocate         |

```bash
MODEL_DIR=/path/to/models \
SAVE_DIR=/path/to/save \
bash examples/nemo_gym_agentic/run_nemo_gym_agentic_qwen35_9B_colocate.sh
```

`MODEL_DIR/<model>` 下要放好 HuggingFace checkpoint。默认 adapter 是
`multi_step`；想换 env 就改 `NEMO_GYM_ADAPTER` 并把 `--prompt-data` 换成
对应 env 的 parquet。

## 数据流

1. Relax 写 session input JSON（含 `messages` 和 `metadata`）。
2. `run_agent_app.sh` 用输入/输出路径调起 `python -m app.agent`。
3. `agent.py` 解析 `NEMO_GYM_ADAPTER` → `app/envs/<name>.py`，然后遍历
   adapter 的 `CONFIG_PATHS`，给每个 yaml 申请一个空闲端口、实例化里
   面声明的 server、用 uvicorn 跑起来。
4. `agent.py` 把控制权交给 `adapter.drive(messages, metadata, agent_port, resources_port)`，由 adapter 跟 env 专属协议交互
   （`multi_step` 走 Responses 工具循环，`multi_turn_gymnasium` 走
   reset/step）。
5. adapter 内部所有模型调用都过 `vllm_model`，由后者翻成 Chat 调用
   Relax chat-completions 端点（带 `Authorization: Bearer $RELAX_SESSION_ID`）。
6. adapter 返回 `{reward, metadata}`；`agent.py` 写到 `RELAX_OUTPUT_JSON`。
7. agent 进程退出，所有起的 uvicorn server 跟着销毁。
8. Relax 把这条 session 收尾成训练样本，reward 用 adapter 给的值。

## 不同模型族注意

`VLLMConverter` 解析的 tool-call 格式对应 Qwen3 系列 chat template 和
reasoning parser。换模型族时，启动 agent 前给以下 `POLICY_*` 环境变量
设好（`app/agent.py` 会读，并作为 nemo_gym 插值覆盖透传）：

| 环境变量                                    | nemo_gym key                                |
| ------------------------------------------- | ------------------------------------------- |
| `POLICY_USES_REASONING_PARSER`              | `policy_uses_reasoning_parser`              |
| `POLICY_USES_INTERLEAVED_REASONING`         | `policy_uses_interleaved_reasoning`         |
| `POLICY_IS_RESPONSES_NATIVE`                | `policy_is_responses_native`                |
| `POLICY_REPLACE_DEVELOPER_ROLE_WITH_SYSTEM` | `policy_replace_developer_role_with_system` |
| `POLICY_RETURN_TOKEN_ID_INFORMATION`        | `policy_return_token_id_information`        |

同时保证模型 chat template 输出的 tool-call 格式是
`VLLMConverter.postprocess_chat_response` 认识的——见下文
Troubleshooting 关于 `tool_calls` 解析的那条。

## Reward

`example_multi_step` 的 `/verify` 返回 `reward = float(extracted == expected)`，比的是最后一次 `extract_synonym_values` 调用的 arguments 是否
等于 `expected_synonym_values`。verify response 还会带 `set_overlap`、
`original_term_minefield_hit`、`order_instruction_following_failure`
等可用于分析的字段——目前 Relax 没用，要的话改
`app/envs/multi_step.py` 把它们 forward 到 session metadata。

## 接入一个新 nemo_gym env

照 `app/envs/multi_step.py` 抄就行，契约很小。

**1. 确定要起哪些上游 server。** 选好 `resources_servers/<env_name>/`
目录、responses-API model adapter（基本永远是 `vllm_model`），如果 env
的 agent 是 server 形态、再加 `responses_api_agents/<agent_name>/`。
纯客户端驱动的循环不用在 `CONFIG_PATHS` 里挂 agent server。

**2. 在 `app/envs/<your_env>.py` 写一个模块**，对外暴露两个东西：

```python
CONFIG_PATHS: list[str] = [
    "resources_servers/<env_name>/configs/<env_name>.yaml",
    "responses_api_models/vllm_model/configs/vllm_model.yaml",
    # 可选：responses_api_agents/<agent_name>/configs/<agent_name>.yaml
]

async def drive(
    messages: list[dict],
    metadata: dict,
    *,
    agent_port: int,            # 首个 responses_api_agents server 的端口
    resources_port: int | None, # 首个 resources_servers server 的端口
) -> dict[str, Any]:
    ...
    return {
        "reward": float(...),       # 必填
        "metadata": {...},          # 可选；会被 merge 到 session metadata
    }
```

`agent.py` 已经替你把 `CONFIG_PATHS` 走一遍 nemo_gym config parser、抢
空闲端口、起 uvicorn，然后**在同一个 event loop 里** `await` 你的
`drive()`。

**3. 选驱动风格。**

| 风格                                     | 适用                                                                                                                       | 实现                                                                                 |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **agent server 驱动**（`multi_step`）    | 上游有 SimpleAgent 风格的 server 自己跑工具循环                                                                            | 对 `agent_port` POST `/v1/responses`，再对 `resources_port` POST `/verify` 拿 reward |
| **客户端驱动**（`multi_turn_gymnasium`） | 上游是有状态 env（`/reset`、`/step`）。要么自己驱动，要么对 gymnasium agent POST `/run` 让 *它* 来驱动（当前示例采用后者） | 对 `agent_port` POST `/run`，一次性拿回 `{response, reward, terminated, ...}`        |

复用 `app.agent` 里的 `_build_responses_body(messages, metadata)` 和
`_BODY_METADATA_KEYS`，让所有 adapter 的 input 解析（tools 转换、
developer role 注入等）保持一致。

**4. 数据集。** parquet schema 是固定的：一行要有 `prompt`（字符串）
和 `metadata`（dict）。`metadata` 装一切 env 专属的东西——工具 schema、
ground truth、env 初始化参数，任何 `drive()` 要读的字段。可以在
`scripts/convert_dataset.py` 加一个 split，也可以在 `scripts/` 下新写一个
converter。下游跑之前用 `pyarrow.parquet.read_table(...).to_pylist()[0]`
肉眼检查一下。

**5. 训练前先 smoke。** 仿照现有的烟测脚本写：

- `scripts/run_smoke.sh` / `run_smoke_gymnasium.py` —— 拿一个外部
  OpenAI 端点跑 in-process nemo_gym 栈。
- `scripts/run_agent_app_smoke.sh` —— 本地端到端跑一次完整的
  `agent.py` session，不走 Relax。

设 `AGENT_TRACE=1` 会让 adapter 把 per-session reward、工具调用、
模型文本/工具输出 dump 到 stdout —— 配合 `AGENT_DEBUG_LOG_DIR` 一起设
就会直接落到 `${AGENT_DEBUG_LOG_DIR}/<session_id>.log`。

**6. 启动。** 在 launcher 加 `--agent-env "NEMO_GYM_ADAPTER=<your_env>"`、
把 `--prompt-data` 换成新 parquet 就完事，launcher 其他东西都不用改。

## Troubleshooting

bring-up 阶段踩过三种失败模式。前两个上层报的错一字不差，**靠
`${AGENT_DEBUG_LOG_DIR}/agentic_session_*.log` 里的 per-session traceback
区分**。第三种是哑的：session "成功"了但根本没产学习信号，所以一定要
在让训练跑下去之前先看一眼 `rollout_result/train/0.jsonl`。

### Worker 容器缺 nemo_gym venv

**症状：** 训练日志末尾报
`RuntimeError: Prepare-owned managed agent session completed before producing a chat IR ... agentic resident dataflow loop failed`。

**真因：** per-session 日志里其实是 `ModuleNotFoundError: No module named 'nemo_gym'`。多节点 Ray 集群上 head 容器装了 `${GYM_REPO}/.venv`
但 worker 容器没装，agent 子进程 import 阶段就崩。Rollout actor 报的
`ip=...` 跟你跑 `setup_env.sh` 的容器不是同一个。

**修：** 每个跑 rollout actor 的容器里都跑一次 `scripts/setup_env.sh`。
`run_agent_app.sh` 现在 venv 缺失会 hard-fail，让错误暴露在正确的层，而
不是兜底成上层那段 dataflow-loop 错。

### `RELAX_BASE_URL` 末尾斜杠

**症状：** 和上面**一字不差**的
`Prepare-owned managed agent session completed before producing a chat IR`。

**真因：** per-session 日志里是 aiohttp 404。Relax 的
`resolve_chat_api_base_url()` 返回的 URL 末尾带 `/`，nemo_gym
`NeMoGymAsyncOpenAI` 是字符串拼接 `f"{base_url}/chat/completions"`，
拼出 `//chat/completions`，匹配不到任何 FastAPI 路由。

**修：** `run_agent_app.sh:13` 加了 `export OPENAI_BASE_URL="${RELAX_BASE_URL%/}"` 去尾斜杠。任何新加的 nemo_gym
agent example、或自写 aiohttp 客户端，都要带这行。

### Launcher 没配 tool-call / reasoning parser

**症状：** 训练全程不报错，但 `rollout_result/train/*.jsonl` 里
**reward 恒为 0、agent_turns 恒为 1、status 是 completed**。看
`response` 字段，模型其实老老实实输出了 `<tool_call>...</tool_call>`，
格式跟 prompt 要的对得上。

**真因：** `relax/agentic/session/service.py:_postprocess_assistant_message`
是否填 `tool_calls` 由 `--agentic-tool-call-parser` 是否提供决定，没传就
直接 fallthrough 返回 `{"role": "assistant", "content": raw_text}`。下游
`Gym/responses_api_models/vllm_model/app.py` 的
`VLLMConverter.postprocess_assistant_message_dict` 读
`message_dict.get("tool_calls", []) or []` —— 空 → Responses 输出里没有
`function_call` 项 → adapter 看不到工具调用 → 第 1 个 model turn 就收尾
→ `/verify` 永远拿不到终止工具（如 `extract_synonym_values`）→ reward=0。

**判别捷径：** *reward=0 + agent_turns=1 + status=completed + response
里能看见 `<tool_call>...`* 这个组合是本 bug 独有的，区别于前两个会
session 直接挂掉的 bug。

**修：** launcher 的 `ROLLOUT_ARGS` 里加上：

```
--agentic-tool-call-parser mimo
--agentic-reasoning-parser qwen3
```

`mimo` 解析的就是 prompt 教模型输出的
`<tool_call><function=NAME><parameter=NAME>value</parameter></function></tool_call>`
（见 `sglang/srt/function_call/mimo_detector.py` 的 docstring）。`hermes` /
`qwen25` parser 走的是 `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
JSON 变体，跟当前 prompt 教的 XML 变体**不匹配**，选错一样 reward=0。

`qwen3` reasoning parser 把模型续写出的 `</think>`（Qwen3 chat template
在 assistant 前缀里隐式注入 `<think>`）切掉，避免推理内容污染下一轮历史。

换其他模型族就把 parser 也换掉，可选项见
`sglang/srt/function_call/function_call_parser.py` 里的 `ToolParserDict`
（`hermes`、`qwen25`、`qwen3_coder`、`mimo`、`pythonic`、`mistral`、
`llama32` 等）和 `sglang/srt/parser/reasoning_parser.py`（`deepseek-r1`、
`qwen3`、`kimi` 等）。
