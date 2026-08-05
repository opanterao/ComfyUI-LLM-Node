# ComfyUI 多参考图描述节点

一个 ComfyUI 自定义节点，可对接**任意 OpenAI 兼容的 chat/completion 端点**
（例如本地 vLLM/Ollama 网关、私有代理，或任何暴露 OpenAI 风格 REST API 的
厂商）。**不内置任何默认提供商** —— 你必须自行提供 `base_url`。

该节点（`CSYIDC-RefDescribe`）使用视觉语言模型（VLM）**独立描述最多 15 张
参考图**，然后根据你的系统提示词，将这些描述合并成一段完整的文本输出。

---

## 节点

| 节点 | 用途 |
|------|------|
| **多参考图描述**（`CSYIDC-RefDescribe`） | 用 VLM 描述最多 15 张参考图，再合并为一段文本输出 |

节点使用 OpenAI **Chat Completions** 格式连接到你的端点。

---

## 安装

1. 将此仓库克隆到你的 ComfyUI `custom_nodes` 目录。
2. 执行 `pip install -r requirements.txt`。
3. 重启 ComfyUI。

`numpy` 和 `torch` 由 ComfyUI 自带，无需单独安装。

> 本节点不捆绑任何 API 密钥或提供商默认值，是刻意保持的"提供商无关"设计。

---

## 配置

每个节点都需要一个 **base_url** —— OpenAI 兼容端点的根地址，例如：

- `https://your-proxy.example/v1`
- `http://localhost:8000/v1`（本地 vLLM / llama.cpp / Ollama 兼容服务器）
- `https://api.deepseek.com/v1` 或任意 OpenAI 兼容厂商

可选字段 **api_key** 仅在填写时作为 `Bearer` token 发送。不需要鉴权的端点可留空。

### 模型列表

**model** 下拉框通过查询 `{base_url}/models` 填充，若找不到再尝试
`{base_url}/v1/models`。如果端点不提供模型列表，或请求失败，会显示一个
自由文本框，你可以手动输入模型 ID。

---

## 输入与输出

### 必填输入

- **base_url** — 你的端点根地址（必填）。
- **api_key** — 可选 `Bearer` token。
- **system_prompt** — 系统提示词，控制如何将各图描述合并为最终的完整输出。
- **image_instruction** — 用于描述每张图片的提示词。
- **model** — 模型 ID（下拉选择或手动输入）。
- **temperature** — 0.0–2.0。
- **reasoning_effort** — `auto | none | minimal | low | medium | high | xhigh`
  （仅在非 `auto` 时以 `reasoning_effort` 字段发送）。
- **max_tokens** — 最大生成长度；`-1` 表示"不发送"。
- **max_context_tokens** — 发送给端点的文本的近似上下文窗口上限；`-1` 表示
  无限（不截断）。
- **seed** — 整数随机种子。
- **top_p** — 核采样（0.0–1.0）。
- **top_k** — top-k 采样；`-1` 表示"不发送"。
- **frequency_penalty** — −2.0 到 2.0。
- **presence_penalty** — −2.0 到 2.0。
- **stop** — 逗号分隔的停止序列（可选）。
- **request_timeout** — 请求超时时间（秒）。

### 可选输入

- **user_instruction** — 用于框定最终合并步骤的可选指令（会与图片描述一起
  传入模型）。

### 动态图片输入

- **ref_image** 输入是动态的：`ref_image_1` 始终存在；连接后会出现
  `ref_image_2`，以此类推，最多到 `ref_image_15`。

### 输出

- **Output** — `STRING` 类型；最终合并的文本响应。

---

## 工作原理

1. 每张已连接的参考图都会使用 `image_instruction` 提示词**独立**请求描述一次。
2. 收集所有描述，连同你的 `system_prompt` 一起再次发送给模型，由它控制如何
   将这些描述合并成一段连贯的输出。
3. 可选字段 `user_instruction` 会被合并进来以框定最终输出；图片描述始终会被
   包含，确保模型拥有完整上下文。

> 需要一个支持图像输入的多模态（视觉）模型。

---

## 示例

[`workflow/`](workflow/) 目录下提供了一份可直接运行的示例工作流和截图。

![CSYIDC-RefDescribe 示例工作流](workflow/img_01.png)

- [workflow/llmWorkflow.json](workflow/llmWorkflow.json) — 一份示例工作流：
  加载 3 张参考图，输入到 `CSYIDC-RefDescribe`，并用 [PreviewAny] 节点预览
  合并后的文本输出。

使用方法：

1. 在 ComfyUI 中，将 `workflow/llmWorkflow.json` 拖拽到画布上（或通过
   **Workflow → Open** 选择该文件）。
2. 在节点中填写你自己的 `base_url` / `api_key` / `model`（示例自带的是作者的
   端点 —— **请替换为你的**）。
3. 连接你自己的参考图片并运行。

> **安全提示：** 示例工作流中可能含有存储在 widget 元数据里的 `api_key`
> （ComfyUI 会把 widget 的值直接写进 JSON）。如果你要分享这份工作流，请先
> 移除或轮换该密钥。

---

## 示例用法：MiniMax H3 视频生成提示词

本节点可用作视频生成管线的"大脑"。典型流程：喂入一张或多张参考图，让 LLM
描述它们，再利用生成的文本为 **MiniMax H3** 这类视频生成模型生成结构完整、
符合格式的视频提示词。

下面是一份可放入节点 `system_prompt` / `image_instruction` 字段（或作为模板
复用）的**示例系统提示词**，让 LLM 写出完整、格式合规的视频提示词。它包含
角色定义以及参考格式输出：

```
你是一个MiniMax H3视频生成的提示词专家，请根据图片识别出的图片内容以及用户提供的剧情内容，写出完整的视频生成提示词，如果用户提到台词内容，请按照用户输入的原文（中文或其他语言）完整的在视频生成提示词中体现出来。

以下是完整的视频生成提示词示例，请按照此格式撰写提示词：

subject_definitions:
<Picture 1> is the reference image of the young woman with long loose hair, fair skin and dark eyes, wearing a lightweight white fitness-style top, standing in a cramped sci-fi environment.
<Picture 2> is the reference image of the young man with short dark hair, wearing a fitted dark crew-neck suit, also set in the same sci-fi surroundings.
<Picture 3> is the reference image of the spacecraft cabin interior, a narrow enclosed sci-fi space with curved metal walls, glowing control panels, soft blue-white lighting and floating dust in the air.

summary:
[reference generation] The target video is a 10-second cinematic sci-fi action short set inside <Picture 3>, showing <Picture 1> beginning to mutate as her eyes turn bloodshot and her pupils dilate, before she leaps into an airborne scissor takedown, whipping her body and the man around in a full rotation and landing on one knee against <Picture 2>.

retention_analysis:
<Picture 1> (female hero throughout): partially_preserved - her hair, skin, build and white top are retained, while her eyes transform to bloodshot with dilated pupils as the mutation takes hold.
<Picture 2> (male target throughout): fully_preserved - the young man's dark hair, fitted dark suit and appearance are retained across all shots.
<Picture 3> (set / environment): fully_preserved - the spacecraft cabin with its curved metal walls, glowing panels, and blue-white light is retained as the backdrop.

detailed_description:
The target video is in a gritty cinematic sci-fi action style with desaturated blue-metal tones, harsh practical lighting from glowing panels, and dynamic, unstable handheld and rapid-cut camera moves.
[Shot 1] A medium shot establishes <Picture 3>, the cramped spacecraft cabin with curved metal walls and softly glowing control panels, blue-white light flickering and thin dust hanging in the air. The young woman from <Picture 1>, in her white top with loose long hair, sways side to side unsteadily in the center, her posture rigid and strained. A slow push-in tightens as the camera edges toward her face. Cutting to a close-up, her eyes fill the frame: fine red veins appear and spread across her sclera as her dark pupils dilate unnaturally, the bloodshot effect intensifying, a low involuntary breath escaping her.
[Shot 2] At 00:03.000, the shot cuts to a medium two-shot of the young man from <Picture 2>, in his fitted dark suit, standing alert near the far wall, and the now-dilated young woman from <Picture 1> facing him. She drops into a low stance, springs explosively upward and launches her whole body into the air, twisting sideways as both legs drive out into an open scissors shape; her thighs and shins clamp around the man's neck and shoulders. Hooking her legs, she rotates her hips and torso through a full spin, carrying the man's upper body around with her in a complete 360-degree rotation before slamming down. She lands on one knee, planted firmly on the metal floor with a sharp metallic clang, the man crumpled and driven to the ground in front of her.
[Shot 3] At 00:06.000, the shot cuts to a tight close-up of the young woman's upper body from <Picture 1>, her hair disheveled and falling across her face, her bloodshot eyes with dilated pupils locked forward, breathing hard, a faint sheen of sweat on her skin under the flickering blue-white light. The camera holds a shaky close framing as her expression hardens, then begins a gentle pull-back into the dim metal gloom of the cabin as the segment ends.

overall_soundscape:
A low, tense mechanical hum of the spacecraft cabin, faint electronic beeps from the control panels, the rustle of fabric, heavy breathing, and the sharp metallic clang of the landing impact continue throughout.

non_diegetic_music:
A pulsing, brooding sci-fi score with low synth drones and a steady dark heartbeat-like pulse, intensifying during the mutation close-up and cutting sharply in the fight, then fading into uneasy silence at the end.
```

将以上内容放入节点的 `system_prompt`，通过 `image_instruction` 描述你的参考
图片，节点就会按上述格式返回一段完整的视频生成提示词。

---

## 安全说明

在节点中输入你的 `api_key` 会将其保存在任何已保存工作流图片的
**工作流元数据**中。对于共享的工作流，建议使用无需密钥的端点，或轮换被
共享的密钥。

---

## 开发与测试

```bash
python -m unittest discover -s tests
```

测试套件会模拟网络/依赖模块，无需真实端点即可运行。

## 许可证

MIT
