/**
 * Dynamic image inputs + widget hints for the ComfyUI LLM nodes.
 * Based on the cozy_ex_dynamic pattern for clean dynamic inputs.
 */

import { app } from "../../../scripts/app.js"

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

// Dynamic input prefix per node. Only nodes present here get dynamic slots.
const NODE_PREFIXES = {
    "CSYIDC-RefDescribe": "ref_image",
};

// Maximum number of connected dynamic slots per node.
const MAX_SLOTS = {
    "CSYIDC-RefDescribe": 15,
};

const TYPE = "IMAGE";

app.registerExtension({
    name: 'CSYIDCAPI.DynamicImageInputs',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Skip if not one of our nodes.
        const PREFIX = NODE_PREFIXES[nodeData.name];
        if (!PREFIX) {
            return;
        }
        const MAX = MAX_SLOTS[nodeData.name] || 10;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);

            // Bilingual (EN/CN) widget display labels. Internal widget names are
            // kept unchanged so saved workflows remain compatible.
            // 中英双语 widget 显示名（内部键名保持不变，确保工作流兼容）。
            const widgetLabels = {
                "base_url": "base_url (接口地址，必填)",
                "api_key": "api_key (接口密钥/Bearer Token)",
                "system_prompt": "system_prompt (系统提示词)",
                "image_instruction": "image_instruction (单图描述指令)",
                "model": "model (模型ID)",
                "temperature": "temperature (温度)",
                "reasoning_effort": "reasoning_effort (推理强度)",
                "max_tokens": "max_tokens (最大生成长度)",
                "max_context_tokens": "max_context_tokens (最大上下文长度)",
                "seed": "seed (随机种子)",
                "top_p": "top_p (核采样)",
                "top_k": "top_k (top-k 采样)",
                "frequency_penalty": "frequency_penalty (频率惩罚)",
                "presence_penalty": "presence_penalty (存在惩罚)",
                "stop": "stop (停止序列)",
                "request_timeout": "request_timeout (请求超时/秒)",
                "max_image_size": "max_image_size (图片最长边/px, 0=不缩放) ",
                "user_instruction": "user_instruction (用户指令)",
            };

            for (const w of this.widgets ?? []) {
                if (widgetLabels[w.name]) {
                    w.label = widgetLabels[w.name];
                }
            }

            // Ensure there is always one empty trailing dynamic slot so the
            // user can connect another image. Names stay unique/sequential.
            const last = this.getLastDynamicSlot(PREFIX);
            if (!last || last.link !== null) {
                this.addInput(this._nextPlaceholderName(PREFIX), TYPE);
                const slot = this.inputs[this.inputs.length - 1];
                if (slot) {
                    slot.color_off = "#666";
                }
            }

            return me;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);

            if (slotType !== TypeSlot.Input) {
                return me;
            }
            if (node_slot && !this.isDynamicSlot(node_slot.name)) {
                return me;
            }

            if (link_info && event === TypeSlotEvent.Connect) {
                const fromNode = this.graph?._nodes.find(
                    (otherNode) => otherNode.id == link_info.origin_id
                );
                if (fromNode) {
                    const parent_link = fromNode.outputs[link_info.origin_slot];
                    if (parent_link) {
                        node_slot.type = parent_link.type;
                    }
                }
                node_slot.name = this._nextPlaceholderName(PREFIX);
            }

            // Remove empty dynamic slots from highest index to lowest,
            // keeping at most the trailing (last) empty one.
            let toRemove = [];
            for (let i = this.inputs.length - 2; i >= 0; i--) {
                const s = this.inputs[i];
                if (this.isDynamicSlot(s.name) && s.link === null) {
                    toRemove.push(i);
                }
            }
            for (const removeIdx of toRemove) {
                this.removeInput(removeIdx);
            }

            // Collect the dynamic slots, in display order, keeping the list of
            // connected ones first followed by at most one empty trailing slot.
            const dynamic = ((this.inputs || []).filter(s => this.isDynamicSlot(s.name)));
            const connected = dynamic.filter(s => s.link !== null);
            const emptyTrail = dynamic.filter(s => s.link === null && s === this.getLastDynamicSlot(PREFIX));

            // Renumber continuously: connected slots get ref_image_1..N, the
            // single empty trailing slot (if any) gets the next number.
            connected.forEach((s, idx) => { s.name = `${PREFIX}_${idx + 1}`; });
            if (emptyTrail.length) {
                emptyTrail[0].name = `${PREFIX}_${connected.length + 1}`;
            }
            const dynamicCount = connected.length;

            // Ensure an empty trailing slot exists, respecting the max count.
            const lastInput = this.getLastDynamicSlot(PREFIX);
            if (dynamicCount < MAX && (!lastInput || lastInput.link !== null)) {
                this.addInput(`${PREFIX}_${dynamicCount + 1}`, TYPE);
                const newSlot = this.inputs[this.inputs.length - 1];
                if (newSlot) {
                    newSlot.color_off = "#666";
                }
            }

            this?.graph?.setDirtyCanvas(true);
            return me;
        };

        // Prototype helpers for this node.
        const proto = nodeType.prototype;
        proto.isDynamicSlot = function (name) {
            return typeof name === "string" && name.startsWith(PREFIX + "_");
        };
        proto._nextPlaceholderName = function (prefix) {
            let n = 1;
            const names = new Set((this.inputs || []).map(i => i.name));
            while (names.has(`${prefix}_${n}`)) {
                n += 1;
            }
            return `${prefix}_${n}`;
        };
        proto.getLastDynamicSlot = function (prefix) {
            let last = null;
            for (let i = (this.inputs || []).length - 1; i >= 0; i--) {
                const s = this.inputs[i];
                if (s.name.startsWith(prefix + "_")) {
                    last = s;
                    break;
                }
            }
            return last;
        };

        return nodeType;
    },
});
