# SPDX-License-Identifier: Apache-2.0
# ============================================================
# CosyVoice2 vLLM 推理模型定义 — 华为 NPU 910B 适配版
#
# 与 GPU 版 (cosyvoice2_vllm.py) 的差异:
#   1. 兼容 vLLM 0.9 (V0 引擎) 和 vLLM 0.17 (V1 引擎)
#   2. vLLM 0.17 的 V1 引擎使用 spawn 子进程:
#      - 注册用字符串格式 "module:class" (跨进程传递)
#      - patch is_text_generation_model (在 register_model 之前)
#   3. is_text_generation_model 类属性 (vLLM 0.17 检查)
#
# 使用方式:
#   docker-compose.npu.yml 中挂载:
#     - ./cosyvoice2_npu.py:/workspace/CosyVoice/cosyvoice/vllm/cosyvoice2.py
# ============================================================

"""Inference-only Qwen2-based CosyVoice2 model for vLLM (NPU version)."""

from typing import Iterable, Optional, Union
from packaging.version import parse as vparse
import torch
from torch import nn

import vllm

# ── 版本检测 ──
VLLM_VERSION = vparse(vllm.__version__.split('+')[0])
VLLM_V1_ENGINE_ONLY: bool = VLLM_VERSION >= vparse("0.11.0")

if VLLM_V1_ENGINE_ONLY:
    from vllm.v1.sample.metadata import SamplingMetadata

from vllm.model_executor.models.qwen2 import *


class CosyVoice2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """
    CosyVoice2 语音 Token 生成模型 (vLLM 推理专用)。
    基于 Qwen2 架构, 支持 vLLM 0.9 和 0.17。
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # ── vLLM 0.17: 声明为文本生成模型 ──
    is_text_generation_model = True

    @classmethod
    def is_backend_compatible(cls):
        """vLLM 0.9+ 兼容性检查。"""
        return True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size, config.hidden_size, True,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> Optional[torch.Tensor]:
        if VLLM_V1_ENGINE_ONLY:
            logits = self.logits_processor(self.lm_head, hidden_states,
                                           self.lm_head.bias)
        else:
            logits = self.logits_processor(self.lm_head, hidden_states,
                                           sampling_metadata, self.lm_head.bias)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)


# ══════════════════════════════════════════════════════════════
#  注册到 vLLM ModelRegistry
# ══════════════════════════════════════════════════════════════

# vLLM 0.17: patch is_text_generation_model, 必须在 register_model 之前
# register_model 内部调 _ModelInfo.from_model_cls → is_text_generation_model()
# 原始函数用 isinstance(model, VllmModelForTextGeneration) 判断,
# 我们的类不继承该 Protocol, 需要 patch 让它识别 CosyVoice2ForCausalLM
if VLLM_V1_ENGINE_ONLY:
    try:
        import vllm.model_executor.models.registry as _reg_module
        _original_is_text_gen = _reg_module.is_text_generation_model
        def _patched_is_text_gen(model):
            if getattr(model, "__name__", "") == "CosyVoice2ForCausalLM":
                return True
            return _original_is_text_gen(model)
        _reg_module.is_text_generation_model = _patched_is_text_gen
    except Exception:
        pass

from vllm import ModelRegistry

if VLLM_V1_ENGINE_ONLY:
    # vLLM 0.17: 字符串格式注册, 支持 spawn 子进程
    # 子进程会自动 import cosyvoice.vllm.cosyvoice2 并加载 CosyVoice2ForCausalLM
    ModelRegistry.register_model(
        "CosyVoice2ForCausalLM",
        "cosyvoice.vllm.cosyvoice2:CosyVoice2ForCausalLM",
    )
else:
    # vLLM 0.9: 直接传类实例
    ModelRegistry.register_model(
        "CosyVoice2ForCausalLM",
        CosyVoice2ForCausalLM,
    )
