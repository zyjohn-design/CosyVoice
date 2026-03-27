# SPDX-License-Identifier: Apache-2.0
# ============================================================
# CosyVoice2 vLLM 推理模型定义
# 适配: vLLM 0.9.0 + transformers 4.51.3
#
# 策略: 直接注册到 vLLM 的 ModelRegistry,
#       绕过 transformers AutoModel.from_config() 的兼容性检查。
#       这是 CosyVoice 官方 vllm_example.py 的预期加载方式。
# ============================================================

"""Inference-only Qwen2-based CosyVoice2 model for vLLM."""

from typing import Iterable, Optional, Union
from packaging.version import parse as vparse
import torch
from torch import nn

import vllm

VLLM_V1_ENGINE_ONLY: bool = vparse(vllm.__version__) >= vparse("0.11.0")
if VLLM_V1_ENGINE_ONLY:
    from vllm.v1.sample.metadata import SamplingMetadata

from vllm.model_executor.models.qwen2 import *


class CosyVoice2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """
    CosyVoice2 语音 Token 生成模型 (vLLM 推理专用)。
    基于 Qwen2 架构, 直接注册到 vLLM ModelRegistry。
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def is_backend_compatible(cls):
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


# ── 注册到 vLLM ModelRegistry ──
# 这样 vLLM 通过 architectures 字段直接找到这个类,
# 不经过 transformers 的 AutoModel / auto_map 路径
from vllm import ModelRegistry
ModelRegistry.register_model(
    "CosyVoice2ForCausalLM",
    CosyVoice2ForCausalLM,
)
