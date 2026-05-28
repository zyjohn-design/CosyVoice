# npu_optimizer.py
"""
CosyVoice NPU 910B 推理优化套件 v4 (★ v5 已转为 fallback 路径)
==============================================================

★ v5 重要变更: 主推理路径已切换到 vLLM-Ascend (LOAD_VLLM=true)
   - 见 docker-compose.npu.yml: LOAD_VLLM 默认 true
   - 见 app_npu.py:675: LOAD_VLLM=true 时本模块完全不加载
   - v3/v4 自研 NPU Fused Attention 有早停 bug (本模块代码已实证)
   本文件保留供 vLLM 跑不通时作为 fallback (LOAD_VLLM=false)。
   Flow 优化 (timesteps=4, CFG-free) 在 fallback 路径下仍然有效。

通过 monkey-patch 实现优化，不修改主项目代码。
在 app_npu.py 模型加载后调用 apply_npu_compile() 即可 (仅 fallback 路径)。

v4 更新 — 尝试修复 v3 早停 bug (实测未根治)
──────────────────────────────────────────────────────────────
v3 的 LLM Fused Attention 实测有 60-80% 速度提升，但 LLM 输出 token
早停 (speech_len 仅为正常的 26-47%)，导致语音被截断。
v4 尝试的层选择性 (skip_last_4) + inner_precise=0 + decode sparse=0
组合实测后 speech_len 仍只有正常的 24-50%, 未根治。结论: 算子级
monkey-patch 与 transformers 框架语义对齐困难, 应改用 vLLM-Ascend。

v4 通过三个互补修复解决数值漂移问题：

  ★ 修复 1: 层选择性 fused attention (主)
    - NPU_FUSED_ATTN_LAYERS="skip_last_4" (默认)
    - 最后 4 层走 eager 保留 logit 精度，前 20 层用 fused 加速
    - 原理: FP16 softmax 误差沿 24 层累积，最后几层直接影响 lm_head
            输出 logits, 对早停判断最敏感

  ★ 修复 2: inner_precise=0 (高精度 softmax)
    - npu_fusion_attention 显式声明 inner_precise=0
    - 强制 softmax 中间累加用 FP32, 不让 NPU 自动降到 FP16 模式

  ★ 修复 3: decode 路径 sparse_mode=0 (无 mask)
    - decode (S_q=1) 时单个 query 应看到所有 K, 不需 causal mask
    - sparse_mode=3 在 S_q << S_kv 时可能有边界 bug

优化策略 (按收益排序):
──────────────────────────────────────────────────────────────
[1] Flow Matching n_timesteps: 10 → 4                  ★★★ (生效)
[2] CFG-free solve_euler: batch 2 → 1                  ★★★ (生效)
[3] NPU Stream 上下文修复                              ★   (生效)
[4] 模块级 profiling 诊断                              📊  (诊断)
[5] torch.compile flow.estimator (实验)                ⚙   (默认关)
[6] LLM dtype 诊断 + 选择性 FP16                       📊  (默认关)
[7] torch.compile LLM (实验)                           ⚙   (默认关)
[8] ★★★ NPU Fused Attention (v4 主战场)               ★★★ (待验证)

核心环境变量:
    NPU_FLOW_TIMESTEPS=4               Flow ODE 步数 (3-6)
    NPU_DISABLE_CFG=true               禁用 CFG，batch=1 推理
    NPU_PROFILE=false                  启用 NPU 同步耗时诊断
    NPU_FUSED_ATTN=true                ★ 启用 NPU 融合 attention
    NPU_FUSED_ATTN_LAYERS=skip_last_4  ★ 仅前 20 层用 fused
    NPU_FUSED_ATTN_INNER_PRECISE=0     ★ 高精度 softmax (FP32 累加)
    NPU_FUSED_ATTN_DECODE_SPARSE=0     ★ decode 无 mask
"""

import os
import math
import time
import logging
import functools
from contextlib import nullcontext
from collections import defaultdict

import torch

logger = logging.getLogger("cosyvoice-api")

# ── 配置 ──
NPU_FLOW_TIMESTEPS = int(os.getenv("NPU_FLOW_TIMESTEPS", "4"))
NPU_DISABLE_CFG = os.getenv("NPU_DISABLE_CFG", "true").lower() in ("true", "1")
NPU_PROFILE = os.getenv("NPU_PROFILE", "false").lower() in ("true", "1")
NPU_COMPILE_ENABLED = os.getenv("NPU_COMPILE", "false").lower() in ("true", "1")
# ── LLM 优化（针对 LLM 占总耗时 97% 的瓶颈）──
NPU_COMPILE_LLM = os.getenv("NPU_COMPILE_LLM", "false").lower() in ("true", "1")
# NPU_FORCE_LLM_FP16: "true" 全转 / "selective" 仅 Linear/Conv / "false" 不转
NPU_FORCE_LLM_FP16 = os.getenv("NPU_FORCE_LLM_FP16", "false").lower() in ("true", "1", "selective")
# NPU_FUSED_ATTN: 启用 torch_npu.npu_*_flash_attention 替代 eager attention（重点优化）
NPU_FUSED_ATTN = os.getenv("NPU_FUSED_ATTN", "false").lower() in ("true", "1", "all", "decode_only", "prefill_only")
# NPU_FUSED_ATTN_MODE:
#   "all"          (默认) — prefill 和 decode 都用 fused
#   "decode_only"  — 只 decode 用 fused，prefill 走 eager（调试用）
#   "prefill_only" — 只 prefill 用 fused，decode 走 eager（调试用）
NPU_FUSED_ATTN_MODE = os.getenv("NPU_FUSED_ATTN_MODE", "all").lower()
# 直接传 "decode_only"/"prefill_only" 给 NPU_FUSED_ATTN 也能切换 mode
_attn_short = os.getenv("NPU_FUSED_ATTN", "false").lower()
if _attn_short in ("decode_only", "prefill_only"):
    NPU_FUSED_ATTN_MODE = _attn_short
# NPU_FUSED_ATTN_DEBUG:
#   "true"/"1"/"compare": 同时跑 fused + eager，记录 diff，返回 eager (默认 debug)
#   "eager_only": 完全不调 NPU，验证钩子本身（dispatch + eager_attention_forward）是否会破坏输出
NPU_FUSED_ATTN_DEBUG = os.getenv("NPU_FUSED_ATTN_DEBUG", "false").lower() in (
    "true", "1", "compare", "eager_only"
)
NPU_FUSED_ATTN_DEBUG_N = int(os.getenv("NPU_FUSED_ATTN_DEBUG_N", "8"))

# ══════════════════════════════════════════════════════════════
#  v4 修复：解决 LLM 早停 bug 的关键参数
# ══════════════════════════════════════════════════════════════
# NPU_FUSED_ATTN_LAYERS: 选择性应用 fused attention 到指定层
#   - "all"        (默认旧版) — 所有层都用 fused (会导致早停)
#   - "skip_last_N" — 跳过最后 N 层 (推荐: skip_last_4)，保留 logit 层精度
#   - "0-19"       — 仅 layer 0 到 19 使用 fused (Qwen2.5-0.5B 共 24 层)
#   - "0,2,4-10"   — 任意组合
# 原理: 早停 bug 源于 FP16 softmax 精度损失沿 24 层累积，影响最终 logits。
#       last layers 直接输出到 lm_head，对精度最敏感；前面 layers 可以容忍。
NPU_FUSED_ATTN_LAYERS = os.getenv("NPU_FUSED_ATTN_LAYERS", "skip_last_4")

# NPU_FUSED_ATTN_INNER_PRECISE: npu_fusion_attention 的 inner_precise 参数
#   - 0 (默认推荐) — 高精度模式，softmax 中间 tensor 用 FP32 累加
#   - 1            — 高性能模式，全 FP16，可能造成数值漂移 → 早停
NPU_FUSED_ATTN_INNER_PRECISE = int(os.getenv("NPU_FUSED_ATTN_INNER_PRECISE", "0"))

# NPU_FUSED_ATTN_DECODE_SPARSE: decode 路径 (S_q=1) 的 sparse_mode
#   - 0 (默认)    — 无 mask, 单 query 看到所有 K (语义上最正确)
#   - 3           — rightDownCausal (理论等价但可能有边界 bug)
NPU_FUSED_ATTN_DECODE_SPARSE = int(os.getenv("NPU_FUSED_ATTN_DECODE_SPARSE", "0"))


# ══════════════════════════════════════════════════════════════
#  模块级 Profiler
# ══════════════════════════════════════════════════════════════

class _NpuProfiler:
    """
    收集 llm/flow/hift 各模块的推理耗时统计。

    NPU 计算是异步的，必须在测量前后调用 torch.npu.synchronize()
    才能得到真实耗时。这会引入额外同步开销，故默认关闭。
    """

    def __init__(self):
        self.stats = defaultdict(list)
        self.enabled = False

    def record(self, name: str, duration_s: float):
        if not self.enabled:
            return
        self.stats[name].append(duration_s)
        # 单次推理实时日志
        logger.info(f"[NPU profile] {name:6s}: {duration_s*1000:7.1f} ms")
        # 每 10 次推理输出一次汇总
        if len(self.stats[name]) % 10 == 0:
            self._log_summary()

    def _log_summary(self):
        lines = ["[NPU profile] ─── 阶段耗时汇总 ───"]
        total_avg = 0.0
        for name in ["llm", "flow", "hift"]:
            durs = self.stats.get(name, [])
            if not durs:
                continue
            avg_ms = sum(durs) / len(durs) * 1000
            total_avg += avg_ms
            lines.append(
                f"  {name:6s}: n={len(durs):3d} | "
                f"avg={avg_ms:7.1f}ms | "
                f"min={min(durs)*1000:6.1f}ms | "
                f"max={max(durs)*1000:6.1f}ms"
            )
        if total_avg > 0:
            lines.append(f"  合计 avg = {total_avg:.1f}ms / 次")
            for name in ["llm", "flow", "hift"]:
                durs = self.stats.get(name, [])
                if durs:
                    pct = sum(durs) / len(durs) * 1000 / total_avg * 100
                    lines.append(f"     {name:6s} 占比 {pct:5.1f}%")
        logger.info("\n".join(lines))

    def report(self) -> str:
        """手动获取统计报告（可在 /health 等接口里返回）"""
        if not self.stats:
            return "no profile data"
        self._log_summary()
        return str({k: {
            "count": len(v),
            "avg_ms": sum(v) / len(v) * 1000,
            "total_s": sum(v),
        } for k, v in self.stats.items()})

    def reset(self):
        self.stats.clear()


_profiler = _NpuProfiler()


def dump_profile() -> str:
    """对外接口，返回当前 profiling 汇总。"""
    return _profiler.report()


def reset_profile():
    """对外接口，清空 profiling 数据。"""
    _profiler.reset()


def _wrap_inference(module, name: str, is_generator: bool = False):
    """
    包装 module.inference 做 NPU 同步耗时统计。

    Args:
        module:       目标 nn.Module (llm / flow / hift)
        name:         统计标签
        is_generator: True 表示原方法返回生成器（LLM 自回归解码用）
    """
    if module is None or not hasattr(module, "inference"):
        return
    if getattr(module, "_npu_timed", False):
        return

    orig = module.inference
    _sync = torch.npu.synchronize if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize") else (lambda: None)

    if is_generator:
        @functools.wraps(orig)
        def timed(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            try:
                gen = orig(*args, **kwargs)
                for item in gen:
                    yield item
            finally:
                _sync()
                _profiler.record(name, time.perf_counter() - t0)
    else:
        @functools.wraps(orig)
        def timed(*args, **kwargs):
            _sync()
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                _sync()
                _profiler.record(name, time.perf_counter() - t0)

    module.inference = timed
    module._npu_timed = True
    logger.info(f"[NPU profile] 已包装 {name}.inference 做耗时统计")


def _enable_profiling(inner):
    """对 llm / flow / hift 三个模块分别添加耗时统计。"""
    _profiler.enabled = True
    _wrap_inference(getattr(inner, "llm", None), "llm", is_generator=True)
    _wrap_inference(getattr(inner, "flow", None), "flow", is_generator=False)
    _wrap_inference(getattr(inner, "hift", None), "hift", is_generator=False)


# ══════════════════════════════════════════════════════════════
#  优化 1: Flow Matching n_timesteps 缩减
# ══════════════════════════════════════════════════════════════

def _patch_flow_decoder_timesteps(flow):
    """覆盖 flow.decoder.forward 的 n_timesteps 为 NPU_FLOW_TIMESTEPS。"""
    decoder = getattr(flow, "decoder", None)
    if decoder is None:
        logger.warning("[NPU] flow.decoder 不存在，跳过 timesteps patch")
        return False

    cls = type(decoder)
    if getattr(cls, "_npu_ts_patched", False):
        return True

    _orig = cls.forward
    _n = NPU_FLOW_TIMESTEPS

    def _fast_forward(self, mu, mask, n_timesteps, *args, **kwargs):
        return _orig(self, mu, mask, _n, *args, **kwargs)

    cls.forward = _fast_forward
    cls._npu_ts_patched = True
    logger.info(
        f"[NPU优化1] flow.decoder ({cls.__name__}) n_timesteps: 10 → {_n} "
        f"(Flow 步数减少 {int((1 - _n/10)*100)}%)"
    )
    return True


# ══════════════════════════════════════════════════════════════
#  优化 2: CFG-free solve_euler (batch 2 → 1)
# ══════════════════════════════════════════════════════════════

def _patch_solve_euler_cfg_free(flow):
    """
    替换 CausalConditionalCFM.solve_euler 为 CFG-free 版本。

    原始实现 (cosyvoice/flow/flow_matching.py:71-124):
      - 每步用 batch=2 跑 estimator (一份带条件，一份无条件)
      - 然后做 CFG 线性组合: (1+r)*cond - r*uncond
      - batch=2 的 estimator forward 是计算瓶颈

    替换后:
      - batch=1，跳过 CFG 分支
      - estimator forward 计算量减半（NPU 上接近线性收益）
      - 数学等价于 inference_cfg_rate=0 但避免无效 batch 计算

    质量影响:
      - 损失 CFG 带来的「条件强化」效果
      - 对短文本 / 单语种 / 已注册说话人场景影响极小
      - 对长文本跨语种克隆，相似度可能下降 5-10%
    """
    decoder = getattr(flow, "decoder", None)
    if decoder is None:
        return False

    cls = type(decoder)
    if getattr(cls, "_npu_cfg_free", False):
        return True

    def _cfg_free_solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        """CFG-free Euler ODE solver, batch=1."""
        t = t_span[0].unsqueeze(dim=0)
        dt = t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            dphi_dt = self.forward_estimator(
                x, mask, mu, t, spks, cond, streaming
            )
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x.float()

    cls.solve_euler = _cfg_free_solve_euler
    cls._npu_cfg_free = True
    logger.info(
        f"[NPU优化2] {cls.__name__}.solve_euler 替换为 CFG-free 版本 "
        f"(estimator batch 2→1, Flow 计算量再减 ~45%)"
    )
    return True


# ══════════════════════════════════════════════════════════════
#  优化 3: NPU Stream 上下文修复
# ══════════════════════════════════════════════════════════════

def _fix_npu_stream_context(inner):
    """修复 llm_context: nullcontext → torch.npu.Stream（如有必要）。"""
    llm_ctx = getattr(inner, "llm_context", None)
    if not isinstance(llm_ctx, nullcontext):
        logger.debug("[NPU] llm_context 已是非空上下文，跳过修复")
        return

    try:
        npu_stream = torch.npu.Stream(inner.device)
        inner.llm_context = torch.npu.stream(npu_stream)
        logger.info("[NPU优化3] llm_context → torch.npu.Stream (LLM 独立 NPU 流)")
    except Exception as e:
        logger.warning(f"[NPU] llm_context 修复失败，保持 nullcontext: {e}")


# ══════════════════════════════════════════════════════════════
#  优化 5: torch.compile (实验性)
# ══════════════════════════════════════════════════════════════

_COMPILE_BACKENDS = ["inductor", "aot_eager", "eager"]


def _try_compile_estimator(flow):
    """对 flow.decoder.estimator 尝试 torch.compile (受 NPU_COMPILE 控制)。"""
    if not NPU_COMPILE_ENABLED:
        return

    decoder = getattr(flow, "decoder", None)
    if decoder is None:
        return

    estimator = getattr(decoder, "estimator", None)
    if estimator is None or not isinstance(estimator, torch.nn.Module):
        return

    for backend in _COMPILE_BACKENDS:
        try:
            compiled = torch.compile(
                estimator, backend=backend, dynamic=True, fullgraph=False,
            )
            decoder.estimator = compiled
            logger.info(f"[NPU优化5] flow.decoder.estimator 编译成功 (backend={backend})")
            return
        except Exception as e:
            logger.debug(f"[NPU compile] backend={backend} 失败: {e}")

    logger.info("[NPU compile] 所有 backend 失败，使用 eager 模式")


# ══════════════════════════════════════════════════════════════
#  优化 6: LLM dtype 诊断 + 强制 FP16
# ══════════════════════════════════════════════════════════════

def _diagnose_llm_dtype(inner):
    """
    诊断 LLM 各子模块的权重 dtype，确认 FP16 是否真的生效。

    背景: cosyvoice_model 加载时即使传 fp16=True，也只是设置了
    self.fp16=True 然后用于 torch.cuda.amp.autocast 包装。
    autocast 只影响中间张量，不改变模型权重。如果权重本身是 FP32，
    每次 matmul 都会上转 FP32，等于没开 FP16。

    本函数遍历 inner.llm 找到 transformer 主体，打印权重 dtype。
    """
    llm = getattr(inner, "llm", None)
    if llm is None:
        return None

    # CosyVoice2 的 LLM 结构: inner.llm.llm 是 HuggingFace Qwen2 主体
    candidates = []
    for attr in ("llm", "text_encoder", "speech_encoder"):
        sub = getattr(llm, attr, None)
        if isinstance(sub, torch.nn.Module):
            candidates.append((f"llm.{attr}", sub))

    if not candidates:
        # 退化：直接看 llm 本身
        if isinstance(llm, torch.nn.Module):
            candidates.append(("llm", llm))

    dtypes_seen = set()
    for name, mod in candidates:
        try:
            first_param = next(mod.parameters(), None)
            if first_param is not None:
                dt = str(first_param.dtype)
                dtypes_seen.add(dt)
                logger.info(f"[NPU诊断] {name} 首参数 dtype = {dt}")
        except Exception as e:
            logger.debug(f"[NPU诊断] {name} 读取失败: {e}")

    return dtypes_seen


def _force_llm_half(inner):
    """
    强制 LLM 主体权重转为 FP16。

    NPU_FORCE_LLM_FP16=true       → 全部 .half()
    NPU_FORCE_LLM_FP16=selective  → 仅 Linear / Conv，保留 LayerNorm/Embedding FP32

    背景: 实测「全 .half()」反而让 LLM 慢 ~10%。猜测是 LayerNorm/Embedding
    在 NPU 的 FP16 路径未充分优化，或与 autocast 冲突造成多次 dtype 转换。
    selective 模式只动权重 matmul 密集的 Linear，最安全。
    """
    llm = getattr(inner, "llm", None)
    if llm is None:
        return

    # 找到 HF transformer 主体
    target = getattr(llm, "llm", None)
    if target is None or not isinstance(target, torch.nn.Module):
        target = llm if isinstance(llm, torch.nn.Module) else None

    if target is None:
        logger.warning("[NPU优化6] 找不到 LLM transformer 主体，跳过 FP16 转换")
        return

    try:
        first_param = next(target.parameters(), None)
        if first_param is None:
            return
        orig_dtype = first_param.dtype
        if orig_dtype == torch.float16:
            logger.info("[NPU优化6] LLM 权重已是 FP16，无需转换")
            return

        mode = os.getenv("NPU_FORCE_LLM_FP16", "false").lower()
        if mode == "selective":
            # 选择性: 只转 Linear/Conv 系列
            count = 0
            target_types = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
            for m in target.modules():
                if isinstance(m, target_types):
                    m.half()
                    count += 1
            logger.info(
                f"[NPU优化6] LLM 选择性 FP16: {count} 个 Linear/Conv 已转换 "
                f"(LayerNorm/Embedding 保持 {orig_dtype})"
            )
        else:
            # 默认: 全部转
            target.half()
            logger.info(
                f"[NPU优化6] LLM transformer 全部转为 FP16 "
                f"({orig_dtype} → torch.float16)"
            )
    except Exception as e:
        logger.warning(f"[NPU优化6] LLM FP16 转换失败: {e}")


# ══════════════════════════════════════════════════════════════
#  优化 7: torch.compile LLM (主战场)
# ══════════════════════════════════════════════════════════════

def _try_compile_llm(inner):
    """
    对 LLM transformer 主体尝试 torch.compile。

    profiling 显示 LLM 占总耗时 97-98%，每 token ~44ms。
    torch.compile 可减少 Python 解释器开销 + kernel launch 开销，
    对自回归解码循环理论收益 30-50%。

    策略: 编译 inner.llm.llm（HF 模型主体），不动外层包装。
    backend 优先级: inductor → aot_eager → eager (兜底)
    """
    if not NPU_COMPILE_LLM:
        return

    llm = getattr(inner, "llm", None)
    if llm is None:
        return

    target = getattr(llm, "llm", None)
    if target is None or not isinstance(target, torch.nn.Module):
        logger.info("[NPU优化7] llm.llm 不是 nn.Module，跳过 compile")
        return

    for backend in _COMPILE_BACKENDS:
        try:
            compiled = torch.compile(
                target, backend=backend, dynamic=True, fullgraph=False,
            )
            llm.llm = compiled
            logger.info(f"[NPU优化7] LLM 编译成功 (backend={backend})")
            return
        except Exception as e:
            logger.debug(f"[NPU compile LLM] backend={backend} 失败: {e}")

    logger.info("[NPU compile LLM] 所有 backend 失败，使用 eager 模式")


# ══════════════════════════════════════════════════════════════
#  优化 8: NPU Fused Attention (★★★ 主战场)
# ══════════════════════════════════════════════════════════════
#
# 通过替换 transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS 注册自定义
# attention impl，把 Qwen2Attention 的 SDPA/eager 路径换成 NPU 融合算子。
#
# decode 路径（每 token，热路径，占 LLM 99%）:
#   torch_npu.npu_incre_flash_attention
#   - 专为 Q_seq=1 的增量解码设计
#   - 单 kernel 完成 QK^T + softmax + V 乘
#   - 比 eager 路径快 2-3 倍
#
# prefill 路径（首次输入，1 次）:
#   torch_npu.npu_prompt_flash_attention
#   - 处理 Q_seq>1 的并行 attention
#   - sparse_mode=2 自动处理 causal mask
#
# 收益预估: LLM 每 token 44ms → 20-25ms，RTF 1.25 → ~0.60
# 风险: API 与 transformers 版本耦合，输出可能与 eager 有微小数值差异
# 兜底: 任何错误自动回退到 eager_attention_forward
# ══════════════════════════════════════════════════════════════

# 注意失败时只记录一次，避免日志刷屏
_NPU_ATTN_FAILED_ONCE = {"flag": False}
# 各种 mode skip 各只记录一次
_MODE_SKIP_LOGGED = {}

# ── v4 层选择性统计 ──
# 记录每一层 fused 调用次数，便于诊断
_LAYER_STATS = {"fused_by_layer": {}, "eager_by_layer": {}, "total_layers_seen": set()}
_TOTAL_LAYER_COUNT = {"value": None}  # 自动推断的 LLM 总层数


def _log_mode_skip(label):
    """模式跳过 fused 时打首次日志，证明 mode 真生效了。"""
    if label not in _MODE_SKIP_LOGGED:
        logger.info(f"[NPU attention] {label} 走 eager (mode={NPU_FUSED_ATTN_MODE})")
        _MODE_SKIP_LOGGED[label] = True


def _parse_layer_spec(spec: str, layer_idx: int, total_layers: int = 24) -> bool:
    """
    解析层选择规格，判断 layer_idx 是否应使用 fused attention。

    支持格式:
        "all"                — 所有层 (返回 True)
        "none"               — 无层 (返回 False)
        "skip_last_4"        — 跳过最后 4 层
        "skip_first_2"       — 跳过前 2 层
        "0-19"               — 仅 0 到 19 层
        "0,2,4-10"           — 离散+区间组合

    Args:
        spec:         配置字符串
        layer_idx:    当前层索引
        total_layers: LLM 总层数（用于解析 skip_last_N）

    Returns:
        True = 使用 fused; False = 使用 eager
    """
    s = spec.strip().lower()
    if s in ("all", "true", "1", ""):
        return True
    if s in ("none", "false", "0"):
        return False

    if s.startswith("skip_last_"):
        try:
            n = int(s.replace("skip_last_", ""))
            return layer_idx < (total_layers - n)
        except ValueError:
            return True

    if s.startswith("skip_first_"):
        try:
            n = int(s.replace("skip_first_", ""))
            return layer_idx >= n
        except ValueError:
            return True

    # 解析 "0-19" / "0,2,4-10" 格式
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                lo, hi = part.split('-', 1)
                if int(lo) <= layer_idx <= int(hi):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(part) == layer_idx:
                    return True
            except ValueError:
                continue
    return False


def _should_use_fused_for_layer(module) -> bool:
    """根据 NPU_FUSED_ATTN_LAYERS 配置 + module.layer_idx 决定是否走 fused。"""
    if NPU_FUSED_ATTN_LAYERS in ("all", ""):
        return True

    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is None:
        # 没有层信息，默认走 fused
        return True

    total = _TOTAL_LAYER_COUNT["value"] or 24  # Qwen2.5-0.5B 默认 24 层
    return _parse_layer_spec(NPU_FUSED_ATTN_LAYERS, layer_idx, total)


def _record_layer_call(module, used_fused: bool):
    """记录每一层 fused/eager 调用次数（用于诊断分布）。"""
    layer_idx = getattr(module, "layer_idx", -1)
    _LAYER_STATS["total_layers_seen"].add(layer_idx)
    bucket = "fused_by_layer" if used_fused else "eager_by_layer"
    _LAYER_STATS[bucket][layer_idx] = _LAYER_STATS[bucket].get(layer_idx, 0) + 1


def dump_attention_layer_stats() -> str:
    """对外接口：返回各层 fused/eager 调用分布。"""
    layers = sorted(_LAYER_STATS["total_layers_seen"])
    lines = ["[NPU attention layer stats]"]
    for L in layers:
        f = _LAYER_STATS["fused_by_layer"].get(L, 0)
        e = _LAYER_STATS["eager_by_layer"].get(L, 0)
        total = f + e
        if total == 0:
            continue
        pct = f / total * 100
        lines.append(f"  layer {L:2d}: fused={f:5d} eager={e:5d} ({pct:5.1f}% fused)")
    return "\n".join(lines)


def _npu_fused_attention_forward(module, query, key, value, attention_mask,
                                  scaling=None, dropout=0.0, is_causal=None, **kwargs):
    """
    NPU 融合 attention 实现，签名兼容 transformers attention_interface 协议。

    Args (transformers 约定):
        query:            [B, num_heads,    S_q,  head_dim]
        key:              [B, num_kv_heads, S_kv, head_dim]
        value:            [B, num_kv_heads, S_kv, head_dim]
        attention_mask:   可能 None / [B, 1, S_q, S_kv]
        scaling:          1/sqrt(head_dim) 或 module.scaling

    Returns:
        (attn_output, None)
        shape: [B, S_q, num_heads, head_dim]  ← 注意是 BSND 不是 BNSD!
        因为 transformers 后续会 attn_output.reshape(*input_shape, -1)，
        其中 input_shape=(B, S_q)，所以输入必须是 [B, S, ...] 形式。
        eager_attention_forward/sdpa_attention_forward 都在末尾做了 .transpose(1, 2)。
    """
    import torch_npu

    orig_dtype = query.dtype
    # ── 强制 FP16（NPU 融合算子只支持 FP16/BF16/INT8）──
    # transformers 的 autocast 在 Linear 层做了 FP16 cast，但有些路径会回到 FP32
    # 这里兜底，保证 NPU 算子永远拿到 FP16 输入
    if orig_dtype not in (torch.float16, torch.bfloat16):
        query = query.to(torch.float16)
        key = key.to(torch.float16)
        value = value.to(torch.float16)

    B, N_q, S_q, D = query.shape
    _, N_kv, S_kv, _ = key.shape

    # ── 调试模式: 部分路径强制走 eager fallback ──
    if NPU_FUSED_ATTN_MODE == "decode_only" and S_q > 1:
        _log_mode_skip("prefill (S_q={})".format(S_q))
        raise NotImplementedError("NPU_FUSED_ATTN_MODE=decode_only: prefill 走 eager")
    if NPU_FUSED_ATTN_MODE == "prefill_only" and S_q == 1:
        _log_mode_skip("decode (S_q=1)")
        raise NotImplementedError("NPU_FUSED_ATTN_MODE=prefill_only: decode 走 eager")

    # ── GQA 显式展开 ──
    # Qwen2.5-0.5B 是 14 Q heads + 2 KV heads (ratio 7)。
    if N_kv != N_q:
        rep = N_q // N_kv
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
        N_kv = N_q

    if scaling is None:
        scaling = 1.0 / math.sqrt(D)

    # ══════════════════════════════════════════════════════════
    # 统一融合算子: torch_npu.npu_fusion_attention (v4 改进版)
    # ══════════════════════════════════════════════════════════
    # 解决 v3 早停 bug 的关键：
    #   1. inner_precise=0 — 强制 softmax 中间累加用 FP32 (默认是 0，显式声明)
    #   2. decode (S_q=1)  — sparse_mode=0 (无 mask)，避免 sparse_mode=3 边界 bug
    #   3. prefill (S_q>1) — sparse_mode=3 (rightDownCausal causal mask)
    #   4. next_tockens=0  — 显式声明无未来 token (严格 causal)
    #
    # 返回 7-tuple: (output, softmax_max, softmax_sum, softmax_out, seed, offset, numels)
    # 推理时只取 [0]
    _is_decode = (S_q == 1)
    if _is_decode:
        # decode: 单 query 应看到所有 K (S_kv 个位置)，理论上无需 mask
        _sparse = NPU_FUSED_ATTN_DECODE_SPARSE  # 默认 0 (无 mask)
    else:
        # prefill: 标准 causal mask
        _sparse = 3  # rightDownCausal

    result = torch_npu.npu_fusion_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        head_num=N_q,
        input_layout="BNSD",
        scale=float(scaling),
        keep_prob=1.0,
        pre_tockens=65536,                          # 看所有历史 K
        next_tockens=0,                             # 严格 causal: 不看未来
        sparse_mode=_sparse,
        inner_precise=NPU_FUSED_ATTN_INNER_PRECISE, # 默认 0 = 高精度 FP32 softmax
    )
    out = result[0]  # [B, N, S, D]

    # BNSD → BSND（transformers 期望 [B, S, N, D]）
    out = out.transpose(1, 2).contiguous()

    # 还原 caller 期望的 dtype（与 query 一致）
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)

    return out, None


_DEBUG_DIFF_COUNT = {"prefill": 0, "decode": 0}


# 保存原始 sdpa attention 函数（由 _patch_qwen2_attention 在替换前赋值）
_ORIGINAL_SDPA_ATTN = None


def _run_eager_attention(module, query, key, value, attention_mask, **kwargs):
    """
    兜底 attention 调用入口。

    优先级:
      1. _ORIGINAL_SDPA_ATTN — 替换前保存的原版 sdpa
         ← 与基线行为完全一致，最稳妥（mask 是 sdpa 风格）
      2. transformers 当前注册的 sdpa（可能已被我们替换，不推荐）
      3. eager_attention_forward
      4. 手写 manual eager
    """
    # ── 1. 原始 sdpa（最稳）──
    if _ORIGINAL_SDPA_ATTN is not None:
        try:
            return _ORIGINAL_SDPA_ATTN(module, query, key, value, attention_mask, **kwargs)
        except Exception as e:
            logger.debug(f"[NPU attn] original sdpa 失败: {e}")

    # ── 2. 备选：手写兜底（与 eager 等价，避免可能被替换的注册表）──
    scaling = kwargs.get("scaling") or (1.0 / math.sqrt(query.shape[-1]))
    B, N_q, S_q, D = query.shape
    _, N_kv, S_kv, _ = key.shape
    if N_kv != N_q:
        rep = N_q // N_kv
        key = key.repeat_interleave(rep, dim=1)
        value = value.repeat_interleave(rep, dim=1)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(probs, value)
    out = out.transpose(1, 2).contiguous()
    return out, None


def _npu_fused_attention_with_fallback(module, query, key, value, attention_mask, **kwargs):
    """带兜底的 NPU attention：单 token 失败自动退回 eager，避免推理崩溃。

    NPU_FUSED_ATTN_DEBUG=true 时进入诊断模式：
      - 同时跑 fused 和 eager
      - 对比 max_abs_diff / rel_diff
      - 前 NPU_FUSED_ATTN_DEBUG_N 次 (prefill 和 decode 各 N 次) 打 INFO 日志
      - 强制返回 eager 输出（保证模型正确性）

    v4 新增：层选择性
      - 检查 NPU_FUSED_ATTN_LAYERS 配置
      - 不在白名单内的层直接走 eager（保留 logit 层精度，解决早停 bug）
    """
    # ── v4: 层选择性检查（最关键的早停 bug 修复）──
    use_fused = _should_use_fused_for_layer(module)
    _record_layer_call(module, used_fused=use_fused)
    if not use_fused:
        # 该层被排除（如 skip_last_4），直接走 eager 保留精度
        return _run_eager_attention(module, query, key, value, attention_mask, **kwargs)
    # ── 调试模式 ──
    # NPU_FUSED_ATTN_DEBUG=true 或 "compare": 同时跑 fused 和 eager，记录 diff，返回 eager
    # NPU_FUSED_ATTN_DEBUG=eager_only: 不调用 NPU，纯 eager（验证钩子本身是否有问题）
    if NPU_FUSED_ATTN_DEBUG:
        S_q = query.shape[2]
        S_kv = key.shape[2]
        is_decode = (S_q == 1)
        key_label = "decode" if is_decode else "prefill"

        # 模式: eager_only — 完全不调 NPU
        if os.getenv("NPU_FUSED_ATTN_DEBUG", "false").lower() == "eager_only":
            if _DEBUG_DIFF_COUNT[key_label] < NPU_FUSED_ATTN_DEBUG_N:
                _DEBUG_DIFF_COUNT[key_label] += 1
                logger.info(
                    f"[NPU attn debug] {key_label} #{_DEBUG_DIFF_COUNT[key_label]} "
                    f"S_q={S_q} S_kv={S_kv} | eager_only (NPU 未调用)"
                )
            return _run_eager_attention(module, query, key, value, attention_mask, **kwargs)

        # 模式: compare (默认 debug 模式)
        try:
            fused_out, _ = _npu_fused_attention_forward(
                module, query, key, value, attention_mask, **kwargs
            )
            eager_out, _ = _run_eager_attention(
                module, query, key, value, attention_mask, **kwargs
            )
            if _DEBUG_DIFF_COUNT[key_label] < NPU_FUSED_ATTN_DEBUG_N:
                diff_t = (fused_out.float() - eager_out.float()).abs()
                max_diff = diff_t.max().item()
                mean_diff = diff_t.mean().item()
                eager_max = eager_out.float().abs().max().item()
                rel = max_diff / max(eager_max, 1e-6)
                _DEBUG_DIFF_COUNT[key_label] += 1
                logger.info(
                    f"[NPU attn debug] {key_label} #{_DEBUG_DIFF_COUNT[key_label]} "
                    f"S_q={S_q} S_kv={S_kv} | "
                    f"max_diff={max_diff:.4f} mean_diff={mean_diff:.5f} "
                    f"eager_max_abs={eager_max:.3f} rel={rel:.2%}"
                )
            return eager_out, None
        except Exception as e:
            logger.warning(f"[NPU attn debug] 对比失败，仅用 eager: {e}")
            return _run_eager_attention(module, query, key, value, attention_mask, **kwargs)

    # ── 正常模式: fused 优先，失败 fallback eager ──
    try:
        return _npu_fused_attention_forward(module, query, key, value, attention_mask, **kwargs)
    except Exception as e:
        if not _NPU_ATTN_FAILED_ONCE["flag"]:
            logger.warning(
                f"[NPU attention] 调用失败，回退 eager: {type(e).__name__}: {e}"
            )
            _NPU_ATTN_FAILED_ONCE["flag"] = True
        # 兜底：使用 transformers 的 eager 实现
        try:
            from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
            return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
        except Exception:
            # 再兜底：手写 eager，注意输出 shape 必须是 [B, S, N, D]
            scaling = kwargs.get("scaling") or (1.0 / math.sqrt(query.shape[-1]))
            # GQA 时需要 repeat_kv 展开
            B, N_q, S_q, D = query.shape
            _, N_kv, S_kv, _ = key.shape
            if N_kv != N_q:
                rep = N_q // N_kv
                key = key.repeat_interleave(rep, dim=1)
                value = value.repeat_interleave(rep, dim=1)
            scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
            if attention_mask is not None:
                scores = scores + attention_mask
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            out = torch.matmul(probs, value)        # [B, N, S, D]
            out = out.transpose(1, 2).contiguous()  # [B, S, N, D] ← 关键
            return out, None


def _find_hf_transformer(root):
    """
    BFS 搜索 LLM 子树，找到带 config._attn_implementation 的 HF 模型主体。

    CosyVoice2 的 LLM 嵌套结构:
      Qwen2LM (无 config) → Qwen2Encoder (无 config) → Qwen2ForCausalLM (有 config)

    因此不能假设固定层数，需要遍历找到 HF transformer 本体。
    """
    if not isinstance(root, torch.nn.Module):
        return None
    for module in root.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            return module
    return None


def _patch_qwen2_attention(inner):
    """
    替换 transformers 注册表中的 sdpa attention 函数为 NPU 融合实现。

    ★ 关键策略（v4）★
    不再修改 config._attn_implementation。保持 "sdpa" 不变，让 transformers 的
    Qwen2Model._update_causal_mask 按 sdpa 路径准备 attention_mask（多数情况返回
    None，由 SDPA 内部 is_causal 处理）。

    然后直接替换 ALL_ATTENTION_FUNCTIONS["sdpa"] 的函数指针为我们的 NPU 实现。
    这样：
      1. mask 准备与基线一致 (sdpa 风格)
      2. dispatch 路径不变 (config._attn_implementation 还是 "sdpa")
      3. 真正的 attention 计算被劫持到 NPU 融合算子
      4. 失败时 fallback 到保存的原版 sdpa，与基线完全一致
    """
    global _ORIGINAL_SDPA_ATTN

    if not NPU_FUSED_ATTN:
        return

    try:
        import torch_npu
    except ImportError:
        logger.warning("[NPU优化8] torch_npu 未安装，跳过 fused attention")
        return

    needed = ["npu_fusion_attention"]
    missing = [n for n in needed if not hasattr(torch_npu, n)]
    if missing:
        logger.warning(f"[NPU优化8] torch_npu 缺失算子 {missing}，跳过")
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        logger.warning("[NPU优化8] transformers 不支持 ALL_ATTENTION_FUNCTIONS，跳过")
        return

    # ── 检查 sdpa 是否在注册表中 ──
    if "sdpa" not in ALL_ATTENTION_FUNCTIONS:
        logger.warning("[NPU优化8] ALL_ATTENTION_FUNCTIONS 中没有 sdpa，跳过")
        return

    # ── 验证 LLM 真的存在 ──
    llm = getattr(inner, "llm", None)
    if llm is None or not isinstance(llm, torch.nn.Module):
        logger.warning("[NPU优化8] inner.llm 不存在或不是 nn.Module，跳过")
        return

    target = _find_hf_transformer(llm)
    if target is None:
        logger.warning("[NPU优化8] 找不到 HF transformer，跳过")
        return

    cfg = target.config
    original_impl = getattr(cfg, "_attn_implementation", "eager")

    # ── v4: 自动探测 LLM 总层数（用于 skip_last_N 解析）──
    n_layers = (
        getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
        or 24  # Qwen2.5-0.5B 默认
    )
    _TOTAL_LAYER_COUNT["value"] = n_layers

    logger.info(
        f"[NPU优化8] 找到 HF 模型主体: {type(target).__name__} "
        f"(原 _attn_implementation={original_impl}, num_layers={n_layers})"
    )

    # ── ★ 保存原始 sdpa，然后替换 ──
    _ORIGINAL_SDPA_ATTN = ALL_ATTENTION_FUNCTIONS["sdpa"]
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _npu_fused_attention_with_fallback

    # ── 不修改 config._attn_implementation！保持 sdpa ──
    # cfg._attn_implementation 不动

    # ── 统计 attention 子模块（仅日志用，不修改）──
    attn_classes = {}
    for module in target.modules():
        cls_name = type(module).__name__
        if "Attention" in cls_name:
            attn_classes[cls_name] = attn_classes.get(cls_name, 0) + 1

    # ── v4: 预计算并展示哪些层会走 fused / eager ──
    if NPU_FUSED_ATTN_LAYERS not in ("all", ""):
        fused_layers = [i for i in range(n_layers)
                        if _parse_layer_spec(NPU_FUSED_ATTN_LAYERS, i, n_layers)]
        eager_layers = [i for i in range(n_layers) if i not in fused_layers]
        logger.info(
            f"[NPU优化8] 层选择策略 NPU_FUSED_ATTN_LAYERS='{NPU_FUSED_ATTN_LAYERS}': "
            f"fused={fused_layers} ({len(fused_layers)}层), "
            f"eager={eager_layers} ({len(eager_layers)}层)"
        )

    logger.info(
        f"[NPU优化8] 替换 ALL_ATTENTION_FUNCTIONS['sdpa'] 为 NPU 融合实现 "
        f"(_attn_implementation 保持 {original_impl}, "
        f"attention 子模块: {attn_classes}, "
        f"inner_precise={NPU_FUSED_ATTN_INNER_PRECISE}, "
        f"decode_sparse_mode={NPU_FUSED_ATTN_DECODE_SPARSE})"
    )


# ══════════════════════════════════════════════════════════════
#  优化入口
# ══════════════════════════════════════════════════════════════

def apply_npu_compile(cosyvoice, enable_flow=True, enable_hift=False):
    """
    CosyVoice NPU 推理全面优化入口（供 app_npu.py 调用）。

    Args:
        cosyvoice:    CosyVoice2 实例（cosyvoice_model）
        enable_flow:  是否对 flow.decoder.estimator 尝试 torch.compile
        enable_hift:  保留参数，始终 False（CPU istft patch 不兼容编译）

    Returns:
        cosyvoice (原实例，已在内存中 patch)
    """
    # ── 前置检查: NPU 必须真正可用 ──
    # 即使 app_npu.py 已判断 DEVICE_TYPE='npu'，也可能在 import 后
    # NPU 状态变化（如其他进程占用）。这里二次确认，避免 patch 失败级联。
    try:
        if not torch.npu.is_available():
            logger.warning("[NPU] torch.npu.is_available() == False，跳过所有 NPU 优化")
            return cosyvoice
        # 强制触发 lazy_init，验证 device 真能用
        torch.npu.device_count()
    except Exception as e:
        logger.warning(f"[NPU] NPU 设备探测失败，跳过所有优化: {e}")
        return cosyvoice

    inner = getattr(cosyvoice, "model", None)
    if inner is None:
        logger.warning("[NPU] cosyvoice.model 不存在，跳过所有优化")
        return cosyvoice

    flow = getattr(inner, "flow", None)

    # ── v5 提示: 当前走 fallback 路径 ──
    logger.warning(
        "[NPU优化] ⚠ 你正在走 fallback 路径 (LOAD_VLLM=false)。"
        "推荐切换到 vLLM-Ascend (LOAD_VLLM=true) 获得无早停 bug 的 LLM 加速。"
    )
    if NPU_FUSED_ATTN:
        logger.warning(
            "[NPU优化] ⚠⚠ NPU_FUSED_ATTN=true 实测有早停 bug "
            "(speech_len 仅为正常的 24-50%), 仅在已知接受质量损失时启用。"
        )

    logger.info(
        f"[NPU优化] 配置 | "
        f"timesteps={NPU_FLOW_TIMESTEPS} | "
        f"disable_cfg={NPU_DISABLE_CFG} | "
        f"profile={NPU_PROFILE} | "
        f"compile={NPU_COMPILE_ENABLED} | "
        f"compile_llm={NPU_COMPILE_LLM} | "
        f"force_llm_fp16={NPU_FORCE_LLM_FP16} | "
        f"fused_attn={NPU_FUSED_ATTN} | "
        f"fused_attn_mode={NPU_FUSED_ATTN_MODE} | "
        f"fused_attn_layers='{NPU_FUSED_ATTN_LAYERS}' | "
        f"inner_precise={NPU_FUSED_ATTN_INNER_PRECISE} | "
        f"decode_sparse={NPU_FUSED_ATTN_DECODE_SPARSE} | "
        f"fused_attn_debug={NPU_FUSED_ATTN_DEBUG}"
    )

    # ── 优化 1: Flow Matching 步数缩减 ──
    if flow is not None:
        _patch_flow_decoder_timesteps(flow)

    # ── 优化 2: CFG-free (batch 2→1) ──
    if NPU_DISABLE_CFG and flow is not None:
        _patch_solve_euler_cfg_free(flow)

    # ── 优化 3: NPU Stream 修复 ──
    _fix_npu_stream_context(inner)

    # ── 优化 6: LLM dtype 诊断（无副作用，总是执行）──
    _diagnose_llm_dtype(inner)

    # ── 优化 6b: 强制 LLM FP16（实测后启用）──
    if NPU_FORCE_LLM_FP16:
        _force_llm_half(inner)
        _diagnose_llm_dtype(inner)  # 再诊断一次确认

    # ── 优化 4: profiling 诊断 ──
    if NPU_PROFILE:
        _enable_profiling(inner)

    # ── 优化 5: torch.compile flow.estimator (实验性) ──
    if enable_flow and flow is not None:
        _try_compile_estimator(flow)

    # ── 优化 7: torch.compile LLM ──
    if NPU_COMPILE_LLM:
        _try_compile_llm(inner)

    # ── 优化 8: NPU Fused Attention (★★★ 真正的主战场)──
    if NPU_FUSED_ATTN:
        _patch_qwen2_attention(inner)

    logger.info("[NPU优化] 所有优化已应用，模型就绪")
    return cosyvoice
