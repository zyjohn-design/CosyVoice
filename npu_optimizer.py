# npu_optimizer.py
"""
CosyVoice NPU 910B 推理优化套件 v3
===================================

通过 monkey-patch 实现优化，不修改主项目代码。
在 app_npu.py 模型加载后调用 apply_npu_compile() 即可。

优化策略 (按收益排序):
──────────────────────────────────────────────────────────────
[1] Flow Matching n_timesteps: 10 → 4                  ★★★
    - Euler ODE solver 步数减半，Flow 时间砍 60%
    - NPU_FLOW_TIMESTEPS=4 (默认)

[2] CFG-free solve_euler: batch 2 → 1                  ★★★
    - 替换 solve_euler 跳过 Classifier-Free Guidance 分支
    - 每步 estimator forward 从 batch=2 降到 batch=1
    - Flow estimator 计算量再减 ~45%
    - NPU_DISABLE_CFG=true (默认)
    - 质量影响: 声音克隆相似度略降，中文 TTS 几乎无感

[3] NPU Stream 上下文修复                              ★
    - 修复 llm_context 退化为 nullcontext() 的问题

[4] 模块级 profiling 诊断                              📊
    - 包装 llm/flow/hift 的 inference 方法做精确耗时统计
    - NPU_PROFILE=true 启用，默认关闭（避免 synchronize 开销）
    - 每次推理 INFO 级日志；可调用 dump_profile() 拿汇总

[5] torch.compile 选择性编译 (实验性)                  ⚙
    - NPU_COMPILE=false (默认关闭)

环境变量:
    NPU_FLOW_TIMESTEPS=4    Flow ODE 步数 (3-6)
    NPU_DISABLE_CFG=true    禁用 CFG，batch=1 推理
    NPU_PROFILE=false       启用 NPU 同步耗时诊断
    NPU_COMPILE=false       启用 torch.compile (实验)

预期 RTF (基线 1.0):
    + n_timesteps=4         → ~0.70
    + CFG-free              → ~0.50
    + 两者叠加               → ~0.40-0.50  ✓ 满足 0.5-0.6 目标
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


def _log_mode_skip(label):
    """模式跳过 fused 时打首次日志，证明 mode 真生效了。"""
    if label not in _MODE_SKIP_LOGGED:
        logger.info(f"[NPU attention] {label} 走 eager (mode={NPU_FUSED_ATTN_MODE})")
        _MODE_SKIP_LOGGED[label] = True


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

    # ── 改用 BNSD 布局 ──
    # 直接用 transformers 给的 [B, N, S, D]，不做 BSH 转换，减少 view/transpose 出错
    q_in = query.contiguous()
    k_in = key.contiguous()
    v_in = value.contiguous()

    if scaling is None:
        scaling = 1.0 / math.sqrt(D)

    if S_q == 1:
        # ── 解码路径（每 token 调用一次，主热点）──
        # ★ 新增 actual_seq_lengths：显式告知有效 KV 长度，避免算子假设 padding
        out = torch_npu.npu_incre_flash_attention(
            q_in, k_in, v_in,
            num_heads=N_q,
            input_layout="BNSD",
            scale_value=float(scaling),
            num_key_value_heads=0,
            actual_seq_lengths=[S_kv] * B,
        )
    else:
        # ── Prefill 路径 ──
        # 显式 causal mask + actual_seq_lengths
        mask = torch.triu(
            torch.ones((S_q, S_kv), dtype=torch.bool, device=query.device),
            diagonal=S_kv - S_q + 1,
        )
        out = torch_npu.npu_prompt_flash_attention(
            q_in, k_in, v_in,
            num_heads=N_q,
            input_layout="BNSD",
            scale_value=float(scaling),
            num_key_value_heads=0,
            atten_mask=mask,
            sparse_mode=0,
            actual_seq_lengths=[S_q] * B,
            actual_seq_lengths_kv=[S_kv] * B,
        )

    # BNSD 输出 [B, N, S, D] → [B, S, N, D]（transformers 期望）
    out = out.transpose(1, 2).contiguous()

    # 还原 caller 期望的 dtype（与 query 一致）
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)

    return out, None


def _npu_fused_attention_with_fallback(module, query, key, value, attention_mask, **kwargs):
    """带兜底的 NPU attention：单 token 失败自动退回 eager，避免推理崩溃。"""
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
    """注册 NPU fused attention 并将 LLM 切换到这个实现。"""
    if not NPU_FUSED_ATTN:
        return

    try:
        import torch_npu
    except ImportError:
        logger.warning("[NPU优化8] torch_npu 未安装，跳过 fused attention")
        return

    # 检查算子可用性
    needed = ["npu_incre_flash_attention", "npu_prompt_flash_attention"]
    missing = [n for n in needed if not hasattr(torch_npu, n)]
    if missing:
        logger.warning(f"[NPU优化8] torch_npu 缺失算子 {missing}，跳过")
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        logger.warning("[NPU优化8] transformers 版本不支持 ALL_ATTENTION_FUNCTIONS")
        return

    # 注册自定义 impl
    ALL_ATTENTION_FUNCTIONS["npu_fused"] = _npu_fused_attention_with_fallback

    # 找到 LLM 根节点
    llm = getattr(inner, "llm", None)
    if llm is None or not isinstance(llm, torch.nn.Module):
        logger.warning("[NPU优化8] inner.llm 不存在或不是 nn.Module，跳过")
        return

    # ── 找到真正的 HF transformer ──
    target = _find_hf_transformer(llm)
    if target is None:
        logger.warning(
            "[NPU优化8] 在 LLM 子树中找不到 HF transformer "
            "(无 config._attn_implementation)，跳过"
        )
        return

    logger.info(f"[NPU优化8] 找到 HF 模型主体: {type(target).__name__}")

    # ── 切换 attention 实现 ──
    cfg = target.config
    original_impl = getattr(cfg, "_attn_implementation", "eager")
    cfg._attn_implementation = "npu_fused"

    # ── 统计 + 设置所有 Attention 子模块的 _attn_implementation ──
    # transformers 4.45+ 在 Attention 模块上也维护一个 _attn_implementation 属性
    attn_classes = {}
    switched = 0
    for module in target.modules():
        cls_name = type(module).__name__
        if "Attention" in cls_name:
            attn_classes[cls_name] = attn_classes.get(cls_name, 0) + 1
            if hasattr(module, "_attn_implementation"):
                module._attn_implementation = "npu_fused"
                switched += 1
            # 同时更新模块自己持有的 config 引用（防御性）
            mod_cfg = getattr(module, "config", None)
            if mod_cfg is not None and mod_cfg is not cfg:
                mod_cfg._attn_implementation = "npu_fused"

    logger.info(
        f"[NPU优化8] Qwen2Attention 切换到 npu_fused "
        f"({original_impl} → npu_fused, attention 子模块: {attn_classes}, "
        f"_attn_implementation 已设置: {switched})"
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

    logger.info(
        f"[NPU优化] 配置 | "
        f"timesteps={NPU_FLOW_TIMESTEPS} | "
        f"disable_cfg={NPU_DISABLE_CFG} | "
        f"profile={NPU_PROFILE} | "
        f"compile={NPU_COMPILE_ENABLED} | "
        f"compile_llm={NPU_COMPILE_LLM} | "
        f"force_llm_fp16={NPU_FORCE_LLM_FP16} | "
        f"fused_attn={NPU_FUSED_ATTN} | "
        f"fused_attn_mode={NPU_FUSED_ATTN_MODE}"
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
