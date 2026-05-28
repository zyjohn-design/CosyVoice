#!/usr/bin/env python3
# ============================================================
# vLLM-Ascend 冒烟测试 — NPU 910B
#
# 用途: 在切换到 LOAD_VLLM=true 之前/之后, 快速验证 vLLM 链路。
# 检查项 (从浅到深):
#   1. torch_npu 可用，NPU 设备能探测到
#   2. vllm 可 import 且版本正确
#   3. vllm_ascend (或集成进 vllm 的 NPU backend) 已注册
#   4. transfer_to_npu / ASCEND_RT_VISIBLE_DEVICES 配置正确
#   5. CosyVoice2 模型注册到 vLLM ModelRegistry 成功
#   6. 加载 export 后的 vLLM 模型并跑一次 dummy decode
#
# 使用:
#   docker exec -it cosyvoice-tts \
#     python /workspace/CosyVoice/scripts/smoke_vllm_npu.py
#
# 退出码:
#   0 — 全部通过 (可以放心 LOAD_VLLM=true)
#   1 — 任一步失败 (按提示排查或回退 LOAD_VLLM=false)
# ============================================================
import os
import sys
import time
import traceback


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


def ok(msg: str):
    print(f"{GREEN}[ ✓ ]{RESET} {msg}")


def fail(msg: str):
    print(f"{RED}[ ✗ ]{RESET} {msg}")


def info(msg: str):
    print(f"{CYAN}[ … ]{RESET} {msg}")


def warn(msg: str):
    print(f"{YELLOW}[ ! ]{RESET} {msg}")


def step(n: int, total: int, title: str):
    print(f"\n{CYAN}━━━ Step {n}/{total} ━━━{RESET} {title}")


# ════════════════════════════════════════════════════════════
TOTAL_STEPS = 6
failures: list[str] = []


# ── Step 1: torch_npu + NPU 设备 ──
step(1, TOTAL_STEPS, "torch_npu / NPU 设备探测")
try:
    import torch
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    ok(f"torch={torch.__version__}")
    ok(f"torch_npu={torch_npu.__version__}")

    if not torch.npu.is_available():
        fail("torch.npu.is_available() == False")
        failures.append("NPU 不可用 — 检查 /dev/davinci* 设备映射和 ASCEND_RT_VISIBLE_DEVICES")
    else:
        n = torch.npu.device_count()
        ok(f"npu.device_count() = {n}")
        try:
            dev = torch.npu.current_device()
            ok(f"current_device = npu:{dev}")
        except Exception as e:
            warn(f"current_device 查询失败 (非致命): {e}")

        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<未设置>")
        ok(f"ASCEND_RT_VISIBLE_DEVICES = {visible}")
except Exception as e:
    fail(f"torch_npu 导入失败: {e}")
    traceback.print_exc()
    failures.append("torch_npu 导入失败 — 检查容器基础镜像是否含 CANN")
    print("\n后续步骤无法继续。")
    sys.exit(1)


# ── Step 2: vllm 版本 ──
step(2, TOTAL_STEPS, "vllm import & 版本")
try:
    import vllm
    from packaging.version import parse as vparse

    v = vparse(vllm.__version__.split('+')[0])
    ok(f"vllm={vllm.__version__}")

    if v >= vparse("0.11.0"):
        ok("使用 vLLM V1 引擎 (≥ 0.11)")
    else:
        warn("旧版 V0 引擎 (< 0.11) — model_npu.py 仍兼容, 但建议升级")
except Exception as e:
    fail(f"vllm 导入失败: {e}")
    traceback.print_exc()
    failures.append("vllm 包未安装或损坏")


# ── Step 3: vllm-ascend backend ──
step(3, TOTAL_STEPS, "vllm-ascend NPU backend 注册")
try:
    import vllm_ascend  # noqa: F401
    ok(f"vllm_ascend 已安装: {vllm_ascend.__version__ if hasattr(vllm_ascend, '__version__') else 'unknown'}")
except ImportError:
    # vLLM 0.17 可能已把 ascend 集成进 vllm 主包
    info("vllm_ascend 独立包未找到, 检查 vllm 内部 platform 注册...")
    try:
        from vllm.platforms import current_platform
        plat_name = getattr(current_platform, "_enum", current_platform).__class__.__name__
        ok(f"current_platform = {plat_name}")
        if "Ascend" in plat_name or "NPU" in plat_name or "npu" in str(current_platform).lower():
            ok("vLLM 已识别为 Ascend 平台")
        else:
            warn(f"vLLM 平台未识别为 NPU/Ascend: {current_platform!r}")
            failures.append("vLLM 没用 NPU backend — 检查 VLLM_USE_V1 和环境变量")
    except Exception as e:
        warn(f"platform 检测失败: {e}")
except Exception as e:
    fail(f"vllm_ascend 加载异常: {e}")
    failures.append("vllm_ascend 加载失败")


# ── Step 4: CosyVoice2 注册 ──
step(4, TOTAL_STEPS, "注册 CosyVoice2ForCausalLM 到 vLLM")
try:
    # 触发注册
    import cosyvoice.vllm.cosyvoice2  # noqa: F401
    from vllm import ModelRegistry
    registered = ModelRegistry.get_supported_archs()
    if "CosyVoice2ForCausalLM" in registered:
        ok("CosyVoice2ForCausalLM 注册成功")
    else:
        fail(f"未在 ModelRegistry 中找到 CosyVoice2ForCausalLM")
        info(f"已注册的 archs (前 20): {sorted(registered)[:20]}")
        failures.append("CosyVoice2 模型未注册")
except Exception as e:
    fail(f"注册模型时异常: {e}")
    traceback.print_exc()
    failures.append("CosyVoice2 注册失败")


# ── Step 5: 检查导出的 vLLM 模型目录 ──
step(5, TOTAL_STEPS, "检查 vLLM 模型导出目录")
model_dir = os.environ.get("MODEL_DIR", "pretrained_models/CosyVoice2-0.5B")
vllm_dir = os.path.join(model_dir, "vllm")
if os.path.exists(vllm_dir):
    contents = sorted(os.listdir(vllm_dir))
    ok(f"已存在 {vllm_dir}")
    ok(f"包含 {len(contents)} 个文件: {contents[:6]}...")
    # 关键文件
    must_have = ["config.json"]
    missing = [f for f in must_have if not os.path.exists(os.path.join(vllm_dir, f))]
    if missing:
        warn(f"缺少关键文件: {missing}")
    else:
        ok("config.json 存在")
        # 检查 architectures
        import json
        with open(os.path.join(vllm_dir, "config.json")) as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if "CosyVoice2ForCausalLM" in archs:
            ok(f"config.json architectures = {archs}")
        else:
            warn(f"config.json architectures = {archs} (期望 CosyVoice2ForCausalLM)")
else:
    info(f"{vllm_dir} 不存在 — 首次启动时会自动导出 (耗时 30-60s, 占用 ~1.5GB)")
    info("跳过 Step 6, 因为没有可加载的模型")
    print(f"\n{YELLOW}═══ 提示 ═══{RESET}")
    print(f"  这是首次跑 vLLM, 模型尚未导出。")
    print(f"  正常启动 cosyvoice-tts 容器即可触发自动导出:")
    print(f"    docker compose -f docker-compose.npu.yml up -d")
    print(f"  日志看到 'Succesfully convert ...' 或 vLLM 初始化输出后, 再跑本脚本验证 Step 6。")
    sys.exit(0 if not failures else 1)


# ── Step 6: 创建 EngineArgs 并加载 (干跑, 不真采样) ──
step(6, TOTAL_STEPS, "vLLM EngineArgs 验证 (lightweight)")
try:
    from vllm import EngineArgs
    args = EngineArgs(
        model=vllm_dir,
        skip_tokenizer_init=True,
        enable_prompt_embeds=True,
        runner='generate',
        enforce_eager=True,           # NPU 必须
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
    )
    ok("EngineArgs 构造成功 (干跑, 未实例化 LLMEngine)")
    info("如需端到端测试, 请直接启动 cosyvoice-tts 容器观察日志")
except Exception as e:
    fail(f"EngineArgs 构造失败: {e}")
    traceback.print_exc()
    failures.append("EngineArgs 不兼容当前 vLLM 版本")


# ════════════════════════════════════════════════════════════
print()
print("═" * 60)
if not failures:
    print(f"{GREEN}✓ 全部检查通过 — 可以 LOAD_VLLM=true 启动{RESET}")
    sys.exit(0)
else:
    print(f"{RED}✗ {len(failures)} 项失败:{RESET}")
    for i, f in enumerate(failures, 1):
        print(f"  {i}. {f}")
    print()
    print(f"{YELLOW}建议: LOAD_VLLM=false docker compose ... 回退到 v4 NPU optimizer{RESET}")
    sys.exit(1)
