#!/usr/bin/env python3
"""
CosyVoice TTS 接口性能测试
==========================
- 10 条不同长度文本 (短句 → 长段落)
- 说话人: 10001-男声.pt
- 记录: 文本长度、推理耗时、音频时长、RTF (实时率)
- 输出: 控制台表格 + CSV 文件 + 音频文件保存

用法:
    python test_tts.py
    python test_tts.py --host 10.16.1.190 --port 8080
"""

import os
import sys
import re
import time
import json
import argparse
import requests
from datetime import datetime

# ═══════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
SPEAKER = "1111-男声"       # 说话人 (不带 .pt 后缀)
OUTPUT_DIR = "test_output"   # 音频输出目录

# ═══════════════════════════════════════════════════
#  10 条测试文本 (从短到长)
# ═══════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": 1,
        "desc": "5字",
        "text": "你好，世界啊！",
    },
    {
        "id": 2,
        "desc": "10字",
        "text": "今天天气真好，适合出门走走。",
    },
    {
        "id": 3,
        "desc": "15字",
        "text": "武汉是一座历史悠久的城市，长江穿城而过。",
    },
    {
        "id": 4,
        "desc": "20字",
        "text": "随着人工智能技术的飞速发展，语音合成已经能够生成自然语音。",
    },
    {
        "id": 5,
        "desc": "25字",
        "text": "二零二六年武汉马拉松将于四月举行，届时数万名跑者将齐聚江城，共同奔跑。",
    },
    {
        "id": 6,
        "desc": "30字",
        "text": "各位市民朋友大家好！为保障马拉松赛事顺利进行，交管部门将对部分道路实施临时交通管制。",
    },
    {
        "id": 7,
        "desc": "35字",
        "text": "智慧城市建设是推动城市治理现代化的重要抓手，武汉市积极探索人工智能和大数据在城市管理中的创新应用。",
    },
    {
        "id": 8,
        "desc": "40字",
        "text": "在光电子信息产业方面，中国光谷已经发展成为全球最大的光纤光缆生产基地，光电子器件占全国市场份额的百分之六十以上。",
    },
    {
        "id": 9,
        "desc": "45字",
        "text": "尊敬的各位领导、各位来宾，大家上午好！欢迎参加武汉市智慧交通管理系统发布会，今天我们正式发布基于人工智能技术的新一代城市交通平台。",
    },
    {
        "id": 10,
        "desc": "50字",
        "text": "经过半年的试运行，该系统在早晚高峰时段的道路通行效率提升了百分之十五，交通事故响应时间缩短了百分之三十，市民出行体验得到了显著改善。",
    },
]

def count_chars(text: str) -> int:
    """计算纯字数（不含标点符号）"""
    return len(re.sub(r'[，。！？、；：""''（）【】《》…—\s,.!?;:\'\"()\[\]{}<>]', '', text))



def run_test(base_url: str, case: dict, output_dir: str) -> dict:
    """执行单条测试，返回结果 dict"""

    text = case["text"]
    text_len = count_chars(text)

    # 构造请求参数
    payload = {
        "text": text,
        "voice": SPEAKER,
        "speed": 5,
        "wav": 0,         # 返回 JSON (含 inference_time, audio_duration)
    }

    url = f"{base_url}/tts"

    # 客户端计时 (含网络开销)
    t_start = time.time()
    try:
        resp = requests.post(url, data=payload, timeout=120)
        t_end = time.time()
    except Exception as e:
        return {
            "id": case["id"],
            "desc": case["desc"],
            "text_len": text_len,
            "status": "ERROR",
            "error": str(e),
        }

    client_time = round(t_end - t_start, 3)

    if resp.status_code != 200:
        return {
            "id": case["id"],
            "desc": case["desc"],
            "text_len": text_len,
            "status": f"HTTP {resp.status_code}",
            "error": resp.text[:200],
        }

    try:
        data = resp.json()
    except Exception:
        return {
            "id": case["id"],
            "desc": case["desc"],
            "text_len": text_len,
            "status": "JSON_ERROR",
            "error": resp.text[:200],
        }

    if data.get("code") != 0:
        return {
            "id": case["id"],
            "desc": case["desc"],
            "text_len": text_len,
            "status": "API_ERROR",
            "error": data.get("msg", ""),
        }

    # 提取服务端指标
    af = data["audio_files"][0] if data.get("audio_files") else {}
    inference_time = af.get("inference_time", 0)
    audio_duration = af.get("audio_duration", 0)
    audio_url = af.get("url", "")

    # RTF = 推理耗时 / 音频时长 (越小越好, <1 表示实时)
    rtf = round(inference_time / audio_duration, 3) if audio_duration > 0 else -1

    # 每秒合成字数
    chars_per_sec = round(text_len / inference_time, 1) if inference_time > 0 else -1

    # 下载音频保存
    wav_file = ""
    if audio_url:
        try:
            wav_resp = requests.get(audio_url, timeout=30)
            if wav_resp.status_code == 200:
                wav_file = os.path.join(output_dir, f"test_{case['id']:02d}_{text_len}chars.wav")
                with open(wav_file, "wb") as f:
                    f.write(wav_resp.content)
        except Exception:
            pass

    return {
        "id": case["id"],
        "desc": case["desc"],
        "text_len": text_len,
        "status": "OK",
        "inference_time": inference_time,
        "audio_duration": audio_duration,
        "client_time": client_time,
        "rtf": rtf,
        "chars_per_sec": chars_per_sec,
        "wav_file": wav_file,
    }


def print_table(results: list):
    """打印结果表格"""

    # 表头
    header = (
        f"{'#':>3}  {'描述':<16}  {'字数':>4}  {'推理(s)':>8}  "
        f"{'音频(s)':>8}  {'客户端(s)':>9}  {'RTF':>6}  {'字/秒':>6}  {'状态':<8}"
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(f"  CosyVoice TTS 性能测试报告")
    print(f"  说话人: {SPEAKER}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{sep}")
    print(header)
    print(sep)

    total_infer = 0
    total_audio = 0
    total_chars = 0
    ok_count = 0

    for r in results:
        if r["status"] == "OK":
            ok_count += 1
            total_infer += r["inference_time"]
            total_audio += r["audio_duration"]
            total_chars += r["text_len"]
            print(
                f"{r['id']:>3}  {r['desc']:<16}  {r['text_len']:>4}  "
                f"{r['inference_time']:>8.2f}  {r['audio_duration']:>8.2f}  "
                f"{r['client_time']:>9.2f}  {r['rtf']:>6.3f}  "
                f"{r['chars_per_sec']:>6.1f}  {r['status']:<8}"
            )
        else:
            print(
                f"{r['id']:>3}  {r['desc']:<16}  {r['text_len']:>4}  "
                f"{'--':>8}  {'--':>8}  {'--':>9}  {'--':>6}  "
                f"{'--':>6}  {r['status']:<8}  {r.get('error', '')[:40]}"
            )

    print(sep)

    if ok_count > 0:
        avg_rtf = round(total_infer / total_audio, 3) if total_audio > 0 else -1
        avg_cps = round(total_chars / total_infer, 1) if total_infer > 0 else -1
        print(
            f"{'合计':>3}  {'':16}  {total_chars:>4}  "
            f"{total_infer:>8.2f}  {total_audio:>8.2f}  "
            f"{'':>9}  {avg_rtf:>6.3f}  {avg_cps:>6.1f}  "
            f"{ok_count}/{len(results)} OK"
        )
    print(sep)
    print()


def save_csv(results: list, filepath: str):
    """保存为 CSV"""
    import csv
    fields = [
        "id", "desc", "text_len", "status",
        "inference_time", "audio_duration", "client_time",
        "rtf", "chars_per_sec", "wav_file",
    ]
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"CSV 报告已保存: {filepath}")


def save_json(results: list, filepath: str):
    """保存为 JSON"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "speaker": SPEAKER,
            "test_time": datetime.now().isoformat(),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告已保存: {filepath}")


# ═══════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CosyVoice TTS 性能测试")
    parser.add_argument("--host", default=DEFAULT_HOST, help="API 地址")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="API 端口")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # 检查服务是否可用
    print(f"测试目标: {base_url}")
    print(f"说话人: {SPEAKER}")
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        h = health.json()
        print(f"服务状态: {h.get('status', 'unknown')}")
        print(f"模型: {h.get('model_dir', 'unknown')}")
        print(f"采样率: {h.get('sample_rate', 'unknown')}")
    except Exception as e:
        print(f"⚠ 无法连接服务: {e}")
        print(f"  请确认服务已启动: python app.py")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 执行测试
    print(f"\n开始测试 ({len(TEST_CASES)} 条) ...\n")
    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {case['desc']} ({len(case['text'])}字) ...", end=" ", flush=True)
        r = run_test(base_url, case, OUTPUT_DIR)
        if r["status"] == "OK":
            print(f"✓ 推理 {r['inference_time']}s / 音频 {r['audio_duration']}s / RTF {r['rtf']}")
        else:
            print(f"✗ {r['status']}: {r.get('error', '')[:60]}")
        results.append(r)

    # 输出报告
    print_table(results)

    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_csv(results, os.path.join(OUTPUT_DIR, f"tts_benchmark_{timestamp}.csv"))
    save_json(results, os.path.join(OUTPUT_DIR, f"tts_benchmark_{timestamp}.json"))

    print(f"音频文件保存在: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == "__main__":
    main()