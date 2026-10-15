#!/usr/bin/env python3
"""使用实际 wetext 运行时验证 FST 文件、标注类型和文本读法，不加载 TTS。"""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=Path.home()/'.cache/modelscope/hub/pengzhendong/wetext')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    lock = json.loads((ROOT/'upstream.lock.json').read_text())
    if version('wetext') != lock['runtime_wetext_version']:
        parser.error(f'本发布版本针对 wetext=={lock["runtime_wetext_version"]} 验证')
    model_dir = args.model_dir.resolve()
    manifest = model_dir/'SHA256SUMS'
    if not manifest.is_file():
        parser.error(f'缺少发布校验清单：{manifest}')
    for line in manifest.read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (model_dir/relative).resolve()
        if not path.is_relative_to(model_dir):
            raise ValueError(relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'文件校验失败：{path}')
    # 规则测试只通过 wetext 的公共 API 执行，不导入任何 CosyVoice 预处理代码。
    from wetext import Normalizer
    cases = json.loads((ROOT/'tests.json').read_text())
    results = []
    for verbalizer in ('verbalizer.fst', 'verbalizer_remove_erhua.fst'):
        normalizer = Normalizer(tagger_path=str(model_dir/'zh/tn/tagger.fst'),
                                verbalizer_path=str(model_dir/'zh/tn'/verbalizer), lang='zh')
        for case in cases:
            tagged = normalizer.tag(case['input'])
            actual = normalizer.normalize(case['input'])
            errors = []
            if 'expected' in case and actual != case['expected']:
                errors.append('文本读法不匹配')
            for token_type in case.get('require_types', []):
                if token_type + ' {' not in tagged:
                    errors.append(f'缺少 {token_type} 标注')
            for token_type in case.get('exclude_types', []):
                if token_type + ' {' in tagged:
                    errors.append(f'不应匹配 {token_type}')
            results.append({**case, 'verbalizer': verbalizer, 'tagged': tagged,
                            'actual': actual, 'passed': not errors, 'errors': errors})
    passed = sum(result['passed'] for result in results)
    report = {'runtime': version('wetext'), 'python': sys.executable,
              'model_dir': str(model_dir), 'passed': passed,
              'total': len(results), 'results': results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for result in results:
        if not result['passed']:
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(f'wetext {report["runtime"]}: {passed}/{len(results)} 项检查通过。')
    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
