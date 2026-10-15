#!/usr/bin/env python3
"""使用实际 wetext 运行时验证 FST 文件、标注类型和文本读法，不加载 TTS。"""
import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=Path.home()/'.cache/modelscope/hub/pengzhendong/wetext')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--check-frontend', action='store_true',
                        help='额外调用实际 CosyVoice 文本前端验证语言分支与分句，不加载权重')
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
        normalizers = {
            'zh': Normalizer(tagger_path=str(model_dir/'zh/tn/tagger.fst'),
                             verbalizer_path=str(model_dir/'zh/tn'/verbalizer), lang='zh'),
            'en': Normalizer(tagger_path=str(model_dir/'en/tn/tagger.fst'),
                             verbalizer_path=str(model_dir/'en/tn/verbalizer.fst'), lang='en'),
        }
        # 只替换测试资源定位，仍由实际 wetext 的 auto 逻辑选择语言；避免联网。
        with patch('wetext.wetext.snapshot_download', return_value=str(model_dir)):
            normalizers['auto'] = Normalizer(lang='auto', remove_erhua=verbalizer != 'verbalizer.fst')
        for case in cases:
            lang = case.get('lang', 'zh')
            normalizer = normalizers[lang]
            selected_lang = ('zh' if normalizer.contains_chinese(case['input']) else 'en') if lang == 'auto' else lang
            tagged = normalizer.tag(case['input'], lang=selected_lang)
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
            results.append({**case, 'verbalizer': verbalizer, 'selected_lang': selected_lang, 'tagged': tagged,
                            'actual': actual, 'passed': not errors, 'errors': errors})
    if args.check_frontend:
        from types import SimpleNamespace
        import inflect
        if (ROOT.parent/'cosyvoice/cli/frontend.py').is_file():
            sys.path.insert(0, str(ROOT.parent))
        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        frontend = CosyVoiceFrontEnd.__new__(CosyVoiceFrontEnd)
        frontend.text_frontend = 'wetext'
        frontend.zh_tn_model = frontend.en_tn_model = normalizers['auto']
        frontend.inflect_parser = inflect.engine()
        frontend.allowed_special = 'all'
        frontend.tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))
        for case in cases:
            if case.get('lang') != 'auto' or 'expected' not in case:
                continue
            modes = (False, True) if 'zh_number' in case.get('require_types', []) else (False,)
            for split in modes:
                actual = frontend.text_normalize(case['input'], split=split)
                if split:
                    actual = ''.join(actual)
                matches = actual == case['expected']
                results.append({**case, 'stage': 'CosyVoiceFrontEnd', 'split': split,
                                'actual': actual, 'passed': matches,
                                'errors': [] if matches else ['实际前端分支输出不匹配']})
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
