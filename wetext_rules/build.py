#!/usr/bin/env python3
"""将本地业务规则编译为 wetext 0.0.4 可加载的 FST 目录。"""
import argparse
import hashlib
import json
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-dir', type=Path, default=ROOT/'_upstream')
    parser.add_argument('--base-model-dir', type=Path,
                        default=Path.home()/'.cache/modelscope/hub/pengzhendong/wetext')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT/'upstream.lock.json').read_text())
    upstream = args.upstream_dir.resolve()
    marker = upstream/'UPSTREAM_COMMIT'
    if not marker.is_file() or marker.read_text().strip() != lock['commit']:
        parser.error('请先用 fetch_upstream.py 获取固定版本源码')
    if version('pynini') != lock['pynini_version']:
        parser.error(f'编译使用 pynini=={lock["pynini_version"]}')
    output = args.output.resolve()
    base = args.base_model_dir.resolve()
    if output.exists() or output.is_relative_to(base):
        parser.error('输出必须是独立的新目录，不能覆盖现用模型或在其中生成')
    for relative in ('zh/tn/tagger.fst', 'zh/tn/verbalizer.fst',
                     'en/tn/tagger.fst', 'en/tn/verbalizer.fst'):
        if not (base/relative).is_file():
            parser.error(f'基础模型缺少文件：{base/relative}')
    sys.path.insert(0, str(upstream))
    from normalizer import BusinessNormalizer

    models = []
    for remove_erhua in (False, True):
        print(f'编译中文 TN，remove_erhua={remove_erhua}', flush=True)
        models.append(BusinessNormalizer(ROOT/'data', remove_erhua=remove_erhua))
    # 所有图先编译成功，再创建新的发布目录。
    shutil.copytree(base, output,
                    ignore=shutil.ignore_patterns('.*', '__pycache__', 'SHA256SUMS', 'custom_rules'))
    normal, without_erhua = models
    normal.tagger.optimize().write(str(output/'zh/tn/tagger.fst'))
    normal.verbalizer.optimize().write(str(output/'zh/tn/verbalizer.fst'))
    without_erhua.verbalizer.optimize().write(str(output/'zh/tn/verbalizer_remove_erhua.fst'))
    # 将来源、配置、构建版本一同发布，便于以后复现。
    sources = [ROOT/name for name in ('rules.py', 'normalizer.py', 'build.py', 'fetch_upstream.py',
                                     'verify.py', 'tests.json', 'upstream.lock.json',
                                     'requirements-build.txt')]
    sources += sorted((ROOT/'data').glob('*.tsv'))
    source_dir = output/'custom_rules'
    for path in sources:
        target = source_dir/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(upstream/'LICENSE', source_dir/'UPSTREAM_LICENSE')
    metadata = {'upstream': lock, 'pynini': version('pynini'),
                'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                'upstream_source_sha256': {
                    str(p.relative_to(upstream)): digest(p)
                    for p in sorted((upstream/'tn').rglob('*'))
                    if p.is_file() and p.suffix in ('.py', '.tsv', '.far')},
                'base_zh_sha256': {name: digest(base/'zh/tn'/name)
                                  for name in ('tagger.fst', 'verbalizer.fst')}}
    (output/'custom_rules/build.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    files = sorted(p for p in output.rglob('*') if p.is_file())
    (output/'SHA256SUMS').write_text(
        ''.join(f'{digest(p)}  {p.relative_to(output).as_posix()}\n' for p in files), encoding='utf-8')
    print(f'已导出：{output}\n发布前请运行 verify.py 验证真实 wetext 运行时。', flush=True)


if __name__ == '__main__':
    main()
