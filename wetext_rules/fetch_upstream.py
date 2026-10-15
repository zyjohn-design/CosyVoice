#!/usr/bin/env python3
"""下载并验证固定版本的规则源码，不修改 CosyVoice 或现用模型缓存。"""
import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'_upstream')
    args = parser.parse_args()
    lock = json.loads((ROOT/'upstream.lock.json').read_text())
    destination = args.output.resolve()
    if destination.exists():
        parser.error(f'输出目录已存在，请使用新目录：{destination}')
    url = f'https://codeload.github.com/wenet-e2e/WeTextProcessing/tar.gz/{lock["commit"]}'
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != lock['archive_sha256']:
        raise ValueError('上游源码归档校验失败')
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        links = []
        for member in archive.getmembers():
            original = Path(member.name)
            if original.is_absolute() or '..' in original.parts:
                raise ValueError(member.name)
            relative = Path(*original.parts[1:])
            if not relative.parts:
                continue
            target = destination/relative
            if member.issym() or member.islnk():
                links.append((member, target))
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
        # 上游测试目录有指向仓库内文件的符号链接；只复制安全的仓库内目标。
        for member, target in links:
            source = ((target.parent/member.linkname).resolve() if member.issym() else
                      (destination/Path(*Path(member.linkname).parts[1:])).resolve())
            if not source.is_relative_to(destination):
                raise ValueError(member.linkname)
            target.write_bytes(source.read_bytes())
    (destination/'UPSTREAM_COMMIT').write_text(lock['commit'] + '\n')
    print(destination)


if __name__ == '__main__':
    main()
