# 在 wetext 中维护中文业务规则

所有业务规则均编译到 `tagger.fst` / `verbalizer.fst` 中，CosyVoice 沿用原来的 `wetext.Normalizer` 调用。没有应用层数字预处理，也不修改已安装的 wetext Python 包。

本项目固定 WeTextProcessing 源码提交、Pynini 编译版本和 wetext 运行时版本，记录在 `upstream.lock.json`。构建时使用 Pynini 2.1.7，发布目标为 `wetext==0.0.4`。编译器只在构建环境中使用，TTS 运行时不需要安装 Pynini。

## 规则在哪里

| 文件 | 用途 |
| --- | --- |
| `data/identifier_fields.tsv` | 需要逐位读的字段名，第一列匹配文字，第二列字段读法，用 TAB 分隔 |
| `data/digits.tsv` | 每个数字的读法，目前 1 读“一”，支持半角、全角数字 |
| `data/zh_numeric_punctuation.tsv` | 无上下文数字使用中文读法的标点信号及输出标点 |
| `rules.py` | 原生 Identifier FST 语法及纯数字编号边界约束 |
| `normalizer.py` | 注册 Identifier，并在英文图中加入整句中文标点数字路径 |
| `tests.json` | 输入、期望读法、必须出现或不应出现的标注类型 |
| `build.py` | 编译并导出完整的新模型目录，不覆盖原模型 |
| `verify.py` | 直接用 wetext 0.0.4 加载 FST 验证，不导入 CosyVoice |

当前字段包括验证码、校验码、设备编号、订单编号、订单号。例如：

```text
本次验证码是001024，请勿泄露。
→ 本次验证码是零零一零二四，请勿泄露。

设备编号为1024，订单编号为202610090001。
→ 设备编号为一零二四，订单编号为二零二六一零零九零零零一。
```

中间标注为 `identifier { value: "验证码是001024" }`，原始数字保留在 tagger 字段内，verbalizer 负责转成中文。读取规则和权重的操作发生在构建阶段；生成的 FST 自身完成分类和转换。

目前 Identifier 只处理明确字段后的纯数字值。含字母、横线、斜线或小数点的复合编号不匹配 Identifier，交给原规则处理；不对其读法作逐位读保证。带空格分组的手机号仍需要另行定义电话规则。

## 没有汉字的数字文本

CosyVoice 和 wetext 0.0.4 都会将没有汉字的文本选入英文路径，中文句号本身不算汉字。因此只修改中文 FST 不能修复 `1024。` 的英文读法。

英文 FST 现在增加了一条整句规则：只有整段由数字、小数/百分比和中文标点组成时，使用中文数值读法。例如 `1024。` 输出 `一千零二十四.`，`100000。` 输出 `十万.`。末尾中文标点映射为 ASCII，兼容 CosyVoice 现有英文分句逻辑，避免追加重复句号。

这一规则只作为整个输入的候选路径注册，不参与英文句子内部的局部匹配。`Room 1024。` 和 `There are 1024 items.` 仍使用英文读法。没有中文标点的 `1024`、`1024.` 保留原来的英文默认行为；缺乏语种信号时不强行将英文数字改成中文。

## 以后新增字段

例如让“合同编号”也逐位读，在 `data/identifier_fields.tsv` 中增加一行（两列用 TAB 分隔）：

```text
合同编号	合同编号
```

然后在 `tests.json` 增加测试，例如：

```json
{
  "input": "合同编号为001024。",
  "expected": "合同编号为零零一零二四。",
  "require_types": ["identifier"]
}
```

重新编译、验证，再发布新的 FST。只编辑 TSV 后重启服务不会改变已经编译的规则。模型目录 `custom_rules/` 内的文件是随版本发布的源码快照，日常维护请编辑这里的源文件后重新构建。

新增读法类型（例如车牌或版本号）时，在 `rules.py` 添加规则类，在 `normalizer.py` 的 `rules` 清单中注册相应 `RuleSpec`，同时添加测试。tagger 和 verbalizer 使用同一个规则清单，避免只注册其中一侧。

## 准备构建环境

Linux x86_64 可以在独立 Python 3.11 环境中安装：

```bash
python3.11 -m venv .venv-wetext-build
.venv-wetext-build/bin/python -m pip install -r wetext_rules/requirements-build.txt
```

macOS 可使用 conda-forge 的 Python 3.11、Pynini 2.1.7 环境。不要在 TTS 运行容器中替换依赖。

从项目根目录下载经校验的固定上游源码（只需一次）：

```bash
python wetext_rules/fetch_upstream.py
```

下载目录默认为 `wetext_rules/_upstream`，已在该目录内的 `.gitignore` 中忽略。发布包也包含这份源码，可离线使用。

## 编译和验证

从项目根目录，指定原始完整 wetext 模型目录和一个新的输出目录：

```bash
.venv-wetext-build/bin/python wetext_rules/build.py \
  --base-model-dir /实际路径/pengzhendong/wetext \
  --output /新的路径/wetext-business-v1
```

默认从 `wetext_rules/_upstream` 读取源码。需要指定已解压的源码目录时增加 `--upstream-dir /路径/WeTextProcessing`，目录内必须带 `UPSTREAM_COMMIT` 版本标记。

编译器复制基础模型的公开文件，重新生成中文 TN 的 tagger、默认 verbalizer 和去儿化音 verbalizer，同时扩展英文 TN 的整句中文标点数字路径，保留中英文 ITN 和其他辅助文件。英文基础图保存在发布目录的 `custom_rules/base_en/` 中，后续重编译从该原始图开始，避免旧业务规则叠加。原始模型不会被改动，输出目录已存在时拒绝覆盖。

再使用原 TTS 的 wetext 0.0.4 Python 环境验证，例如本地项目环境：

```bash
.venv/bin/python wetext_rules/verify.py \
  --model-dir /新的路径/wetext-business-v1 \
  --report /新的路径/validation.json
```

检查包括 SHA256、实际读法和 tagger 类型，两套 verbalizer 都要通过。报告只验证文本转换，不代表已经生成或试听音频。

测试集同时覆盖实际 `lang=auto` 的语言选择、英文路径和正常英文句子。项目环境还可增加 `--check-frontend`，调用实际 `CosyVoiceFrontEnd.text_normalize` 验证语言分支和分句；仅构造文本方法所需状态，不加载模型权重。

## Docker 发布

已编译的发布包内有独立的 Docker 更新说明。基本流程是：解压校验，备份宿主机挂载的旧规则目录，完整替换模型目录，再用原 Compose 配置重新创建容器，让内存中的规则重新加载。

仅发布 FST 不需要改动 API、重新训练语音模型或重建镜像。若服务器之前应用过旧版 CosyVoice 预处理补丁，需要先撤回该补丁并重建一次镜像；本地项目的预处理补丁已撤回。
