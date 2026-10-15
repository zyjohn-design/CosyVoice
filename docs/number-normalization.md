# 数字读法统一由 wetext 规则维护

之前的 CosyVoice 前端编号预处理已撤回。验证码、设备编号和订单编号的逐位读法现在由原生 WFST 规则完成，应用程序继续调用原有 wetext 接口。

规则源码、字段词典、固定上游版本、编译和回归测试都在 [`wetext_rules/`](../wetext_rules/README.md)。

- 新增字段：编辑 `wetext_rules/data/identifier_fields.tsv` 并增加 `tests.json` 测试。
- 新增读法类型：修改 `rules.py` 并在 `normalizer.py` 的规则清单中注册。
- 发布：编译新的 FST，使用 wetext 0.0.4 验证，再更新 Docker 挂载的模型目录并重新创建容器。

当前验证仅涵盖文本规范化，服务器仍需检查 `synthesis text` 日志并试听音频。
