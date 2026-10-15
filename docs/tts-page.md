# CosyVoice HTML 语音测试页

## 远端没有配置 CORS 时

直接双击 HTML 无法绕过远端的 CORS 限制。可以用 Python 标准库启动本地页面及请求转发器，无需安装项目、模型或任何第三方依赖：

```bash
python3 tools/tts_test_server.py --target http://10.16.1.86:8080
```

浏览器打开 `http://127.0.0.1:8088/`，点击「测试连接 / 加载音色」。页面地址自动设置为本地转发接口，无需再次填写远端地址。修改远端服务时停止启动器并重新设置 `--target`。本地端口可用 `--port` 更改。

浏览器只请求同源的本地服务，由 Python 访问远端。这种方式无需更改远端 CORS，但电脑仍须能够通过网络访问远端。转发器仅监听 `127.0.0.1`，只转发 `/health`、`/speakers` 和 `/tts` 三个接口。复制使用时一并保留根目录的 `cosyvoice-test.html` 和 `tools/tts_test_server.py`。

## 直接访问 CosyVoice 服务

启动本项目服务：

```bash
python app.py
```

浏览器打开 `http://127.0.0.1:8080/static/tts.html`。远程部署时将主机和端口替换为实际地址。

Docker Compose 已挂载整个静态目录 `./static:/workspace/CosyVoice/static`（GPU、NPU 配置均支持）。修改挂载配置后运行 `docker compose up -d --force-recreate`；NPU 使用 `docker compose -f docker-compose.npu.yml up -d --force-recreate`。确保宿主机 `static/tts.html` 已上传到部署目录。直接通过服务的 `/static/tts.html` 打开页面时，默认请求同源 API，无需跨域配置。

1. 填写 CosyVoice 服务根地址，例如 `http://192.168.1.10:8080`，或带代理前缀的 `https://example.com/cosyvoice`。
2. 点击「测试连接 / 加载音色」，选择已注册的音色。
3. 输入文字，按需调整语速和随机种子，点击「生成语音」。
4. 使用播放器试听，或点击「下载 WAV 文件」保存到本机。

页面是独立 HTML，不需要安装前端依赖，也可以直接打开 `static/tts.html` 文件后填写服务地址。地址和音色选择保存在当前浏览器中，文字及音频不保存在浏览器存储中。

页面使用 `GET /health`、`GET /speakers`，并以表单方式请求 `POST /tts`，传入 `text`、`voice`、`speed`（1–9，5 为正常语速）、`text_seed`、`wav=1`、`is_stream=0`。音频通过 Blob URL 供播放和下载。

`app.py` 支持通过 `CORS_ORIGINS` 配置允许的页面来源，逗号分隔，默认 `*`。例如：

```bash
CORS_ORIGINS='https://tts.example.com,http://localhost:8080' python app.py
```

HTTPS 页面需要连接 HTTPS 服务。没有音色时，在 `speakers/` 中添加参考音频和对应文本后重启服务。连接测试超时为 15 秒；生成等待超时为 5 分钟。取消等待仅停止浏览器等待，不保证中断服务端推理。
