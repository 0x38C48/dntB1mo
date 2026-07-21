# WeChat Persona Chatbot MVP

一个轻量版微信聊天人格模拟项目。当前版本重点是把 `backup/user` 侧聊天风格蒸馏成可运行的本地 Web 聊天机器人，并通过 RAG 检索相似历史片段辅助回复。

## 功能

- 五层人格蒸馏：
  - 怎么说话：语气、节奏、短句、用词偏好
  - 怎么想：心智模型和认知框架
  - 怎么判断：决策启发式
  - 什么不做：反模式和价值观底线
  - 知道局限：诚实边界和隐私边界
- 本地词法 RAG，最多返回 10 条相关记忆。
- 梗词、谐音和拼音缩写扩展检索，例如 `xswl`、`丸辣`、`尊嘟假嘟`。
- 可选接入模型 API，未配置密钥时自动使用本地兜底回复。
- 前后端一体的 Python 标准库服务，无重型框架依赖。

## 运行

在 `F:\dntB1mo\chatbot` 下运行：

```powershell
python scripts\build_persona.py
.\scripts\start_chatbot.ps1
```

打开：

```text
http://127.0.0.1:8765
```

## API 配置

推荐把密钥只放在本机环境变量里，不要写入代码或提交到仓库：

```powershell
$env:SOPHNET_API_KEY="..."
$env:SOPHNET_MODEL="DeepSeek-V4-Flash"
```

也可以使用 OpenAI 兼容配置：

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_MODEL="gpt-4o-mini"
```

## 隐私说明

仓库默认忽略原始聊天记录、蒸馏出的 persona 文件、头像、日志和运行缓存。公开提交前请确认 `.gitignore` 仍然生效。
