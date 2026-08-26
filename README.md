# pi Web Search

为 [pi](https://github.com/earendil-works/pi) 增加 provider-native 的 `web_search` 工具。

## 安装

```bash
pi install npm:@yankewei/pi-web-search
```

安装后重启 pi 即可。

## 原理

1. 扩展向 pi 注册 `web_search` 工具。
2. 模型需要实时信息时调用该工具，扩展把查询交给当前 provider 的原生 Web Search。
3. DeepSeek 使用 Responses API 的 `web_search`；GPT-5.6 使用 pi 的 `openai-codex-responses` 和现有 OAuth 登录。
4. 返回结果中的正文引用会转换为可点击的 `Sources` 列表；不支持的模型会返回明确错误。

扩展不实现搜索引擎，也不需要单独维护搜索 API。认证、`baseUrl`、headers 和环境配置由 pi 管理。

## 支持模型

DeepSeek：

- `deepseek-v4-flash`

GPT-5.6（必须使用 `openai-codex` provider）：

- `openai-codex/gpt-5.6-luna`
- `openai-codex/gpt-5.6-sol`
- `openai-codex/gpt-5.6-terra`

例如：

```bash
pi --model openai-codex/gpt-5.6-luna
```

Codex 使用 pi 已有的 OAuth 凭证，不需要额外配置 `OPENAI_API_KEY`。

## 本地开发

```bash
npm install
npm run check
npm test
```

本地加载扩展：

```bash
pi -e ./src/index.ts
```
