# Changelog

## 0.1.1 (2025-08-08)

- 发布准备：新增 `py.typed` 类型标记、SPDX license 元数据、`[project.urls]`、
  CI（GitHub Actions，Python 3.12/3.13 矩阵）、CHANGELOG
- 版本单一来源：由 `src/openai_io/__init__.py` 的 `__version__` 动态生成

## 0.1.0

- 首个版本：`chat.completions` / `completions` / `embeddings` 同步与异步客户端（基于 httpx）
- langchain 风格 message 体系（`SystemMessage` / `HumanMessage` / `AIMessage` /
  `ToolMessage` / `FunctionMessage` / `ChatMessage`），兼容 dict 混合输入
- SSE 流式输出（`Stream` / `AsyncStream`）
- 异常体系与 openai SDK 对齐，自动重试（指数退避 + 抖动）
- 完整类型注解（pyright strict 0 错误），发布 `py.typed`
