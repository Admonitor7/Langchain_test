# LangChain 代码库问答项目

这是一个基于 LangChain/FastAPI 的代码库问答项目，支持命令行问答和 API 服务。

## 当前能力

- 单次问答
- 多轮聊天
- 文本摘要
- 本地知识库问答
- 代码库问答
- 离线代码索引
- Chroma 向量检索
- BM25 关键词检索
- RRF 结果融合
- FastAPI 接口
- SSE 流式返回

## 架构

```text
用户提问
    ↓
FastAPI 接口层
    ↓
检索层 ←————————— 索引层（离线）
向量检索 + 关键词检索        代码文件解析
    ↓                       ↓
上下文组装              Embedding 向量化
    ↓                       ↓
LLM 生成层              存入 Chroma 向量数据库
    ↓
SSE 流式返回答案
```

## 项目结构

```text
Langchain_test/
├── indexer/
│   ├── parser.py        # 代码解析、按函数/类/标题分块
│   └── embedder.py      # Embedding 向量化 + Chroma 存储
├── retriever/
│   ├── vector_search.py # Chroma 向量检索
│   ├── bm25_search.py   # BM25 关键词检索
│   ├── fusion.py        # RRF 结果融合
│   └── hybrid.py        # 混合检索入口
├── generator/
│   └── llm.py           # LLM 调用 + prompt
├── storage/
│   └── chat_store.py    # 本地聊天历史
├── api/
│   └── main.py          # FastAPI 路由
├── config.py            # 配置读取
├── main.py              # 命令行入口
└── requirements.txt     # 依赖列表
```

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 配置

`.env` 中 LLM 相关配置继续使用：

```env
LLM_PROVIDER=openai_compatible
LLM_API_KEY=你的模型服务密钥
LLM_BASE_URL=https://你的模型服务/v1
LLM_MODEL=你的模型名称
LLM_TEMPERATURE=0.2
```

代码库索引相关配置：

```env
EMBEDDING_PROVIDER=local_hash
EMBEDDING_MODEL=local-hash
CHROMA_PERSIST_DIR=D:\workspace\Langchain_test\data\chroma
CHROMA_COLLECTION=code_chunks
CODE_INDEX_DIR=D:\workspace\Langchain_test\data\code_index
INDEX_REPO_PATH=D:\workspace\Langchain_test
```

如果你的服务支持 OpenAI-compatible Embedding，可以改成：

```env
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BASE_URL=https://你的模型服务/v1
EMBEDDING_API_KEY=你的 Embedding 密钥
```

`local_hash` 适合本地演示完整流程；生产环境建议使用真实 Embedding 模型。

## 启动 API

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

## 构建索引

```powershell
curl -X POST http://127.0.0.1:8000/index `
  -H "Content-Type: application/json" `
  -d "{\"repo_path\": \"D:\\\\workspace\\\\Langchain_test\", \"reset\": true}"
```

接口会完成：

- 解析代码文件
- Python 按函数、方法、类切分
- Markdown 按标题切分
- 配置文件按段落或顶层键切分
- Embedding 向量化
- 写入 Chroma
- 写入 BM25 本地片段存储

默认跳过：`.env`、`.idea`、`__pycache__`、虚拟环境、日志、数据库和编译缓存。

## 流式提问

```powershell
curl -N -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"main.py 如何处理模型错误\", \"top_k\": 8, \"stream\": true}"
```

返回 SSE 事件：

```text
event: metadata
data: {"chat_id":"...","sources":[...]}

event: delta
data: {"content":"..."}

event: done
data: {"chat_id":"..."}
```

## 非流式提问

```powershell
curl -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d "{\"question\": \"配置是怎么读取的\", \"stream\": false}"
```

## 获取历史对话

```powershell
curl http://127.0.0.1:8000/chat/{chat_id}
```

## 清空索引

```powershell
curl -X DELETE http://127.0.0.1:8000/index
```

## 命令行功能

查看配置：

```powershell
python main.py config
```

命令行代码库问答：

```powershell
python main.py code main.py 的入口流程是什么
```

多轮聊天：

```powershell
python main.py chat
```

## 后续可扩展方向

- 增量索引：对比 git diff，只重新索引变更文件
- GitHub 跳转：根据仓库地址生成具体行链接
- 权限控制：不同用户只能查自己有权限的仓库
- 多轮代码问答：把历史问题加入检索和生成上下文
- 前端页面：使用 EventSource 接收 SSE 实时显示答案
