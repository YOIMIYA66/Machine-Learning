# config.py
import os
from dotenv import load_dotenv

load_dotenv() # 从 .env 文件加载环境变量

# 百度AI Studio API 配置
AI_STUDIO_API_KEY = os.getenv("AI_STUDIO_API_KEY")
AI_STUDIO_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3" # 文档中指定的API服务域名

# 大语言模型ID (根据你的文档和需求选择)
# 例如: "ernie-3.5-8k", "ernie-4.0-8k", "ernie-speed-8k" 等
BAIDU_LLM_MODEL_NAME = "ernie-4.5-turbo-128k"

# Embedding 模型ID (根据你的文档和需求选择)
# 例如: "embedding-v1", "bge-large-zh"
BAIDU_EMBEDDING_MODEL_NAME = "bge-large-zh"

# 知识库路径
KNOWLEDGE_BASE_DIR = "knowledge_base"

# ChromaDB 持久化存储路径
CHROMA_PERSIST_DIR = "chroma_db"

# JSON 文件加载配置 (非常重要 - 需要根据你的JSON文件结构调整)
# 此 jq_schema 用于从JSON文件中提取要进行向量化的文本内容。
# 示例:
# 如果JSON是: [{"title": "案例A", "text_body": "详细内容A"}, {"title": "案例B", "text_body": "详细内容B"}]
# 则 jq_schema 可以是: '.[] | .text_body'
# 或者 '.[] | "标题: " + .title + "\n正文: " + .text_body'
# 如果JSON是: {"documents": [{"passage": "文本1"}, {"passage": "文本2"}]}
# 则 jq_schema 可以是: '.documents[].passage'
# 如果你的 `离婚诉讼文本.json` 文件中每个对象有一个名为 `sentence` 的字段包含主要文本:
# IMPORTANT: 请根据你的实际JSON文件结构（例如 `离婚诉讼文本.json`）验证或修改此 jq_schema。
# 当前的 '.[] | .sentence' 假定文件是一个包含对象的列表，每个对象都有一个 'sentence' 字段。
JSON_JQ_SCHEMA = '.[] | .sentence'

# 文本分割参数
CHUNK_SIZE = 450
CHUNK_OVERLAP = int(CHUNK_SIZE * 0.2)

# Embedding API 调用时的批处理大小 (embedding-v1 API限制每次最多16个输入)
EMBEDDING_BATCH_SIZE = 16