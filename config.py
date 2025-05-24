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

# config.py (在文件末尾或合适位置添加以下内容)

# --- RAG 和 LLM 结果评估相关配置 ---
UNCERTAINTY_PHRASES = [
    "无法找到", "没有相关信息", "未能找到", "无法提供", "不确定",
    "我不知道", "无法确定", "没有足够信息", "目前无法回答",
    "To", "I cannot", "I don't", "Unable to", "not find", "no information",
    "对不起，我无法", "抱歉，我无法" # 添加更多常见的不确定性短语
]
RAG_SCORE_THRESHOLD = 0.45  # RAG 文档相关性得分阈值 (可根据实际效果调整)
RAG_ANSWER_MIN_LENGTH = 25  # RAG 回答最小长度阈值 (字符数, 可调整)

# --- 机器学习关键词列表 ---
ML_KEYWORDS = [
    '机器学习', '模型', '训练', '预测', '分类', '回归', '聚类', '算法', '特征', '数据',
    '随机森林', '决策树', '支持向量机', 'svm', 'knn', 'k近邻', '逻辑回归', '线性回归',
    '神经网络', '深度学习', '朴素贝叶斯', 'k-means', 'xgboost', 'lightgbm', 'catboost',
    '准确率', '精确率', '召回率', 'f1分数', 'auc', 'roc', 'mse', 'rmse', 'mae', 'r方', 'r2',
    '超参数', '验证集', '测试集', '过拟合', '欠拟合', '特征工程', '降维', 'pca',
    'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'sklearn', 'paddlepaddle', 'paddle'
]

ML_OPS_KEYWORDS = [
    '训练', '预测', '比较', '评估', '构建', '解释', '优化', '部署', '监控', '保存', '加载',
    '选择模型', '调整参数', '分析特征', '生成报告', '自动化', '工作流',
    '版本控制', '流水线', 'pipeline', 'finetune', '微调', '自动机器学习', 'automl'
]

# --- 应用行为相关配置 (示例，您可以按需添加更多) ---
# 例如，上传文件存储位置，虽然您在 app.py 中定义了 UPLOADS_DIR，但也可以考虑放在这里
# UPLOADS_DIR = os.path.join(os.getcwd(), "uploads")
# MODELS_STORAGE_DIR = os.path.join(os.getcwd(), "ml_models")

# 默认的预览行数
DEFAULT_PREVIEW_ROWS = 10
