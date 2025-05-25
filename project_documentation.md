# 项目名称

AI机器学习助手 Pro

## 项目目的

这是一个集成了RAG检索增强生成和机器学习模型的智能助手系统，可以回答机器学习相关问题，并提供模型训练、预测、分析和可视化功能。

## 后端应用 (Flask - app.py)

`app.py` 是项目的主要后端应用程序，基于 Flask 框架构建。它负责处理客户端请求，调度机器学习任务，并与 RAG 系统交互。

### 主要 API 端点:

*   **`/api/chat` (POST)**
    *   **功能**: 核心交互接口。根据用户查询的类型（通用知识、机器学习操作、数据分析、代码生成、模型教程等）和提供的上下文（如上传的数据预览、选择的目标列或模型），智能地将请求路由到 RAG 系统、机器学习代理 (ML Agent) 或直接调用大语言模型 (LLM) 进行处理。
    *   **请求体**: 包含 `query` (用户输入) 和可选的 `mode` (如 `data_analysis`, `general_llm`), `data_preview`, `target_column`, `model_name` 等。
    *   **响应**: 返回包含答案、可能的源文档、是否为机器学习查询等信息的 JSON 对象。

*   **`/api/ml/train` (POST)**
    *   **功能**: 训练新的机器学习模型。
    *   **请求体**: 包含 `model_type`, `data_path`, `target_column`, 以及可选的 `model_name`, `categorical_columns`, `numerical_columns`, `model_params`, `test_size`。
    *   **响应**: 返回包含模型名称、类型、评估指标和成功消息的 JSON 对象。

*   **`/api/ml/predict` (POST)**
    *   **功能**: 使用已训练的模型进行预测。
    *   **请求体**: 包含 `model_name` 和 `input_data`。
    *   **响应**: 返回包含预测结果和输入数据的 JSON 对象。

*   **`/api/ml/analyze` (POST)**
    *   **功能**: 分析数据集，提供统计信息、相关性分析和推荐模型。
    *   **请求体**: 包含 `data_path` 和可选的 `target_column`。
    *   **响应**: 返回包含详细分析结果的 JSON 对象。

*   **`/api/ml/upload` (POST)**
    *   **功能**: 上传数据文件 (CSV, XLSX, JSON)。
    *   **请求体**: `multipart/form-data` 包含文件。
    *   **响应**: 返回文件路径、列名、数据预览等信息的 JSON 对象。

### 代码示例: `/api/ml/train` 端点

```python
@app.route('/api/ml/train', methods=['POST'])
def train_model_endpoint():
    """训练机器学习模型的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    required_fields = ['model_type', 'data_path', 'target_column']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 直接调用ml_models.py中的train_model函数
        from ml_models import train_model

        model_type = data['model_type']
        data_path = data['data_path']
        target_column = data['target_column']
        model_name = data.get('model_name')
        categorical_columns = data.get('categorical_columns', [])
        numerical_columns = data.get('numerical_columns', [])
        model_params = data.get('model_params', {})
        test_size = data.get('test_size', 0.2)

        # 如果是Excel文件，转换为CSV以便更好地处理
        if data_path.endswith('.xlsx'):
            import pandas as pd
            df = pd.read_excel(data_path)
            csv_path = data_path.replace('.xlsx', '_processed.csv')
            df.to_csv(csv_path, index=False)
            data_path = csv_path
        
        # 训练模型
        result = train_model(
            model_type=model_type,
            data=data_path,
            target_column=target_column,
            model_name=model_name,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            model_params=model_params,
            test_size=test_size
        )

        # 格式化结果以便前端显示
        formatted_result = {
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "metrics": result["metrics"],
            "message": f"成功训练{model_type}模型，模型名称为{result['model_name']}"
        }
        return jsonify(formatted_result), 200
    except Exception as e:
        # app.logger.error(f"/api/ml/train 接口发生错误: {e}", exc_info=True) # Assuming app.logger is configured
        return jsonify({"error": f"训练模型时发生错误: {str(e)}"}), 500
```
此外, `app.py` 还包含了模型版本控制、模型比较、集成模型构建、模型部署与取消部署、模型解释以及自动模型选择等功能的API端点。它也负责初始化RAG系统并在应用启动时检查各项配置。

## 机器学习核心 (`ml_models.py`)

`ml_models.py` 模块是项目中所有机器学习操作的核心。它封装了数据预处理、模型训练、评估、预测以及更高级的ML功能，旨在提供一个统一的接口来处理各种机器学习任务。

### 主要功能:

-   **数据预处理 (`preprocess_data`)**: 包含对数据进行清洗、特征编码（如标签编码）、特征标准化等步骤，为模型训练准备合适格式的数据。
-   **模型训练 (`train_model`)**: 支持多种类型的模型训练，包括回归模型（如线性回归）、分类模型（如逻辑回归、决策树、随机森林、K-近邻、支持向量机、朴素贝叶斯）和聚类模型（如K-Means）。函数接收模型类型、数据、目标列、模型名称、特征列等参数，并返回包含训练结果和评估指标的字典。
-   **模型预测 (`predict`)**: 使用已保存和加载的模型进行预测。
-   **模型加载与列出 (`load_model`, `list_available_models`)**: 提供加载已保存模型和列出所有可用模型的功能。
-   **模型版本控制 (`save_model_with_version`, `list_model_versions`, `load_model_version`)**: 支持模型版本化，可以保存模型的不同版本，并加载特定版本。
-   **集成学习 (`create_ensemble_model`)**: 支持创建集成模型，如投票(Voting)、堆叠(Stacking)和装袋(Bagging)集成，以提升模型性能。
-   **自动机器学习 (`auto_model_selection`)**: 提供自动化模型选择和超参数优化的能力，帮助用户找到最适合特定数据集和任务的模型。
-   **模型解释 (`explain_model_prediction`)**: 对模型的预测结果进行解释，分析特征重要性和贡献。
-   **模型比较 (`compare_models`)**: 在同一测试数据集上比较多个模型的性能。

### 代码示例: `train_model` 函数

```python
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd # Assuming pandas is used, adjust as necessary

def train_model(
    model_type: str,
    data: Union[pd.DataFrame, str],
    target_column: str,
    model_name: str = None,
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None,
    model_params: Dict[str, Any] = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    训练机器学习模型并保存
    
    参数:
        model_type: 模型类型（例如："linear_regression", "logistic_regression"等）
        data: DataFrame数据或CSV文件路径
        target_column: 目标变量列名
        model_name: 模型保存名称（如果为None，则使用模型类型作为名称）
        categorical_columns: 分类特征列表
        numerical_columns: 数值特征列表
        model_params: 模型参数字典
        test_size: 测试集比例
        
    返回:
        Dictionary包含训练结果和评估指标
        
    可能抛出的异常:
        ValueError: 当模型类型不支持或数据格式不正确时
        FileNotFoundError: 当数据文件不存在时
        KeyError: 当目标列不存在时
    """
    # ... (Implementation would involve:
    # 1. Loading data if 'data' is a path.
    # 2. Preprocessing data (handling categorical/numerical features, splitting).
    # 3. Initializing model_cls based on model_type.
    # 4. Fitting the model: model.fit(X_train, y_train).
    # 5. Evaluating the model: calculating metrics.
    # 6. Saving the model and preprocessors.
    # 7. Returning a dictionary with model_name, metrics, etc.)
    pass # Actual implementation is in the ml_models.py file
```
该模块通过这些功能，为AI助手提供了强大的机器学习后端支持。

## 机器学习智能代理 (`ml_agents.py`)

`ml_agents.py` 模块利用 LangChain 框架，将 `ml_models.py` 中的核心机器学习功能包装成可供语言模型调用的“工具”（Tools）。这使得系统能够通过自然语言接口执行复杂的机器学习任务，例如模型训练、数据分析和结果可视化。

### 核心机制:

-   **工具封装**: `ml_models.py` 中的关键函数（如 `train_model`, `predict`, `auto_model_selection`, `generate_visualization` 等）被定义为 LangChain 的 `StructuredTool`。每个工具都具有明确的输入模式，这些模式通常通过 Pydantic 模型（例如 `TrainModelInput`, `PredictInput`）来定义，确保了类型安全和参数的清晰描述。
-   **Agent 执行器**: 系统采用 LangChain Agent（如 `OpenAIFunctionsAgent` 或 `StructuredChatAgent` 结合 `AgentExecutor`）来解析用户的自然语言查询。Agent 的职责是理解用户意图，从可用的工具集中选择最合适的工具或工具序列，并从查询中提取执行这些工具所需的参数。
-   **自然语言驱动的ML任务**: 用户可以用日常语言提出请求，例如“用逻辑回归模型训练一个客户流失预测模型，目标列是 'Churn'，数据集是 'data.csv'”。Agent 会将此请求解析，并调用相应的 `_train_model` 工具，同时传入从请求中提取的参数。

### 主要功能与交互流程:

1.  **任务理解与工具选择**: `query_ml_agent` 函数是与 ML Agent 交互的主要入口。它接收用户查询后，LangChain Agent 会分析查询内容，判断用户意图，并从一系列预定义的 ML 工具中选择一个或多个来执行。
2.  **参数提取**: Agent 负责从用户的自然语言查询中准确提取执行工具所需的参数，如模型类型、数据文件路径、目标列名称、要生成的图表类型等。
3.  **工具执行**: 一旦选定工具并提取了参数，Agent 就会调用该工具。这实际上会触发执行 `ml_models.py` 中对应的底层函数。
4.  **结果整合与返回**: 工具执行后返回的结果（例如模型训练的评估指标、预测输出、Base64 编码的图表图像、图表相关的表格数据等）会先返回给 Agent。Agent 随后可以将这些信息整合成一个结构化的响应（通常是 JSON），或者生成一段自然语言描述，最终呈现给用户。

### 可视化能力:

-   在`ml_agents.py`中，作为工具暴露给Agent的**包装函数**（wrapper functions）在调用`ml_models.py`中的核心ML操作（如模型训练、特征分析）后，会进一步调用相应的可视化函数（例如`ml_models.generate_visualization`, `ml_models.visualize_feature_importance`等）。
-   这些包装函数在获得核心ML操作（如模型训练、特征分析）的结果后，会进一步调用可视化函数生成图表。
-   生成的图表会**编码为 Base64 字符串**，方便在Web界面中直接嵌入显示。
-   除了图像数据，这些工具通常还会返回与图表直接相关的**表格数据**（例如，特征重要性得分列表、相关系数矩阵的数值）。这使得信息可以以多种形式（图形和表格）呈现给用户，增强了可解释性。
-   因此，当用户请求一个会产生可视化的任务时（例如“训练模型并显示特征重要性”），Agent调用的工具的包装函数会负责调用核心ML函数和相应的可视化函数，并将所有结果（文本、Base64图像、表格数据）整合后返回。

### 代码示例: 工具输入定义 (`TrainModelInput`)

以下代码片段展示了用于 `_train_model` 工具的输入参数Pydantic模型 `TrainModelInput`。这个模型定义了工具期望接收的参数、它们的类型以及描述，确保了 Agent 能够准确地从用户自然语言中提取信息并传递给工具。

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TrainModelInput(BaseModel):
    model_type: str = Field(..., description="Type of the model to train (e.g., 'logistic_regression', 'random_forest_classifier').")
    data_path: str = Field(..., description="Path to the dataset file (CSV, Excel, or JSON).")
    target_column: str = Field(..., description="Name of the target variable column in the dataset.")
    model_name: Optional[str] = Field(None, description="Optional name to save the trained model.")
    categorical_columns: Optional[List[str]] = Field(None, description="List of categorical feature column names.")
    numerical_columns: Optional[List[str]] = Field(None, description="List of numerical feature column names.")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Dictionary of model-specific hyperparameters.")
    test_size: Optional[float] = Field(0.2, description="Proportion of the dataset to include in the test split.")

# Conceptual example of tool creation:
# from langchain.tools import StructuredTool
# def _train_model_wrapper_for_agent(input_args: TrainModelInput) -> dict:
#    # This wrapper would call the actual ml_models.train_model
#    # using input_args.model_type, input_args.data_path etc.
#    # and then format the result, potentially adding visualizations.
#    # result_from_ml_models = ml_models.train_model(**input_args.dict())
#    # return formatted_result_with_visuals
#    pass 
#
# train_model_tool_for_agent = StructuredTool.from_function(
#     func=_train_model_wrapper_for_agent,
#     name="TrainModel",
#     description="Trains a machine learning model based on specified parameters and dataset, and returns results including metrics and visualizations.",
#     args_schema=TrainModelInput
# )
```
通过这种方式，`ml_agents.py` 作为自然语言接口与底层机器学习功能之间的桥梁，极大地增强了系统的易用性和交互性，使得非技术用户也能方便地利用强大的机器学习能力。

## 检索增强生成核心 (`rag_core.py`)

`rag_core.py` 模块是项目中实现检索增强生成 (Retrieval Augmented Generation, RAG) 功能的核心。它使得系统能够基于本地知识库中的文档内容，结合大型语言模型 (LLM) 来回答用户的问题，提供更具上下文和事实依据的答案。

### 主要功能与流程:

1.  **文档加载与处理**:
    *   `load_documents_from_kb()`: 从 `KNOWLEDGE_BASE_DIR` (在 `config.py` 中定义) 加载多种格式的文档，包括：PDF (`PyPDFLoader`), Word 文档 (`UnstructuredWordDocumentLoader`), 纯文本 (`TextLoader`), 结构化 JSON (`load_and_parse_custom_json` 使用 `config.JSON_JQ_SCHEMA` 即 `'.[] | .sentence'` 配置提取特定字段), 和 CSV 文件 (summarized by `generate_csv_summary_documents`)。
    *   `split_documents()`: 加载后的文档被分割成较小的文本块 (chunks) 使用 `RecursiveCharacterTextSplitter`，以便于后续的向量化和检索。分割大小和重叠部分由 `config.py` 中的 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 控制。

2.  **向量存储创建与管理**:
    *   `get_vector_store()`:
        *   使用百度文心提供的 Embedding 模型 (`config.BAIDU_EMBEDDING_MODEL_NAME`, 即 `bge-large-zh`) 将文本块转换为向量表示。
        *   这些向量存储在 `ChromaDB` 向量数据库中，该数据库会持久化到 `CHROMA_PERSIST_DIR` (在 `config.py` 中定义)。
        *   系统会尝试从磁盘加载已存在的数据库，如果不存在或强制重建，则会重新处理知识库文档并创建新的数据库。
    *   `initialize_rag_system()`: 在应用启动时调用，确保向量数据库和QA链准备就绪。可以配置为强制重建向量数据库。

3.  **问答 (QA) 链机制**:
    *   `get_qa_chain()`:
        *   创建一个 `RetrievalQA` 链。
        *   该链使用从 ChromaDB 构建的检索器 (retriever) 根据用户问题查询相关的文本块。
        *   检索到的文本块与原始问题一起被传递给一个大型语言模型 (`config.BAIDU_LLM_MODEL_NAME`, 即 `ernie-4.5-turbo-128k`)。
        *   LLM 基于这些信息生成最终答案。
    *   `query_rag()`: 这是与 RAG 系统交互的主要函数。它接收用户问题，调用 QA 链，并返回包含答案和源文档信息的字典。
    *   `direct_query_llm()`: 提供一个直接调用LLM的接口，不经过RAG检索过程。

### 代码示例: `query_rag` 函数

```python
from typing import Dict, Any
from langchain_core.documents import Document # Or from langchain.schema import Document

# Assuming qa_chain_instance is a global or class-level variable, 
# properly initialized by initialize_rag_system() before this function is called.
# For snippet purposes, its direct availability is assumed.
# global qa_chain_instance 

def query_rag(question: str) -> Dict[str, Any]:
    """
    使用RAG系统查询问题的答案。

    Args:
        question: 用户提出的问题字符串。

    Returns:
        一个字典，包含:
        - "answer": LLM生成的答案。
        - "source_documents": 一个列表，包含检索到的源文档片段及其元数据。
    """
    # global qa_chain_instance # Uncomment if it's a global variable accessed here
    if 'qa_chain_instance' not in globals() or qa_chain_instance is None: # A way to check if initialized
        # This would typically be handled by initialize_rag_system or raise an error
        print("警告: RAG问答链未初始化。请先调用 initialize_rag_system().")
        return {"answer": "错误: RAG问答链未初始化。", "source_documents": []}

    try:
        print(f"RAG系统接收到查询: {question}")
        # Conceptual: result = qa_chain_instance.invoke({"query": question})
        # For snippet purposes, we'll mock a realistic-looking result structure.
        result = {
            "result": f"Generated answer to: {question}", 
            "source_documents": [
                Document(page_content="Relevant context from a source document...", metadata={"source": "knowledge_base/doc1.pdf"})
            ]
        }
        answer = result.get("result", "未能找到明确的答案。")
        source_docs_raw = result.get("source_documents", [])
        
        formatted_sources = []
        if source_docs_raw:
            for doc_item in source_docs_raw:
                if isinstance(doc_item, Document): 
                    source_info = {
                        "page_content": doc_item.page_content[:100] + "..." if len(doc_item.page_content) > 100 else doc_item.page_content, # Limit snippet length
                        "metadata": doc_item.metadata
                    }
                    formatted_sources.append(source_info)
        
        return {"answer": answer, "source_documents": formatted_sources}
    except Exception as e:
        # In a real app, use proper logging: print(f"RAG查询过程中发生错误: {e}", exc_info=True)
        # For snippet:
        print(f"RAG查询过程中发生错误: {e}")
        return {"answer": f"处理您的问题时发生错误: {str(e)}", "source_documents": []}
```

通过这些组件，`rag_core.py` 为系统提供了强大的知识检索和智能问答能力。

## 项目配置 (`config.py` 与 `.env`)

项目的配置管理采用分层方式，结合使用 `.env` 文件和 `config.py` 脚本，以实现灵活性和安全性。

### 1. 环境变量 (`.env` 文件)

-   **核心作用**: `.env` 文件是存储所有敏感配置信息的地方，例如 API 密钥、数据库连接字符串以及特定于部署环境的变量（如 `FLASK_ENV`）。**此文件绝对不能提交到版本控制系统（例如 Git）**，以避免安全凭证泄露。通常，项目中会包含一个 `.env.example` 文件，作为用户配置实际 `.env` 文件的模板。
-   **加载机制**: 应用在启动时（通常在 `config.py` 或主应用脚本 `app.py` 的早期），使用 `python-dotenv` 库的 `load_dotenv()` 函数来读取 `.env` 文件，并将其中的键值对加载到操作系统的环境变量中。
-   **主要配置项**:
    *   `AI_STUDIO_API_KEY`: 用于访问百度AI Studio大模型平台服务的API密钥，主要用于文心千帆大模型及Embedding服务。
    *   `ERNIE_API_KEY` / `ERNIE_SECRET_KEY` (可选): 如果项目直接使用百度智能云文心千帆SDK，则可能需要配置这些更具体的凭证。
    *   `FLASK_ENV`: 指定Flask应用的运行环境，如 `development` 或 `production`。
    *   `FLASK_DEBUG`: 控制是否开启Flask的调试模式 (通常在开发环境设为 `True`)。

### 2. 应用固定配置 (`config.py`)

-   **核心作用**: `config.py` 脚本负责定义项目中非敏感的、相对固定的配置参数。它首先会尝试从环境变量中加载由 `.env` 文件设置的敏感信息，然后定义其他应用层面的参数。
-   **参数类型与示例**:
    *   **API密钥与模型名称**:
        *   从环境变量中读取 `AI_STUDIO_API_KEY` 等。
        *   定义默认使用的LLM模型名称，如 `BAIDU_LLM_MODEL_NAME = "ernie-4.5-turbo-128k"`。
        *   定义默认使用的Embedding模型名称，如 `BAIDU_EMBEDDING_MODEL_NAME = "bge-large-zh"`。
    *   **文件与目录路径**:
        *   `BASE_DIR`: 项目的根目录。
        *   `KNOWLEDGE_BASE_DIR`: 存放RAG知识库文档的目录。
        *   `CHROMA_PERSIST_DIR`: ChromaDB向量数据库的持久化存储目录。
        *   `LOG_FILE_PATH`: 日志文件的输出路径。
    *   **RAG文本处理参数**:
        *   `CHUNK_SIZE`: 文档分割时每个文本块的目标大小。
        *   `CHUNK_OVERLAP`: 分割后相邻文本块之间的重叠字符数。
    *   **RAG检索与Agent行为参数**:
        *   `RAG_SEARCH_TYPE`: 向量检索类型（如 `similarity`, `mmr`）。
        *   `RAG_K`: 检索时返回的最相关文档数量。
        *   `RAG_SCORE_THRESHOLD`: 认定检索文档为相关的最低分数阈值。
        *   `RAG_ANSWER_MIN_LENGTH`: RAG系统生成答案的期望最小长度。
        *   `UNCERTAINTY_PHRASES`: 用于识别LLM回答不确定性的短语列表。
        *   `ML_KEYWORDS`, `ML_OPS_KEYWORDS`: 用于辅助判断用户查询是否与机器学习任务相关的关键词列表。
    *   **其他应用参数**:
        *   `JSON_JQ_SCHEMA`: 用于从特定结构的JSON文件中提取内容的 `jq` 模式 (`'.[] | .sentence'`)。
        *   `LOG_LEVEL`: 应用的日志记录级别（如 `INFO`, `DEBUG`）。

### 代码示例: `config.py`

以下是 `config.py` 中部分参数定义的示例，展示了如何从环境变量加载敏感数据并定义其他应用参数：

```python
import os
from dotenv import load_dotenv

# 加载 .env 文件中定义的环境变量
# 这使得 os.getenv 可以访问 .env 中设置的值
load_dotenv()

# --- API密钥与模型配置 ---
# 从环境变量中获取百度AI Studio的API Key
AI_STUDIO_API_KEY = os.getenv("AI_STUDIO_API_KEY")
# 同样的方式可以获取ERNIE_API_KEY和ERNIE_SECRET_KEY，如果它们被使用

# 定义默认使用的语言模型和Embedding模型
BAIDU_LLM_MODEL_NAME = "ernie-4.5-turbo-128k" 
BAIDU_EMBEDDING_MODEL_NAME = "bge-large-zh" # 百度 Ernie Embeddings官方推荐使用此模型名

# --- 路径配置 ---
# BASE_DIR 定义为 config.py 文件所在的目录 (即项目根目录, 假设config.py在根目录)
BASE_DIR = os.path.abspath(os.path.dirname(__file__)) 
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "vector_store", "chroma_db_ernie") 
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE_PATH = os.path.join(LOG_DIR, "ai_assistant.log")
# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

# --- RAG文本分割参数 ---
CHUNK_SIZE = 800        # 每个文本块的目标大小 (字符数)
CHUNK_OVERLAP = 100     # 文本块之间的重叠大小 (字符数)

# --- RAG检索与答案生成参数 ---
RAG_SEARCH_TYPE = "similarity"  # 向量检索类型, 可选 "mmr" (Maximal Marginal Relevance)
RAG_K = 5                       # 检索时返回的最相关文档数量
RAG_SCORE_THRESHOLD = 0.35      # 检索文档的最低相关性得分阈值 (0到1之间)
RAG_ANSWER_MIN_LENGTH = 30      # RAG系统生成答案的期望最小长度 (字符数)

# --- Agent行为与关键词 ---
# 用于初步判断用户查询意图的关键词列表
ML_KEYWORDS = ["模型", "训练", "预测", "分析", "特征", "评估", "算法", "machine learning", "model", "train", "predict", "analyze", "plot", "visualize"]
ML_OPS_KEYWORDS = ["保存", "加载", "版本", "部署", "监控", "save", "load", "version", "deploy", "monitor"]
# 用于识别模型回答中不确定性的短语
UNCERTAINTY_PHRASES = ["无法找到", "不确定", "不知道", "没有相关信息", "未能", "无法提供", "I'm sorry", "I cannot", "I don't know", "Unable to answer"]

# --- 其他 ---
JSON_JQ_SCHEMA = "'.[] | .sentence'" # 用于从特定JSON结构中提取文本的jq schéma
LOG_LEVEL = "INFO" # 应用的日志级别
```
通过这种配置分离策略，项目在保证敏感信息安全的同时，也为不同部署环境和应用行为的调整提供了便利。
