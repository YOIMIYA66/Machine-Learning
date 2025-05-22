# ✨ 启航者 AI - 您的个性化学习导航与智能预测引擎 ✨

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/) <!-- 假设使用Flask -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- 假设MIT -->
[![Status](https://img.shields.io/badge/status-in%20development-orange.svg)](./)

**启航者 AI** 是一个创新的学习辅助系统。它不仅利用大语言模型（LLM）和检索增强生成（RAG）技术提供个性化的学习路径规划与智能问答，更核心的是**集成了课程所学的机器学习模型，通过对话方式对学习效果（如知识点掌握度）和学习进度（如模块完成时间）进行预测与模拟，并提供对这些模拟结果进行分析与数据处理的框架。** 本项目旨在探索将多种AI技术融合，为学习者提供前瞻性的、数据驱动的智能学习伙伴。

## 目录

- [项目亮点](#项目亮点)
- [核心问题定义](#核心问题定义)
- [核心功能详解](#核心功能详解)
  - [1. 智能学习导航 (对话式交互)](#1-智能学习导航-对话式交互)
  - [2. 个性化学习路径规划](#2-个性化学习路径规划)
  - [3. **学习效果与进度预测模拟 (ML集成核心)**](#3-学习效果与进度预测模拟-ml集成核心)
  - [4. 交互式知识库探索 (RAG驱动)](#4-交互式知识库探索-rag驱动)
  - [5. **技术实验室与模拟分析 (ML集成演示与结果评估)**](#5-技术实验室与模拟分析-ml集成演示与结果评估)
     - [核心学习预测引擎概览](#核心学习预测引擎概览)
     - [ML模型集成策略与效果对比 (模拟)](#ml模型集成策略与效果对比-模拟)
     - [模拟结果分析与数据处理 (规划)](#模拟结果分析与数据处理-规划)
- [核心技术](#核心技术)
- [系统架构概览](#系统架构概览)
- [安装与运行](#安装与运行)
  - [环境准备](#环境准备)
  - [配置](#配置)
  - [启动服务](#启动服务)
- [API接口核心](#api接口核心)
- [项目结构 (建议)](#项目结构-建议)
- [未来展望](#未来展望)
- [贡献](#贡献)
- [许可证](#许可证)

## 项目亮点

*   🧭 **个性化路径**：根据用户的学习目标和现有水平，动态生成定制化的学习计划。
*   🚀 **ML驱动的智能预测**：集成课程机器学习模型（如线性回归、逻辑回归、决策树等）**对学习效果和进度进行量化预测与模拟。
*   💬 **对话驱动**：通过自然语言与AI学习伙伴交流，获取指导、解答疑问，并进行预测模拟交互。
*   📊 **结果分析与可视化**：对机器学习模型的预测结果进行处理和可视化展示，辅助用户理解和决策。
*   📚 **RAG增强知识库**：结合本地学习资料库，提供更精准、更具上下文的知识解答。
*   💡 **透明化技术探索**：“技术实验室”让用户了解背后AI及**集成机器学习模型**的工作原理与模拟效果。
*   🧠 **百度文心大模型驱动**：利用强大的LLM进行理解、生成、解释和规划。

## 核心问题定义

本项目旨在解决的核心问题是：如何结合大语言模型和课程所学的机器学习模型，通过对话交互的方式，对学习者在特定学习任务上的**未来表现（如知识点掌握程度、任务完成时间）进行预测与模拟**，并对这些预测结果进行有效的**分析和数据处理**，从而为学习者提供更智能、更具洞察力的学习辅助。

## 核心功能详解

### 1. 智能学习导航 (对话式交互)

用户通过主界面的“学习导航”标签页与启航者AI进行对话。这是系统的核心交互入口：

-   **目标设定**：用户可以通过自然语言描述学习目标，AI将引导用户明确需求。
-   **智能问答**：针对学习过程中遇到的问题，AI结合RAG知识库和自身知识进行解答。
-   **动态引导**：根据用户当前的学习阶段和上下文，提供相关的提问建议或操作引导。

### 2. 个性化学习路径规划

一旦学习目标明确，启航者AI将为其规划一条个性化的学习路径：

-   **模块化路径**：学习路径由一系列有序的学习模块组成，每个模块包含关键知识点。
-   **动态生成**：路径的生成基于LLM对用户目标、现有知识水平以及可选课程大纲的综合理解。
-   **可视化展示**：“我的路径”标签页清晰展示学习模块、预估时长、进度和模块描述。
-   **路径调整**：用户可以与AI对话，调整学习路径。

### 3. 学习效果与进度预测模拟 (ML集成核心)

这是本项目的核心功能之一，通过对话式交互调用集成的机器学习模型进行预测：

1.  **预测流程**：
    *   用户通过自然语言描述预测需求（例如："预测我掌握线性回归这个知识点需要多久？"，"如果我每周学习10个小时，完成这个模块的概率有多大？"）。
    *   LLM解析用户意图，提取关键参数（如学习目标、声明的投入时间、学习习惯等）。
    *   特征工程模块将这些参数以及学习内容本身的特征（如模块难度、知识点关联性）转化为机器学习模型可接受的输入。
    *   集成的预测引擎根据任务类型（如掌握度预测-分类，时间预测-回归）调用相应的机器学习模型或模型组合。
    *   系统返回预测结果（如掌握概率、预计完成时间），并可附带置信区间或解释。

2.  **预测目标示例**：
    *   **知识点/模块掌握度预测** (可能使用分类模型如逻辑回归、决策树、随机森林):
        *   输入特征可能包括：用户先前相关知识点掌握情况、计划学习强度、模块复杂度等。
        *   输出：预测的掌握概率（例如：75%的概率掌握“线性回归”）。
        *   集成策略示例：基于各模型在验证集上的表现进行加权投票。
    *   **学习模块/任务完成时间预测** (可能使用回归模型如线性回归、支持向量回归SVR、梯度提升树):
        *   输入特征可能包括：模块包含的知识点数量、用户平均学习速度、计划的每日/每周学习时长等。
        *   输出：预测的完成时长（例如：预计需要12.5小时）。
        *   集成策略示例：使用Stacking，其中基础模型的预测作为元模型（如岭回归）的输入。

3.  **交互式参数调整与“What-if”分析**：
    *   用户可以通过对话调整输入参数，例如：“如果我把每周学习时间从8小时增加到12小时，掌握这个模块的可能性会提高多少？”或“如果我优先学习A知识点，再学习B知识点，完成时间会缩短吗？”。
    *   系统基于调整后的参数，重新调用机器学习模型进行预测，展示不同策略下的模拟结果。

4.  **LLM解读与可视化**：
    *   LLM将机器学习模型输出的原始数值（如概率0.75，时长12.5小时）转化为易于理解的自然语言解释和总结。
    *   预测结果可以通过图表（如掌握度条形图表示概率，甘特图概念表示时间规划）在前端进行可视化呈现。

### 4. 交互式知识库探索 (RAG驱动)

“知识库探索”标签页允许用户主动探索系统内置的学习资料：

-   **语义搜索**：用户输入问题或关键词，系统通过RAG技术从知识库（如课程讲义、笔记、FAQ文档）中检索最相关的内容片段。
-   **知识点卡片**：展示检索到的知识点详情、定义、示例等，可由LLM进一步解释或总结。
-   **上下文感知引用**：在对话或路径规划中引用知识库内容时，明确标出信息来源，增强可信度。

### 5. 技术实验室与模拟分析 (ML集成演示与结果评估)

此模块不仅向用户科普背后AI技术原理，更核心的是**演示课程中所学的机器学习模型如何被集成应用、模拟不同模型或集成策略在标准场景下的预测效果，并提供对这些模拟实验结果进行初步分析与数据处理的框架**。

#### 核心学习预测引擎概览

-   展示系统中用于“学习效果与进度预测模拟”的核心机器学习模型（例如：线性回归、逻辑回归、决策树、朴素贝叶斯、支持向量机等——**确保这些与课程教学内容对应**）。
-   用户可以选择查看某个模型，LLM将结合预设信息解释该模型的基本原理、**在本学习预测任务中的具体作用（例如，线性回归用于预测时间，逻辑回归用于预测掌握概率）、其典型的输入特征（如学习时长、历史成绩、模块难度等）和输出结果的解读方式**。

#### ML模型集成策略与效果对比 (模拟)

-   **模拟集成配置**：允许用户选择一部分课程中学过的基础机器学习模型，并选择一种或几种简单的**集成学习策略**（如投票法、平均法、简单的Stacking等）来**虚拟构建一个“集成预测器”**。
-   **模拟预测与对比**：
    *   用户可以针对一个**预设的、标准化的学习场景/数据集**（或由用户输入的简化场景参数，如“假设一个中等难度模块，学生每周学习5小时，基础一般”）。
    *   系统将使用用户选择的单个模型、不同参数的同一模型，或配置的“集成预测器”在该场景下进行**模拟预测**。
    *   **对比展示**不同模型或集成策略的预测结果（例如，A模型预测掌握概率0.6，B模型预测0.7，集成预测器预测0.68）。结果可以用表格或简单图表展示。
    *   LLM辅助解读不同模型或策略可能带来的优势、劣势，或解释结果差异的潜在原因（如“决策树可能对这个场景的非线性关系捕捉得更好，但容易过拟合；集成模型通过综合多个模型意见，试图提高稳定性和准确性”）。

#### 模拟结果分析与数据处理 (规划)

-   **结果记录与导出 (规划)**:
    *   设想用户在技术实验室中进行的多次模拟预测（包含不同输入参数、选择的不同模型/集成策略、以及对应的预测输出）的结果可以被记录下来。
    *   提供将这些**模拟实验数据**（例如：场景描述、模型配置、输入特征、预测概率、预测时长等）**导出为CSV或JSON格式**的功能。这使得用户可以将数据导入到外部工具（如Excel, Python Pandas, R）中进行更深入的统计分析、可视化和数据处理。
-   **基本统计与可视化 (规划)**:
    *   对于记录下来的多次模拟结果，系统可提供基本的描述性统计（例如，针对某一预测目标，不同模型预测值的均值、中位数、范围、标准差等）。
    *   提供简单的可视化图表（例如，对比不同模型/策略在同一场景下预测效果的柱状图、箱线图等），帮助用户直观理解结果差异。
-   **偏差分析与讨论 (LLM辅助)**:
    *   引导用户思考模拟预测结果与真实学习情况之间可能存在的偏差（Bias and Variance的概念可以被引入）。
    *   LLM可以辅助讨论影响预测准确性的因素，如模型的局限性、特征选择的重要性、数据质量（对于预设场景）、集成策略的有效性等。例如：“这些预测是基于模型的假设和我们提供的简化数据，实际学习过程会更复杂。思考一下，哪些未被包含的因素可能会影响真实结果？”
    *   *此部分主要体现“对实验结果进行分析与数据处理”的规划和思考，旨在培养用户的数据分析思维，不一定需要实现非常复杂的实时、自动化分析系统，但提供数据导出和引导性分析是关键。*

## 核心技术

*   **大语言模型 (LLM)**：核心驱动力，采用 **百度文心系列大模型 (如 Ernie 4.5 Turbo 128k)**，负责自然语言理解、对话管理、内容生成、逻辑规划、**ML结果解释、分析引导**。
*   **检索增强生成 (RAG)**：结合 **百度文心Embedding模型** 和 **ChromaDB** (或其他向量数据库如FAISS) 向量数据库，增强LLM知识获取能力，为问答和内容生成提供上下文。
*   **机器学习 (预测与模拟引擎)**：
    *   **课程模型集成**：主要集成和实现**课程中讲授的经典机器学习模型**，如：
        *   **回归模型** (用于时间预测等)：线性回归、多项式回归、岭回归、Lasso回归、支持向量回归 (SVR)、决策树回归、随机森林回归、梯度提升回归 (GBRT)。
        *   **分类模型** (用于掌握度预测等)：逻辑回归、K近邻 (KNN)、朴素贝叶斯、决策树分类、随机森林分类、支持向量机 (SVM)、梯度提升分类。
    *   **集成学习方法**：实现课程中可能涉及的集成方法，如：
        *   **投票法 (Voting)**: 用于分类任务，可实现硬投票和软投票。
        *   **平均法 (Averaging)**: 用于回归任务，对多个模型预测结果取平均。
        *   **堆叠法 (Stacking)**: 实现两层结构，基础模型预测结果作为元模型的输入特征。
    *   **模型调用与管理**：使用Scikit-learn构建和训练模型，使用Joblib或Pickle进行模型持久化（保存为`.pkl`或`.joblib`文件），支持动态加载和版本控制（通过文件名或元数据管理）。
    *   **特征工程**：将用户的对话输入（如学习习惯、投入时间）、学习内容特征（如模块难度、知识点数量）、以及可能的历史学习数据，转化为适合机器学习模型输入的数值化特征。
*   **前端**：HTML, CSS (考虑使用 Tailwind CSS, DaisyUI 或类似框架简化样式), JavaScript (Vanilla JS, 或轻量级框架如Vue.js/React的CDN版本, Chart.js for visualization, Marked.js for Markdown渲染)。
*   **后端**：Python, Flask (或 FastAPI，根据对异步和性能的需求选择)。
*   **核心Python库**：
    *   `langchain`, `langchain-community`, `langchain-baidu-ernie` (或直接使用百度SDK): LLM交互、Agent构建、RAG流程。
    *   `scikit-learn`: 机器学习模型实现、训练、评估。
    *   `pandas`: 数据处理，尤其用于特征工程和模拟结果的组织。
    *   `numpy`: 高效数值计算。
    *   `chromadb-client` (或对应向量数据库的库): 向量存储与检索。
    *   `matplotlib`, `seaborn` (可选，主要用于notebooks中的分析，或后端生成图表数据API)。
    *   `python-dotenv`: 环境变量管理。
    *   `flask`, `flask-cors` (如果前后端分离)。
*   **配置管理**：`python-dotenv`加载`.env`文件，`config.py`存储应用配置。

## 系统架构概览

```mermaid
graph LR
    subgraph 用户端 (Browser)
        UI[前端界面 <br/> (HTML, CSS, JavaScript, Chart.js)]
    end

    subgraph 后端服务 (Python: Flask/FastAPI)
        API_Gateway[API网关 <br/> (e.g., /api/chat, /api/predict, /api/techlab/simulate)]

        Router_Agent[LLM智能路由/Agent核心 <br/> (LangChain, Baidu Ernie LLM)]

        subgraph 核心逻辑模块
            DialogManager[对话管理模块]
            PathPlanner[学习路径规划模块 <br/> (learning_planner.py)]
            RAG_Core[RAG检索模块 <br/> (rag_core.py)]
            ML_Predictor[**ML预测与模拟引擎** <br/> (ml_predictor.py - 集成课程ML模型)]
            TechLab_Analyzer[**技术实验室与分析模块** <br/> (tech_lab_analyzer.py - 模拟与结果处理)]
        end

        subgraph 数据与外部服务
            LLM_Service[百度文心LLM API]
            Embedding_Service[百度文心Embedding API]
            VectorDB[向量数据库 <br/> (ChromaDB / FAISS)]
            ML_Model_Store[**机器学习模型库** <br/> (Scikit-learn models: .pkl/.joblib files, model_metadata.json)]
            KB_Docs[知识库文档 <br/> (.md, .txt, .pdf)]
            Sim_Data_Store[模拟结果数据存储 (规划) <br/> (e.g., CSV, JSON files, or simple DB)]
        end
    end

    UI -- HTTP/WebSocket 请求 --> API_Gateway
    API_Gateway -- 路由 --> Router_Agent

    Router_Agent -- 调用 --> DialogManager
    Router_Agent -- 意图: 规划学习路径 --> PathPlanner
    Router_Agent -- 意图: 查询知识 --> RAG_Core
    Router_Agent -- 意图: **学习效果预测/模拟** --> ML_Predictor
    Router_Agent -- 意图: **技术实验室操作/分析** --> TechLab_Analyzer
    Router_Agent -- 与LLM交互 --> LLM_Service

    PathPlanner -- 调用LLM进行规划 --> LLM_Service
    PathPlanner -- 可能查询知识 --> RAG_Core

    RAG_Core -- 生成Embeddings --> Embedding_Service
    RAG_Core -- 存取向量 --> VectorDB
    VectorDB -- 加载自 --> KB_Docs
    RAG_Core -- 构建上下文给LLM --> LLM_Service

    ML_Predictor -- **加载/调用** --> ML_Model_Store
    ML_Predictor -- **执行预测/集成** --> ML_Model_Store
    ML_Predictor -- 预测结果 --> Router_Agent
    Router_Agent -- 结果解释 --> LLM_Service

    TechLab_Analyzer -- **加载模型元数据/调用模型** --> ML_Model_Store
    TechLab_Analyzer -- **执行模拟预测** (可调用 ML_Predictor)
    TechLab_Analyzer -- **处理模拟结果** --> Sim_Data_Store (记录/导出)
    TechLab_Analyzer -- 分析报告/数据 --> Router_Agent
    Router_Agent -- 分析解释 --> LLM_Service
    
    Router_Agent -- 结构化响应 --> API_Gateway
    API_Gateway -- JSON响应 --> UI
```

**架构说明:**

1.  **用户端 (Frontend)**: 用户通过浏览器与前端界面交互，发送请求。
2.  **API网关 (API Gateway)**: 后端Flask/FastAPI应用接收HTTP请求，作为所有服务的入口。
3.  **LLM智能路由/Agent核心 (LLM Router/Agent Core)**: 使用LangChain和百度文心LLM解析用户意图，管理对话流程，并根据意图将任务分发给相应的核心逻辑模块。
4.  **核心逻辑模块 (Core Logic Modules)**:
    *   **对话管理模块**: 维护对话历史和上下文。
    *   **学习路径规划模块**: 结合LLM能力和用户目标生成个性化学习路径。
    *   **RAG检索模块**: 从知识库中检索信息，为LLM提供上下文。
    *   **ML预测与模拟引擎 (ml_predictor.py)**: **核心！** 加载和调用存储在`ML_Model_Store`中的、基于课程所学训练好的机器学习模型（如线性回归、逻辑回归等），执行学习效果和进度的预测与模拟。
    *   **技术实验室与分析模块 (tech_lab_analyzer.py)**: **核心！** 支持用户在技术实验室中选择不同ML模型和集成策略进行模拟预测，对比效果，并对模拟实验结果进行记录、导出（至`Sim_Data_Store`）和初步分析（借助LLM进行解读）。
5.  **数据与外部服务 (Data & External Services)**:
    *   **LLM/Embedding服务**: 调用百度文心API。
    *   **向量数据库**: 存储和检索知识库文档的向量。
    *   **机器学习模型库**: 存放训练好的`.pkl`或`.joblib`模型文件及描述其元数据（如适用特征、模型类型）的`model_metadata.json`。
    *   **知识库文档**: 原始学习资料。
    *   **模拟结果数据存储**: 用于存放技术实验室模拟实验的数据，供用户导出和进一步分析。

## 安装与运行

### 环境准备

1.  **克隆仓库**:
    ```bash
    git clone https://your-repository-url/voyager-ai-learning-navigator.git # 替换为你的仓库URL和项目名
    cd voyager-ai-learning-navigator
    ```
2.  **Python环境**: 推荐 Python 3.8+。强烈建议使用虚拟环境：
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    核心依赖一般包括: `flask`, `python-dotenv`, `langchain`, `langchain-community`, `langchain-baidu-ernie` (或百度官方SDK `erniebot`), `scikit-learn`, `pandas`, `numpy`, `chromadb-client` (或对应向量库)。请根据实际使用的库更新`requirements.txt`。

### 配置

1.  **环境变量**: 复制 `.env.example` 文件为 `.env`，并填入必要的API密钥等信息：
    ```env
    # .env
    BAIDU_API_KEY="your_baidu_ernie_api_key"
    BAIDU_SECRET_KEY="your_baidu_ernie_secret_key" # 部分百度服务可能需要
    # 其他配置，如默认LLM模型名等
    LLM_MODEL_NAME="ernie-4.0-8k" # 示例
    EMBEDDING_MODEL_NAME="ernie-text-embedding" # 示例
    ```
2.  **系统配置 (config.py)**: 在 `config.py` (或主应用 `app.py`中直接定义) 设置路径等：
    ```python
    # config.py (示例)
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # API Keys from .env
    BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
    BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "ernie-4.0-8k")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "ernie-text-embedding")

    # Paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    KNOWLEDGE_BASE_DIR = os.path.join(ROOT_DIR, "knowledge_base")
    CHROMA_PERSIST_DIR = os.path.join(ROOT_DIR, "vector_store")
    MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "models") # 存储 .pkl 文件的目录
    SIMULATION_DATA_DIR = os.path.join(ROOT_DIR, "simulation_data") # 存储模拟实验结果

    # RAG Parameters
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Ensure directories exist
    os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(SIMULATION_DATA_DIR, exist_ok=True)
    ```

### 启动服务

1.  **初始化知识库和检查ML模型 (首次运行或更新后)**:
    *   **知识库处理**: 可能需要一个脚本来处理 `KNOWLEDGE_BASE_DIR` 下的文档，生成Embeddings并存入向量数据库。这可以在应用启动时检查并执行，或者提供一个管理命令。
        ```bash
        # 示例命令 (需要自行实现 manage.py 或类似机制)
        # python manage.py process_kb
        ```
    *   **ML模型**: 确保 `MODEL_SAVE_PATH` 目录中有所需的预训练机器学习模型文件 (如 `.pkl`) 和可能的元数据文件。这些模型应在 `notebooks/` 中训练和导出。
2.  **启动后端Flask/FastAPI服务**:
    ```bash
    # 如果使用Flask (假设主应用文件为 app.py)
    flask run --host=0.0.0.0 --port=5000

    # 或者直接运行Python脚本
    # python app.py
    ```
    服务启动后，默认可以通过 `http://127.0.0.1:5000` 访问 (如果前端也由此服务提供)。如果前后端分离，前端需配置正确的API地址。

## API接口核心

这里列出一些核心API的设想，具体实现时可以调整。

1.  **`/api/chat` (POST)**: 主要的对话交互接口。
    *   **请求体**:
        ```json
        {
            "query": "用户的自然语言输入",
            "session_id": "用户会话ID (用于上下文管理)",
            "current_context": { // 可选，前端传递的当前学习上下文
                "learning_goal": "...",
                "current_path_id": "...",
                "active_module_id": "...",
                "mode": "navigation | prediction_query | tech_lab" // 指示当前用户可能所处的模式
            }
        }
        ```
    *   **响应体**: (结构化JSON，具体字段取决于 `response_type`)
        ```json
        {
            "response_type": "general_answer | path_generated | prediction_prompt | simulation_prompt | tech_lab_info | error",
            "message": "LLM生成的自然语言回复/总结/引导问题",
            "data": { /* 特定类型数据，例如：
                        path_generated: { path_details: [...] },
                        prediction_prompt: { required_features: ["weekly_study_hours", "prior_knowledge_level"] },
                        tech_lab_info: { available_models: [...], available_strategies: [...] }
                     */ }
        }
        ```

2.  **`/api/predict/learning_outcome` (POST)**: 专门用于执行学习效果/进度预测的接口。
    *   **请求体**: (由前端在用户通过对话确认预测意图并提供参数后调用)
        ```json
        {
            "session_id": "...",
            "module_id": "要预测的模块ID或学习目标描述",
            "learning_parameters": { // 用户设定的学习参数，作为ML模型特征
                "weekly_study_hours": 10,
                "study_focus_level": "high", // "high", "medium", "low" -> 会被特征工程转换
                "prior_knowledge_level": "medium", // -> 会被特征工程转换
                // ... 其他课程ML模型所需的、从用户对话中提取或默认的特征 ...
            },
            "target_prediction": "mastery_probability | completion_time" // "mastery_probability" 或 "completion_time"
        }
        ```
    *   **响应体**:
        ```json
        {
            "module_id": "...",
            "prediction_type": "mastery_probability", // 或 "completion_time_hours"
            "predicted_value": 0.82, // 示例：掌握概率
            // "predicted_value": 15.0, // 示例：完成时间 (小时)
            "confidence_interval": [0.75, 0.88], // 可选
            "model_used": "integrated_logistic_regression_v2", // 使用的模型标识
            "llm_interpretation": "根据您每周投入10小时且高度专注，并具备中等先验知识的计划，我们预测您有82%的概率掌握此模块...",
            "raw_model_output": { /* 原始模型输出，供调试或高级分析 */ }
        }
        ```

3.  **`/api/techlab/simulate` (POST)**: 用于技术实验室模拟不同模型/集成策略。
    *   **请求体**:
        ```json
        {
            "session_id": "...",
            "scenario_id": "predefined_scenario_1", // 或自定义场景参数
            "custom_scenario_params": { /* 如果不是预定义场景 */
                "difficulty": "medium",
                "study_hours_week": 5
            },
            "model_configurations": [ // 用户选择要对比的模型或集成策略
                {"type": "single_model", "name": "linear_regression_v1"},
                {"type": "single_model", "name": "decision_tree_regressor_v1"},
                {"type": "ensemble", "name": "voting_regressor_avg", "base_models": ["linear_regression_v1", "svr_v1"]}
            ],
            "prediction_target": "completion_time" // 或 "mastery_probability"
        }
        ```
    *   **响应体**:
        ```json
        {
            "scenario_description": "...",
            "simulation_results": [
                {
                    "configuration_name": "linear_regression_v1",
                    "predicted_value": 10.5,
                    "raw_output": "..."
                },
                {
                    "configuration_name": "decision_tree_regressor_v1",
                    "predicted_value": 9.8,
                    "raw_output": "..."
                },
                {
                    "configuration_name": "voting_regressor_avg",
                    "predicted_value": 10.1,
                    "raw_output": "..."
                }
            ],
            "llm_analysis_summary": "在此模拟场景下，决策树预测的完成时间最短，为9.8小时。线性回归和投票集成预测器结果相近...",
            "exportable_data_reference_id": "sim_run_xyz123" // 可选，用于后续导出完整数据
        }
        ```
4.  **`/api/techlab/export_simulation_data` (GET)**:
    *   **请求参数**: `simulation_id=sim_run_xyz123`
    *   **响应**: CSV 或 JSON 文件下载。

5.  **其他辅助API**: 如 `/api/knowledge/search` (GET), `/api/techlab/models` (GET, 获取可用模型列表和元数据)。

## 项目结构 (建议)

```
.
├── app.py                     # Flask/FastAPI主应用与核心路由
├── config.py                  # 系统配置 (API密钥, 路径, LLM模型名等)
├── agents/                    # (可选) LangChain Agents 定义
│   └── main_dialog_agent.py
├── core/                      # 核心业务逻辑模块
│   ├── llm_services.py        # 封装百度文心LLM和Embedding API调用
│   ├── rag_core.py            # RAG核心逻辑 (文档加载, embedding, 检索)
│   ├── learning_planner.py    # 学习路径规划逻辑
│   ├── feature_engineering.py # 用于将用户输入和上下文转换为ML模型特征
│   ├── ml_predictor.py        # **机器学习预测与模拟引擎** (加载模型, 执行预测, 集成逻辑)
│   └── tech_lab_analyzer.py   # **技术实验室模拟与分析逻辑** (处理模拟请求, 结果聚合, 准备导出数据)
├── models/                    # 存放训练好的机器学习模型 (.pkl 或 .joblib) 和元数据
│   ├── linear_regression_time_v1.pkl
│   ├── logistic_regression_mastery_v1.pkl
│   ├── decision_tree_time_v1.pkl
│   └── model_metadata.json    # 描述每个模型、期望特征、用途、版本等
├── knowledge_base/            # RAG源学习文档 (如 .md, .txt, .pdf 文件)
│   ├── module1_basics.md
│   └── module2_advanced_topics.pdf
├── vector_store/              # ChromaDB (或选定向量库) 的持久化数据目录
├── simulation_data/           # (规划) 技术实验室模拟实验结果的存储 (如CSV, JSON)
│   └── sim_run_xyz123.csv
├── static/                    # 前端静态文件 (如果Flask同时提供前端)
│   ├── js/
│   │   └── app.js             # 前端主逻辑
│   │   └── chart.min.js       # Chart.js库
│   ├── css/
│   │   └── style.css          # 自定义样式
│   └── favicon.ico
├── templates/                 # 前端HTML模板 (如果Flask同时提供前端)
│   └── index.html
├── notebooks/                 # (强烈推荐) Jupyter notebooks 用于数据探索、模型训练、评估、实验和结果分析
│   ├── 00_environment_setup.ipynb
│   ├── 01_data_preprocessing_and_feature_exploration.ipynb # 数据清洗、特征工程探索
│   ├── 02_model_training_completion_time.ipynb             # 训练各类回归模型 (线性回归, 决策树等)
│   ├── 03_model_training_mastery_probability.ipynb         # 训练各类分类模型 (逻辑回归, 决策树等)
│   ├── 04_model_ensemble_and_evaluation.ipynb              # 模型集成实验 (Voting, Stacking) 和评估
│   ├── 05_simulation_result_analysis_template.ipynb        # 分析和可视化 `simulation_data` 中导出的结果
│   └── utils/                                              # Notebooks中可能用到的辅助函数
│       └── plot_helpers.py
├── .env                       # 存储环境变量 (API密钥等 - 不提交到git)
├── .env.example               # 环境变量模板
├── requirements.txt           # Python依赖包列表
├── Dockerfile                 # (可选) 用于容器化部署
├── docker-compose.yml         # (可选) 用于容器化部署
└── README.md                  # 本文件
```

**关键模块说明:**

*   `core/ml_predictor.py`: 核心模块，负责加载`models/`目录下的预训练机器学习模型。它将接收来自用户对话（经过`feature_engineering.py`处理）的特征，调用相应的模型进行预测（如学习时间、掌握概率）。它也应包含实现不同集成策略（如投票、平均、Stacking）的逻辑。
*   `core/tech_lab_analyzer.py`: 支持“技术实验室”功能。它会根据用户请求，调用`ml_predictor.py`中的模型或集成逻辑在预设/自定义场景下进行模拟。关键在于它会收集这些模拟的输入、配置和输出，将结果存储到`simulation_data/`目录（例如CSV文件），并准备数据供用户导出或进行初步的LLM辅助分析。
*   `models/`: **至关重要**。这里存放所有课程中学习并训练好的机器学习模型文件（如`linear_regression_time_v1.pkl`）以及一个`model_metadata.json`文件，该文件描述了每个模型的信息（例如，它预测什么，需要哪些输入特征，特征的期望格式/范围，版本号等），便于`ml_predictor.py`和`tech_lab_analyzer.py`动态加载和使用。
*   `notebooks/`: **强烈建议**。这里是进行所有机器学习模型开发、训练、评估和实验的地方。
    *   `01_...ipynb`：用于探索性数据分析和特征工程。
    *   `02_...ipynb` 和 `03_...ipynb`：分别针对不同预测目标（如时间、掌握度）训练和保存（导出到`models/`目录）课程中讲解的各种ML模型。
    *   `04_...ipynb`：专门用于实验不同的模型集成方法，并评估其性能，最终的集成模型也应保存到`models/`。
    *   `05_...ipynb`：提供一个模板，用于加载和分析从`simulation_data/`目录中导出的**模拟实验结果**。这直接对应了项目需求中“对实验结果进行分析与数据处理”的部分。
*   `simulation_data/`: 专门用于存储技术实验室中进行的模拟实验的详细数据，方便用户后续进行更深入的离线分析。

## 未来展望

*   **更精细化的机器学习模型**：引入更多上下文特征（如用户历史学习行为的详细数据、知识点之间的依赖关系图），使用更高级的机器学习模型或深度学习模型（如果适用且课程有涉及），探索更复杂的集成方法（如动态模型选择），以持续提升预测的准确性和个性化程度。
*   **在线学习与模型迭代 (Online Learning & Model Retraining)**：设计机制收集用户实际的学习效果和进度数据（在用户同意的前提下），定期或在数据达到一定规模时，使用这些新数据重新训练或微调机器学习模型，实现模型的持续优化。
*   **严格的模型评估与可解释性增强**：建立更完善、自动化的模型评估流水线，定期对预测模型的性能进行严格验证。引入更多模型可解释性技术（如SHAP, LIME），帮助用户理解为什么模型会做出某个特定的预测。
*   **交互式结果分析与可视化工具**：在前端提供更丰富的、交互式的图表和工具，让用户能更直观地探索和理解模拟预测结果，例如，通过拖动滑块调整输入参数实时查看预测变化，或者对多次模拟结果进行多维度对比。
*   **基于预测的自适应路径调整**：根据ML模型的预测结果（如预测用户可能在某模块遇到困难）和用户的实际学习进度，动态地、智能地向用户推荐调整学习路径、补充额外学习资料或改变学习策略。
*   **支持更广泛的知识源与学习场景**：扩展知识库的接入能力，支持更多格式的文档或在线资源。将预测模型泛化到更多的学习任务和学科领域。
*   **引入强化学习进行策略优化**：探索使用强化学习来优化学习路径推荐或学习干预策略，最大化用户的学习效率和效果。

## 贡献

欢迎对此项目感兴趣的开发者一同贡献！如果您有任何建议、发现bug或想要添加新功能，请先创建一个Issue进行讨论，或者直接提交Pull Request。

在贡献代码前，请确保：
1.  代码风格与项目现有代码保持一致。
2.  添加了必要的注释和文档。
3.  相关的测试用例已添加并通过。

## 许可证

本项目采用 [MIT许可证](https://opensource.org/licenses/MIT)。