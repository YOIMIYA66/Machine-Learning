# AI机器学习助手 Pro

## 项目介绍

这是一个集成了RAG检索增强生成和机器学习模型的智能助手系统，可以回答机器学习相关问题，并提供模型训练、预测、分析和可视化功能。

## 主要功能

### 1. 模型集成

- **投票集成 (Voting)**：组合多个模型的预测结果，分类问题可使用硬投票或软投票，回归问题使用加权平均
- **堆叠集成 (Stacking)**：使用元学习器组合基础模型的预测结果，可以学习更复杂的组合方式
- **装袋集成 (Bagging)**：对同一模型进行多次训练，每次使用数据的不同子集，减少过拟合风险

示例用法：
```python
from ml_models import create_ensemble_model

# 创建投票集成分类器
ensemble = create_ensemble_model(
    base_models=['random_forest_classifier_air_quality', 'decision_tree_air_quality'],
    ensemble_type='voting',
    save_name='my_voting_ensemble'
)
```

### 2. 模型版本控制

- **版本创建**：为模型创建版本标记，保存模型状态和元数据
- **版本列表**：查看模型的所有历史版本
- **版本加载**：指定加载特定版本的模型

示例用法：
```python
from ml_models import save_model_with_version, list_model_versions

# 创建模型版本
version_info = save_model_with_version(
    model=model_object,
    model_name='my_model',
    metadata={'description': '优化后的版本', 'accuracy': 0.95}
)

# 列出模型所有版本
versions = list_model_versions('my_model')
```

### 3. 自动模型选择

- **超参数优化**：使用网格搜索自动寻找最佳超参数
- **多模型比较**：比较多种算法的性能，选择最适合任务的模型
- **交叉验证**：使用K折交叉验证确保模型的泛化能力

示例用法：
```python
from ml_models import auto_model_selection

# 自动选择最佳模型
result = auto_model_selection(
    data_path='my_data.csv',
    target_column='target',
    categorical_columns=['cat1', 'cat2'],
    numerical_columns=['num1', 'num2']
)
```

### 4. 模型解释

- **特征重要性**：分析各特征对模型预测的影响程度
- **特征贡献**：计算特定输入数据中各特征对预测结果的贡献
- **可视化解释**：生成特征重要性条形图、雷达图等可视化解释

示例用法：
```python
from ml_models import explain_model_prediction

# 解释模型预测
explanation = explain_model_prediction(
    model_name='my_model',
    input_data={'feature1': 10, 'feature2': 20}
)
```

### 5. 模型比较

- **多指标比较**：同时比较多个模型在多个指标上的表现
- **可视化对比**：生成对比条形图，直观展示模型性能差异
- **最佳模型推荐**：自动推荐最适合特定任务的模型

示例用法：
```python
from ml_models import compare_models

# 比较多个模型
comparison = compare_models(
    model_names=['model1', 'model2', 'model3'],
    test_data_path='test_data.csv',
    target_column='target'
)
```

### 6. 丰富的可视化

- **多种图表类型**：条形图、折线图、饼图、散点图、热力图、雷达图、气泡图等
- **特征分析可视化**：特征重要性、特征分布、相关性分析等
- **模型评估可视化**：混淆矩阵、ROC曲线、精确率-召回率曲线等
- **聚类结果可视化**：支持PCA和t-SNE降维方法

### 7. API接口

- **集成模型API**：创建和管理集成模型的API端点
- **自动选择API**：自动选择最佳模型的API端点
- **模型解释API**：解释模型预测结果的API端点
- **模型比较API**：比较多个模型性能的API端点
- **版本控制API**：管理模型版本的API端点
- **聊天接口 (`/api/chat`)**：核心交互接口，根据用户查询类型（通用知识、ML操作、数据分析、预测等）智能路由到RAG、ML Agent或直接LLM。

### 8. RAG检索增强生成

- **知识库**：系统内置了机器学习相关的文档知识库，用于增强LLM的回答。
- **文档加载与向量化**：支持多种文档格式（PDF, DOCX, TXT, JSON, CSV），自动进行文本分割和向量化存储。
- **检索与生成**：根据用户查询从知识库中检索相关文档片段，结合LLM生成更准确、更具上下文的回答。

### 9. 增强型ML Agent与高级特征分析

- **智能路由**：ML Agent能够理解用户意图，自动调用相应的机器学习功能（训练、预测、评估、比较等）。
- **高级数据分析**：提供基础、综合和高级（特征重要性、特征交互、特征稳定性）的数据和特征分析能力，帮助用户理解数据。
- **可视化支持**：分析结果支持多种可视化图表生成。

### 10. RAG与机器学习模型集成

- **预测集成**：对于包含预测需求的查询，系统尝试从知识库中提取预测所需信息，查找合适的已训练模型进行预测，并将预测结果与RAG的知识回答相结合。
- **分析结果集成**：将模型训练、评估、特征分析等结果与RAG生成的解释性文本集成，提供更全面的信息。

### 11. 大语言模型与Embedding模型

- **核心LLM**：使用百度文心系列大语言模型（如 Ernie 4.5 Turbo 128k）进行文本生成和理解。
- **Embedding模型**：使用百度文心系列Embedding模型（如 BGE-Large-zh）进行文本向量化，支持RAG检索和语义匹配。

### 12. 配置说明

- **`.env` 文件**：用于配置敏感信息和关键参数，如 `AI_STUDIO_API_KEY`。
- **`config.py` 文件**：包含系统运行时的各种配置，如LLM和Embedding模型名称、知识库路径、ChromaDB存储路径、文本分割参数、JSON解析JQ Schema等。用户可以根据需要修改此文件来调整系统行为。

## 安装和使用

1. 确保已安装依赖库：
```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

2. 导入所需模块：
```python
from ml_models import train_model, predict, list_available_models, create_ensemble_model
from ml_agents import query_ml_agent
```

3. 开始使用各项功能，详见示例代码。

## 注意事项

- 模型保存在`ml_models`目录中
- 模型版本保存在`ml_models/{model_name}_versions`子目录中
- 对于大型数据集，建议使用`auto_model_selection`功能前先进行特征选择