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