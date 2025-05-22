# ml_models.py
import os
import pickle
import numpy as np
import pandas as pd
import json
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    classification_report, confusion_matrix
)

# 模型保存目录
MODELS_DIR = "ml_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 模型映射表，便于通过名称查找模型
MODEL_TYPES = {
    # 回归模型
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
    
    # 分类模型
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest_classifier": RandomForestClassifier,
    "knn_classifier": KNeighborsClassifier,
    "svm_classifier": SVC,
    "naive_bayes": MultinomialNB,
    
    # 聚类模型
    "kmeans": KMeans,
    
    # 集成模型类型
    "voting_classifier": VotingClassifier,
    "voting_regressor": VotingRegressor,
    "stacking_classifier": StackingClassifier,
    "stacking_regressor": StackingRegressor,
    "bagging_classifier": BaggingClassifier,
    "bagging_regressor": BaggingRegressor
}

# 模型类别分组，便于前端展示和选择
MODEL_CATEGORIES = {
    "regression": ["linear_regression"],
    "classification": ["logistic_regression", "knn_classifier", "decision_tree", "svm_classifier", "naive_bayes"],
    "clustering": ["kmeans"]
}

# 模型详细信息，包含图标和描述
MODEL_DETAILS = {
    "linear_regression": {
        "display_name": "线性回归模型",
        "icon_class": "fa-chart-line",
        "description": "线性回归是一种基本的统计模型，用于预测连续型变量。它通过建立自变量与因变量之间的线性关系，找出最佳拟合直线，适用于简单的数值预测任务。"
    },
    "logistic_regression": {
        "display_name": "逻辑回归模型",
        "icon_class": "fa-code-branch",
        "description": "逻辑回归是一种用于二分类问题的统计模型，通过Sigmoid函数将线性模型的输出转换为概率值。它计算效率高，易于实现，适合处理线性可分的分类问题。"
    },
    "knn_classifier": {
        "display_name": "K-近邻法预测模型(KNN)",
        "icon_class": "fa-project-diagram",
        "description": "K-近邻算法是一种基于实例的学习方法，通过计算新样本与训练集中所有样本的距离，选取最近的K个邻居进行投票或平均，从而进行分类或回归预测。"
    },
    "decision_tree": {
        "display_name": "决策树",
        "icon_class": "fa-sitemap",
        "description": "决策树是一种树形结构的分类模型，通过一系列条件判断将数据划分为不同类别。它直观易懂，可解释性强，能够处理非线性关系，但容易过拟合。"
    },
    "svm_classifier": {
        "display_name": "向量机模型",
        "icon_class": "fa-vector-square",
        "description": "支持向量机(SVM)是一种强大的分类算法，通过寻找最优超平面来区分不同类别的数据点。它在高维空间中表现良好，可以通过核函数处理非线性问题，适合小型复杂数据集。"
    },
    "naive_bayes": {
        "display_name": "朴素贝叶斯分类器",
        "icon_class": "fa-percentage",
        "description": "朴素贝叶斯是基于贝叶斯定理的概率分类器，假设特征之间相互独立。它训练速度快，需要较少的训练数据，特别适合文本分类和多分类问题，但对特征相关性较强的数据效果可能不佳。"
    },
    "kmeans": {
        "display_name": "K-Means 模型",
        "icon_class": "fa-object-group",
        "description": "K-Means是一种常用的聚类算法，通过迭代优化将数据点分配到K个簇中。它实现简单，计算效率高，适合大规模数据集的无监督学习，但对初始聚类中心敏感，且难以处理非球形簇。"
    }
}

# 模型默认参数
DEFAULT_MODEL_PARAMS = {
    "linear_regression": {},
    "logistic_regression": {"max_iter": 1000, "C": 1.0},
    "decision_tree": {"max_depth": 5},
    "random_forest_classifier": {"n_estimators": 100, "max_depth": 5},
    "random_forest_regressor": {"n_estimators": 100, "max_depth": 5},
    "knn_classifier": {"n_neighbors": 5},
    "svm_classifier": {"C": 1.0, "kernel": "rbf"},
    "naive_bayes": {"alpha": 1.0},
    "kmeans": {"n_clusters": 3, "random_state": 42}
}

# 数据预处理函数
def preprocess_data(
    data: pd.DataFrame,
    target_column: str,
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None,
    test_size: float = 0.2,
    scale_data: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    预处理数据用于机器学习模型训练。
    
    Args:
        data: 输入的DataFrame
        target_column: 目标列名称
        categorical_columns: 需要进行编码的分类特征列表
        numerical_columns: 需要进行标准化的数值特征列表
        test_size: 测试集比例
        scale_data: 是否标准化数值特征
        
    Returns:
        X_train: 训练特征
        X_test: 测试特征
        y_train: 训练目标
        y_test: 测试目标
        preprocessors: 预处理器字典，包含编码器和缩放器
    """
    # 拷贝数据，避免修改原始DataFrame
    df = data.copy()
    
    # 准备预处理器字典
    preprocessors = {}
    
    # 处理分类特征
    if categorical_columns:
        encoders = {}
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
        preprocessors['label_encoders'] = encoders
    
    # 分离特征和目标
    if target_column in df.columns:
        y = df[target_column].values
        X = df.drop(columns=[target_column])
    else:
        raise ValueError(f"目标列 '{target_column}' 未在数据中找到")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 标准化数值特征
    if scale_data and numerical_columns:
        scaler = StandardScaler()
        numeric_cols_present = [col for col in numerical_columns if col in X.columns]
        if numeric_cols_present:
            X_train_numeric = X_train[numeric_cols_present]
            X_train[numeric_cols_present] = scaler.fit_transform(X_train_numeric)
            
            X_test_numeric = X_test[numeric_cols_present]
            X_test[numeric_cols_present] = scaler.transform(X_test_numeric)
            
            preprocessors['scaler'] = scaler
    
    return X_train.values, X_test.values, y_train, y_test, preprocessors

# 训练机器学习模型的函数
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
    print(f"[INFO] 开始训练 {model_type} 模型...")
    # 加载数据
    if isinstance(data, str):
        if data.endswith('.csv'):
            data = pd.read_csv(data, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
        elif data.endswith( ('.xls', '.xlsx') ):
            data = pd.read_excel(data, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
        elif data.endswith('.json'):
            data = pd.read_json(data)
            data = data.fillna('')
        else:
            raise ValueError("支持的文件格式: CSV, Excel (.xls, .xlsx), JSON (.json)")
        
        # 检查数据是否为空
        if data.empty:
            raise ValueError("加载的数据为空，请检查文件内容")
            
        # 处理缺失值
        if data.isna().any().any():
            print("警告：数据中存在缺失值，将使用适当的策略处理")
            # 对数值列使用均值填充，对分类列使用众数填充
            for col in data.columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    data[col] = data[col].fillna(data[col].mean())
                else:
                    data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else '')
    
    # 检查模型类型是否支持
    if model_type not in MODEL_TYPES:
        error_msg = f"不支持的模型类型: {model_type}. 支持的类型有: {', '.join(MODEL_TYPES.keys())}"
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)
    
    # 预处理数据
    X_train, X_test, y_train, y_test, preprocessors = preprocess_data(
        data,
        target_column,
        categorical_columns,
        numerical_columns,
        test_size
    )
    
    # 创建并训练模型
    model_cls = MODEL_TYPES[model_type]
    model = model_cls(**(model_params or {}))
    model.fit(X_train, y_train)
    
    # 进行预测
    y_pred = model.predict(X_test)
    
    # 评估模型
    metrics = {}
    if model_type in ["linear_regression", "random_forest_regressor"]:
        # 回归模型评估
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
    else:
        # 分类模型评估
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
    
    # 生成模型名称
    if model_name is None:
        model_name = f"{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
    # 保存模型和预处理器
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, preprocessors), f)
    
    return {
        "model_name": model_name,
        "model_type": model_type,
        "model_path": model_path,
        "target_column": target_column,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "metrics": metrics,
        "feature_importance": getattr(model, "feature_importances_", None)
    }

# 加载模型函数
def load_model(model_name: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    从文件加载机器学习模型和预处理器
    
    返回:
        (模型对象, 预处理器, 元数据)
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, "rb") as f:
        loaded_data = pickle.load(f)
        
    # 处理不同格式的模型文件
    if isinstance(loaded_data, tuple):
        if len(loaded_data) == 2:
            # 旧格式: (model, preprocessors)
            model, preprocessors = loaded_data
            metadata = {}
        elif len(loaded_data) == 3:
            # 新格式: (model, preprocessors, metadata)
            model, preprocessors, metadata = loaded_data
        else:
            raise ValueError(f"未知的模型文件格式，元组长度: {len(loaded_data)}")
    else:
        # 单一对象格式
        model = loaded_data
        preprocessors = {}
        metadata = {}
    
    return model, preprocessors, metadata

# 预测函数 - 使用已训练模型进行预测
def predict(
    model_name: str,
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    target_column: str = None
) -> Dict[str, Any]:
    """
    使用保存的模型进行预测
    
    Args:
        model_name: 模型名称
        input_data: 输入数据（DataFrame或字典或字典列表）
        target_column: 目标列名（如果在输入数据中存在）
        
    Returns:
        Dictionary包含预测结果和相关信息
    """
    # 将输入数据转换为DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    # 保存副本
    input_df = input_data.copy()
    
    # 加载模型和预处理器
    model, preprocessors, _ = load_model(model_name)
    
    # 应用预处理变换
    if 'label_encoders' in preprocessors:
        for col, encoder in preprocessors['label_encoders'].items():
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError:
                    # 处理未知类别
                    print(f"警告: 列 '{col}' 包含模型训练期间未见过的类别。将使用默认值0。")
                    input_df[col] = 0
    
    # 标准化数值特征
    if 'scaler' in preprocessors and hasattr(preprocessors['scaler'], 'transform'):
        numeric_cols = [col for col in input_df.columns if col in preprocessors['scaler'].feature_names_in_]
        if numeric_cols:
            input_df[numeric_cols] = preprocessors['scaler'].transform(input_df[numeric_cols])
    
    # 删除目标列（如果存在）
    if target_column and target_column in input_df.columns:
        actual_values = input_df[target_column].values
        X = input_df.drop(columns=[target_column]).values
    else:
        actual_values = None
        X = input_df.values
    
    # 进行预测
    predictions = model.predict(X)
    
    # 准备返回结果
    result = {
        "model_name": model_name,
        "predictions": predictions.tolist(),
        "input_data": input_data.to_dict('records')
    }
    
    # 如果有实际值，计算评估指标
    if actual_values is not None:
        if hasattr(model, "predict_proba"):
            result["is_classification"] = True
            result["accuracy"] = accuracy_score(actual_values, predictions)
        else:
            result["is_classification"] = False
            result["mse"] = mean_squared_error(actual_values, predictions)
            result["r2"] = r2_score(actual_values, predictions)
    
    return result

# 列出所有可用模型
def list_available_models() -> List[Dict[str, Any]]:  # 修改返回类型提示
    """
    列出所有可用的模型，仅返回可JSON序列化的元数据。
    """
    models_metadata = []  # 改名为 models_metadata
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            model_name = os.path.splitext(filename)[0]
            try:
                # 我们仍然需要加载模型来获取其类型和可能的元数据
                # 但我们不会将模型对象本身放入返回的字典中
                model_obj, preprocessors, metadata = load_model(model_name)  # load_model现在返回3个值

                # 从模型对象推断类型 (如果元数据中没有)
                model_type_from_obj = next((k for k, v in MODEL_TYPES.items() if isinstance(model_obj, v)), "unknown")

                # 优先使用元数据中的类型，其次是推断的类型
                model_type = metadata.get("model_type", model_type_from_obj)

                # 准备模型元数据字典
                detail_info = MODEL_DETAILS.get(model_type, {})
                model_info = {
                    "name": model_name,
                    "type": model_type,
                    "path": os.path.join(MODELS_DIR, filename),
                    "params": metadata.get("model_params", {}),
                    "display_name": detail_info.get("display_name", model_name.replace("_", " ").title()),
                    "icon_class": detail_info.get("icon_class", "fa-brain"),
                    "description": detail_info.get("description", f"A {model_type} model."),
                    "created_at": metadata.get("created_at", None),
                    "internal_name": model_name, # 与前端 data-model-name 对应
                }

                # 确保所有值都是可序列化的
                for key, value in list(model_info.items()):  # 使用list复制，因为可能在迭代中修改
                    if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        model_info[key] = str(value)  # 将不可序列化的值转为字符串

                if metadata:  # 将其他元数据项也加入，确保它们可序列化
                    for meta_key, meta_value in metadata.items():
                        if meta_key not in model_info:  # 避免覆盖已有的键
                            if isinstance(meta_value, (str, int, float, bool, list, dict, type(None))):
                                model_info[meta_key] = meta_value
                            else:
                                model_info[meta_key] = str(meta_value)

                models_metadata.append(model_info)
            except Exception as e:
                # 使用 app.logger (如果已配置) 或 print
                print(f"加载或处理模型元数据 {model_name} 时出错: {e}")
                # 可以选择跳过此模型或添加一个带错误标记的条目
                models_metadata.append({
                    "name": model_name,
                    "type": "error",
                    "path": os.path.join(MODELS_DIR, filename),
                    "error_message": str(e)
                })
    return models_metadata

# 根据问题描述选择合适的模型
def select_model_for_task(task_description: str) -> Optional[str]:
    """
    基于任务描述选择最合适的模型
    """
    available_models = list_available_models()
    if not available_models:
        return None
    
    # 这里可以实现更智能的模型选择逻辑
    # 例如使用关键词匹配或基于任务特征进行选择
    
    # 关键词映射
    keywords = {
        "回归": ["linear_regression", "random_forest_regressor"],
        "预测数值": ["linear_regression", "random_forest_regressor"],
        "价格预测": ["linear_regression", "random_forest_regressor"],
        "销量预测": ["linear_regression", "random_forest_regressor"],
        "分类": ["logistic_regression", "decision_tree", "random_forest_classifier"],
        "是否": ["logistic_regression", "decision_tree", "random_forest_classifier"],
        "风险识别": ["logistic_regression", "random_forest_classifier"],
        "信贷风险": ["decision_tree", "random_forest_classifier"],
        "健康风险": ["logistic_regression", "random_forest_classifier"],
        "客户流失": ["logistic_regression", "random_forest_classifier"]
    }
    
    # 根据任务描述匹配关键词
    preferred_types = []
    for keyword, model_types in keywords.items():
        if keyword in task_description:
            preferred_types.extend(model_types)
    
    # 如果找到匹配的模型类型，选择该类型的第一个可用模型
    if preferred_types:
        for model_info in available_models:
            if model_info["type"] in preferred_types:
                return model_info["name"]
    
    # 如果没有找到匹配的模型，返回第一个可用模型
    return available_models[0]["name"] if available_models else None 

# 模型版本控制和管理
def save_model_with_version(model, model_name, preprocessors=None, metadata=None, version=None):
    """
    保存模型并进行版本管理
    
    Args:
        model: 模型对象
        model_name: 模型基本名称
        preprocessors: 预处理器字典
        metadata: 模型元数据
        version: 版本号(可选)，如不提供则使用时间戳
        
    Returns:
        包含版本信息的字典
    """
    if version is None:
        # 自动生成版本号
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 版本文件夹
    version_dir = os.path.join(MODELS_DIR, f"{model_name}_versions")
    os.makedirs(version_dir, exist_ok=True)
    
    # 完整版本模型名称
    versioned_model_name = f"{model_name}_v{version}"
    model_path = os.path.join(version_dir, f"{versioned_model_name}.pkl")
    
    # 保存模型和预处理器
    with open(model_path, "wb") as f:
        pickle.dump((model, preprocessors or {}, metadata or {}), f)
    
    # 保存版本元数据
    version_info = {
        "model_name": model_name,
        "version": version,
        "timestamp": datetime.datetime.now().isoformat(),
        "path": model_path,
    }
    
    # 如果提供了元数据，将其合并
    if metadata:
        version_info.update(metadata)
    
    # 保存版本信息
    version_info_path = os.path.join(version_dir, f"{versioned_model_name}_info.json")
    with open(version_info_path, "w") as f:
        json.dump(version_info, f, indent=2)
    
    return version_info

def list_model_versions(model_name):
    """
    列出模型的所有版本
    
    Args:
        model_name: 模型基本名称
        
    Returns:
        版本信息列表
    """
    version_dir = os.path.join(MODELS_DIR, f"{model_name}_versions")
    if not os.path.exists(version_dir):
        return []
    
    versions = []
    for filename in os.listdir(version_dir):
        if filename.endswith("_info.json"):
            with open(os.path.join(version_dir, filename), "r") as f:
                version_info = json.load(f)
                versions.append(version_info)
    
    # 按时间戳排序
    versions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return versions

def load_model_version(model_name, version):
    """
    加载特定版本的模型
    
    Args:
        model_name: 模型基本名称
        version: 版本号
        
    Returns:
        (模型, 预处理器, 元数据)
    """
    version_dir = os.path.join(MODELS_DIR, f"{model_name}_versions")
    model_path = os.path.join(version_dir, f"{model_name}_v{version}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型版本不存在: {model_path}")
    
    with open(model_path, "rb") as f:
        model, preprocessors, metadata = pickle.load(f)
    
    return model, preprocessors, metadata

# 模型集成功能
def create_ensemble_model(
    base_models,
    ensemble_type='voting',
    weights=None,
    final_estimator=None,
    meta_features=False,
    save_name=None
):
    """
    创建集成模型
    
    Args:
        base_models: 基础模型列表（模型名称或(名称,模型)元组）
        ensemble_type: 集成类型，'voting'|'stacking'|'bagging'
        weights: 各模型权重（用于voting）
        final_estimator: 最终估计器（用于stacking）
        meta_features: 是否使用元特征（用于stacking）
        save_name: 保存集成模型的名称
        
    Returns:
        集成模型对象，以及相关信息
    """
    # 加载模型和预处理器
    model_list = []
    is_classifier = True
    preprocessors_dict = {}
    metadata_dict = {}
    
    for item in base_models:
        try:
            if isinstance(item, tuple) and len(item) == 2:
                # 已提供(名称,模型)元组
                model_name, model = item
                _, preprocessors, metadata = load_model(model_name)
            else:
                # 只提供了模型名称
                model_name = item
                model, preprocessors, metadata = load_model(model_name)
            
            # 判断是分类还是回归
            if not hasattr(model, "predict_proba"):
                is_classifier = False
            
            # 确保模型名称是字符串类型
            if not isinstance(model_name, str):
                model_name = str(model_name)
                
            model_list.append((model_name, model))
            preprocessors_dict[model_name] = preprocessors
            metadata_dict[model_name] = metadata
        except Exception as e:
            print(f"加载模型 {model_name} 时出错: {e}")
            raise ValueError(f"无法加载模型 {model_name}: {str(e)}")
    
    # 创建集成模型
    ensemble_model = None
    if ensemble_type == 'voting':
        if is_classifier:
            ensemble_model = VotingClassifier(
                estimators=model_list,
                voting='soft' if all(hasattr(m[1], "predict_proba") for m in model_list) else 'hard',
                weights=weights
            )
        else:
            ensemble_model = VotingRegressor(
                estimators=model_list,
                weights=weights
            )
    elif ensemble_type == 'stacking':
        if is_classifier:
            ensemble_model = StackingClassifier(
                estimators=model_list,
                final_estimator=final_estimator,
                passthrough=meta_features
            )
        else:
            ensemble_model = StackingRegressor(
                estimators=model_list,
                final_estimator=final_estimator,
                passthrough=meta_features
            )
    elif ensemble_type == 'bagging':
        # 对于bagging，我们使用第一个模型作为基础估计器
        base_estimator = model_list[0][1]
        if is_classifier:
            ensemble_model = BaggingClassifier(base_estimator=base_estimator)
        else:
            ensemble_model = BaggingRegressor(base_estimator=base_estimator)
    else:
        raise ValueError(f"不支持的集成类型: {ensemble_type}")
    
    # 创建集成元数据
    ensemble_metadata = {
        'ensemble_type': ensemble_type,
        'base_models': [m[0] for m in model_list],
        'is_classifier': is_classifier,
        'weights': weights,
        'description': f"{ensemble_type.capitalize()} 集成模型，基于 {', '.join([m[0] for m in model_list])}"
    }
    
    # 保存集成模型
    if save_name:
        ensemble_name = save_name
    else:
        ensemble_name = f"{ensemble_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建组合预处理器
    ensemble_preprocessors = {
        'base_model_preprocessors': preprocessors_dict
    }
    
    model_path = os.path.join(MODELS_DIR, f"{ensemble_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((ensemble_model, ensemble_preprocessors, ensemble_metadata), f)
    
    return {
        'model': ensemble_model,
        'preprocessors': ensemble_preprocessors,
        'metadata': ensemble_metadata,
        'model_name': ensemble_name,
        'model_path': model_path
    }

# 自动模型选择和超参数优化
def auto_model_selection(
    data_path,
    target_column,
    categorical_columns=None,
    numerical_columns=None,
    cv=5,
    metric='auto',
    models_to_try=None
):
    """
    自动选择最佳模型和参数
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名
        categorical_columns: 分类特征列表
        numerical_columns: 数值特征列表
        cv: 交叉验证折数
        metric: 评估指标，'auto'表示自动选择
        models_to_try: 要尝试的模型类型列表，None表示尝试所有合适的模型
        
    Returns:
        最佳模型及其信息
    """
    # 加载数据
    if data_path.endswith('.csv'):
        data = pd.read_csv(data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    elif data_path.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    elif data_path.endswith('.json'):
        data = pd.read_json(data_path)
        data = data.fillna('')
    else:
        raise ValueError("目前只支持CSV、Excel和JSON文件格式")
        
    # 确保目标列存在于数据中
    if target_column not in data.columns:
        raise ValueError(f"目标列 '{target_column}' 不在数据中。可用列: {', '.join(data.columns)}")
        
    # 处理缺失值
    if data.isna().any().any():
        print("警告：数据中存在缺失值，将使用适当的策略处理")
        # 对数值列使用均值填充，对分类列使用众数填充
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else '')
    
    # 预处理数据
    X_train, X_test, y_train, y_test, preprocessors = preprocess_data(
        data,
        target_column,
        categorical_columns,
        numerical_columns
    )
    
    # 自动判断是分类还是回归任务
    unique_values = np.unique(y_train)
    is_classification = len(unique_values) < 20  # 简单判断，可优化
    
    # 定义要尝试的模型和参数网格
    if models_to_try is None:
        if is_classification:
            models_to_try = ['logistic_regression', 'decision_tree', 'random_forest_classifier', 'knn_classifier', 'svm_classifier']
        else:
            models_to_try = ['linear_regression', 'decision_tree', 'random_forest_regressor']
    
    # 为每个模型定义参数网格
    param_grids = {
        'logistic_regression': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        },
        'decision_tree': {
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'random_forest_classifier': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        'random_forest_regressor': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        },
        'knn_classifier': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'svm_classifier': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf']
        },
        'linear_regression': {
            'fit_intercept': [True, False],
            'positive': [False, True]
        }
    }
    
    # 设置评估指标
    if metric == 'auto':
        if is_classification:
            metric = 'f1_weighted'
        else:
            metric = 'neg_mean_squared_error'
    
    # 每个模型搜索最佳参数
    best_score = float('-inf')
    best_model = None
    best_params = None
    best_name = None
    all_models_results = []
    
    for model_type in models_to_try:
        if model_type not in MODEL_TYPES:
            print(f"警告：模型类型 {model_type} 不在支持列表中，将跳过。")
            continue
        
        print(f"正在训练和优化 {model_type} 模型...")
        
        # 获取模型类和参数网格
        model_class = MODEL_TYPES[model_type]
        param_grid = param_grids.get(model_type, {})
        
        try:
            # 创建和训练网格搜索
            grid_search = GridSearchCV(
                model_class(),
                param_grid,
                cv=cv,
                scoring=metric,
                n_jobs=-1 if model_type != 'svm_classifier' else 1  # SVM可能很慢
            )
            grid_search.fit(X_train, y_train)
            
            # 评估最佳模型
            best_estimator = grid_search.best_estimator_
            
            if is_classification:
                y_pred = best_estimator.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                test_score = f1  # 使用F1作为测试集评分
                model_metrics = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                }
            else:
                y_pred = best_estimator.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                test_score = -mse  # 使用负MSE作为测试集评分（越高越好）
                model_metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
            
            # 记录此模型的结果
            model_result = {
                'model_type': model_type,
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
                'test_score': test_score,
                'metrics': model_metrics
            }
            all_models_results.append(model_result)
            
            # 更新最佳模型（基于CV分数）
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = best_estimator
                best_params = grid_search.best_params_
                best_name = model_type
            
            print(f"  {model_type} 优化完成。CV分数: {grid_search.best_score_:.4f}, 测试分数: {test_score:.4f}")
            
        except Exception as e:
            print(f"训练 {model_type} 时出错: {e}")
    
    if best_model is None:
        raise ValueError("没有找到有效的模型")
    
    # 保存最佳模型
    model_name = f"automl_{best_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata = {
        'model_type': best_name,
        'model_params': best_params,
        'cv_score': best_score,
        'all_models_results': all_models_results,
        'best_model_idx': [i for i, r in enumerate(all_models_results) if r['model_type'] == best_name][0],
        'is_classification': is_classification,
        'metric': metric,
        'created_by': 'auto_model_selection'
    }
    
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((best_model, preprocessors, metadata), f)
    
    return {
        'model_name': model_name,
        'model_type': best_name,
        'model_path': model_path,
        'model': best_model,
        'preprocessors': preprocessors,
        'params': best_params,
        'cv_score': best_score,
        'is_classification': is_classification,
        'all_models_results': all_models_results
    }

# 模型可解释性功能
def explain_model_prediction(model_name, input_data):
    """
    解释模型预测结果
    
    Args:
        model_name: 模型名称
        input_data: 输入数据（字典或数据框）
        
    Returns:
        解释结果，包含特征重要性和其他解释信息
    """
    # 加载模型
    model, preprocessors, _ = load_model(model_name)
    
    # 转换输入数据为DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, list) and all(isinstance(item, dict) for item in input_data):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data.copy()
    
    # 应用预处理
    # 处理分类特征
    if 'label_encoders' in preprocessors:
        for col, encoder in preprocessors['label_encoders'].items():
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)
                try:
                    input_df[col] = encoder.transform(input_df[col])
                except ValueError:
                    # 处理未知类别
                    print(f"警告: 列 '{col}' 包含模型训练期间未见过的类别。将使用默认值0。")
                    input_df[col] = 0
    
    # 标准化数值特征
    if 'scaler' in preprocessors and hasattr(preprocessors['scaler'], 'transform'):
        numeric_cols = [col for col in input_df.columns if col in preprocessors['scaler'].feature_names_in_]
        if numeric_cols:
            input_df[numeric_cols] = preprocessors['scaler'].transform(input_df[numeric_cols])
    
    # 获取预测
    prediction = model.predict(input_df)
    
    # 计算特征重要性
    feature_importance = {}
    
    # 对于树模型，我们可以直接获取特征重要性
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        for i, col in enumerate(input_df.columns):
            feature_importance[col] = float(importance[i])
    # 对于线性模型，我们可以获取系数
    elif hasattr(model, 'coef_'):
        coef = model.coef_
        if coef.ndim == 1:
            for i, col in enumerate(input_df.columns):
                feature_importance[col] = float(coef[i])
        else:
            # 对于多类别分类
            for i, col in enumerate(input_df.columns):
                feature_importance[col] = float(np.mean(abs(coef[:, i])))
    
    # 排序并规范化特征重要性
    if feature_importance:
        # 转换为列表并排序
        feature_imp_list = [(col, imp) for col, imp in feature_importance.items()]
        feature_imp_list.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 计算当前预测的特征贡献
        feature_contributions = []
        for col, imp in feature_imp_list:
            idx = list(input_df.columns).index(col)
            value = input_df.iloc[0, idx]
            
            # 计算贡献 (仅适用于线性模型，对于其他模型这是一个近似)
            if hasattr(model, 'coef_'):
                if model.coef_.ndim == 1:
                    contribution = float(value * model.coef_[idx])
                else:
                    contribution = float(value * np.mean(model.coef_[:, idx]))
            else:
                contribution = float(value * imp)
            
            feature_contributions.append({
                'feature': col,
                'importance': float(imp),
                'value': float(value),
                'contribution': contribution
            })
    else:
        feature_contributions = []
    
    return {
        'prediction': prediction.tolist(),
        'feature_importance': feature_importance,
        'feature_contributions': feature_contributions,
        'input_data': input_df.to_dict('records')
    }

# 模型评估与比较
def compare_models(model_names, test_data_path, target_column):
    """
    比较多个模型在测试集上的表现
    
    Args:
        model_names: 要比较的模型名称列表
        test_data_path: 测试数据路径
        target_column: 目标列名
        
    Returns:
        比较结果
    """
    # 加载测试数据
    if test_data_path.endswith('.csv'):
        test_data = pd.read_csv(test_data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    elif test_data_path.endswith(('.xls', '.xlsx')):
        test_data = pd.read_excel(test_data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    elif test_data_path.endswith('.json'):
        test_data = pd.read_json(test_data_path)
        test_data = test_data.fillna('')
    else:
        raise ValueError("目前只支持CSV、Excel和JSON文件格式")
    
    # 确保目标列存在于数据中
    if target_column not in test_data.columns:
        raise ValueError(f"目标列 '{target_column}' 不在测试数据中。可用列: {', '.join(test_data.columns)}")
        
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    # 存储每个模型的评估结果
    results = []
    
    for model_name in model_names:
        # 加载模型
        try:
            model, preprocessors, _ = load_model(model_name)
            
            # 应用预处理
            X_processed = X_test.copy()
            
            # 应用标签编码器
            if 'label_encoders' in preprocessors:
                for col, encoder in preprocessors['label_encoders'].items():
                    if col in X_processed.columns:
                        X_processed[col] = X_processed[col].astype(str)
                        try:
                            X_processed[col] = encoder.transform(X_processed[col])
                        except ValueError:
                            # 处理未知类别
                            print(f"警告: 列 '{col}' 包含模型训练期间未见过的类别。将使用默认值0。")
                            X_processed[col] = 0
            
            # 应用标准化
            if 'scaler' in preprocessors and hasattr(preprocessors['scaler'], 'transform'):
                numeric_cols = [col for col in X_processed.columns if col in preprocessors['scaler'].feature_names_in_]
                if numeric_cols:
                    X_processed[numeric_cols] = preprocessors['scaler'].transform(X_processed[numeric_cols])
            
            # 判断是分类还是回归
            is_classifier = hasattr(model, "predict_proba")
            
            # 进行预测
            y_pred = model.predict(X_processed)
            
            # 计算评估指标
            metrics = {}
            if is_classifier:
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
                metrics["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            else:
                metrics["mse"] = mean_squared_error(y_test, y_pred)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                metrics["mae"] = mean_absolute_error(y_test, y_pred)
                metrics["r2"] = r2_score(y_test, y_pred)
            
            # 存储结果
            results.append({
                "model_name": model_name,
                "is_classifier": is_classifier,
                "metrics": metrics
            })
            
        except Exception as e:
            print(f"评估模型 {model_name} 时出错: {e}")
            results.append({
                "model_name": model_name,
                "error": str(e)
            })
    
    # 整理比较结果
    comparison = {
        "models": results,
        "test_data": test_data_path,
        "target_column": target_column,
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # 对于分类模型和回归模型分别找出最佳模型
    best_classifier = None
    best_regressor = None
    best_classifier_score = -float('inf')
    best_regressor_score = -float('inf')
    
    for result in results:
        if "error" in result:
            continue
            
        if result["is_classifier"]:
            # 使用F1分数作为分类器评分
            if result["metrics"]["f1"] > best_classifier_score:
                best_classifier_score = result["metrics"]["f1"]
                best_classifier = result["model_name"]
        else:
            # 使用R2分数作为回归器评分
            if result["metrics"]["r2"] > best_regressor_score:
                best_regressor_score = result["metrics"]["r2"]
                best_regressor = result["model_name"]
    
    comparison["best_classifier"] = best_classifier
    comparison["best_regressor"] = best_regressor
    
    return comparison 

def optimize_ensemble_weights(base_models, X, y, n_trials=10):
    """优化集成模型权重"""
    loaded_models = []
    for model_name in base_models:
        model, preprocessors, _ = load_model(model_name)
        loaded_models.append(model)
    
    # 使用贝叶斯优化寻找最佳权重
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    import numpy as np
    
    best_score = 0
    best_weights = None
    
    for _ in range(n_trials):
        # 生成随机权重
        weights = np.random.dirichlet(np.ones(len(loaded_models)))
        
        # 创建加权集成
        if hasattr(loaded_models[0], 'predict_proba'):
            ensemble = VotingClassifier(
                estimators=[(f"model_{i}", m) for i, m in enumerate(loaded_models)],
                weights=weights,
                voting='soft'
            )
        else:
            ensemble = VotingRegressor(
                estimators=[(f"model_{i}", m) for i, m in enumerate(loaded_models)],
                weights=weights
            )
        
        # 评估性能
        score = np.mean(cross_val_score(ensemble, X, y, cv=5))
        
        if score > best_score:
            best_score = score
            best_weights = weights
    
    return best_weights