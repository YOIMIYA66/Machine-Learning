#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
机器学习预测模块
用于预测学习效果和进度
"""

import os
import json
import datetime
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
import joblib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型存储目录
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 学习数据存储目录
LEARNING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'learning_data')
os.makedirs(LEARNING_DATA_DIR, exist_ok=True)

# 默认特征映射
FEATURE_MAPPING = {
    # 学习者特征
    'prior_knowledge_level': {
        'none': 0,
        'basic': 1,
        'intermediate': 2,
        'advanced': 3
    },
    'focus_level': {
        'low': 0,
        'medium': 1,
        'high': 2
    },
    # 学习内容特征
    'content_difficulty': {
        'beginner': 0,
        'intermediate': 1,
        'advanced': 2
    }
}

def _encode_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    将特征编码为数值
    
    Args:
        features: 特征字典
        
    Returns:
        编码后的特征字典
    """
    encoded_features = {}
    
    for key, value in features.items():
        if isinstance(value, (int, float)):
            encoded_features[key] = value
        elif key in FEATURE_MAPPING and value in FEATURE_MAPPING[key]:
            encoded_features[key] = FEATURE_MAPPING[key][value]
        else:
            encoded_features[key] = value
            
    return encoded_features

def predict_learning_outcome(module_id: str, learning_parameters: Dict[str, Any],
                          target_prediction: str = "mastery_probability") -> Dict[str, Any]:
    """
    预测学习效果
    
    Args:
        module_id: 学习模块ID
        learning_parameters: 学习参数
        target_prediction: 预测目标 (mastery_probability 或 completion_time)
        
    Returns:
        预测结果
    """
    try:
        logger.info(f"预测学习效果: 模块 {module_id}, 目标 {target_prediction}")
        
        # 编码特征
        encoded_params = _encode_features(learning_parameters)
        
        # 根据预测目标选择模型
        if target_prediction == "mastery_probability":
            # 掌握概率预测 - 使用分类模型
            model_path = os.path.join(MODELS_DIR, "mastery_prediction_model.pkl")
            
            # 如果模型不存在，使用默认的逻辑回归模型
            if not os.path.exists(model_path):
                logger.warning(f"模型文件 {model_path} 不存在，使用备用预测逻辑")
                # 使用简单的规则进行预测
                probability = _predict_mastery_fallback(encoded_params)
                confidence_interval = [max(0, probability - 0.1), min(1, probability + 0.1)]
                
                result = {
                    "module_id": module_id,
                    "prediction_type": "mastery_probability",
                    "predicted_value": round(probability * 100, 1),  # 转换为百分比
                    "confidence_interval": [round(ci * 100, 1) for ci in confidence_interval],
                    "model_used": "fallback_rule_based",
                    "raw_model_output": encoded_params,
                    "feature_importance": {
                        "weekly_study_hours": 30,
                        "prior_knowledge_level": 25,
                        "focus_level": 20,
                        "content_difficulty": 25
                    }
                }
            else:
                # 加载模型
                model = joblib.load(model_path)
                
                # 准备特征
                features = pd.DataFrame([encoded_params])
                
                # 进行预测
                probability = model.predict_proba(features)[0][1]  # 假设是二分类，取正类概率
                
                # 计算置信区间 (简化版)
                confidence_interval = [max(0, probability - 0.1), min(1, probability + 0.1)]
                
                # 获取特征重要性 (如果模型支持)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = list(encoded_params.keys())
                    for i, name in enumerate(feature_names):
                        if i < len(importances):
                            feature_importance[name] = int(importances[i] * 100)
                
                result = {
                    "module_id": module_id,
                    "prediction_type": "mastery_probability",
                    "predicted_value": round(probability * 100, 1),  # 转换为百分比
                    "confidence_interval": [round(ci * 100, 1) for ci in confidence_interval],
                    "model_used": type(model).__name__,
                    "feature_importance": feature_importance
                }
                
        elif target_prediction == "completion_time":
            # 完成时间预测 - 使用回归模型
            model_path = os.path.join(MODELS_DIR, "completion_time_model.pkl")
            
            # 如果模型不存在，使用默认的线性回归模型
            if not os.path.exists(model_path):
                logger.warning(f"模型文件 {model_path} 不存在，使用备用预测逻辑")
                # 使用简单的规则进行预测
                hours = _predict_completion_time_fallback(encoded_params)
                
                result = {
                    "module_id": module_id,
                    "prediction_type": "completion_time_hours",
                    "predicted_value": round(hours, 1),
                    "confidence_interval": [round(max(0, hours * 0.8), 1), round(hours * 1.2, 1)],
                    "model_used": "fallback_rule_based",
                    "raw_model_output": encoded_params,
                    "feature_importance": {
                        "weekly_study_hours": 40,
                        "prior_knowledge_level": 30,
                        "focus_level": 20,
                        "content_difficulty": 10
                    }
                }
            else:
                # 加载模型
                model = joblib.load(model_path)
                
                # 准备特征
                features = pd.DataFrame([encoded_params])
                
                # 进行预测
                hours = model.predict(features)[0]
                
                # 计算置信区间 (简化版)
                confidence_interval = [max(0, hours * 0.8), hours * 1.2]
                
                # 获取特征重要性 (如果模型支持)
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_names = list(encoded_params.keys())
                    for i, name in enumerate(feature_names):
                        if i < len(importances):
                            feature_importance[name] = int(importances[i] * 100)
                
                result = {
                    "module_id": module_id,
                    "prediction_type": "completion_time_hours",
                    "predicted_value": round(hours, 1),
                    "confidence_interval": [round(ci, 1) for ci in confidence_interval],
                    "model_used": type(model).__name__,
                    "feature_importance": feature_importance
                }
        else:
            raise ValueError(f"不支持的预测目标: {target_prediction}")
        
        # 添加时间戳
        result["timestamp"] = datetime.datetime.now().isoformat()
        
        return result
    except Exception as e:
        logger.error(f"预测学习效果时出错: {str(e)}")
        raise

def _predict_mastery_fallback(params: Dict[str, Any]) -> float:
    """
    备用的掌握概率预测逻辑
    
    Args:
        params: 编码后的特征
        
    Returns:
        掌握概率 (0-1)
    """
    # 基础概率为0.5
    base_probability = 0.5
    
    # 每周学习时间影响 (每周10小时及以上获得最大加成)
    weekly_hours = params.get('weekly_study_hours', 0)
    time_factor = min(0.3, weekly_hours / 10.0 * 0.3)
    
    # 先验知识水平影响
    prior_knowledge = params.get('prior_knowledge_level', 0)
    knowledge_factor = prior_knowledge * 0.1  # 0-0.3
    
    # 专注度影响
    focus_level = params.get('focus_level', 1)
    focus_factor = focus_level * 0.1  # 0-0.2
    
    # 内容难度影响 (难度越大，概率越低)
    content_difficulty = params.get('content_difficulty', 1)
    difficulty_factor = -0.1 * content_difficulty  # 0 to -0.2
    
    # 计算最终概率
    probability = base_probability + time_factor + knowledge_factor + focus_factor + difficulty_factor
    
    # 限制在0.1-0.95之间
    probability = min(0.95, max(0.1, probability))
    
    return probability

def _predict_completion_time_fallback(params: Dict[str, Any]) -> float:
    """
    备用的完成时间预测逻辑
    
    Args:
        params: 编码后的特征
        
    Returns:
        预计完成时间 (小时)
    """
    # 基础时间 (假设一个中等难度模块需要10小时)
    base_hours = 10.0
    
    # 内容难度影响
    content_difficulty = params.get('content_difficulty', 1)
    difficulty_factor = content_difficulty * 5  # 0, 5, 10 额外小时
    
    # 先验知识水平影响 (降低所需时间)
    prior_knowledge = params.get('prior_knowledge_level', 0)
    knowledge_factor = -prior_knowledge * 2  # -0, -2, -4, -6 小时
    
    # 专注度影响
    focus_level = params.get('focus_level', 1)
    focus_factor = (1 - focus_level * 0.2)  # 1, 0.8, 0.6 倍时间
    
    # 计算最终时间
    total_hours = (base_hours + difficulty_factor + knowledge_factor) * focus_factor
    
    # 最少需要1小时
    total_hours = max(1.0, total_hours)
    
    return total_hours


def train_custom_prediction_model(data_path: str, target_column: str, 
                                model_type: str, output_model_name: str) -> Dict[str, Any]:
    """
    训练自定义预测模型
    
    Args:
        data_path: 数据文件路径
        target_column: 目标列名
        model_type: 模型类型
        output_model_name: 输出模型名称
        
    Returns:
        训练结果
    """
    try:
        logger.info(f"训练自定义预测模型: {model_type}, 目标列: {target_column}")
        
        # 读取数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的文件格式，仅支持CSV和Excel")
            
        # 数据预处理
        df = df.dropna()  # 简单起见，删除缺失值
        
        # 准备特征和目标
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 选择模型
        model = None
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "logistic_regression":
            model = LogisticRegression(max_iter=1000)
        elif model_type == "decision_tree_regressor":
            model = DecisionTreeRegressor(random_state=42)
        elif model_type == "decision_tree_classifier":
            model = DecisionTreeClassifier(random_state=42)
        elif model_type == "random_forest_regressor":
            model = RandomForestRegressor(random_state=42, n_estimators=100)
        elif model_type == "random_forest_classifier":
            model = RandomForestClassifier(random_state=42, n_estimators=100)
        elif model_type == "svr":
            model = SVR()
        elif model_type == "svc":
            model = SVC(probability=True)
        elif model_type == "knn_regressor":
            model = KNeighborsRegressor(n_neighbors=5)
        elif model_type == "knn_classifier":
            model = KNeighborsClassifier(n_neighbors=5)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估模型
        metrics = {}
        if model_type in ["linear_regression", "decision_tree_regressor", "random_forest_regressor", "svr", "knn_regressor"]:
            # 回归模型评估
            y_pred = model.predict(X_test)
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
            metrics["r2"] = r2_score(y_test, y_pred)
        else:
            # 分类模型评估
            y_pred = model.predict(X_test)
            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            if len(np.unique(y)) == 2:  # 二分类
                metrics["precision"] = precision_score(y_test, y_pred, average='binary')
                metrics["recall"] = recall_score(y_test, y_pred, average='binary')
            else:  # 多分类
                metrics["precision"] = precision_score(y_test, y_pred, average='weighted')
                metrics["recall"] = recall_score(y_test, y_pred, average='weighted')
                
        # 保存模型
        model_path = os.path.join(MODELS_DIR, f"{output_model_name}.pkl")
        joblib.dump(model, model_path)
        
        # 保存模型元数据
        model_metadata = {
            "model_name": output_model_name,
            "model_type": model_type,
            "target_column": target_column,
            "feature_columns": X.columns.tolist(),
            "metrics": metrics,
            "created_at": datetime.datetime.now().isoformat(),
            "data_path": data_path,
            "data_rows": len(df),
            "data_columns": len(df.columns)
        }
        
        metadata_path = os.path.join(MODELS_DIR, f"{output_model_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(model_metadata, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "model_name": output_model_name,
            "model_path": model_path,
            "metrics": metrics,
            "feature_columns": X.columns.tolist()
        }
    except Exception as e:
        logger.error(f"训练自定义预测模型时出错: {str(e)}")
        raise 