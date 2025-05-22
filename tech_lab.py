#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术实验室模块
用于模型测试、性能分析和场景模拟
"""

import os
import json
import datetime
import logging
import pandas as pd
import numpy as np
import uuid
from typing import Dict, List, Any, Optional, Union
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, explained_variance_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 实验结果存储目录
EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'experiments')
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# 模型存储目录
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)

# 上传数据存储目录
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_available_models() -> List[Dict[str, Any]]:
    """
    获取可用模型列表
    
    Returns:
        模型信息列表
    """
    try:
        logger.info("获取可用模型列表")
        
        models = []
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pkl'):
                model_name = filename[:-4]
                
                # 检查是否有元数据文件
                metadata_file = os.path.join(MODELS_DIR, f"{model_name}_metadata.json")
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        models.append({
                            "id": model_name,
                            "name": metadata.get('model_name', model_name),
                            "type": metadata.get('model_type', 'unknown'),
                            "target": metadata.get('target_column', 'unknown'),
                            "features": metadata.get('feature_columns', []),
                            "metrics": metadata.get('metrics', {}),
                            "created_at": metadata.get('created_at', ''),
                            "has_metadata": True
                        })
                else:
                    # 没有元数据文件，添加基本信息
                    models.append({
                        "id": model_name,
                        "name": model_name,
                        "type": "unknown",
                        "has_metadata": False
                    })
        
        return models
    except Exception as e:
        logger.error(f"获取可用模型列表失败: {str(e)}")
        raise

def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    获取模型详细信息
    
    Args:
        model_id: 模型ID
        
    Returns:
        模型详细信息
    """
    try:
        logger.info(f"获取模型 {model_id} 详细信息")
        
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_path):
            raise ValueError(f"模型 {model_id} 不存在")
            
        # 加载模型
        model = joblib.load(model_path)
        
        # 检查是否有元数据文件
        metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "model_name": model_id,
                "model_type": type(model).__name__,
                "created_at": datetime.datetime.now().isoformat()
            }
            
        # 获取模型详情
        details = {
            "id": model_id,
            "name": metadata.get('model_name', model_id),
            "type": metadata.get('model_type', type(model).__name__),
            "target": metadata.get('target_column', 'unknown'),
            "features": metadata.get('feature_columns', []),
            "metrics": metadata.get('metrics', {}),
            "created_at": metadata.get('created_at', ''),
            "data_path": metadata.get('data_path', ''),
            "data_rows": metadata.get('data_rows', 0),
            "data_columns": metadata.get('data_columns', 0),
            "model_params": _get_model_params(model),
            "has_metadata": True
        }
            
        return details
    except Exception as e:
        logger.error(f"获取模型详细信息失败: {str(e)}")
        raise

def _get_model_params(model: Any) -> Dict[str, Any]:
    """
    获取模型参数
    
    Args:
        model: 模型对象
        
    Returns:
        模型参数
    """
    params = {}
    
    # 尝试获取模型参数
    if hasattr(model, 'get_params'):
        try:
            model_params = model.get_params()
            for key, value in model_params.items():
                if isinstance(value, (int, float, str, bool, type(None))):
                    params[key] = value
                else:
                    params[key] = str(value)
        except:
            pass
            
    # 尝试获取特征重要性
    if hasattr(model, 'feature_importances_'):
        params['feature_importances'] = model.feature_importances_.tolist()
    
    # 尝试获取回归系数
    if hasattr(model, 'coef_'):
        if isinstance(model.coef_, np.ndarray):
            if model.coef_.ndim == 1:
                params['coefficients'] = model.coef_.tolist()
            else:
                params['coefficients'] = model.coef_.tolist()
                
    return params

def create_experiment(name: str, description: str, model_id: str, 
                    experiment_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    创建实验
    
    Args:
        name: 实验名称
        description: 实验描述
        model_id: 模型ID
        experiment_type: 实验类型 (prediction, analysis, comparison)
        config: 实验配置
        
    Returns:
        实验信息
    """
    try:
        logger.info(f"创建实验: {name}, 类型: {experiment_type}")
        
        # 验证模型存在
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_path):
            raise ValueError(f"模型 {model_id} 不存在")
            
        # 创建实验ID
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        # 创建实验记录
        experiment = {
            "id": experiment_id,
            "name": name,
            "description": description,
            "model_id": model_id,
            "type": experiment_type,
            "config": config,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "created",
            "results": None
        }
        
        # 保存实验记录
        experiment_file = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, ensure_ascii=False, indent=2)
            
        return experiment
    except Exception as e:
        logger.error(f"创建实验失败: {str(e)}")
        raise

def run_experiment(experiment_id: str, data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    运行实验
    
    Args:
        experiment_id: 实验ID
        data_path: 数据文件路径 (可选)
        
    Returns:
        实验结果
    """
    try:
        logger.info(f"运行实验: {experiment_id}")
        
        # 加载实验配置
        experiment_file = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        if not os.path.exists(experiment_file):
            raise ValueError(f"实验 {experiment_id} 不存在")
            
        with open(experiment_file, 'r', encoding='utf-8') as f:
            experiment = json.load(f)
            
        # 更新实验状态
        experiment['status'] = "running"
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, ensure_ascii=False, indent=2)
            
        # 获取实验类型和配置
        experiment_type = experiment.get('type')
        config = experiment.get('config', {})
        model_id = experiment.get('model_id')
        
        # 根据实验类型执行不同操作
        if experiment_type == "prediction":
            results = _run_prediction_experiment(model_id, config, data_path)
        elif experiment_type == "analysis":
            results = _run_analysis_experiment(model_id, config, data_path)
        elif experiment_type == "comparison":
            results = _run_comparison_experiment(config, data_path)
        else:
            raise ValueError(f"不支持的实验类型: {experiment_type}")
            
        # 更新实验结果和状态
        experiment['results'] = results
        experiment['status'] = "completed"
        experiment['completed_at'] = datetime.datetime.now().isoformat()
        
        # 保存更新后的实验记录
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, ensure_ascii=False, indent=2)
            
        return experiment
    except Exception as e:
        logger.error(f"运行实验失败: {str(e)}")
        
        # 更新实验状态为失败
        try:
            experiment_file = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
            if os.path.exists(experiment_file):
                with open(experiment_file, 'r', encoding='utf-8') as f:
                    experiment = json.load(f)
                    
                experiment['status'] = "failed"
                experiment['error'] = str(e)
                
                with open(experiment_file, 'w', encoding='utf-8') as f:
                    json.dump(experiment, f, ensure_ascii=False, indent=2)
        except:
            pass
            
        raise

def _run_prediction_experiment(model_id: str, config: Dict[str, Any], 
                             data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    运行预测实验
    
    Args:
        model_id: 模型ID
        config: 实验配置
        data_path: 数据文件路径 (可选)
        
    Returns:
        实验结果
    """
    # 加载模型
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    model = joblib.load(model_path)
    
    # 加载元数据
    metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}
        
    # 确定数据源
    if data_path is None:
        data_path = metadata.get('data_path')
        if not data_path or not os.path.exists(data_path):
            raise ValueError("没有指定数据文件，且模型元数据中没有有效的数据路径")
    
    # 加载数据
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("不支持的文件格式，仅支持CSV和Excel")
        
    # 检查目标列是否存在
    target_column = metadata.get('target_column')
    if not target_column or target_column not in df.columns:
        raise ValueError(f"目标列 {target_column} 不存在于数据中")
        
    # 准备特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 获取特征列列表
    feature_columns = metadata.get('feature_columns', X.columns.tolist())
    
    # 确保X中只包含模型使用的特征
    X = X[feature_columns]
    
    # 进行预测
    if hasattr(model, 'predict_proba') and config.get('use_probability', False):
        y_pred_proba = model.predict_proba(X)
        if y_pred_proba.shape[1] == 2:  # 二分类
            y_pred_proba = y_pred_proba[:, 1]
        y_pred = model.predict(X)
    else:
        y_pred = model.predict(X)
        y_pred_proba = None
    
    # 计算评估指标
    metrics = {}
    model_type = metadata.get('model_type', '')
    
    # 分类模型
    if 'classifier' in model_type.lower() or hasattr(model, 'predict_proba'):
        metrics['accuracy'] = accuracy_score(y, y_pred)
        metrics['precision'] = precision_score(y, y_pred, average='weighted')
        metrics['recall'] = recall_score(y, y_pred, average='weighted')
        metrics['f1'] = f1_score(y, y_pred, average='weighted')
        
        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC (二分类)
        if len(np.unique(y)) == 2 and y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
    
    # 回归模型
    else:
        metrics['mse'] = mean_squared_error(y, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y, y_pred)
        metrics['explained_variance'] = explained_variance_score(y, y_pred)
    
    # 生成可视化
    visualizations = {}
    
    # 预测值vs实际值
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 实际值')
    plt.grid(True)
    
    # 将图保存到内存
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # 将图像转换为base64编码
    pred_vs_actual_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    visualizations['pred_vs_actual'] = pred_vs_actual_b64
    plt.close()
    
    # 生成预测结果
    results = {
        "metrics": metrics,
        "visualizations": visualizations,
        "data": {
            "total_samples": len(X),
            "feature_names": feature_columns,
            "target_column": target_column
        }
    }
    
    return results

def _run_analysis_experiment(model_id: str, config: Dict[str, Any], 
                           data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    运行分析实验
    
    Args:
        model_id: 模型ID
        config: 实验配置
        data_path: 数据文件路径 (可选)
        
    Returns:
        实验结果
    """
    # 加载模型
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    model = joblib.load(model_path)
    
    # 加载元数据
    metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # 分析类型
    analysis_type = config.get('analysis_type', 'feature_importance')
    
    results = {
        "analysis_type": analysis_type,
        "model_id": model_id,
        "model_type": metadata.get('model_type', type(model).__name__)
    }
    
    # 根据分析类型执行不同操作
    if analysis_type == "feature_importance":
        # 获取特征重要性
        feature_importance = None
        feature_names = metadata.get('feature_columns', [])
        
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = np.abs(model.coef_)
            if feature_importance.ndim > 1:
                feature_importance = np.mean(feature_importance, axis=0)
                
        if feature_importance is not None and len(feature_names) == len(feature_importance):
            # 创建特征重要性的数据框
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            })
            
            # 按重要性排序
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            # 可视化特征重要性
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('重要性')
            plt.ylabel('特征')
            plt.title('特征重要性')
            plt.grid(True, axis='x')
            plt.tight_layout()
            
            # 将图保存到内存
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # 将图像转换为base64编码
            importance_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            results['feature_importance'] = importance_df.to_dict(orient='records')
            results['visualization'] = importance_b64
        else:
            results['error'] = "该模型不支持特征重要性分析"
    
    elif analysis_type == "partial_dependence":
        # 确定数据源
        if data_path is None:
            data_path = metadata.get('data_path')
            if not data_path or not os.path.exists(data_path):
                raise ValueError("没有指定数据文件，且模型元数据中没有有效的数据路径")
        
        # 加载数据
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("不支持的文件格式，仅支持CSV和Excel")
            
        # 检查目标列是否存在
        target_column = metadata.get('target_column')
        if not target_column or target_column not in df.columns:
            raise ValueError(f"目标列 {target_column} 不存在于数据中")
            
        # 获取特征列列表
        feature_columns = metadata.get('feature_columns', [])
        if not feature_columns:
            feature_columns = [c for c in df.columns if c != target_column]
            
        # 选择要分析的特征
        target_feature = config.get('target_feature')
        if not target_feature:
            # 选择第一个特征
            target_feature = feature_columns[0]
            
        if target_feature not in feature_columns:
            raise ValueError(f"特征 {target_feature} 不在特征列表中")
            
        # 准备特征和目标
        X = df[feature_columns]
        
        # 计算部分依赖
        feature_values = np.linspace(
            X[target_feature].min(),
            X[target_feature].max(),
            num=config.get('num_points', 20)
        )
        
        mean_predictions = []
        for value in feature_values:
            X_temp = X.copy()
            X_temp[target_feature] = value
            predictions = model.predict(X_temp)
            mean_predictions.append(np.mean(predictions))
            
        # 可视化部分依赖
        plt.figure(figsize=(10, 6))
        plt.plot(feature_values, mean_predictions)
        plt.xlabel(target_feature)
        plt.ylabel('预测值的平均值')
        plt.title(f'{target_feature} 的部分依赖图')
        plt.grid(True)
        
        # 将图保存到内存
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # 将图像转换为base64编码
        pdp_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        results['partial_dependence'] = {
            'feature': target_feature,
            'values': feature_values.tolist(),
            'mean_predictions': mean_predictions
        }
        results['visualization'] = pdp_b64
    
    elif analysis_type == "hyperparameter_sensitivity":
        # 此分析需要重新训练模型，较为复杂
        results['error'] = "超参数敏感性分析需要重新训练模型，请使用专门的工具进行"
    
    else:
        results['error'] = f"不支持的分析类型: {analysis_type}"
    
    return results

def _run_comparison_experiment(config: Dict[str, Any], 
                             data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    运行模型比较实验
    
    Args:
        config: 实验配置
        data_path: 数据文件路径 (可选)
        
    Returns:
        实验结果
    """
    # 获取要比较的模型ID列表
    model_ids = config.get('model_ids', [])
    if not model_ids:
        raise ValueError("没有指定要比较的模型")
        
    # 确定数据源
    if data_path is None:
        raise ValueError("模型比较实验需要指定数据文件")
    
    # 加载数据
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("不支持的文件格式，仅支持CSV和Excel")
        
    # 检查目标列是否存在
    target_column = config.get('target_column')
    if not target_column or target_column not in df.columns:
        raise ValueError(f"目标列 {target_column} 不存在于数据中")
        
    # 检查特征列是否存在
    feature_columns = config.get('feature_columns', [])
    if not feature_columns:
        feature_columns = [c for c in df.columns if c != target_column]
        
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"特征列 {col} 不存在于数据中")
            
    # 准备特征和目标
    X = df[feature_columns]
    y = df[target_column]
    
    # 比较结果
    comparison_results = []
    
    # 加载并评估每个模型
    for model_id in model_ids:
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_path):
            comparison_results.append({
                "model_id": model_id,
                "error": f"模型 {model_id} 不存在"
            })
            continue
            
        # 加载模型
        model = joblib.load(model_path)
        
        # 加载元数据
        metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "model_name": model_id,
                "model_type": type(model).__name__
            }
            
        # 进行预测
        try:
            y_pred = model.predict(X)
            
            # 计算评估指标
            metrics = {}
            model_type = metadata.get('model_type', '')
            
            # 分类模型
            if 'classifier' in model_type.lower() or hasattr(model, 'predict_proba'):
                metrics['accuracy'] = accuracy_score(y, y_pred)
                metrics['precision'] = precision_score(y, y_pred, average='weighted')
                metrics['recall'] = recall_score(y, y_pred, average='weighted')
                metrics['f1'] = f1_score(y, y_pred, average='weighted')
                
                # ROC AUC (二分类)
                if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
            
            # 回归模型
            else:
                metrics['mse'] = mean_squared_error(y, y_pred)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(y, y_pred)
                metrics['explained_variance'] = explained_variance_score(y, y_pred)
                
            comparison_results.append({
                "model_id": model_id,
                "model_name": metadata.get('model_name', model_id),
                "model_type": metadata.get('model_type', type(model).__name__),
                "metrics": metrics
            })
        except Exception as e:
            comparison_results.append({
                "model_id": model_id,
                "model_name": metadata.get('model_name', model_id),
                "model_type": metadata.get('model_type', type(model).__name__),
                "error": str(e)
            })
    
    # 创建比较可视化
    visualizations = {}
    
    # 如果都是同类型的模型，创建性能比较图
    valid_results = [r for r in comparison_results if 'metrics' in r]
    if valid_results:
        # 检查所有模型是否都是同一类型
        first_model_type = 'classifier' if 'accuracy' in valid_results[0]['metrics'] else 'regressor'
        all_same_type = all(
            ('accuracy' in r['metrics']) == (first_model_type == 'classifier') 
            for r in valid_results
        )
        
        if all_same_type:
            # 对于分类器，比较准确率、精确率、召回率、F1分数
            if first_model_type == 'classifier':
                metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
                metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
                
                # 创建比较图
                plt.figure(figsize=(12, 8))
                width = 0.2
                x = np.arange(len(metrics_labels))
                
                for i, result in enumerate(valid_results):
                    model_metrics = [result['metrics'].get(m, 0) for m in metrics_to_compare]
                    plt.bar(x + i*width, model_metrics, width, label=result['model_name'])
                
                plt.xlabel('评估指标')
                plt.ylabel('分数')
                plt.title('模型性能比较')
                plt.xticks(x + width * (len(valid_results) - 1) / 2, metrics_labels)
                plt.legend()
                plt.grid(True, axis='y')
                
                # 将图保存到内存
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                
                # 将图像转换为base64编码
                comp_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['classifier_comparison'] = comp_b64
                plt.close()
                
            # 对于回归器，比较MSE、RMSE、R2
            else:
                metrics_to_compare = ['mse', 'rmse', 'r2', 'explained_variance']
                metrics_labels = ['MSE', 'RMSE', 'R²', '解释方差']
                
                # R2和解释方差单独绘制
                plt.figure(figsize=(12, 10))
                
                # 创建子图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # 绘制MSE和RMSE
                width = 0.4
                x1 = np.arange(2)
                
                for i, result in enumerate(valid_results):
                    model_metrics = [result['metrics'].get('mse', 0), result['metrics'].get('rmse', 0)]
                    ax1.bar(x1 + i*width, model_metrics, width, label=result['model_name'])
                
                ax1.set_xlabel('评估指标')
                ax1.set_ylabel('误差')
                ax1.set_title('误差指标比较')
                ax1.set_xticks(x1 + width * (len(valid_results) - 1) / 2)
                ax1.set_xticklabels(['MSE', 'RMSE'])
                ax1.legend()
                ax1.grid(True, axis='y')
                
                # 绘制R2和解释方差
                x2 = np.arange(2)
                
                for i, result in enumerate(valid_results):
                    model_metrics = [result['metrics'].get('r2', 0), result['metrics'].get('explained_variance', 0)]
                    ax2.bar(x2 + i*width, model_metrics, width, label=result['model_name'])
                
                ax2.set_xlabel('评估指标')
                ax2.set_ylabel('分数')
                ax2.set_title('拟合优度指标比较')
                ax2.set_xticks(x2 + width * (len(valid_results) - 1) / 2)
                ax2.set_xticklabels(['R²', '解释方差'])
                ax2.legend()
                ax2.grid(True, axis='y')
                
                plt.tight_layout()
                
                # 将图保存到内存
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                
                # 将图像转换为base64编码
                comp_b64 = base64.b64encode(buffer.read()).decode('utf-8')
                visualizations['regressor_comparison'] = comp_b64
                plt.close()
    
    results = {
        "models_compared": len(model_ids),
        "comparison_results": comparison_results,
        "visualizations": visualizations,
        "data": {
            "total_samples": len(X),
            "feature_names": feature_columns,
            "target_column": target_column
        }
    }
    
    return results

def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """
    获取实验详情
    
    Args:
        experiment_id: 实验ID
        
    Returns:
        实验详情
    """
    try:
        logger.info(f"获取实验 {experiment_id} 详情")
        
        experiment_file = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        if not os.path.exists(experiment_file):
            raise ValueError(f"实验 {experiment_id} 不存在")
            
        with open(experiment_file, 'r', encoding='utf-8') as f:
            experiment = json.load(f)
            
        return experiment
    except Exception as e:
        logger.error(f"获取实验详情失败: {str(e)}")
        raise

def get_all_experiments() -> List[Dict[str, Any]]:
    """
    获取所有实验
    
    Returns:
        实验列表
    """
    try:
        logger.info("获取所有实验")
        
        experiments = []
        for filename in os.listdir(EXPERIMENTS_DIR):
            if filename.endswith('.json'):
                experiment_file = os.path.join(EXPERIMENTS_DIR, filename)
                with open(experiment_file, 'r', encoding='utf-8') as f:
                    experiment = json.load(f)
                    
                    # 简化返回的信息
                    experiments.append({
                        "id": experiment.get('id'),
                        "name": experiment.get('name'),
                        "description": experiment.get('description'),
                        "model_id": experiment.get('model_id'),
                        "type": experiment.get('type'),
                        "created_at": experiment.get('created_at'),
                        "status": experiment.get('status'),
                        "has_results": experiment.get('results') is not None
                    })
        
        # 按创建时间排序，最新的在前
        experiments.sort(key=lambda p: p.get('created_at', ''), reverse=True)
        
        return experiments
    except Exception as e:
        logger.error(f"获取所有实验失败: {str(e)}")
        raise 