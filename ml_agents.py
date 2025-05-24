# ml_agents.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
import colorsys
import uuid
import traceback

from langchain.tools import StructuredTool
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import Field
from pydantic import BaseModel
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import PromptTemplate

from baidu_llm import BaiduErnieLLM
from ml_models import (
    train_model as actual_train_model, predict as actual_predict, list_available_models as actual_list_models,
    select_model_for_task as actual_select_model, load_model,
    create_ensemble_model, auto_model_selection as actual_auto_select,
    explain_model_prediction as actual_explain_prediction, compare_models as actual_compare_models,
    save_model_with_version, list_model_versions
)
from sklearn.metrics import confusion_matrix, classification_report, f1_score, mean_absolute_error, r2_score, \
    precision_score, accuracy_score, recall_score, mean_squared_error
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 定义工具输入模型
class TrainModelInput(BaseModel):
    model_type: str = Field(..., description="模型类型，如 'linear_regression', 'logistic_regression', 'decision_tree', 'random_forest_classifier' 等")
    data_path: str = Field(..., description="数据文件路径，支持CSV和Excel格式")
    target_column: str = Field(..., description="目标列名")
    model_name: Optional[str] = Field(None, description="模型保存名称，如不提供则自动生成")
    categorical_columns: Optional[List[str]] = Field(None, description="分类特征列表")
    numerical_columns: Optional[List[str]] = Field(None, description="数值特征列表")

class PredictInput(BaseModel):
    model_name: str = Field(..., description="模型名称")
    input_data: Dict[str, Any] = Field(..., description="输入数据，格式为字段名到值的映射")

# 为其他工具定义Pydantic模型
class RecommendModelInput(BaseModel):
    task_description: str = Field(..., description="对任务的描述，例如'预测房价'或'分类垃圾邮件'")

class DataAnalysisInput(BaseModel):
    file_path: str = Field(..., description="数据文件路径，支持CSV和Excel格式")
    target_column: Optional[str] = Field(None, description="目标列名，用于分析与目标相关的特征")
    analysis_type: Optional[str] = Field(None, description="分析类型，例如 'statistics', 'feature_relevance'")

class EvaluateModelInput(BaseModel):
    model_name: str = Field(..., description="模型名称")
    test_data_path: str = Field(..., description="测试数据文件路径")
    target_column: str = Field(..., description="目标列名")

class EnsembleModelInput(BaseModel):
    base_models: List[str] = Field(..., description="基础模型名称列表")
    ensemble_type: str = Field("voting", description="集成类型，可选 'voting', 'stacking', 'bagging'")
    weights: Optional[List[float]] = Field(None, description="基础模型权重，仅用于voting集成")
    save_name: Optional[str] = Field(None, description="保存的模型名称")

class AutoSelectModelInput(BaseModel):
    data_path: str = Field(..., description="数据文件路径")
    target_column: str = Field(..., description="目标列名")
    categorical_columns: Optional[List[str]] = Field(None, description="分类特征列表")
    numerical_columns: Optional[List[str]] = Field(None, description="数值特征列表")

class ExplainPredictionInput(BaseModel):
    model_name: str = Field(..., description="模型名称")
    input_data: Dict[str, Any] = Field(..., description="输入数据，格式为字段名到值的映射")

class CompareModelsInput(BaseModel):
    model_names: List[str] = Field(..., description="模型名称列表")
    test_data_path: str = Field(..., description="测试数据文件路径")
    target_column: str = Field(..., description="目标列名")

class VersionModelInput(BaseModel):
    model_name: str = Field(..., description="模型名称")
    version: Optional[str] = Field(None, description="版本号，如不提供则使用时间戳")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外的元数据")

class ListModelVersionsInput(BaseModel):
    model_name: str = Field(..., description="模型名称")

# 生成渐变色彩列表
def generate_gradient_colors(n_colors):
    """生成渐变色彩列表，用于图表"""
    colors = []
    for i in range(n_colors):
        # 从蓝紫色渐变到天蓝色
        hue = 0.6 + (0.2 * i / max(1, n_colors - 1))  # 色相从0.6(蓝紫)到0.8(天蓝)
        saturation = 0.7  # 饱和度
        value = 0.9  # 亮度
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # 转换为rgba格式
        rgba = (rgb[0], rgb[1], rgb[2], 0.7)
        colors.append(rgba)
    return colors

# 可视化函数
def generate_visualization(data_type, labels, values, title=None, options=None):
    """生成可视化图表并返回base64编码"""
    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    options = options or {}
    colors = generate_gradient_colors(len(values) if isinstance(values, list) else 5)

    if data_type == 'bar':
        plt.bar(labels, values, color=colors)
        plt.ylabel('值')
    elif data_type == 'line':
        plt.plot(labels, values, marker='o', color='#4F46E5', linewidth=2)
        plt.ylabel('值')
        plt.grid(True, alpha=0.3)
    elif data_type == 'pie':
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                shadow=True, colors=colors)
        plt.axis('equal')  # 使饼图为正圆形
    elif data_type == 'scatter':
        plt.scatter(labels, values, color='#4F46E5', alpha=0.7, s=options.get('point_size', 70))
        plt.ylabel('值')
    elif data_type == 'heatmap':
        # 热力图需要二维数据
        if 'matrix' in options:
            sns.heatmap(options['matrix'], annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=labels, yticklabels=options.get('y_labels', labels))
    elif data_type == 'radar':
        # 雷达图(极坐标条形图)
        theta = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        values = np.array(values)
        ax = plt.subplot(111, polar=True)
        ax.fill(theta, values, color='#4F46E5', alpha=0.25)
        ax.plot(theta, values, color='#4F46E5', linewidth=2)
        ax.set_xticks(theta)
        ax.set_xticklabels(labels)
        ax.grid(True)
    elif data_type == 'bubble':
        # 气泡图 - 需要x, y坐标和大小数据
        sizes = options.get('sizes', [50] * len(values))
        plt.scatter(labels, values, s=sizes, alpha=0.6, c=colors)
        plt.ylabel('值')

    plt.title(title or '数据可视化')
    if data_type not in ['pie', 'radar', 'heatmap']:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return {
        'type': data_type,
        'labels': labels,
        'values': values,
        'title': title,
        'options': options,
        'image': base64.b64encode(image_png).decode('utf-8')
    }

def visualize_feature_importance(model, feature_names, n_features=10):
    """可视化特征重要性"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        top_n = min(n_features, len(feature_names))  # 最多显示前n个特征
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')
        plt.title('特征重要性')

        # 使用渐变色彩
        colors = generate_gradient_colors(top_n)

        plt.bar(range(top_n), importances[indices][:top_n], color=colors, align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=45, ha='right')
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.tight_layout()

        # 将图像转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        # 同时生成表格数据
        table_data = {
            'columns': ['特征', '重要性'],
            'data': [[feature_names[i], float(importances[indices][j])] for j, i in enumerate(indices[:top_n])]
        }

        return {
            'type': 'bar',
            'labels': [feature_names[i] for i in indices][:top_n],
            'values': importances[indices][:top_n].tolist(),
            'title': '特征重要性',
            'image': base64.b64encode(image_png).decode('utf-8'),
            'table_data': table_data
        }
    return None

def visualize_confusion_matrix(y_true, y_pred, class_names=None):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()

    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    # 计算精确率、召回率等指标
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)

    # 处理除以零的情况
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)

    # 生成表格数据
    metrics_data = []
    for i, class_name in enumerate(class_names):
        metrics_data.append([
            class_name,
            int(np.sum(cm[i, :])),  # 该类的总样本数
            int(cm[i, i]),  # 正确预测数
            float(precision[i]),  # 精确率
            float(recall[i])  # 召回率
        ])

    table_data = {
        'columns': ['类别', '样本数', '正确预测', '精确率', '召回率'],
        'data': metrics_data
    }

    return {
        'type': 'confusion_matrix',
        'matrix': cm.tolist(),
        'class_names': class_names,
        'image': base64.b64encode(image_png).decode('utf-8'),
        'table_data': table_data
    }

def visualize_metrics(metrics_dict):
    """可视化评估指标"""
    # 过滤掉非数值型指标
    numeric_metrics = {}
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric_metrics[k] = v

    if not numeric_metrics:
        return None

    plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    # 使用渐变色彩
    colors = generate_gradient_colors(len(numeric_metrics))

    plt.bar(numeric_metrics.keys(), numeric_metrics.values(), color=colors)
    plt.title('模型评估指标')
    plt.ylim(0, max(1.0, max(numeric_metrics.values()) * 1.1))  # 根据数值动态调整Y轴
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    # 生成表格数据
    table_data = {
        'columns': ['指标', '值'],
        'data': [[k, v] for k, v in numeric_metrics.items()]
    }

    return {
        'type': 'bar',
        'labels': list(numeric_metrics.keys()),
        'values': list(numeric_metrics.values()),
        'title': '模型评估指标',
        'image': base64.b64encode(image_png).decode('utf-8'),
        'table_data': table_data
    }

def visualize_clusters(X, labels, feature_names=None, method='pca'):
    """可视化聚类结果"""
    # 如果特征维度大于2，使用降维
    if X.shape[1] > 2:
        if method == 'pca':
            reducer = PCA(n_components=2)
        else:  # 使用t-SNE
            reducer = TSNE(n_components=2, random_state=42)

        X_2d = reducer.fit_transform(X)
    else:
        X_2d = X

    # 获取唯一的聚类标签
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # 生成颜色
    colors = generate_gradient_colors(n_clusters)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[colors[i]],
                   label=f'聚类 {label}',
                   alpha=0.7,
                   s=80,
                   edgecolors='w')

    plt.legend()
    plt.title('聚类可视化')
    if method == 'pca':
        plt.xlabel('主成分 1')
        plt.ylabel('主成分 2')
    else:
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    # 计算每个聚类的样本数量
    cluster_counts = np.bincount(labels.astype(int))

    # 生成表格数据
    table_data = {
        'columns': ['聚类', '样本数量', '比例'],
        'data': [[f'聚类 {i}', int(count), float(count)/len(labels)] for i, count in enumerate(cluster_counts)]
    }

    return {
        'type': 'scatter',
        'image': base64.b64encode(image_png).decode('utf-8'),
        'table_data': table_data,
        'method': method,
        'n_clusters': n_clusters
    }

def generate_data_table(data, columns=None, max_rows=100):
    """将数据转换为表格格式"""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and all(isinstance(data[k], (list, np.ndarray)) for k in data):
        df = pd.DataFrame(data)
    else:
        try:
            df = pd.DataFrame(data)
        except:
            return None

    # 截取数据
    if len(df) > max_rows:
        df = df.head(max_rows)

    # 使用指定列或所有列
    if columns:
        df = df[columns]

    # 转换为表格数据
    return {
        'columns': df.columns.tolist(),
        'data': df.values.tolist()
    }

# 添加特征重要性雷达图可视化
def visualize_feature_importance_radar(feature_importance, title='特征重要性雷达图'):
    """生成特征重要性雷达图"""
    # 准备数据
    features = list(feature_importance.keys())
    values = list(feature_importance.values())

    # 确保值为正数，并且标准化到0-1范围
    values = np.array(values)
    if np.any(values < 0):
        # 对于有负值的情况，使用MinMaxScaler将范围缩放到0-1
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
    else:
        # 只需要标准化到0-1范围
        values = values / np.max(values) if np.max(values) > 0 else values

    # 创建雷达图
    plt.figure(figsize=(10, 8))

    # 计算角度变量
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)

    # 闭合图形
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    features = features + [features[0]]

    # 绘制雷达图
    ax = plt.subplot(111, polar=True)
    ax.fill(angles, values, color='#4F46E5', alpha=0.25)
    ax.plot(angles, values, 'o-', color='#4F46E5', linewidth=2)

    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features[:-1])

    # 设置y轴刻度
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])

    # 添加标题
    plt.title(title)

    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return {
        'type': 'radar',
        'labels': features[:-1],  # 移除闭合点
        'values': values[:-1].tolist(),  # 移除闭合点
        'title': title,
        'image': base64.b64encode(image_png).decode('utf-8')
    }

# 可视化模型比较结果
def visualize_model_comparison(comparison_results, metric='auto'):
    """
    可视化模型比较结果

    Args:
        comparison_results: 模型比较结果
        metric: 使用的指标，'auto'表示自动选择

    Returns:
        可视化数据
    """
    models = comparison_results['models']

    # 按模型类型分组
    classifiers = [model for model in models if model.get('is_classifier', False)]
    regressors = [model for model in models if not model.get('is_classifier', False) and 'error' not in model]

    visualizations = []
    tables = []

    # 处理分类器
    if classifiers:
        model_names = [model['model_name'] for model in classifiers]

        # 为每个指标创建条形图
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric_name in metrics:
            values = [model['metrics'].get(metric_name, 0) for model in classifiers]

            plt.figure(figsize=(10, 6))
            plt.style.use('ggplot')

            colors = generate_gradient_colors(len(values))
            plt.bar(model_names, values, color=colors)
            plt.title(f'分类器比较 - {metric_name}')
            plt.ylim(0, 1.05)  # 分类指标通常在0-1范围内
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # 将图像转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()

            visualizations.append({
                'type': 'bar',
                'title': f'分类器比较 - {metric_name}',
                'labels': model_names,
                'values': values,
                'image': base64.b64encode(image_png).decode('utf-8')
            })

        # 为分类器创建综合表格
        classifier_table = {
            'columns': ['模型', '准确率', '精确率', '召回率', 'F1分数'],
            'data': [
                [
                    model['model_name'],
                    model['metrics'].get('accuracy', '-'),
                    model['metrics'].get('precision', '-'),
                    model['metrics'].get('recall', '-'),
                    model['metrics'].get('f1', '-')
                ]
                for model in classifiers
            ]
        }
        tables.append({'title': '分类器性能对比', 'data': classifier_table})

    # 处理回归器
    if regressors:
        model_names = [model['model_name'] for model in regressors]

        # 为每个指标创建条形图
        metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric_name in metrics:
            values = [model['metrics'].get(metric_name, 0) for model in regressors]

            # 对于R2，我们希望值越高越好；对于其他误差指标，值越低越好
            if metric_name == 'r2':
                plt.figure(figsize=(10, 6))
                plt.style.use('ggplot')

                colors = generate_gradient_colors(len(values))
                plt.bar(model_names, values, color=colors)
                plt.title(f'回归器比较 - {metric_name}')
                plt.ylim(min(0, min(values) - 0.1), max(1, max(values) + 0.1))
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
            else:
                plt.figure(figsize=(10, 6))
                plt.style.use('ggplot')

                colors = generate_gradient_colors(len(values))
                plt.bar(model_names, values, color=colors)
                plt.title(f'回归器比较 - {metric_name} (越低越好)')
                plt.ylim(0, max(values) * 1.2)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()

            # 将图像转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            plt.close()

            visualizations.append({
                'type': 'bar',
                'title': f'回归器比较 - {metric_name}',
                'labels': model_names,
                'values': values,
                'image': base64.b64encode(image_png).decode('utf-8')
            })

        # 为回归器创建综合表格
        regressor_table = {
            'columns': ['模型', 'MSE', 'RMSE', 'MAE', 'R²'],
            'data': [
                [
                    model['model_name'],
                    model['metrics'].get('mse', '-'),
                    model['metrics'].get('rmse', '-'),
                    model['metrics'].get('mae', '-'),
                    model['metrics'].get('r2', '-')
                ]
                for model in regressors
            ]
        }
        tables.append({'title': '回归器性能对比', 'data': regressor_table})

    # 创建最佳模型综合视图
    if comparison_results.get('best_classifier') or comparison_results.get('best_regressor'):
        best_model_info = []

        if comparison_results.get('best_classifier'):
            best_model_info.append({
                'type': '最佳分类器',
                'name': comparison_results['best_classifier'],
                'metric': 'F1分数'
            })

        if comparison_results.get('best_regressor'):
            best_model_info.append({
                'type': '最佳回归器',
                'name': comparison_results['best_regressor'],
                'metric': 'R²分数'
            })

        best_model_table = {
            'columns': ['模型类型', '模型名称', '评估指标'],
            'data': [[info['type'], info['name'], info['metric']] for info in best_model_info]
        }
        tables.append({'title': '最佳模型', 'data': best_model_table})

    return {
        'visualizations': visualizations,
        'tables': tables
    }

# 可视化模型解释结果
def visualize_model_explanation(explanation_result):
    """
    可视化模型解释结果

    Args:
        explanation_result: 模型解释结果

    Returns:
        可视化数据
    """
    visualizations = []
    tables = []

    # 特征重要性可视化
    feature_importance = explanation_result.get('feature_importance', {})
    if feature_importance:
        # 转换为排序后的列表
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:10]  # 最多展示前10个特征

        feature_names = [item[0] for item in top_features]
        importance_values = [item[1] for item in top_features]

        # 创建条形图
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')

        colors = generate_gradient_colors(len(top_features))
        bars = plt.bar(feature_names, importance_values, color=colors)

        # 添加正负值不同颜色
        for i, value in enumerate(importance_values):
            if value < 0:
                bars[i].set_color('tomato')

        plt.title('特征重要性')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 将图像转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        visualizations.append({
            'type': 'bar',
            'title': '特征重要性',
            'labels': feature_names,
            'values': importance_values,
            'image': base64.b64encode(image_png).decode('utf-8')
        })

        # 如果有足够多的特征，创建雷达图
        if len(feature_importance) >= 3:
            radar_viz = visualize_feature_importance_radar(
                {k: abs(v) for k, v in feature_importance.items()},
                title='特征重要性雷达图'
            )
            visualizations.append(radar_viz)

        # 创建特征重要性表格
        feature_table = {
            'columns': ['特征', '重要性'],
            'data': [[name, value] for name, value in sorted_features]
        }
        tables.append({'title': '特征重要性', 'data': feature_table})

    # 特征贡献可视化
    feature_contributions = explanation_result.get('feature_contributions', [])
    if feature_contributions:
        # 最多展示前10个特征贡献
        top_contributions = feature_contributions[:10]

        feature_names = [item['feature'] for item in top_contributions]
        contribution_values = [item['contribution'] for item in top_contributions]

        # 创建条形图
        plt.figure(figsize=(10, 6))
        plt.style.use('ggplot')

        colors = generate_gradient_colors(len(top_contributions))
        bars = plt.bar(feature_names, contribution_values, color=colors)

        # 添加正负值不同颜色
        for i, value in enumerate(contribution_values):
            if value < 0:
                bars[i].set_color('tomato')

        plt.title('特征贡献')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()

        # 将图像转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()

        visualizations.append({
            'type': 'bar',
            'title': '特征贡献',
            'labels': feature_names,
            'values': contribution_values,
            'image': base64.b64encode(image_png).decode('utf-8')
        })

        # 创建特征贡献表格
        contribution_table = {
            'columns': ['特征', '值', '重要性', '贡献'],
            'data': [
                [
                    item['feature'],
                    item['value'],
                    item['importance'],
                    item['contribution']
                ]
                for item in feature_contributions
            ]
        }
        tables.append({'title': '特征贡献', 'data': contribution_table})

    # 添加预测结果表格
    prediction = explanation_result.get('prediction', [])
    if prediction:
        prediction_table = {
            'columns': ['预测结果'],
            'data': [[p] for p in prediction]
        }
        tables.append({'title': '预测结果', 'data': prediction_table})

    return {
        'visualizations': visualizations,
        'tables': tables
    }

# 创建机器学习工具
def create_ml_tools():
    """创建机器学习工具集"""

    def _train_model(
        model_type: str,
        data_path: str,
        target_column: str,
        model_name: str = None,
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None
    ) -> str:
        """训练一个机器学习模型"""
        try:
            # Handle .xlsx input: convert to CSV before calling actual_train_model
            processed_data_path = data_path
            if data_path.lower().endswith( ('.xls', '.xlsx') ):
                try:
                    df = pd.read_excel(data_path)
                    # Create a temporary CSV file path in the uploads directory
                    # Generate a unique filename to avoid conflicts
                    temp_csv_filename = f"{os.path.splitext(os.path.basename(data_path))[0]}_temp_{uuid.uuid4().hex}.csv"
                    uploads_dir = os.path.join(os.getcwd(), 'uploads') # Assuming 'uploads' is in the current working directory
                    os.makedirs(uploads_dir, exist_ok=True)
                    processed_data_path = os.path.join(uploads_dir, temp_csv_filename)
                    df.to_csv(processed_data_path, index=False)
                    print(f"Converted Excel file {data_path} to temporary CSV: {processed_data_path}")
                except Exception as e:
                    return json.dumps({"text": f"❌ 转换Excel文件为CSV时发生错误: {str(e)}", "error": f"转换Excel文件为CSV时发生错误: {str(e)}"}) # Unicode for cross mark

            result = actual_train_model(
                model_type=model_type,
                data=processed_data_path, # Use the potentially converted path
                target_column=target_column,
                model_name=model_name,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns
            )
            
            # Optionally, clean up the temporary CSV file after training
            if processed_data_path != data_path and os.path.exists(processed_data_path):
                 try:
                     os.remove(processed_data_path)
                     print(f"Cleaned up temporary CSV file: {processed_data_path}")
                 except Exception as e:
                     print(f"Warning: Failed to clean up temporary CSV file {processed_data_path}: {e}")

            # Format result for Agent output
            formatted_result = {
                "model_name": result.get("model_name", model_name or model_type),
                "model_type": result.get("model_type", model_type),
                "metrics": result.get("metrics", {}),
                "message": f"✅ 成功训练了{result.get('model_type', model_type)}模型。模型名称为: {result.get('model_name', '')}。评估指标: {json.dumps(result.get('metrics', {}), ensure_ascii=False)}"
            }
            return json.dumps(formatted_result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ 训练模型时发生错误: {str(e)}", "error": str(e)}) # Unicode for cross mark

    def _predict_with_model(model_name: str, input_data: Dict[str, Any]) -> str:
        """使用训练好的模型进行预测"""
        try:
            result = actual_predict(model_name=model_name, input_data=input_data)

            # 尝试加载模型获取更多信息
            model, _, metadata = load_model(model_name)
            model_info = {}

            # 检查是分类还是回归
            is_classification = hasattr(model, "predict_proba")

            # 格式化输出
            predictions = result["predictions"]
            pred_text = ""
            if len(predictions) == 1:  # 单个预测结果
                pred_text = f"预测结果: {predictions[0]}"

                # 如果是分类模型并且有概率输出
                if is_classification and "probabilities" in result:
                    probs = result["probabilities"][0]

                    # 可视化概率分布
                    if isinstance(probs, dict):  # 类别到概率的映射
                        classes = list(probs.keys())
                        probabilities = list(probs.values())

                        vis_data = generate_visualization(
                            'bar',
                            classes,
                            probabilities,
                            title='预测概率分布'
                        )

                        table_data = {
                            'columns': ['类别', '概率'],
                            'data': [[c, p] for c, p in zip(classes, probabilities)]
                        }

                        return json.dumps({
                            "text": pred_text,
                            "visualization_data": vis_data,
                            "table_data": table_data,
                            "predictions": predictions,
                            "probabilities": probs
                        })
            else:  # 多个预测结果
                pred_text = f"预测结果:\n" + "\n".join([f"- {p}" for p in predictions[:10]])
                if len(predictions) > 10:
                    pred_text += f"\n... 等 {len(predictions)} 条结果"

                # 生成预测结果表格
                table_data = {
                    'columns': ['索引', '预测值'],
                    'data': [[i, p] for i, p in enumerate(predictions[:100])]
                }

                # 如果是分类预测，尝试可视化类别分布
                if is_classification:
                    # 统计类别频率
                    from collections import Counter
                    counts = Counter(predictions)
                    categories = list(counts.keys())
                    frequencies = list(counts.values())

                    vis_data = generate_visualization(
                        'pie',
                        categories,
                        frequencies,
                        title='预测类别分布'
                    )

                    return json.dumps({
                        "text": pred_text,
                        "visualization_data": vis_data,
                        "table_data": table_data,
                        "predictions": predictions
                    })

            return json.dumps({
                "text": pred_text,
                "predictions": predictions
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 预测时发生错误: {str(e)}",
                "error": str(e)
            })

    def _list_models() -> str:
        """列出所有可用的模型"""
        try:
            models = actual_list_models()

            if not models:
                return json.dumps({
                    "text": "📚 没有找到任何已训练的模型。",
                    "models": []
                })

            # 生成表格数据
            table_data = {
                'columns': ['模型名称', '模型类型', '特征数量', '训练时间'],
                'data': []
            }

            # 收集模型类型计数用于饼图
            model_types = {}

            model_list_text = "📚 可用模型列表:\n\n"
            for model in models:
                model_list_text += f"- {model['name']} ({model['type']})\n"

                # 添加到表格数据
                table_data['data'].append([
                    model['name'],
                    model['type'],
                    model.get('n_features', '未知'),
                    model.get('created_at', '未知')
                ])

                # 统计模型类型
                model_type = model['type']
                if model_type in model_types:
                    model_types[model_type] += 1
                else:
                    model_types[model_type] = 1

            # 创建模型类型分布图
            types = list(model_types.keys())
            counts = list(model_types.values())

            if types:
                vis_data = generate_visualization(
                    'pie',
                    types,
                    counts,
                    title='模型类型分布'
                )
            else:
                vis_data = None

            return json.dumps({
                "text": model_list_text,
                "visualization_data": vis_data,
                "table_data": table_data,
                "models": models
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 获取模型列表时发生错误: {str(e)}",
                "error": str(e)
            })

    def _recommend_model(task_description: str) -> str:
        """根据任务描述推荐模型"""
        try:
            result = actual_select_model(task_description)

            if not result:
                return json.dumps({
                    "text": "❌ 无法为此任务推荐合适的模型。请提供更详细的任务描述，或尝试使用通用模型如随机森林。"
                })

            # 推荐多个模型
            recommendations = result.get('recommendations', [])
            if recommendations:
                rec_text = f"📊 根据任务 \"{task_description}\" 的推荐模型:\n\n"

                for idx, rec in enumerate(recommendations):
                    model_type = rec.get('model_type', '未知')
                    confidence = rec.get('confidence', 0) * 100
                    reason = rec.get('reason', '无说明')

                    rec_text += f"{idx+1}. {model_type} (置信度: {confidence:.1f}%)\n   原因: {reason}\n\n"

                # 生成可视化
                models = [r.get('model_type', '其他') for r in recommendations]
                scores = [r.get('confidence', 0) * 100 for r in recommendations]

                vis_data = generate_visualization(
                    'bar',
                    models,
                    scores,
                    title='模型推荐评分'
                )

                # 生成表格
                table_data = {
                    'columns': ['模型类型', '置信度', '推荐理由'],
                    'data': [
                        [r.get('model_type', '未知'), f"{r.get('confidence', 0)*100:.1f}%", r.get('reason', '无说明')]
                        for r in recommendations
                    ]
                }

                return json.dumps({
                    "text": rec_text,
                    "visualization_data": vis_data,
                    "table_data": table_data,
                    "recommendations": recommendations
                })
            else:
                rec_model = result.get('recommended_model', '未知')
                reason = result.get('reason', '无说明')

                return json.dumps({
                    "text": f"📊 对于任务 \"{task_description}\"，推荐使用 {rec_model} 模型。\n\n原因: {reason}",
                    "recommended_model": rec_model,
                    "reason": reason
                })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 推荐模型时发生错误: {str(e)}",
                "error": str(e)
            })

    def _data_analysis(file_path: str, target_column: Optional[str] = None, analysis_type: Optional[str] = None) -> str:
        """分析数据集并提供统计信息和可视化"""
        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return json.dumps({
                    "text": f"❌ 文件 {file_path} 不存在。"
                })

            # 根据文件类型加载数据
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                return json.dumps({
                    "text": f"❌ 不支持的文件格式。目前支持 CSV 和 Excel 文件。"
                })

            # 基本数据信息
            n_rows, n_cols = df.shape
            missing_values = df.isnull().sum().sum()
            missing_percent = missing_values / (n_rows * n_cols) * 100

            # 数据类型分析
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_cols = df.select_dtypes(include=numerics).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

            # 基本统计信息
            basic_stats = df.describe().transpose()

            # 准备响应文本
            analysis_text = (f"📊 数据分析报告 - {file_path.split('/')[-1]}\n\n"
                           f"▶ 基本信息:\n"
                           f"  - 行数: {n_rows}\n"
                           f"  - 列数: {n_cols}\n"
                           f"  - 缺失值: {missing_values} ({missing_percent:.2f}%)\n\n"
                           f"▶ 列类型分布:\n"
                           f"  - 数值型: {len(numeric_cols)} 列\n"
                           f"  - 类别型: {len(categorical_cols)} 列\n"
                           f"  - 时间型: {len(datetime_cols)} 列\n")

            # 准备可视化
            visualizations = []
            tables = []

            # 1. 列类型分布饼图
            type_vis = generate_visualization(
                'pie',
                ['数值型', '类别型', '时间型'],
                [len(numeric_cols), len(categorical_cols), len(datetime_cols)],
                title='列类型分布'
            )
            visualizations.append(type_vis)

            # 2. 为数值型列创建统计表格
            if numeric_cols:
                numeric_stats = df[numeric_cols].describe().transpose().reset_index()
                numeric_stats.columns = ['列名', '计数', '均值', '标准差', '最小值', '25%', '50%', '75%', '最大值']

                # 对齐小数位数
                for col in numeric_stats.columns[2:]:
                    numeric_stats[col] = numeric_stats[col].apply(lambda x: f"{x:.4f}")

                numeric_table = {
                    'columns': numeric_stats.columns.tolist(),
                    'data': numeric_stats.values.tolist()
                }
                tables.append({
                    'title': '数值型列统计',
                    'data': numeric_table
                })

                # 3. 生成一些数值列的分布图 (选择前5个)
                for i, col in enumerate(numeric_cols[:5]):
                    if df[col].nunique() > 1:  # 确保不是常数列
                        try:
                            plt.figure(figsize=(10, 6))
                            sns.histplot(df[col].dropna(), kde=True)
                            plt.title(f'{col} 分布')
                            plt.tight_layout()

                            # 将图像转换为base64
                            buffer = BytesIO()
                            plt.savefig(buffer, format='png', dpi=100)
                            buffer.seek(0)
                            image_png = buffer.getvalue()
                            buffer.close()
                            plt.close()

                            visualizations.append({
                                'type': 'histogram',
                                'title': f'{col} 分布',
                                'image': base64.b64encode(image_png).decode('utf-8')
                            })
                        except:
                            # 略过无法可视化的列
                            pass

            # 4. 类别型列的频率分析
            if categorical_cols:
                cat_tables = []
                for col in categorical_cols[:5]:  # 分析前5个类别列
                    if df[col].nunique() <= 20:  # 限制值的数量
                        freq = df[col].value_counts().reset_index()
                        freq.columns = ['值', '频率']
                        freq['百分比'] = freq['频率'] / freq['频率'].sum() * 100
                        freq['百分比'] = freq['百分比'].apply(lambda x: f"{x:.2f}%")

                        cat_table = {
                            'columns': freq.columns.tolist(),
                            'data': freq.values.tolist()
                        }

                        cat_tables.append({
                            'title': f'{col} 频率分布',
                            'data': cat_table
                        })

                        # 生成类别频率图
                        try:
                            top_n = min(10, df[col].nunique())
                            top_cats = df[col].value_counts().nlargest(top_n)

                            plt.figure(figsize=(10, 6))
                            sns.barplot(x=top_cats.index, y=top_cats.values)
                            plt.title(f'{col} 前{top_n}类别频率')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()

                            # 将图像转换为base64
                            buffer = BytesIO()
                            plt.savefig(buffer, format='png', dpi=100)
                            buffer.seek(0)
                            image_png = buffer.getvalue()
                            buffer.close()
                            plt.close()

                            visualizations.append({
                                'type': 'bar',
                                'title': f'{col} 前{top_n}类别频率',
                                'image': base64.b64encode(image_png).decode('utf-8')
                            })
                        except:
                            # 略过无法可视化的列
                            pass

                tables.extend(cat_tables)

            # 5. 相关性分析
            if analysis_type == 'feature_relevance' and target_column and target_column in df.columns and numeric_cols:
                analysis_text += f"\n🎯 与目标变量 '{target_column}' 相关的特征分析:\n"
                # 确保目标列是数值类型才能计算相关性
                if target_column not in numeric_cols:
                    analysis_text += f"警告: 目标列 '{target_column}' 不是数值类型，无法计算其与其他数值特征的相关系数。\n"
                else:
                    try:
                        # 计算所有数值列与目标列的相关性
                        target_corr = df[numeric_cols].corr()[target_column].sort_values(ascending=False)
                        analysis_text += "特征与目标变量的相关性:\n"
                        # 过滤掉目标列自身的相关性（值为1）
                        filtered_target_corr = target_corr[target_corr.index != target_column]
                        analysis_text += filtered_target_corr.to_string() + "\n"

                        # 可视化与目标变量的相关性 (条形图)
                        plt.figure(figsize=(10, max(6, len(filtered_target_corr) * 0.3)))
                        # 使用generate_gradient_colors生成颜色列表
                        colors_for_target_corr = generate_gradient_colors(len(filtered_target_corr))
                        filtered_target_corr.plot(kind='barh', color=colors_for_target_corr)
                        plt.title(f'特征与目标变量 "{target_column}" 的相关性')
                        plt.xlabel('相关系数')
                        plt.tight_layout()
                        
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png', dpi=100)
                        buffer.seek(0)
                        image_png = buffer.getvalue()
                        buffer.close()
                        plt.close()

                        visualizations.append({
                            'type': 'bar',
                            'title': f'特征与目标变量 "{target_column}" 的相关性',
                            'labels': filtered_target_corr.index.tolist(),
                            'values': filtered_target_corr.values.tolist(),
                            'image': base64.b64encode(image_png).decode('utf-8')
                        })

                        tables.append({
                            'title': f'与目标 "{target_column}" 的相关性',
                            'data': {
                                'columns': ['特征', '相关系数'],
                                'data': [[idx, round(val, 4)] for idx, val in filtered_target_corr.items()]
                            }
                        })
                    except Exception as e_corr_target:
                        analysis_text += f"计算与目标变量 '{target_column}' 相关性时出错: {str(e_corr_target)}\n"
                        # pass # 保持错误信息可见
            
            elif len(numeric_cols) >= 2: # 执行一般的相关性分析 (当 analysis_type 不是 feature_relevance 或 target_column 未提供时)
                try:
                    corr = df[numeric_cols].corr()

                    # 相关性热图
                    plt.figure(figsize=(12, 10))
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
                               linewidths=.5, cbar_kws={'shrink': .8})
                    plt.title('特征相关性矩阵 (所有数值特征)')
                    plt.tight_layout()

                    # 将图像转换为base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100)
                    buffer.seek(0)
                    image_png = buffer.getvalue()
                    buffer.close()
                    plt.close()

                    visualizations.append({
                        'type': 'heatmap',
                        'title': '特征相关性矩阵',
                        'image': base64.b64encode(image_png).decode('utf-8')
                    })

                    # 提取强相关特征对
                    strong_corr = []
                    for i in range(len(corr.columns)):
                        for j in range(i+1, len(corr.columns)):
                            if abs(corr.iloc[i, j]) > 0.5:  # 强相关阈值
                                strong_corr.append([
                                    corr.columns[i],
                                    corr.columns[j],
                                    corr.iloc[i, j]
                                ])

                    # 提取强相关特征对 (绝对值大于0.7)
                    strong_corr_pairs = []
                    for i in range(len(corr.columns)):
                        for j in range(i + 1, len(corr.columns)):
                            if abs(corr.iloc[i, j]) > 0.7:
                                strong_corr_pairs.append([
                                    corr.columns[i],
                                    corr.columns[j],
                                    round(corr.iloc[i, j], 4)
                                ])
                    # 按相关性绝对值降序排序，取前15个
                    strong_corr_pairs_sorted = sorted(strong_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:15]

                    if strong_corr_pairs_sorted:
                        corr_table_data = {
                            'columns': ['特征1', '特征2', '相关系数'],
                            'data': strong_corr_pairs_sorted
                        }
                        tables.append({
                            'title': '强相关特征对 (Top 15, |相关系数| > 0.7)',
                            'data': corr_table_data
                        })
                except Exception as e_corr_general:
                    analysis_text += f"计算通用相关性矩阵时出错: {str(e_corr_general)}\n"
                    # pass

            # 6. 数据预览表格
            preview_table = {
                'columns': df.columns.tolist(),
                'data': df.head(10).values.tolist()
            }
            tables.append({
                'title': '数据预览',
                'data': preview_table
            })

            # 返回JSON响应
            return json.dumps({
                "text": analysis_text,
                "visualizations": visualizations,
                "tables": tables,
                "df_info": {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "numeric_columns": numeric_cols,
                    "categorical_columns": categorical_cols,
                    "datetime_columns": datetime_cols
                }
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 分析数据时发生错误: {str(e)}",
                "error": str(e)
            })

    def _evaluate_model(model_name: str, test_data_path: str, target_column: str) -> str:
        """
        评估模型在测试数据上的表现
        
        参数:
            model_name: 模型名称
            test_data_path: 测试数据文件路径
            target_column: 目标列名
            
        返回:
            JSON字符串包含评估结果或错误信息
        """
        try:
            # 记录评估开始
            print(f"[INFO] 开始评估模型 {model_name}...")
            
            # 检查模型和数据是否存在
            model_path = os.path.join('ml_models', f"{model_name}.pkl")
            if not os.path.exists(model_path):
                error_msg = f"❌ 模型 {model_name} 不存在。"
                print(f"[ERROR] {error_msg}")
                return json.dumps({
                    "text": error_msg,
                    "error": "MODEL_NOT_FOUND"
                })

            if not os.path.exists(test_data_path):
                error_msg = f"❌ 测试数据文件 {test_data_path} 不存在。"
                print(f"[ERROR] {error_msg}")
                return json.dumps({
                    "text": error_msg,
                    "error": "TEST_DATA_NOT_FOUND"
                })

            # 加载模型
            model, preprocessors, _ = load_model(model_name)

            # 加载测试数据
            if test_data_path.endswith('.csv'):
                test_data = pd.read_csv(test_data_path)
            elif test_data_path.endswith(('.xls', '.xlsx')):
                test_data = pd.read_excel(test_data_path)
            else:
                error_msg = "❌ 不支持的测试数据格式。目前支持 CSV 和 Excel 文件。"
                print(f"[ERROR] {error_msg}")
                return json.dumps({
                    "text": error_msg,
                    "error": "UNSUPPORTED_FILE_FORMAT"
                })

            # 检查目标列是否存在
            if target_column not in test_data.columns:
                error_msg = f"❌ 目标列 {target_column} 在测试数据中不存在。"
                print(f"[ERROR] {error_msg}")
                return json.dumps({
                    "text": error_msg,
                    "error": "TARGET_COLUMN_NOT_FOUND"
                })

            # 准备测试数据
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            # 应用预处理
            if 'label_encoders' in preprocessors:
                for col, encoder in preprocessors['label_encoders'].items():
                    if col in X_test.columns:
                        X_test[col] = X_test[col].astype(str)
                        try:
                            X_test[col] = encoder.transform(X_test[col])
                        except:
                            # 处理未知类别
                            X_test[col] = X_test[col].map(lambda x: 0 if x not in encoder.classes_ else encoder.transform([x])[0])

            if 'scaler' in preprocessors and hasattr(preprocessors['scaler'], 'transform'):
                if hasattr(preprocessors['scaler'], 'feature_names_in_'):
                    # Scikit-learn 1.0+
                    common_cols = [col for col in preprocessors['scaler'].feature_names_in_ if col in X_test.columns]
                    if common_cols:
                        X_test[common_cols] = preprocessors['scaler'].transform(X_test[common_cols])
            else:
                    # 尝试识别数值列并缩放
                    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                    numeric_cols = X_test.select_dtypes(include=numerics).columns.tolist()
                    if numeric_cols:
                        X_test[numeric_cols] = preprocessors['scaler'].transform(X_test[numeric_cols])

            # 进行预测
            y_pred = model.predict(X_test)

            # 检查模型类型
            is_classifier = hasattr(model, "predict_proba")
            is_regressor = not is_classifier

            # 根据模型类型计算评估指标
            metrics = {}
            if is_classifier:
                # 分类指标
                metrics["accuracy"] = accuracy_score(y_test, y_pred)
                metrics["precision"] = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                metrics["recall"] = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                metrics["f1"] = f1_score(y_test, y_pred, average="weighted", zero_division=0)

                # 获取类别名称
                if 'label_encoders' in preprocessors and target_column in preprocessors['label_encoders']:
                    class_names = preprocessors['label_encoders'][target_column].classes_
                else:
                    class_names = sorted(set(y_test.unique()))

                # 详细的分类报告
                report = classification_report(y_test, y_pred, output_dict=True)

                # 混淆矩阵可视化
                cm_vis = visualize_confusion_matrix(y_test, y_pred, class_names=class_names)

                # 为分类报告创建表格
                report_data = []
                for class_name, metrics_dict in report.items():
                    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                        report_data.append([
                            class_name,
                            metrics_dict['precision'],
                            metrics_dict['recall'],
                            metrics_dict['f1-score'],
                            metrics_dict['support']
                        ])

                # 添加平均指标
                for avg_type in ['macro avg', 'weighted avg']:
                    if avg_type in report:
                        report_data.append([
                            avg_type,
                            report[avg_type]['precision'],
                            report[avg_type]['recall'],
                            report[avg_type]['f1-score'],
                            report[avg_type]['support']
                        ])

                report_table = {
                    'columns': ['类别', '精确率', '召回率', 'F1分数', '样本数'],
                    'data': report_data
                }

                # 创建指标可视化
                metrics_vis = visualize_metrics(metrics)

                # 格式化评估结果文本
                eval_text = (f"📊 模型 {model_name} 评估结果:\n\n"
                           f"▶ 模型类型: 分类器\n"
                           f"▶ 测试样本数: {len(y_test)}\n"
                           f"▶ 主要指标:\n"
                           f"  - 准确率: {metrics['accuracy']:.4f}\n"
                           f"  - 精确率: {metrics['precision']:.4f}\n"
                           f"  - 召回率: {metrics['recall']:.4f}\n"
                           f"  - F1分数: {metrics['f1']:.4f}\n")

                return json.dumps({
                    "text": eval_text,
                    "visualizations": [cm_vis, metrics_vis],
                    "tables": [
                        {'title': '分类报告', 'data': report_table},
                        {'title': '混淆矩阵', 'data': cm_vis.get('table_data')}
                    ],
                    "metrics": metrics,
                    "model_type": "classifier"
                })

            else:
                # 回归指标
                metrics["mse"] = mean_squared_error(y_test, y_pred)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                metrics["mae"] = mean_absolute_error(y_test, y_pred)
                metrics["r2"] = r2_score(y_test, y_pred)

                # 创建预测vs实际值散点图
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('实际值')
                plt.ylabel('预测值')
                plt.title('预测值 vs 实际值')
                plt.tight_layout()

                # 将图像转换为base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plt.close()

                scatter_vis = {
                    'type': 'scatter',
                    'title': '预测值 vs 实际值',
                    'image': base64.b64encode(image_png).decode('utf-8')
                }

                # 创建残差图
                residuals = y_test - y_pred
                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('预测值')
                plt.ylabel('残差')
                plt.title('残差图')
                plt.tight_layout()

                # 将图像转换为base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                image_png = buffer.getvalue()
                buffer.close()
                plt.close()

                residual_vis = {
                    'type': 'scatter',
                    'title': '残差图',
                    'image': base64.b64encode(image_png).decode('utf-8')
                }

                # 创建指标可视化
                metrics_vis = visualize_metrics(metrics)

                # 为残差创建表格（前20个样本）
                residual_data = pd.DataFrame({
                    '实际值': y_test.iloc[:20],
                    '预测值': y_pred[:20],
                    '残差': residuals.iloc[:20]
                }).reset_index().rename(columns={'index': '样本索引'})

                residual_table = {
                    'columns': residual_data.columns.tolist(),
                    'data': residual_data.values.tolist()
                }

                # 格式化评估结果文本
                eval_text = (f"📊 模型 {model_name} 评估结果:\n\n"
                           f"▶ 模型类型: 回归器\n"
                           f"▶ 测试样本数: {len(y_test)}\n"
                           f"▶ 主要指标:\n"
                           f"  - MSE (均方误差): {metrics['mse']:.4f}\n"
                           f"  - RMSE (均方根误差): {metrics['rmse']:.4f}\n"
                           f"  - MAE (平均绝对误差): {metrics['mae']:.4f}\n"
                           f"  - R² (决定系数): {metrics['r2']:.4f}\n")

                return json.dumps({
                    "text": eval_text,
                    "visualizations": [scatter_vis, residual_vis, metrics_vis],
                    "tables": [
                        {'title': '残差分析', 'data': residual_table}
                    ],
                    "metrics": metrics,
                    "model_type": "regressor"
                })

        except Exception as e:
            import traceback
            error_msg = f"❌ 评估模型时发生错误: {str(e)}"
            print(f"[ERROR] {error_msg}")
            print(f"[DEBUG] 错误详情: {traceback.format_exc()}")
            return json.dumps({
                "text": error_msg,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            })

    # 添加新的工具函数
    def _create_ensemble_model(
        base_models: List[str],
        ensemble_type: str = 'voting',
        weights: List[float] = None,
        save_name: str = None
    ) -> str:
        """创建集成模型"""
        try:
            # 将 voting_classifier 映射到 voting
            if ensemble_type == 'voting_classifier':
                ensemble_type = 'voting'
            
            result = create_ensemble_model(
                base_models=base_models,
                ensemble_type=ensemble_type,
                weights=weights,
                save_name=save_name
            )

            # 格式化输出
            model_info = f"✅ 集成模型创建成功!\n\n"
            model_info += f"📊 模型名称: {result['model_name']}\n"
            model_info += f"📈 集成类型: {ensemble_type}\n"
            model_info += f"📑 基础模型: {', '.join(base_models)}\n"

            if weights:
                model_info += f"⚖️ 权重: {weights}\n"

            return json.dumps({
                "text": model_info,
                "result": {
                    "model_name": result['model_name'],
                    "ensemble_type": ensemble_type,
                    "base_models": base_models
                }
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 创建集成模型时发生错误: {str(e)}",
                "error": str(e)
            })

    def _auto_select_model(
        data_path: str,
        target_column: str,
        categorical_columns: List[str] = None,
        numerical_columns: List[str] = None
    ) -> str:
        """自动选择最佳模型并优化超参数"""
        try:
            result = actual_auto_select(
                data_path=data_path,
                target_column=target_column,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns
            )

            # 获取模型类型和参数
            model_type = result['model_type']
            params = result['params']
            cv_score = result['cv_score']

            # 格式化参数输出
            params_str = ", ".join([f"{k}={v}" for k, v in params.items()])

            # 格式化输出
            model_info = f"✅ 自动模型选择完成!\n\n"
            model_info += f"🏆 最佳模型: {model_type}\n"
            model_info += f"📊 模型名称: {result['model_name']}\n"
            model_info += f"⚙️ 最佳参数: {params_str}\n"
            model_info += f"📈 交叉验证分数: {cv_score:.4f}\n\n"

            # 添加所有模型的比较结果
            model_info += "📊 所有模型比较:\n"
            for idx, model_result in enumerate(result['all_models_results']):
                model_info += f"{idx+1}. {model_result['model_type']}: CV={model_result['cv_score']:.4f}, Test={model_result['test_score']:.4f}\n"

            # 准备可视化数据 - 模型比较图
            model_types = [m['model_type'] for m in result['all_models_results']]
            cv_scores = [m['cv_score'] for m in result['all_models_results']]

            vis_data = generate_visualization(
                'bar',
                model_types,
                cv_scores,
                title='模型CV分数比较'
            )

            # 创建比较表格
            table_data = {
                'columns': ['模型类型', 'CV分数', '测试分数', '参数'],
                'data': [
                    [
                        m['model_type'],
                        f"{m['cv_score']:.4f}",
                        f"{m['test_score']:.4f}",
                        str({k: v for k, v in m['best_params'].items()})
                    ]
                    for m in result['all_models_results']
                ]
            }

            return json.dumps({
                "text": model_info,
                "visualization_data": vis_data,
                "table_data": table_data,
                "result": result
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 自动选择模型时发生错误: {str(e)}",
                "error": str(e)
            })

    def _explain_prediction(
        model_name: str,
        input_data: Dict[str, Any]
    ) -> str:
        """解释模型预测结果"""
        try:
            explanation = actual_explain_prediction(model_name, input_data)

            # 获取预测结果
            prediction = explanation['prediction']
            prediction_str = str(prediction[0]) if len(prediction) == 1 else str(prediction)

            # 格式化输出
            explanation_text = f"✅ 模型预测解释完成!\n\n"
            explanation_text += f"📊 预测结果: {prediction_str}\n\n"

            # 添加特征重要性信息
            if explanation['feature_importance']:
                explanation_text += "📑 特征重要性 (前5项):\n"
                sorted_features = sorted(
                    explanation['feature_importance'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]

                for feature, importance in sorted_features:
                    explanation_text += f"  - {feature}: {importance:.4f}\n"

            # 获取可视化结果
            viz_results = visualize_model_explanation(explanation)

            return json.dumps({
                "text": explanation_text,
                "visualizations": viz_results['visualizations'],
                "tables": viz_results['tables'],
                "explanation": explanation
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 解释预测结果时发生错误: {str(e)}",
                "error": str(e)
            })

    def _compare_models(
        model_names: List[str],
        test_data_path: str,
        target_column: str
    ) -> str:
        """比较多个模型的性能"""
        try:
            if not model_names:
                return json.dumps({
                    "text": "❌ 模型比较失败：必须至少提供一个模型名称。",
                    "error": "模型名称列表不能为空。",
                    "comparison": None
                })
            # actual_compare_models is an alias for compare_models from ml_models
            comparison = actual_compare_models(model_names, test_data_path, target_column)

            # 格式化输出
            comparison_text = f"✅ 模型比较完成!\n\n"
            comparison_text += f"📊 测试数据: {test_data_path}\n"
            comparison_text += f"🎯 目标列: {target_column}\n\n"

            # 添加最佳模型信息
            if comparison.get('best_classifier'):
                comparison_text += f"🏆 最佳分类器: {comparison['best_classifier']}\n"

            if comparison.get('best_regressor'):
                comparison_text += f"🏆 最佳回归器: {comparison['best_regressor']}\n"

            # 获取可视化结果
            viz_results = visualize_model_comparison(comparison)

            return json.dumps({
                "text": comparison_text,
                "visualizations": viz_results['visualizations'],
                "tables": viz_results['tables'],
                "comparison": comparison
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 比较模型时发生错误: {str(e)}",
                "error": str(e)
            })

    def _version_model(
        model_name: str,
        version: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """创建模型的版本"""
        try:
            # 先加载模型
            loaded_data = load_model(model_name) # load_model is imported from ml_models
            if loaded_data is None or loaded_data[0] is None:
                error_msg = f"模型 '{model_name}' 未找到或加载失败。"
                if loaded_data is None:
                    error_msg = f"模型文件 '{model_name}' 未找到或无法加载。"
                elif loaded_data[0] is None:
                    error_msg = f"模型 '{model_name}' 加载成功，但模型对象为空。"
                return json.dumps({
                    "text": f"❌ 创建模型版本时发生错误: {error_msg}",
                    "error": error_msg,
                    "version_info": None
                })
            model, preprocessors, _ = loaded_data

            # 保存版本
            # save_model_with_version is imported from ml_models
            version_info = save_model_with_version(model, model_name, preprocessors, metadata, version)

            # 格式化输出
            version_text = f"✅ 模型版本保存成功!\n\n"
            version_text += f"📊 模型名称: {version_info['model_name']}\n"
            version_text += f"🔖 版本号: {version_info['version']}\n"
            version_text += f"⏱️ 时间戳: {version_info['timestamp']}\n"

            # 创建表格数据
            table_data = {
                'columns': ['字段', '值'],
                'data': [
                    [k, str(v)] for k, v in version_info.items()
                    if k not in ('path', 'timestamp')  # 排除一些不需要显示的字段
                ]
            }

            return json.dumps({
                "text": version_text,
                "table_data": table_data,
                "version_info": version_info
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 创建模型版本时发生错误: {str(e)}",
                "error": str(e)
            })

    def _list_model_versions(model_name: str) -> str:
        """列出模型的所有版本"""
        try:
            versions = list_model_versions(model_name)

            if not versions:
                return json.dumps({
                    "text": f"📊 模型 {model_name} 没有找到任何版本。"
                })

            # 格式化输出
            version_text = f"📊 模型 {model_name} 的版本列表:\n\n"

            for idx, version in enumerate(versions):
                v_num = version.get('version', 'unknown')
                v_time = version.get('timestamp', 'unknown')
                version_text += f"{idx+1}. 版本 {v_num} (时间: {v_time})\n"

            # 创建表格数据
            table_data = {
                'columns': ['版本号', '创建时间', '路径'],
                'data': [
                    [
                        v.get('version', '-'),
                        v.get('timestamp', '-'),
                        v.get('path', '-')
                    ]
                    for v in versions
                ]
            }

            return json.dumps({
                "text": version_text,
                "table_data": table_data,
                "versions": versions
            })
        except Exception as e:
            return json.dumps({
                "text": f"❌ 获取模型版本列表时发生错误: {str(e)}",
                "error": str(e)
            })

    # 创建工具集
    tools = [
        StructuredTool.from_function(
            func=_train_model,
            name="train_model",
            description="训练机器学习模型，支持多种类型的模型，如线性回归、逻辑回归、决策树、随机森林等",
            args_schema=TrainModelInput
        ),
        StructuredTool.from_function(
            func=_predict_with_model,
            name="predict_with_model",
            description="使用已训练的模型进行预测",
            args_schema=PredictInput
        ),
        Tool.from_function(
            func=_list_models,
            name="list_models",
            description="列出所有可用的机器学习模型"
        ),
        Tool.from_function(
            func=_recommend_model,
            name="recommend_model",
            description="根据任务描述推荐适合的机器学习模型",
            args_schema=RecommendModelInput
        ),
        Tool.from_function(
            func=_data_analysis,
            name="analyze_data",
            description="分析数据集的特征、统计信息，并能找出与指定目标变量最相关的特征。",
            args_schema=DataAnalysisInput
        ),
        Tool.from_function(
            func=_evaluate_model,
            name="evaluate_model",
            description="评估模型在测试数据上的表现",
            args_schema=EvaluateModelInput
        ),
        Tool.from_function(
            func=_create_ensemble_model,
            name="create_ensemble_model",
            description="创建集成模型，组合多个基础模型",
            args_schema=EnsembleModelInput
        ),
        Tool.from_function(
            func=_auto_select_model,
            name="auto_select_model",
            description="自动选择最佳模型并优化超参数",
            args_schema=AutoSelectModelInput
        ),
        Tool.from_function(
            func=_explain_prediction,
            name="explain_prediction",
            description="解释模型预测结果，提供特征重要性和贡献分析",
            args_schema=ExplainPredictionInput
        ),
        Tool.from_function(
            func=_compare_models,
            name="compare_models",
            description="比较多个模型在测试集上的性能",
            args_schema=CompareModelsInput
        ),
        Tool.from_function(
            func=_version_model,
            name="version_model",
            description="为模型创建一个新版本",
            args_schema=VersionModelInput
        ),
        Tool.from_function(
            func=_list_model_versions,
            name="list_model_versions",
            description="列出模型的所有版本",
            args_schema=ListModelVersionsInput
        )
    ]

    return tools

def create_ml_agent(use_existing_model: bool = True):
    """创建机器学习代理"""
    model = BaiduErnieLLM()
    tools = create_ml_tools()

    # 提取工具名称列表
    # 提示模板，确保包含所有必需变量
    # 注意：`tools`变量将由create_structured_chat_agent用工具的格式化描述填充
    prompt_template_base = """
{tools}
Tool Names: {tool_names}
Input: {input}
{agent_scratchpad}
"""
    model_preference_text = ""
    if use_existing_model:
        model_preference_text = "\n重要提示：当前设置为优先使用已训练好的模型。除非用户明确要求或没有合适的现有模型，否则请不要重新训练模型。\n"

    # model_preference_text is already defined above.
    # prompt_template_base is already defined above and contains the necessary placeholders:
    # """
    # {tools}
    # Tool Names: {tool_names}
    # Input: {input}
    # {agent_scratchpad}
    # """

    # Combine model preference text with the base template
    effective_template_str = model_preference_text + prompt_template_base

    prompt = PromptTemplate(
        template=effective_template_str,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
        # Using default validate_template=True is recommended
    )

    # 创建代理
    agent = create_structured_chat_agent(
        llm=model,
        tools=tools,
        prompt=prompt
    )

    # 创建代理执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=lambda e: f"JSON解析错误: {str(e)}\n原始响应:\n```json\n{e.response.replace('{', '{{').replace('}', '}}')}\n```\n",
        max_iterations=1,  # 减少最大迭代次数以避免超时
        return_intermediate_steps=True,
        max_execution_time=180  # 设置最大执行时间为30秒
    )

    return agent_executor

def query_ml_agent(question: str, use_existing_model: bool = True) -> Dict[str, Any]:
    """
    查询机器学习代理

    Args:
        question: 用户问题

    Returns:
        包含回答和可能的可视化数据的字典
    """
    agent_response_output = ""
    visualization_data = None
    table_data = None
    is_ml_query_flag = True # 默认为 True
    error_message = None
    error_details = None
    expected_format_info = None


    try:
        # 创建并查询代理
        agent = create_ml_agent(use_existing_model=use_existing_model)
        response = agent.invoke({"input": question})

        # 解析JSON响应
        agent_response_output = response.get("output", "") # 使用局部变量
        steps = response.get("intermediate_steps", [])

        # 提取可能包含的可视化数据
        # visualization_data = None # 已在函数开头初始化
        # table_data = None # 已在函数开头初始化

        # 内部 try-except 用于处理工具输出的解析
        try:
            if steps and len(steps) > 0:
                last_step = steps[-1]
                tool_output = last_step[1]

                if isinstance(tool_output, str) and tool_output.strip().startswith('{'):
                    try:
                        json_output = json.loads(tool_output)
                        if 'text' in json_output:
                            agent_response_output = json_output['text']
                        if 'visualization_data' in json_output:
                            visualization_data = json_output['visualization_data']
                        if 'table_data' in json_output:
                            table_data = json_output['table_data']
                        if 'visualizations' in json_output:
                            if json_output['visualizations']:
                                visualization_data = json_output['visualizations'][0]
                        if 'tables' in json_output:
                            if json_output['tables']:
                                table_data = json_output['tables'][0]['data']
                    except (json.JSONDecodeError, ValueError) as e_json:
                        # 这个 return 会提前结束函数，如果这是期望行为则保留
                        # 否则，应该设置错误信息并继续到函数末尾的 return
                        error_message = "模型响应格式错误"
                        error_details = f"请严格使用要求的JSON格式。错误信息: {str(e_json)}"
                        expected_format_info = {
                                "action": "tool_name",
                                "action_input": {"param1": "value"}
                            }
                        # 如果希望在这里就返回，那么：
                        return {
                            "answer": agent_response_output, # 或者一个固定的错误提示
                            "visualization_data": visualization_data,
                            "table_data": table_data,
                            "is_ml_query": is_ml_query_flag,
                            "error": error_message,
                            "details": error_details,
                            "expected_format": expected_format_info
                        }
                # else: # 不是JSON格式，保持原样
                #    pass # 这个 pass 是不必要的
        except Exception as e_inner_parse:
            # 提取可视化数据出错，可以选择记录日志或设置错误信息
            print(f"提取或解析工具输出时发生错误: {str(e_inner_parse)}")
            # 你可能想在这里也设置 error_message

    except TimeoutError as te:
        error_msg_detail = f"查询处理超时: {str(te)}\n{traceback.format_exc()}\n"
        print(error_msg_detail)
        agent_response_output = "查询超时，请尝试简化您的请求或使用更小的数据集"
        error_message = "请求超时"
    except Exception as e_outer:
        error_msg_detail = f"处理机器学习查询时发生错误: {str(e_outer)}\n{traceback.format_exc()}\n"
        print(error_msg_detail)
        agent_response_output = f"处理查询时发生错误: {str(e_outer)}"
        error_message = str(e_outer)

    # 统一的返回点
    result = {
        "answer": agent_response_output,
        "visualization_data": visualization_data,
        "table_data": table_data,
        "is_ml_query": is_ml_query_flag
    }
    if error_message:
        result["error"] = error_message
    if error_details:
        result["details"] = error_details
    if expected_format_info:
        result["expected_format"] = expected_format_info

    return result

def enhance_ml_query_with_rag(query, ml_response):
    """使用RAG系统增强机器学习查询结果"""
    from rag_core import query_rag

    # 获取原始ML回答
    ml_answer = ml_response.get("answer", "")
    return ml_answer  # 或者增强后的结果
