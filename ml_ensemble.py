#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
启航者 AI - 机器学习模型集成系统
整合多种机器学习算法，提供集成学习功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, classification_report,
    silhouette_score, hamming_loss
)

# 模型导入
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor,
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier

# 文本处理
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MLEnsembleSystem:
    """机器学习模型集成系统"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.data_cache = {}
        self.scalers = {}
        
        # 初始化可用模型
        self._init_available_models()
    
    def _init_available_models(self):
        """初始化可用模型字典"""
        self.available_models = {
            # 回归模型
            'linear_regression': {
                'class': LinearRegression,
                'type': 'regression',
                'name': '线性回归',
                'description': '用于连续变量预测的基本线性模型'
            },
            'decision_tree_regressor': {
                'class': DecisionTreeRegressor,
                'type': 'regression', 
                'name': '决策树回归',
                'description': '基于树结构的回归模型'
            },
            'random_forest_regressor': {
                'class': RandomForestRegressor,
                'type': 'regression',
                'name': '随机森林回归',
                'description': '集成多棵决策树的回归模型'
            },
            'svr': {
                'class': SVR,
                'type': 'regression',
                'name': '支持向量回归',
                'description': '支持向量机回归模型'
            },
            'knn_regressor': {
                'class': KNeighborsRegressor,
                'type': 'regression',
                'name': 'K近邻回归',
                'description': '基于最近邻的回归模型'
            },
            
            # 分类模型
            'logistic_regression': {
                'class': LogisticRegression,
                'type': 'classification',
                'name': '逻辑回归',
                'description': '用于分类问题的概率模型'
            },
            'decision_tree': {
                'class': DecisionTreeClassifier,
                'type': 'classification',
                'name': '决策树',
                'description': '基于树结构的分类模型'
            },
            'random_forest': {
                'class': RandomForestClassifier,
                'type': 'classification',
                'name': '随机森林',
                'description': '集成多棵决策树的分类模型'
            },
            'svm': {
                'class': SVC,
                'type': 'classification',
                'name': '支持向量机',
                'description': '支持向量机分类模型'
            },
            'knn': {
                'class': KNeighborsClassifier,
                'type': 'classification',
                'name': 'K近邻',
                'description': '基于最近邻的分类模型'
            },
            'naive_bayes': {
                'class': GaussianNB,
                'type': 'classification',
                'name': '朴素贝叶斯',
                'description': '基于贝叶斯定理的分类模型'
            },
            'multinomial_nb': {
                'class': MultinomialNB,
                'type': 'classification',
                'name': '多项式朴素贝叶斯',
                'description': '适用于文本分类的朴素贝叶斯模型'
            },
            
            # 聚类模型
            'kmeans': {
                'class': KMeans,
                'type': 'clustering',
                'name': 'K均值聚类',
                'description': '基于距离的聚类算法'
            }
        }
    
    def load_data(self, data_path: str, data_type: str = 'auto') -> Dict[str, Any]:
        """
        加载数据集
        
        Args:
            data_path: 数据文件路径
            data_type: 数据类型 ('csv', 'excel', 'json', 'auto')
        
        Returns:
            数据加载结果
        """
        try:
            if data_type == 'auto':
                # 自动检测文件类型
                if data_path.endswith('.csv'):
                    data_type = 'csv'
                elif data_path.endswith(('.xlsx', '.xls')):
                    data_type = 'excel'
                elif data_path.endswith('.json'):
                    data_type = 'json'
                else:
                    raise ValueError(f"无法识别的文件类型: {data_path}")
            
            # 加载数据
            if data_type == 'csv':
                df = pd.read_csv(data_path, encoding='utf-8')
            elif data_type == 'excel':
                df = pd.read_excel(data_path)
            elif data_type == 'json':
                # 处理JSON文件（可能是文本分类数据）
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                else:
                    raise ValueError("JSON文件格式不正确")
            
            # 缓存数据
            self.data_cache[data_path] = df
            
            # 分析数据
            analysis = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'sample_data': df.head().to_dict('records')
            }
            
            return {
                'success': True,
                'data': df,
                'analysis': analysis,
                'message': f"成功加载数据: {df.shape[0]}行, {df.shape[1]}列"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"数据加载失败: {str(e)}"
            }
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str, 
                       problem_type: str = 'auto') -> Dict[str, Any]:
        """
        数据预处理
        
        Args:
            df: 数据框
            target_column: 目标列名
            problem_type: 问题类型 ('classification', 'regression', 'clustering', 'auto')
        
        Returns:
            预处理结果
        """
        try:
            # 复制数据避免修改原始数据
            processed_df = df.copy()
            
            # 处理缺失值
            # 数值列用均值填充
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            processed_df[numeric_cols] = processed_df[numeric_cols].fillna(
                processed_df[numeric_cols].mean()
            )
            
            # 分类列用众数填充
            categorical_cols = processed_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != target_column:
                    mode_value = processed_df[col].mode()
                    if len(mode_value) > 0:
                        processed_df[col] = processed_df[col].fillna(mode_value[0])
            
            # 准备特征和目标变量
            if target_column in processed_df.columns:
                X = processed_df.drop(columns=[target_column])
                y = processed_df[target_column]
                
                # 自动检测问题类型
                if problem_type == 'auto':
                    if pd.api.types.is_numeric_dtype(y):
                        unique_values = y.nunique()
                        if unique_values <= 10:
                            problem_type = 'classification'
                        else:
                            problem_type = 'regression'
                    else:
                        problem_type = 'classification'
            else:
                # 聚类问题，没有目标变量
                X = processed_df
                y = None
                problem_type = 'clustering'
            
            # 编码分类特征
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
            
            # 处理目标变量
            target_encoder = None
            if y is not None and not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
            
            # 特征缩放
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # 缓存预处理器
            preprocessing_info = {
                'scaler': scaler,
                'label_encoders': label_encoders,
                'target_encoder': target_encoder,
                'feature_columns': X.columns.tolist(),
                'problem_type': problem_type
            }
            
            return {
                'success': True,
                'X': X_scaled,
                'y': y,
                'X_original': X,
                'preprocessing_info': preprocessing_info,
                'problem_type': problem_type,
                'message': f"数据预处理完成，问题类型: {problem_type}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"数据预处理失败: {str(e)}"
            }
    
    def train_single_model(self, model_name: str, X: pd.DataFrame, y: np.ndarray,
                          test_size: float = 0.2, **kwargs) -> Dict[str, Any]:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            X: 特征数据
            y: 目标变量
            test_size: 测试集比例
            **kwargs: 模型参数
        
        Returns:
            训练结果
        """
        try:
            if model_name not in self.available_models:
                raise ValueError(f"不支持的模型: {model_name}")
            
            model_info = self.available_models[model_name]
            model_class = model_info['class']
            
            # 创建模型实例
            if kwargs:
                model = model_class(**kwargs)
            else:
                # 使用默认参数
                if model_name == 'svm':
                    model = model_class(probability=True, random_state=42)
                elif model_name == 'random_forest':
                    model = model_class(n_estimators=100, random_state=42)
                elif model_name == 'kmeans':
                    model = model_class(n_clusters=3, random_state=42)
                else:
                    try:
                        model = model_class(random_state=42)
                    except:
                        model = model_class()
            
            # 训练模型
            start_time = datetime.now()
            
            if model_info['type'] == 'clustering':
                # 聚类不需要分割数据
                model.fit(X)
                predictions = model.predict(X)
                
                # 聚类评估
                if len(set(predictions)) > 1:
                    silhouette = silhouette_score(X, predictions)
                else:
                    silhouette = -1
                
                results = {
                    'model': model,
                    'model_name': model_name,
                    'model_type': model_info['type'],
                    'predictions': predictions,
                    'silhouette_score': silhouette,
                    'n_clusters': len(set(predictions)),
                    'training_time': (datetime.now() - start_time).total_seconds()
                }
            else:
                # 分类/回归需要分割数据
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 计算评估指标
                if model_info['type'] == 'classification':
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    results = {
                        'model': model,
                        'model_name': model_name,
                        'model_type': model_info['type'],
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'confusion_matrix': confusion_matrix(y_test, y_pred),
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'training_time': (datetime.now() - start_time).total_seconds()
                    }
                else:  # regression
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    results = {
                        'model': model,
                        'model_name': model_name,
                        'model_type': model_info['type'],
                        'mse': mse,
                        'rmse': np.sqrt(mse),
                        'r2_score': r2,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'training_time': (datetime.now() - start_time).total_seconds()
                    }
            
            # 缓存结果
            self.results[model_name] = results
            
            return {
                'success': True,
                'results': results,
                'message': f"模型 {model_name} 训练完成"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"模型 {model_name} 训练失败: {str(e)}"
            }
    
    def train_ensemble_models(self, base_models: List[str], X: pd.DataFrame, y: np.ndarray,
                            ensemble_method: str = 'voting', test_size: float = 0.2) -> Dict[str, Any]:
        """
        训练集成模型
        
        Args:
            base_models: 基础模型列表
            X: 特征数据
            y: 目标变量
            ensemble_method: 集成方法 ('voting', 'stacking', 'bagging')
            test_size: 测试集比例
        
        Returns:
            集成模型训练结果
        """
        try:
            # 检查问题类型
            if pd.api.types.is_numeric_dtype(y):
                unique_values = pd.Series(y).nunique()
                if unique_values <= 10:
                    problem_type = 'classification'
                else:
                    problem_type = 'regression'
            else:
                problem_type = 'classification'
            
            # 准备基础模型
            estimators = []
            for model_name in base_models:
                if model_name not in self.available_models:
                    continue
                
                model_info = self.available_models[model_name]
                if model_info['type'] != problem_type:
                    continue
                
                # 创建模型实例
                if model_name == 'svm' and problem_type == 'classification':
                    model = SVC(probability=True, random_state=42)
                elif model_name in ['random_forest', 'random_forest_regressor']:
                    if problem_type == 'classification':
                        model = RandomForestClassifier(n_estimators=50, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                else:
                    model_class = model_info['class']
                    try:
                        model = model_class(random_state=42)
                    except:
                        model = model_class()
                
                estimators.append((model_name, model))
            
            if len(estimators) < 2:
                raise ValueError("至少需要2个有效的基础模型进行集成")
            
            # 创建集成模型
            if ensemble_method == 'voting':
                if problem_type == 'classification':
                    ensemble_model = VotingClassifier(
                        estimators=estimators,
                        voting='soft'
                    )
                else:
                    ensemble_model = VotingRegressor(estimators=estimators)
            
            elif ensemble_method == 'stacking':
                if problem_type == 'classification':
                    ensemble_model = StackingClassifier(
                        estimators=estimators,
                        final_estimator=LogisticRegression(random_state=42),
                        cv=3
                    )
                else:
                    ensemble_model = StackingRegressor(
                        estimators=estimators,
                        final_estimator=LinearRegression(),
                        cv=3
                    )
            
            elif ensemble_method == 'bagging':
                # 使用第一个模型作为基础估计器
                base_estimator = estimators[0][1]
                if problem_type == 'classification':
                    ensemble_model = BaggingClassifier(
                        base_estimator=base_estimator,
                        n_estimators=10,
                        random_state=42
                    )
                else:
                    ensemble_model = BaggingRegressor(
                        base_estimator=base_estimator,
                        n_estimators=10,
                        random_state=42
                    )
            
            # 训练集成模型
            start_time = datetime.now()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            ensemble_model.fit(X_train, y_train)
            y_pred = ensemble_model.predict(X_test)
            
            # 计算评估指标
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                ensemble_results = {
                    'model': ensemble_model,
                    'model_name': f'{ensemble_method}_ensemble',
                    'model_type': problem_type,
                    'ensemble_method': ensemble_method,
                    'base_models': base_models,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'training_time': (datetime.now() - start_time).total_seconds()
                }
            else:  # regression
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                ensemble_results = {
                    'model': ensemble_model,
                    'model_name': f'{ensemble_method}_ensemble',
                    'model_type': problem_type,
                    'ensemble_method': ensemble_method,
                    'base_models': base_models,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2_score': r2,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'training_time': (datetime.now() - start_time).total_seconds()
                }
            
            # 训练基础模型进行比较
            base_results = []
            for model_name in base_models:
                result = self.train_single_model(model_name, X, y, test_size)
                if result['success']:
                    base_results.append(result['results'])
            
            return {
                'success': True,
                'ensemble_results': ensemble_results,
                'base_results': base_results,
                'message': f"{ensemble_method}集成模型训练完成"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"集成模型训练失败: {str(e)}"
            }
    
    def compare_models(self, results_list: List[Dict]) -> Dict[str, Any]:
        """
        比较模型性能
        
        Args:
            results_list: 模型结果列表
        
        Returns:
            比较结果
        """
        try:
            if not results_list:
                return {'success': False, 'message': '没有模型结果可比较'}
            
            comparison_data = []
            problem_type = results_list[0].get('model_type', 'unknown')
            
            for result in results_list:
                model_data = {
                    'model_name': result.get('model_name', 'unknown'),
                    'model_type': result.get('model_type', 'unknown'),
                    'training_time': result.get('training_time', 0)
                }
                
                if problem_type == 'classification':
                    model_data.update({
                        'accuracy': result.get('accuracy', 0),
                        'precision': result.get('precision', 0),
                        'recall': result.get('recall', 0),
                        'f1_score': result.get('f1_score', 0)
                    })
                elif problem_type == 'regression':
                    model_data.update({
                        'mse': result.get('mse', float('inf')),
                        'rmse': result.get('rmse', float('inf')),
                        'r2_score': result.get('r2_score', 0)
                    })
                elif problem_type == 'clustering':
                    model_data.update({
                        'silhouette_score': result.get('silhouette_score', -1),
                        'n_clusters': result.get('n_clusters', 0)
                    })
                
                comparison_data.append(model_data)
            
            # 找到最佳模型
            if problem_type == 'classification':
                best_model = max(comparison_data, key=lambda x: x.get('accuracy', 0))
                metric_name = 'accuracy'
            elif problem_type == 'regression':
                best_model = min(comparison_data, key=lambda x: x.get('mse', float('inf')))
                metric_name = 'mse'
            elif problem_type == 'clustering':
                best_model = max(comparison_data, key=lambda x: x.get('silhouette_score', -1))
                metric_name = 'silhouette_score'
            else:
                best_model = comparison_data[0]
                metric_name = 'unknown'
            
            return {
                'success': True,
                'comparison_data': comparison_data,
                'best_model': best_model,
                'problem_type': problem_type,
                'metric_name': metric_name,
                'message': f"模型比较完成，最佳模型: {best_model['model_name']}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f"模型比较失败: {str(e)}"
            }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """获取可用模型列表"""
        return [
            {
                'name': name,
                'type': info['type'],
                'display_name': info['name'],
                'description': info['description']
            }
            for name, info in self.available_models.items()
        ]
    
    def save_model(self, model_name: str, model, file_path: str) -> bool:
        """保存模型到文件"""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            return True
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False
    
    def load_model(self, file_path: str):
        """从文件加载模型"""
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"加载模型失败: {e}")
            return None

# 全局实例
ml_ensemble = MLEnsembleSystem()

def analyze_dataset_with_llm(data_path: str, llm_func) -> str:
    """
    使用大模型分析数据集并推荐合适的机器学习方法
    
    Args:
        data_path: 数据文件路径
        llm_func: 大模型函数
    
    Returns:
        大模型的分析建议
    """
    try:
        # 加载数据
        load_result = ml_ensemble.load_data(data_path)
        if not load_result['success']:
            return f"数据加载失败: {load_result['message']}"
        
        analysis = load_result['analysis']
        
        # 构建分析提示
        prompt = f"""
作为机器学习专家，请分析以下数据集并提供建议：

数据集信息：
- 形状: {analysis['shape'][0]}行 × {analysis['shape'][1]}列
- 数值列: {analysis['numeric_columns']}
- 分类列: {analysis['categorical_columns']}
- 缺失值: {analysis['missing_values']}

样本数据:
{analysis['sample_data'][:3]}

请提供：
1. 数据集类型分析
2. 推荐的机器学习问题类型（分类/回归/聚类）
3. 建议的特征工程方法
4. 推荐的算法（包括集成方法）
5. 预期的性能指标

请以专业但易懂的方式回答。
"""
        
        # 调用大模型
        return llm_func(prompt)
        
    except Exception as e:
        return f"分析过程中出现错误: {str(e)}"

def run_ml_experiment_with_dialogue(query: str, data_path: str = None, 
                                  llm_func = None) -> Dict[str, Any]:
    """
    通过对话方式运行机器学习实验
    
    Args:
        query: 用户查询
        data_path: 数据文件路径  
        llm_func: 大模型函数
    
    Returns:
        实验结果
    """
    try:
        # 解析用户意图
        query_lower = query.lower()
        
        if not data_path:
            return {
                'success': False,
                'message': '请先上传数据文件',
                'suggestion': '您可以上传CSV、Excel或JSON格式的数据文件'
            }
        
        # 分析数据集
        if '分析数据' in query or 'analyze' in query_lower:
            if llm_func:
                analysis = analyze_dataset_with_llm(data_path, llm_func)
                return {
                    'success': True,
                    'type': 'analysis',
                    'result': analysis,
                    'message': '数据集分析完成'
                }
            else:
                load_result = ml_ensemble.load_data(data_path)
                return {
                    'success': load_result['success'],
                    'type': 'analysis', 
                    'result': load_result.get('analysis', {}),
                    'message': load_result['message']
                }
        
        # 模型训练
        elif any(keyword in query for keyword in ['训练模型', '建模', '预测', 'train', 'model']):
            # 加载和预处理数据
            load_result = ml_ensemble.load_data(data_path)
            if not load_result['success']:
                return load_result
            
            df = load_result['data']
            
            # 尝试自动识别目标列
            target_column = None
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_column = numeric_cols[-1]  # 使用最后一个数值列
            
            if not target_column:
                return {
                    'success': False,
                    'message': '无法自动识别目标列，请指定目标变量',
                    'suggestion': f'可选择的列: {df.columns.tolist()}'
                }
            
            # 数据预处理
            preprocess_result = ml_ensemble.preprocess_data(df, target_column)
            if not preprocess_result['success']:
                return preprocess_result
            
            X = preprocess_result['X']
            y = preprocess_result['y']
            problem_type = preprocess_result['problem_type']
            
            # 根据问题类型选择模型
            if problem_type == 'classification':
                models_to_try = ['logistic_regression', 'decision_tree', 'random_forest', 'svm']
            elif problem_type == 'regression':
                models_to_try = ['linear_regression', 'decision_tree_regressor', 'random_forest_regressor']
            else:
                models_to_try = ['kmeans']
            
            # 检查是否要求集成学习
            if any(keyword in query for keyword in ['集成', '投票', 'ensemble', 'voting']):
                ensemble_result = ml_ensemble.train_ensemble_models(
                    models_to_try[:3], X, y, 'voting'
                )
                return {
                    'success': ensemble_result['success'],
                    'type': 'ensemble_training',
                    'result': ensemble_result,
                    'message': ensemble_result['message']
                }
            else:
                # 训练单个模型
                results = []
                for model_name in models_to_try[:3]:  # 限制模型数量
                    result = ml_ensemble.train_single_model(model_name, X, y)
                    if result['success']:
                        results.append(result['results'])
                
                # 比较模型
                comparison = ml_ensemble.compare_models(results)
                
                return {
                    'success': True,
                    'type': 'model_training',
                    'result': {
                        'individual_results': results,
                        'comparison': comparison
                    },
                    'message': f'成功训练了{len(results)}个模型'
                }
        
        # 模型比较
        elif any(keyword in query for keyword in ['比较', '对比', 'compare']):
            if not ml_ensemble.results:
                return {
                    'success': False,
                    'message': '没有已训练的模型可供比较',
                    'suggestion': '请先训练一些模型'
                }
            
            results = list(ml_ensemble.results.values())
            comparison = ml_ensemble.compare_models(results)
            
            return {
                'success': comparison['success'],
                'type': 'model_comparison',
                'result': comparison,
                'message': comparison['message']
            }
        
        else:
            # 通用帮助信息
            return {
                'success': True,
                'type': 'help',
                'result': {
                    'available_commands': [
                        '分析数据 - 分析上传的数据集',
                        '训练模型 - 训练机器学习模型',
                        '集成学习 - 使用集成方法训练模型',
                        '比较模型 - 比较不同模型的性能',
                    ],
                    'available_models': ml_ensemble.get_available_models()
                },
                'message': '我可以帮您进行机器学习实验，请告诉我您想做什么'
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f'实验执行失败: {str(e)}'
        } 