# advanced_feature_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple
import colorsys
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest
from sklearn.feature_selection import f_regression, f_classif, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

# 导入现有的特征分析功能
from feature_analysis import (
    analyze_feature_correlations, 
    calculate_mutual_information, 
    analyze_feature_importance,
    analyze_pairwise_relationships,
    comprehensive_feature_analysis,
    generate_gradient_colors
)


def analyze_feature_stability(df: pd.DataFrame, target_column: str, 
                            categorical_features: List[str] = None,
                            n_splits: int = 5, random_state: int = 42) -> Dict:
    """
    分析特征稳定性，通过交叉验证评估特征重要性的稳定性
    
    参数:
        df: 数据框
        target_column: 目标列名称
        categorical_features: 分类特征列表
        n_splits: 交叉验证折数
        random_state: 随机种子
        
    返回:
        包含特征稳定性分析结果的字典
    """
    if target_column not in df.columns:
        return {"error": f"目标列 {target_column} 不存在"}
    
    # 分离特征和目标
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 确定目标类型（分类或回归）
    is_classification = False
    if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or y.nunique() < 10:
        is_classification = True
    
    # 处理分类特征
    if categorical_features:
        for col in categorical_features:
            if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.Categorical(X[col]).codes
    
    # 确保所有特征都是数值型
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    X_numeric = X[numeric_cols]
    
    # 初始化交叉验证
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 存储每个折的特征重要性
    importances_per_fold = []
    
    # 对每个折训练模型并计算特征重要性
    for train_index, test_index in kf.split(X_numeric):
        X_train, X_test = X_numeric.iloc[train_index], X_numeric.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # 初始化模型
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 存储特征重要性
        importances_per_fold.append(model.feature_importances_)
    
    # 计算每个特征的重要性均值和标准差
    importances_array = np.array(importances_per_fold)
    mean_importances = np.mean(importances_array, axis=0)
    std_importances = np.std(importances_array, axis=0)
    
    # 计算变异系数（标准差/均值）作为稳定性指标
    # 注意：避免除以零
    stability_scores = np.zeros_like(mean_importances)
    for i, mean_imp in enumerate(mean_importances):
        if mean_imp > 0:
            stability_scores[i] = 1 - (std_importances[i] / mean_imp)  # 1减去变异系数，使得值越高表示越稳定
        else:
            stability_scores[i] = 0  # 如果均值为0，则稳定性为0
    
    # 创建结果数据框
    stability_df = pd.DataFrame({
        '特征': numeric_cols,
        '平均重要性': mean_importances,
        '标准差': std_importances,
        '稳定性分数': stability_scores
    })
    stability_df = stability_df.sort_values('平均重要性', ascending=False)
    
    # 生成条形图
    plt.figure(figsize=(12, max(8, len(stability_df) * 0.4)))
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, len(stability_df) * 0.4)))
    
    # 特征重要性图
    colors = generate_gradient_colors(len(stability_df))
    ax1.barh(stability_df['特征'], stability_df['平均重要性'], xerr=stability_df['标准差'], color=colors)
    ax1.set_title('特征重要性及其变异性')
    ax1.set_xlabel('重要性')
    
    # 稳定性分数图
    ax2.barh(stability_df['特征'], stability_df['稳定性分数'], color=colors)
    ax2.set_title('特征稳定性分数 (越高越稳定)')
    ax2.set_xlabel('稳定性分数')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    
    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return {
        "stability_analysis": stability_df.to_dict('records'),
        "feature_names": numeric_cols,
        "mean_importance": mean_importances.tolist(),
        "std_importance": std_importances.tolist(),
        "stability_scores": stability_scores.tolist(),
        "is_classification": is_classification,
        "image": base64.b64encode(image_png).decode('utf-8')
    }


def analyze_feature_interactions(df: pd.DataFrame, target_column: str, 
                               top_features: int = 5,
                               categorical_features: List[str] = None) -> Dict:
    """
    分析特征之间的交互作用对目标变量的影响
    
    参数:
        df: 数据框
        target_column: 目标列名称
        top_features: 要分析的顶级特征数量
        categorical_features: 分类特征列表
        
    返回:
        包含特征交互分析结果的字典
    """
    if target_column not in df.columns:
        return {"error": f"目标列 {target_column} 不存在"}
    
    # 首先获取最重要的特征
    importance_result = analyze_feature_importance(df, target_column, categorical_features)
    if "error" in importance_result:
        return importance_result
    
    # 获取最重要的特征名称
    top_feature_names = importance_result["feature_names"][:top_features]
    
    # 分离特征和目标
    X = df[top_feature_names].copy()
    y = df[target_column]
    
    # 处理分类特征
    if categorical_features:
        for col in categorical_features:
            if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.Categorical(X[col]).codes
    
    # 确保所有特征都是数值型
    X = X.select_dtypes(include=['number'])
    
    # 创建特征交互项
    interaction_results = []
    
    for i in range(len(top_feature_names)):
        for j in range(i+1, len(top_feature_names)):
            feature1 = top_feature_names[i]
            feature2 = top_feature_names[j]
            
            # 创建交互特征
            interaction_name = f"{feature1} * {feature2}"
            X[interaction_name] = X[feature1] * X[feature2]
            
            # 评估交互特征的影响
            if pd.api.types.is_numeric_dtype(y):
                # 回归问题：计算相关性
                corr, p_value = pearsonr(X[interaction_name], y)
                base_corr1, _ = pearsonr(X[feature1], y)
                base_corr2, _ = pearsonr(X[feature2], y)
                
                # 计算交互增益（交互特征相关性与单独特征相关性的最大值之差）
                interaction_gain = abs(corr) - max(abs(base_corr1), abs(base_corr2))
                
                interaction_results.append({
                    "feature1": feature1,
                    "feature2": feature2,
                    "interaction": interaction_name,
                    "correlation": corr,
                    "p_value": p_value,
                    "base_corr1": base_corr1,
                    "base_corr2": base_corr2,
                    "interaction_gain": interaction_gain
                })
            else:
                # 分类问题：计算互信息
                mi = mutual_info_classif(X[[interaction_name]], y, random_state=42)[0]
                base_mi1 = mutual_info_classif(X[[feature1]], y, random_state=42)[0]
                base_mi2 = mutual_info_classif(X[[feature2]], y, random_state=42)[0]
                
                # 计算交互增益
                interaction_gain = mi - max(base_mi1, base_mi2)
                
                interaction_results.append({
                    "feature1": feature1,
                    "feature2": feature2,
                    "interaction": interaction_name,
                    "mutual_info": mi,
                    "base_mi1": base_mi1,
                    "base_mi2": base_mi2,
                    "interaction_gain": interaction_gain
                })
    
    # 按交互增益排序
    interaction_results = sorted(interaction_results, key=lambda x: x["interaction_gain"], reverse=True)
    
    # 可视化交互增益
    plt.figure(figsize=(12, 8))
    
    # 提取数据
    interactions = [result["interaction"] for result in interaction_results]
    gains = [result["interaction_gain"] for result in interaction_results]
    
    # 绘制条形图
    colors = generate_gradient_colors(len(interaction_results))
    plt.barh(interactions, gains, color=colors)
    plt.title('特征交互增益')
    plt.xlabel('增益值')
    plt.tight_layout()
    
    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # 创建热图显示交互矩阵
    interaction_matrix = np.zeros((len(top_feature_names), len(top_feature_names)))
    for result in interaction_results:
        i = top_feature_names.index(result["feature1"])
        j = top_feature_names.index(result["feature2"])
        interaction_matrix[i, j] = result["interaction_gain"]
        interaction_matrix[j, i] = result["interaction_gain"]  # 对称矩阵
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, annot=True, fmt='.3f', cmap='coolwarm',
               xticklabels=top_feature_names, yticklabels=top_feature_names)
    plt.title('特征交互矩阵')
    plt.tight_layout()
    
    # 将热图转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    heatmap_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return {
        "interaction_results": interaction_results,
        "top_features": top_feature_names,
        "is_classification": not pd.api.types.is_numeric_dtype(y),
        "bar_image": base64.b64encode(image_png).decode('utf-8'),
        "heatmap_image": base64.b64encode(heatmap_png).decode('utf-8')
    }


def analyze_feature_nonlinearity(df: pd.DataFrame, target_column: str, 
                               categorical_features: List[str] = None,
                               top_features: int = 5) -> Dict:
    """
    分析特征与目标变量之间的非线性关系
    
    参数:
        df: 数据框
        target_column: 目标列名称
        categorical_features: 分类特征列表
        top_features: 要分析的顶级特征数量
        
    返回:
        包含非线性关系分析结果的字典
    """
    if target_column not in df.columns:
        return {"error": f"目标列 {target_column} 不存在"}
    
    # 只对回归问题进行非线性分析
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return {"error": "非线性分析仅适用于回归问题（数值型目标变量）"}
    
    # 首先获取最重要的特征
    importance_result = analyze_feature_importance(df, target_column, categorical_features)
    if "error" in importance_result:
        return importance_result
    
    # 获取最重要的特征名称
    top_feature_names = importance_result["feature_names"][:top_features]
    
    # 分离特征和目标
    X = df[top_feature_names].copy()
    y = df[target_column]
    
    # 处理分类特征
    if categorical_features:
        for col in categorical_features:
            if col in X.columns and not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.Categorical(X[col]).codes
    
    # 确保所有特征都是数值型
    X = X.select_dtypes(include=['number'])
    
    # 分析每个特征的非线性关系
    nonlinearity_results = []
    
    # 创建多个子图
    fig, axes = plt.subplots(len(top_feature_names), 3, figsize=(18, 5 * len(top_feature_names)))
    
    for i, feature in enumerate(top_feature_names):
        # 线性模型
        X_feature = X[[feature]]
        linear_model = RandomForestRegressor(n_estimators=1, max_depth=1)  # 使用深度为1的决策树作为线性近似
        linear_model.fit(X_feature, y)
        y_pred_linear = linear_model.predict(X_feature)
        linear_r2 = r2_score(y, y_pred_linear)
        linear_mse = mean_squared_error(y, y_pred_linear)
        
        # 非线性模型（随机森林）
        nonlinear_model = RandomForestRegressor(n_estimators=100, max_depth=None)
        nonlinear_model.fit(X_feature, y)
        y_pred_nonlinear = nonlinear_model.predict(X_feature)
        nonlinear_r2 = r2_score(y, y_pred_nonlinear)
        nonlinear_mse = mean_squared_error(y, y_pred_nonlinear)
        
        # 计算非线性增益
        nonlinearity_gain = nonlinear_r2 - linear_r2
        
        # 存储结果
        nonlinearity_results.append({
            "feature": feature,
            "linear_r2": linear_r2,
            "nonlinear_r2": nonlinear_r2,
            "linear_mse": linear_mse,
            "nonlinear_mse": nonlinear_mse,
            "nonlinearity_gain": nonlinearity_gain
        })
        
        # 绘制散点图和拟合曲线
        # 原始散点图
        axes[i, 0].scatter(X_feature, y, alpha=0.5, color='#4F46E5')
        axes[i, 0].set_title(f'{feature} vs {target_column} (原始数据)')
        axes[i, 0].set_xlabel(feature)
        axes[i, 0].set_ylabel(target_column)
        
        # 线性拟合
        # 按特征值排序以便绘制平滑曲线
        sort_idx = np.argsort(X_feature.values.ravel())
        axes[i, 1].scatter(X_feature.values.ravel(), y, alpha=0.3, color='#4F46E5')
        axes[i, 1].plot(X_feature.values.ravel()[sort_idx], y_pred_linear[sort_idx], color='red', linewidth=2)
        axes[i, 1].set_title(f'线性拟合 (R² = {linear_r2:.3f})')
        axes[i, 1].set_xlabel(feature)
        axes[i, 1].set_ylabel(target_column)
        
        # 非线性拟合
        axes[i, 2].scatter(X_feature.values.ravel(), y, alpha=0.3, color='#4F46E5')
        axes[i, 2].plot(X_feature.values.ravel()[sort_idx], y_pred_nonlinear[sort_idx], color='green', linewidth=2)
        axes[i, 2].set_title(f'非线性拟合 (R² = {nonlinear_r2:.3f})')
        axes[i, 2].set_xlabel(feature)
        axes[i, 2].set_ylabel(target_column)
    
    plt.tight_layout()
    
    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # 绘制非线性增益条形图
    plt.figure(figsize=(10, 6))
    features = [result["feature"] for result in nonlinearity_results]
    gains = [result["nonlinearity_gain"] for result in nonlinearity_results]
    
    # 按非线性增益排序
    sorted_indices = np.argsort(gains)
    sorted_features = [features[i] for i in sorted_indices]
    sorted_gains = [gains[i] for i in sorted_indices]
    
    colors = generate_gradient_colors(len(features))
    plt.barh(sorted_features, sorted_gains, color=colors)
    plt.title('特征非线性增益 (非线性R² - 线性R²)')
    plt.xlabel('非线性增益')
    plt.tight_layout()
    
    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    gain_image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return {
        "nonlinearity_results": nonlinearity_results,
        "features": features,
        "nonlinearity_gains": gains,
        "fit_curves_image": base64.b64encode(image_png).decode('utf-8'),
        "gain_image": base64.b64encode(gain_image_png).decode('utf-8')
    }


def advanced_feature_analysis(df: pd.DataFrame, target_column: str, 
                            categorical_features: List[str] = None,
                            analysis_types: List[str] = None) -> Dict:
    """
    执行高级特征分析，包括特征稳定性、交互作用和非线性关系分析
    
    参数:
        df: 数据框
        target_column: 目标列名称
        categorical_features: 分类特征列表
        analysis_types: 要执行的分析类型列表，可选 'stability', 'interaction', 'nonlinearity', 'all'
        
    返回:
        包含高级分析结果的字典
    """
    if analysis_types is None or 'all' in analysis_types:
        analysis_types = ['stability', 'interaction', 'nonlinearity']
    
    results = {}
    
    # 基础特征分析
    basic_analysis = comprehensive_feature_analysis(df, target_column, categorical_features)
    results["basic_analysis"] = basic_analysis
    
    # 特征稳定性分析
    if 'stability' in analysis_types:
        try:
            stability_results = analyze_feature_stability(df, target_column, categorical_features)
            results["stability_analysis"] = stability_results
        except Exception as e:
            results["stability_analysis"] = {"error": str(e)}
    
    # 特征交互分析
    if 'interaction' in analysis_types:
        try:
            interaction_results = analyze_feature_interactions(df, target_column, 5, categorical_features)
            results["interaction_analysis"] = interaction_results
        except Exception as e:
            results["interaction_analysis"] = {"error": str(e)}
    
    # 非线性关系分析（仅适用于回归问题）
    if 'nonlinearity' in analysis_types and pd.api.types.is_numeric_dtype(df[target_column]):
        try:
            nonlinearity_results = analyze_feature_nonlinearity(df, target_column, categorical_features)
            results["nonlinearity_analysis"] = nonlinearity_results
        except Exception as e:
            results["nonlinearity_analysis"] = {"error": str(e)}
    
    return results


def integrate_ml_with_rag(query_result: Dict, ml_model_name: str, feature_data: Dict) -> Dict:
    """
    将机器学习模型的预测结果与RAG模型的回答集成
    
    参数:
        query_result: RAG查询结果
        ml_model_name: 机器学习模型名称
        feature_data: 特征数据和分析结果
        
    返回:
        增强的查询结果
    """
    # 获取原始回答
    original_answer = query_result.get("answer", "")
    
    # 构建增强回答
    enhanced_answer = original_answer
    
    # 添加模型预测信息
    if "prediction" in feature_data:
        prediction_info = f"\n\n根据'{ml_model_name}'模型的预测结果: {feature_data['prediction']}"
        enhanced_answer += prediction_info
    
    # 添加特征重要性信息
    if "feature_importance" in feature_data:
        # 兼容两种可能的键名：feature_names或top_features
        feature_key = "top_features" if "top_features" in feature_data["feature_importance"] else "feature_names"
        top_features = feature_data["feature_importance"][feature_key][:3]  # 取前3个重要特征
        importance_values = feature_data["feature_importance"]["importance_values"][:3]
        
        importance_info = "\n\n影响这一结果的主要特征是: "
        for i, (feature, importance) in enumerate(zip(top_features, importance_values)):
            importance_info += f"\n{i+1}. {feature} (重要性: {importance:.4f})"
        
        enhanced_answer += importance_info
    
    # 添加模型性能信息
    if "model_metrics" in feature_data:
        metrics = feature_data["model_metrics"]
        metrics_info = "\n\n模型性能指标: "
        for metric_name, metric_value in metrics.items():
            metrics_info += f"\n- {metric_name}: {metric_value:.4f}"
        
        enhanced_answer += metrics_info
    
    # 更新查询结果
    query_result["answer"] = enhanced_answer
    query_result["ml_enhanced"] = True
    query_result["ml_model_used"] = ml_model_name
    query_result["feature_analysis"] = feature_data
    
    return query_result