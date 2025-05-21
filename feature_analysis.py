# feature_analysis.py
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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr, kendalltau


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


def analyze_feature_correlations(df: pd.DataFrame, target_column: Optional[str] = None, method: str = 'pearson') -> Dict:
    """
    分析特征之间的相关性，并可选择性地分析与目标变量的相关性
    
    参数:
        df: 数据框
        target_column: 目标列名称（可选）
        method: 相关性计算方法，可选 'pearson', 'spearman', 'kendall'
        
    返回:
        包含相关性分析结果的字典
    """
    # 确保所有列都是数值型
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return {
            "error": "没有找到数值型列，无法计算相关性"
        }
    
    # 计算相关性矩阵
    corr_matrix = numeric_df.corr(method=method)
    
    # 生成热图
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt='.2f',
               linewidths=.5, cbar_kws={'shrink': .8})
    plt.title(f'特征相关性矩阵 ({method}方法)')
    plt.tight_layout()
    
    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # 提取强相关特征对
    strong_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:  # 强相关阈值
                strong_corr_pairs.append({
                    "feature1": corr_matrix.columns[i],
                    "feature2": corr_matrix.columns[j],
                    "correlation": round(corr_matrix.iloc[i, j], 4)
                })
    
    # 按相关性绝对值排序
    strong_corr_pairs = sorted(strong_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)
    
    # 目标变量相关性分析
    target_correlations = None
    target_corr_image = None
    if target_column and target_column in numeric_df.columns:
        target_corr = corr_matrix[target_column].drop(target_column).sort_values(ascending=False)
        
        # 生成条形图
        plt.figure(figsize=(10, max(6, len(target_corr) * 0.3)))
        colors = generate_gradient_colors(len(target_corr))
        target_corr.plot(kind='barh', color=colors)
        plt.title(f'特征与目标变量 "{target_column}" 的相关性')
        plt.xlabel('相关系数')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        target_corr_image = buffer.getvalue()
        buffer.close()
        plt.close()
        
        target_correlations = {
            "feature_names": target_corr.index.tolist(),
            "correlation_values": target_corr.values.tolist(),
            "image": base64.b64encode(target_corr_image).decode('utf-8')
        }
    
    return {
        "correlation_matrix": corr_matrix.to_dict(),
        "correlation_image": base64.b64encode(image_png).decode('utf-8'),
        "strong_correlations": strong_corr_pairs,
        "target_correlations": target_correlations,
        "method": method
    }


def calculate_mutual_information(df: pd.DataFrame, target_column: str, categorical_features: List[str] = None) -> Dict:
    """
    计算特征与目标变量之间的互信息
    
    参数:
        df: 数据框
        target_column: 目标列名称
        categorical_features: 分类特征列表
        
    返回:
        包含互信息分析结果的字典
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
    
    # 计算互信息
    if is_classification:
        mi_scores = mutual_info_classif(X_numeric, y)
    else:
        mi_scores = mutual_info_regression(X_numeric, y)
    
    # 创建结果数据框
    mi_df = pd.DataFrame({'特征': numeric_cols, '互信息分数': mi_scores})
    mi_df = mi_df.sort_values('互信息分数', ascending=False)
    
    # 生成条形图
    plt.figure(figsize=(10, max(6, len(mi_df) * 0.3)))
    colors = generate_gradient_colors(len(mi_df))
    plt.barh(mi_df['特征'], mi_df['互信息分数'], color=colors)
    plt.title(f'特征与目标变量 "{target_column}" 的互信息分数')
    plt.xlabel('互信息分数')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return {
        "mutual_information": mi_df.to_dict('records'),
        "feature_names": mi_df['特征'].tolist(),
        "mi_scores": mi_df['互信息分数'].tolist(),
        "is_classification": is_classification,
        "image": base64.b64encode(image_png).decode('utf-8')
    }


def analyze_feature_importance(df: pd.DataFrame, target_column: str, categorical_features: List[str] = None, 
                              method: str = 'random_forest', n_estimators: int = 100) -> Dict:
    """
    使用机器学习模型分析特征重要性
    
    参数:
        df: 数据框
        target_column: 目标列名称
        categorical_features: 分类特征列表
        method: 特征重要性计算方法，可选 'random_forest', 'permutation'
        n_estimators: 随机森林的树数量
        
    返回:
        包含特征重要性分析结果的字典
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
    
    # 初始化模型
    if is_classification:
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    
    # 训练模型
    model.fit(X_numeric, y)
    
    # 计算特征重要性
    if method == 'random_forest':
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    elif method == 'permutation':
        result = permutation_importance(model, X_numeric, y, n_repeats=10, random_state=42)
        importances = result.importances_mean
        std = result.importances_std
    else:
        return {"error": f"不支持的方法: {method}"}
    
    # 创建结果数据框
    importance_df = pd.DataFrame({
        '特征': numeric_cols,
        '重要性': importances,
        '标准差': std
    })
    importance_df = importance_df.sort_values('重要性', ascending=False)
    
    # 生成条形图
    plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
    colors = generate_gradient_colors(len(importance_df))
    plt.barh(importance_df['特征'], importance_df['重要性'], xerr=importance_df['标准差'], color=colors)
    plt.title(f'特征重要性 ({method}方法)')
    plt.xlabel('重要性')
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    # 生成雷达图
    radar_image = None
    if len(importance_df) >= 3:
        # 准备数据
        top_features = importance_df.head(min(10, len(importance_df)))  # 最多取前10个特征
        features = top_features['特征'].tolist()
        values = top_features['重要性'].values
        
        # 标准化到0-1范围
        values = values / values.max() if values.max() > 0 else values
        
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
        plt.title('特征重要性雷达图')
        
        # 将图像转换为base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        radar_image = buffer.getvalue()
        buffer.close()
        plt.close()
    
    return {
        "feature_importance": importance_df.to_dict('records'),
        "feature_names": importance_df['特征'].tolist(),
        "importance_values": importance_df['重要性'].tolist(),
        "std_values": importance_df['标准差'].tolist(),
        "is_classification": is_classification,
        "method": method,
        "image": base64.b64encode(image_png).decode('utf-8'),
        "radar_image": base64.b64encode(radar_image).decode('utf-8') if radar_image else None
    }


def analyze_pairwise_relationships(df: pd.DataFrame, target_column: Optional[str] = None, 
                                  max_features: int = 5) -> Dict:
    """
    分析特征之间的成对关系，生成散点图矩阵
    
    参数:
        df: 数据框
        target_column: 目标列名称（可选）
        max_features: 最大特征数量
        
    返回:
        包含成对关系分析结果的字典
    """
    # 确保所有列都是数值型
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return {"error": "没有找到数值型列，无法分析成对关系"}
    
    # 如果有目标列，确保它在数据框中
    if target_column and target_column not in numeric_df.columns:
        return {"error": f"目标列 {target_column} 不是数值型或不存在"}
    
    # 选择要分析的特征
    if target_column:
        # 计算与目标的相关性
        correlations = numeric_df.corr()[target_column].abs().sort_values(ascending=False)
        # 选择相关性最高的特征（不包括目标本身）
        top_features = correlations.drop(target_column).head(max_features).index.tolist()
        # 确保目标列在选择的特征中
        selected_features = [target_column] + top_features
    else:
        # 如果没有目标列，选择方差最大的特征
        variances = numeric_df.var().sort_values(ascending=False)
        selected_features = variances.head(max_features).index.tolist()
    
    # 创建成对关系图
    plt.figure(figsize=(12, 10))
    sns.pairplot(numeric_df[selected_features], diag_kind='kde', 
                 plot_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
                 hue=target_column if target_column and pd.api.types.is_categorical_dtype(df[target_column]) else None)
    plt.suptitle('特征成对关系图', y=1.02, fontsize=16)
    plt.tight_layout()
    
    # 将图像转换为base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    return {
        "selected_features": selected_features,
        "image": base64.b64encode(image_png).decode('utf-8')
    }


def comprehensive_feature_analysis(df: pd.DataFrame, target_column: Optional[str] = None, 
                                 categorical_features: List[str] = None) -> Dict:
    """
    执行全面的特征分析，包括相关性、互信息和特征重要性
    
    参数:
        df: 数据框
        target_column: 目标列名称（可选）
        categorical_features: 分类特征列表
        
    返回:
        包含全面分析结果的字典
    """
    results = {
        "basic_info": {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_columns": df.select_dtypes(exclude=['number']).columns.tolist()
        }
    }
    
    # 相关性分析
    try:
        correlation_results = analyze_feature_correlations(df, target_column)
        results["correlation_analysis"] = correlation_results
    except Exception as e:
        results["correlation_analysis"] = {"error": str(e)}
    
    # 如果有目标列，执行更多分析
    if target_column and target_column in df.columns:
        # 互信息分析
        try:
            mi_results = calculate_mutual_information(df, target_column, categorical_features)
            results["mutual_information"] = mi_results
        except Exception as e:
            results["mutual_information"] = {"error": str(e)}
        
        # 特征重要性分析
        try:
            importance_results = analyze_feature_importance(df, target_column, categorical_features)
            results["feature_importance"] = importance_results
        except Exception as e:
            results["feature_importance"] = {"error": str(e)}
        
        # 成对关系分析
        try:
            pairwise_results = analyze_pairwise_relationships(df, target_column)
            results["pairwise_relationships"] = pairwise_results
        except Exception as e:
            results["pairwise_relationships"] = {"error": str(e)}
    
    return results