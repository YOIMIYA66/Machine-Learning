# ml_agents_enhanced.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple
import colorsys
import uuid

from langchain.tools import StructuredTool
from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field
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

# 导入特征分析模块
from feature_analysis import (
    analyze_feature_correlations, 
    calculate_mutual_information, 
    analyze_feature_importance,
    analyze_pairwise_relationships,
    comprehensive_feature_analysis
)

# 导入高级特征分析模块
from advanced_feature_analysis import (
    analyze_feature_stability,
    analyze_feature_interactions,
    analyze_feature_nonlinearity,
    advanced_feature_analysis,
    integrate_ml_with_rag
)

# 从原始ml_agents.py导入必要的函数
from ml_agents import (
    query_ml_agent, generate_gradient_colors, generate_visualization,
    visualize_feature_importance, visualize_confusion_matrix,
    visualize_metrics, visualize_clusters, generate_data_table,
    visualize_feature_importance_radar, visualize_model_comparison,
    TrainModelInput, PredictInput, RecommendModelInput, DataAnalysisInput,
    EvaluateModelInput, EnsembleModelInput, AutoSelectModelInput,
    ExplainPredictionInput, CompareModelsInput, VersionModelInput,
    ListModelVersionsInput
)


def enhanced_data_analysis(file_path: str, target_column: Optional[str] = None, 
                         analysis_type: str = 'comprehensive', 
                         categorical_columns: List[str] = None) -> Dict:
    """
    执行增强的数据分析，包括基础分析和高级特征分析
    
    参数:
        file_path: 数据文件路径
        target_column: 目标列名称
        analysis_type: 分析类型，可选 'basic', 'comprehensive', 'advanced'
        categorical_columns: 分类特征列表
        
    返回:
        包含分析结果的字典
    """
    try:
        # 读取数据文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            return {"error": "不支持的文件格式，仅支持CSV、Excel和JSON"}
        
        # 基本数据信息
        basic_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "data_types": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
            "unique_values": {col: int(df[col].nunique()) for col in df.columns}
        }
        
        # 根据分析类型执行不同级别的分析
        if analysis_type == 'basic':
            # 基础分析
            correlation_results = analyze_feature_correlations(df, target_column)
            
            return {
                "basic_info": basic_info,
                "correlation_analysis": correlation_results,
                "analysis_type": analysis_type
            }
            
        elif analysis_type == 'comprehensive':
            # 综合分析
            comprehensive_results = comprehensive_feature_analysis(df, target_column, categorical_columns)
            
            return {
                "basic_info": basic_info,
                "comprehensive_analysis": comprehensive_results,
                "analysis_type": analysis_type
            }
            
        elif analysis_type == 'advanced':
            # 高级分析
            advanced_results = advanced_feature_analysis(df, target_column, categorical_columns)
            
            return {
                "basic_info": basic_info,
                "advanced_analysis": advanced_results,
                "analysis_type": analysis_type
            }
        else:
            return {"error": f"不支持的分析类型: {analysis_type}"}
            
    except Exception as e:
        return {"error": f"数据分析过程中出错: {str(e)}"}


def enhanced_query_ml_agent(query: str, use_existing_model: bool = True, 
                          integrate_with_rag: bool = False, rag_result: Dict = None) -> Dict:
    """
    增强版的ML代理查询函数，支持高级特征分析和与RAG模型的集成
    
    参数:
        query: 用户查询
        use_existing_model: 是否使用现有模型
        integrate_with_rag: 是否与RAG模型集成
        rag_result: RAG查询结果
        
    返回:
        包含回答和分析结果的字典
    """
    # 首先调用原始的ML代理查询函数
    result = query_ml_agent(query, use_existing_model)
    
    # 检查是否需要进行高级特征分析
    if "需要高级特征分析" in query or "特征重要性" in query or "特征交互" in query or "特征稳定性" in query:
        # 提取查询中可能包含的数据文件路径和目标列
        # 这里使用简单的启发式方法，实际应用中可能需要更复杂的NLP技术
        file_path = None
        target_column = None
        analysis_type = 'advanced'
        
        # 尝试从查询中提取文件路径
        if "文件" in query or "数据" in query:
            # 简单启发式：查找.csv, .xlsx, .json等文件扩展名
            for word in query.split():
                if word.endswith(('.csv', '.xlsx', '.xls', '.json')):
                    file_path = word
                    break
        
        # 尝试从查询中提取目标列
        if "目标" in query or "预测" in query:
            # 简单启发式：查找"目标是"或"预测"后面的词
            parts = query.split("目标是")
            if len(parts) > 1:
                target_column = parts[1].split()[0].strip()
            else:
                parts = query.split("预测")
                if len(parts) > 1:
                    target_column = parts[1].split()[0].strip()
        
        # 如果找到文件路径，执行高级特征分析
        if file_path:
            analysis_result = enhanced_data_analysis(file_path, target_column, analysis_type)
            result["feature_analysis"] = analysis_result
            
            # 更新回答，包含特征分析结果
            original_answer = result.get("answer", "")
            enhanced_answer = original_answer + "\n\n我已经对数据进行了高级特征分析，以下是主要发现：\n"
            
            # 添加特征重要性信息
            if "advanced_analysis" in analysis_result and "feature_importance" in analysis_result["advanced_analysis"]:
                feature_importance = analysis_result["advanced_analysis"]["feature_importance"]
                top_features = feature_importance.get("feature_names", [])[:3]  # 取前3个重要特征
                importance_values = feature_importance.get("importance_values", [])[:3]
                
                enhanced_answer += "\n主要特征重要性："
                for i, (feature, importance) in enumerate(zip(top_features, importance_values)):
                    enhanced_answer += f"\n{i+1}. {feature} (重要性: {importance:.4f})"
            
            # 添加特征交互信息
            if "advanced_analysis" in analysis_result and "interaction_analysis" in analysis_result["advanced_analysis"]:
                interaction = analysis_result["advanced_analysis"]["interaction_analysis"]
                if "interaction_results" in interaction and len(interaction["interaction_results"]) > 0:
                    top_interaction = interaction["interaction_results"][0]
                    enhanced_answer += f"\n\n发现最显著的特征交互：{top_interaction.get('feature1', '')} 与 {top_interaction.get('feature2', '')}"
            
            # 添加非线性关系信息
            if "advanced_analysis" in analysis_result and "nonlinearity_analysis" in analysis_result["advanced_analysis"]:
                nonlinearity = analysis_result["advanced_analysis"]["nonlinearity_analysis"]
                if "nonlinearity_results" in nonlinearity and len(nonlinearity["nonlinearity_results"]) > 0:
                    # 按非线性增益排序
                    sorted_results = sorted(nonlinearity["nonlinearity_results"], 
                                          key=lambda x: x.get("nonlinearity_gain", 0), 
                                          reverse=True)
                    if len(sorted_results) > 0:
                        top_nonlinear = sorted_results[0]
                        enhanced_answer += f"\n\n发现最显著的非线性关系：{top_nonlinear.get('feature', '')} (非线性增益: {top_nonlinear.get('nonlinearity_gain', 0):.4f})"
            
            result["answer"] = enhanced_answer
    
    # 检查是否需要与RAG模型集成
    if integrate_with_rag and rag_result:
        # 提取可能使用的模型名称
        model_name = result.get("model_used", "未知模型")
        
        # 提取特征分析数据
        feature_data = result.get("feature_analysis", {})
        
        # 集成ML结果与RAG结果
        integrated_result = integrate_ml_with_rag(rag_result, model_name, feature_data)
        
        # 返回集成结果
        return integrated_result
    
    return result