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
import traceback

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
    
    # 检查是否是预测或模拟请求
    # 简单的启发式：检查查询中是否包含"预测"、"模拟"或"当...时预测"等关键词
    if any(keyword in query for keyword in ['预测', '模拟']) or ('当' in query and '时' in query and '预测' in query):
        try:
            # 尝试从查询中提取预测目标和特征
            prediction_target, features = extract_prediction_info(query)

            if prediction_target and features:
                # 查找适合该预测任务的模型
                # 优先使用用户已选模型（如果前端传递了），如果没有，则尝试自动选择或查找匹配目标列的模型
                # 注意：当前enhanced_query_ml_agent没有接收selected_model参数，这里先尝试根据目标查找
                model_name = find_suitable_model(prediction_target)

                if model_name:
                    # 使用模型进行预测
                    prediction_result = make_prediction_with_model(model_name, features)

                    # 将预测结果添加到ml_context中，以便LLM生成回答
                    ml_context_for_llm = {
                        "model_name": model_name,
                        "prediction": prediction_result.get("prediction"),
                        "features_used": features,
                        "model_metrics": prediction_result.get("metrics"),
                        "feature_importance": prediction_result.get("feature_importance")
                    }

                    # 使用增强版LLM生成回答，包含预测结果
                    # 注意：这里直接调用enhanced_direct_query_llm，而不是通过RAG流程
                    llm_response = enhanced_direct_query_llm(query, ml_context=ml_context_for_llm)

                    # 将LLM的回答和可能的其他结果合并到最终结果中
                    result["answer"] = llm_response.get("answer", result.get("answer", ""))
                    result["ml_enhanced"] = True
                    # 将预测结果、模型指标和特征重要性添加到最终结果中，以便前端结构化展示
                    result["prediction"] = prediction_result.get("prediction")
                    result["model_metrics"] = prediction_result.get("metrics")
                    result["feature_importance"] = prediction_result.get("feature_importance")

                    return result # 返回包含预测结果的增强结果

                else:
                    result["answer"] = result.get("answer", "") + "\n\n抱歉，未能找到适合进行预测的模型。" # 如果找不到模型，更新回答
            else:
                 result["answer"] = result.get("answer", "") + "\n\n抱歉，未能从您的查询中提取出完整的预测所需特征信息。请提供明确的特征名称和值。" # 如果提取特征失败，更新回答
        except Exception as e:
            print(f"处理预测请求时出错: {str(e)}")
            traceback.print_exc() # 打印详细错误信息
            result["answer"] = result.get("answer", "") + "\n\n处理预测请求时发生错误: {}".format(str(e)) # 发生错误时更新回答

    return result # 对于非预测请求或处理失败的情况，返回原始或部分更新的结果


def enhanced_direct_query_llm(query: str, ml_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    增强版的直接大模型查询，可以包含机器学习上下文
    """

def extract_prediction_info(query: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    从用户查询中提取预测目标和特征信息
    
    参数:
        query: 用户查询字符串
        
    返回:
        (预测目标, 特征字典)
    """
    try:
        # 初始化返回值
        prediction_target = None
        features = {}
        
        # 提取预测目标
        # 查找常见的目标提示词
        target_keywords = ["预测", "预估", "估计", "计算", "判断"]
        target_found = False
        
        for keyword in target_keywords:
            if keyword in query:
                parts = query.split(keyword)
                if len(parts) > 1:
                    # 假设预测目标跟在关键词后面
                    target_candidate = parts[1].split()[0].strip()
                    if target_candidate:
                        prediction_target = target_candidate
                        target_found = True
                        break
        
        # 如果没有通过关键词找到，尝试寻找被引号包围的可能目标
        if not target_found:
            import re
            quoted_terms = re.findall(r'["\'](.*?)["\']', query)
            if quoted_terms:
                prediction_target = quoted_terms[0]
        
        # 提取特征信息
        # 查找常见的特征表达模式
        feature_patterns = [
            r'(\w+)\s*[=:：]\s*(\d+\.?\d*|\w+)',  # 变量=值
            r'(\w+)\s*为\s*(\d+\.?\d*|\w+)',      # 变量为值
            r'(\w+)\s*是\s*(\d+\.?\d*|\w+)',      # 变量是值
            r'当\s*(\w+)\s*[为是]\s*(\d+\.?\d*|\w+)'  # 当变量为/是值
        ]
        
        import re
        for pattern in feature_patterns:
            matches = re.findall(pattern, query)
            for match in matches:
                feature_name, feature_value = match
                # 尝试将数值转换为浮点数
                try:
                    if feature_value.replace('.', '', 1).isdigit():
                        feature_value = float(feature_value)
                except:
                    pass  # 如果不是数字，保持原样
                
                features[feature_name] = feature_value
        
        return prediction_target, features
    
    except Exception as e:
        print(f"提取预测信息时出错: {str(e)}")
        return None, None

def find_suitable_model(prediction_target: str) -> Optional[str]:
    """
    根据预测目标找到适合的模型
    
    参数:
        prediction_target: 预测目标字段名
        
    返回:
        适合的模型名称，如果没有找到则返回None
    """
    try:
        # 从ml_models模块导入列出可用模型的函数
        from ml_models import list_available_models
        
        # 获取所有可用模型
        available_models = list_available_models()
        
        # 首先，寻找名称中包含目标字段的模型
        target_models = []
        for model in available_models:
            model_name = model.get("name", "").lower()
            if prediction_target.lower() in model_name:
                target_models.append(model.get("name"))
        
        # 如果找到了包含目标字段的模型，返回第一个
        if target_models:
            return target_models[0]
        
        # 如果没有找到匹配的模型，尝试根据目标字段推断模型类型
        # 例如，某些字段名称可能暗示分类或回归任务
        regression_keywords = ["价格", "收入", "销量", "数量", "分数", "得分", "年龄", "工资"]
        classification_keywords = ["类别", "分类", "类型", "是否", "标签", "等级"]
        
        is_regression = any(keyword in prediction_target.lower() for keyword in regression_keywords)
        is_classification = any(keyword in prediction_target.lower() for keyword in classification_keywords)
        
        # 根据推断的任务类型选择模型
        for model in available_models:
            model_type = model.get("type", "").lower()
            model_name = model.get("name", "")
            
            if is_regression and "regression" in model_type:
                return model_name
            elif is_classification and ("classifier" in model_type or "classification" in model_type):
                return model_name
        
        # 如果仍然没有找到，返回任何可用的机器学习模型
        for model in available_models:
            if "model" in model.get("name", "").lower():
                return model.get("name")
        
        # 如果有任何模型，返回第一个
        if available_models:
            return available_models[0].get("name")
        
        # 没有可用模型
        return None
    
    except Exception as e:
        print(f"查找适合模型时出错: {str(e)}")
        return None

def make_prediction_with_model(model_name: str, features: Dict) -> Dict:
    """
    使用指定模型对给定特征进行预测
    
    参数:
        model_name: 模型名称
        features: 特征字典 {特征名: 特征值}
        
    返回:
        包含预测结果、特征重要性和模型指标的字典
    """
    try:
        # 导入需要的函数
        from ml_models import load_model, predict, explain_model_prediction
        
        # 准备输入数据格式
        input_data = pd.DataFrame([features])
        
        # 加载模型
        model, model_metadata = load_model(model_name)
        if not model:
            return {
                "error": f"无法加载模型 {model_name}",
                "predictions": None
            }
        
        # 进行预测
        prediction_result = predict(model_name=model_name, input_data=features)
        
        # 提取预测结果
        predictions = prediction_result.get("predictions")
        
        # 尝试解释预测结果
        try:
            explanation = explain_model_prediction(model=model, 
                                                 features=features, 
                                                 prediction=predictions[0] if isinstance(predictions, list) else predictions)
            
            feature_importance = explanation.get("feature_importance", {})
        except Exception as e:
            print(f"解释预测结果时出错: {str(e)}")
            feature_importance = {}
        
        # 获取模型指标
        metrics = {}
        if "accuracy" in prediction_result:
            metrics["accuracy"] = prediction_result["accuracy"]
        if "mse" in prediction_result:
            metrics["mse"] = prediction_result["mse"]
        if "r2" in prediction_result:
            metrics["r2"] = prediction_result["r2"]
        
        # 返回结果
        return {
            "predictions": predictions,
            "feature_importance": feature_importance,
            "metrics": metrics,
            "model_used": model_name
        }
    
    except Exception as e:
        print(f"使用模型进行预测时出错: {str(e)}")
        traceback.print_exc()  # 打印详细错误信息
        return {
            "error": f"预测过程中出错: {str(e)}",
            "predictions": None
        }