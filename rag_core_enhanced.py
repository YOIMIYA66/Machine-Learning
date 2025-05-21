# rag_core_enhanced.py
import os
import json
import pandas as pd
from typing import List, Optional, Dict, Any, Union
import numpy as np

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from config import (
    KNOWLEDGE_BASE_DIR, CHROMA_PERSIST_DIR, JSON_JQ_SCHEMA,
    CHUNK_SIZE, CHUNK_OVERLAP, AI_STUDIO_API_KEY
)
from baidu_llm import BaiduErnieEmbeddings, BaiduErnieLLM

# 导入机器学习相关模块
from ml_agents_enhanced import enhanced_query_ml_agent
from advanced_feature_analysis import integrate_ml_with_rag
from ml_models import load_model, predict

# 从原始rag_core.py导入函数
from rag_core import (
    load_and_parse_custom_json,
    generate_csv_summary_documents,
    load_documents_from_kb,
    initialize_rag_system,
    query_rag,
    direct_query_llm
)

# ---- 全局变量，用于缓存，避免重复加载 ----
_VECTOR_STORE: Optional[Chroma] = None
_QA_CHAIN: Optional[RetrievalQA] = None
_ML_MODELS_CACHE: Dict[str, Any] = {}
# --------------------------------


def enhanced_query_rag(query: str, ml_integration: bool = True) -> Dict[str, Any]:
    """
    增强版的RAG查询函数，支持与机器学习模型集成
    
    参数:
        query: 用户查询
        ml_integration: 是否启用机器学习模型集成
        
    返回:
        包含回答和相关文档的字典
    """
    # 首先使用原始RAG系统进行查询
    rag_result = query_rag(query)
    
    # 如果不需要机器学习集成，直接返回RAG结果
    if not ml_integration:
        return rag_result
    
    # 检测查询是否与机器学习相关
    ml_keywords = [
        '机器学习', '模型', '训练', '预测', '分类', '回归', '聚类',
        '随机森林', '决策树', '线性回归', '逻辑回归', 'KNN', 'SVM',
        '朴素贝叶斯', 'K-Means', '数据', '特征', '准确率', 'MSE', 'RMSE'
    ]
    ml_ops_keywords = ['训练', '预测', '比较', '评估', '构建', '解释', '自动', '集成', '版本', '分析', '推荐']
    
    is_ml_query = any(keyword.lower() in query.lower() for keyword in ml_keywords)
    is_ml_ops = any(op in query for op in ml_ops_keywords)
    
    # 如果是机器学习操作类查询，使用ML代理处理
    if is_ml_query and is_ml_ops:
        try:
            # 使用增强版ML代理，并将RAG结果传入以便集成
            ml_result = enhanced_query_ml_agent(query, use_existing_model=True, 
                                              integrate_with_rag=True, rag_result=rag_result)
            return ml_result
        except Exception as e:
            print(f"ML代理处理时出错，回退到RAG结果: {str(e)}")
            return rag_result
    
    # 如果是知识类查询但可能需要模型预测，尝试找到合适的模型并集成预测结果
    elif is_ml_query and "预测" in query:
        try:
            # 尝试从查询中提取预测目标和特征
            prediction_target, features = extract_prediction_info(query)
            
            if prediction_target and features:
                # 查找适合该预测任务的模型
                model_name = find_suitable_model(prediction_target)
                
                if model_name:
                    # 加载模型并进行预测
                    model_result = make_prediction_with_model(model_name, features)
                    
                    # 将模型预测结果与RAG结果集成
                    enhanced_result = integrate_ml_with_rag(rag_result, model_name, {
                        "prediction": model_result.get("prediction"),
                        "feature_importance": model_result.get("feature_importance"),
                        "model_metrics": model_result.get("metrics")
                    })
                    
                    return enhanced_result
        except Exception as e:
            print(f"模型预测集成时出错，回退到RAG结果: {str(e)}")
    
    # 对于其他情况，返回原始RAG结果
    return rag_result


def extract_prediction_info(query: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    从查询中提取预测目标和特征信息
    
    参数:
        query: 用户查询
        
    返回:
        (预测目标, 特征字典)
    """
    # 这里使用简单的启发式方法，实际应用中可能需要更复杂的NLP技术
    prediction_target = None
    features = {}
    
    # 尝试提取预测目标
    if "预测" in query:
        parts = query.split("预测")
        if len(parts) > 1:
            # 取预测后面的第一个词作为目标
            words = parts[1].strip().split()
            if words:
                prediction_target = words[0].strip(",.?!;:")
    
    # 尝试提取特征
    feature_indicators = ["特征", "参数", "条件", "值", "是", "为"]
    for indicator in feature_indicators:
        if indicator in query:
            parts = query.split(indicator)
            for i in range(1, len(parts)):
                # 尝试提取"X是Y"或"X为Y"这样的模式
                words = parts[i].strip().split()
                if len(words) >= 2:
                    feature_name = words[0].strip(",.?!;:")
                    feature_value = words[1].strip(",.?!;:")
                    # 尝试将特征值转换为数值
                    try:
                        if '.' in feature_value:
                            features[feature_name] = float(feature_value)
                        else:
                            features[feature_name] = int(feature_value)
                    except ValueError:
                        features[feature_name] = feature_value
    
    return prediction_target, features if features else None


def find_suitable_model(prediction_target: str) -> Optional[str]:
    """
    根据预测目标找到合适的模型
    
    参数:
        prediction_target: 预测目标
        
    返回:
        模型名称
    """
    # 导入模型列表函数
    from ml_models import list_available_models
    
    # 获取所有可用模型
    models = list_available_models()
    
    # 根据预测目标匹配模型
    for model in models:
        model_metadata = model.get("metadata", {})
        target_name = model_metadata.get("target_name", "")
        description = model_metadata.get("description", "")
        
        # 检查目标名称或描述是否匹配预测目标
        if (prediction_target.lower() in target_name.lower() or 
            prediction_target.lower() in description.lower()):
            return model["name"]
    
    # 如果没有找到匹配的模型，返回None
    return None


def make_prediction_with_model(model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用指定模型进行预测
    
    参数:
        model_name: 模型名称
        features: 特征字典
        
    返回:
        预测结果字典
    """
    # 检查模型是否已缓存
    if model_name not in _ML_MODELS_CACHE:
        # 加载模型
        model, preprocessors, metadata = load_model(model_name)
        _ML_MODELS_CACHE[model_name] = (model, preprocessors, metadata)
    else:
        model, preprocessors, metadata = _ML_MODELS_CACHE[model_name]
    
    # 使用模型进行预测
    prediction_result = predict(model_name=model_name, input_data=features)
    
    return prediction_result


def enhanced_direct_query_llm(query: str, ml_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    增强版的直接大模型查询，可以包含机器学习上下文
    
    参数:
        query: 用户查询
        ml_context: 机器学习相关上下文信息
        
    返回:
        包含回答的字典
    """
    # 初始化LLM
    llm = BaiduErnieLLM(api_key=AI_STUDIO_API_KEY)
    
    # 构建提示
    if ml_context:
        # 如果有机器学习上下文，将其添加到提示中
        prompt = f"""请回答以下问题，并参考提供的机器学习模型信息：

问题: {query}

机器学习上下文:
"""
        
        # 添加模型名称
        if "model_name" in ml_context:
            prompt += f"\n- 使用的模型: {ml_context['model_name']}"
        
        # 添加预测结果
        if "prediction" in ml_context:
            prompt += f"\n- 模型预测结果: {ml_context['prediction']}"
        
        # 添加特征重要性
        if "feature_importance" in ml_context and "top_features" in ml_context["feature_importance"]:
            prompt += "\n- 重要特征:"  
            for i, feature in enumerate(ml_context["feature_importance"]["top_features"][:3]):
                importance = ml_context["feature_importance"]["importance_values"][i] if i < len(ml_context["feature_importance"]["importance_values"]) else "未知"
                prompt += f"\n  {i+1}. {feature} (重要性: {importance})"
        
        # 添加模型性能指标
        if "model_metrics" in ml_context:
            prompt += "\n- 模型性能指标:"
            for metric_name, metric_value in ml_context["model_metrics"].items():
                prompt += f"\n  - {metric_name}: {metric_value}"
        
        prompt += "\n\n请结合上述机器学习信息和你的知识，给出全面、准确的回答。"
    else:
        # 如果没有机器学习上下文，使用原始提示
        prompt = f"请回答以下问题: {query}"
    
    # 调用LLM
    response = llm.invoke(prompt)
    
    return {
        "answer": response,
        "is_direct_answer": True,
        "ml_enhanced": ml_context is not None
    }


def enhanced_initialize_rag_system(force_recreate_vs: bool = False) -> None:
    """
    增强版的RAG系统初始化函数，包括机器学习模型缓存初始化
    
    参数:
        force_recreate_vs: 是否强制重新创建向量存储
    """
    # 调用原始初始化函数
    initialize_rag_system(force_recreate_vs)
    
    # 清空模型缓存
    global _ML_MODELS_CACHE
    _ML_MODELS_CACHE = {}
    
    # 预加载常用模型
    try:
        from ml_models import list_available_models
        models = list_available_models()
        
        # 只预加载前几个模型，避免占用过多内存
        for model in models[:3]:  # 只预加载前3个模型
            model_name = model["name"]
            try:
                model_obj, preprocessors, metadata = load_model(model_name)
                _ML_MODELS_CACHE[model_name] = (model_obj, preprocessors, metadata)
                print(f"预加载模型: {model_name}")
            except Exception as e:
                print(f"预加载模型 {model_name} 时出错: {str(e)}")
    except Exception as e:
        print(f"预加载模型时出错: {str(e)}")