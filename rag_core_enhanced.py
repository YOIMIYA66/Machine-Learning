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
    #  从查询中提取预测目标和特征信息。
     
    #  @param query 用户查询字符串。
    #  @returns 包含预测目标和特征字典的元组。
     
    prediction_target = None
    features = {}

    # 尝试提取预测目标 (更灵活的方式)
    prediction_phrases = ["预测", "预测什么", "预测一下", "计算", "估计"] # 增加更多预测相关的词汇
    for phrase in prediction_phrases:
        if phrase in query:
            parts = query.split(phrase, 1) # 只分割一次
            if len(parts) > 1:
                # 尝试从分割后的第二部分提取目标
                remaining_query = parts[1].strip()
                # 简单启发式：取第一个名词或关键短语作为目标
                # 这里可以进一步增强，例如使用NLP库进行词性标注和实体识别
                words = remaining_query.split()
                if words:
                    # 过滤掉一些非目标词汇，例如“的”、“值”等
                    potential_target = words[0].strip(",.?!;:的的值")
                    if potential_target and len(potential_target) > 1: # 避免单字符或标点作为目标
                         prediction_target = potential_target
                         break # 找到目标后停止搜索

    # 尝试提取特征 (更灵活的方式)
    # 查找“当...时”、“如果...”、“在...情况下”、“...是...”、“...为...”等模式
    import re
    # 匹配 "特征名 是/为 特征值" 或 "特征名 为 特征值" 等模式
    # 使用正则表达式查找 "词语 是/为 词语" 的模式
    feature_pattern = re.compile(r'(\S+)\s*[是为]\s*(\S+)')
    
    # 检查整个查询字符串
    matches = feature_pattern.findall(query)
    
    for name, value_str in matches:
        feature_name = name.strip(",.?!;:")
        feature_value_str = value_str.strip(",.?!;:")
        
        # 尝试将特征值转换为数值
        try:
            if '.' in feature_value_str:
                features[feature_name] = float(feature_value_str)
            else:
                features[feature_name] = int(feature_value_str)
        except ValueError:
            # 如果不是数值，保留为字符串
            features[feature_name] = feature_value_str
            
    # 进一步尝试从预测目标后的剩余查询中提取特征
    # 如果找到了预测目标，只分析目标后的部分
    if prediction_target:
        prediction_phrases = [phrase.lower() for phrase in ["预测", "预测什么", "预测一下", "计算", "估计"]]
        query_lower = query.lower()
        remaining_query = query
        for phrase in prediction_phrases:
            if phrase in query_lower:
                parts = query.split(phrase, 1)
                if len(parts) > 1:
                    remaining_query = parts[1].strip()
                    break
        
        # 在剩余查询中查找特征模式
        matches_remaining = feature_pattern.findall(remaining_query)
        for name, value_str in matches_remaining:
            feature_name = name.strip(",.?!;:")
            feature_value_str = value_str.strip(",.?!;:")
            
            # 避免重复添加已提取的特征
            if feature_name not in features:
                 # 尝试将特征值转换为数值
                try:
                    if '.' in feature_value_str:
                        features[feature_name] = float(feature_value_str)
                    else:
                        features[feature_name] = int(feature_value_str)
                except ValueError:
                    # 如果不是数值，保留为字符串
                    features[feature_name] = feature_value_str

    # 如果提取到了特征，返回特征字典，否则返回None
    return prediction_target, features if features else None


def find_suitable_model(prediction_target: str) -> Optional[str]:
    # /**
    #  * 根据预测目标找到合适的模型。
    #  *
    #  * @param prediction_target 预测目标字符串。
    #  * @returns 匹配到的模型名称，如果没有找到则返回None。
    #  */
    # 导入模型选择函数
    from ml_models import select_model_for_task
    
    # 直接调用 ml_models 中的模型选择函数
    # 将预测目标作为任务描述传入
    model_name = select_model_for_task(prediction_target)
    
    return model_name


def make_prediction_with_model(model_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
    # /**
    #  * 使用指定模型进行预测。
    #  *
    #  * @param model_name 模型名称。
    #  * @param features 特征字典，用于模型预测。
    #  * @returns 包含预测结果和相关信息的字典，结构与 integrate_ml_with_rag 期望的 ml_prediction_info 参数一致。
    #  */
    # 检查模型是否已缓存
    if model_name not in _ML_MODELS_CACHE:
        # 加载模型
        try:
            model, preprocessors, metadata = load_model(model_name)
            _ML_MODELS_CACHE[model_name] = (model, preprocessors, metadata)
        except FileNotFoundError:
            print(f"错误: 未找到模型文件 '{model_name}'.pkl")
            return {"error": f"模型 '{model_name}' 未找到"}
        except Exception as e:
            print(f"加载模型 '{model_name}' 时出错: {str(e)}")
            return {"error": f"加载模型 '{model_name}' 失败: {str(e)}"}
    else:
        model, preprocessors, metadata = _ML_MODELS_CACHE[model_name]
    
    # 使用模型进行预测
    try:
        prediction_output = predict(model_name=model_name, input_data=features)
        
        # 提取预测结果，假设输入是单个数据点，取predictions列表的第一个元素
        prediction_value = prediction_output.get("predictions", [None])[0]
        
        # 构建返回字典，与integrate_ml_with_rag的期望结构对齐
        # 注意：ml_models.predict 目前不直接返回 feature_importance 或 model_metrics
        # 如果需要这些信息，可能需要在训练时保存到metadata并在load_model时加载
        return {
            "prediction": prediction_value,
            "model_name": model_name,
            "feature_importance": None, # 预测时通常不计算特征重要性
            "model_metrics": None # 预测时通常不计算模型指标
            # "raw_prediction_output": prediction_output # 可以选择保留原始输出用于调试
        }
        
    except Exception as e:
        print(f"使用模型 '{model_name}' 进行预测时出错: {str(e)}")
        return {"error": f"预测失败: {str(e)}"}


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
    if ml_context and ml_context.get('generate_tutorial'):
        # 为教程生成构建特定提示
        data_preview_str = "数据预览:\n"
        if ml_context.get('data_preview'):
            try:
                preview_df = pd.DataFrame(ml_context['data_preview'])
                data_preview_str += preview_df.to_string(index=False, max_rows=5)
            except Exception as e:
                data_preview_str += f"(无法格式化预览: {str(e)})\n{json.dumps(ml_context['data_preview'], indent=2, ensure_ascii=False)}"
        else:
            data_preview_str += "无可用数据预览。"

        prompt = f"""您是一位机器学习辅导老师。请根据以下信息，为用户生成一份详细的教程，解释如何使用Python的sklearn库来实现指定的机器学习模型。

用户信息：
- 选择的模型: {ml_context.get('model_name', '未指定')}
- 选择的目标列: {ml_context.get('target_column', '未指定')}
- {data_preview_str}

教程应包含以下内容：
1.  对所选机器学习模型 ({ml_context.get('model_name', '未指定')}) 的基本原理、适用场景、优点和缺点的详细介绍。
2.  对用户提供的数据集（基于以上预览）进行简要分析，特别是目标列 '{ml_context.get('target_column', '未指定')}' 的特性（例如，是分类还是回归，数据类型等）。
3.  提供一个使用sklearn库实现所选模型的完整Python代码示例。代码应包含：
    a.  必要的库导入 (如 pandas, sklearn.model_selection, 以及选定模型的sklearn实现)。
    b.  假设数据已加载到名为 `df` 的Pandas DataFrame中，展示如何准备特征 (X) 和目标 (y)。明确指出如何处理目标列 '{ml_context.get('target_column', '未指定')}'。
    c.  数据预处理步骤的建议（例如，处理缺失值、分类特征编码、数值特征缩放等），并提供相关代码片段（如果适用）。
    d.  将数据划分为训练集和测试集。
    e.  初始化、训练选定的模型。
    f.  （如果适用）使用训练好的模型在测试集上进行预测。
    g.  （如果适用）展示如何评估模型性能（例如，分类任务的准确率、精确率、召回率、F1分数、混淆矩阵；回归任务的MSE, R2分数等）。
4.  对代码中每个关键步骤进行清晰的解释。
5.  总结，并给出一些关于如何进一步改进模型或应用的建议。

请确保教程内容详实、易于理解，并且代码示例可以直接运行（假设用户已安装必要的库并将数据加载到`df`中）。
"""
    elif ml_context:
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