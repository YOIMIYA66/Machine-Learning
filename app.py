# app.py
import sys
import os
import logging
import json
import uuid
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
import datetime
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# 导入配置和核心功能模块
from config import KNOWLEDGE_BASE_DIR, AI_STUDIO_API_KEY
from rag_core import query_rag, initialize_rag_system, direct_query_llm
from ml_agents import query_ml_agent

# 导入增强版RAG和ML集成功能
from rag_core_enhanced import enhanced_query_rag, enhanced_direct_query_llm
from ml_agents_enhanced import enhanced_query_ml_agent
from advanced_feature_analysis import integrate_ml_with_rag

# Helper functions moved to the top
def is_rag_result_poor(query, rag_result):
    """
    评估RAG结果质量是否较差

    评估指标:
    1. 相关性 - 检查RAG的回答是否与问题相关
    2. 确定性 - 检查答案是否包含"未找到"、"没有相关信息"等不确定表述
    3. 置信度 - 检查文档检索的分数是否过低
    """
    answer = rag_result.get("answer", "")

    # 检查不确定性表达
    uncertainty_phrases = [
        "无法找到", "没有相关信息", "未能找到", "无法提供",
        "我不知道", "无法确定", "没有足够信息",
        "To", "I cannot", "I don't", "Unable to"  # 英文回答中的不确定性表达
    ]

    if any(phrase in answer for phrase in uncertainty_phrases):
        return True

    # 检查文档检索分数
    source_docs = rag_result.get("source_documents", [])
    if source_docs:
        # 获取最高相关性分数
        max_score = max(
            [doc.get("score", 0) for doc in source_docs]
            if all("score" in doc for doc in source_docs)
            else [0]
        )
        # 如果最高分数低于阈值，认为结果质量较差
        if max_score < 0.45:  # 可根据实际情况调整阈值
            return True

    # 如果回答过短或过长也可能表示质量问题
    if len(answer.strip()) < 30 or "The answer is" in answer:
        return True

    return False

# 创建一个线程池执行器用于异步任务
executor = ThreadPoolExecutor(max_workers=4)

app = Flask(__name__)  # Flask会自动查找同级的 'templates' 文件夹
CORS(app)

# --- 日志配置 ---
# 基本配置，确保在 app.run() 之前设置，或者由 Flask 的 debug 模式自动处理
# 如果不是在debug模式下运行，或者需要更精细的控制，可以取消注释并调整下面的配置
# if not app.debug:
#     log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     # To console
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(log_formatter)
#     app.logger.addHandler(stream_handler)
#     # Optionally, to a file
#     # file_handler = logging.FileHandler('app.log')
#     # file_handler.setFormatter(log_formatter)
#     # app.logger.addHandler(file_handler)
#     app.logger.setLevel(logging.INFO)
# else:
#     # Debug模式下，Flask通常有自己的日志处理器，这里确保级别
app.logger.setLevel(logging.INFO)
# -----------------

@app.route('/')
def index():
    """渲染主HTML页面。"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_endpoint():
    """处理用户查询的端点"""
    try:
        data = request.json
        query = data.get('query', '')

        if not query:
            return jsonify({"error": "请提供查询文本"}), 400

        app.logger.info(f"接收到查询请求: {query}")

        # 机器学习相关查询检测
        ml_keywords = [
            '机器学习', '模型', '训练', '预测', '分类', '回归', '聚类',
            '随机森林', '决策树', '线性回归', '逻辑回归', 'KNN', 'SVM',
            '朴素贝叶斯', 'K-Means', '数据', '特征', '准确率', 'MSE', 'RMSE'
        ]
        # 操作类关键词
        ml_ops_keywords = ['训练', '预测', '比较', '评估', '构建', '解释', '自动', '集成', '版本', '分析', '推荐']

        is_ml_query = any(keyword.lower() in query.lower() for keyword in ml_keywords)
        is_ml_ops = any(op in query for op in ml_ops_keywords)

        # 1. 操作类问题优先走增强版ML Agent
        if is_ml_query and is_ml_ops:
            try:
                app.logger.info("检测到机器学习操作类查询，使用增强版ML Agent处理")
                result = enhanced_query_ml_agent(query, use_existing_model=True)
                return jsonify(result)
            except Exception as e:
                app.logger.error(f"增强版ML Agent处理时出错，回退到RAG: {str(e)}")
                # 尝试使用标准ML Agent
                try:
                    app.logger.info("尝试使用标准ML Agent处理")
                    result = query_ml_agent(query)
                    return jsonify(result)
                except Exception as e2:
                    app.logger.error(f"标准ML Agent处理时出错，回退到RAG: {str(e2)}")
                    # 机器学习处理失败时回退到RAG系统

        # 2. 专业知识问答优先走增强版RAG
        app.logger.info("使用增强版RAG系统处理常规/知识类查询")
        try:
            # 尝试使用增强版RAG处理
            result = enhanced_query_rag(query)
        except Exception as e:
            app.logger.warning(f"增强版RAG处理失败，回退到标准RAG: {str(e)}")
            # 回退到标准RAG
            result = query_rag(query)
            
        # 3. RAG效果不佳时兜底增强版LLM
        if is_rag_result_poor(query, result):
            app.logger.info("RAG结果质量不佳，切换到直接大模型回答")
            try:
                direct_llm_response = enhanced_direct_query_llm(query)
            except Exception as e:
                app.logger.warning(f"增强版LLM处理失败，回退到标准LLM: {str(e)}")
                direct_llm_response = direct_query_llm(query)
                
            result["answer"] = direct_llm_response["answer"]
            result["is_direct_answer"] = direct_llm_response.get("is_direct_answer", True)
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"处理查询时出错: {str(e)}")
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500

@app.route('/api/models/ml_models', methods=['GET'])
def get_ml_models():
    """
    获取ml_models目录中的模型列表
    
    返回:
        JSON格式的模型列表
    """
    try:
        model_dir = os.path.join(os.path.dirname(__file__), 'ml_models')
        
        # 确保目录存在
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            return jsonify({"models": [], "message": "ml_models目录已创建"})
            
        # 获取所有模型文件，支持多种格式
        model_files = [f for f in os.listdir(model_dir) 
                      if f.endswith(('.pkl', '.joblib', '.h5', '.keras')) 
                      and os.path.isfile(os.path.join(model_dir, f))]
        
        # 提取模型名称(去掉扩展名)
        model_names = [os.path.splitext(f)[0] for f in model_files]
        
        # 添加模型描述信息
        models_info = [
            {
                "name": name,
                "path": os.path.join(model_dir, f),
                "size": os.path.getsize(os.path.join(model_dir, f)),
                "last_modified": os.path.getmtime(os.path.join(model_dir, f))
            } 
            for name, f in zip(model_names, model_files)
        ]
        
        return jsonify({"models": models_info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """处理聊天请求的API端点。"""
    data = request.get_json()
    if not data or 'query' not in data:
        app.logger.warning("API请求缺少 'query' 字段。请求体: %s", data)
        return jsonify({"error": "请求体中缺少 'query' 字段"}), 400

    user_query = data.get('query') # 使用 .get() 更安全
    use_existing_model = data.get('use_existing_model', True) # 默认为True，优先使用现有模型
    if not isinstance(user_query, str) or not user_query.strip():
        app.logger.warning(f"API接收到无效查询: '{user_query}' (类型: {type(user_query)})")
        return jsonify({"error": "查询必须是非空字符串"}), 400

    app.logger.info(f"API接收到查询: '{user_query}'")
    ml_keywords = ['机器学习', '模型', '训练', '预测', '回归', '分类', 'ML', '决策树', '随机森林',
                   '线性回归', '逻辑回归', '数据分析', '特征', '权重', '参数', '准确率', 'accuracy',
                   'precision', 'recall']
    ml_ops_keywords = ['训练', '预测', '比较', '评估', '构建', '解释', '自动', '集成', '版本', '分析', '推荐']
    is_ml_query = any(keyword in user_query for keyword in ml_keywords)
    is_ml_ops = any(op in user_query for op in ml_ops_keywords)
    try:
        # 优先处理通用大模型回答模式
        # 前端实际传递的通用大模型模式的 mode 值为 'general_llm'
        if data.get('mode') == 'general_llm': 
            app.logger.info("检测到通用大模型回答模式，直接调用LLM API")
            try:
                direct_llm_response = enhanced_direct_query_llm(user_query)
                return jsonify({
                    "answer": direct_llm_response.get("answer", "未能获取回答。"),
                    "source_documents": direct_llm_response.get("source_documents", []),
                    "is_ml_query": False,
                    "is_direct_answer": True,
                    "model_used": direct_llm_response.get("model_name", "General LLM (Enhanced)")
                })
            except Exception as e_enhanced_llm:
                app.logger.error(f"增强版通用大模型LLM调用失败: {str(e_enhanced_llm)}，尝试标准LLM", exc_info=True)
                try:
                    direct_llm_response = direct_query_llm(user_query)
                    return jsonify({
                        "answer": direct_llm_response.get("answer", "未能获取回答。"),
                        "source_documents": direct_llm_response.get("source_documents", []),
                        "is_ml_query": False,
                        "is_direct_answer": True,
                        "model_used": "General LLM (Standard)"
                    })
                except Exception as e_standard_llm:
                    app.logger.error(f"标准通用大模型LLM调用也失败: {str(e_standard_llm)}", exc_info=True)
                    return jsonify({"error": f"通用大模型处理时出错: {str(e_standard_llm)}"}), 500
        
        # 检查是否为教程生成请求
        elif (data.get('mode') == 'data_analysis' and
            data.get('data_preview') and
            data.get('model_name') and
            data.get('target_column')):
            
            app.logger.info(f"检测到教程生成请求: 模型 '{data.get('model_name')}', 目标列 '{data.get('target_column')}'")
            llm_ml_context = {
                'data_preview': data.get('data_preview'),
                'model_name': data.get('model_name'),
                'target_column': data.get('target_column'),
                'generate_tutorial': True
            }
            
            try:
                # user_query 也传递给LLM，以便它了解用户的原始意图
                direct_llm_response = enhanced_direct_query_llm(user_query, llm_ml_context)
                return jsonify({
                    "answer": direct_llm_response.get("answer", "未能生成教程内容。"),
                    "source_documents": [], 
                    "is_ml_query": True, 
                    "is_tutorial": True, 
                    "ml_model_used": data.get('model_name')
                })
            except Exception as e:
                app.logger.error(f"教程生成LLM调用失败: {str(e)}", exc_info=True)
                return jsonify({"error": f"生成教程时出错: {str(e)}"}), 500
        
        elif is_ml_query and is_ml_ops:
            app.logger.info(f"检测到机器学习操作类查询，将使用增强版ML Agent处理")
            try:
                # 尝试使用增强版ML代理
                result = enhanced_query_ml_agent(user_query, use_existing_model=use_existing_model)
            except Exception as e:
                app.logger.warning(f"增强版ML代理处理失败，回退到标准ML代理: {str(e)}")
                # 回退到标准ML代理
                result = query_ml_agent(user_query, use_existing_model=use_existing_model)
            
            # 返回结果，保留特征分析数据和预测结果
            response_data = {
                "answer": result["answer"],
                "source_documents": [],
                "is_ml_query": True,
                "feature_analysis": result.get("feature_analysis", {}),
                "ml_model_used": result.get("model_used", "未知模型")
            }
            # 如果结果中包含预测，添加到响应中
            if "prediction" in result:
                response_data["prediction"] = result["prediction"]
            return jsonify(response_data)
        else:
            app.logger.info(f"使用增强版RAG系统处理常规/知识类查询")
            try:
                # 尝试使用增强版RAG处理，启用机器学习集成
                result = enhanced_query_rag(user_query, ml_integration=True)
            except Exception as e:
                app.logger.warning(f"增强版RAG处理失败，回退到标准RAG: {str(e)}")
                # 回退到标准RAG
                result = query_rag(user_query)
                
            result["is_ml_query"] = False
            
            # 检查是否需要进行机器学习集成
            if is_ml_query and not is_ml_ops and "预测" in user_query:
                app.logger.info("检测到预测类查询，尝试集成机器学习模型结果")
                try:
                    # 提取可能的预测目标和特征
                    from rag_core_enhanced import extract_prediction_info, find_suitable_model, make_prediction_with_model
                    prediction_target, features = extract_prediction_info(user_query)
                    
                    if prediction_target and features:
                        # 查找适合该预测任务的模型
                        model_name = find_suitable_model(prediction_target)
                        
                        if model_name:
                            # 加载模型并进行预测
                            model_result = make_prediction_with_model(model_name, features)
                            
                            # 将模型预测结果与RAG结果集成
                            result = integrate_ml_with_rag(result, model_name, {
                                "prediction": model_result.get("predictions"),
                                "feature_importance": model_result.get("feature_importance", {}),
                                "model_metrics": model_result.get("metrics", {})
                            })
                except Exception as e:
                    app.logger.warning(f"机器学习集成失败: {str(e)}")
            
            # 如果RAG结果质量不佳，使用增强版LLM
            if is_rag_result_poor(user_query, result):
                app.logger.info("RAG结果质量不佳，切换到直接大模型回答")
                try:
                    # 如果有机器学习相关信息，将其传递给增强版LLM
                    ml_context = None
                    if result.get("ml_enhanced") or result.get("feature_analysis"):
                        ml_context = {
                            "model_name": result.get("ml_model_used", "未知模型"),
                            "prediction": result.get("prediction"),
                            "feature_importance": result.get("feature_analysis", {}).get("feature_importance", {}),
                            "model_metrics": result.get("model_metrics", {})
                        }
                    
                    direct_llm_response = enhanced_direct_query_llm(user_query, ml_context)
                except Exception as e:
                    app.logger.warning(f"增强版LLM处理失败，回退到标准LLM: {str(e)}")
                    direct_llm_response = direct_query_llm(user_query)
                    
                result["answer"] = direct_llm_response["answer"]
                result["is_direct_answer"] = direct_llm_response.get("is_direct_answer", True)
                result["ml_enhanced_llm"] = direct_llm_response.get("ml_enhanced", False)
            
            # 如果结果中包含预测、模型指标或特征重要性，添加到响应中
            if "prediction" in result:
                result["prediction"] = result["prediction"]
            if "model_metrics" in result:
                result["model_metrics"] = result["model_metrics"]
            if "feature_importance" in result:
                result["feature_importance"] = result["feature_importance"]

            return jsonify(result)
    except Exception as e:
        app.logger.error(f"/api/chat 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"服务器内部错误，请稍后重试或联系管理员。"}), 500

@app.route('/api/rebuild_vector_store', methods=['POST'])
def rebuild_vector_store_endpoint():
    """强制重新构建向量数据库的API端点。"""
    app.logger.info("接收到重建向量数据库的请求。")
    try:
        # 启动异步任务重建向量库
        executor.submit(initialize_rag_system, force_recreate_vs=True)
        app.logger.info("向量数据库重建流程已异步启动。")
        return jsonify({"message": "向量数据库重建流程已异步启动，请稍后查询状态。"}), 202
    except Exception as e:
        app.logger.error(f"/api/rebuild_vector_store 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"重建向量数据库失败: {str(e)}"}), 500

@app.route('/api/ml/train', methods=['POST'])
def train_model_endpoint():
    """训练机器学习模型的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    required_fields = ['model_type', 'data_path', 'target_column']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 创建任务ID
        task_id = str(uuid.uuid4())
        
        # 复制请求数据以便异步任务使用
        task_data = data.copy()
        
        # 异步执行模型训练
        def async_train_model(task_id, task_data):
            try:
                from ml_models import train_model

                model_type = task_data['model_type']
                data_path = task_data['data_path']
                target_column = task_data['target_column']
                model_name = task_data.get('model_name')
                categorical_columns = task_data.get('categorical_columns', [])
                numerical_columns = task_data.get('numerical_columns', [])
                model_params = task_data.get('model_params', {})
                test_size = task_data.get('test_size', 0.2)
                
                app.logger.info(f"开始异步训练任务 {task_id}: {model_type} 模型，目标列: {target_column}")
                
                # 如果是Excel文件，转换为CSV以便更好地处理
                if data_path.endswith('.xlsx'):
                    import pandas as pd
                    df = pd.read_excel(data_path)
                    csv_path = data_path.replace('.xlsx', f'_processed_{task_id}.csv')
                    df.to_csv(csv_path, index=False)
                    data_path = csv_path
                
                # 执行训练
                result = train_model(
                    model_type=model_type,
                    data=data_path,
                    target_column=target_column,
                    model_name=model_name,
                    categorical_columns=categorical_columns,
                    numerical_columns=numerical_columns,
                    model_params=model_params,
                    test_size=test_size
                )
                
                # 将临时CSV清理掉
                if data_path.endswith(f'_processed_{task_id}.csv') and os.path.exists(data_path):
                    try:
                        os.remove(data_path)
                    except Exception as e:
                        app.logger.warning(f"清理临时文件 {data_path} 失败: {str(e)}")
                
                app.logger.info(f"异步训练任务 {task_id} 完成: {result.get('model_name')}")
            except Exception as e:
                app.logger.error(f"异步训练任务 {task_id} 失败: {str(e)}", exc_info=True)
        
        # 提交异步任务
        executor.submit(async_train_model, task_id, task_data)
        
        return jsonify({
            "message": f"模型训练任务已异步启动 (ID: {task_id})",
            "task_id": task_id,
            "status": "processing",
            "model_type": data['model_type'],
            "target_column": data['target_column']
        }), 202
    except Exception as e:
        app.logger.error(f"/api/ml/train 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"启动训练任务时发生错误: {str(e)}"}), 500

@app.route('/api/ml/predict', methods=['POST'])
def predict_endpoint():
    """使用机器学习模型进行预测的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    required_fields = ['model_name', 'input_data']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 直接调用ml_models.py中的predict函数
        from ml_models import predict

        model_name = data['model_name']
        input_data = data['input_data']
        target_column = data.get('target_column')

        # 进行预测
        result = predict(
            model_name=model_name,
            input_data=input_data,
            target_column=target_column
        )

        # 格式化结果以便前端显示
        formatted_result = {
            "model_name": result["model_name"],
            "predictions": result["predictions"],
            "input_data": result["input_data"]
        }

        # 如果有评估指标，添加到结果中
        if "accuracy" in result:
            formatted_result["accuracy"] = result["accuracy"]
        if "mse" in result:
            formatted_result["mse"] = result["mse"]
            formatted_result["r2"] = result["r2"]

        return jsonify(formatted_result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/predict 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"进行预测时发生错误: {str(e)}"}), 500

@app.route('/api/ml/models', methods=['GET'])
def list_models_endpoint():
    """列出所有可用机器学习模型的API端点"""
    try:
        # 直接调用ml_models.py中的list_available_models函数
        from ml_models import list_available_models, MODEL_CATEGORIES

        models = list_available_models()

        # 按类别组织模型
        categorized_models = {}
        for category, model_types in MODEL_CATEGORIES.items():
            categorized_models[category] = []
            for model in models:
                if model["type"] in model_types:
                    categorized_models[category].append({
                        "name": model["name"],
                        "type": model["type"],
                        "path": model["path"]
                    })

        return jsonify({
            "models": models,
            "categorized_models": categorized_models,
            "total_count": len(models)
        }), 200
    except Exception as e:
        app.logger.error(f"/api/ml/models 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"列出模型时发生错误: {str(e)}"}), 500

@app.route('/api/ml/upload', methods=['POST'])
def upload_data_endpoint():
    """上传数据文件的API端点"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件部分"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
        
    # 安全检查文件扩展名
    allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        return jsonify({"error": f"不支持的文件类型。仅支持 {', '.join(allowed_extensions)}"}), 400

    try:
        # 确保上传目录存在
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        # 生成安全的文件名 (使用UUID避免文件名冲突)
        safe_filename = f"{str(uuid.uuid4())}{file_ext}"
        file_path = os.path.join(uploads_dir, safe_filename)
        
        # 保存原始文件名与安全文件名的映射关系
        original_filename = file.filename
        
        # 保存文件
        file.save(file_path)
        app.logger.info(f"文件上传成功: {original_filename} -> {file_path}")

        # 读取数据并处理不同格式
        df = None
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
            elif file_ext == '.json':
                df = pd.read_json(file_path, orient='records')
                df = df.fillna('')
        except Exception as e:
            # 清理已上传的文件
            if os.path.exists(file_path):
                os.remove(file_path)
            app.logger.error(f"读取文件 {original_filename} 失败: {str(e)}")
            return jsonify({"error": f"读取文件失败: {str(e)}"}), 400
            
        if df is None or df.empty:
            # 清理已上传的文件
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"error": "文件为空或格式不正确"}), 400

        # 推断每列的数据类型
        column_types = {}
        categorical_columns = []
        numerical_columns = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() < min(10, len(df) // 10):  # 如果唯一值较少，仍视为分类
                    categorical_columns.append(col)
                    column_types[col] = 'categorical'
                else:
                    numerical_columns.append(col)
                    column_types[col] = 'numerical'
            else:
                categorical_columns.append(col)
                column_types[col] = 'categorical'

        # 使用json_compatible_result处理结果，确保没有NaN值
        result = json_compatible_result({
            "message": "文件上传成功",
            "file_path": file_path,
            "original_filename": original_filename,
            "columns": df.columns.tolist(),
            "column_types": column_types,
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            "preview": df.head(5).to_dict('records'),
            "row_count": len(df),
            "column_count": len(df.columns)
        })

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"上传文件时出错: {str(e)}", exc_info=True)
        return jsonify({"error": f"上传文件时出错: {str(e)}"}), 500

def json_compatible_result(data):
    """确保数据可以被JSON序列化"""
    if isinstance(data, dict):
        return {k: json_compatible_result(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_compatible_result(item) for item in data]
    elif isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
        return int(data)
    elif isinstance(data, (np.float64, np.float32, np.float16)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif pd and isinstance(data, pd.Series):
        return json_compatible_result(data.tolist())
    elif pd and isinstance(data, pd.DataFrame):
        return json_compatible_result(data.to_dict(orient='records'))
    elif pd and isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif isinstance(data, np.ndarray):
        return json_compatible_result(data.tolist())
    elif isinstance(data, (datetime.datetime, datetime.date)):
        return data.isoformat()
    return data

@app.route('/api/ml/analyze', methods=['POST'])
def analyze_data_endpoint():
    """分析数据集的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    required_fields = ['data_path']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 读取数据文件
        data_path = data['data_path']
        target_column = data.get('target_column')

        # 读取数据，处理NaN值
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
        elif data_path.endswith('.json'):
            # 支持JSON文件
            df = pd.read_json(data_path)
        else:
            return jsonify({"error": "不支持的文件格式，仅支持CSV、Excel和JSON"}), 400

        # 基本统计信息
        basic_stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                basic_stats[col] = {
                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    "median": float(df[col].median()) if pd.notna(df[col].median()) else None,
                    "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                    "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                    "missing": int(df[col].isna().sum())
                }
            else:
                value_counts = df[col].value_counts().to_dict()
                # 将键转换为字符串，确保JSON序列化不会出错
                value_counts = {str(k): int(v) for k, v in value_counts.items()}
                basic_stats[col] = {
                    "unique_values": int(df[col].nunique()),
                    "missing": int(df[col].isna().sum()),
                    "most_common": json.loads(json.dumps(value_counts)) # Ensure this is serializable
                }

        # 相关性分析（仅对数值列）
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation = None
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr().round(3)
            # Convert NaN/Inf in correlation matrix to None
            # Use json_compatible_result here too for safety
            correlation = json_compatible_result(corr_matrix.to_dict())

        # 目标列分析（如果提供）
        target_analysis = None
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # 回归问题分析
                target_analysis = {
                    "type": "regression",
                    "distribution": {
                        "mean": float(df[target_column].mean()) if pd.notna(df[target_column].mean()) else None,
                        "median": float(df[target_column].median()) if pd.notna(df[target_column].median()) else None,
                        "skewness": float(stats.skew(df[target_column].dropna())) if not np.isnan(stats.skew(df[target_column].dropna())) else None,
                        "kurtosis": float(stats.kurtosis(df[target_column].dropna())) if not np.isnan(stats.kurtosis(df[target_column].dropna())) else None
                    }
                }

                # 计算与目标列的相关性
                if len(numeric_cols) > 1:
                    target_corr = df[numeric_cols].corr()[target_column].drop(target_column).sort_values(ascending=False)
                    # Convert NaN/Inf in target correlation to None
                     # Use json_compatible_result here too for safety
                    target_analysis["correlations"] = json_compatible_result(target_corr.to_dict())
            else:
                # 分类问题分析
                class_distribution = df[target_column].value_counts().to_dict()
                # Ensure keys and values are serializable
                class_distribution = {str(k): (int(v) if pd.notna(v) else None) for k, v in class_distribution.items()}
                target_analysis = {
                    "type": "classification",
                    "class_distribution": class_distribution,
                    "class_count": len(class_distribution)
                }

        # 推荐模型
        recommended_models = []
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                # 回归问题推荐模型：线性回归
                recommended_models = ["linear_regression"]
            elif df[target_column].nunique() > 0 and df[target_column].nunique() < len(df) * 0.5: # 假设分类问题类别数小于总样本数的一半
                # 分类问题推荐模型：逻辑回归、K-近邻、决策树、向量机、朴素贝叶斯
                recommended_models = ["logistic_regression", "knn_classifier", "decision_tree", "svm_classifier", "naive_bayes"]
            else:
                # 如果目标列是其他类型或类别过多，暂不推荐监督模型
                recommended_models = []
        else:
            # 无监督学习推荐模型：K-Means
            recommended_models = ["kmeans"]

        # Prepare the result dictionary
        analysis_result = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "basic_stats": basic_stats,
            "correlation": correlation,
            "target_analysis": target_analysis,
            "recommended_models": recommended_models,
            "message": "数据分析完成"
        }

        # Recursively convert NaN/Inf to None before sending
        return jsonify(json_compatible_result(analysis_result)), 200
    except Exception as e:
        app.logger.error(f"/api/ml/analyze 接口发生错误: {e}", exc_info=True)
        # Ensure error response is also JSON compatible
        return jsonify(json_compatible_result({"error": f"分析数据时发生错误: {str(e)}"})), 500

@app.route('/api/ml/analyze', methods=['GET'])
def analyze_data_get_endpoint():
    """处理GET请求的数据分析，用于模型比较等场景"""
    try:
        file_path = request.args.get('file_path')
        if not file_path:
            return jsonify({"error": "未提供文件路径参数"}), 400
            
        # 确保文件存在
        if not os.path.exists(file_path):
            return jsonify({"error": f"文件不存在: {file_path}"}), 404
            
        # 读取并分析数据
        df, error = load_dataframe(file_path)
        if error:
            return jsonify({"error": f"读取文件失败: {error}"}), 400
            
        # 获取列信息
        columns = df.columns.tolist()
        
        # 简单分析
        result = {
            "columns": columns,
            "row_count": len(df),
            "column_count": len(columns),
            "file_path": file_path
        }
        
        return jsonify(result)
    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"分析数据失败: {str(e)}\n{traceback_str}")
        return jsonify({"error": f"分析数据失败: {str(e)}"}), 500

@app.route('/api/ml/model_versions', methods=['POST'])
def create_model_version_endpoint():
    """创建模型新版本的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400

    required_fields = ['model_name', 'version_info']
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 导入模型版本管理函数
        from ml_api_endpoints import save_model_version

        model_name = data['model_name']
        version_info = data['version_info']

        # 创建模型版本
        result = save_model_version(
            model_name=model_name,
            version_info=version_info
        )

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/model_versions 接口发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"创建模型版本时发生错误: {str(e)}"}), 500

@app.route('/api/ml/model_versions/<model_name>', methods=['GET'])
def get_model_versions_endpoint(model_name):
    """获取模型所有版本的API端点"""
    try:
        # 导入获取模型版本函数
        from ml_api_endpoints import get_model_versions

        # 获取模型版本
        result = get_model_versions(model_name)

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/model_versions/{model_name} 接口发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"获取模型版本时发生错误: {str(e)}"}), 500

@app.route('/api/ml/compare_models', methods=['POST'])
def compare_models_endpoint():
    """比较多个模型性能的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400

    required_fields = ['model_names', 'test_data_path', 'target_column']
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 导入修复后的模型比较函数
        from ml_api_endpoints_fix import compare_models_api

        model_names_raw = data['model_names']
        test_data_path = data['test_data_path']
        target_column = data['target_column']

        model_names = []
        if isinstance(model_names_raw, str):
            model_names = [name.strip() for name in model_names_raw.split(',') if name.strip()]
        elif isinstance(model_names_raw, list):
            #确保列表中的每个元素都是字符串
            model_names = [str(name).strip() for name in model_names_raw if str(name).strip()]
        
        if not model_names or len(model_names) < 2:
            return jsonify({"success": False, "error": "进行模型比较至少需要选择两个模型，并以正确格式提供 (列表或逗号分隔的字符串)。"}), 400

        # 比较模型
        result = compare_models_api(
            model_names=model_names,
            test_data_path=test_data_path,
            target_column=target_column
        )

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/compare_models 接口发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"比较模型时发生错误: {str(e)}"}), 500

@app.route('/api/ml/ensemble', methods=['POST'])
def build_ensemble_model_endpoint():
    """构建集成模型的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400

    required_fields = ['base_models', 'ensemble_type']
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 导入修复后的集成模型构建函数
        from ml_api_endpoints_fix import build_ensemble_model

        base_models = data['base_models']
        ensemble_type = data['ensemble_type']
        save_name = data.get('save_name')

        # 构建集成模型
        result = build_ensemble_model(
            base_models=base_models,
            ensemble_type=ensemble_type,
            save_name=save_name
        )

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result), 201
    except Exception as e:
        app.logger.error(f"/api/ml/ensemble 接口发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"构建集成模型时发生错误: {str(e)}"}), 500

@app.route('/api/ml/deploy', methods=['POST'])
def deploy_model_endpoint():
    """部署模型的API端点 (后端生成端点)"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400

    required_fields = ['model_name', 'environment']
    for field in required_fields:
        if field not in data:
            return jsonify({"success": False, "error": f"缺少必要字段 '{field}'"}), 400

    try:
        from ml_api_endpoints import deploy_model

        model_name = data['model_name']
        environment = data['environment']
        
        # 后端生成唯一的端点路径
        model_name_slug = model_name.lower().replace(' ', '-').replace('_', '-')
        unique_id = str(uuid.uuid4())[:8]
        generated_endpoint = f"/api/predict/{model_name_slug}/{unique_id}"
        
        app.logger.info(f"为模型 '{model_name}' 在环境 '{environment}' 生成的部署端点: {generated_endpoint}")

        # 部署模型，传递生成的端点
        result = deploy_model(
            model_name=model_name,
            environment=environment,
            endpoint=generated_endpoint
        )

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result), 201
    except Exception as e:
        app.logger.error(f"/api/ml/deploy 接口发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"部署模型时发生错误: {str(e)}"}), 500

@app.route('/api/ml/deployments', methods=['GET'])
def get_deployed_models_endpoint():
    """获取已部署模型列表的API端点"""
    try:
        # 导入获取已部署模型列表函数
        from ml_api_endpoints import get_deployed_models

        # 获取已部署模型列表
        result = get_deployed_models()

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/deployments 接口发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"获取已部署模型列表时发生错误: {str(e)}"}), 500

@app.route('/api/ml/undeploy/<deployment_id>', methods=['POST'])
def undeploy_model_endpoint(deployment_id):
    """取消部署模型的API端点"""
    try:
        # 导入取消部署模型函数
        from ml_api_endpoints import undeploy_model

        # 取消部署模型
        result = undeploy_model(deployment_id)

        if not result.get("success", False):
            return jsonify(result), 400

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/undeploy/{deployment_id} 接口发生错误: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"取消部署模型时发生错误: {str(e)}"}), 500

@app.route('/api/ml/explain', methods=['POST'])
def explain_model_endpoint():
    """解释模型预测的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    required_fields = ['model_name', 'data_path', 'target_column']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 导入必要的库
        import matplotlib.pyplot as plt
        import io
        import base64
        from sklearn.inspection import permutation_importance
        import pickle
        import os
        import shap

        model_name = data['model_name']
        data_path = data['data_path']
        target_column = data['target_column']

        # 加载模型
        model_path = os.path.join("ml_models", f"{model_name}.pkl")
        if not os.path.exists(model_path):
            return jsonify({"error": f"模型 {model_name} 不存在"}), 404

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # 加载数据
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(data_path)
            else:
                return jsonify({"error": "不支持的文件格式，仅支持CSV和Excel"}), 400
        except Exception as e:
            app.logger.error(f"加载数据文件时出错: {e}")
            return jsonify({"error": f"加载数据文件时出错: {str(e)}"}), 500

        # 准备特征和目标
        if target_column not in df.columns:
            return jsonify({"error": f"目标列 {target_column} 不在数据集中"}), 400

        # 处理缺失值
        df = df.dropna()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 获取模型类型
        model_type = type(model).__name__

        # 特征重要性
        feature_importance = {}
        feature_importance_plot = ""
        shap_plot = ""
        model_params = {}

        # 获取模型参数
        try:
            model_params = model.get_params()
        except Exception as e:
            app.logger.warning(f"获取模型参数时出错: {e}")
            model_params = {"error": "无法获取模型参数"}

        if hasattr(model, 'feature_importances_'):
            # 对于随机森林、决策树等有feature_importances_属性的模型
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            feature_names = X.columns

            for i, idx in enumerate(indices):
                feature_importance[feature_names[idx]] = float(importances[idx])

            # 创建特征重要性图
            plt.figure(figsize=(10, 6))
            plt.title("特征重要性")
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()

            # 将图像转换为base64字符串
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            feature_importance_plot = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

            # 尝试生成SHAP值解释
            try:
                 # 对于树模型，使用TreeExplainer
                 if hasattr(model, 'estimators_') or 'Tree' in model_type:
                     explainer = shap.TreeExplainer(model)
                     shap_values = explainer.shap_values(X.iloc[:100])  # 使用前100个样本以提高性能

                     plt.figure(figsize=(12, 8))
                     if isinstance(shap_values, list):
                         # 分类模型可能返回每个类别的SHAP值列表
                         shap.summary_plot(shap_values[0], X.iloc[:100], show=False)
                     else:
                         # 回归模型返回单个SHAP值数组
                         shap.summary_plot(shap_values, X.iloc[:100], show=False)

                     buf = io.BytesIO()
                     plt.savefig(buf, format='png')
                     buf.seek(0)
                     shap_plot = base64.b64encode(buf.read()).decode('utf-8')
                     plt.close()
            except Exception as e:
                 app.logger.warning(f"生成SHAP解释时出错: {e}")
                 # 错误不会中断流程，只是没有SHAP图
            plt.tight_layout()

            # 将图转换为base64字符串
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            feature_importance_plot = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        elif hasattr(model, 'coef_'):
            # 对于线性模型
            coefficients = model.coef_
            if len(coefficients.shape) == 1:
                # 单目标回归或二分类
                feature_names = X.columns
                for i, name in enumerate(feature_names):
                    feature_importance[name] = float(abs(coefficients[i]))

                # 创建系数图
                plt.figure(figsize=(10, 6))
                plt.title("特征系数")
                plt.bar(feature_names, coefficients)
                plt.xticks(rotation=90)
                plt.tight_layout()

                # 将图转换为base64字符串
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                feature_importance_plot = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
            else:
                # 多分类
                feature_names = X.columns
                avg_importance = np.mean(np.abs(coefficients), axis=0)
                for i, name in enumerate(feature_names):
                    feature_importance[name] = float(avg_importance[i])

                # 创建平均系数图
                plt.figure(figsize=(10, 6))
                plt.title("平均特征系数 (绝对值)")
                plt.bar(feature_names, avg_importance)
                plt.xticks(rotation=90)
                plt.tight_layout()

                # 将图转换为base64字符串
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                feature_importance_plot = base64.b64encode(buf.read()).decode('utf-8')
                plt.close()
        else:
            # 对于其他模型，使用排列重要性
            perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            feature_names = X.columns
            for i, name in enumerate(feature_names):
                feature_importance[name] = float(perm_importance.importances_mean[i])

            # 创建排列重要性图
            plt.figure(figsize=(10, 6))
            plt.title("排列特征重要性")
            sorted_idx = perm_importance.importances_mean.argsort()[::-1]
            plt.bar(range(X.shape[1]), perm_importance.importances_mean[sorted_idx])
            plt.xticks(range(X.shape[1]), [feature_names[i] for i in sorted_idx], rotation=90)
            plt.tight_layout()

            # 将图转换为base64字符串
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            feature_importance_plot = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()

        # 模型参数
        model_params = {}
        for param, value in model.get_params().items():
            # 确保值是JSON可序列化的
            if isinstance(value, (int, float, str, bool, type(None))):
                model_params[param] = value
            else:
                model_params[param] = str(value)

        # 准备模型解释结果
        result = {
            "model_name": model_name,
            "model_type": model_type,
            "feature_importance": feature_importance,
            "feature_importance_plot": feature_importance_plot,
            "model_params": model_params,
            "data_shape": {"rows": X.shape[0], "columns": X.shape[1]},
            "column_names": X.columns.tolist(),
            "message": f"成功解释{model_type}模型"
        }

        # 如果有SHAP图，添加到结果中
        if shap_plot:
            result["shap_plot"] = shap_plot
            result["has_shap_explanation"] = True
        else:
            result["has_shap_explanation"] = False

        # 添加模型特定的解释信息
        if hasattr(model, 'classes_'):
            result["classes"] = model.classes_.tolist() if hasattr(model.classes_, 'tolist') else [str(c) for c in model.classes_]
            result["problem_type"] = "classification"
        else:
            result["problem_type"] = "regression"

        return jsonify(result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/explain 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"解释模型时发生错误: {str(e)}"}), 500

@app.route('/api/ml/auto_select', methods=['POST'])
def auto_model_selection_endpoint():
    """自动选择最佳模型的API端点"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求体为空"}), 400

    required_fields = ['data_path', 'target_column']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"缺少必要字段 '{field}'"}), 400

    try:
        # 直接调用ml_models.py中的auto_model_selection函数
        from ml_models import auto_model_selection

        data_path = data['data_path']
        target_column = data['target_column']
        categorical_columns = data.get('categorical_columns', [])
        numerical_columns = data.get('numerical_columns', [])
        cv = data.get('cv', 5)
        metric = data.get('metric', 'auto')
        models_to_try = data.get('models_to_try')

        # 自动选择最佳模型
        result = auto_model_selection(
            data_path=data_path,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            cv=cv,
            metric=metric,
            models_to_try=models_to_try
        )

        # 格式化结果以便前端显示
        formatted_result = {
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "params": result["params"],
            "cv_score": result["cv_score"],
            "is_classification": result["is_classification"],
            "all_models_results": result["all_models_results"],
            "message": f"成功选择最佳模型: {result['model_type']}，模型名称为{result['model_name']}"
        }

        return jsonify(formatted_result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/auto_select 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"自动选择模型时发生错误: {str(e)}"}), 500

def check_config_and_kb():
    """检查基本配置和知识库目录。"""
    config_valid = True

    # 检查API密钥配置
    if not AI_STUDIO_API_KEY:
        app.logger.error("错误：AI_STUDIO_API_KEY 未在 .env 文件或环境变量中配置。")
        config_valid = False

    # 检查知识库目录
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        app.logger.warning(f"警告：知识库目录 '{KNOWLEDGE_BASE_DIR}' 不存在。")
        try:
            os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True) # exist_ok=True 避免目录已存在时报错
            app.logger.info(f"已自动创建知识库目录: {KNOWLEDGE_BASE_DIR}。请将您的文档放入此目录。")
        except OSError as e:
            app.logger.error(f"无法创建知识库目录 {KNOWLEDGE_BASE_DIR}: {e}。请手动创建。")
            config_valid = False
    else:
        # 检查目录是否为空
        current_files = os.listdir(KNOWLEDGE_BASE_DIR)
        if not current_files:
            app.logger.warning(f"警告：知识库目录 '{KNOWLEDGE_BASE_DIR}' 为空。RAG系统将没有可查询的数据源。")

    # 检查上传目录
    uploads_dir = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(uploads_dir):
        app.logger.warning(f"警告：上传目录 '{uploads_dir}' 不存在。")
        try:
            os.makedirs(uploads_dir, exist_ok=True)
            app.logger.info(f"已自动创建上传目录: {uploads_dir}。")
        except OSError as e:
            app.logger.error(f"无法创建上传目录 {uploads_dir}: {e}。请手动创建。")
            config_valid = False

    # 检查模型目录
    models_dir = os.path.join(os.getcwd(), 'ml_models')
    if not os.path.exists(models_dir):
        app.logger.warning(f"警告：模型目录 '{models_dir}' 不存在。")
        try:
            os.makedirs(models_dir, exist_ok=True)
            app.logger.info(f"已自动创建模型目录: {models_dir}。")
        except OSError as e:
            app.logger.error(f"无法创建模型目录 {models_dir}: {e}。请手动创建。")
            config_valid = False

    return config_valid

# 配置静态文件路径，使前端能够正确加载JavaScript文件
# Functions is_rag_result_poor and get_direct_llm_answer have been moved to the top of the file.
@app.route('/templates/<path:filename>')
def serve_template_file(filename):
    """提供模板目录中的静态文件"""
    try:
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        return send_from_directory(template_dir, filename)
    except Exception as e:
        app.logger.error(f"加载模板文件 {filename} 时出错: {e}")
        return f"无法加载文件 {filename}", 404

@app.route('/static/<path:filename>')
def serve_static_file(filename):
    """提供静态文件目录中的文件"""
    try:
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        return send_from_directory(static_dir, filename)
    except Exception as e:
        app.logger.error(f"加载静态文件 {filename} 时出错: {e}")
        return f"无法加载文件 {filename}", 404

# 初始化应用程序
def init_app():
    """初始化应用程序，包括配置检查和RAG系统初始化"""
    if not check_config_and_kb():
        app.logger.critical("配置检查未通过，请修复上述问题后重试。程序即将退出。")
        return False

    app.logger.info("正在初始化RAG系统 (百度文心版)...")
    try:
        # 首次运行时，force_recreate_vs=False。如果chroma_db不存在，会自动创建。
        # 如果希望每次启动都强制重建（比如知识库文件经常变动），可以设为True，但会很慢。
        initialize_rag_system(force_recreate_vs=False)
        app.logger.info("RAG系统初始化完成。")
        return True
    except Exception as e:
        app.logger.critical(f"RAG系统初始化过程中发生严重错误: {e}。", exc_info=True)
        return False

# 主函数，启动Flask应用
if __name__ == '__main__':
    try:
        # 检查依赖项
        try:
            import langchain_community
            import langchain
            import chromadb
        except ImportError as e:
            app.logger.critical(f"缺少必要的依赖项: {e}。请运行 'pip install -r requirements.txt' 安装所有依赖。")
            print(f"\n错误: 缺少必要的依赖项: {e}")
            print("请运行 'pip install -r requirements.txt' 安装所有依赖。\n")
            sys.exit(1)

        # 初始化应用
        if not init_app():
            app.logger.critical("应用初始化失败，程序即将退出。")
            print("\n错误: 应用初始化失败，请检查日志获取详细信息。\n")
            sys.exit(1)

        # 确保静态文件目录存在
        static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir, exist_ok=True)
            app.logger.info(f"已创建静态文件目录: {static_dir}")

        # 确保templates目录存在
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        if not os.path.exists(templates_dir):
            app.logger.warning(f"模板目录不存在: {templates_dir}，将创建该目录")
            os.makedirs(templates_dir, exist_ok=True)
            app.logger.info(f"已创建模板目录: {templates_dir}")

        # 启动服务器
        app.logger.info(f"Flask服务器正在启动，请访问 http://localhost:5000 或 http://127.0.0.1:5000")
        print(f"\n服务器启动成功! 请访问: http://localhost:5000\n")
        # debug=True 用于开发，它会自动重载代码并提供调试器。生产环境应设为 False。
        # use_reloader=False 可以防止Flask在debug模式下启动两次（一次主进程，一次重载进程），
        # 这对于避免 initialize_rag_system 被执行两次可能有用。
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except Exception as e:
        app.logger.critical(f"启动服务器时发生未预期的错误: {e}", exc_info=True)
        print(f"\n错误: 启动服务器时发生未预期的错误: {e}\n")
        sys.exit(1)