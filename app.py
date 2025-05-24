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
import re
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# 导入配置和核心功能模块
from config import (
    KNOWLEDGE_BASE_DIR, AI_STUDIO_API_KEY,
    ML_KEYWORDS as APP_ML_KEYWORDS, # 使用别名避免与局部变量冲突
    ML_OPS_KEYWORDS as APP_ML_OPS_KEYWORDS,
    UNCERTAINTY_PHRASES, RAG_SCORE_THRESHOLD, RAG_ANSWER_MIN_LENGTH
)
from rag_core import query_rag, initialize_rag_system, direct_query_llm
from ml_agents import query_ml_agent
import logging
logger = logging.getLogger(__name__)
# 导入增强版RAG和ML集成功能
from rag_core_enhanced import enhanced_query_rag, enhanced_direct_query_llm
from ml_agents_enhanced import enhanced_query_ml_agent
from advanced_feature_analysis import integrate_ml_with_rag
from werkzeug.utils import secure_filename # 新增导入
# Helper functions moved to the top
def extract_and_parse_json_from_llm(llm_response_str: str, endpoint_name: str = "LLM_JSON_Parser") -> tuple[
    Optional[dict], Optional[str]]:
    """
    从LLM的响应字符串中提取并解析JSON。

    Args:
        llm_response_str: LLM返回的原始字符串。
        endpoint_name: 调用此函数的端点名称，用于日志记录。

    Returns:
        A tuple (parsed_json, error_message).
        If successful, parsed_json is the dict, error_message is None.
        If failed, parsed_json is None, error_message contains the error.
    """
    if not llm_response_str:
        logger.warning(f"[{endpoint_name}] LLM响应为空字符串。")
        return None, "LLM响应为空。"

    logger.debug(f"[{endpoint_name}] 原始LLM响应 (前500字符): {llm_response_str[:500]}")

    extracted_json_str = None

    # 1. 尝试匹配Markdown代码块 ```json ... ```
    match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", llm_response_str, re.DOTALL)
    if match:
        extracted_json_str = match.group(1)
        logger.debug(f"[{endpoint_name}] 从Markdown代码块中提取到JSON字符串。")
    else:
        # 2. 如果没有Markdown块，尝试查找最外层的 '{' 和 '}'
        #    这需要更小心，因为LLM的文本中可能包含其他花括号
        #    一个稍微健壮一点的方法是找到第一个 '{' 和最后一个 '}'
        #    但这仍然不完美，如果JSON本身被包裹在更多文本中且文本中也有花括号
        json_start = llm_response_str.find('{')
        json_end = llm_response_str.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            # 尝试验证括号是否匹配，这比较复杂，这里简化处理
            # 我们可以先假设这个提取是初步的
            potential_json = llm_response_str[json_start: json_end + 1]
            # 尝试直接解析这个初步提取的部分
            try:
                json.loads(potential_json)  # 尝试解析，如果成功，就用它
                extracted_json_str = potential_json
                logger.debug(f"[{endpoint_name}] 通过查找首尾花括号提取到潜在JSON字符串。")
            except json.JSONDecodeError:
                # 如果初步提取的无法解析，回退到使用整个字符串，寄希望于它本身就是JSON
                logger.warning(
                    f"[{endpoint_name}] 初步提取的JSON '{potential_json[:100]}...' 无法解析，将尝试解析整个LLM响应。")
                extracted_json_str = llm_response_str  # Fallback
        else:
            # 如果连首尾花括号都找不到，直接用原始字符串尝试
            extracted_json_str = llm_response_str
            logger.debug(f"[{endpoint_name}] 未找到明确的JSON结构标记，将尝试解析整个LLM响应。")

    if not extracted_json_str:  # 应该不会到这里，因为上面总会给 extracted_json_str 赋值
        logger.error(f"[{endpoint_name}] 无法提取任何JSON候选字符串。")
        return None, "无法从LLM响应中提取JSON内容。"

    try:
        parsed_json = json.loads(extracted_json_str)
        logger.info(f"[{endpoint_name}] 成功解析JSON。")
        return parsed_json, None
    except json.JSONDecodeError as e:
        err_msg = f"LLM返回的内容无法解析为有效的JSON。错误: {e}. 内容 (前500字符): {extracted_json_str[:500]}"
        logger.error(f"[{endpoint_name}] {err_msg}")
        # 注意：在实际返回给前端的错误信息中，可能不需要包含具体的解码错误 e，以免泄露过多细节
        return None, "大模型未能返回有效的JSON格式。请检查Prompt或重试。"  # 返回给前端的通用错误
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
UPLOADS_DIR = os.path.join(os.getcwd(), 'uploads')
MODELS_DIR = os.path.join(os.getcwd(), 'ml_models') # 如果您有模型存储目录
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'} # 定义允许的文件扩展名
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)  # Flask会自动查找同级的 'templates' 文件夹
CORS(app)
app.config['UPLOADS_DIR'] = UPLOADS_DIR
app.config['MODELS_DIR'] = MODELS_DIR
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

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
    data = request.get_json()
    if not data or 'query' not in data:
        app.logger.warning("API请求缺少 'query' 字段。请求体: %s", data)
        return jsonify({"error": "请求体中缺少 'query' 字段"}), 400

    user_query = data.get('query', '').strip()
    mode = data.get('mode')  # 'data_analysis' 或 'general_llm'

    # 从请求中获取前端可能传递的上下文信息
    data_preview = data.get('data_preview')  # 前端传来的数据预览 (list of dicts)
    target_column = data.get('target_column')
    selected_model_name = data.get('model_name')  # 前端选择的模型
    data_path = data.get('data_path') # 如果需要完整数据路径

    if not user_query:
        app.logger.warning("API接收到空查询")
        return jsonify({"error": "查询不能为空"}), 400

    app.logger.info(f"API接收到查询: '{user_query[:100]}...', 模式: {mode}")

    try:
        # 功能 4: 用户使用通用大模型模式
        if mode == 'general_llm':
            app.logger.info("处理通用大模型模式查询")
            # 直接调用LLM，Prompt可以简单包装或直接使用用户查询
            prompt = f"请回答以下问题：\n{user_query}"
            llm_response = enhanced_direct_query_llm(prompt)  # 假设它返回 {"answer": "...", ...}
            return jsonify({
                "answer": llm_response.get("answer", "未能获取回答。"),
                "source_documents": llm_response.get("source_documents", []),  # RAG核心的LLM也可能返回空
                "is_ml_query": False,  # 通常通用查询不是特定ML操作
                "is_direct_answer": True,
                "model_used": llm_response.get("model_name", "General LLM")
            }), 200

        # --- 数据分析模式 (mode == 'data_analysis') ---
        # 功能 3: 用户上传数据集和选择目标列和模型,然后为其生成具体的包含代码的教程
        # 我们通过一个关键词或前端特定标记来识别教程生成请求
        # 假设前端会在query中包含“生成教程”或通过一个额外参数标记
        is_tutorial_request = ("教程" in user_query.lower() or "generate_tutorial" in data) and \
                              data_preview and selected_model_name and target_column

        if is_tutorial_request:
            app.logger.info(f"处理教程生成请求: 模型 '{selected_model_name}', 目标列 '{target_column}'")
            prompt_parts = [
                f"请为以下场景生成一个详细的Python机器学习教程，使用语言为中文：",
                f"用户原始问题（供参考，主要按以下要求生成教程）: {user_query}",
                f"要使用的模型: {selected_model_name}",
                f"目标预测列: {target_column}",
                "教程应包含清晰的步骤、Python代码示例和必要的解释。代码应尽可能通用，并使用常见的库如 pandas 和 scikit-learn。",
                "步骤应包括（但不限于）："
                "  1. 简介和目标说明。",
                "  2. 数据加载与探索性数据分析 (EDA)：假设用户已有一个Pandas DataFrame，其列名和数据类型可参考以下数据预览。",
                "  3. 数据预处理：根据数据预览和常见场景，讨论可能需要的预处理步骤（如处理缺失值、分类特征编码、数值特征缩放）。请提供代码片段作为示例。",
                "  4. 特征工程（如果适用）。",
                "  5. 将数据集拆分为特征 (X) 和目标 (y)，然后划分为训练集和测试集。",
                f"  6. 模型初始化与训练：实例化 '{selected_model_name}' 模型并进行训练。",
                "  7. 使用模型进行预测。",
                "  8. 模型评估：选择适合该模型和任务（分类/回归）的评估指标，并解释如何解读它们。",
                "  9. 总结和后续步骤建议。",
                "数据预览（前几行，用于理解数据结构和生成相关代码示例）：",
                json.dumps(data_preview, indent=2, ensure_ascii=False),
                "\n请确保代码块使用Markdown格式正确标识。"
            ]
            tutorial_prompt = "\n".join(prompt_parts)
            app.logger.debug(f"教程生成Prompt (部分): {tutorial_prompt[:500]}...")

            llm_response = enhanced_direct_query_llm(tutorial_prompt)
            return jsonify({
                "answer": llm_response.get("answer", "未能生成教程内容。"),
                "source_documents": [],  # 教程通常不依赖RAG源文档
                "is_ml_query": True,
                "is_tutorial": True,
                "ml_model_used": selected_model_name,
                "target_column_for_tutorial": target_column
            }), 200

        # 功能 1 & 2: 用户基于上传的数据进行提问
        if data_preview:  # 必须有数据预览才能进行这类分析
            app.logger.info("处理基于数据的分析查询")
            prompt_parts = [f"用户问题: {user_query}\n"]
            prompt_parts.append("请根据以下提供的数据预览信息来回答用户的问题。")
            prompt_parts.append("数据预览 (前几行):")
            prompt_parts.append(json.dumps(data_preview, indent=2, ensure_ascii=False) + "\n")

            if target_column:
                prompt_parts.append(f"用户已指定的目标列 (用于预测或分析相关性): '{target_column}'\n")
            if selected_model_name:
                prompt_parts.append(f"用户当前选择或提及的模型是: '{selected_model_name}'\n")

            # 根据问题类型调整指示
            if "适合用什么模型进行分析" in user_query or "推荐模型" in user_query:
                prompt_parts.append(
                    "请基于数据的特点（如列名、数据类型暗示、值的分布等）推荐合适的机器学习模型，并解释推荐理由。")
            elif "哪些特征最重要" in user_query and target_column:
                prompt_parts.append(
                    f"请分析对于预测目标列 '{target_column}'，数据预览中的哪些其他列（特征）可能最重要，并说明判断依据。")
            else:
                prompt_parts.append("请针对用户的问题，结合数据预览给出专业的分析和回答。")

            data_context_prompt = "".join(prompt_parts)
            app.logger.debug(f"数据上下文Prompt (部分): {data_context_prompt[:500]}...")

            llm_response = enhanced_direct_query_llm(data_context_prompt)
            return jsonify({
                "answer": llm_response.get("answer", "未能基于数据分析回答。"),
                "source_documents": [],
                "is_ml_query": True,  # 假设这类问题与ML相关
                "data_context_used": True,
                "model_used": llm_response.get("model_name", "Contextual LLM")
            }), 200

        # 如果是数据分析模式，但没有足够上下文（如数据预览），则可能无法很好回答
        # 可以尝试RAG或通用ML Agent，或者提示用户上传数据
        app.logger.warning(f"数据分析模式查询 '{user_query}'，但缺少足够的数据上下文 (如data_preview)。")
        # 尝试使用ML Agent (如果问题看起来是操作性的)
        is_ml_query = any(keyword.lower() in user_query.lower() for keyword in APP_ML_KEYWORDS)
        is_ml_ops = any(op.lower() in user_query.lower() for op in APP_ML_OPS_KEYWORDS)
        if is_ml_query and is_ml_ops and enhanced_query_ml_agent:
            try:
                app.logger.info("尝试使用ML Agent处理（无数据上下文的操作类查询）")
                # 注意：这里的ML Agent可能无法执行需要数据的操作
                agent_result = enhanced_query_ml_agent(user_query,
                                                       use_existing_model=data.get('use_existing_model', True))
                return jsonify({
                    "answer": agent_result.get("answer", "ML Agent未能处理此请求。"),
                    "is_ml_query": True,
                    "ml_model_used": agent_result.get("model_used", "ML Agent")
                    # 其他 agent_result 中的字段，如 feature_analysis, prediction 等
                }), 200
            except Exception as e_agent:
                app.logger.error(f"ML Agent处理失败: {str(e_agent)}", exc_info=True)
                # 继续执行后续的RAG/LLM兜底

        # 默认兜底：尝试RAG，如果RAG不好再用纯LLM (主要用于知识性问答)
        app.logger.info(f"数据分析模式查询 '{user_query}' 无特定处理路径，尝试RAG")
        rag_response = enhanced_query_rag(user_query)
        if not is_rag_result_poor(user_query, rag_response):  # 使用您已有的 is_rag_result_poor
            app.logger.info("RAG结果尚可")
            return jsonify(rag_response), 200
        else:
            app.logger.info("RAG结果不佳，转通用LLM处理")
            prompt = f"请回答以下问题：\n{user_query}"
            llm_response = enhanced_direct_query_llm(prompt)
            return jsonify({
                "answer": llm_response.get("answer", "未能获取回答。"),
                "source_documents": llm_response.get("source_documents", []),
                "is_direct_answer": True,
                "model_used": llm_response.get("model_name", "Fallback LLM")
            }), 200

    except Exception as e:
        app.logger.error(f"/api/chat 接口发生错误: {str(e)}", exc_info=True)
        return jsonify({"error": f"服务器内部错误，请稍后重试或联系管理员。"}), 500

@app.route('/api/rebuild_vector_store', methods=['POST'])
def rebuild_vector_store_endpoint():
    """强制重新构建向量数据库的API端点。"""
    app.logger.info("接收到重建向量数据库的请求。")
    try:
        # 确保这个函数调用不会阻塞太久，或者考虑异步处理
        initialize_rag_system(force_recreate_vs=True)
        app.logger.info("向量数据库重建流程已完成。")
        return jsonify({"message": "向量数据库重建流程已启动并完成。"}), 200
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
        # 直接调用ml_models.py中的train_model函数
        from ml_models import train_model

        model_type = data['model_type']
        data_path = data['data_path']
        target_column = data['target_column']
        model_name = data.get('model_name')
        categorical_columns = data.get('categorical_columns', [])
        numerical_columns = data.get('numerical_columns', [])
        model_params = data.get('model_params', {})
        test_size = data.get('test_size', 0.2)

        # 如果是Excel文件，转换为CSV以便更好地处理
        if data_path.endswith('.xlsx'):
            import pandas as pd
            df = pd.read_excel(data_path)
            csv_path = data_path.replace('.xlsx', '_processed.csv')
            df.to_csv(csv_path, index=False)
            data_path = csv_path
            
        # 训练模型
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

        # 格式化结果以便前端显示
        formatted_result = {
            "model_name": result["model_name"],
            "model_type": result["model_type"],
            "metrics": result["metrics"],
            "message": f"成功训练{model_type}模型，模型名称为{result['model_name']}"
        }

        return jsonify(formatted_result), 200
    except Exception as e:
        app.logger.error(f"/api/ml/train 接口发生错误: {e}", exc_info=True)
        return jsonify({"error": f"训练模型时发生错误: {str(e)}"}), 500


# 在 app.py 的路由部分添加以下新端点

# 功能 5.2: 模型比较 (模拟)
@app.route('/api/simulate_model_comparison', methods=['POST'])
def simulate_model_comparison_endpoint():
    data = request.get_json()
    if not data or not all(k in data for k in ['model_names', 'test_data_identifier', 'target_column']):
        app.logger.warning(f"模拟模型比较请求缺少参数: {data}")
        return jsonify({"error": "缺少必要参数 (model_names, test_data_identifier, target_column)"}), 400

    model_names_list = data['model_names']
    if not isinstance(model_names_list, list) or len(model_names_list) < 2:
        return jsonify({"error": "model_names 必须是至少包含两个模型的列表"}), 400

    prompt_parts = [
        "请模拟以下机器学习模型的比较过程，并以中文回答：",
        f"模型列表：{', '.join(model_names_list)}",
        f"测试数据集标识：'{data['test_data_identifier']}' (例如：'当前上传的关于客户流失预测的数据' 或 '公开的鸢尾花分类数据集')",
        f"目标列：'{data['target_column']}'\n",
        "要求：",
        "1. 为列表中的每个模型生成一组合理的、符合其典型应用场景的模拟评估指标。",
        "   - 如果目标列暗示分类任务（例如，目标列是字符型或少数唯一值），请使用分类指标如：准确率 (Accuracy), 精确率 (Precision), 召回率 (Recall), F1分数 (F1-score)。数值请在0.6到0.98之间随机模拟。",
        "   - 如果目标列暗示回归任务（例如，目标列是连续数值型），请使用回归指标如：R²分数 (R2 Score), 均方误差 (MSE), 平均绝对误差 (MAE)。R2分数在0.5到0.95之间，MSE/MAE根据常见场景模拟。",
        "2. 对模拟结果进行简短总结，指出哪个模型在模拟中可能表现更优，并给出推测的理由。",
        "3. 以严格的JSON格式返回结果，结构如下：",
        """
{
  "comparison_results": [
    {
      "model_name": "模型A的名称", 
      "metrics": {"指标1": "模拟值1", "指标2": "模拟值2"}
    },
    {
      "model_name": "模型B的名称", 
      "metrics": {"指标1": "模拟值1", "指标2": "模拟值2"}
    }
    // ... 更多模型
  ],
  "summary": "这里是模拟的总结文本...",
  "test_data_info": { 
      "identifier": "{data['test_data_identifier']}", 
      "simulated_rows": "例如：约1000行", 
      "simulated_features": "例如：约10个特征"
  }
}
""",
        "请确保'metrics'对象中的值是数值类型（如果适用，如准确率）或字符串。请务必确保您的整个输出就是一个单一的、完整且严格符合上述结构的JSON对象。不要在JSON对象之前或之后包含任何其他文本、注释、解释或Markdown的```json ```标记，直接输出JSON本身。"
    ]
    comparison_prompt = "\n".join(prompt_parts)
    app.logger.info(f"模拟模型比较Prompt (部分): {comparison_prompt[:300]}...")

    try:
        # enhanced_direct_query_llm 应该返回一个字典，其中 "answer" 键包含LLM的原始文本输出
        llm_raw_output_dict = enhanced_direct_query_llm(comparison_prompt)
        if not llm_raw_output_dict or "answer" not in llm_raw_output_dict:
            app.logger.error("enhanced_direct_query_llm 未返回预期的包含 'answer' 的字典。")
            return jsonify({"error": "调用大模型时发生内部错误 (无响应)。"}), 500

        llm_response_str = llm_raw_output_dict.get("answer", "")

        simulated_results, error_msg = extract_and_parse_json_from_llm(llm_response_str, "ModelComparison")

        if error_msg:  # 如果解析失败
            return jsonify({
                "error": error_msg,  # 使用辅助函数返回的错误信息
                "raw_llm_response": llm_response_str  # 仍然返回原始响应
            }), 500

        # 基本的结构验证 (simulated_results 不为 None)
        if not isinstance(simulated_results, dict) or \
                "comparison_results" not in simulated_results or \
                not isinstance(simulated_results["comparison_results"], list) or \
                "summary" not in simulated_results:
            app.logger.error(f"LLM返回的模拟比较结果JSON结构不符合预期: {simulated_results}")
            return jsonify({
                "error": "大模型返回的模拟比较结果JSON结构不正确。",
                "parsed_response": simulated_results,  # 返回已解析（但结构错误）的内容
                "raw_llm_response": llm_response_str
            }), 500

        # 可以添加更细致的结构验证，例如检查 comparison_results 列表中的元素
        for item in simulated_results["comparison_results"]:
            if not isinstance(item, dict) or "model_name" not in item or "metrics" not in item:
                app.logger.error(f"模拟比较结果中 'comparison_results' 列表内元素结构错误: {item}")
                return jsonify({
                    "error": "模拟比较结果内部数据结构不正确。",
                    "parsed_response": simulated_results,
                    "raw_llm_response": llm_response_str
                }), 500

        return jsonify(simulated_results), 200
    except Exception as e:
        app.logger.error(f"模拟模型比较过程中发生错误: {str(e)}", exc_info=True)
        return jsonify({"error": f"模拟模型比较时发生内部错误: {str(e)}"}), 500


# 功能 5.3: 集成模型构建 (模拟)
@app.route('/api/simulate_ensemble_building', methods=['POST'])
def simulate_ensemble_building_endpoint():
    data = request.get_json()
    if not data or not all(k in data for k in ['base_models', 'ensemble_type', 'ensemble_name']):
        app.logger.warning(f"模拟集成构建请求缺少参数: {data}")
        return jsonify({"error": "缺少必要参数 (base_models, ensemble_type, ensemble_name)"}), 400

    base_models_list = data['base_models']
    ensemble_type = data['ensemble_type']
    ensemble_name = data['ensemble_name']

    if not isinstance(base_models_list, list) or len(base_models_list) < 2:
        return jsonify({"error": "base_models 必须是至少包含两个模型的列表"}), 400
    if not ensemble_name.strip():
        return jsonify({"error": "ensemble_name 不能为空"}), 400

    prompt_parts = [
        f"请模拟构建一个名为 '{ensemble_name}' 的集成学习模型，并以中文回答：",
        f"基础模型列表：{', '.join(base_models_list)}",
        f"集成类型：{ensemble_type} (例如：Voting Classifier, Stacking Regressor, BaggingClassifier等)\n",
        "请在回答中包含以下内容：",
        "1. 一个模拟的“构建成功”或“已创建”的消息。",
        f"2. 对这个名为 '{ensemble_name}' 的模拟集成模型的工作原理进行简要描述（根据其类型和基础模型）。",
        "3. 列出这个模拟集成模型相对于其基础模型可能的潜在优势。",
        "4. 简要描述它可能适用于什么样的数据集或问题场景。",
        "5. 提供一些关于这个模拟集成模型的假设性元数据，例如模拟的创建时间戳、组合方式等。",
        "请以严格的JSON格式返回结果，结构如下：",
        """
{
  "success": true,
  "message": "模拟的构建成功消息，例如：集成模型 '[ensemble_name]' 已成功模拟创建！",
  "ensemble_name": "{ensemble_name}",
  "ensemble_type": "{ensemble_type}",
  "base_models_used": {base_models_list}, 
  "description": "这里是集成模型工作原理的模拟描述...",
  "potential_advantages": "这里是模拟的潜在优势列表或描述...",
  "suitable_scenarios": "这里是模拟的适用场景描述...",
  "model_info": {
    "simulated_created_at": "例如：一个ISO格式的时间戳，如 YYYY-MM-DDTHH:MM:SSZ",
    "simulated_combination_method": "例如：对投票分类器是'多数投票'或'加权投票'，对Stacking是'使用元学习器组合预测'等"
  }
}
""",
        "请确保整个响应是单一的、格式正确的JSON对象。"
    ]
    ensemble_prompt = "\n".join(prompt_parts)
    app.logger.info(f"模拟集成构建Prompt (部分): {ensemble_prompt[:300]}...")

    try:
        llm_raw_output_dict = enhanced_direct_query_llm(ensemble_prompt)  # √ 使用正确的 ensemble_prompt
        if not llm_raw_output_dict or "answer" not in llm_raw_output_dict:
            app.logger.error("enhanced_direct_query_llm 未返回预期的包含 'answer' 的字典。")
            return jsonify({"error": "调用大模型时发生内部错误 (无响应)。"}), 500

        llm_response_str = llm_raw_output_dict.get("answer", "")

        # --- 修改这里的日志标记 ---
        simulated_results, error_msg = extract_and_parse_json_from_llm(llm_response_str,
                                                                       "EnsembleBuilding")  # 修改为 "EnsembleBuilding"

        if error_msg:
            return jsonify({
                "error": error_msg,
                "raw_llm_response": llm_response_str
            }), 500

        # --- 修改这里的JSON结构验证 ---
        expected_keys = ["success", "ensemble_name", "ensemble_type", "base_models_used",
                         "description", "potential_advantages", "suitable_scenarios", "model_info"]

        if not isinstance(simulated_results, dict):
            app.logger.error(f"LLM返回的模拟集成结果不是一个字典: {simulated_results}")  # √ 日志文本正确
            return jsonify({
                "error": "大模型返回的模拟集成结果格式不正确 (非字典)。",  # √ 错误信息文本正确
                "parsed_response": simulated_results,
                "raw_llm_response": llm_response_str
            }), 500

        missing_keys = [key for key in expected_keys if key not in simulated_results]
        if missing_keys:
            app.logger.error(
                f"LLM返回的模拟集成结果JSON缺少关键字段: {missing_keys}. 结果: {simulated_results}")  # √ 日志文本正确
            return jsonify({
                "error": f"大模型返回的模拟集成结果JSON缺少必要字段: {', '.join(missing_keys)}。",  # √ 错误信息文本正确
                "parsed_response": simulated_results,
                "raw_llm_response": llm_response_str
            }), 500

        # 可以选择性地添加对特定字段类型的进一步验证
        if not isinstance(simulated_results.get("success"), bool):
            app.logger.error(f"模拟集成结果中 'success' 字段类型错误. 结果: {simulated_results}")
            return jsonify({
                "error": "模拟集成结果中 'success' 字段类型非布尔值。",
                "parsed_response": simulated_results, "raw_llm_response": llm_response_str
            }), 500
        if not isinstance(simulated_results.get("base_models_used"), list):
            app.logger.error(f"模拟集成结果中 'base_models_used' 字段类型错误. 结果: {simulated_results}")
            return jsonify({
                "error": "模拟集成结果中 'base_models_used' 字段类型非列表。",
                "parsed_response": simulated_results, "raw_llm_response": llm_response_str
            }), 500
        # ... 可以为 model_info 等其他字段添加类似检查 ...

        # 移除或注释掉针对 comparison_results 的 for 循环验证
        # for item in simulated_results["comparison_results"]: <--- 这个是错误的，应该移除

        return jsonify(simulated_results), 201  # √ 使用 201 Created
    except Exception as e:
        app.logger.error(f"模拟集成模型构建过程中发生错误: {str(e)}", exc_info=True)  # √ 日志文本正确
        return jsonify({"error": f"模拟集成模型构建时发生内部错误: {str(e)}"}), 500  # √ 错误信息文本正确

    @app.route('/api/ml/model_versions', methods=['POST'])
    def create_model_version_placeholder():
        # data = request.get_json() # 可以接收数据但不处理
        # model_name = data.get('model_name')
        # version_info = data.get('version_info')
        # app.logger.info(f"接收到创建模型版本请求 (前端模拟): {model_name} - {version_info}")
        return jsonify({"success": True, "message": "模型版本信息已在前端记录 (模拟)。"}), 200

    @app.route('/api/ml/model_versions/<model_name>', methods=['GET'])
    def get_model_versions_placeholder(model_name):
        # app.logger.info(f"接收到获取模型版本请求 (前端模拟): {model_name}")
        # 模拟返回空列表或一个示例结构
        return jsonify({"success": True, "versions": [], "message": "模型版本历史在前端管理 (模拟)。"}), 200

    # 对部署相关的API也做类似处理

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

    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx') or file.filename.endswith('.json')):
        try:
            # 确保上传目录存在
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)

            # 保存文件
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)

            # 如果是Excel文件，转换为CSV以便更好地处理
            if file.filename.endswith('.xlsx'):
                import pandas as pd
                df = pd.read_excel(file_path)
                csv_path = os.path.join(uploads_dir, file.filename.replace('.xlsx', '_processed.csv'))
                df.to_csv(csv_path, index=False)
                file_path = csv_path

            # 读取数据的前几行，获取列名和数据类型
            import pandas as pd
            # 处理NaN值，避免JSON序列化错误
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
            elif file_path.endswith('.json'):
                # 读取JSON文件
                df = pd.read_json(file_path, orient='records')
                # 处理可能的NaN值
                df = df.fillna('')

            # 推断每列的数据类型
            column_types = {}
            categorical_columns = []
            numerical_columns = []

            for col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    categorical_columns.append(col)
                    column_types[col] = 'categorical'
                else:
                    numerical_columns.append(col)
                    column_types[col] = 'numerical'

            # 使用json_compatible_result处理结果，确保没有NaN值
            result = json_compatible_result({
                "message": "文件上传成功",
                "file_path": file_path,
                "columns": df.columns.tolist(),
                "column_types": column_types,
                "categorical_columns": categorical_columns,
                "numerical_columns": numerical_columns,
                "row_count": len(df),
                "preview": df.head(10).to_dict('records')
            })

            return jsonify(result), 200
        except Exception as e:
            app.logger.error(f"/api/ml/upload 接口发生错误: {e}", exc_info=True)
            return jsonify({"error": f"处理上传文件时发生错误: {str(e)}"}), 500
    else:
        return jsonify({"error": "只支持CSV、Excel和JSON文件"}), 400

def json_compatible_result(data):
    """
    Recursively converts numpy NaN/Inf and other non-JSON serializable types
    to None or string.
    """
    if isinstance(data, dict):
        return {k: json_compatible_result(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_compatible_result(item) for item in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    elif isinstance(data, (np.int64, np.float64, np.bool_)):
        return data.item() # Convert numpy types to Python native types
    elif isinstance(data, np.ndarray):
        return data.tolist() # Convert numpy arrays to lists
    elif pd.isna(data): # Catch Pandas NaN specifically
        return None
    # Add other types if needed
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