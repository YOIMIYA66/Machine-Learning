# ml_api_endpoints_logs.py
# 这个文件包含部署日志相关的API端点函数

import os
import json
import datetime
import logging
import uuid
from typing import Dict, Any, List, Optional

# 设置日志记录器
logger = logging.getLogger('ml_api_endpoints')

# 部署日志文件路径
DEPLOYMENTS_LOG_DIR = "ml_deployments/logs"
os.makedirs(DEPLOYMENTS_LOG_DIR, exist_ok=True)

def get_deployment_logs(deployment_id: str, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """
    获取指定部署的日志。

    Args:
        deployment_id (str): 部署ID。
        limit (int): 返回的日志条数限制，默认为100。
        offset (int): 日志起始偏移量，默认为0。
        
    Returns:
        Dict[str, Any]: 包含日志信息的字典。如果操作成功，包含 'success': True 和日志数据；
                        如果失败，包含 'success': False 和错误信息。
    
    Raises:
        Exception: 处理过程中可能发生的任何异常。
    """
    try:
        # 验证参数
        if not deployment_id or not deployment_id.strip():
            return {
                "success": False,
                "error": "部署ID不能为空"
            }
            
        # 获取所有部署信息
        from deployment_utils import _get_all_deployments
        deployments = _get_all_deployments()
        
        # 查找部署记录
        deployment = next((dep for dep in deployments if dep['id'] == deployment_id), None)
        if not deployment:
            return {
                "success": False,
                "error": f"找不到部署ID: {deployment_id}"
            }
        
        # 构建日志文件路径
        log_file_path = os.path.join(DEPLOYMENTS_LOG_DIR, f"{deployment_id}.json")
        
        # 如果日志文件不存在，创建一个空的日志文件
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "deployment_id": deployment_id,
                    "model_name": deployment.get('model_name', 'unknown'),
                    "environment": deployment.get('environment', 'unknown'),
                    "endpoint": deployment.get('endpoint', 'unknown'),
                    "logs": []
                }, f, ensure_ascii=False, indent=2)
        
        # 读取日志文件
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        # 获取指定范围的日志
        logs = log_data.get('logs', [])
        total_logs = len(logs)
        logs_slice = logs[offset:offset+limit] if offset < total_logs else []
        
        return {
            "success": True,
            "deployment_id": deployment_id,
            "model_name": deployment.get('model_name', 'unknown'),
            "environment": deployment.get('environment', 'unknown'),
            "endpoint": deployment.get('endpoint', 'unknown'),
            "status": deployment.get('status', 'unknown'),
            "logs": logs_slice,
            "total_logs": total_logs,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total_logs
        }
    except Exception as e:
        logger.error(f"获取部署日志时出错: {str(e)}")
        return {
            "success": False,
            "error": f"获取部署日志时出错: {str(e)}"
        }

def add_deployment_log(deployment_id: str, log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    添加部署日志条目。
    
    Args:
        deployment_id (str): 部署ID。
        log_entry (Dict[str, Any]): 日志条目，应包含 'level', 'message' 等字段。
        
    Returns:
        Dict[str, Any]: 操作结果。如果操作成功，包含 'success': True 和日志ID；
                        如果失败，包含 'success': False 和错误信息。

    Raises:
        Exception: 处理过程中可能发生的任何异常。
    """
    try:
        # 验证参数
        if not deployment_id or not deployment_id.strip():
            return {
                "success": False,
                "error": "部署ID不能为空"
            }
            
        if not log_entry or not isinstance(log_entry, dict):
            return {
                "success": False,
                "error": "日志条目必须是一个字典"
            }
        
        # 获取所有部署信息
        from deployment_utils import _get_all_deployments
        deployments = _get_all_deployments()
        
        # 查找部署记录
        deployment = next((dep for dep in deployments if dep['id'] == deployment_id), None)
        if not deployment:
            return {
                "success": False,
                "error": f"找不到部署ID: {deployment_id}"
            }
        
        # 构建日志文件路径
        log_file_path = os.path.join(DEPLOYMENTS_LOG_DIR, f"{deployment_id}.json")
        
        # 如果日志文件不存在，创建一个空的日志文件
        if not os.path.exists(log_file_path):
            with open(log_file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "deployment_id": deployment_id,
                    "model_name": deployment.get('model_name', 'unknown'),
                    "environment": deployment.get('environment', 'unknown'),
                    "endpoint": deployment.get('endpoint', 'unknown'),
                    "logs": []
                }, f, ensure_ascii=False, indent=2)
        
        # 读取日志文件
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        
        # 添加时间戳和ID到日志条目
        log_entry["timestamp"] = datetime.datetime.now().isoformat()
        log_entry["id"] = str(uuid.uuid4())[:8]
        
        # 添加日志条目
        log_data["logs"].append(log_entry)
        
        # 写回日志文件
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "deployment_id": deployment_id,
            "log_id": log_entry["id"],
            "message": "日志条目已添加"
        }
    except Exception as e:
        logger.error(f"添加部署日志时出错: {str(e)}")
        return {
            "success": False,
            "error": f"添加部署日志时出错: {str(e)}"
        }

def get_deployment_metrics(deployment_id: str) -> Dict[str, Any]:
    """
    获取指定部署的性能指标。
    注意: 当前实现包含一个从 'ml_api_endpoints' 导入 '_get_all_deployments' 的操作，
    这可能与 'ml_api_endpoints.py' 中对此文件的导入形成循环依赖，建议重构以避免此问题。

    Args:
        deployment_id (str): 部署ID。

    Returns:
        Dict[str, Any]: 包含性能指标的字典。如果操作成功，包含 'success': True 和指标数据；
                        如果失败，包含 'success': False 和错误信息。
    """
    logger.info(f"Fetching metrics for deployment ID: {deployment_id}")
    # 此处为示例实现，实际应从日志、监控系统或数据库中获取真实指标
    # 例如，可以解析日志文件中的请求响应时间、错误率等

    # 模拟指标数据
    # 在实际应用中，这些数据应该动态生成或从存储中读取
    if not deployment_id or not deployment_id.strip():
        return {
            "success": False,
            "error": "部署ID不能为空"
        }

    # 尝试从日志文件中聚合一些基本指标，或者从专门的指标存储中获取
    # 这里我们先返回一些模拟数据
    # 检查部署是否存在 (可以复用 get_deployment_logs 中的逻辑或 _get_all_deployments)
    from deployment_utils import _get_all_deployments
    deployments = _get_all_deployments()
    deployment = next((dep for dep in deployments if dep['id'] == deployment_id), None)

    if not deployment:
        return {
            "success": False,
            "error": f"找不到部署ID: {deployment_id}"
        }

    # 模拟指标
    # TODO: 实现真实的指标收集逻辑
    # 例如，可以分析日志文件中的请求数量、平均响应时间、错误率等
    # log_file_path = os.path.join(DEPLOYMENTS_LOG_DIR, f"{deployment_id}.json")
    # request_count = 0
    # total_response_time = 0
    # error_count = 0
    # if os.path.exists(log_file_path):
    #     with open(log_file_path, 'r', encoding='utf-8') as f:
    #         log_data = json.load(f)
    #         logs = log_data.get('logs', [])
    #         # 假设日志条目中有 'type' (e.g., 'request', 'error') 和 'response_time_ms'
    #         for log_entry in logs:
    #             if log_entry.get('type') == 'request':
    #                 request_count += 1
    #                 total_response_time += log_entry.get('response_time_ms', 0)
    #             elif log_entry.get('level') == 'ERROR' or log_entry.get('type') == 'error':
    #                 error_count += 1
    # avg_response_time = (total_response_time / request_count) if request_count > 0 else 0
    # error_rate = (error_count / request_count) * 100 if request_count > 0 else 0

    # 简化版模拟数据
    import random
    request_count = random.randint(100, 1000)
    avg_response_time = random.uniform(50, 500) # ms
    error_rate = random.uniform(0, 5) # percentage
    uptime_percentage = random.uniform(99, 100)

    return {
        "success": True,
        "deployment_id": deployment_id,
        "model_name": deployment.get('model_name', 'unknown'),
        "metrics": {
            "total_requests": request_count,
            "avg_response_time_ms": round(avg_response_time, 2),
            "error_rate_percent": round(error_rate, 2),
            "uptime_percentage": round(uptime_percentage, 2),
            "cpu_usage_percent": random.uniform(10, 70), # 模拟
            "memory_usage_mb": random.uniform(100, 500) # 模拟
        },
        "last_updated": datetime.datetime.now().isoformat()
    }

# 示例：如何记录一个包含性能信息的日志条目 (可以在模型预测端点中调用)
# def record_prediction_performance(deployment_id: str, response_time_ms: float, success: bool):
#     log_level = "INFO" if success else "ERROR"
#     message = f"Prediction processed in {response_time_ms:.2f}ms"
#     if not success:
#         message = f"Prediction failed after {response_time_ms:.2f}ms"
    
#     add_deployment_log(deployment_id, {
#         "level": log_level,
#         "message": message,
#         "type": "request_metric", # 自定义类型，便于后续解析
#         "response_time_ms": response_time_ms,
#         "status_code": 200 if success else 500 # 假设的HTTP状态码
#     })