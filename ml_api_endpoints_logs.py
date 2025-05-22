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
    此函数会尝试从部署的日志文件 ({deployment_id}.json) 中解析和聚合真实的性能指标，
    例如请求总数、平均响应时间和错误率。
    期望日志条目中包含 type='request_metric', response_time_ms, level, 和 status_code 字段来计算这些指标。
    其他指标（如CPU/内存使用率、正常运行时间百分比）当前仍为模拟数据。

    注意: 当前实现包含一个从 'deployment_utils' 导入 '_get_all_deployments' 的操作。
    如果 'deployment_utils' 或其依赖间接导入此模块，可能需要注意循环依赖问题。

    Args:
        deployment_id (str): 部署ID。

    Returns:
        Dict[str, Any]: 包含性能指标的字典。如果操作成功，包含 'success': True 和指标数据；
                        如果失败，包含 'success': False 和错误信息。
    """
    logger.info(f"Fetching metrics for deployment ID: {deployment_id}")

    if not deployment_id or not deployment_id.strip():
        return {
            "success": False,
            "error": "部署ID不能为空"
        }

    from deployment_utils import _get_all_deployments
    deployments = _get_all_deployments()
    deployment = next((dep for dep in deployments if dep['id'] == deployment_id), None)

    if not deployment:
        return {
            "success": False,
            "error": f"找不到部署ID: {deployment_id}"
        }

    # 从日志文件聚合真实指标
    log_file_path = os.path.join(DEPLOYMENTS_LOG_DIR, f"{deployment_id}.json")
    request_count = 0
    total_response_time_ms = 0.0
    error_count = 0

    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            logs = log_data.get('logs', [])
            for log_entry in logs:
                # 假设 'request_metric' 类型的日志包含性能数据
                if log_entry.get('type') == 'request_metric':
                    request_count += 1
                    total_response_time_ms += float(log_entry.get('response_time_ms', 0.0))
                    # 假设 'level': 'ERROR' 或非200的 status_code 表示请求处理错误
                    if log_entry.get('level') == 'ERROR' or \
                       (log_entry.get('status_code') is not None and \
                        not str(log_entry.get('status_code')).startswith('2')):
                        error_count += 1
        except json.JSONDecodeError as je:
            logger.error(f"解析日志文件 {log_file_path} 时出错: {str(je)}")
        except Exception as ex:
            logger.error(f"读取或处理日志文件 {log_file_path} 时出错: {str(ex)}")

    avg_response_time_ms = (total_response_time_ms / request_count) if request_count > 0 else 0.0
    error_rate_percent = (error_count / request_count) * 100 if request_count > 0 else 0.0

    # 其他指标可以暂时保留模拟数据或后续实现
    import random # 保留 random 导入，因为其他指标仍使用它
    uptime_percentage = random.uniform(99, 100) # 模拟
    cpu_usage_percent = random.uniform(10, 70) # 模拟
    memory_usage_mb = random.uniform(100, 500) # 模拟

    return {
        "success": True,
        "deployment_id": deployment_id,
        "model_name": deployment.get('model_name', 'unknown'),
        "metrics": {
            "total_requests": request_count,
            "avg_response_time_ms": round(avg_response_time_ms, 2),
            "error_rate_percent": round(error_rate_percent, 2),
            "uptime_percentage": round(uptime_percentage, 2),
            "cpu_usage_percent": round(cpu_usage_percent, 2),
            "memory_usage_mb": round(memory_usage_mb, 2)
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