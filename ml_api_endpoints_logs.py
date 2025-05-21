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
    获取指定部署的日志
    
    Args:
        deployment_id: 部署ID
        limit: 返回的日志条数限制
        offset: 日志起始偏移量
        
    Returns:
        包含日志信息的字典
    """
    try:
        # 验证参数
        if not deployment_id or not deployment_id.strip():
            return {
                "success": False,
                "error": "部署ID不能为空"
            }
            
        # 获取所有部署信息
        from ml_api_endpoints import _get_all_deployments
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
    添加部署日志条目
    
    Args:
        deployment_id: 部署ID
        log_entry: 日志条目，应包含level, message等字段
        
    Returns:
        操作结果
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
        from ml_api_endpoints import _get_all_deployments
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