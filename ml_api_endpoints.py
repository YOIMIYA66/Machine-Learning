# ml_api_endpoints.py
import os
import json
import uuid
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple

# 导入机器学习相关模块
from ml_models import (
    list_available_models, 
    load_model, 
    predict, 
    save_model_with_version, 
    list_model_versions,
    create_ensemble_model,
    compare_models
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='ml_api.log'
)
logger = logging.getLogger('ml_api_endpoints')

# 部署模型存储
DEPLOYMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deployments.json')
DEPLOYMENTS_BACKUP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deployments_backup.json')

# 确保部署文件存在
def _ensure_deployments_file():
    """确保部署文件存在，如果不存在则创建"""
    if not os.path.exists(DEPLOYMENTS_FILE):
        try:
            with open(DEPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            logger.info("创建新的部署文件")
        except Exception as e:
            logger.error(f"创建部署文件时出错: {str(e)}")
            raise

# 备份部署文件
def _backup_deployments_file():
    """创建部署文件的备份"""
    try:
        if os.path.exists(DEPLOYMENTS_FILE):
            with open(DEPLOYMENTS_FILE, 'r', encoding='utf-8') as src:
                data = json.load(src)
                with open(DEPLOYMENTS_BACKUP_FILE, 'w', encoding='utf-8') as dst:
                    json.dump(data, dst, ensure_ascii=False, indent=2)
            logger.info("已创建部署文件备份")
    except Exception as e:
        logger.error(f"备份部署文件时出错: {str(e)}")

# 获取所有部署
def _get_all_deployments() -> List[Dict[str, Any]]:
    """获取所有已部署的模型信息"""
    _ensure_deployments_file()
    try:
        with open(DEPLOYMENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("部署文件格式错误，尝试从备份恢复")
        try:
            # 尝试从备份恢复
            if os.path.exists(DEPLOYMENTS_BACKUP_FILE):
                with open(DEPLOYMENTS_BACKUP_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 恢复主文件
                with open(DEPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info("已从备份恢复部署文件")
                return data
        except Exception as backup_error:
            logger.error(f"从备份恢复失败: {str(backup_error)}")
        
        # 如果无法恢复，重置为空列表
        with open(DEPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []
    except Exception as e:
        logger.error(f"读取部署文件时出错: {str(e)}")
        return []

# 保存部署信息
def _save_deployments(deployments: List[Dict[str, Any]]):
    """保存部署信息到文件"""
    _ensure_deployments_file()
    # 先创建备份
    _backup_deployments_file()
    try:
        with open(DEPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(deployments, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存{len(deployments)}个部署信息")
    except Exception as e:
        logger.error(f"保存部署信息时出错: {str(e)}")
        raise

# API端点实现
def save_model_version(model_name: str, version_info: Dict[str, Any]) -> Dict[str, Any]:
    """保存模型的新版本
    
    Args:
        model_name: 模型名称
        version_info: 版本信息，包含description和performance等
        
    Returns:
        包含版本信息的字典
    """
    try:
        # 验证必要字段
        required_fields = ['description', 'performance']
        for field in required_fields:
            if field not in version_info:
                return {
                    "success": False,
                    "error": f"缺少必要的版本信息字段: {field}"
                }
        
        # 检查模型是否存在
        models = list_available_models()
        model_exists = any(m['name'] == model_name for m in models)
        if not model_exists:
            return {
                "success": False,
                "error": f"模型 '{model_name}' 不存在"
            }
        
        # 生成版本号
        current_versions = list_model_versions(model_name)
        version_number = len(current_versions) + 1
        version_id = f"v{version_number}"
        
        # 创建版本信息
        version_data = {
            "version_id": version_id,
            "created_at": datetime.datetime.now().isoformat(),
            "description": version_info.get('description', ''),
            "performance": version_info.get('performance', {}),
            "metadata": version_info.get('metadata', {})
        }
        
        # 保存版本
        save_model_with_version(model_name, version_data)
        
        return {
            "success": True,
            "model_name": model_name,
            "version": version_id,
            "version_info": version_data
        }
    except Exception as e:
        logger.error(f"保存模型版本时出错: {str(e)}")
        return {
            "success": False,
            "error": f"保存模型版本时出错: {str(e)}"
        }

def get_model_versions(model_name: str) -> Dict[str, Any]:
    """获取模型的所有版本信息
    
    Args:
        model_name: 模型名称
        
    Returns:
        包含版本列表的字典
    """
    try:
        # 检查模型是否存在
        models = list_available_models()
        model_exists = any(m['name'] == model_name for m in models)
        if not model_exists:
            return {
                "success": False,
                "error": f"模型 '{model_name}' 不存在"
            }
        
        # 获取版本信息
        versions = list_model_versions(model_name)
        
        # 获取模型类型
        model_type = next((m['type'] for m in models if m['name'] == model_name), None)
        
        return {
            "success": True,
            "model_name": model_name,
            "model_type": model_type,
            "versions": versions,
            "version_count": len(versions)
        }
    except Exception as e:
        logger.error(f"获取模型版本时出错: {str(e)}")
        return {
            "success": False,
            "error": f"获取模型版本时出错: {str(e)}"
        }

def build_ensemble_model(base_models: List[str], ensemble_type: str, save_name: Optional[str] = None) -> Dict[str, Any]:
    """构建集成模型
    
    Args:
        base_models: 基础模型名称列表
        ensemble_type: 集成类型 ('voting', 'stacking', 'bagging')
        save_name: 保存的模型名称
        
    Returns:
        包含集成模型信息的字典
    """
    try:
        # 验证参数
        if not base_models or len(base_models) < 2:
            return {
                "success": False,
                "error": "至少需要两个基础模型来构建集成模型"
            }
            
        if ensemble_type not in ['voting', 'stacking', 'bagging']:
            return {
                "success": False,
                "error": f"不支持的集成类型: {ensemble_type}，支持的类型有: voting, stacking, bagging"
            }
        
        # 检查所有基础模型是否存在
        models = list_available_models()
        model_names = [m['name'] for m in models]
        missing_models = [m for m in base_models if m not in model_names]
        
        if missing_models:
            return {
                "success": False,
                "error": f"以下模型不存在: {', '.join(missing_models)}"
            }
        
        # 构建集成模型
        result = create_ensemble_model(
            model_list=[(m, 1.0) for m in base_models],  # 默认权重为1.0
            ensemble_type=ensemble_type,
            save_name=save_name
        )
        
        # 格式化返回结果
        return {
            "success": True,
            "model_name": result['model_name'],
            "ensemble_type": ensemble_type,
            "base_models": base_models,
            "model_info": {
                "type": "ensemble",
                "ensemble_type": ensemble_type,
                "base_models": base_models,
                "created_at": datetime.datetime.now().isoformat(),
                "description": result.get('metadata', {}).get('description', f"{ensemble_type.capitalize()} 集成模型")
            }
        }
    except Exception as e:
        logger.error(f"构建集成模型时出错: {str(e)}")
        return {
            "success": False,
            "error": f"构建集成模型时出错: {str(e)}"
        }

def compare_models_api(model_names: List[str], test_data_path: str, target_column: str) -> Dict[str, Any]:
    """比较多个模型的性能
    
    Args:
        model_names: 要比较的模型名称列表
        test_data_path: 测试数据路径
        target_column: 目标列名
        
    Returns:
        包含比较结果的字典
    """
    try:
        # 验证参数
        if not model_names or len(model_names) < 2:
            return {
                "success": False,
                "error": "至少需要两个模型进行比较"
            }
            
        if not os.path.exists(test_data_path):
            return {
                "success": False,
                "error": f"测试数据文件不存在: {test_data_path}"
            }
        
        # 检查所有模型是否存在
        models = list_available_models()
        model_names_available = [m['name'] for m in models]
        missing_models = [m for m in model_names if m not in model_names_available]
        
        if missing_models:
            return {
                "success": False,
                "error": f"以下模型不存在: {', '.join(missing_models)}"
            }
        
        # 比较模型
        comparison_result = compare_models(model_names, test_data_path, target_column)
        
        # 格式化返回结果
        return {
            "success": True,
            "models": model_names,
            "test_data": test_data_path,
            "target_column": target_column,
            "comparison_result": comparison_result,
            "best_model": comparison_result.get('best_model', {}),
            "metrics": comparison_result.get('metrics', {}),
            "visualization_data": comparison_result.get('visualization_data', {})
        }
    except Exception as e:
        logger.error(f"比较模型时出错: {str(e)}")
        return {
            "success": False,
            "error": f"比较模型时出错: {str(e)}"
        }

def deploy_model(model_name: str, environment: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
    """部署模型到指定环境
    
    Args:
        model_name: 模型名称
        environment: 部署环境 ('development', 'staging', 'production')
        endpoint: API端点路径，如果为None则自动生成
        
    Returns:
        包含部署信息的字典
    """
    try:
        logger.info(f"开始部署模型 '{model_name}' 到 {environment} 环境")
        
        # 验证参数
        valid_environments = ['development', 'staging', 'production']
        if not model_name or not model_name.strip():
            return {
                "success": False,
                "error": "模型名称不能为空"
            }
            
        if environment not in valid_environments:
            return {
                "success": False,
                "error": f"无效的部署环境: {environment}，有效环境: {', '.join(valid_environments)}"
            }
            
        # 检查模型是否存在
        models = list_available_models()
        model_info = next((m for m in models if m['name'] == model_name), None)
        if not model_info:
            return {
                "success": False,
                "error": f"模型 '{model_name}' 不存在"
            }
        
        # 如果未提供端点，则自动生成
        if not endpoint:
            # 根据模型名称和环境生成端点
            model_type = model_info.get('type', 'model').lower()
            safe_name = model_name.replace('.', '_').lower()
            endpoint = f"/api/{environment}/{model_type}/{safe_name}"
            logger.info(f"自动生成端点: {endpoint}")
        else:
            # 规范化端点路径
            if not endpoint.startswith('/'):
                endpoint = f"/{endpoint}"
            
        # 检查端点是否已被使用
        deployments = _get_all_deployments()
        for dep in deployments:
            if dep['endpoint'] == endpoint and dep['environment'] == environment:
                # 如果是同一个模型的重新部署，则更新部署信息
                if dep['model_name'] == model_name:
                    dep['status'] = "运行中"
                    dep['updated_at'] = datetime.datetime.now().isoformat()
                    _save_deployments(deployments)
                    
                    return {
                        "success": True,
                        "deployment_id": dep['id'],
                        "model_name": model_name,
                        "environment": environment,
                        "endpoint_url": endpoint,
                        "status": "运行中",
                        "message": f"模型 '{model_name}' 已重新部署到 {environment} 环境的 {endpoint} 端点"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"端点 '{endpoint}' 在 {environment} 环境中已被模型 '{dep['model_name']}' 使用"
                    }
        
        # 创建部署ID
        deployment_id = f"dep_{str(uuid.uuid4())[:8]}"
        
        # 创建部署记录
        deployment = {
            "id": deployment_id,
            "model_name": model_name,
            "model_type": model_info.get('type', 'unknown'),
            "environment": environment,
            "endpoint": endpoint,
            "status": "运行中",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat(),
            "metrics": {
                "requests": 0,
                "avg_response_time": 0,
                "last_request": None,
                "success_count": 0,
                "error_count": 0
            }
        }
        
        # 保存部署信息
        deployments.append(deployment)
        _save_deployments(deployments)
        
        logger.info(f"模型 '{model_name}' 已成功部署到 {environment} 环境的 {endpoint} 端点")
        
        return {
            "success": True,
            "deployment_id": deployment_id,
            "model_name": model_name,
            "model_type": model_info.get('type', 'unknown'),
            "environment": environment,
            "endpoint_url": endpoint,
            "status": "运行中",
            "created_at": deployment['created_at'],
            "updated_at": deployment['updated_at'],
            "message": f"模型 '{model_name}' 已成功部署到 {environment} 环境的 {endpoint} 端点"
        }
    except Exception as e:
        logger.error(f"部署模型时出错: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"部署模型时出错: {str(e)}"
        }

def get_deployed_models() -> Dict[str, Any]:
    """获取所有已部署的模型信息
    
    Returns:
        包含部署列表的字典
    """
    try:
        logger.info("获取已部署模型列表")
        deployments = _get_all_deployments()
        
        # 按环境分组
        deployments_by_env = {
            "production": [],
            "staging": [],
            "development": []
        }
        
        for dep in deployments:
            env = dep.get('environment', 'development')
            if env in deployments_by_env:
                deployments_by_env[env].append(dep)
        
        # 计算统计信息
        total_requests = sum(dep.get('metrics', {}).get('requests', 0) for dep in deployments)
        running_deployments = [dep for dep in deployments if dep.get('status') == '运行中']
        avg_response_times = [dep.get('metrics', {}).get('avg_response_time', 0) for dep in running_deployments if dep.get('metrics', {}).get('avg_response_time', 0) > 0]
        overall_avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0
        
        # 计算成功率
        total_success = sum(dep.get('metrics', {}).get('success_count', 0) for dep in deployments)
        total_errors = sum(dep.get('metrics', {}).get('error_count', 0) for dep in deployments)
        success_rate = (total_success / (total_success + total_errors) * 100) if (total_success + total_errors) > 0 else 0
        
        # 按模型类型分组统计
        model_types = {}
        for dep in deployments:
            model_type = dep.get('model_type', 'unknown')
            if model_type not in model_types:
                model_types[model_type] = 0
            model_types[model_type] += 1
        
        logger.info(f"找到 {len(deployments)} 个部署，其中 {len(running_deployments)} 个正在运行")
        
        return {
            "success": True,
            "deployments": deployments,
            "deployments_by_env": deployments_by_env,
            "count": len(deployments),
            "running_count": len(running_deployments),
            "total_requests": total_requests,
            "avg_response_time": overall_avg_response_time,
            "success_rate": round(success_rate, 2),
            "model_types": model_types,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"获取已部署模型时出错: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"获取已部署模型时出错: {str(e)}",
            "deployments": [],
            "count": 0
        }

def undeploy_model(deployment_id: str) -> Dict[str, Any]:
    """取消部署模型
    
    Args:
        deployment_id: 部署ID
        
    Returns:
        包含操作结果的字典
    """
    if not deployment_id or not deployment_id.strip():
        return {
            "success": False,
            "error": "部署ID不能为空"
        }
        
    logger.info(f"请求停止部署模型，部署ID: {deployment_id}")
    try:
        deployments = _get_all_deployments()
        
        # 查找部署记录
        deployment_index = None
        for i, dep in enumerate(deployments):
            if dep['id'] == deployment_id:
                deployment_index = i
                break
                
        if deployment_index is None:
            logger.warning(f"尝试停止不存在的部署，ID: {deployment_id}")
            return {
                "success": False,
                "error": f"找不到部署ID: {deployment_id}"
            }
            
        # 获取部署信息用于返回
        deployment = deployments[deployment_index]
        model_name = deployment.get('model_name', 'unknown')
        environment = deployment.get('environment', 'unknown')
        endpoint = deployment.get('endpoint', 'unknown')
        status = deployment.get('status', 'unknown')
        
        # 检查部署状态
        if status != '运行中':
            logger.warning(f"尝试停止非运行状态的部署，ID: {deployment_id}, 当前状态: {status}")
            return {
                "success": False,
                "error": f"部署 '{model_name}' 当前状态为 '{status}'，无法停止"
            }
        
        # 更新部署状态为已停止，而不是直接删除
        deployment['status'] = '已停止'
        deployment['updated_at'] = datetime.datetime.now().isoformat()
        deployment['stop_reason'] = '用户请求'
        _save_deployments(deployments)
        
        logger.info(f"已成功停止部署，ID: {deployment_id}, 模型: {model_name}, 环境: {environment}")
        
        return {
            "success": True,
            "deployment_id": deployment_id,
            "model_name": model_name,
            "environment": environment,
            "endpoint": endpoint,
            "previous_status": "运行中",
            "current_status": "已停止",
            "updated_at": deployment['updated_at'],
            "message": f"模型 '{model_name}' 已从 {environment} 环境的 {endpoint} 端点取消部署"
        }
    except Exception as e:
        logger.error(f"取消部署模型时出错: {str(e)}")
        return {
            "success": False,
            "error": f"取消部署模型时出错: {str(e)}"
        }