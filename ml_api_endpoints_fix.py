# ml_api_endpoints_fix.py
# 这个文件包含修复后的API端点函数，用于解决compare_models和ensemble API的400错误

# 导入原始ml_api_endpoints.py中的所有函数
from ml_api_endpoints import (
    save_model_version,
    get_model_versions,
    deploy_model,
    get_deployed_models,
    undeploy_model,
    get_deployment_logs,
    get_deployment_metrics
)

# 修复后的compare_models_api函数
def compare_models_api(model_names, test_data_path, target_column):
    """
    比较多个模型的性能
    
    Args:
        model_names: 要比较的模型名称列表
        test_data_path: 测试数据路径
        target_column: 目标列名
        
    Returns:
        包含比较结果的字典
    """
    import os
    import logging
    from ml_models import list_available_models, compare_models
    
    logger = logging.getLogger('ml_api_endpoints')
    
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

# 修复后的build_ensemble_model函数
def build_ensemble_model(base_models, ensemble_type, save_name=None):
    """
    构建集成模型
    
    Args:
        base_models: 基础模型名称列表
        ensemble_type: 集成类型 ('voting', 'stacking', 'bagging')
        save_name: 保存的模型名称
        
    Returns:
        包含集成模型信息的字典
    """
    import datetime
    import logging
    from ml_models import list_available_models, create_ensemble_model
    
    logger = logging.getLogger('ml_api_endpoints')
    
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