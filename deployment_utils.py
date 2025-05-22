# deployment_utils.py
import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger('deployment_utils')

DEPLOYMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deployments.json')
DEPLOYMENTS_BACKUP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deployments_backup.json')

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

def _get_all_deployments() -> List[Dict[str, Any]]:
    """获取所有已部署的模型信息"""
    _ensure_deployments_file()
    try:
        with open(DEPLOYMENTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("部署文件格式错误，尝试从备份恢复")
        try:
            if os.path.exists(DEPLOYMENTS_BACKUP_FILE):
                with open(DEPLOYMENTS_BACKUP_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                with open(DEPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info("已从备份恢复部署文件")
                return data
        except Exception as backup_error:
            logger.error(f"从备份恢复失败: {str(backup_error)}")
        
        with open(DEPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []
    except Exception as e:
        logger.error(f"读取部署文件时出错: {str(e)}")
        return []

def _save_deployments(deployments: List[Dict[str, Any]]):
    """保存部署信息到文件"""
    _ensure_deployments_file()
    _backup_deployments_file()
    try:
        with open(DEPLOYMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(deployments, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存{len(deployments)}个部署信息")
    except Exception as e:
        logger.error(f"保存部署信息时出错: {str(e)}")
        raise