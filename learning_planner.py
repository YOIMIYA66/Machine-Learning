#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
学习路径规划模块
用于生成和管理个性化学习路径
"""

import os
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional, Union
import logging
import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 学习路径存储目录
LEARNING_PATHS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'learning_paths')
os.makedirs(LEARNING_PATHS_DIR, exist_ok=True)

# 学习主题和模块数据 (示例数据，实际应用中可能来自数据库)
LEARNING_TOPICS = {
    "machine_learning": {
        "display_name": "机器学习",
        "description": "机器学习基础与应用",
        "modules": [
            {
                "id": "ml_intro",
                "title": "机器学习导论",
                "description": "了解机器学习的基本概念、应用场景和常见算法类型",
                "difficulty": "beginner",
                "estimated_hours": 3,
                "prerequisites": []
            },
            {
                "id": "data_preprocessing",
                "title": "数据预处理",
                "description": "学习数据清洗、特征工程和数据转换的基本技术",
                "difficulty": "beginner",
                "estimated_hours": 5,
                "prerequisites": ["ml_intro"]
            },
            {
                "id": "supervised_learning",
                "title": "监督学习算法",
                "description": "掌握线性回归、逻辑回归、决策树等基础算法",
                "difficulty": "intermediate",
                "estimated_hours": 8,
                "prerequisites": ["data_preprocessing"]
            },
            {
                "id": "unsupervised_learning",
                "title": "无监督学习算法",
                "description": "学习聚类、降维等无监督学习方法",
                "difficulty": "intermediate",
                "estimated_hours": 6,
                "prerequisites": ["supervised_learning"]
            },
            {
                "id": "model_evaluation",
                "title": "模型评估与优化",
                "description": "学习模型评估指标、交叉验证和超参数调优",
                "difficulty": "intermediate",
                "estimated_hours": 5,
                "prerequisites": ["supervised_learning"]
            },
            {
                "id": "ensemble_methods",
                "title": "集成学习方法",
                "description": "掌握随机森林、Boosting等集成学习算法",
                "difficulty": "advanced",
                "estimated_hours": 6,
                "prerequisites": ["model_evaluation"]
            },
            {
                "id": "ml_project",
                "title": "机器学习项目实践",
                "description": "从数据收集到部署，完成一个完整的机器学习项目",
                "difficulty": "advanced",
                "estimated_hours": 12,
                "prerequisites": ["ensemble_methods"]
            }
        ]
    },
    "deep_learning": {
        "display_name": "深度学习",
        "description": "深度学习原理与应用",
        "modules": [
            {
                "id": "dl_intro",
                "title": "深度学习导论",
                "description": "了解深度学习的基本概念、历史和应用",
                "difficulty": "beginner",
                "estimated_hours": 4,
                "prerequisites": ["machine_learning.supervised_learning"]
            },
            {
                "id": "neural_networks",
                "title": "神经网络基础",
                "description": "学习神经网络的结构、前向传播和反向传播算法",
                "difficulty": "intermediate",
                "estimated_hours": 8,
                "prerequisites": ["dl_intro"]
            },
            # 更多模块...
        ]
    }
}


def generate_learning_path(user_id: str, goal: str, prior_knowledge: str, weekly_hours: float) -> Dict[str, Any]:
    """
    基于用户目标和背景生成个性化学习路径
    
    Args:
        user_id: 用户ID
        goal: 学习目标描述
        prior_knowledge: 用户已有知识水平
        weekly_hours: 每周计划学习时间
        
    Returns:
        包含个性化学习路径的字典
    """
    logger.info(f"为用户 {user_id} 生成学习路径，目标: {goal}")
    
    # 生成路径ID
    path_id = str(uuid.uuid4())
    
    # 此处应该调用大模型API来分析用户目标，选择合适的主题和模块
    # 以下为简化实现，实际应用中可能基于LLM分析
    
    # 基于目标关键词选择主题
    selected_topic = "machine_learning"  # 默认选择机器学习
    if "深度" in goal or "neural" in goal.lower() or "deep" in goal.lower():
        selected_topic = "deep_learning"
    
    # 获取主题信息
    topic_info = LEARNING_TOPICS.get(selected_topic, LEARNING_TOPICS["machine_learning"])
    
    # 基于先验知识选择起始模块
    all_modules = topic_info["modules"]
    
    # 根据先验知识水平确定起始模块
    if "高级" in prior_knowledge or "advanced" in prior_knowledge.lower():
        # 高级用户可以跳过基础模块
        starting_module_index = min(2, len(all_modules) - 1)
    elif "中级" in prior_knowledge or "intermediate" in prior_knowledge.lower():
        # 中级用户可以跳过入门模块
        starting_module_index = min(1, len(all_modules) - 1)
    else:
        # 初学者从头开始
        starting_module_index = 0
    
    # 根据起始模块和依赖关系，生成模块序列
    selected_modules = all_modules[starting_module_index:]
    
    # 计算总学习时间和预计完成日期
    total_hours = sum(module["estimated_hours"] for module in selected_modules)
    weeks_needed = total_hours / weekly_hours
    
    # 创建今天的日期对象
    today = datetime.datetime.now().date()
    
    # 计算预计完成日期
    estimated_completion_date = today + datetime.timedelta(days=int(weeks_needed * 7))
    
    # 构建学习路径
    learning_path = {
        "path_id": path_id,
        "user_id": user_id,
        "created_at": datetime.datetime.now().isoformat(),
        "goal": goal,
        "prior_knowledge": prior_knowledge,
        "weekly_hours": weekly_hours,
        "topic": {
            "id": selected_topic,
            "name": topic_info["display_name"],
            "description": topic_info["description"]
        },
        "modules": selected_modules,
        "total_hours": total_hours,
        "estimated_completion_date": estimated_completion_date.isoformat(),
        "weeks_needed": round(weeks_needed, 1),
        "progress": {
            "completed_modules": [],
            "current_module": selected_modules[0]["id"] if selected_modules else None,
            "completion_percentage": 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
    }
    
    # 保存学习路径
    save_learning_path(path_id, learning_path)
    
    return learning_path


def save_learning_path(path_id: str, learning_path: Dict[str, Any]) -> bool:
    """
    将学习路径保存到文件
    
    Args:
        path_id: 路径ID
        learning_path: 学习路径数据
        
    Returns:
        保存是否成功
    """
    try:
        file_path = os.path.join(LEARNING_PATHS_DIR, f"{path_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(learning_path, f, ensure_ascii=False, indent=2)
        logger.info(f"学习路径 {path_id} 已保存")
        return True
    except Exception as e:
        logger.error(f"保存学习路径 {path_id} 时出错: {str(e)}")
        return False


def get_user_learning_path(user_id: str) -> List[Dict[str, Any]]:
    """
    获取用户的所有学习路径
    
    Args:
        user_id: 用户ID
        
    Returns:
        用户的学习路径列表
    """
    paths = []
    
    try:
        for filename in os.listdir(LEARNING_PATHS_DIR):
            if filename.endswith('.json'):
                file_path = os.path.join(LEARNING_PATHS_DIR, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    path_data = json.load(f)
                    if path_data.get('user_id') == user_id:
                        paths.append(path_data)
        
        # 按创建时间排序，最新的在前
        paths.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return paths
    except Exception as e:
        logger.error(f"获取用户 {user_id} 的学习路径时出错: {str(e)}")
        return []


def update_path_progress(path_id: str, completed_module_id: Optional[str] = None, 
                        current_module_id: Optional[str] = None) -> Dict[str, Any]:
    """
    更新学习路径的进度
    
    Args:
        path_id: 路径ID
        completed_module_id: 完成的模块ID (可选)
        current_module_id: 当前进行的模块ID (可选)
        
    Returns:
        更新后的学习路径
    """
    try:
        # 读取现有路径
        file_path = os.path.join(LEARNING_PATHS_DIR, f"{path_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"学习路径 {path_id} 不存在")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            learning_path = json.load(f)
        
        # 更新进度信息
        progress = learning_path["progress"]
        
        # 如果提供了完成的模块ID，将其添加到已完成模块列表
        if completed_module_id and completed_module_id not in progress["completed_modules"]:
            progress["completed_modules"].append(completed_module_id)
        
        # 如果提供了当前模块ID，更新当前模块
        if current_module_id:
            progress["current_module"] = current_module_id
            
        # 计算完成百分比
        total_modules = len(learning_path["modules"])
        if total_modules > 0:
            progress["completion_percentage"] = round(len(progress["completed_modules"]) / total_modules * 100, 1)
        else:
            progress["completion_percentage"] = 0
            
        # 更新时间戳
        progress["last_updated"] = datetime.datetime.now().isoformat()
        
        # 保存更新后的路径
        save_learning_path(path_id, learning_path)
        
        return learning_path
    except Exception as e:
        logger.error(f"更新学习路径 {path_id} 的进度时出错: {str(e)}")
        raise


def predict_module_mastery(user_id: str, module_id: str, weekly_hours: float, 
                          focus_level: str = "medium") -> Dict[str, Any]:
    """
    预测用户掌握特定模块的概率
    
    Args:
        user_id: 用户ID
        module_id: 模块ID
        weekly_hours: 每周学习时间
        focus_level: 学习专注程度 (low, medium, high)
        
    Returns:
        包含预测结果的字典
    """
    try:
        # 查找模块信息
        module_info = None
        module_topic = None
        
        for topic_id, topic_data in LEARNING_TOPICS.items():
            for module in topic_data["modules"]:
                if module["id"] == module_id:
                    module_info = module
                    module_topic = topic_id
                    break
            if module_info:
                break
                
        if not module_info:
            raise ValueError(f"模块 {module_id} 不存在")
        
        # 获取用户学习路径
        user_paths = get_user_learning_path(user_id)
        
        # 计算用户已完成的相关模块数量
        completed_modules = []
        if user_paths:
            for path in user_paths:
                completed_modules.extend(path["progress"]["completed_modules"])
        
        # 计算已完成的前置条件比例
        prerequisites = module_info.get("prerequisites", [])
        if prerequisites:
            completed_prereqs = sum(1 for prereq in prerequisites if prereq in completed_modules)
            prereq_completion_ratio = completed_prereqs / len(prerequisites)
        else:
            prereq_completion_ratio = 1.0  # 没有前置条件
        
        # 基于学习时间和专注度计算基础概率
        base_probability = 0.5  # 基础概率
        
        # 调整系数
        time_factor = min(1.0, weekly_hours / 10.0) * 0.3  # 每周学习10小时以上获得最大时间加成
        
        focus_factors = {
            "low": 0.1,
            "medium": 0.2,
            "high": 0.3
        }
        focus_factor = focus_factors.get(focus_level, 0.2)
        
        difficulty_factors = {
            "beginner": 0.2,
            "intermediate": 0.1,
            "advanced": 0.0
        }
        difficulty_factor = difficulty_factors.get(module_info.get("difficulty", "intermediate"), 0.1)
        
        # 计算最终概率
        mastery_probability = base_probability + time_factor + focus_factor + difficulty_factor + prereq_completion_ratio * 0.2
        mastery_probability = min(0.95, max(0.05, mastery_probability))  # 限制在5%-95%之间
        
        # 构建结果
        result = {
            "user_id": user_id,
            "module_id": module_id,
            "module_title": module_info["title"],
            "weekly_hours": weekly_hours,
            "focus_level": focus_level,
            "mastery_probability": round(mastery_probability * 100, 1),  # 转换为百分比
            "factors": {
                "time_factor": round(time_factor * 100, 1),
                "focus_factor": round(focus_factor * 100, 1),
                "difficulty_factor": round(difficulty_factor * 100, 1),
                "prereq_completion": round(prereq_completion_ratio * 100, 1)
            },
            "predicted_at": datetime.datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        logger.error(f"预测模块掌握概率时出错: {str(e)}")
        raise


def predict_completion_time(user_id: str, path_id: str, weekly_hours: float) -> Dict[str, Any]:
    """
    预测完成学习路径所需的时间
    
    Args:
        user_id: 用户ID
        path_id: 路径ID
        weekly_hours: 每周学习时间
        
    Returns:
        包含预测结果的字典
    """
    try:
        # 读取学习路径
        file_path = os.path.join(LEARNING_PATHS_DIR, f"{path_id}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"学习路径 {path_id} 不存在")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            learning_path = json.load(f)
            
        # 确认用户ID匹配
        if learning_path["user_id"] != user_id:
            raise ValueError("用户ID不匹配")
            
        # 获取路径中的所有模块
        modules = learning_path["modules"]
        
        # 获取已完成的模块
        completed_modules = learning_path["progress"]["completed_modules"]
        
        # 计算剩余模块的总学习时间
        remaining_hours = 0
        for module in modules:
            if module["id"] not in completed_modules:
                remaining_hours += module["estimated_hours"]
                
        # 计算预计完成时间
        if weekly_hours <= 0:
            raise ValueError("每周学习时间必须大于0")
            
        weeks_needed = remaining_hours / weekly_hours
        days_needed = int(weeks_needed * 7)
        
        # 计算预计完成日期
        today = datetime.datetime.now().date()
        estimated_completion_date = today + datetime.timedelta(days=days_needed)
        
        # 构建结果
        result = {
            "user_id": user_id,
            "path_id": path_id,
            "path_name": f"{learning_path['topic']['name']} 学习路径",
            "total_modules": len(modules),
            "completed_modules": len(completed_modules),
            "remaining_modules": len(modules) - len(completed_modules),
            "original_total_hours": learning_path["total_hours"],
            "remaining_hours": remaining_hours,
            "weekly_hours": weekly_hours,
            "weeks_needed": round(weeks_needed, 1),
            "days_needed": days_needed,
            "estimated_completion_date": estimated_completion_date.isoformat(),
            "prediction_scenarios": [
                {"weekly_hours": max(1, weekly_hours - 5), "weeks_needed": round(remaining_hours / max(1, weekly_hours - 5), 1)},
                {"weekly_hours": weekly_hours, "weeks_needed": round(weeks_needed, 1)},
                {"weekly_hours": weekly_hours + 5, "weeks_needed": round(remaining_hours / (weekly_hours + 5), 1)}
            ],
            "predicted_at": datetime.datetime.now().isoformat()
        }
        
        return result
    except Exception as e:
        logger.error(f"预测完成时间时出错: {str(e)}")
        raise 