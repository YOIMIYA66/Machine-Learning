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

# 知识图谱和学习资源目录
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'knowledge')
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# 默认知识图谱文件
DEFAULT_KNOWLEDGE_GRAPH = os.path.join(KNOWLEDGE_DIR, 'knowledge_graph.json')

# 如果知识图谱文件不存在，创建一个简单的示例
if not os.path.exists(DEFAULT_KNOWLEDGE_GRAPH):
    default_graph = {
        "domains": [
            {
                "id": "ml",
                "name": "机器学习",
                "topics": [
                    {
                        "id": "ml_basics",
                        "name": "机器学习基础",
                        "difficulty": "beginner",
                        "modules": [
                            {
                                "id": "ml_intro",
                                "name": "机器学习介绍",
                                "description": "机器学习的基本概念和类型",
                                "estimated_hours": 2,
                                "resources": ["video:intro_to_ml", "article:ml_overview"],
                                "prerequisites": []
                            },
                            {
                                "id": "ml_workflow",
                                "name": "机器学习工作流程",
                                "description": "机器学习项目的端到端流程",
                                "estimated_hours": 3,
                                "resources": ["video:ml_workflow", "notebook:ml_pipeline"],
                                "prerequisites": ["ml_intro"]
                            }
                        ]
                    },
                    {
                        "id": "supervised_learning",
                        "name": "监督学习",
                        "difficulty": "intermediate",
                        "modules": [
                            {
                                "id": "linear_regression",
                                "name": "线性回归",
                                "description": "线性回归模型的原理和实现",
                                "estimated_hours": 4,
                                "resources": ["video:linear_regression", "notebook:linear_regression_impl"],
                                "prerequisites": ["ml_workflow"]
                            },
                            {
                                "id": "classification",
                                "name": "分类算法",
                                "description": "常见分类算法介绍和应用",
                                "estimated_hours": 5,
                                "resources": ["video:classification_intro", "notebook:classification_algorithms"],
                                "prerequisites": ["linear_regression"]
                            }
                        ]
                    },
                    {
                        "id": "unsupervised_learning",
                        "name": "无监督学习",
                        "difficulty": "intermediate",
                        "modules": [
                            {
                                "id": "clustering",
                                "name": "聚类算法",
                                "description": "常见聚类算法介绍和应用",
                                "estimated_hours": 4,
                                "resources": ["video:clustering_intro", "notebook:clustering_algorithms"],
                                "prerequisites": ["ml_workflow"]
                            }
                        ]
                    },
                    {
                        "id": "deep_learning",
                        "name": "深度学习",
                        "difficulty": "advanced",
                        "modules": [
                            {
                                "id": "neural_networks",
                                "name": "神经网络基础",
                                "description": "神经网络的基本概念和结构",
                                "estimated_hours": 6,
                                "resources": ["video:neural_networks_intro", "notebook:simple_neural_network"],
                                "prerequisites": ["linear_regression", "classification"]
                            },
                            {
                                "id": "cnn",
                                "name": "卷积神经网络",
                                "description": "CNN的原理和应用",
                                "estimated_hours": 8,
                                "resources": ["video:cnn_intro", "notebook:cnn_implementation"],
                                "prerequisites": ["neural_networks"]
                            }
                        ]
                    }
                ]
            },
            {
                "id": "data_science",
                "name": "数据科学",
                "topics": [
                    {
                        "id": "data_preprocessing",
                        "name": "数据预处理",
                        "difficulty": "beginner",
                        "modules": [
                            {
                                "id": "data_cleaning",
                                "name": "数据清洗",
                                "description": "数据清洗和处理的技术和方法",
                                "estimated_hours": 3,
                                "resources": ["video:data_cleaning", "notebook:data_cleaning_techniques"],
                                "prerequisites": []
                            },
                            {
                                "id": "feature_engineering",
                                "name": "特征工程",
                                "description": "特征选择和转换的方法",
                                "estimated_hours": 4,
                                "resources": ["video:feature_engineering", "notebook:feature_engineering_examples"],
                                "prerequisites": ["data_cleaning"]
                            }
                        ]
                    },
                    {
                        "id": "data_visualization",
                        "name": "数据可视化",
                        "difficulty": "intermediate",
                        "modules": [
                            {
                                "id": "visualization_basics",
                                "name": "可视化基础",
                                "description": "数据可视化的基本原则和工具",
                                "estimated_hours": 3,
                                "resources": ["video:visualization_intro", "notebook:basic_visualization"],
                                "prerequisites": ["data_cleaning"]
                            },
                            {
                                "id": "advanced_visualization",
                                "name": "高级可视化技术",
                                "description": "交互式和高级可视化方法",
                                "estimated_hours": 5,
                                "resources": ["video:advanced_viz", "notebook:interactive_visualization"],
                                "prerequisites": ["visualization_basics"]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    with open(DEFAULT_KNOWLEDGE_GRAPH, 'w', encoding='utf-8') as f:
        json.dump(default_graph, f, ensure_ascii=False, indent=2)

def load_knowledge_graph(graph_path: str = DEFAULT_KNOWLEDGE_GRAPH) -> Dict:
    """
    加载知识图谱
    
    Args:
        graph_path: 知识图谱文件路径
        
    Returns:
        知识图谱数据
    """
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载知识图谱失败: {str(e)}")
        raise

def generate_learning_path(user_id: str, goal: str, prior_knowledge: List[str], 
                         weekly_hours: int, max_modules: int = 20) -> Dict[str, Any]:
    """
    生成个性化学习路径
    
    Args:
        user_id: 用户ID
        goal: 学习目标
        prior_knowledge: 已有知识点列表
        weekly_hours: 每周可用学习时间(小时)
        max_modules: 最大模块数量
        
    Returns:
        学习路径对象
    """
    try:
        logger.info(f"为用户 {user_id} 生成学习路径，目标: {goal}")
        
        # 加载知识图谱
        knowledge_graph = load_knowledge_graph()
        
        # 解析学习目标，获取相关领域和主题
        target_domains, target_topics = _parse_learning_goal(goal, knowledge_graph)
        
        # 获取所有模块列表
        all_modules = _get_all_modules(knowledge_graph)
        
        # 根据先验知识标记已完成的模块
        completed_modules = _get_completed_modules(prior_knowledge, all_modules)
        
        # 为目标主题构建学习路径
        modules_to_learn = _build_learning_path(target_topics, all_modules, completed_modules, max_modules)
        
        # 估算完成时间
        total_hours = sum([m.get('estimated_hours', 0) for m in modules_to_learn])
        weeks_to_complete = round(total_hours / weekly_hours, 1)
        
        # 创建学习路径对象
        path_id = f"path_{uuid.uuid4().hex[:8]}"
        learning_path = {
            "path_id": path_id,
            "user_id": user_id,
            "created_at": datetime.datetime.now().isoformat(),
            "goal": goal,
            "prior_knowledge": prior_knowledge,
            "weekly_hours": weekly_hours,
            "modules": modules_to_learn,
            "total_modules": len(modules_to_learn),
            "completed_modules": [],
            "current_module_id": modules_to_learn[0]['id'] if modules_to_learn else None,
            "estimated_total_hours": total_hours,
            "estimated_weeks": weeks_to_complete,
            "progress_percentage": 0,
            "last_updated": datetime.datetime.now().isoformat()
        }
        
        # 保存学习路径
        _save_learning_path(learning_path)
        
        return learning_path
    except Exception as e:
        logger.error(f"生成学习路径失败: {str(e)}")
        raise

def _parse_learning_goal(goal: str, knowledge_graph: Dict) -> tuple:
    """
    解析学习目标，返回相关领域和主题
    
    Args:
        goal: 学习目标
        knowledge_graph: 知识图谱
        
    Returns:
        (目标领域列表, 目标主题列表)
    """
    # 这里是简化实现，实际应用中可以使用NLP技术进行更精确的解析
    goal_lower = goal.lower()
    
    target_domains = []
    target_topics = []
    
    # 关键词映射
    keyword_mappings = {
        "机器学习": ["ml"],
        "深度学习": ["deep_learning"],
        "神经网络": ["neural_networks", "deep_learning"],
        "监督学习": ["supervised_learning"],
        "无监督学习": ["unsupervised_learning"],
        "聚类": ["clustering", "unsupervised_learning"],
        "分类": ["classification", "supervised_learning"],
        "回归": ["linear_regression", "supervised_learning"],
        "数据科学": ["data_science"],
        "数据清洗": ["data_cleaning", "data_preprocessing"],
        "特征工程": ["feature_engineering", "data_preprocessing"],
        "数据可视化": ["data_visualization"],
        "卷积神经网络": ["cnn", "deep_learning"]
    }
    
    # 检查关键词匹配
    for keyword, topics in keyword_mappings.items():
        if keyword in goal_lower:
            for topic_id in topics:
                # 添加主题
                if topic_id not in target_topics:
                    target_topics.append(topic_id)
                
                # 寻找对应的领域
                for domain in knowledge_graph.get('domains', []):
                    domain_id = domain.get('id')
                    for topic in domain.get('topics', []):
                        if topic.get('id') == topic_id and domain_id not in target_domains:
                            target_domains.append(domain_id)
    
    # 默认情况，如果没有匹配到任何关键词，返回机器学习领域
    if not target_domains:
        target_domains = ["ml"]
        target_topics = ["ml_basics"]
    
    return target_domains, target_topics

def _get_all_modules(knowledge_graph: Dict) -> List[Dict]:
    """
    从知识图谱中获取所有学习模块
    
    Args:
        knowledge_graph: 知识图谱
        
    Returns:
        所有模块列表
    """
    all_modules = []
    
    for domain in knowledge_graph.get('domains', []):
        domain_id = domain.get('id')
        domain_name = domain.get('name')
        
        for topic in domain.get('topics', []):
            topic_id = topic.get('id')
            topic_name = topic.get('name')
            topic_difficulty = topic.get('difficulty')
            
            for module in topic.get('modules', []):
                # 复制模块并添加额外信息
                module_copy = module.copy()
                module_copy['domain_id'] = domain_id
                module_copy['domain_name'] = domain_name
                module_copy['topic_id'] = topic_id
                module_copy['topic_name'] = topic_name
                module_copy['difficulty'] = topic_difficulty
                
                all_modules.append(module_copy)
    
    return all_modules

def _get_completed_modules(prior_knowledge: List[str], all_modules: List[Dict]) -> List[str]:
    """
    根据先验知识确定已完成的模块
    
    Args:
        prior_knowledge: 先验知识列表
        all_modules: 所有模块列表
        
    Returns:
        已完成的模块ID列表
    """
    completed_module_ids = []
    
    # 直接匹配模块ID
    for module in all_modules:
        module_id = module.get('id')
        if module_id in prior_knowledge:
            completed_module_ids.append(module_id)
    
    # 匹配模块名称（不区分大小写）
    for knowledge_item in prior_knowledge:
        knowledge_lower = knowledge_item.lower()
        for module in all_modules:
            if module.get('id') in completed_module_ids:
                continue
                
            if module.get('name', '').lower() == knowledge_lower:
                completed_module_ids.append(module.get('id'))
    
    return completed_module_ids

def _build_learning_path(target_topics: List[str], all_modules: List[Dict], 
                       completed_modules: List[str], max_modules: int) -> List[Dict]:
    """
    构建学习路径
    
    Args:
        target_topics: 目标主题ID列表
        all_modules: 所有模块列表
        completed_modules: 已完成的模块ID列表
        max_modules: 最大模块数量
        
    Returns:
        学习路径模块列表
    """
    # 收集所有目标主题的模块
    target_modules = []
    for module in all_modules:
        if module.get('topic_id') in target_topics:
            target_modules.append(module)
    
    # 确保添加先决条件模块
    modules_to_add = target_modules.copy()
    i = 0
    while i < len(modules_to_add):
        module = modules_to_add[i]
        for prereq_id in module.get('prerequisites', []):
            # 检查先决条件是否已经在列表中或已完成
            if prereq_id not in completed_modules and not any(m.get('id') == prereq_id for m in modules_to_add):
                # 寻找先决条件模块并添加
                for m in all_modules:
                    if m.get('id') == prereq_id:
                        modules_to_add.append(m)
                        break
        i += 1
    
    # 按难度和依赖关系排序
    difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2}
    
    # 创建依赖图
    dependency_graph = {}
    for module in modules_to_add:
        module_id = module.get('id')
        dependency_graph[module_id] = set(module.get('prerequisites', []))
    
    # 拓扑排序
    sorted_modules = []
    no_prereqs = [m for m in modules_to_add if not m.get('prerequisites', [])]
    
    # 先按难度排序无先决条件的模块
    no_prereqs.sort(key=lambda m: difficulty_order.get(m.get('difficulty'), 1))
    
    while no_prereqs:
        # 获取下一个没有依赖的模块
        next_module = no_prereqs.pop(0)
        module_id = next_module.get('id')
        
        if module_id in completed_modules:
            continue
            
        sorted_modules.append(next_module)
        
        # 移除当前模块作为其他模块的依赖
        for m in modules_to_add:
            m_id = m.get('id')
            if m_id != module_id and module_id in m.get('prerequisites', []):
                dependency_graph[m_id].remove(module_id)
                if not dependency_graph[m_id] and m not in no_prereqs and m_id not in completed_modules:
                    no_prereqs.append(m)
                    # 重新按难度排序
                    no_prereqs.sort(key=lambda x: difficulty_order.get(x.get('difficulty'), 1))
    
    # 限制模块数量
    return sorted_modules[:max_modules]

def _save_learning_path(learning_path: Dict) -> None:
    """
    保存学习路径
    
    Args:
        learning_path: 学习路径对象
    """
    path_id = learning_path.get('path_id')
    user_id = learning_path.get('user_id')
    
    # 创建用户目录
    user_dir = os.path.join(LEARNING_PATHS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    
    # 保存学习路径
    path_file = os.path.join(user_dir, f"{path_id}.json")
    with open(path_file, 'w', encoding='utf-8') as f:
        json.dump(learning_path, f, ensure_ascii=False, indent=2)

def get_user_learning_paths(user_id: str) -> List[Dict]:
    """
    获取用户的所有学习路径
    
    Args:
        user_id: 用户ID
        
    Returns:
        学习路径列表
    """
    try:
        logger.info(f"获取用户 {user_id} 的学习路径")
        
        user_dir = os.path.join(LEARNING_PATHS_DIR, user_id)
        if not os.path.exists(user_dir):
            return []
            
        paths = []
        for filename in os.listdir(user_dir):
            if filename.endswith('.json'):
                path_file = os.path.join(user_dir, filename)
                with open(path_file, 'r', encoding='utf-8') as f:
                    path = json.load(f)
                    paths.append(path)
        
        # 按创建时间排序，最新的在前
        paths.sort(key=lambda p: p.get('created_at', ''), reverse=True)
        
        return paths
    except Exception as e:
        logger.error(f"获取用户学习路径失败: {str(e)}")
        raise

def get_user_learning_path(user_id: str) -> List[Dict]:
    """
    获取用户的所有学习路径（兼容函数）
    
    Args:
        user_id: 用户ID
        
    Returns:
        学习路径列表
    """
    return get_user_learning_paths(user_id)

def get_learning_path(path_id: str) -> Dict:
    """
    获取指定的学习路径
    
    Args:
        path_id: 学习路径ID
        
    Returns:
        学习路径对象
    """
    try:
        logger.info(f"获取学习路径 {path_id}")
        
        # 遍历所有用户目录查找
        for user_id in os.listdir(LEARNING_PATHS_DIR):
            user_dir = os.path.join(LEARNING_PATHS_DIR, user_id)
            if os.path.isdir(user_dir):
                path_file = os.path.join(user_dir, f"{path_id}.json")
                if os.path.exists(path_file):
                    with open(path_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
        
        raise ValueError(f"未找到学习路径: {path_id}")
    except Exception as e:
        logger.error(f"获取学习路径失败: {str(e)}")
        raise

def update_path_progress(path_id: str, completed_module_id: Optional[str] = None, 
                        current_module_id: Optional[str] = None) -> Dict:
    """
    更新学习路径进度
    
    Args:
        path_id: 学习路径ID
        completed_module_id: 完成的模块ID
        current_module_id: 当前学习的模块ID
        
    Returns:
        更新后的学习路径
    """
    try:
        logger.info(f"更新学习路径 {path_id} 进度")
        
        # 获取学习路径
        path = get_learning_path(path_id)
        user_id = path.get('user_id')
        
        # 添加完成的模块
        if completed_module_id:
            if completed_module_id not in path.get('completed_modules', []):
                path['completed_modules'].append(completed_module_id)
        
        # 更新当前模块
        if current_module_id:
            path['current_module_id'] = current_module_id
        
        # 更新进度百分比
        total_modules = path.get('total_modules', 0)
        completed_count = len(path.get('completed_modules', []))
        
        if total_modules > 0:
            path['progress_percentage'] = round((completed_count / total_modules) * 100, 1)
        
        # 更新时间戳
        path['last_updated'] = datetime.datetime.now().isoformat()
        
        # 保存更新后的学习路径
        user_dir = os.path.join(LEARNING_PATHS_DIR, user_id)
        path_file = os.path.join(user_dir, f"{path_id}.json")
        with open(path_file, 'w', encoding='utf-8') as f:
            json.dump(path, f, ensure_ascii=False, indent=2)
        
        return path
    except Exception as e:
        logger.error(f"更新学习路径进度失败: {str(e)}")
        raise

def predict_module_mastery(user_id: str, module_id: str, weekly_hours: int, 
                         focus_level: str = "medium") -> Dict:
    """
    预测用户掌握特定模块的概率
    
    Args:
        user_id: 用户ID
        module_id: 模块ID
        weekly_hours: 每周学习时间
        focus_level: 专注度 (low, medium, high)
        
    Returns:
        预测结果
    """
    try:
        logger.info(f"预测用户 {user_id} 掌握模块 {module_id} 的概率")
        
        # 导入预测模块
        from ml_predictor import predict_learning_outcome
        
        # 加载知识图谱
        knowledge_graph = load_knowledge_graph()
        
        # 获取所有模块
        all_modules = _get_all_modules(knowledge_graph)
        
        # 查找目标模块
        target_module = None
        for module in all_modules:
            if module.get('id') == module_id:
                target_module = module
                break
        
        if not target_module:
            raise ValueError(f"未找到模块: {module_id}")
        
        # 获取用户的学习路径
        user_paths = get_user_learning_paths(user_id)
        
        # 确定用户的先验知识水平
        completed_modules = []
        for path in user_paths:
            completed_modules.extend(path.get('completed_modules', []))
        
        # 简单估计先验知识水平
        prior_knowledge_count = len(set(completed_modules))
        if prior_knowledge_count > 10:
            prior_knowledge_level = "advanced"
        elif prior_knowledge_count > 5:
            prior_knowledge_level = "intermediate"
        elif prior_knowledge_count > 0:
            prior_knowledge_level = "basic"
        else:
            prior_knowledge_level = "none"
        
        # 准备预测参数
        learning_parameters = {
            "weekly_study_hours": weekly_hours,
            "prior_knowledge_level": prior_knowledge_level,
            "focus_level": focus_level,
            "content_difficulty": target_module.get('difficulty', 'intermediate')
        }
        
        # 预测掌握概率
        prediction = predict_learning_outcome(module_id, learning_parameters, "mastery_probability")
        
        return prediction
    except Exception as e:
        logger.error(f"预测模块掌握概率失败: {str(e)}")
        raise

def predict_completion_time(user_id: str, path_id: str, weekly_hours: int) -> Dict:
    """
    预测完成学习路径的时间
    
    Args:
        user_id: 用户ID
        path_id: 学习路径ID
        weekly_hours: 每周学习时间
        
    Returns:
        预测结果
    """
    try:
        logger.info(f"预测用户 {user_id} 完成学习路径 {path_id} 的时间")
        
        # 导入预测模块
        from ml_predictor import predict_learning_outcome
        
        # 获取学习路径
        path = get_learning_path(path_id)
        
        # 检查是否是该用户的路径
        if path.get('user_id') != user_id:
            raise ValueError("学习路径不属于该用户")
        
        # 获取未完成的模块
        completed_modules = path.get('completed_modules', [])
        remaining_modules = [m for m in path.get('modules', []) if m.get('id') not in completed_modules]
        
        if not remaining_modules:
            return {
                "path_id": path_id,
                "user_id": user_id,
                "prediction_type": "completion_time",
                "remaining_modules": 0,
                "predicted_hours": 0,
                "predicted_weeks": 0,
                "confidence_interval_weeks": [0, 0],
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        # 估算总学习时间
        total_estimated_hours = sum([m.get('estimated_hours', 0) for m in remaining_modules])
        
        # 根据每周学习时间计算周数
        predicted_weeks = total_estimated_hours / weekly_hours
        
        # 95%置信区间 (简化版)
        confidence_interval = [max(0.5, predicted_weeks * 0.8), predicted_weeks * 1.2]
        
        return {
            "path_id": path_id,
            "user_id": user_id,
            "prediction_type": "completion_time",
            "remaining_modules": len(remaining_modules),
            "predicted_hours": round(total_estimated_hours, 1),
            "predicted_weeks": round(predicted_weeks, 1),
            "confidence_interval_weeks": [round(ci, 1) for ci in confidence_interval],
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"预测完成时间失败: {str(e)}")
        raise 