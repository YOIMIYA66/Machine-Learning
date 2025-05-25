#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Learning Path Planning Module
Used to generate and manage personalized learning paths.
"""

import datetime
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Learning paths storage directory
LEARNING_PATHS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'learning_paths'
)
os.makedirs(LEARNING_PATHS_DIR, exist_ok=True)

# Knowledge graph and learning resources directory
KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'knowledge'
)
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Default knowledge graph file
DEFAULT_KNOWLEDGE_GRAPH = os.path.join(KNOWLEDGE_DIR, 'knowledge_graph.json')

# If the knowledge graph file does not exist, create a simple example
if not os.path.exists(DEFAULT_KNOWLEDGE_GRAPH):
    default_graph_data = {
        "domains": [
            {
                "id": "ml",
                "name": "Machine Learning",
                "topics": [
                    {
                        "id": "ml_basics",
                        "name": "Machine Learning Basics",
                        "difficulty": "beginner",
                        "modules": [
                            {
                                "id": "ml_intro",
                                "name": "Introduction to Machine Learning",
                                "description": "Basic concepts and types of machine learning.",
                                "estimated_hours": 2,
                                "resources": ["video:intro_to_ml", "article:ml_overview"],
                                "prerequisites": []
                            },
                            {
                                "id": "ml_workflow",
                                "name": "Machine Learning Workflow",
                                "description": "End-to-end process of a machine learning project.",
                                "estimated_hours": 3,
                                "resources": ["video:ml_workflow", "notebook:ml_pipeline"],
                                "prerequisites": ["ml_intro"]
                            }
                        ]
                    },
                    {
                        "id": "supervised_learning",
                        "name": "Supervised Learning",
                        "difficulty": "intermediate",
                        "modules": [
                            {
                                "id": "linear_regression",
                                "name": "Linear Regression",
                                "description": "Principles and implementation of linear regression models.",
                                "estimated_hours": 4,
                                "resources": ["video:linear_regression", "notebook:linear_regression_impl"],
                                "prerequisites": ["ml_workflow"]
                            },
                            {
                                "id": "classification",
                                "name": "Classification Algorithms",
                                "description": "Introduction and application of common classification algorithms.",
                                "estimated_hours": 5,
                                "resources": ["video:classification_intro", "notebook:classification_algorithms"],
                                "prerequisites": ["linear_regression"]
                            }
                        ]
                    },
                    {
                        "id": "unsupervised_learning",
                        "name": "Unsupervised Learning",
                        "difficulty": "intermediate",
                        "modules": [
                            {
                                "id": "clustering",
                                "name": "Clustering Algorithms",
                                "description": "Introduction and application of common clustering algorithms.",
                                "estimated_hours": 4,
                                "resources": ["video:clustering_intro", "notebook:clustering_algorithms"],
                                "prerequisites": ["ml_workflow"]
                            }
                        ]
                    },
                    {
                        "id": "deep_learning",
                        "name": "Deep Learning",
                        "difficulty": "advanced",
                        "modules": [
                            {
                                "id": "neural_networks",
                                "name": "Neural Network Basics",
                                "description": "Basic concepts and structure of neural networks.",
                                "estimated_hours": 6,
                                "resources": ["video:neural_networks_intro", "notebook:simple_neural_network"],
                                "prerequisites": ["linear_regression", "classification"]
                            },
                            {
                                "id": "cnn",
                                "name": "Convolutional Neural Networks",
                                "description": "Principles and applications of CNNs.",
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
                "name": "Data Science",
                "topics": [
                    {
                        "id": "data_preprocessing",
                        "name": "Data Preprocessing",
                        "difficulty": "beginner",
                        "modules": [
                            {
                                "id": "data_cleaning",
                                "name": "Data Cleaning",
                                "description": "Techniques and methods for data cleaning and processing.",
                                "estimated_hours": 3,
                                "resources": ["video:data_cleaning", "notebook:data_cleaning_techniques"],
                                "prerequisites": []
                            },
                            {
                                "id": "feature_engineering",
                                "name": "Feature Engineering",
                                "description": "Methods for feature selection and transformation.",
                                "estimated_hours": 4,
                                "resources": ["video:feature_engineering", "notebook:feature_engineering_examples"],
                                "prerequisites": ["data_cleaning"]
                            }
                        ]
                    },
                    {
                        "id": "data_visualization",
                        "name": "Data Visualization",
                        "difficulty": "intermediate",
                        "modules": [
                            {
                                "id": "visualization_basics",
                                "name": "Visualization Basics",
                                "description": "Basic principles and tools for data visualization.",
                                "estimated_hours": 3,
                                "resources": ["video:visualization_intro", "notebook:basic_visualization"],
                                "prerequisites": ["data_cleaning"]
                            },
                            {
                                "id": "advanced_visualization",
                                "name": "Advanced Visualization Techniques",
                                "description": "Interactive and advanced visualization methods.",
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
    with open(DEFAULT_KNOWLEDGE_GRAPH, 'w', encoding='utf-8') as kg_file:
        json.dump(default_graph_data, kg_file, ensure_ascii=False, indent=2)
    logger.info(f"Created default knowledge graph at {DEFAULT_KNOWLEDGE_GRAPH}")

def load_knowledge_graph(graph_path: str = DEFAULT_KNOWLEDGE_GRAPH) -> Dict:
    """
    Loads the knowledge graph.

    Args:
        graph_path: Path to the knowledge graph file.

    Returns:
        Knowledge graph data.
    """
    try:
        with open(graph_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load knowledge graph: {str(e)}")
        raise


def generate_learning_path(
    user_id: str,
    goal: str,
    prior_knowledge: List[str],
    weekly_hours: int,
    max_modules: int = 20
) -> Dict[str, Any]:
    """
    Generates a personalized learning path.

    Args:
        user_id: User ID.
        goal: Learning goal.
        prior_knowledge: List of already known knowledge points/module IDs.
        weekly_hours: Available learning hours per week.
        max_modules: Maximum number of modules in the path.

    Returns:
        Learning path object.
    """
    try:
        logger.info(f"Generating learning path for user {user_id}, goal: {goal}")

        knowledge_graph = load_knowledge_graph()
        _, target_topics = _parse_learning_goal(goal, knowledge_graph)
        all_modules = _get_all_modules(knowledge_graph)
        completed_modules = _get_completed_modules(prior_knowledge, all_modules)
        modules_to_learn = _build_learning_path(
            target_topics, all_modules, completed_modules, max_modules
        )

        total_hours = sum(m.get('estimated_hours', 0) for m in modules_to_learn)
        weeks_to_complete = round(total_hours / weekly_hours, 1) if weekly_hours > 0 else float('inf')

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
            "completed_modules": [], # Initially empty
            "current_module_id": modules_to_learn[0]['id'] if modules_to_learn else None,
            "estimated_total_hours": total_hours,
            "estimated_weeks": weeks_to_complete,
            "progress_percentage": 0,
            "last_updated": datetime.datetime.now().isoformat()
        }

        _save_learning_path(learning_path)
        return learning_path
    except Exception as e:
        logger.error(f"Failed to generate learning path: {str(e)}")
        raise


def _parse_learning_goal(goal: str, knowledge_graph: Dict) -> tuple[List[str], List[str]]:
    """
    Parses the learning goal and returns relevant domains and topics.
    (Simplified implementation)

    Args:
        goal: Learning goal description.
        knowledge_graph: The knowledge graph.

    Returns:
        A tuple containing (list of target domain IDs, list of target topic IDs).
    """
    # This is a simplified implementation; NLP techniques could be used for more precision.
    goal_lower = goal.lower()
    target_domains: List[str] = []
    target_topics: List[str] = []

    # Keyword mapping (English keywords for an English codebase)
    keyword_mappings = {
        "machine learning": ["ml"],
        "deep learning": ["deep_learning"],
        "neural networks": ["neural_networks", "deep_learning"],
        "supervised learning": ["supervised_learning"],
        "unsupervised learning": ["unsupervised_learning"],
        "clustering": ["clustering", "unsupervised_learning"],
        "classification": ["classification", "supervised_learning"],
        "regression": ["linear_regression", "supervised_learning"],
        "data science": ["data_science"],
        "data cleaning": ["data_cleaning", "data_preprocessing"],
        "feature engineering": ["feature_engineering", "data_preprocessing"],
        "data visualization": ["data_visualization"],
        "convolutional neural networks": ["cnn", "deep_learning"],
        "cnn": ["cnn", "deep_learning"]
    }

    for keyword, topics in keyword_mappings.items():
        if keyword in goal_lower:
            for topic_id in topics:
                if topic_id not in target_topics:
                    target_topics.append(topic_id)
                for domain in knowledge_graph.get('domains', []):
                    domain_id = domain.get('id')
                    if any(t.get('id') == topic_id for t in domain.get('topics', [])):
                        if domain_id not in target_domains:
                            target_domains.append(domain_id)
    
    # Default if no keywords match
    if not target_domains and not target_topics: # Check both
        logger.warning(f"No specific keywords matched for goal: '{goal}'. Defaulting to ML basics.")
        target_domains = ["ml"]
        target_topics = ["ml_basics"]
    elif not target_topics and target_domains: # If only domain found, try to add all its topics or a default
        logger.warning(f"Only domain(s) {target_domains} matched for goal: '{goal}'. Adding default topics.")
        # Example: add first topic of first matched domain, or a generic one
        first_domain_topics = next((d.get('topics', []) for d in knowledge_graph.get('domains', []) if d.get('id') == target_domains[0]), [])
        if first_domain_topics:
            target_topics.append(first_domain_topics[0].get('id'))
        else: # Fallback if domain has no topics
            target_topics = ["ml_basics"]


    return target_domains, target_topics


def _get_all_modules(knowledge_graph: Dict) -> List[Dict]:
    """
    Retrieves all learning modules from the knowledge graph.

    Args:
        knowledge_graph: The knowledge graph.

    Returns:
        A list of all modules with added domain and topic info.
    """
    all_modules = []
    for domain in knowledge_graph.get('domains', []):
        for topic in domain.get('topics', []):
            for module in topic.get('modules', []):
                module_copy = module.copy()
                module_copy['domain_id'] = domain.get('id')
                module_copy['domain_name'] = domain.get('name')
                module_copy['topic_id'] = topic.get('id')
                module_copy['topic_name'] = topic.get('name')
                module_copy['difficulty'] = topic.get('difficulty')
                all_modules.append(module_copy)
    return all_modules


def _get_completed_modules(prior_knowledge: List[str], all_modules: List[Dict]) -> List[str]:
    """
    Determines completed module IDs based on prior knowledge (IDs or names).

    Args:
        prior_knowledge: List of prior knowledge items (module IDs or names).
        all_modules: List of all available modules.

    Returns:
        A list of completed module IDs.
    """
    completed_module_ids = []
    prior_knowledge_lower = [pk.lower() for pk in prior_knowledge]

    for module in all_modules:
        module_id = module.get('id', '')
        module_name_lower = module.get('name', '').lower()
        if module_id in prior_knowledge or module_name_lower in prior_knowledge_lower:
            if module_id not in completed_module_ids:
                 completed_module_ids.append(module_id)
    return completed_module_ids


def _build_learning_path(
    target_topics: List[str],
    all_modules: List[Dict],
    completed_modules: List[str],
    max_modules: int
) -> List[Dict]:
    """
    Builds the learning path by selecting and ordering modules.

    Args:
        target_topics: List of target topic IDs.
        all_modules: List of all available modules.
        completed_modules: List of IDs of completed modules.
        max_modules: Maximum number of modules for the path.

    Returns:
        A list of modules forming the learning path.
    """
    # Collect modules related to target topics
    candidate_modules = [m for m in all_modules if m.get('topic_id') in target_topics]

    # Add prerequisite modules recursively
    modules_to_add_set = {m['id'] for m in candidate_modules} # Use a set for efficient lookups
    queue = [m for m in candidate_modules if m['id'] not in completed_modules]
    
    final_module_candidates = [] # Store module objects
    visited_for_prereqs = set() # To avoid redundant processing

    idx = 0
    while idx < len(queue):
        module = queue[idx]
        idx += 1

        if module['id'] in visited_for_prereqs or module['id'] in completed_modules:
            continue
        visited_for_prereqs.add(module['id'])
        final_module_candidates.append(module)

        for prereq_id in module.get('prerequisites', []):
            if prereq_id not in completed_modules and prereq_id not in modules_to_add_set:
                prereq_module = next((m for m in all_modules if m.get('id') == prereq_id), None)
                if prereq_module:
                    modules_to_add_set.add(prereq_id)
                    queue.append(prereq_module) # Add object to queue
    
    # Filter out already completed modules from the final list
    final_modules_for_path = [m for m in final_module_candidates if m['id'] not in completed_modules]


    # Sort modules by difficulty and then topologically (simplified)
    difficulty_order = {"beginner": 0, "intermediate": 1, "advanced": 2, "unknown": 3}
    
    # Create dependency graph for topological sort
    adj = {m['id']: [] for m in final_modules_for_path}
    in_degree = {m['id']: 0 for m in final_modules_for_path}
    module_dict = {m['id']: m for m in final_modules_for_path}

    for m_obj in final_modules_for_path:
        m_id = m_obj['id']
        for prereq_id in m_obj.get('prerequisites', []):
            if prereq_id in module_dict: # Ensure prereq is part of the path modules
                adj[prereq_id].append(m_id)
                in_degree[m_id] += 1
    
    # Topological sort queue
    topo_q = [m_id for m_id in in_degree if in_degree[m_id] == 0]
    # Sort initial queue by difficulty
    topo_q.sort(key=lambda m_id: difficulty_order.get(module_dict[m_id].get('difficulty'), 3))
    
    sorted_path_modules = []
    while topo_q:
        u_id = topo_q.pop(0)
        sorted_path_modules.append(module_dict[u_id])
        
        # Sort neighbors by difficulty before adding to queue to maintain order
        sorted_neighbors = sorted(adj.get(u_id, []), key=lambda v_id: difficulty_order.get(module_dict[v_id].get('difficulty'), 3))

        for v_id in sorted_neighbors:
            in_degree[v_id] -= 1
            if in_degree[v_id] == 0:
                # Insert into topo_q while maintaining difficulty sort
                # This is a simplified insertion, for strict ordering a bisect_left might be better
                inserted = False
                v_difficulty = difficulty_order.get(module_dict[v_id].get('difficulty'),3)
                for i in range(len(topo_q)):
                    if v_difficulty < difficulty_order.get(module_dict[topo_q[i]].get('difficulty'),3):
                        topo_q.insert(i, v_id)
                        inserted = True
                        break
                if not inserted:
                    topo_q.append(v_id)

    if len(sorted_path_modules) != len(final_modules_for_path):
        logger.warning("Cycle detected or issue in topological sort. Path may be incomplete.")
        # Fallback or error handling for cycles
        # For now, just use what was sorted, or could return a subset of non-cyclic modules

    return sorted_path_modules[:max_modules]


def _save_learning_path(learning_path: Dict) -> None:
    """
    Saves the learning path to a file.

    Args:
        learning_path: Learning path object.
    """
    path_id = learning_path.get('path_id')
    user_id = learning_path.get('user_id')
    if not path_id or not user_id:
        logger.error("Cannot save learning path: path_id or user_id is missing.")
        return

    user_dir = os.path.join(LEARNING_PATHS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    path_file = os.path.join(user_dir, f"{path_id}.json")
    try:
        with open(path_file, 'w', encoding='utf-8') as f:
            json.dump(learning_path, f, ensure_ascii=False, indent=2)
        logger.info(f"Learning path {path_id} saved for user {user_id}.")
    except IOError as e:
        logger.error(f"Failed to save learning path {path_id}: {e}")


def get_user_learning_paths(user_id: str) -> List[Dict]:
    """
    Retrieves all learning paths for a user.

    Args:
        user_id: User ID.

    Returns:
        A list of learning paths.
    """
    try:
        logger.info(f"Fetching learning paths for user {user_id}")
        user_dir = os.path.join(LEARNING_PATHS_DIR, user_id)
        if not os.path.exists(user_dir):
            return []

        paths = []
        for filename in os.listdir(user_dir):
            if filename.endswith('.json'):
                path_file = os.path.join(user_dir, filename)
                try:
                    with open(path_file, 'r', encoding='utf-8') as f:
                        paths.append(json.load(f))
                except json.JSONDecodeError:
                    logger.error(f"Could not decode JSON for path file: {path_file}")
                except Exception as e:
                    logger.error(f"Error reading path file {path_file}: {e}")
        
        paths.sort(key=lambda p: p.get('created_at', ''), reverse=True) # Sort newest first
        return paths
    except Exception as e:
        logger.error(f"Failed to get user learning paths: {str(e)}")
        raise


def get_user_learning_path(user_id: str) -> List[Dict]: # Renamed for clarity in task
    """
    Retrieves all learning paths for a user (compatibility function).

    Args:
        user_id: User ID.

    Returns:
        A list of learning paths.
    """
    return get_user_learning_paths(user_id)


def get_learning_path(path_id: str) -> Optional[Dict]:
    """
    Retrieves a specific learning path by its ID.

    Args:
        path_id: Learning path ID.

    Returns:
        The learning path object, or None if not found.
    """
    try:
        logger.info(f"Fetching learning path {path_id}")
        # Iterate through all user directories to find the path_id
        for user_id in os.listdir(LEARNING_PATHS_DIR):
            user_dir = os.path.join(LEARNING_PATHS_DIR, user_id)
            if os.path.isdir(user_dir):
                path_file = os.path.join(user_dir, f"{path_id}.json")
                if os.path.exists(path_file):
                    with open(path_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
        logger.warning(f"Learning path not found: {path_id}")
        return None # Changed from raising ValueError to returning None for graceful handling
    except Exception as e:
        logger.error(f"Failed to get learning path {path_id}: {str(e)}")
        raise


def update_path_progress(
    path_id: str,
    completed_module_id: Optional[str] = None,
    current_module_id: Optional[str] = None
) -> Optional[Dict]:
    """
    Updates the progress of a learning path.

    Args:
        path_id: Learning path ID.
        completed_module_id: ID of the module just completed.
        current_module_id: ID of the module now current.

    Returns:
        The updated learning path, or None if path not found.
    """
    try:
        logger.info(f"Updating progress for learning path {path_id}")
        path = get_learning_path(path_id)
        if not path:
            logger.error(f"Path {path_id} not found for update.")
            return None
        
        user_id = path.get('user_id')

        if completed_module_id and completed_module_id not in path.get('completed_modules', []):
            path.setdefault('completed_modules', []).append(completed_module_id)

        if current_module_id:
            path['current_module_id'] = current_module_id

        total_modules = path.get('total_modules', 0)
        completed_count = len(path.get('completed_modules', []))
        if total_modules > 0:
            path['progress_percentage'] = round((completed_count / total_modules) * 100, 1)
        else:
            path['progress_percentage'] = 0


        path['last_updated'] = datetime.datetime.now().isoformat()
        _save_learning_path(path) # Re-save the updated path
        return path
    except Exception as e:
        logger.error(f"Failed to update learning path progress for {path_id}: {str(e)}")
        raise


def predict_module_mastery(
    user_id: str, module_id: str, weekly_hours: int, focus_level: str = "medium"
) -> Optional[Dict]:
    """
    Predicts the user's mastery probability for a specific module.

    Args:
        user_id: User ID.
        module_id: Module ID.
        weekly_hours: Weekly study hours.
        focus_level: User's focus level ('low', 'medium', 'high').

    Returns:
        Prediction result dictionary, or None on failure.
    """
    try:
        logger.info(f"Predicting mastery for user {user_id}, module {module_id}")
        from ml_predictor import predict_learning_outcome # Local import if ml_predictor exists

        knowledge_graph = load_knowledge_graph()
        all_modules = _get_all_modules(knowledge_graph)
        target_module = next((m for m in all_modules if m.get('id') == module_id), None)

        if not target_module:
            raise ValueError(f"Module not found: {module_id}")

        user_paths = get_user_learning_paths(user_id)
        completed_modules = [cm_id for p in user_paths for cm_id in p.get('completed_modules', [])]
        
        prior_knowledge_count = len(set(completed_modules))
        if prior_knowledge_count > 10: prior_knowledge_level = "advanced"
        elif prior_knowledge_count > 5: prior_knowledge_level = "intermediate"
        elif prior_knowledge_count > 0: prior_knowledge_level = "basic"
        else: prior_knowledge_level = "none"

        learning_params = {
            "weekly_study_hours": weekly_hours,
            "prior_knowledge_level": prior_knowledge_level,
            "focus_level": focus_level,
            "content_difficulty": target_module.get('difficulty', 'intermediate')
        }
        prediction = predict_learning_outcome(module_id, learning_params, "mastery_probability")
        return prediction
    except ModuleNotFoundError:
        logger.error("Module 'ml_predictor' not found. Cannot predict module mastery.")
        return None
    except Exception as e:
        logger.error(f"Failed to predict module mastery: {str(e)}")
        raise


def predict_completion_time(user_id: str, path_id: str, weekly_hours: int) -> Dict:
    """
    Predicts the completion time for a learning path.
    Currently uses simple estimation; intended to use ml_predictor.

    Args:
        user_id: User ID.
        path_id: Learning path ID.
        weekly_hours: Weekly study hours.

    Returns:
        Prediction result dictionary.
    """
    try:
        logger.info(f"Predicting completion time for user {user_id}, path {path_id}")
        # from ml_predictor import predict_learning_outcome # Intended usage

        path = get_learning_path(path_id)
        if not path:
            raise ValueError(f"Learning path {path_id} not found.")
        if path.get('user_id') != user_id:
            raise ValueError("Learning path does not belong to this user.")

        completed_module_ids = path.get('completed_modules', [])
        remaining_modules = [
            m for m in path.get('modules', []) if m.get('id') not in completed_module_ids
        ]

        if not remaining_modules:
            return {
                "path_id": path_id, "user_id": user_id,
                "prediction_type": "completion_time", "remaining_modules": 0,
                "predicted_hours": 0, "predicted_weeks": 0,
                "confidence_interval_weeks": [0, 0],
                "timestamp": datetime.datetime.now().isoformat()
            }

        total_estimated_hours = sum(m.get('estimated_hours', 0) for m in remaining_modules)
        predicted_weeks = (total_estimated_hours / weekly_hours) if weekly_hours > 0 else float('inf')
        
        # Simplified confidence interval
        ci_low = max(0.5, predicted_weeks * 0.8) if predicted_weeks != float('inf') else float('inf')
        ci_high = predicted_weeks * 1.2 if predicted_weeks != float('inf') else float('inf')
        
        return {
            "path_id": path_id, "user_id": user_id,
            "prediction_type": "completion_time",
            "remaining_modules": len(remaining_modules),
            "predicted_hours": round(total_estimated_hours, 1),
            "predicted_weeks": round(predicted_weeks, 1) if predicted_weeks != float('inf') else 'N/A',
            "confidence_interval_weeks": [
                round(ci_low, 1) if ci_low != float('inf') else 'N/A',
                round(ci_high, 1) if ci_high != float('inf') else 'N/A'
            ],
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to predict completion time: {str(e)}")
        raise