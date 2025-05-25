#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tech Lab Module
For model testing, performance analysis, and scenario simulation.
"""

import base64
import datetime
import json
import logging
import os
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, explained_variance_score,
                             f1_score, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Experiment results storage directory
EXPERIMENTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'experiments'
)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# Model storage directory
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'ml_models'
) # Assuming ml_models.py and its stored models are in a sibling directory
os.makedirs(MODELS_DIR, exist_ok=True)

# Uploaded data storage directory (if needed by tech_lab independently)
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'uploads'
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_available_models() -> List[Dict[str, Any]]:
    """
    Retrieves a list of available models.

    Returns:
        A list of dictionaries, each containing model information.
    """
    try:
        logger.info("Fetching list of available models.")
        models = []
        if not os.path.exists(MODELS_DIR):
            logger.warning(f"Models directory '{MODELS_DIR}' not found.")
            return []

        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pkl'):
                model_name_id = filename[:-4]
                metadata_file = os.path.join(MODELS_DIR, f"{model_name_id}_metadata.json")
                model_info = {
                    "id": model_name_id,
                    "name": model_name_id, # Default name is id
                    "type": "unknown",
                    "has_metadata": False
                }
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        model_info.update({
                            "name": metadata.get('model_name', model_name_id),
                            "type": metadata.get('model_type', 'unknown'),
                            "target": metadata.get('target_column', 'unknown'),
                            "features": metadata.get('feature_columns', []),
                            "metrics": metadata.get('metrics', {}),
                            "created_at": metadata.get('created_at', ''),
                            "has_metadata": True
                        })
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding metadata for model {model_name_id}.")
                    except Exception as e:
                        logger.error(f"Error loading metadata for {model_name_id}: {e}")
                models.append(model_info)
        return models
    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        raise


def get_model_details(model_id: str) -> Dict[str, Any]:
    """
    Retrieves detailed information for a specific model.

    Args:
        model_id: The ID of the model.

    Returns:
        A dictionary containing model details.
    """
    try:
        logger.info(f"Fetching details for model {model_id}.")
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_path):
            raise ValueError(f"Model {model_id} does not exist at {model_path}.")

        model = joblib.load(model_path)
        metadata = {}
        metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else: # Basic metadata if file is missing
            metadata = {
                "model_name": model_id,
                "model_type": type(model).__name__,
                "created_at": datetime.datetime.now().isoformat(),
                 "has_metadata": False
            }
        if "has_metadata" not in metadata : metadata["has_metadata"] = True


        details = {
            "id": model_id,
            "name": metadata.get('model_name', model_id),
            "type": metadata.get('model_type', type(model).__name__),
            "target": metadata.get('target_column', 'N/A'),
            "features": metadata.get('feature_columns', []),
            "metrics": metadata.get('metrics', {}),
            "created_at": metadata.get('created_at', 'N/A'),
            "data_path": metadata.get('data_path', 'N/A'),
            "data_rows": metadata.get('data_rows', 0),
            "data_columns": metadata.get('data_columns', 0),
            "model_params": _get_model_params(model),
            "has_metadata": metadata.get("has_metadata")
        }
        return details
    except Exception as e:
        logger.error(f"Failed to get model details for {model_id}: {str(e)}")
        raise


def _get_model_params(model: Any) -> Dict[str, Any]:
    """
    Extracts parameters from a scikit-learn model.

    Args:
        model: The trained model object.

    Returns:
        A dictionary of model parameters.
    """
    params = {}
    if hasattr(model, 'get_params'):
        try:
            model_params = model.get_params()
            for key, value in model_params.items():
                if isinstance(value, (int, float, str, bool)) or value is None:
                    params[key] = value
                else:
                    params[key] = str(value) # Convert complex objects to string
        except Exception as e:
            logger.warning(f"Could not retrieve model parameters: {e}")
            
    if hasattr(model, 'feature_importances_'):
        params['feature_importances_'] = model.feature_importances_.tolist()
    if hasattr(model, 'coef_'):
        params['coef_'] = model.coef_.tolist()
    return params


def create_experiment(
    name: str, description: str, model_id: str,
    experiment_type: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Creates an experiment record.

    Args:
        name: Experiment name.
        description: Experiment description.
        model_id: Model ID to be used in the experiment.
        experiment_type: Type of experiment (e.g., "prediction", "analysis", "comparison").
        config: Experiment-specific configuration.

    Returns:
        The created experiment information.
    """
    try:
        logger.info(f"Creating experiment: {name}, Type: {experiment_type}")
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_path) and experiment_type != "comparison": # Comparison might not need a single model_id
             if not (experiment_type == "comparison" and not model_id): # Allow empty model_id for comparison
                raise ValueError(f"Model {model_id} does not exist.")

        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        experiment = {
            "id": experiment_id, "name": name, "description": description,
            "model_id": model_id, "type": experiment_type, "config": config,
            "created_at": datetime.datetime.now().isoformat(),
            "status": "created", "results": None
        }

        experiment_file = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        with open(experiment_file, 'w', encoding='utf-8') as f:
            json.dump(experiment, f, ensure_ascii=False, indent=2)
        logger.info(f"Experiment {experiment_id} created successfully.")
        return experiment
    except Exception as e:
        logger.error(f"Failed to create experiment: {str(e)}")
        raise


def run_experiment(experiment_id: str, data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs a previously created experiment.

    Args:
        experiment_id: The ID of the experiment to run.
        data_path: Optional path to data file if not in config or needs override.

    Returns:
        The experiment results.
    """
    try:
        logger.info(f"Running experiment: {experiment_id}")
        experiment_file = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        if not os.path.exists(experiment_file):
            raise ValueError(f"Experiment {experiment_id} does not exist.")

        with open(experiment_file, 'r', encoding='utf-8') as f:
            experiment = json.load(f)

        experiment['status'] = "running"
        with open(experiment_file, 'w', encoding='utf-8') as f: # Update status
            json.dump(experiment, f, ensure_ascii=False, indent=2)

        exp_type = experiment.get('type')
        config = experiment.get('config', {})
        model_id = experiment.get('model_id')
        results = {}

        if exp_type == "prediction":
            results = _run_prediction_experiment(model_id, config, data_path)
        elif exp_type == "analysis":
            results = _run_analysis_experiment(model_id, config, data_path)
        elif exp_type == "comparison":
            results = _run_comparison_experiment(config, data_path) # model_id might be a list in config
        else:
            raise ValueError(f"Unsupported experiment type: {exp_type}")

        experiment['results'] = results
        experiment['status'] = "completed"
        experiment['completed_at'] = datetime.datetime.now().isoformat()
        with open(experiment_file, 'w', encoding='utf-8') as f: # Save final state
            json.dump(experiment, f, ensure_ascii=False, indent=2)
        logger.info(f"Experiment {experiment_id} completed.")
        return experiment
    except Exception as e:
        logger.error(f"Failed to run experiment {experiment_id}: {str(e)}")
        # Update status to failed
        try:
            if os.path.exists(experiment_file):
                with open(experiment_file, 'r', encoding='utf-8') as f_read:
                    exp_fail_data = json.load(f_read)
                exp_fail_data['status'] = "failed"
                exp_fail_data['error'] = str(e)
                with open(experiment_file, 'w', encoding='utf-8') as f_write:
                    json.dump(exp_fail_data, f_write, ensure_ascii=False, indent=2)
        except Exception as e_save_fail:
            logger.error(f"Additionally failed to save error status for experiment {experiment_id}: {e_save_fail}")
        raise


def _run_prediction_experiment(
    model_id: str, config: Dict[str, Any], data_path: Optional[str] = None
) -> Dict[str, Any]:
    """Helper to run prediction experiments."""
    logger.info(f"Running prediction experiment for model {model_id}.")
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    model = joblib.load(model_path)
    metadata = {}
    metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    # Determine data source
    data_path = data_path or metadata.get('data_path')
    if not data_path or not os.path.exists(data_path):
        raise ValueError("Data file not specified and no valid data path in model metadata.")

    df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
    target_column = metadata.get('target_column')
    if not target_column or target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    feature_columns = metadata.get('feature_columns', [col for col in df.columns if col != target_column])
    X = df[feature_columns]
    y = df[target_column]

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') and config.get('use_probability', False) else None
    
    metrics = {}
    model_type_hint = metadata.get('model_type', '').lower()
    is_classifier = 'classifier' in model_type_hint or hasattr(model, 'predict_proba')

    if is_classifier:
        metrics['accuracy'] = accuracy_score(y, y_pred)
        metrics['precision'] = precision_score(y, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
        metrics['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
        if y_pred_proba is not None and len(np.unique(y)) == 2 and y_pred_proba.shape[1] == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba[:, 1])
    else: # Regressor
        metrics['mse'] = mean_squared_error(y, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y, y_pred)
        metrics['explained_variance'] = explained_variance_score(y, y_pred)

    visualizations = {}
    fig_pred_actual = plt.figure(figsize=(10,6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('Actual Values'); plt.ylabel('Predicted Values'); plt.title('Predicted vs. Actual Values')
    plt.grid(True)
    visualizations['pred_vs_actual'] = _save_plot_to_base64(fig_pred_actual)
    
    return {
        "metrics": metrics, "visualizations": visualizations,
        "data_summary": {"total_samples": len(X), "feature_names": feature_columns, "target_column": target_column}
    }


def _run_analysis_experiment(
    model_id: str, config: Dict[str, Any], data_path: Optional[str] = None
) -> Dict[str, Any]:
    """Helper to run analysis experiments (e.g., feature importance)."""
    logger.info(f"Running analysis experiment for model {model_id}.")
    model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
    model = joblib.load(model_path)
    metadata = {}
    metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    analysis_type = config.get('analysis_type', 'feature_importance')
    results = {"analysis_type": analysis_type, "model_id": model_id, "model_type": metadata.get('model_type', type(model).__name__)}

    if analysis_type == "feature_importance":
        feature_names = metadata.get('feature_columns', [])
        importances = None
        if hasattr(model, 'feature_importances_'): importances = model.feature_importances_
        elif hasattr(model, 'coef_'): 
            importances = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)

        if importances is not None and len(feature_names) == len(importances):
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
            fig_imp = plt.figure(figsize=(10,8))
            plt.barh(imp_df['feature'][:15], imp_df['importance'][:15]) # Top 15 features
            plt.xlabel('Importance'); plt.ylabel('Feature'); plt.title('Feature Importance')
            plt.gca().invert_yaxis() # Display most important at top
            plt.tight_layout()
            results['feature_importance_data'] = imp_df.to_dict(orient='records')
            results['feature_importance_plot'] = _save_plot_to_base64(fig_imp)
        else:
            results['error'] = "Model does not support feature importance or feature names mismatch."
    # Add other analysis types like partial dependence if needed
    elif analysis_type == "partial_dependence":
        data_path = data_path or metadata.get('data_path')
        if not data_path or not os.path.exists(data_path):
            raise ValueError("Data file not specified for partial dependence plot.")
        
        df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
        target_column = metadata.get('target_column')
        if not target_column or target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")
            
        feature_columns = metadata.get('feature_columns', [c for c in df.columns if c != target_column])
        target_feature = config.get('target_feature', feature_columns[0] if feature_columns else None)
        if not target_feature or target_feature not in feature_columns:
            raise ValueError(f"Target feature '{target_feature}' for PDP not valid.")
            
        X_pdp = df[feature_columns]
        # This is a simplified PDP calculation; proper libraries (like sklearn.inspection) are preferred
        feature_values = np.linspace(X_pdp[target_feature].min(), X_pdp[target_feature].max(), num=config.get('num_points', 20))
        mean_predictions = []
        for value in feature_values:
            X_temp = X_pdp.copy()
            X_temp[target_feature] = value
            predictions = model.predict(X_temp) # Assuming model can handle DataFrame
            mean_predictions.append(np.mean(predictions))
            
        fig_pdp = plt.figure(figsize=(10, 6))
        plt.plot(feature_values, mean_predictions)
        plt.xlabel(target_feature); plt.ylabel('Average Predicted Value'); plt.title(f'Partial Dependence Plot for {target_feature}')
        plt.grid(True); plt.tight_layout()
        results['partial_dependence_plot'] = _save_plot_to_base64(fig_pdp)
        results['partial_dependence_data'] = {'feature': target_feature, 'values': feature_values.tolist(), 'mean_predictions': mean_predictions}

    else:
        results['error'] = f"Unsupported analysis type: {analysis_type}"
    return results


def _run_comparison_experiment(
    config: Dict[str, Any], data_path: Optional[str] = None
) -> Dict[str, Any]:
    """Helper to run model comparison experiments."""
    model_ids = config.get('model_ids', [])
    if not model_ids: raise ValueError("No models specified for comparison.")
    if not data_path: raise ValueError("Data path required for model comparison.")

    logger.info(f"Running comparison experiment for models: {model_ids} on data: {data_path}")
    df = pd.read_csv(data_path) if data_path.endswith('.csv') else pd.read_excel(data_path)
    target_column = config.get('target_column')
    if not target_column or target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in data for comparison.")
    
    feature_columns = config.get('feature_columns', [c for c in df.columns if c != target_column])
    X_data = df[feature_columns]
    y_data = df[target_column]
    comparison_metrics = []

    for model_id in model_ids:
        try:
            model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
            if not os.path.exists(model_path):
                comparison_metrics.append({"model_id": model_id, "error": "Model file not found."})
                continue
            model = joblib.load(model_path)
            metadata_file = os.path.join(MODELS_DIR, f"{model_id}_metadata.json")
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f: metadata = json.load(f)
            
            y_pred = model.predict(X_data) # Assuming X_data is preprocessed if necessary
            metrics = {}
            model_type_hint = metadata.get('model_type', '').lower()
            is_classifier = 'classifier' in model_type_hint or hasattr(model, 'predict_proba')

            if is_classifier:
                metrics.update({
                    'accuracy': accuracy_score(y_data, y_pred),
                    'f1': f1_score(y_data, y_pred, average='weighted', zero_division=0)
                })
            else:
                metrics.update({
                    'mse': mean_squared_error(y_data, y_pred),
                    'r2': r2_score(y_data, y_pred)
                })
            comparison_metrics.append({
                "model_id": model_id, "model_name": metadata.get('model_name', model_id),
                "model_type": metadata.get('model_type', 'unknown'), "metrics": metrics
            })
        except Exception as e:
            comparison_metrics.append({"model_id": model_id, "error": str(e)})
    
    # Basic visualization (can be expanded)
    visualizations = {}
    if comparison_metrics:
        df_comp = pd.DataFrame([
            {'model': r.get('model_name', r['model_id']), **r.get('metrics', {})}
            for r in comparison_metrics if 'error' not in r
        ])
        if not df_comp.empty:
            primary_metric = 'f1' if 'f1' in df_comp.columns else 'r2' if 'r2' in df_comp.columns else None
            if primary_metric:
                df_comp = df_comp.sort_values(by=primary_metric, ascending=False)
                fig_comp = plt.figure(figsize=(10,6))
                sns.barplot(x='model', y=primary_metric, data=df_comp, palette='viridis', ax=fig_comp.gca())
                plt.title(f'Model Comparison by {primary_metric.upper()}')
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                visualizations['comparison_plot'] = _save_plot_to_base64(fig_comp)
    
    return {
        "comparison_summary": comparison_metrics,
        "visualizations": visualizations,
        "data_info": {"samples": len(X_data), "features": feature_columns, "target": target_column}
    }


def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """
    Retrieves details for a specific experiment.

    Args:
        experiment_id: The ID of the experiment.

    Returns:
        A dictionary containing experiment details.
    """
    try:
        logger.info(f"Fetching details for experiment {experiment_id}")
        experiment_file = os.path.join(EXPERIMENTS_DIR, f"{experiment_id}.json")
        if not os.path.exists(experiment_file):
            raise ValueError(f"Experiment {experiment_id} does not exist.")
        with open(experiment_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to get experiment details for {experiment_id}: {str(e)}")
        raise


def get_all_experiments() -> List[Dict[str, Any]]:
    """
    Retrieves a list of all experiments.

    Returns:
        A list of dictionaries, each summarizing an experiment.
    """
    try:
        logger.info("Fetching all experiments.")
        experiments = []
        if not os.path.exists(EXPERIMENTS_DIR):
            logger.warning(f"Experiments directory '{EXPERIMENTS_DIR}' not found.")
            return []

        for filename in os.listdir(EXPERIMENTS_DIR):
            if filename.endswith('.json'):
                experiment_file = os.path.join(EXPERIMENTS_DIR, filename)
                try:
                    with open(experiment_file, 'r', encoding='utf-8') as f:
                        exp_data = json.load(f)
                    experiments.append({
                        "id": exp_data.get('id'), "name": exp_data.get('name'),
                        "description": exp_data.get('description'),
                        "model_id": exp_data.get('model_id'), "type": exp_data.get('type'),
                        "created_at": exp_data.get('created_at'), "status": exp_data.get('status'),
                        "has_results": exp_data.get('results') is not None
                    })
                except json.JSONDecodeError:
                     logger.error(f"Error decoding JSON for experiment file: {filename}")
                except Exception as e:
                    logger.error(f"Error reading experiment file {filename}: {e}")
        
        experiments.sort(key=lambda p: p.get('created_at', ''), reverse=True) # Newest first
        return experiments
    except Exception as e:
        logger.error(f"Failed to get all experiments: {str(e)}")
        raise
\ No newline at end of file