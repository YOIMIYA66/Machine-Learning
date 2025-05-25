# ml_agents.py
import base64
import colorsys
import json
import os
import traceback
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import PromptTemplate
# Pydantic V1 is used by Langchain for now
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import Tool
from langchain.tools import StructuredTool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score)
from sklearn.preprocessing import MinMaxScaler

from baidu_llm import BaiduErnieLLM
# Import specific model functions from ml_models
from ml_models import (
    auto_model_selection as actual_auto_select,
    compare_models as actual_compare_models, create_ensemble_model,
    explain_model_prediction as actual_explain_prediction,
    list_available_models as actual_list_models, list_model_versions,
    load_model, predict as actual_predict,
    save_model_with_version,
    select_model_for_task as actual_select_model,
    train_model as actual_train_model)

# --- Input Schemas for Tools ---
class TrainModelInput(BaseModel):
    model_type: str = Field(
        ...,
        description=(
            "Type of model, e.g., 'linear_regression', 'logistic_regression', "
            "'decision_tree', 'random_forest_classifier', etc."
        )
    )
    data_path: str = Field(
        ...,
        description="Path to the data file, supports CSV and Excel formats."
    )
    target_column: str = Field(..., description="Name of the target column.")
    model_name: Optional[str] = Field(
        None,
        description="Name to save the model as. Auto-generated if not provided."
    )
    categorical_columns: Optional[List[str]] = Field(
        None,
        description="List of categorical feature names."
    )
    numerical_columns: Optional[List[str]] = Field(
        None,
        description="List of numerical feature names."
    )

class PredictInput(BaseModel):
    model_name: str = Field(..., description="Name of the model to use.")
    input_data: Dict[str, Any] = Field(
        ...,
        description="Input data as a dictionary mapping feature names to values."
    )

class RecommendModelInput(BaseModel):
    task_description: str = Field(
        ...,
        description="Description of the task, e.g., 'predict house prices' or 'classify spam emails'."
    )

class DataAnalysisInput(BaseModel):
    file_path: str = Field(
        ...,
        description="Path to the data file, supports CSV and Excel formats."
    )
    target_column: Optional[str] = Field(
        None,
        description="Target column name, for analyzing feature relevance to the target."
    )
    analysis_type: Optional[str] = Field(
        None,
        description="Type of analysis, e.g., 'statistics', 'feature_relevance'."
    )

class EvaluateModelInput(BaseModel):
    model_name: str = Field(..., description="Name of the model to evaluate.")
    test_data_path: str = Field(..., description="Path to the test data file.")
    target_column: str = Field(..., description="Name of the target column.")

class EnsembleModelInput(BaseModel):
    base_models: List[str] = Field(
        ...,
        description="List of base model names for ensembling."
    )
    ensemble_type: str = Field(
        "voting",
        description="Type of ensemble, e.g., 'voting', 'stacking', 'bagging'."
    )
    weights: Optional[List[float]] = Field(
        None,
        description="Weights for base models, used only for 'voting' ensemble."
    )
    save_name: Optional[str] = Field(
        None,
        description="Name to save the ensemble model as."
    )

class AutoSelectModelInput(BaseModel):
    data_path: str = Field(..., description="Path to the data file.")
    target_column: str = Field(..., description="Name of the target column.")
    categorical_columns: Optional[List[str]] = Field(
        None,
        description="List of categorical feature names."
    )
    numerical_columns: Optional[List[str]] = Field(
        None,
        description="List of numerical feature names."
    )

class ExplainPredictionInput(BaseModel):
    model_name: str = Field(..., description="Name of the model.")
    input_data: Dict[str, Any] = Field(
        ...,
        description="Input data as a dictionary mapping feature names to values."
    )

class CompareModelsInput(BaseModel):
    model_names: List[str] = Field(
        ...,
        description="List of model names to compare."
    )
    test_data_path: str = Field(..., description="Path to the test data file.")
    target_column: str = Field(..., description="Name of the target column.")

class VersionModelInput(BaseModel):
    model_name: str = Field(..., description="Name of the model to version.")
    version: Optional[str] = Field(
        None,
        description="Version number. Uses timestamp if not provided."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for the version."
    )

class ListModelVersionsInput(BaseModel):
    model_name: str = Field(..., description="Name of the model.")


# --- Visualization Helper Functions ---
def generate_gradient_colors(n_colors: int) -> List[Tuple[float, float, float, float]]:
    """Generates a list of gradient colors for charts."""
    colors = []
    for i in range(n_colors):
        # Gradient from blue-purple to sky blue
        hue = 0.6 + (0.2 * i / max(1, n_colors - 1))  # Hue from 0.6 (blue-purple) to 0.8 (sky blue)
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgba = (rgb[0], rgb[1], rgb[2], 0.7)  # RGBA format
        colors.append(rgba)
    return colors

def _save_plot_to_base64(plt_figure) -> str:
    """Saves the current matplotlib figure to a base64 encoded string."""
    buffer = BytesIO()
    plt_figure.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt_figure.close()  # Close the figure to free memory
    return base64.b64encode(image_png).decode('utf-8')

def generate_visualization(
    data_type: str,
    labels: List[Any],
    values: List[Any],
    title: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Generates a visualization chart and returns its base64 encoding and data."""
    fig = plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    options = options or {}
    num_colors = len(values) if isinstance(values, list) and len(values) > 0 else 5
    colors = generate_gradient_colors(num_colors)

    if data_type == 'bar':
        plt.bar(labels, values, color=colors)
        plt.ylabel('Value')
    elif data_type == 'line':
        plt.plot(labels, values, marker='o', color=colors[0] if colors else '#4F46E5', linewidth=2)
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    elif data_type == 'pie':
        if not values or sum(values) == 0: # Avoid error for empty or all-zero pie charts
             plt.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center')
        else:
            plt.pie(
                values, labels=labels, autopct='%1.1f%%',
                startangle=90, shadow=True, colors=colors
            )
        plt.axis('equal')
    elif data_type == 'scatter':
        plt.scatter(
            labels, values, color=colors[0] if colors else '#4F46E5',
            alpha=0.7, s=options.get('point_size', 70)
        )
        plt.ylabel('Value')
    elif data_type == 'heatmap':
        if 'matrix' in options and options['matrix'] is not None:
            sns.heatmap(
                options['matrix'], annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=options.get('y_labels', labels)
            )
        else:
            plt.text(0.5, 0.5, 'Matrix data not provided for heatmap', ha='center', va='center')
    elif data_type == 'radar':
        if labels and values:
            theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            plot_values = np.array(values)
            ax = plt.subplot(111, polar=True)
            ax.fill(theta, plot_values, color=colors[0] if colors else '#4F46E5', alpha=0.25)
            ax.plot(theta, plot_values, color=colors[0] if colors else '#4F46E5', linewidth=2)
            ax.set_xticks(theta)
            ax.set_xticklabels(labels)
            ax.grid(True)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for radar chart', ha='center', va='center')

    elif data_type == 'bubble':
        sizes = options.get('sizes', [50] * len(values))
        plt.scatter(labels, values, s=sizes, alpha=0.6, c=colors)
        plt.ylabel('Value')

    plt.title(title or 'Data Visualization')
    if data_type not in ['pie', 'radar', 'heatmap'] and labels:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    image_base64 = _save_plot_to_base64(fig)

    return {
        'type': data_type,
        'labels': labels,
        'values': values,
        'title': title,
        'options': options,
        'image': image_base64
    }

def visualize_feature_importance(
    model: Any, feature_names: List[str], n_features: int = 10
) -> Optional[Dict[str, Any]]:
    """Visualizes feature importance if the model supports it."""
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(n_features, len(feature_names))

    fig = plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    plt.title('Feature Importance')
    colors = generate_gradient_colors(top_n)
    plt.bar(
        range(top_n), importances[indices][:top_n],
        color=colors, align='center'
    )
    plt.xticks(
        range(top_n),
        [feature_names[i] for i in indices][:top_n],
        rotation=45, ha='right'
    )
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    image_base64 = _save_plot_to_base64(fig)

    table_data = {
        'columns': ['Feature', 'Importance'],
        'data': [
            [feature_names[i], float(importances[indices][j])]
            for j, i in enumerate(indices[:top_n])
        ]
    }
    return {
        'type': 'bar',
        'labels': [feature_names[i] for i in indices][:top_n],
        'values': importances[indices][:top_n].tolist(),
        'title': 'Feature Importance',
        'image': image_base64,
        'table_data': table_data
    }

def visualize_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Visualizes a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    image_base64 = _save_plot_to_base64(fig)

    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)

    metrics_data = []
    if class_names is None: # Handle case where class_names might not be available
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    for i, class_name in enumerate(class_names):
        metrics_data.append([
            class_name,
            int(np.sum(cm[i, :])),  # Total samples for this class
            int(cm[i, i]),          # Correctly predicted
            float(precision[i]) if i < len(precision) else 0.0, # Precision
            float(recall[i]) if i < len(recall) else 0.0        # Recall
        ])
    table_data = {
        'columns': ['Class', 'Samples', 'Correct', 'Precision', 'Recall'],
        'data': metrics_data
    }
    return {
        'type': 'confusion_matrix',
        'matrix': cm.tolist(),
        'class_names': class_names,
        'image': image_base64,
        'table_data': table_data
    }

def visualize_metrics(metrics_dict: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Visualizes model evaluation metrics as a bar chart."""
    numeric_metrics = {
        k: v for k, v in metrics_dict.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not numeric_metrics:
        return None

    fig = plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    colors = generate_gradient_colors(len(numeric_metrics))
    plt.bar(numeric_metrics.keys(), numeric_metrics.values(), color=colors)
    plt.title('Model Evaluation Metrics')
    y_max_limit = 1.0
    if numeric_metrics: # Check if dictionary is not empty
        y_max_limit = max(1.0, max(numeric_metrics.values()) * 1.1)

    plt.ylim(0, y_max_limit)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    image_base64 = _save_plot_to_base64(fig)

    table_data = {
        'columns': ['Metric', 'Value'],
        'data': [[k, v] for k, v in numeric_metrics.items()]
    }
    return {
        'type': 'bar',
        'labels': list(numeric_metrics.keys()),
        'values': list(numeric_metrics.values()),
        'title': 'Model Evaluation Metrics',
        'image': image_base64,
        'table_data': table_data
    }

def visualize_clusters(
    X: np.ndarray, labels: np.ndarray, method: str = 'pca'
) -> Dict[str, Any]:
    """Visualizes clustering results using PCA or t-SNE for dimensionality reduction."""
    if X.shape[1] > 2:
        reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    else:
        X_2d = X

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = generate_gradient_colors(n_clusters)

    fig = plt.figure(figsize=(10, 8))
    for i, label_val in enumerate(unique_labels):
        mask = labels == label_val
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=[colors[i % len(colors)]], # Use modulo for color safety
            label=f'Cluster {label_val}', alpha=0.7, s=80, edgecolors='w'
        )
    plt.legend()
    plt.title('Cluster Visualization')
    plt.xlabel('Component 1 (PCA)' if method == 'pca' else 't-SNE Dimension 1')
    plt.ylabel('Component 2 (PCA)' if method == 'pca' else 't-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    image_base64 = _save_plot_to_base64(fig)

    cluster_counts = np.bincount(labels.astype(int))
    table_data = {
        'columns': ['Cluster', 'Sample Count', 'Proportion'],
        'data': [
            [f'Cluster {i}', int(count), float(count) / len(labels)]
            for i, count in enumerate(cluster_counts)
        ]
    }
    return {
        'type': 'scatter',
        'image': image_base64,
        'table_data': table_data,
        'method': method,
        'n_clusters': n_clusters
    }

def generate_data_table(
    data: Any, columns: Optional[List[str]] = None, max_rows: int = 100
) -> Optional[Dict[str, List[Any]]]:
    """Converts various data types to a table format (list of lists)."""
    df = None
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, dict) and all(isinstance(data[k], (list, np.ndarray)) for k in data):
        df = pd.DataFrame(data)
    else:
        try:
            df = pd.DataFrame(data)
        except ValueError: # More specific exception
            return None

    if df.empty:
        return {'columns': [], 'data': []}


    df = df.head(max_rows)
    if columns:
        df = df[[col for col in columns if col in df.columns]] # Ensure columns exist

    return {'columns': df.columns.tolist(), 'data': df.values.tolist()}

def visualize_feature_importance_radar(
    feature_importance: Dict[str, float], title: str = 'Feature Importance Radar Chart'
) -> Dict[str, Any]:
    """Generates a radar chart for feature importance."""
    features = list(feature_importance.keys())
    values = np.array(list(feature_importance.values()))

    # Normalize values to 0-1 range
    if np.any(values < 0): # Scale to 0-1 if negative values exist
        values = (values - np.min(values)) / (np.max(values) - np.min(values))
    elif np.max(values) > 0: # Normalize if positive
        values = values / np.max(values)
    # else: values are all zero or empty, chart will be blank or have zero radius

    fig = plt.figure(figsize=(10, 8))
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False)
    
    # Close the radar chart
    plot_values = np.concatenate((values, [values[0]]))
    plot_angles = np.concatenate((angles, [angles[0]]))
    
    ax = plt.subplot(111, polar=True)
    ax.fill(plot_angles, plot_values, color=generate_gradient_colors(1)[0], alpha=0.25) # Use first gradient color
    ax.plot(plot_angles, plot_values, 'o-', color=generate_gradient_colors(1)[0], linewidth=2)
    ax.set_xticks(angles)
    ax.set_xticklabels(features)
    ax.set_yticks(np.linspace(0, 1, 6)) # Y-ticks from 0 to 1
    plt.title(title)
    image_base64 = _save_plot_to_base64(fig)

    return {
        'type': 'radar',
        'labels': features,
        'values': values.tolist(),
        'title': title,
        'image': image_base64
    }

def visualize_model_comparison(
    comparison_results: Dict[str, Any], metric: str = 'auto'
) -> Dict[str, List[Any]]:
    """Visualizes model comparison results."""
    models = comparison_results.get('models', [])
    classifiers = [m for m in models if m.get('is_classifier', False) and 'error' not in m]
    regressors = [m for m in models if not m.get('is_classifier', False) and 'error' not in m]
    visualizations = []
    tables = []

    # Process classifiers
    if classifiers:
        model_names = [m['model_name'] for m in classifiers]
        classifier_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric_name in classifier_metrics:
            values = [m['metrics'].get(metric_name, 0) for m in classifiers]
            vis_data = generate_visualization(
                'bar', model_names, values,
                title=f'Classifier Comparison - {metric_name.capitalize()}'
            )
            visualizations.append(vis_data)
        tables.append({
            'title': 'Classifier Performance Comparison',
            'data': generate_data_table(
                [{'Model': m['model_name'], **m['metrics']} for m in classifiers]
            )
        })

    # Process regressors
    if regressors:
        model_names = [m['model_name'] for m in regressors]
        regressor_metrics = ['mse', 'rmse', 'mae', 'r2']
        for metric_name in regressor_metrics:
            values = [m['metrics'].get(metric_name, 0) for m in regressors]
            title_suffix = " (Lower is Better)" if metric_name != 'r2' else ""
            vis_data = generate_visualization(
                'bar', model_names, values,
                title=f'Regressor Comparison - {metric_name.upper()}{title_suffix}'
            )
            visualizations.append(vis_data)
        tables.append({
            'title': 'Regressor Performance Comparison',
            'data': generate_data_table(
                 [{'Model': m['model_name'], **m['metrics']} for m in regressors]
            )
        })

    # Best model summary table
    if comparison_results.get('best_classifier') or comparison_results.get('best_regressor'):
        best_model_info = []
        if comparison_results.get('best_classifier'):
            best_model_info.append({
                'Type': 'Best Classifier',
                'Name': comparison_results['best_classifier'],
                'Metric': 'F1 Score (higher is better)'
            })
        if comparison_results.get('best_regressor'):
             best_model_info.append({
                'Type': 'Best Regressor',
                'Name': comparison_results['best_regressor'],
                'Metric': 'R² Score (higher is better)'
            })
        tables.append({
            'title': 'Best Performing Models',
            'data': generate_data_table(best_model_info)
        })
    return {'visualizations': visualizations, 'tables': tables}

def visualize_model_explanation(
    explanation_result: Dict[str, Any]
) -> Dict[str, List[Any]]:
    """Visualizes model explanation results."""
    visualizations = []
    tables = []
    feature_importance = explanation_result.get('feature_importance', {})

    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = sorted_features[:10]
        feature_names = [item[0] for item in top_features]
        importance_values = [item[1] for item in top_features]

        vis_data_bar = generate_visualization(
            'bar', feature_names, importance_values, title='Feature Importance'
        )
        # Custom color for negative values (optional, if generate_visualization doesn't handle)
        # ...
        visualizations.append(vis_data_bar)

        if len(feature_importance) >= 3:
            radar_abs_importance = {k: abs(v) for k,v in feature_importance.items()}
            vis_data_radar = visualize_feature_importance_radar(
                radar_abs_importance, title='Feature Importance Strength (Radar)'
            )
            visualizations.append(vis_data_radar)

        tables.append({
            'title': 'Feature Importance',
            'data': generate_data_table(
                [{'Feature': name, 'Importance': val} for name, val in sorted_features]
            )
        })

    feature_contributions = explanation_result.get('feature_contributions', [])
    if feature_contributions:
        top_contributions = feature_contributions[:10]
        feature_names_contrib = [item['feature'] for item in top_contributions]
        contribution_values = [item['contribution'] for item in top_contributions]
        vis_data_contrib = generate_visualization(
            'bar', feature_names_contrib, contribution_values, title='Feature Contributions'
        )
        # Custom color for negative contributions ...
        visualizations.append(vis_data_contrib)
        tables.append({
            'title': 'Feature Contributions',
            'data': generate_data_table(feature_contributions)
        })

    prediction = explanation_result.get('prediction', [])
    if prediction:
        tables.append({
            'title': 'Prediction Result',
            'data': generate_data_table([{'Prediction': p} for p in prediction])
        })
    return {'visualizations': visualizations, 'tables': tables}


# --- Tool Definitions ---
def create_ml_tools() -> List[Tool]:
    """Creates a list of structured tools for the ML agent."""

    def _train_model(
        model_type: str, data_path: str, target_column: str,
        model_name: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ) -> str:
        """Trains a machine learning model."""
        try:
            processed_data_path = data_path
            if data_path.lower().endswith(('.xls', '.xlsx')):
                try:
                    df = pd.read_excel(data_path)
                    temp_csv_filename = (
                        f"{os.path.splitext(os.path.basename(data_path))[0]}"
                        f"_temp_{uuid.uuid4().hex}.csv"
                    )
                    # Assume 'uploads' dir exists or is created by another process if needed
                    # For robustness, ensure 'uploads' dir is available or use tempfile module
                    uploads_dir = os.path.join(os.getcwd(), 'uploads')
                    if not os.path.exists(uploads_dir):
                        os.makedirs(uploads_dir, exist_ok=True)
                    processed_data_path = os.path.join(uploads_dir, temp_csv_filename)
                    df.to_csv(processed_data_path, index=False)
                    print(f"Converted Excel file {data_path} to temporary CSV: {processed_data_path}")
                except Exception as e:
                    return json.dumps({
                        "text": f"❌ Error converting Excel to CSV: {str(e)}",
                        "error": f"Error converting Excel to CSV: {str(e)}"
                    })

            result = actual_train_model(
                model_type=model_type, data=processed_data_path,
                target_column=target_column, model_name=model_name,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns
            )
            
            if processed_data_path != data_path and os.path.exists(processed_data_path):
                 try:
                     os.remove(processed_data_path)
                     print(f"Cleaned up temporary CSV file: {processed_data_path}")
                 except Exception as e:
                     print(f"Warning: Failed to clean up temporary CSV: {processed_data_path}: {e}")

            metrics_json = json.dumps(result.get('metrics', {}), ensure_ascii=False)
            message = (
                f"✅ Successfully trained {result.get('model_type', model_type)} model. "
                f"Model name: {result.get('model_name', '')}. "
                f"Evaluation metrics: {metrics_json}"
            )
            formatted_result = {
                "model_name": result.get("model_name", model_name or model_type),
                "model_type": result.get("model_type", model_type),
                "metrics": result.get("metrics", {}),
                "message": message
            }
            return json.dumps(formatted_result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error training model: {str(e)}", "error": str(e)})

    def _predict_with_model(model_name: str, input_data: Dict[str, Any]) -> str:
        """Makes predictions using a trained model."""
        try:
            result = actual_predict(model_name=model_name, input_data=input_data)
            model, _, _ = load_model(model_name) # Load model for type check
            is_classification = hasattr(model, "predict_proba")
            predictions = result["predictions"]
            pred_text = f"Predictions: {predictions}"

            response_data = {"text": pred_text, "predictions": predictions}

            if len(predictions) == 1 and is_classification and "probabilities" in result:
                probs = result["probabilities"][0]
                if isinstance(probs, dict): # Class-probability map
                    vis_data = generate_visualization(
                        'bar', list(probs.keys()), list(probs.values()),
                        title='Prediction Probability Distribution'
                    )
                    table_data = generate_data_table(
                        [{'Class': c, 'Probability': p} for c, p in probs.items()]
                    )
                    response_data.update({
                        "visualization_data": vis_data,
                        "table_data": table_data,
                        "probabilities": probs
                    })
            elif len(predictions) > 1: # Multiple predictions
                response_data["table_data"] = generate_data_table(
                    [{'Index': i, 'Prediction': p} for i,p in enumerate(predictions[:100])]
                )
                if is_classification:
                    from collections import Counter
                    counts = Counter(predictions)
                    vis_data_pie = generate_visualization(
                        'pie', list(counts.keys()), list(counts.values()),
                        title='Predicted Class Distribution'
                    )
                    response_data["visualization_data"] = vis_data_pie
            return json.dumps(response_data, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error during prediction: {str(e)}", "error": str(e)})

    def _list_models() -> str:
        """Lists all available trained models."""
        try:
            models = actual_list_models()
            if not models:
                return json.dumps({"text": "📚 No trained models found.", "models": []})

            model_list_text = "📚 Available models:\n\n" + "\n".join(
                [f"- {model['name']} ({model['type']})" for model in models]
            )
            table_data = generate_data_table(models)
            
            model_types_counts = pd.Series([m['type'] for m in models]).value_counts().to_dict()
            vis_data = None
            if model_types_counts:
                vis_data = generate_visualization(
                    'pie', list(model_types_counts.keys()),
                    list(model_types_counts.values()), title='Model Type Distribution'
                )
            return json.dumps({
                "text": model_list_text, "visualization_data": vis_data,
                "table_data": table_data, "models": models
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error listing models: {str(e)}", "error": str(e)})

    def _recommend_model(task_description: str) -> str:
        """Recommends a model based on the task description."""
        try:
            result = actual_select_model(task_description)
            if not result or not result.get('recommendations'): # Check if recommendations list exists and is not empty
                return json.dumps({
                    "text": (
                        "❌ Could not recommend a suitable model for this task. "
                        "Please provide a more detailed task description or try a general model."
                    )
                })

            recommendations = result['recommendations']
            rec_text = f"📊 Recommended models for task \"{task_description}\":\n\n"
            for idx, rec in enumerate(recommendations):
                rec_text += (
                    f"{idx+1}. {rec.get('model_type', 'Unknown')} "
                    f"(Confidence: {rec.get('confidence', 0)*100:.1f}%)\n"
                    f"   Reason: {rec.get('reason', 'No reason provided')}\n\n"
                )
            
            vis_data = generate_visualization(
                'bar', [r.get('model_type', 'Other') for r in recommendations],
                [r.get('confidence', 0)*100 for r in recommendations],
                title='Model Recommendation Scores'
            )
            table_data = generate_data_table(recommendations)

            return json.dumps({
                "text": rec_text, "visualization_data": vis_data,
                "table_data": table_data, "recommendations": recommendations
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error recommending model: {str(e)}", "error": str(e)})

    def _data_analysis(
        file_path: str, target_column: Optional[str] = None,
        analysis_type: Optional[str] = None
    ) -> str:
        """Analyzes a dataset and provides statistics and visualizations."""
        try:
            if not os.path.exists(file_path):
                return json.dumps({"text": f"❌ File {file_path} not found.", "error": "File not found"})

            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            n_rows, n_cols = df.shape
            missing_values = df.isnull().sum().sum()
            missing_percent = (missing_values / (n_rows * n_cols) * 100) if (n_rows * n_cols) > 0 else 0


            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numeric_cols = df.select_dtypes(include=numerics).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            analysis_text = (
                f"📊 Data Analysis Report - {os.path.basename(file_path)}\n\n"
                f"▶ Basic Info:\n"
                f"  - Rows: {n_rows}\n  - Columns: {n_cols}\n"
                f"  - Missing values: {missing_values} ({missing_percent:.2f}%)\n\n"
                f"▶ Column Types:\n"
                f"  - Numerical: {len(numeric_cols)}\n  - Categorical: {len(categorical_cols)}\n"
            )
            visualizations = []
            tables = []

            # Column type distribution
            visualizations.append(
                generate_visualization(
                    'pie', ['Numerical', 'Categorical'],
                    [len(numeric_cols), len(categorical_cols)], title='Column Type Distribution'
                )
            )
            # Numerical stats table
            if numeric_cols:
                numeric_stats_df = df[numeric_cols].describe().transpose().reset_index()
                # Format numeric_stats_df for table_data
                formatted_stats_data = []
                for _, row in numeric_stats_df.iterrows():
                    formatted_row = [row['index']] # Column name
                    for col_stat in numeric_stats_df.columns[1:]: # Iterate through stat names
                         formatted_row.append(f"{row[col_stat]:.4f}" if isinstance(row[col_stat], (int,float)) else row[col_stat])
                    formatted_stats_data.append(formatted_row)

                tables.append({
                    'title': 'Numerical Column Statistics',
                    'data': {
                        'columns': ['Feature'] + numeric_stats_df.columns[1:].tolist(),
                        'data': formatted_stats_data
                        }
                })


            # Correlation analysis
            if numeric_cols and len(numeric_cols) >=2:
                corr = df[numeric_cols].corr()
                if analysis_type == 'feature_relevance' and target_column and target_column in numeric_cols:
                    target_corr = corr[target_column].sort_values(ascending=False)
                    filtered_target_corr = target_corr[target_corr.index != target_column]
                    visualizations.append(
                        generate_visualization(
                            'bar', filtered_target_corr.index.tolist(),
                            filtered_target_corr.values.tolist(),
                            title=f'Feature Correlation with "{target_column}"'
                        )
                    )
                    tables.append({
                        'title': f'Correlation with "{target_column}"',
                        'data': generate_data_table(filtered_target_corr.reset_index())
                    })
                else: # General correlation heatmap
                    fig_heatmap = plt.figure(figsize=(12,10)) # Create figure for heatmap
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=fig_heatmap.gca())
                    plt.title('Feature Correlation Matrix')
                    visualizations.append({
                         'type': 'heatmap', 'title': 'Feature Correlation Matrix',
                         'image': _save_plot_to_base64(fig_heatmap) # Pass the figure object
                    })


            # Data preview table
            tables.append({'title': 'Data Preview (First 10 rows)', 'data': generate_data_table(df.head(10))})

            return json.dumps({
                "text": analysis_text, "visualizations": visualizations, "tables": tables,
                "df_info": {"shape": [n_rows, n_cols], "columns": df.columns.tolist()}
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error analyzing data: {str(e)}", "error": str(e)})

    def _evaluate_model(model_name: str, test_data_path: str, target_column: str) -> str:
        """Evaluates a model on test data."""
        try:
            print(f"[INFO] Starting evaluation for model {model_name}...")
            model_path = os.path.join('ml_models', f"{model_name}.pkl")
            if not os.path.exists(model_path):
                return json.dumps({"text": f"❌ Model {model_name} not found.", "error": "MODEL_NOT_FOUND"})
            if not os.path.exists(test_data_path):
                return json.dumps({"text": f"❌ Test data {test_data_path} not found.", "error": "TEST_DATA_NOT_FOUND"})

            model, preprocessors, _ = load_model(model_name)
            test_data = pd.read_csv(test_data_path) if test_data_path.endswith('.csv') \
                else pd.read_excel(test_data_path)

            if target_column not in test_data.columns:
                return json.dumps({
                    "text": f"❌ Target column {target_column} not in test data.",
                    "error": "TARGET_COLUMN_NOT_FOUND"
                })

            X_test_df = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # Apply preprocessing (simplified, ensure it matches training)
            if 'label_encoders' in preprocessors:
                for col, encoder in preprocessors['label_encoders'].items():
                    if col in X_test_df.columns:
                        X_test_df[col] = X_test_df[col].astype(str).map(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1 # Handle unseen
                        )
            if 'scaler' in preprocessors and hasattr(preprocessors['scaler'], 'transform'):
                num_cols = [col for col in X_test_df.columns if col in preprocessors['scaler'].feature_names_in_]
                if num_cols:
                    X_test_df[num_cols] = preprocessors['scaler'].transform(X_test_df[num_cols])
            
            X_test = X_test_df.values # Convert to numpy array for sklearn
            y_pred = model.predict(X_test)
            is_classifier = hasattr(model, "predict_proba")
            metrics, visualizations, tables = {}, [], []

            if is_classifier:
                metrics.update({
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
                })
                class_names = preprocessors.get('label_encoders', {}).get(target_column, None)
                class_names_list = class_names.classes_.tolist() if class_names else sorted(list(y_test.unique()))

                cm_vis = visualize_confusion_matrix(y_test, y_pred, class_names=class_names_list)
                visualizations.append(cm_vis)
                tables.append({'title': 'Confusion Matrix Breakdown', 'data': cm_vis.get('table_data')})
                
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0, target_names=class_names_list)
                # Convert report_dict to a simpler table structure
                report_table_data = []
                for lbl, mtrcs in report_dict.items():
                    if isinstance(mtrcs, dict): # individual classes and averages
                        report_table_data.append({'Class/Avg': lbl, **mtrcs})
                    else: # overall accuracy
                         report_table_data.append({'Class/Avg': lbl, 'value': mtrcs})
                tables.append({'title': 'Classification Report', 'data': generate_data_table(report_table_data)})


            else: # Regressor
                metrics.update({
                    "mse": mean_squared_error(y_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "mae": mean_absolute_error(y_test, y_pred),
                    "r2": r2_score(y_test, y_pred)
                })
                # Scatter plot for actual vs predicted
                fig_scatter = plt.figure(figsize=(8,6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                plt.xlabel('Actual Values'); plt.ylabel('Predicted Values'); plt.title('Actual vs. Predicted')
                visualizations.append({'type': 'scatter', 'title': 'Actual vs. Predicted', 'image': _save_plot_to_base64(fig_scatter)})


            metrics_vis = visualize_metrics(metrics)
            if metrics_vis: visualizations.append(metrics_vis)
            
            eval_text = f"📊 Model {model_name} evaluation results:\n" + \
                        f"▶ Model Type: {'Classifier' if is_classifier else 'Regressor'}\n" + \
                        f"▶ Test Samples: {len(y_test)}\n" + \
                        "\n".join([f"  - {k.capitalize()}: {v:.4f}" for k, v in metrics.items()])

            return json.dumps({
                "text": eval_text, "visualizations": visualizations,
                "tables": tables, "metrics": metrics,
                "model_type": "classifier" if is_classifier else "regressor"
            }, ensure_ascii=False)
        except Exception as e:
            tb_str = traceback.format_exc()
            return json.dumps({
                "text": f"❌ Error evaluating model: {str(e)}", "error": str(e),
                "error_type": type(e).__name__, "traceback": tb_str
            })

    def _create_ensemble_model(
        base_models: List[str], ensemble_type: str = 'voting',
        weights: Optional[List[float]] = None, save_name: Optional[str] = None
    ) -> str:
        """Creates an ensemble model."""
        try:
            if ensemble_type == 'voting_classifier': ensemble_type = 'voting' # Map to common type
            result = create_ensemble_model(
                base_models=base_models, ensemble_type=ensemble_type,
                weights=weights, save_name=save_name
            )
            model_info = (
                f"✅ Ensemble model created successfully!\n\n"
                f"📊 Model Name: {result['model_name']}\n"
                f"📈 Ensemble Type: {ensemble_type}\n"
                f"📑 Base Models: {', '.join(base_models)}\n"
            )
            if weights: model_info += f"⚖️ Weights: {weights}\n"
            return json.dumps({
                "text": model_info,
                "result": {"model_name": result['model_name'], "ensemble_type": ensemble_type, "base_models": base_models}
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error creating ensemble: {str(e)}", "error": str(e)})

    def _auto_select_model(
        data_path: str, target_column: str,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None
    ) -> str:
        """Automatically selects the best model and optimizes hyperparameters."""
        try:
            result = actual_auto_select(
                data_path=data_path, target_column=target_column,
                categorical_columns=categorical_columns, numerical_columns=numerical_columns
            )
            params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
            model_info = (
                f"✅ Auto model selection complete!\n\n"
                f"🏆 Best Model: {result['model_type']}\n"
                f"📊 Model Name: {result['model_name']}\n"
                f"⚙️ Best Parameters: {params_str}\n"
                f"📈 CV Score: {result['cv_score']:.4f}\n\n"
                f"📊 All models comparison:\n" +
                "\n".join([
                    f"{idx+1}. {mr['model_type']}: CV={mr['cv_score']:.4f}, Test={mr['test_score']:.4f}"
                    for idx, mr in enumerate(result['all_models_results'])
                ])
            )
            vis_data = generate_visualization(
                'bar', [m['model_type'] for m in result['all_models_results']],
                [m['cv_score'] for m in result['all_models_results']],
                title='Model CV Score Comparison'
            )
            table_data = generate_data_table([
                {
                    'Model Type': m['model_type'], 'CV Score': f"{m['cv_score']:.4f}",
                    'Test Score': f"{m['test_score']:.4f}", 'Parameters': str(m['best_params'])
                } for m in result['all_models_results']
            ])
            return json.dumps({
                "text": model_info, "visualization_data": vis_data,
                "table_data": table_data, "result": result
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error in auto model selection: {str(e)}", "error": str(e)})

    def _explain_prediction(model_name: str, input_data: Dict[str, Any]) -> str:
        """Explains a model's prediction."""
        try:
            explanation = actual_explain_prediction(model_name, input_data)
            pred_str = str(explanation['prediction'][0]) if len(explanation['prediction']) == 1 else str(explanation['prediction'])
            exp_text = (
                f"✅ Model prediction explanation complete!\n\n"
                f"📊 Prediction: {pred_str}\n\n"
            )
            if explanation.get('feature_importance'):
                exp_text += "📑 Feature Importance (Top 5):\n" + "\n".join([
                    f"  - {feat}: {imp:.4f}" for feat, imp in sorted(
                        explanation['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True
                    )[:5]
                ])
            viz_results = visualize_model_explanation(explanation)
            return json.dumps({
                "text": exp_text, "visualizations": viz_results.get('visualizations', []),
                "tables": viz_results.get('tables', []), "explanation": explanation
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error explaining prediction: {str(e)}", "error": str(e)})

    def _compare_models(
        model_names: List[str], test_data_path: str, target_column: str
    ) -> str:
        """Compares performance of multiple models."""
        try:
            if not model_names:
                return json.dumps({
                    "text": "❌ Model comparison failed: At least one model name must be provided.",
                    "error": "Model name list cannot be empty."
                })
            comparison = actual_compare_models(model_names, test_data_path, target_column)
            comp_text = (
                f"✅ Model comparison complete!\n\n"
                f"📊 Test Data: {test_data_path}\n🎯 Target Column: {target_column}\n\n"
            )
            if comparison.get('best_classifier'):
                comp_text += f"🏆 Best Classifier: {comparison['best_classifier']}\n"
            if comparison.get('best_regressor'):
                comp_text += f"🏆 Best Regressor: {comparison['best_regressor']}\n"
            
            viz_results = visualize_model_comparison(comparison)
            return json.dumps({
                "text": comp_text, "visualizations": viz_results.get('visualizations', []),
                "tables": viz_results.get('tables', []), "comparison": comparison
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error comparing models: {str(e)}", "error": str(e)})

    def _version_model(
        model_name: str, version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Versions an existing model."""
        try:
            loaded_model, preprocessors, _ = load_model(model_name)
            if loaded_model is None:
                 return json.dumps({
                    "text": f"❌ Error versioning model: Model '{model_name}' not found or failed to load.",
                    "error": f"Model '{model_name}' not found or failed to load.",
                })
            version_info = save_model_with_version(
                loaded_model, model_name, preprocessors, metadata, version
            )
            ver_text = (
                f"✅ Model version saved successfully!\n\n"
                f"📊 Model Name: {version_info['model_name']}\n"
                f"🔖 Version: {version_info['version']}\n"
                f"⏱️ Timestamp: {version_info['timestamp']}\n"
            )
            table_data = generate_data_table(
                [{'Field': k, 'Value': str(v)} for k,v in version_info.items() if k not in ('path','timestamp')]
            )
            return json.dumps({
                "text": ver_text, "table_data": table_data, "version_info": version_info
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error versioning model: {str(e)}", "error": str(e)})

    def _list_model_versions(model_name: str) -> str:
        """Lists all versions of a model."""
        try:
            versions = list_model_versions(model_name)
            if not versions:
                return json.dumps({"text": f"📊 No versions found for model {model_name}."})
            
            ver_text = f"📊 Versions for model {model_name}:\n\n" + "\n".join([
                f"{idx+1}. Version {v.get('version', 'N/A')} (Time: {v.get('timestamp', 'N/A')})"
                for idx, v in enumerate(versions)
            ])
            table_data = generate_data_table(versions)
            return json.dumps({
                "text": ver_text, "table_data": table_data, "versions": versions
            }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"text": f"❌ Error listing model versions: {str(e)}", "error": str(e)})

    tools = [
        StructuredTool.from_function(
            func=_train_model, name="train_model",
            description="Trains a machine learning model (e.g., linear regression, logistic regression, decision tree, random forest).",
            args_schema=TrainModelInput
        ),
        StructuredTool.from_function(
            func=_predict_with_model, name="predict_with_model",
            description="Uses a trained model to make predictions.",
            args_schema=PredictInput
        ),
        Tool.from_function(
            func=_list_models, name="list_models",
            description="Lists all available machine learning models."
        ),
        Tool.from_function(
            func=_recommend_model, name="recommend_model",
            description="Recommends suitable machine learning models based on a task description.",
            args_schema=RecommendModelInput
        ),
        Tool.from_function(
            func=_data_analysis, name="analyze_data",
            description="Analyzes dataset features, statistics, and identifies features most relevant to a target variable.",
            args_schema=DataAnalysisInput
        ),
        Tool.from_function(
            func=_evaluate_model, name="evaluate_model",
            description="Evaluates a model's performance on test data.",
            args_schema=EvaluateModelInput
        ),
        Tool.from_function(
            func=_create_ensemble_model, name="create_ensemble_model",
            description="Creates an ensemble model by combining multiple base models.",
            args_schema=EnsembleModelInput
        ),
        Tool.from_function(
            func=_auto_select_model, name="auto_select_model",
            description="Automatically selects the best model and optimizes its hyperparameters.",
            args_schema=AutoSelectModelInput
        ),
        Tool.from_function(
            func=_explain_prediction, name="explain_prediction",
            description="Explains model predictions, providing feature importance and contribution analysis.",
            args_schema=ExplainPredictionInput
        ),
        Tool.from_function(
            func=_compare_models, name="compare_models",
            description="Compares the performance of multiple models on a test set.",
            args_schema=CompareModelsInput
        ),
        Tool.from_function(
            func=_version_model, name="version_model",
            description="Creates a new version for a model.",
            args_schema=VersionModelInput
        ),
        Tool.from_function(
            func=_list_model_versions, name="list_model_versions",
            description="Lists all versions of a specific model.",
            args_schema=ListModelVersionsInput
        )
    ]
    return tools

def create_ml_agent(use_existing_model: bool = True):
    """Creates a machine learning agent."""
    llm = BaiduErnieLLM() # Changed from 'model' to 'llm' for clarity
    tools = create_ml_tools()

    prompt_template_str = """
{tools}
Tool Names: {tool_names}
Input: {input}
{agent_scratchpad}
"""
    model_preference_text = ""
    if use_existing_model:
        model_preference_text = (
            "\nImportant: The current setting prioritizes using pre-trained models. "
            "Do not retrain a model unless explicitly requested by the user or if no suitable existing model is available.\n"
        )
    
    effective_template_str = model_preference_text + prompt_template_str

    prompt = PromptTemplate(
        template=effective_template_str,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
    )

    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True,
        handle_parsing_errors=lambda e: (
            f"JSON parsing error: {str(e)}\n"
            f"Original response:\n```json\n{e.response.replace('{', '{{').replace('}', '}}')}\n```\n"
        ),
        max_iterations=1, # Reduced max_iterations to prevent timeouts
        return_intermediate_steps=True,
        max_execution_time=180 # Set max execution time to 180 seconds
    )
    return agent_executor

def query_ml_agent(question: str, use_existing_model: bool = True) -> Dict[str, Any]:
    """
    Queries the machine learning agent.

    Args:
        question: The user's question.
        use_existing_model: Flag to indicate if existing models should be preferred.

    Returns:
        A dictionary containing the answer and potentially visualization data.
    """
    agent_response_output = "An error occurred, or no output was generated."
    visualization_data = None
    table_data = None
    is_ml_query_flag = True # Default to True, assuming it's an ML query
    error_message = None
    error_details = None
    expected_format_info = None # For parsing errors

    try:
        agent = create_ml_agent(use_existing_model=use_existing_model)
        response = agent.invoke({"input": question})
        agent_response_output = response.get("output", agent_response_output)
        steps = response.get("intermediate_steps", [])

        if steps: # Try to extract structured output from the last tool call
            last_step_output = steps[-1][1] # Tool output is the second element of the tuple
            if isinstance(last_step_output, str) and last_step_output.strip().startswith('{'):
                try:
                    json_output = json.loads(last_step_output)
                    agent_response_output = json_output.get('text', agent_response_output)
                    visualization_data = json_output.get('visualization_data')
                    table_data = json_output.get('table_data')
                    # Handle lists of visualizations/tables if present
                    if 'visualizations' in json_output and json_output['visualizations']:
                        visualization_data = json_output['visualizations'][0] # Take first for simplicity
                    if 'tables' in json_output and json_output['tables']:
                         table_data = json_output['tables'][0].get('data') # Take data from first table
                except (json.JSONDecodeError, ValueError) as e_json:
                    error_message = "Model response format error (tool output)."
                    error_details = f"Expected JSON from tool, got error: {str(e_json)}. Raw: {last_step_output[:200]}"
                    # Keep agent_response_output as the agent's final textual output
            # If tool_output is not a JSON string, agent_response_output (agent's final text) is used
            
    except TimeoutError as te:
        error_msg_detail = f"Query processing timed out: {str(te)}\n{traceback.format_exc()}"
        print(error_msg_detail)
        agent_response_output = "Query timed out. Please try simplifying your request or using a smaller dataset."
        error_message = "Request timed out"
    except Exception as e_outer:
        error_msg_detail = f"Error processing ML query: {str(e_outer)}\n{traceback.format_exc()}"
        print(error_msg_detail)
        agent_response_output = f"Error processing query: {str(e_outer)}"
        error_message = str(e_outer)

    result = {
        "answer": agent_response_output,
        "visualization_data": visualization_data,
        "table_data": table_data,
        "is_ml_query": is_ml_query_flag
    }
    if error_message: result["error"] = error_message
    if error_details: result["details"] = error_details
    if expected_format_info: result["expected_format"] = expected_format_info
    return result

def enhance_ml_query_with_rag(query: str, ml_response: Dict[str, Any]) -> str:
    """Enhances machine learning query results using the RAG system."""
    # This function seems to be a placeholder or needs more context.
    # For now, it just returns the original ML answer.
    # To make it useful, it would need to:
    # 1. Identify parts of the ml_response or query that could be enhanced by RAG.
    # 2. Formulate a new query for the RAG system.
    # 3. Combine the RAG response with the original ml_answer.
    from rag_core import query_rag # Local import to avoid circular dependency issues at module level

    ml_answer = ml_response.get("answer", "")
    # Example: If ML answer is short or mentions a concept, query RAG for more details.
    # rag_query_text = f"Explain more about: {ml_answer}" # Simplistic
    # rag_response = query_rag(rag_query_text)
    # enhanced_answer = ml_answer + "\n\nFurther details from knowledge base:\n" + rag_response.get("answer","")
    # return enhanced_answer
    return ml_answer # Placeholder: returns original ML answer