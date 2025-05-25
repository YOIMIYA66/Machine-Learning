# plotting_utils.py
import base64
import colorsys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_gradient_colors(n_colors: int) -> List[Tuple[float, float, float, float]]:
    """Generates a list of gradient colors for charts."""
    colors = []
    for i in range(n_colors):
        # Gradient from blue-purple to sky blue
        hue = 0.6 + (0.2 * i / max(1, n_colors - 1))
        saturation = 0.7
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        rgba = (rgb[0], rgb[1], rgb[2], 0.7)  # RGBA format
        colors.append(rgba)
    return colors

def _save_plot_to_base64(plt_figure: plt.Figure) -> str:
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
    """
    Generates a generic visualization chart.

    Returns:
        A dictionary containing plot type, labels, values, title, options, and base64 image.
    """
    fig = plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')

    options = options or {}
    num_colors = len(values) if isinstance(values, list) and len(values) > 0 else 5
    plot_colors = generate_gradient_colors(num_colors) # Renamed to avoid conflict

    if data_type == 'bar':
        plt.bar(labels, values, color=plot_colors)
        plt.ylabel('Value')
    elif data_type == 'line':
        plt.plot(labels, values, marker='o', color=plot_colors[0] if plot_colors else '#4F46E5', linewidth=2)
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    elif data_type == 'pie':
        if not values or sum(values) == 0:
             plt.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center')
        else:
            plt.pie(
                values, labels=labels, autopct='%1.1f%%',
                startangle=90, shadow=True, colors=plot_colors
            )
        plt.axis('equal')
    elif data_type == 'scatter':
        plt.scatter(
            labels, values, color=plot_colors[0] if plot_colors else '#4F46E5',
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
        if labels and values and len(labels) == len(values) and len(labels) > 2 : # Radar needs at least 3 points
            theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            plot_values_radar = np.array(values) # Renamed to avoid conflict
            ax = plt.subplot(111, polar=True)
            ax.fill(theta, plot_values_radar, color=plot_colors[0] if plot_colors else '#4F46E5', alpha=0.25)
            ax.plot(theta, plot_values_radar, color=plot_colors[0] if plot_colors else '#4F46E5', linewidth=2)
            ax.set_xticks(theta)
            ax.set_xticklabels(labels)
            ax.grid(True)
        else:
            plt.text(0.5, 0.5, 'Insufficient or mismatched data for radar chart (requires >= 3 points)', ha='center', va='center')

    elif data_type == 'bubble':
        sizes = options.get('sizes', [50] * len(values))
        plt.scatter(labels, values, s=sizes, alpha=0.6, c=plot_colors)
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
    """
    Visualizes feature importance if the model supports it.
    Returns a dict with plot data and table data.
    """
    if not hasattr(model, 'feature_importances_'):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(n_features, len(feature_names))

    fig = plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    plt.title('Feature Importance')
    plot_colors = generate_gradient_colors(top_n)
    plt.bar(
        range(top_n), importances[indices][:top_n],
        color=plot_colors, align='center'
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
    """
    Visualizes a confusion matrix.
    Returns a dict with plot data and table data.
    """
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

    # Handle cases where sum along axis is 0 to avoid division by zero
    sum_pred = np.sum(cm, axis=0)
    sum_true = np.sum(cm, axis=1)
    
    precision = np.zeros_like(np.diag(cm), dtype=float)
    np.divide(np.diag(cm), sum_pred, out=precision, where=sum_pred!=0)

    recall = np.zeros_like(np.diag(cm), dtype=float)
    np.divide(np.diag(cm), sum_true, out=recall, where=sum_true!=0)

    metrics_data = []
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    for i, class_name_val in enumerate(class_names): # Renamed class_name to avoid conflict
        metrics_data.append([
            class_name_val,
            int(sum_true[i]) if i < len(sum_true) else 0,
            int(cm[i, i]) if i < cm.shape[0] else 0,
            float(precision[i]) if i < len(precision) else 0.0,
            float(recall[i]) if i < len(recall) else 0.0
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
    """
    Visualizes model evaluation metrics as a bar chart.
    Returns a dict with plot data and table data.
    """
    numeric_metrics = {
        k: v for k, v in metrics_dict.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not numeric_metrics:
        return None

    fig = plt.figure(figsize=(10, 6))
    plt.style.use('ggplot')
    plot_colors = generate_gradient_colors(len(numeric_metrics))
    plt.bar(numeric_metrics.keys(), numeric_metrics.values(), color=plot_colors)
    plt.title('Model Evaluation Metrics')
    y_max_limit = 1.0
    if numeric_metrics:
        y_max_limit = max(1.0, max(numeric_metrics.values()) * 1.1 if numeric_metrics.values() else 1.0)


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
    """
    Visualizes clustering results using PCA or t-SNE for dimensionality reduction.
    Returns a dict with plot data and table data.
    """
    if X.shape[1] > 2:
        reducer = PCA(n_components=2) if method == 'pca' else TSNE(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    else:
        X_2d = X

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    plot_colors = generate_gradient_colors(n_clusters)

    fig = plt.figure(figsize=(10, 8))
    for i, label_val in enumerate(unique_labels):
        mask = labels == label_val
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1], c=[plot_colors[i % len(plot_colors)]],
            label=f'Cluster {label_val}', alpha=0.7, s=80, edgecolors='w'
        )
    plt.legend()
    plt.title('Cluster Visualization')
    plt.xlabel('Component 1 (PCA)' if method == 'pca' else 't-SNE Dimension 1')
    plt.ylabel('Component 2 (PCA)' if method == 'pca' else 't-SNE Dimension 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    image_base64 = _save_plot_to_base64(fig)

    cluster_counts = np.bincount(labels.astype(int)) if labels.size > 0 else np.array([])
    table_data = {
        'columns': ['Cluster', 'Sample Count', 'Proportion'],
        'data': [
            [f'Cluster {i}', int(count), float(count) / len(labels) if len(labels) > 0 else 0.0]
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

def visualize_feature_importance_radar(
    feature_importance: Dict[str, float], title: str = 'Feature Importance Radar Chart'
) -> Dict[str, Any]:
    """Generates a radar chart for feature importance."""
    features = list(feature_importance.keys())
    values = np.array(list(feature_importance.values()))

    if not features or values.size == 0: # Handle empty input
        return {'type': 'radar', 'labels': [], 'values': [], 'title': title, 'image': ''}


    # Normalize values to 0-1 range
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val: # Avoid division by zero if all values are the same
        values_normalized = np.ones_like(values) * 0.5 if values.size > 0 else np.array([]) # Or all 0, or handle as error
    else:
        values_normalized = (values - min_val) / (max_val - min_val)


    fig = plt.figure(figsize=(10, 8))
    # Ensure there are at least 3 features for a meaningful radar chart
    if len(features) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        plot_values = np.concatenate((values_normalized, [values_normalized[0]]))
        plot_angles = np.concatenate((angles, [angles[0]]))
        
        ax = plt.subplot(111, polar=True)
        ax.fill(plot_angles, plot_values, color=generate_gradient_colors(1)[0], alpha=0.25)
        ax.plot(plot_angles, plot_values, 'o-', color=generate_gradient_colors(1)[0], linewidth=2)
        ax.set_xticks(angles)
        ax.set_xticklabels(features)
        ax.set_yticks(np.linspace(0, 1, 6))
        plt.title(title)
    else:
        plt.text(0.5, 0.5, "Radar chart requires at least 3 features.", horizontalalignment='center', verticalalignment='center')
        plt.title(title)

    image_base64 = _save_plot_to_base64(fig)

    return {
        'type': 'radar',
        'labels': features,
        'values': values_normalized.tolist(),
        'title': title,
        'image': image_base64
    }

def visualize_model_comparison(
    comparison_results: Dict[str, Any] 
) -> Dict[str, List[Any]]: # Removed metric='auto' as it wasn't used
    """Visualizes model comparison results."""
    models = comparison_results.get('models', [])
    classifiers = [m for m in models if m.get('is_classifier', False) and 'error' not in m]
    regressors = [m for m in models if not m.get('is_classifier', False) and 'error' not in m]
    visualizations = [] # Stores dicts from generate_visualization
    tables = [] # Stores dicts for table data structure

    # Process classifiers
    if classifiers:
        model_names = [m['model_name'] for m in classifiers]
        classifier_metrics_map = {'accuracy': 'Accuracy', 'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1 Score'}
        for metric_key, metric_display_name in classifier_metrics_map.items():
            values = [m['metrics'].get(metric_key, 0) for m in classifiers]
            vis_data = generate_visualization(
                'bar', model_names, values,
                title=f'Classifier Comparison - {metric_display_name}'
            )
            visualizations.append(vis_data)
        
        classifier_table_data = []
        for m in classifiers:
            row = {'Model': m['model_name']}
            for mk, md_name in classifier_metrics_map.items():
                row[md_name] = m['metrics'].get(mk, '-')
            classifier_table_data.append(row)
        tables.append({
            'title': 'Classifier Performance Comparison',
            'data': generate_data_table(classifier_table_data)
        })

    # Process regressors
    if regressors:
        model_names = [m['model_name'] for m in regressors]
        regressor_metrics_map = {'mse': 'MSE', 'rmse': 'RMSE', 'mae': 'MAE', 'r2': 'R² Score'}
        for metric_key, metric_display_name in regressor_metrics_map.items():
            values = [m['metrics'].get(metric_key, 0) for m in regressors]
            title_suffix = " (Lower is Better)" if metric_key != 'r2' else " (Higher is Better)"
            vis_data = generate_visualization(
                'bar', model_names, values,
                title=f'Regressor Comparison - {metric_display_name}{title_suffix}'
            )
            visualizations.append(vis_data)

        regressor_table_data = []
        for m in regressors:
            row = {'Model': m['model_name']}
            for mk, md_name in regressor_metrics_map.items():
                 row[md_name] = m['metrics'].get(mk, '-')
            regressor_table_data.append(row)
        tables.append({
            'title': 'Regressor Performance Comparison',
            'data': generate_data_table(regressor_table_data)
        })

    # Best model summary table
    best_model_summary_info = []
    if comparison_results.get('best_classifier'):
        best_model_summary_info.append({
            'Type': 'Best Classifier',
            'Name': comparison_results['best_classifier'],
            'Metric': 'F1 Score (higher is better)'
        })
    if comparison_results.get('best_regressor'):
         best_model_summary_info.append({
            'Type': 'Best Regressor',
            'Name': comparison_results['best_regressor'],
            'Metric': 'R² Score (higher is better)'
        })
    if best_model_summary_info:
        tables.append({
            'title': 'Best Performing Models',
            'data': generate_data_table(best_model_summary_info)
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
        
        # Create a temporary figure for bar chart with potential negative values
        fig_contrib = plt.figure(figsize=(10,6))
        plt.style.use('ggplot')
        bars = plt.bar(feature_names_contrib, contribution_values, color=generate_gradient_colors(len(top_contributions)))
        for i, value in enumerate(contribution_values):
            if value < 0:
                bars[i].set_color('tomato') # Example: color negative contributions red
        plt.title('Feature Contributions')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        image_base64_contrib = _save_plot_to_base64(fig_contrib)
        
        visualizations.append({
            'type': 'bar_custom', # Indicate it's a custom bar chart due to coloring logic
            'title': 'Feature Contributions',
            'labels': feature_names_contrib,
            'values': contribution_values,
            'image': image_base64_contrib
        })

        tables.append({
            'title': 'Feature Contributions',
            'data': generate_data_table(feature_contributions, columns=['feature', 'value', 'importance', 'contribution'])
        })

    prediction = explanation_result.get('prediction', [])
    if prediction:
        tables.append({
            'title': 'Prediction Result',
            'data': generate_data_table([{'Prediction': p} for p in prediction])
        })
    return {'visualizations': visualizations, 'tables': tables}
