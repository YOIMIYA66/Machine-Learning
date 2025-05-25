# ml_models.py
import datetime
import inspect  # For checking model class parameter signatures
import json
import os
import pickle
import threading
import time
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import multiprocessing
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Scikit-learn imports
from sklearn.cluster import KMeans
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor,
                              VotingClassifier, VotingRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, explained_variance_score,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Directory for saving models
MODELS_DIR = "ml_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Model Caching System
_MODEL_CACHE: Dict[str, Tuple[Any, Dict[str, Any], Dict[str, Any]]] = {} # Model cache dictionary
_MODEL_CACHE_LOCK = threading.RLock()  # Cache lock
_MODEL_CACHE_MAX_SIZE = 10  # Maximum number of models to cache
_MODEL_CACHE_ACCESS_TIMES: Dict[str, float] = {}  # Records last access time for each model

# Model mapping for easy lookup by name
MODEL_TYPES = {
    # Regression Models
    "linear_regression": LinearRegression,
    "random_forest_regressor": RandomForestRegressor,
    
    # Classification Models
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest_classifier": RandomForestClassifier,
    "knn_classifier": KNeighborsClassifier,
    "svm_classifier": SVC,
    "naive_bayes": MultinomialNB,
    
    # Clustering Models
    "kmeans": KMeans,
    
    # Ensemble Model Types
    "voting_classifier": VotingClassifier,
    "voting_regressor": VotingRegressor,
    "stacking_classifier": StackingClassifier,
    "stacking_regressor": StackingRegressor,
    "bagging_classifier": BaggingClassifier,
    "bagging_regressor": BaggingRegressor
}

# Model categories for frontend display and selection
MODEL_CATEGORIES = {
    "regression": ["linear_regression", "random_forest_regressor"],
    "classification": [
        "logistic_regression", "knn_classifier", "decision_tree",
        "svm_classifier", "naive_bayes", "random_forest_classifier"
    ],
    "clustering": ["kmeans"]
}

# Detailed model information including icons and descriptions
MODEL_DETAILS = {
    "linear_regression": {
        "display_name": "Linear Regression Model",
        "icon_class": "fa-chart-line",
        "description": "A basic statistical model for predicting continuous variables. "
                       "It establishes a linear relationship between independent and dependent "
                       "variables, finding the best-fitting line. Suitable for simple numerical prediction tasks."
    },
    "logistic_regression": {
        "display_name": "Logistic Regression Model",
        "icon_class": "fa-code-branch",
        "description": "A statistical model for binary classification problems. It converts "
                       "the output of a linear model into probability values using the Sigmoid function. "
                       "It is computationally efficient, easy to implement, and suitable for linearly separable classification problems."
    },
    "knn_classifier": {
        "display_name": "K-Nearest Neighbors (KNN) Prediction Model",
        "icon_class": "fa-project-diagram",
        "description": "An instance-based learning method that classifies or predicts by "
                       "calculating the distance between a new sample and all samples in the training set, "
                       "selecting the K nearest neighbors for voting or averaging."
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "icon_class": "fa-sitemap",
        "description": "A tree-structured classification model that divides data into different "
                       "categories through a series of conditional judgments. It is intuitive, highly interpretable, "
                       "can handle non-linear relationships, but is prone to overfitting."
    },
    "svm_classifier": {
        "display_name": "Support Vector Machine (SVM) Model",
        "icon_class": "fa-vector-square",
        "description": "A powerful classification algorithm that distinguishes different classes "
                       "of data points by finding an optimal hyperplane. It performs well in high-dimensional spaces, "
                       "can handle non-linear problems using kernel functions, and is suitable for small, complex datasets."
    },
    "naive_bayes": {
        "display_name": "Naive Bayes Classifier",
        "icon_class": "fa-percentage",
        "description": "A probabilistic classifier based on Bayes' theorem, assuming features are "
                       "mutually independent. It trains quickly, requires less training data, is particularly "
                       "suited for text classification and multi-class problems, but may not perform well on "
                       "data with strong feature correlations."
    },
    "kmeans": {
        "display_name": "K-Means Model",
        "icon_class": "fa-object-group",
        "description": "A common clustering algorithm that partitions data points into K clusters "
                       "through iterative optimization. It is simple to implement, computationally efficient, "
                       "suitable for large-scale unsupervised learning, but is sensitive to initial cluster centers "
                       "and struggles with non-spherical clusters."
    }
    # Add details for RandomForest and ensemble models if they are user-selectable directly
}

# Default model parameters
DEFAULT_MODEL_PARAMS = {
    "linear_regression": {},
    "logistic_regression": {"max_iter": 1000, "C": 1.0, "solver": "liblinear"}, # Added solver for default
    "decision_tree": {"max_depth": 5, "random_state": 42},
    "random_forest_classifier": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    "random_forest_regressor": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
    "knn_classifier": {"n_neighbors": 5},
    "svm_classifier": {"C": 1.0, "kernel": "rbf", "probability": True, "random_state": 42}, # Added probability for consistency
    "naive_bayes": {"alpha": 1.0},
    "kmeans": {"n_clusters": 3, "random_state": 42, "n_init": "auto"} # Set n_init explicitly
}


def preprocess_data(
    data: pd.DataFrame,
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    scale_data: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Preprocesses data for machine learning model training.

    Args:
        data: Input DataFrame.
        target_column: Name of the target column.
        categorical_columns: List of categorical feature names for encoding.
        numerical_columns: List of numerical feature names for scaling.
        test_size: Proportion of the dataset to include in the test split.
        scale_data: Whether to scale numerical features.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing: X_train, X_test, y_train, y_test, preprocessors dictionary.
    """
    df = data.copy()
    preprocessors: Dict[str, Any] = {'label_encoders': {}, 'scaler': None}

    # Handle categorical features
    if categorical_columns:
        for col in categorical_columns:
            if col in df.columns and col != target_column: # Ensure not to encode target here
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                preprocessors['label_encoders'][col] = le
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    
    # Encode target column if it's categorical (object or many unique numericals)
    if df[target_column].dtype == 'object' or \
       (pd.api.types.is_numeric_dtype(df[target_column]) and df[target_column].nunique() < 0.1 * len(df[target_column])): # Heuristic for categorical numeric
        le_target = LabelEncoder()
        df[target_column] = le_target.fit_transform(df[target_column].astype(str))
        preprocessors['label_encoders'][target_column] = le_target # Store target encoder as well

    y = df[target_column].values
    X_df = df.drop(columns=[target_column])
    
    # Identify feature columns if not fully specified
    if numerical_columns is None and categorical_columns is None:
        numerical_columns = X_df.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = X_df.select_dtypes(exclude=np.number).columns.tolist()
    elif numerical_columns is None:
        numerical_columns = [col for col in X_df.columns if col not in (categorical_columns or [])]
    elif categorical_columns is None:
        categorical_columns = [col for col in X_df.columns if col not in (numerical_columns or [])]

    # Ensure all identified columns exist in X_df
    all_feature_cols = (numerical_columns or []) + (categorical_columns or [])
    X = X_df[[col for col in all_feature_cols if col in X_df.columns]]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=(y if df[target_column].nunique() > 1 else None)
    )

    # Scale numerical features
    if scale_data and numerical_columns:
        scaler = StandardScaler()
        # Filter numerical_columns to only those present in X_train (after split and potential column drops)
        numerical_cols_in_X_train = [col for col in numerical_columns if col in X_train.columns]
        if numerical_cols_in_X_train:
            X_train[numerical_cols_in_X_train] = scaler.fit_transform(X_train[numerical_cols_in_X_train])
            X_test[numerical_cols_in_X_train] = scaler.transform(X_test[numerical_cols_in_X_train])
            preprocessors['scaler'] = scaler
            preprocessors['scaled_numerical_columns'] = numerical_cols_in_X_train


    return X_train.values, X_test.values, y_train, y_test, preprocessors


def train_model(
    model_type: str,
    data: Union[pd.DataFrame, str],
    target_column: str,
    model_name: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a machine learning model and saves it.

    Args:
        model_type: Model type (e.g., "linear_regression").
        data: DataFrame or path to CSV/Excel file.
        target_column: Name of the target variable column.
        model_name: Name to save the model. Auto-generated if None.
        categorical_columns: List of categorical feature names.
        numerical_columns: List of numerical feature names.
        model_params: Dictionary of model parameters.
        test_size: Proportion for the test set split.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing model information.
    """
    if isinstance(data, str):
        if data.lower().endswith('.csv'):
            df = pd.read_csv(data)
        elif data.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data)
        else:
            raise ValueError(f"Unsupported file format: {data}. Only CSV and Excel are supported.")
        
        if len(df) > 10000 or df.memory_usage(deep=True).sum() > 100 * 1024 * 1024: # 10k rows or 100MB
            df = optimize_dataframe_memory(df)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("Data must be a pandas DataFrame or a file path string.")

    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}. Valid types: {list(MODEL_TYPES.keys())}")

    # Auto-identify column types if not specified
    if categorical_columns is None and numerical_columns is None:
        temp_X = df.drop(columns=[target_column], errors='ignore')
        numerical_columns = temp_X.select_dtypes(include=np.number).columns.tolist()
        categorical_columns = temp_X.select_dtypes(exclude=np.number).columns.tolist()
        print(f"Auto-identified numerical columns: {numerical_columns}")
        print(f"Auto-identified categorical columns: {categorical_columns}")


    X_train, X_test, y_train, y_test, preprocessors = preprocess_data(
        df, target_column, categorical_columns, numerical_columns, test_size, random_state=random_state
    )

    current_model_params = DEFAULT_MODEL_PARAMS.get(model_type, {}).copy()
    if model_params: # User-provided params override defaults
        current_model_params.update(model_params)
    
    # Add random_state to model params if model supports it and it's not already set
    model_class = MODEL_TYPES[model_type]
    model_signature = inspect.signature(model_class.__init__)
    if 'random_state' in model_signature.parameters and 'random_state' not in current_model_params:
        current_model_params['random_state'] = random_state


    model = model_class(**current_model_params)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, model_type)

    if model_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{model_type}_{timestamp}"

    metadata = {
        "model_name": model_name, # Ensure model_name is part of metadata
        "model_type": model_type,
        "target_column": target_column,
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns, # Store identified numerical columns
        "scaled_numerical_columns": preprocessors.get('scaled_numerical_columns'), # Store actually scaled columns
        "metrics": metrics,
        "model_params": current_model_params, # Store used parameters
        "created_at": datetime.datetime.now().isoformat(),
        "data_shape": df.shape,
        "data_columns": df.columns.tolist(),
        "feature_names_in": list(X.columns) if isinstance(X, pd.DataFrame) else None # Store feature names if X was DataFrame
    }

    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((model, preprocessors, metadata), f)
    print(f"Model saved to: {model_path}")

    return {
        "model_name": model_name,
        "model_type": model_type,
        "metrics": metrics,
        "feature_importance": getattr(model, "feature_importances_", None) # Keep this for quick access if needed
    }


def load_model(model_name: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Loads a machine learning model, its preprocessors, and metadata from a file.
    Uses an LRU cache for performance optimization.

    Returns:
        A tuple: (model_object, preprocessors_dict, metadata_dict).
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    file_mtime = os.path.getmtime(model_path)
    cache_key = f"{model_name}_{file_mtime}"

    with _MODEL_CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            _MODEL_CACHE_ACCESS_TIMES[cache_key] = time.time()
            print(f"Loading model {model_name} from cache.")
            return _MODEL_CACHE[cache_key]

        print(f"Loading model {model_name} from file: {model_path}")
        start_time = time.time()
        try:
            with open(model_path, "rb") as f:
                loaded_data = pickle.load(f)
            
            if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
                model, preprocessors, metadata = loaded_data
            elif isinstance(loaded_data, tuple) and len(loaded_data) == 2: # Legacy format
                model, preprocessors = loaded_data
                metadata = {} # Create empty metadata for legacy models
                print(f"Warning: Model {model_name} loaded in legacy format (no metadata found in .pkl).")
            else: # Assume it's just the model object for very old formats
                model = loaded_data
                preprocessors = {}
                metadata = {}
                print(f"Warning: Model {model_name} loaded in very old format (only model object found).")


            load_time = time.time() - start_time
            print(f"Model {model_name} loaded in {load_time:.4f} seconds.")

            result = (model, preprocessors, metadata)
            if len(_MODEL_CACHE) >= _MODEL_CACHE_MAX_SIZE:
                _clean_model_cache()
            _MODEL_CACHE[cache_key] = result
            _MODEL_CACHE_ACCESS_TIMES[cache_key] = time.time()
            return result
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise


def _clean_model_cache():
    """Cleans the model cache by removing the least recently used item."""
    if not _MODEL_CACHE:
        return
    # Sort by access time (oldest first)
    oldest_key = min(_MODEL_CACHE_ACCESS_TIMES, key=_MODEL_CACHE_ACCESS_TIMES.get)
    _MODEL_CACHE.pop(oldest_key, None)
    _MODEL_CACHE_ACCESS_TIMES.pop(oldest_key, None)
    print(f"Removed least recently used model from cache: {oldest_key.split('_')[0]}")


def clear_model_cache():
    """Clears the entire model cache."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
        _MODEL_CACHE_ACCESS_TIMES.clear()
    print("Model cache cleared.")


def predict(
    model_name: str,
    input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]: # Removed target_column as it's not used for prediction input
    """
    Makes predictions using a saved model.

    Args:
        model_name: Name of the model.
        input_data: Input data (DataFrame, single dict, or list of dicts).
        
    Returns:
        Dictionary containing predictions and related information.
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise TypeError("input_data must be a pandas DataFrame, a dictionary, or a list of dictionaries.")

    model, preprocessors, metadata = load_model(model_name)
    
    # Use feature names from metadata if available, falling back to model's if any
    feature_names_in = metadata.get("feature_names_in", None)
    if feature_names_in is None and hasattr(model, 'feature_names_in_'):
        feature_names_in = model.feature_names_in_
    
    # Ensure input_df has the correct columns in the correct order
    if feature_names_in is not None:
        missing_cols = set(feature_names_in) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing columns: {missing_cols}")
        input_df_processed = input_df[feature_names_in].copy() # Reorder/select columns
    else:
        # If no feature_names_in, assume input_df is already correctly ordered/featured
        # This might be risky if preprocessors rely on specific column names/order
        print("Warning: 'feature_names_in' not found in model metadata or model object. Assuming input data is correctly ordered.")
        input_df_processed = input_df.copy()


    # Apply preprocessing transformations
    if 'label_encoders' in preprocessors:
        for col, encoder in preprocessors['label_encoders'].items():
            if col in input_df_processed.columns and col != metadata.get("target_column"): # Don't encode target if present
                input_df_processed[col] = input_df_processed[col].astype(str)
                # Handle unseen labels during transform
                input_df_processed[col] = input_df_processed[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1 # Or handle as NaN / specific category
                )
                if (input_df_processed[col] == -1).any():
                     print(f"Warning: Column '{col}' contained unseen labels, encoded as -1.")


    if 'scaler' in preprocessors and preprocessors['scaler'] is not None:
        scaled_num_cols = metadata.get('scaled_numerical_columns', []) # Use columns that were actually scaled
        # Ensure only existing columns are selected for scaling
        cols_to_scale = [col for col in scaled_num_cols if col in input_df_processed.columns]
        if cols_to_scale:
            input_df_processed[cols_to_scale] = preprocessors['scaler'].transform(input_df_processed[cols_to_scale])
    
    X_predict = input_df_processed.values
    predictions = model.predict(X_predict)
    
    result = {
        "model_name": model_name,
        "predictions": predictions.tolist(),
    }
    
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(X_predict)
            # Format probabilities based on number of classes
            if probabilities.shape[1] == 2: # Binary classification
                 result["probabilities"] = probabilities[:, 1].tolist() # Prob of positive class
            else: # Multiclass
                # Return list of probability arrays, or dict mapping class name to prob
                if hasattr(preprocessors.get('label_encoders',{}).get(metadata.get("target_column")), 'classes_'):
                    classes = preprocessors['label_encoders'][metadata.get("target_column")].classes_
                    result["probabilities"] = [{cls_name: prob for cls_name, prob in zip(classes, prob_array)} for prob_array in probabilities]
                else:
                    result["probabilities"] = probabilities.tolist()

        except Exception as e_proba:
            print(f"Could not get probabilities: {e_proba}")


    return result


def list_available_models() -> List[Dict[str, Any]]:
    """
    Lists all available models, returning JSON-serializable metadata.
    """
    models_metadata = []
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Models directory '{MODELS_DIR}' not found when listing models.")
        return []

    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            model_name = os.path.splitext(filename)[0]
            try:
                _, _, metadata = load_model(model_name) # Load to get metadata

                model_type = metadata.get("model_type", "unknown")
                detail_info = MODEL_DETAILS.get(model_type, {})
                
                model_info = {
                    "name": model_name,
                    "type": model_type,
                    "path": os.path.join(MODELS_DIR, filename),
                    "params": metadata.get("model_params", {}), # Already stringified if needed
                    "display_name": detail_info.get("display_name", model_name.replace("_", " ").title()),
                    "icon_class": detail_info.get("icon_class", "fa-brain"), # Default icon
                    "description": detail_info.get("description", f"A {model_type} model."),
                    "created_at": metadata.get("created_at"),
                    "internal_name": model_name, # For frontend data-model-name
                    "target_column": metadata.get("target_column"),
                    "feature_columns": metadata.get("feature_columns_in", metadata.get("categorical_columns", []) + metadata.get("numerical_columns", [])), # Prefer feature_names_in if available
                    "metrics": metadata.get("metrics")
                }
                
                # Ensure all values are serializable (should mostly be handled by metadata saving)
                for key, value in list(model_info.items()):
                    if not isinstance(value, (str, int, float, bool, list, dict)) and value is not None:
                        model_info[key] = str(value)
                
                models_metadata.append(model_info)
            except Exception as e:
                logger.error(f"Error loading or processing metadata for model {model_name}: {e}")
                models_metadata.append({
                    "name": model_name, "type": "error",
                    "path": os.path.join(MODELS_DIR, filename), "error_message": str(e)
                })
    return models_metadata


def select_model_for_task(task_description: str) -> Optional[Dict[str, Any]]: # Return dict for more info
    """
    Selects the most suitable model based on a task description.
    (Simplified keyword-based implementation)
    """
    available_models = list_available_models()
    if not available_models:
        return None

    # English keywords
    keywords_map = {
        "regression": ["linear_regression", "random_forest_regressor"],
        "predict numerical value": ["linear_regression", "random_forest_regressor"],
        "price prediction": ["linear_regression", "random_forest_regressor"],
        "sales forecast": ["linear_regression", "random_forest_regressor"],
        "classification": ["logistic_regression", "decision_tree", "random_forest_classifier", "knn_classifier", "svm_classifier", "naive_bayes"],
        "is it": ["logistic_regression", "decision_tree", "random_forest_classifier"], # "yes/no" type questions
        "risk identification": ["logistic_regression", "random_forest_classifier"],
        "credit risk": ["decision_tree", "random_forest_classifier"],
        "health risk": ["logistic_regression", "random_forest_classifier"],
        "customer churn": ["logistic_regression", "random_forest_classifier"]
    }
    
    task_desc_lower = task_description.lower()
    preferred_model_types = []
    for keyword, model_types_list in keywords_map.items():
        if keyword in task_desc_lower:
            preferred_model_types.extend(model_types_list)
    
    recommendations = []
    # First, check for preferred types
    if preferred_model_types:
        for model_info in available_models:
            if model_info["type"] in preferred_model_types:
                 recommendations.append({
                     "model_type": model_info["type"],
                     "model_name": model_info["name"],
                     "confidence": 0.7, # Higher confidence for keyword match
                     "reason": f"Matches task description keyword related to '{model_info['type']}'."
                 })
    
    # Add other available models with lower confidence if no strong match or to provide options
    for model_info in available_models:
        if not any(r['model_name'] == model_info['name'] for r in recommendations):
            recommendations.append({
                "model_type": model_info["type"],
                "model_name": model_info["name"],
                "confidence": 0.3, # Lower confidence for general models
                "reason": "General purpose model, might be applicable."
            })

    if not recommendations: return None
    
    # Sort by confidence (desc) then by name (asc)
    recommendations.sort(key=lambda x: (-x['confidence'], x['model_name']))

    return {
        "recommended_model": recommendations[0]['model_name'] if recommendations else None, # Top recommendation
        "recommendations": recommendations # List of all recommendations with scores
    }


def save_model_with_version(
    model: Any,
    model_name: str,
    preprocessors: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves a model with version management.

    Args:
        model: The trained model object.
        model_name: Base name for the model.
        preprocessors: Dictionary of preprocessors used.
        metadata: Model metadata.
        version: Optional version string. If None, a timestamp-based version is generated.

    Returns:
        Dictionary containing version information.
    """
    if version is None:
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    version_dir = os.path.join(MODELS_DIR, f"{model_name}_versions")
    os.makedirs(version_dir, exist_ok=True)
    
    versioned_model_name = f"{model_name}_v{version}"
    model_path = os.path.join(version_dir, f"{versioned_model_name}.pkl")
    
    with open(model_path, "wb") as f:
        pickle.dump((model, preprocessors or {}, metadata or {}), f)
    
    version_info = {
        "model_name": model_name, "version": version,
        "timestamp": datetime.datetime.now().isoformat(), "path": model_path,
    }
    if metadata: version_info.update(metadata)
    
    version_info_path = os.path.join(version_dir, f"{versioned_model_name}_info.json")
    with open(version_info_path, "w", encoding='utf-8') as f:
        json.dump(version_info, f, indent=2, ensure_ascii=False)
    
    return version_info


def list_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """
    Lists all versions of a given model.

    Args:
        model_name: Base name of the model.

    Returns:
        A list of version information dictionaries.
    """
    version_dir = os.path.join(MODELS_DIR, f"{model_name}_versions")
    if not os.path.exists(version_dir):
        return []
    
    versions = []
    for filename in os.listdir(version_dir):
        if filename.endswith("_info.json"): # Assuming info files denote versions
            try:
                with open(os.path.join(version_dir, filename), "r", encoding='utf-8') as f:
                    versions.append(json.load(f))
            except json.JSONDecodeError:
                logger.error(f"Could not decode JSON for version info file: {filename}")
            except Exception as e:
                 logger.error(f"Error reading version info file {filename}: {e}")

    versions.sort(key=lambda x: x.get("timestamp", ""), reverse=True) # Newest first
    return versions


def load_model_version(model_name: str, version: str) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Loads a specific version of a model.

    Args:
        model_name: Base name of the model.
        version: Version string.

    Returns:
        A tuple: (model_object, preprocessors_dict, metadata_dict).
    """
    version_dir = os.path.join(MODELS_DIR, f"{model_name}_versions")
    model_path = os.path.join(version_dir, f"{model_name}_v{version}.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model version file not found: {model_path}")
    
    with open(model_path, "rb") as f:
        return pickle.load(f) # Assumes (model, preprocessors, metadata) structure


def create_ensemble_model(
    base_models: List[Union[str, Tuple[str, Any]]], # Can be names or (name, model_obj)
    ensemble_type: str = 'voting',
    weights: Optional[List[float]] = None,
    final_estimator: Optional[Any] = None, # For stacking
    meta_features: bool = False, # For stacking, passthrough original features
    save_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates an ensemble model from base models.

    Args:
        base_models: List of base model names or (name, model_object) tuples.
        ensemble_type: 'voting', 'stacking', or 'bagging'.
        weights: Weights for voting ensemble.
        final_estimator: Meta-learner for stacking.
        meta_features: Whether to use original features in meta-learner for stacking.
        save_name: Name to save the ensemble model.

    Returns:
        Dictionary with ensemble model info.
    """
    loaded_base_models = []
    is_classifier_list = [] # To check consistency
    
    for item in base_models:
        model_obj, model_name_str = None, None
        if isinstance(item, tuple) and len(item) == 2:
            model_name_str, model_obj = item
        elif isinstance(item, str):
            model_name_str = item
            model_obj, _, _ = load_model(model_name_str) # Load preprocessors and metadata too
        else:
            raise ValueError(f"Invalid item in base_models: {item}. Must be name or (name, model_obj).")

        if not isinstance(model_name_str, str): model_name_str = str(model_name_str) # Ensure name is string
        
        loaded_base_models.append((model_name_str, model_obj))
        is_classifier_list.append(hasattr(model_obj, "predict_proba"))

    if not all(is_classifier_list) and not all(not x for x in is_classifier_list):
        raise ValueError("All base models must be of the same type (all classifiers or all regressors).")
    is_ensemble_classifier = is_classifier_list[0]

    ensemble_model: Any = None
    if ensemble_type == 'voting':
        if is_ensemble_classifier:
            ensemble_model = VotingClassifier(
                estimators=loaded_base_models,
                voting='soft' if all(hasattr(m[1], "predict_proba") for m in loaded_base_models) else 'hard',
                weights=weights
            )
        else:
            ensemble_model = VotingRegressor(estimators=loaded_base_models, weights=weights)
    elif ensemble_type == 'stacking':
        if is_ensemble_classifier:
            ensemble_model = StackingClassifier(
                estimators=loaded_base_models, final_estimator=final_estimator, passthrough=meta_features
            )
        else:
            ensemble_model = StackingRegressor(
                estimators=loaded_base_models, final_estimator=final_estimator, passthrough=meta_features
            )
    elif ensemble_type == 'bagging':
        base_estimator_obj = loaded_base_models[0][1] # Bagging uses one type of base estimator
        if is_ensemble_classifier:
            ensemble_model = BaggingClassifier(base_estimator=base_estimator_obj)
        else:
            ensemble_model = BaggingRegressor(base_estimator=base_estimator_obj)
    else:
        raise ValueError(f"Unsupported ensemble type: {ensemble_type}")

    ensemble_name = save_name or f"{ensemble_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensemble_metadata = {
        'model_name': ensemble_name, 'ensemble_type': ensemble_type,
        'base_model_names': [m[0] for m in loaded_base_models],
        'is_classifier': is_ensemble_classifier, 'weights': weights,
        'description': f"{ensemble_type.capitalize()} ensemble model based on {', '.join([m[0] for m in loaded_base_models])}."
    }
    # Ensemble preprocessors might be complex; for now, assume base models are used with their own or compatible preprocessed data.
    # A true ensemble pipeline would need careful handling of preprocessing if base models require different steps.
    ensemble_preprocessors = {'note': 'Preprocessing should be handled before feeding data to this ensemble or ensure base models use compatible preprocessed data.'}


    model_path = os.path.join(MODELS_DIR, f"{ensemble_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((ensemble_model, ensemble_preprocessors, ensemble_metadata), f)
    
    return {
        'model_name': ensemble_name, 'model_path': model_path,
        'model': ensemble_model, 'preprocessors': ensemble_preprocessors, 'metadata': ensemble_metadata
    }


def auto_model_selection(
    data_path: str,
    target_column: str,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None,
    cv: int = 5,
    metric: str = 'auto',
    models_to_try: Optional[List[str]] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Automatically selects the best model and parameters using GridSearchCV.

    Args:
        data_path: Path to the data file (CSV, Excel, JSON).
        target_column: Name of the target column.
        categorical_columns: List of categorical feature names.
        numerical_columns: List of numerical feature names.
        cv: Number of cross-validation folds.
        metric: Evaluation metric ('auto', or specific scikit-learn scorer).
        models_to_try: List of model types to try. None means try all suitable.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with the best model, its parameters, and performance.
    """
    if data_path.lower().endswith('.csv'):
        data = pd.read_csv(data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    elif data_path.lower().endswith(('.xls', '.xlsx')):
        data = pd.read_excel(data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    elif data_path.lower().endswith('.json'): # Added JSON support
        data = pd.read_json(data_path)
        data = data.fillna('') # Simple fill for JSON, might need more sophisticated handling
    else:
        raise ValueError("Unsupported file format. Only CSV, Excel, and JSON are supported.")
        
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not in data. Available columns: {', '.join(data.columns)}")
        
    if data.isna().any().any():
        print("Warning: Missing values detected in data. Applying simple imputation (mean/mode).")
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())
            else:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else '')
    
    X_train_np, X_test_np, y_train, y_test, preprocessors = preprocess_data(
        data, target_column, categorical_columns, numerical_columns, random_state=random_state
    )
    
    unique_y_values = np.unique(y_train)
    is_classification_task = len(unique_y_values) < 20 or not pd.api.types.is_numeric_dtype(y_train)

    if models_to_try is None:
        models_to_try = MODEL_CATEGORIES['classification'] if is_classification_task else MODEL_CATEGORIES['regression']
    
    # Refined param_grids
    param_grids = {
        'logistic_regression': {'C': [0.1, 1.0, 10.0], 'solver': ['liblinear'], 'max_iter': [1000, 2000]},
        'decision_tree': {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1,2,5]},
        'random_forest_classifier': {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'random_forest_regressor': {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        'knn_classifier': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        'svm_classifier': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'linear_regression': {'fit_intercept': [True, False]}
    }
    
    scoring_metric = metric
    if scoring_metric == 'auto':
        scoring_metric = 'f1_weighted' if is_classification_task else 'neg_mean_squared_error'
    
    best_cv_score = -float('inf')
    best_model_instance = None
    best_model_params = None
    best_model_type_name = None
    all_model_run_results = []
    
    for model_type_key in models_to_try:
        if model_type_key not in MODEL_TYPES:
            print(f"Warning: Model type {model_type_key} is not supported. Skipping.")
            continue
        
        print(f"Training and optimizing {model_type_key} model...")
        model_class_ref = MODEL_TYPES[model_type_key]
        current_param_grid = param_grids.get(model_type_key, {})
        
        try:
            # Ensure n_jobs is set if model supports it
            model_init_params = {}
            model_signature = inspect.signature(model_class_ref.__init__)
            if 'random_state' in model_signature.parameters: model_init_params['random_state'] = random_state
            if 'n_jobs' in model_signature.parameters and model_type_key != 'svm_classifier': model_init_params['n_jobs'] = -1


            grid_search_cv = GridSearchCV(
                model_class_ref(**model_init_params), current_param_grid, cv=cv,
                scoring=scoring_metric, n_jobs=(1 if model_type_key == 'svm_classifier' else -1) # SVM can be slow with n_jobs=-1 for some kernels
            )
            grid_search_cv.fit(X_train_np, y_train)
            
            current_best_estimator = grid_search_cv.best_estimator_
            y_pred_test = current_best_estimator.predict(X_test_np)
            
            test_set_score = f1_score(y_test, y_pred_test, average='weighted') if is_classification_task \
                else -mean_squared_error(y_test, y_pred_test) # Use negative MSE for regressors

            current_model_metrics = evaluate_model(current_best_estimator, X_test_np, y_test, model_type_key)

            all_model_run_results.append({
                'model_type': model_type_key, 'best_params': grid_search_cv.best_params_,
                'cv_score': grid_search_cv.best_score_, 'test_score': test_set_score, # Using consistent scoring direction
                'metrics_on_test': current_model_metrics
            })
            
            if grid_search_cv.best_score_ > best_cv_score:
                best_cv_score = grid_search_cv.best_score_
                best_model_instance = current_best_estimator
                best_model_params = grid_search_cv.best_params_
                best_model_type_name = model_type_key
            
            print(f"  {model_type_key} optimization complete. CV Score: {grid_search_cv.best_score_:.4f}, Test Score: {test_set_score:.4f}")
            
        except Exception as e_grid:
            print(f"Error training {model_type_key}: {e_grid}")
    
    if best_model_instance is None:
        raise ValueError("No valid model could be trained successfully.")
    
    final_model_name = f"automl_{best_model_type_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    final_metadata = {
        'model_name': final_model_name, 'model_type': best_model_type_name,
        'model_params': best_model_params, 'cv_score': best_cv_score,
        'all_models_results': all_model_run_results, # Store results of all tried models
        'is_classification': is_classification_task, 'metric_used': scoring_metric,
        'created_by': 'auto_model_selection', 'random_state': random_state,
        'target_column': target_column, 'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'scaled_numerical_columns': preprocessors.get('scaled_numerical_columns'),
        'feature_names_in': list(X.columns) if isinstance(X, pd.DataFrame) else None
    }
    
    final_model_path = os.path.join(MODELS_DIR, f"{final_model_name}.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump((best_model_instance, preprocessors, final_metadata), f)
    print(f"Best auto-selected model '{final_model_name}' saved to {final_model_path}")
    
    return {
        'model_name': final_model_name, 'model_type': best_model_type_name,
        'model_path': final_model_path, 'model': best_model_instance,
        'preprocessors': preprocessors, 'params': best_model_params,
        'cv_score': best_cv_score, 'is_classification': is_classification_task,
        'all_models_results': all_model_run_results
    }


def explain_model_prediction(model_name: str, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
    """
    Explains a model's prediction (basic implementation).

    Args:
        model_name: Name of the model.
        input_data: Input data (single dictionary or DataFrame for multiple).

    Returns:
        Explanation result, including feature importance and contributions.
    """
    model, preprocessors, metadata = load_model(model_name)
    
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise TypeError("input_data must be a dictionary or pandas DataFrame.")

    # Apply preprocessing (consistent with predict function)
    input_df_processed = input_df.copy() # Start with a copy of the original input structure
    
    # Use feature names from metadata if available
    feature_names_in = metadata.get("feature_names_in", None)
    if feature_names_in is None and hasattr(model, 'feature_names_in_'):
        feature_names_in = model.feature_names_in_

    if feature_names_in is not None:
        # Ensure all necessary columns are present and in order
        current_cols = input_df_processed.columns.tolist()
        if not all(fn in current_cols for fn in feature_names_in):
            raise ValueError(f"Input data missing required features. Expected: {feature_names_in}")
        input_df_processed = input_df_processed[feature_names_in]
    else:
        print("Warning: Feature names for model input not found in metadata. Assuming input_data columns are correct and ordered.")


    if 'label_encoders' in preprocessors:
        for col, encoder in preprocessors['label_encoders'].items():
            if col in input_df_processed.columns and col != metadata.get("target_column"):
                input_df_processed[col] = input_df_processed[col].astype(str)
                input_df_processed[col] = input_df_processed[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
    
    if 'scaler' in preprocessors and preprocessors['scaler'] is not None:
        scaled_num_cols = metadata.get('scaled_numerical_columns', [])
        cols_to_scale = [col for col in scaled_num_cols if col in input_df_processed.columns]
        if cols_to_scale:
            input_df_processed[cols_to_scale] = preprocessors['scaler'].transform(input_df_processed[cols_to_scale])
    
    X_explain = input_df_processed.values
    prediction = model.predict(X_explain)
    
    feature_importance_dict = {}
    feature_names = feature_names_in if feature_names_in else input_df_processed.columns.tolist()

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
    elif hasattr(model, 'coef_'):
        coefs = model.coef_
        if coefs.ndim == 1: # Simple linear model or binary logistic
            feature_importance_dict = {name: float(coef) for name, coef in zip(feature_names, coefs)}
        else: # Multiclass logistic, etc. Take mean of absolute coefs per feature
            feature_importance_dict = {name: float(np.mean(np.abs(coefs[:, i]))) for i, name in enumerate(feature_names)}
    
    # Simplified feature contributions (for linear models or tree-based with LIME/SHAP this would be more complex)
    feature_contributions = []
    if feature_importance_dict and not input_df_processed.empty:
        # This is a very basic approximation of contribution
        # For proper contributions, tools like SHAP or LIME are needed.
        first_input_sample = input_df_processed.iloc[0]
        for feature_name, importance_value in feature_importance_dict.items():
            if feature_name in first_input_sample: # Ensure feature exists in the sample
                original_value = input_df[feature_name].iloc[0] # Get original value before preprocessing for display
                processed_value = first_input_sample[feature_name]
                # Basic "contribution" might be feature_value * importance (for some models)
                # This is highly model-dependent and simplified here.
                contribution_approx = processed_value * importance_value 
                feature_contributions.append({
                    'feature': feature_name,
                    'original_value': original_value, # Display original value
                    'processed_value': float(processed_value), # Actual value used by model
                    'importance': float(importance_value),
                    'contribution_approximation': float(contribution_approx)
                })
    
    return {
        'prediction': prediction.tolist(),
        'feature_importance': feature_importance_dict,
        'feature_contributions': feature_contributions, # Simplified
        'input_data_processed': input_df_processed.to_dict('records')
    }


def compare_models(model_names: List[str], test_data_path: str, target_column: str) -> Dict[str, Any]:
    """
    Compares multiple models on a test set.

    Args:
        model_names: List of model names to compare.
        test_data_path: Path to the test data file.
        target_column: Name of the target column.

    Returns:
        Comparison results dictionary.
    """
    if test_data_path.lower().endswith('.csv'):
        test_data = pd.read_csv(test_data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    elif test_data_path.lower().endswith(('.xls', '.xlsx')):
        test_data = pd.read_excel(test_data_path, keep_default_na=False, na_values=['NaN', 'N/A', 'NA', 'nan', 'null'])
    else:
        raise ValueError("Unsupported test data file format. Only CSV and Excel are supported.")
        
    if target_column not in test_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in test data.")
        
    y_test_actual = test_data[target_column]
    
    comparison_results_list = []
    for model_name_item in model_names:
        try:
            model_obj, preprocessors, metadata = load_model(model_name_item)
            
            # Prepare X_test based on this model's features
            feature_cols_for_model = metadata.get("feature_names_in", test_data.drop(columns=[target_column]).columns.tolist())
            
            # Ensure all required feature columns are in test_data
            missing_model_features = set(feature_cols_for_model) - set(test_data.columns)
            if missing_model_features:
                raise ValueError(f"Test data is missing columns required by model {model_name_item}: {missing_model_features}")

            X_test_current_model_df = test_data[feature_cols_for_model].copy()


            # Apply preprocessing specific to this model
            if 'label_encoders' in preprocessors:
                for col, encoder in preprocessors['label_encoders'].items():
                    if col in X_test_current_model_df.columns and col != target_column:
                         X_test_current_model_df[col] = X_test_current_model_df[col].astype(str).apply(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                        )
            
            if 'scaler' in preprocessors and preprocessors['scaler'] is not None:
                scaled_num_cols = metadata.get('scaled_numerical_columns', [])
                cols_to_scale = [col for col in scaled_num_cols if col in X_test_current_model_df.columns]
                if cols_to_scale:
                    X_test_current_model_df[cols_to_scale] = preprocessors['scaler'].transform(X_test_current_model_df[cols_to_scale])

            X_test_values = X_test_current_model_df.values
            y_pred_values = model_obj.predict(X_test_values)
            
            is_classifier_model = hasattr(model_obj, "predict_proba")
            current_metrics = evaluate_model(model_obj, X_test_values, y_test_actual, metadata.get("model_type","unknown")) # Use evaluate_model
            
            comparison_results_list.append({
                "model_name": model_name_item, "is_classifier": is_classifier_model,
                "metrics": current_metrics
            })
        except Exception as e_comp:
            print(f"Error evaluating model {model_name_item} during comparison: {e_comp}")
            comparison_results_list.append({"model_name": model_name_item, "error": str(e_comp)})
    
    # Determine best classifier and regressor from results
    best_classifier_name, best_regressor_name = None, None
    highest_f1, highest_r2 = -float('inf'), -float('inf')

    for res_item in comparison_results_list:
        if "error" not in res_item:
            if res_item["is_classifier"] and res_item["metrics"].get("f1", -1) > highest_f1:
                highest_f1 = res_item["metrics"]["f1"]
                best_classifier_name = res_item["model_name"]
            elif not res_item["is_classifier"] and res_item["metrics"].get("r2", -1) > highest_r2:
                highest_r2 = res_item["metrics"]["r2"]
                best_regressor_name = res_item["model_name"]
                
    return {
        "models": comparison_results_list, "test_data_path": test_data_path,
        "target_column": target_column, "timestamp": datetime.datetime.now().isoformat(),
        "best_classifier": best_classifier_name, "best_regressor": best_regressor_name
    }


def optimize_ensemble_weights(
    base_models: List[str], X: pd.DataFrame, y: pd.Series, n_trials: int = 10, cv_folds: int = 3 # Reduced CV for speed
) -> Optional[List[float]]:
    """Optimizes weights for a voting ensemble (simplified)."""
    from sklearn.model_selection import cross_val_score
    
    loaded_models_list = []
    for model_name_str in base_models:
        model_obj, _, _ = load_model(model_name_str)
        loaded_models_list.append(model_obj)
    
    if not loaded_models_list: return None

    is_classification_ensemble = hasattr(loaded_models_list[0], 'predict_proba')
    best_score_ensemble = -float('inf')
    optimal_weights: Optional[List[float]] = None
    
    for _ in range(n_trials): # Random search for weights
        current_weights = np.random.dirichlet(np.ones(len(loaded_models_list)))
        
        if is_classification_ensemble:
            ensemble_clf = VotingClassifier(
                estimators=[(f"model_{i}", m) for i, m in enumerate(loaded_models_list)],
                weights=current_weights, voting='soft' # Prefer soft voting if possible
            )
            scoring_metric = 'f1_weighted'
        else:
            ensemble_clf = VotingRegressor(
                estimators=[(f"model_{i}", m) for i, m in enumerate(loaded_models_list)],
                weights=current_weights
            )
            scoring_metric = 'r2'
            
        try:
            # Note: X should be preprocessed appropriately before this function if models expect it
            cv_score_mean = np.mean(cross_val_score(ensemble_clf, X, y, cv=cv_folds, scoring=scoring_metric))
            if cv_score_mean > best_score_ensemble:
                best_score_ensemble = cv_score_mean
                optimal_weights = current_weights.tolist()
        except Exception as e_cv:
            print(f"Error during ensemble weight optimization CV: {e_cv}")
            continue # Try next set of weights
            
    return optimal_weights


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes memory usage of a DataFrame, especially useful for large datasets.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame optimized for memory.
    """
    start_mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Original DataFrame memory usage: {start_mem_mb:.2f} MB")
    
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if pd.api.types.is_integer_dtype(col_type):
            c_min, c_max = optimized_df[col].min(), optimized_df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
            else:
                optimized_df[col] = optimized_df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(col_type):
            # Downcast floats to float32 if possible without significant precision loss
            # This is a simple downcast; more sophisticated checks could be added
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
            # Convert object columns that are mostly unique strings to 'category' type
            if optimized_df[col].nunique() / len(optimized_df[col]) < 0.5: # Heuristic
                 try:
                    optimized_df[col] = optimized_df[col].astype('category')
                 except TypeError: # If column contains mixed types that cannot be categorized
                    pass

    end_mem_mb = optimized_df.memory_usage(deep=True).sum() / 1024**2
    reduction_pct = 100 * (start_mem_mb - end_mem_mb) / start_mem_mb if start_mem_mb > 0 else 0
    print(f"Optimized DataFrame memory usage: {end_mem_mb:.2f} MB, reduced by {reduction_pct:.1f}%")
    
    return optimized_df


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_type: str) -> Dict[str, Any]:
    """
    Evaluates model performance and returns relevant metrics.

    Args:
        model: Trained model object.
        X_test: Test set features.
        y_test: Test set labels.
        model_type: String identifier for the model type (e.g., "logistic_regression").

    Returns:
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {}

    # Determine if classification or regression based on model_type or attributes
    is_classifier = "classifier" in model_type or "logreg" in model_type or \
                    hasattr(model, 'predict_proba') or isinstance(model, (SVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, MultinomialNB))
    
    if is_classifier:
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        }
        if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2: # Binary classification ROC AUC
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            except Exception as e_roc:
                print(f"Could not compute ROC AUC: {e_roc}")
        
        # Detailed classification report (text format for now, can be parsed if needed)
        try:
            # Ensure target names are strings for classification_report
            unique_labels = np.unique(np.concatenate((y_test, y_pred))).astype(str)
            report = classification_report(y_test.astype(str), y_pred.astype(str), target_names=unique_labels, output_dict=True, zero_division=0)
            
            # Simplify report for JSON
            simple_report = {}
            for lbl, mtrcs in report.items():
                if isinstance(mtrcs, dict):
                    simple_report[lbl] = {k_m: float(v_m) for k_m, v_m in mtrcs.items() if isinstance(v_m, (int, float))}
                elif isinstance(mtrcs, (int, float)):
                     simple_report[lbl] = float(mtrcs)
            metrics["classification_report_dict"] = simple_report

        except Exception as e_report:
            print(f"Could not generate classification report: {e_report}")
            metrics["classification_report_error"] = str(e_report)

    else: # Regression
        metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "explained_variance": float(explained_variance_score(y_test, y_pred))
        }
    return metrics


def process_large_dataset(
    data_path: str, 
    processing_func: Callable[[pd.DataFrame, Any], Any], # Callable type hint
    chunk_size: int = 100000,
    **kwargs
) -> Any: # Return type depends on processing_func
    """
    Processes large dataset files in chunks.

    Args:
        data_path: Path to the data file.
        processing_func: Function to process each data chunk.
        chunk_size: Number of rows per chunk.
        **kwargs: Additional arguments for the processing function.
        
    Returns:
        Result of processing, type depends on processing_func.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    file_extension = os.path.splitext(data_path)[1].lower()
    processed_results = []
    
    try:
        if file_extension == '.csv':
            file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
            print(f"Processing CSV file: {data_path}, Size: {file_size_mb:.2f} MB")
            
            if file_size_mb < 100: # Process smaller files in one go
                df_full = pd.read_csv(data_path)
                return processing_func(df_full, **kwargs)
            
            chunk_iterator = pd.read_csv(data_path, chunksize=chunk_size)
            for i, chunk_df in enumerate(chunk_iterator):
                print(f"Processing chunk {i+1}, Size: {len(chunk_df)} rows")
                processed_results.append(processing_func(chunk_df, **kwargs))
                
        elif file_extension in ['.xlsx', '.xls']: # Excel files are usually smaller
            df_excel = pd.read_excel(data_path)
            return processing_func(df_excel, **kwargs)
            
        elif file_extension == '.parquet': # Parquet can be read efficiently
            # Parquet can be read in chunks too if needed, but often pandas handles large parquet well
            df_parquet = pd.read_parquet(data_path) 
            # Simple example: if very large, could implement chunking similar to CSV
            if len(df_parquet) > chunk_size * 2: # Arbitrary threshold for example
                 for i in range(0, len(df_parquet), chunk_size):
                    chunk_df = df_parquet.iloc[i:i + chunk_size]
                    print(f"Processing Parquet chunk {i//chunk_size + 1}")
                    processed_results.append(processing_func(chunk_df, **kwargs))
            else:
                return processing_func(df_parquet, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        # Consolidate results if possible (e.g., list of DataFrames)
        if processed_results:
            if all(isinstance(res, pd.DataFrame) for res in processed_results):
                return pd.concat(processed_results)
            # Add other consolidation logic if needed for other types of results
        return processed_results # Return list of results if not DataFrames
            
    except Exception as e:
        print(f"Error processing large dataset: {str(e)}")
        raise


def batch_predict(
    model_name: str, data_path: str,
    output_path: Optional[str] = None, chunk_size: int = 100000
) -> str:
    """
    Performs batch prediction on a large dataset to avoid memory issues.

    Args:
        model_name: Name of the model.
        data_path: Path to the input data file.
        output_path: Optional path for the output results.
        chunk_size: Number of rows to process in each batch.
        
    Returns:
        Path to the output file containing predictions.
    """
    import csv # Local import for file operations
    
    print(f"Starting batch prediction with model: {model_name}")
    
    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(os.path.basename(data_path))
        output_path = f"{base}_predictions_{timestamp}.csv"
    
    model, preprocessors, metadata = load_model(model_name)
    
    # Determine feature columns from metadata or model if possible
    feature_columns = metadata.get("feature_names_in", metadata.get("categorical_columns", []) + metadata.get("numerical_columns", []))
    if not feature_columns and hasattr(model, 'feature_names_in_'):
        feature_columns = model.feature_names_in_
    if not feature_columns:
        # As a last resort, try to infer from a sample of the data, excluding known target if any
        # This is risky and should ideally be avoided by having features in metadata.
        sample_df_features = pd.read_csv(data_path, nrows=5) if data_path.lower().endswith('.csv') else pd.read_excel(data_path, nrows=5)
        if metadata.get("target_column") in sample_df_features.columns:
            feature_columns = sample_df_features.drop(columns=[metadata.get("target_column")]).columns.tolist()
        else:
            feature_columns = sample_df_features.columns.tolist()
        print(f"Warning: Feature columns inferred from data sample: {feature_columns}")


    def _process_chunk_for_prediction(chunk_df: pd.DataFrame) -> pd.DataFrame:
        if not feature_columns: # Should not happen if inference above works
             raise ValueError("Feature columns for prediction could not be determined.")
        
        # Ensure only necessary features are present and in correct order
        missing_cols = set(feature_columns) - set(chunk_df.columns)
        if missing_cols:
            raise ValueError(f"Chunk data is missing columns: {missing_cols}")
        
        X_chunk = chunk_df[feature_columns].copy()
        
        # Apply preprocessors (reusing the apply_preprocessors logic)
        X_processed_values = apply_preprocessors(
            X_chunk, preprocessors, 
            metadata.get("categorical_columns", []), 
            metadata.get("scaled_numerical_columns", metadata.get("numerical_columns", [])) # Prefer scaled_numerical_columns
        )
        
        predictions_chunk = model.predict(X_processed_values)
        
        result_df_chunk = chunk_df.copy()
        result_df_chunk['prediction'] = predictions_chunk
        return result_df_chunk
    
    try:
        file_extension = os.path.splitext(data_path)[1].lower()
        
        if file_extension == '.csv':
            first_chunk = True
            with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
                for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                    result_chunk_df = _process_chunk_for_prediction(chunk)
                    result_chunk_df.to_csv(f_out, header=first_chunk, index=False, mode='a')
                    first_chunk = False
        else: # For Excel or other formats that might not support easy chunking in read_
              # We use the generic process_large_dataset which loads then chunks if large
            all_results_df = process_large_dataset(
                data_path=data_path,
                processing_func=_process_chunk_for_prediction,
                chunk_size=chunk_size
            )
            if isinstance(all_results_df, pd.DataFrame):
                all_results_df.to_csv(output_path, index=False)
            elif isinstance(all_results_df, list) and all(isinstance(item, pd.DataFrame) for item in all_results_df): # If list of DFs
                pd.concat(all_results_df).to_csv(output_path, index=False)

        print(f"Batch prediction complete. Results saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error during batch prediction: {str(e)}")
        raise


def apply_preprocessors(
    X: pd.DataFrame,
    preprocessors: Dict[str, Any],
    categorical_columns: Optional[List[str]],
    numerical_columns: Optional[List[str]] # Should be the list of columns that were scaled
) -> np.ndarray:
    """
    Applies preprocessors to input data.

    Args:
        X: Input features DataFrame.
        preprocessors: Dictionary of preprocessors.
        categorical_columns: List of categorical feature names.
        numerical_columns: List of numerical feature names that were scaled.
        
    Returns:
        Processed feature matrix as a NumPy array.
    """
    X_processed = X.copy()
    
    if 'label_encoders' in preprocessors and categorical_columns:
        for col in categorical_columns:
            if col in X_processed.columns and col in preprocessors['label_encoders']:
                le = preprocessors['label_encoders'][col]
                X_processed[col] = X_processed[col].astype(str)
                
                # Handle unseen labels by mapping them to a default value (e.g., -1 or a specific category)
                # or by fitting the encoder on the new data as well (less ideal for test set)
                unseen_mask = ~X_processed[col].isin(le.classes_)
                if unseen_mask.any():
                    print(f"Warning: Column {col} has unseen labels: {X_processed[col][unseen_mask].unique()}. Mapping to -1.")
                    # Option 1: Map to a default like -1 (if your model can handle it or if it's okay to treat as missing/other)
                    # X_processed.loc[unseen_mask, col] = -1 
                    # Option 2: Map to the most frequent class (less disruptive if -1 is bad)
                    most_frequent_class_val = 0 # Default if no known classes
                    if len(le.classes_) > 0:
                        # Get the encoded value of the most frequent class
                        most_frequent_original_label = pd.Series(le.inverse_transform(X_processed[col][~unseen_mask].astype(int))).mode()
                        if not most_frequent_original_label.empty:
                             most_frequent_class_val = le.transform(most_frequent_original_label[:1])[0]
                        else: # If all values were unseen, or no mode, use first known class
                             most_frequent_class_val = 0 # Default to encoded 0
                    
                    X_processed.loc[unseen_mask, col] = str(le.classes_[most_frequent_class_val] if len(le.classes_) > most_frequent_class_val else le.classes_[0] if len(le.classes_)>0 else 'unknown_category_placeholder')


                # Transform only known labels, leave others (or handle as above)
                known_mask = X_processed[col].isin(le.classes_)
                X_processed.loc[known_mask, col] = le.transform(X_processed.loc[known_mask, col])
                # After transform, unseen (now -1 or placeholder) might need specific handling if model expects numeric
                if (X_processed[col] == 'unknown_category_placeholder').any():
                    # Convert to a numeric placeholder if necessary, e.g. if column must be int
                    # This depends on how your model handles such cases. For now, assuming it might remain string/object
                    # If conversion to int is needed:
                    # X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce').fillna(-1).astype(int)
                    pass


    if 'scaler' in preprocessors and preprocessors['scaler'] is not None and numerical_columns:
        # Ensure only columns present in X_processed and meant for scaling are used
        cols_to_scale = [col for col in numerical_columns if col in X_processed.columns]
        if cols_to_scale:
            X_processed[cols_to_scale] = preprocessors['scaler'].transform(X_processed[cols_to_scale])
    
    return X_processed.values


def parallel_process(
    func: Callable, items: List[Any], n_jobs: Optional[int] = None,
    backend: str = 'multiprocessing', **kwargs
) -> List[Any]:
    """
    Executes a function in parallel to speed up processing.

    Args:
        func: The function to execute in parallel.
        items: A list of items to process.
        n_jobs: Number of parallel tasks (None uses all available CPU cores minus one).
        backend: Parallel backend ('multiprocessing', 'threading', 'loky').
        **kwargs: Additional arguments to pass to func.
        
    Returns:
        A list of processing results.
    """
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1) # Leave one core for the system
    
    print(f"Starting parallel processing with {n_jobs} worker(s) using {backend} backend.")
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(func)(item, **kwargs) for item in items
    )
    return results


def parallel_feature_selection(
    data: pd.DataFrame, target_column: str, n_jobs: int = -1,
    method: str = 'mutual_info', top_k: int = 10
) -> List[str]:
    """
    Accelerates feature selection using parallel computation.

    Args:
        data: Input DataFrame.
        target_column: Name of the target column.
        n_jobs: Number of parallel tasks (-1 uses all available CPUs).
        method: Feature selection method ('mutual_info', 'f_test', 'chi2').
        top_k: Number of top features to select.
        
    Returns:
        List of selected feature names.
    """
    from sklearn.feature_selection import (SelectKBest, f_classif, f_regression,
                                         chi2, mutual_info_classif,
                                         mutual_info_regression)
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    is_classification = y.dtype == 'object' or y.nunique() < 0.1 * len(y) # Heuristic
    
    score_func_map = {
        'mutual_info': mutual_info_classif if is_classification else mutual_info_regression,
        'f_test': f_classif if is_classification else f_regression,
        'chi2': chi2 if is_classification else None # chi2 only for classification
    }
    
    score_func_selected = score_func_map.get(method)
    if score_func_selected is None:
        raise ValueError(f"Unsupported feature selection method: {method} for this task type.")

    if method == 'chi2' and is_classification:
        # Ensure all features are non-negative for chi2
        if (X < 0).any().any():
            print("Warning: Negative values found in features for chi2. Applying offset.")
            X = X - X.min().min() # Simple offset, might need more sophisticated handling

    selector = SelectKBest(score_func=score_func_selected, k=min(top_k, X.shape[1]))
    # Note: SelectKBest itself doesn't directly use n_jobs for fitting all scorers.
    # Parallelism here would be if the score_func itself is parallelizable or if used within a parallel CV.
    selector.fit(X, y)
    
    feature_scores = sorted(list(zip(X.columns, selector.scores_)), key=lambda x: x[1], reverse=True)
    return [f[0] for f in feature_scores[:top_k]]


def parallel_train_model(
    model_type: str, data: pd.DataFrame, target_column: str,
    categorical_columns: Optional[List[str]] = None,
    numerical_columns: Optional[List[str]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = -1, cv: int = 5, random_state: int = 42
) -> Dict[str, Any]:
    """
    Accelerates model training and cross-validation using parallel computation.
    (Note: Cross-validation part is parallelized if model or CV supports n_jobs)
    """
    from sklearn.model_selection import cross_val_score # Local import for this specific use
    
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model type: {model_type}. Valid types: {list(MODEL_TYPES.keys())}")
    
    X_train_np, X_test_np, y_train, y_test, preprocessors = preprocess_data(
        data, target_column, categorical_columns, numerical_columns, random_state=random_state
    )
    
    current_model_params = DEFAULT_MODEL_PARAMS.get(model_type, {}).copy()
    if model_params: current_model_params.update(model_params)

    model_class_ref = MODEL_TYPES[model_type]
    model_signature = inspect.signature(model_class_ref.__init__)
    if 'random_state' in model_signature.parameters and 'random_state' not in current_model_params:
        current_model_params['random_state'] = random_state
    # Pass n_jobs to model if it supports it (e.g., RandomForest)
    if 'n_jobs' in model_signature.parameters:
        current_model_params['n_jobs'] = n_jobs 
        
    model_instance = model_class_ref(**current_model_params)
    
    print(f"Performing {cv}-fold cross-validation with {n_jobs if n_jobs != -1 else 'all'} CPU core(s).")
    scoring_metric = 'accuracy' if "classifier" in model_type or "logreg" in model_type else 'neg_mean_squared_error'
    
    # cross_val_score itself can use n_jobs for some estimators
    cv_scores_array = cross_val_score(model_instance, X_train_np, y_train, cv=cv, scoring=scoring_metric, n_jobs=n_jobs)
    
    model_instance.fit(X_train_np, y_train) # Train final model on full training data
    final_metrics = evaluate_model(model_instance, X_test_np, y_test, model_type)
    
    if scoring_metric == 'neg_mean_squared_error':
        final_metrics['cv_mean_neg_mse'] = cv_scores_array.mean()
        final_metrics['cv_std_neg_mse'] = cv_scores_array.std()
    else: # Assuming accuracy or other classification metric
        final_metrics[f'cv_mean_{scoring_metric}'] = cv_scores_array.mean()
        final_metrics[f'cv_std_{scoring_metric}'] = cv_scores_array.std()
    
    return {'model': model_instance, 'preprocessors': preprocessors, 'metrics': final_metrics}