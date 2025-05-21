# test_ensemble.py
import os
import sys
import pandas as pd
import numpy as np
import time
import json
from sklearn.metrics import accuracy_score, r2_score

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型相关功能
from ml_models import (
    load_model, list_available_models, create_ensemble_model,
    auto_model_selection, explain_model_prediction, compare_models,
    save_model_with_version, list_model_versions
)

def test_ensemble_models():
    """测试创建集成模型功能"""
    print("\n=== 测试集成模型功能 ===")
    
    # 获取可用模型
    models = list_available_models()
    model_names = [model['name'] for model in models]
    
    # 找出分类模型和回归模型
    classification_models = []
    regression_models = []
    
    for model_name in model_names:
        try:
            model, _, _ = load_model(model_name)
            # 判断是分类还是回归
            if hasattr(model, "predict_proba"):
                classification_models.append(model_name)
            else:
                regression_models.append(model_name)
        except Exception as e:
            print(f"加载模型 {model_name} 时出错: {e}")
    
    print(f"找到 {len(classification_models)} 个分类模型: {classification_models}")
    print(f"找到 {len(regression_models)} 个回归模型: {regression_models}")
    
    # 创建投票集成分类器
    if len(classification_models) >= 2:
        print("\n创建投票集成分类器...")
        voting_classifier = create_ensemble_model(
            base_models=classification_models[:2],  # 使用前两个分类模型
            ensemble_type='voting',
            save_name='voting_classifier_test'
        )
        print(f"创建成功: {voting_classifier['model_name']}")
    
    # 创建投票集成回归器
    if len(regression_models) >= 2:
        print("\n创建投票集成回归器...")
        voting_regressor = create_ensemble_model(
            base_models=regression_models[:2],  # 使用前两个回归模型
            ensemble_type='voting',
            save_name='voting_regressor_test'
        )
        print(f"创建成功: {voting_regressor['model_name']}")
    
    # 创建堆叠集成分类器
    if len(classification_models) >= 2:
        print("\n创建堆叠集成分类器...")
        stacking_classifier = create_ensemble_model(
            base_models=classification_models[:2],  # 使用前两个分类模型
            ensemble_type='stacking',
            save_name='stacking_classifier_test'
        )
        print(f"创建成功: {stacking_classifier['model_name']}")
    
    # 创建堆叠集成回归器
    if len(regression_models) >= 2:
        print("\n创建堆叠集成回归器...")
        stacking_regressor = create_ensemble_model(
            base_models=regression_models[:2],  # 使用前两个回归模型
            ensemble_type='stacking',
            save_name='stacking_regressor_test'
        )
        print(f"创建成功: {stacking_regressor['model_name']}")
    
    return True

def test_model_versioning():
    """测试模型版本管理功能"""
    print("\n=== 测试模型版本管理功能 ===")
    
    # 获取可用模型
    models = list_available_models()
    
    if not models:
        print("没有找到可用模型，跳过测试")
        return False
    
    # 选择第一个模型
    model_name = models[0]['name']
    
    # 创建模型版本
    print(f"\n为模型 {model_name} 创建版本...")
    version_info = save_model_with_version(
        model=models[0]['model'],
        model_name=model_name,
        metadata={
            "description": "测试版本",
            "author": "自动测试脚本",
            "accuracy": 0.95
        }
    )
    print(f"版本创建成功: {version_info['version']}")
    
    # 列出模型版本
    print(f"\n列出模型 {model_name} 的所有版本...")
    versions = list_model_versions(model_name)
    print(f"找到 {len(versions)} 个版本")
    for i, ver in enumerate(versions):
        print(f"  {i+1}. 版本: {ver.get('version')}, 时间: {ver.get('timestamp')}")
    
    return True

def test_model_explanation():
    """测试模型解释功能"""
    print("\n=== 测试模型解释功能 ===")
    
    # 获取可用模型
    models = list_available_models()
    
    if not models:
        print("没有找到可用模型，跳过测试")
        return False
    
    # 选择一个有特征重要性的模型
    model_name = None
    for model in models:
        if model['type'] in ["random_forest_classifier", "random_forest_regressor", "decision_tree"]:
            model_name = model['name']
            break
    
    if not model_name:
        model_name = models[0]['name']  # 如果没有找到RF或DT，使用第一个模型
    
    print(f"使用模型 {model_name} 进行解释测试")
    
    # 创建测试数据
    input_data = {
        "PM2.5": 50,
        "PM10": 60,
        "SO2": 20,
        "CO": 0.8,
        "NO2": 40,
        "O3": 30
    }
    
    # 解释预测
    print("\n解释预测结果...")
    explanation = explain_model_prediction(model_name, input_data)
    
    # 输出预测和特征重要性
    print(f"预测结果: {explanation['prediction']}")
    
    if explanation['feature_importance']:
        print("\n特征重要性:")
        for feature, importance in sorted(
            explanation['feature_importance'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]:  # 显示前5个特征
            print(f"  {feature}: {importance:.4f}")
    
    if explanation['feature_contributions']:
        print("\n特征贡献:")
        for item in explanation['feature_contributions'][:5]:  # 显示前5个贡献
            print(f"  {item['feature']}: 值={item['value']:.4f}, 贡献={item['contribution']:.4f}")
    
    return True

def test_model_comparison():
    """测试模型比较功能"""
    print("\n=== 测试模型比较功能 ===")
    
    # 获取可用模型
    models = list_available_models()
    
    # 分离分类器和回归器
    classifiers = []
    regressors = []
    
    for model in models:
        model_obj, _, _ = load_model(model['name'])
        if hasattr(model_obj, "predict_proba"):
            classifiers.append(model['name'])
        else:
            regressors.append(model['name'])
    
    # 测试数据路径
    data_path = "北京市空气质量数据.xlsx"
    if not os.path.exists(data_path):
        print(f"测试数据 {data_path} 不存在，跳过比较测试")
        return False
    
    # 比较分类器
    if len(classifiers) >= 2:
        print("\n比较分类器...")
        comparison = compare_models(
            model_names=classifiers[:3],  # 最多使用前3个分类器
            test_data_path=data_path,
            target_column="质量等级"
        )
        
        print(f"最佳分类器: {comparison.get('best_classifier')}")
        print("\n分类器比较结果:")
        for model in comparison['models']:
            if 'error' in model:
                print(f"  {model['model_name']}: 错误 - {model['error']}")
            else:
                print(f"  {model['model_name']}: 准确率={model['metrics'].get('accuracy', 0):.4f}, F1={model['metrics'].get('f1', 0):.4f}")
    
    # 比较回归器
    if len(regressors) >= 2:
        print("\n比较回归器...")
        comparison = compare_models(
            model_names=regressors[:3],  # 最多使用前3个回归器
            test_data_path=data_path,
            target_column="PM2.5"
        )
        
        print(f"最佳回归器: {comparison.get('best_regressor')}")
        print("\n回归器比较结果:")
        for model in comparison['models']:
            if 'error' in model:
                print(f"  {model['model_name']}: 错误 - {model['error']}")
            else:
                print(f"  {model['model_name']}: RMSE={model['metrics'].get('rmse', 0):.4f}, R²={model['metrics'].get('r2', 0):.4f}")
    
    return True

def prepare_data_for_automl():
    """准备测试数据，处理时间戳类型"""
    print("\n准备自动模型选择的测试数据...")
    
    # 检查数据文件是否存在
    data_path = "北京市空气质量数据.xlsx"
    if not os.path.exists(data_path):
        print(f"数据文件 {data_path} 不存在，跳过数据准备")
        return False
    
    try:
        # 加载数据
        df = pd.read_excel(data_path)
        
        # 检查是否有时间列并处理
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            print(f"发现时间列: {date_columns.tolist()}")
            for col in date_columns:
                # 添加提取的时间特征
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df[f'{col}_hour'] = df[col].dt.hour if hasattr(df[col].dt, 'hour') else 0
                
                # 删除原始时间列
                df = df.drop(columns=[col])
        
        # 保存处理后的数据
        processed_path = "北京市空气质量数据_processed.csv"
        df.to_csv(processed_path, index=False)
        print(f"处理后的数据已保存到 {processed_path}")
        return processed_path
    except Exception as e:
        print(f"预处理数据时出错: {e}")
        return None

def test_automl():
    """测试自动模型选择功能"""
    print("\n=== 测试自动模型选择功能 ===")
    
    # 准备数据
    data_path = prepare_data_for_automl()
    if not data_path:
        data_path = "北京市空气质量数据.xlsx"
    
    # 检查数据是否存在
    if not os.path.exists(data_path):
        print(f"测试数据 {data_path} 不存在，跳过自动模型选择测试")
        return False
    
    try:
        # 自动选择回归模型
        print("\n自动选择回归模型...")
        regression_result = auto_model_selection(
            data_path=data_path,
            target_column="PM2.5",
            models_to_try=['linear_regression', 'random_forest_regressor']
        )
        
        print(f"最佳回归模型: {regression_result['model_type']}")
        print(f"CV分数: {regression_result['cv_score']:.4f}")
        print(f"最佳参数: {regression_result['params']}")
        
        # 自动选择分类模型
        print("\n自动选择分类模型...")
        classification_result = auto_model_selection(
            data_path=data_path,
            target_column="质量等级",
            categorical_columns=["质量等级"],
            models_to_try=['logistic_regression', 'random_forest_classifier']
        )
        
        print(f"最佳分类模型: {classification_result['model_type']}")
        print(f"CV分数: {classification_result['cv_score']:.4f}")
        print(f"最佳参数: {classification_result['params']}")
        
        return True
    except Exception as e:
        print(f"自动模型选择失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 机器学习模型集成与高级功能测试 ===\n")
    start_time = time.time()
    
    # 测试集成模型功能
    try:
        test_ensemble_models()
    except Exception as e:
        print(f"测试集成模型功能失败: {e}")
    
    # 测试模型版本管理功能
    try:
        test_model_versioning()
    except Exception as e:
        print(f"测试模型版本管理功能失败: {e}")
    
    # 测试模型解释功能
    try:
        test_model_explanation()
    except Exception as e:
        print(f"测试模型解释功能失败: {e}")
    
    # 测试模型比较功能
    try:
        test_model_comparison()
    except Exception as e:
        print(f"测试模型比较功能失败: {e}")
    
    # 测试自动模型选择功能
    try:
        test_automl()
    except Exception as e:
        print(f"测试自动模型选择功能失败: {e}")
    
    end_time = time.time()
    print(f"\n=== 测试完成，耗时: {end_time - start_time:.2f}秒 ===") 