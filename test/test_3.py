import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import graphviz # Optional: for visualizing the tree structure
from sklearn.tree import export_graphviz
import pydotplus # Optional: dependency for graphviz display
import os

# --- 1. 配置环境 (Environment Setup) ---
# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# --- 2. 加载数据 (Load Data) ---
file_path = 'D:\PaddlePaddle-EfficientNetV2\PaddleClas-EfficientNet\北京市空气质量数据.xlsx' # <--- 修改为你实际的文件路径
try:
    df = pd.read_excel(file_path)
    print("数据加载成功:")
    print(df.head())
    print("\n数据信息:")
    df.info()
except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载数据时出错: {e}")
    exit()

# --- 3. 数据预处理 (Data Preprocessing) ---
# 假设最后一列是目标变量 'AQI等级'，其余为特征
# target_column = df.columns[-1] # <--- 这是错误假设！
# feature_columns = df.columns[:-1].tolist() # <--- 这包含了不合适的列

target_column = '质量等级' # <--- 修改：明确指定目标列
# 选择合适的特征列，排除非数值列、目标列和可能冗余的列（如AQI本身）
feature_columns = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3'] # <--- 修改：选择实际用于预测的特征

X = df[feature_columns]
y_raw = df[target_column]

# 将分类目标变量编码为数字
le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_ # 保存原始类别名称，方便后续解释

print(f"\n特征列: {feature_columns}")
print(f"目标列: {target_column}")
print(f"编码后的目标类别映射: {dict(zip(le.transform(class_names), class_names))}")
print(f"原始类别分布:\n{y_raw.value_counts()}")

# --- 4. 划分训练集和测试集 (Split Data) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y # stratify 保证类别比例
)
print(f"\n训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# --- 5. 任务二：绘制不同纯度度量函数图像 (Plot Impurity Measures) ---
def entropy(p):
    """计算二分类信息熵"""
    p = np.clip(p, 1e-10, 1 - 1e-10) # 避免 log(0)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def gini_index(p):
    """计算二分类基尼系数"""
    return 2 * p * (1 - p) # 等价于 1 - (p**2 + (1-p)**2)

p_values = np.linspace(0.001, 0.999, 200)
entropy_values = entropy(p_values)
gini_values = gini_index(p_values)

plt.figure(figsize=(8, 6))
plt.plot(p_values, entropy_values, label='熵 (Entropy)', linestyle='-')
plt.plot(p_values, gini_values, label='基尼系数 (Gini)', linestyle='-.') # 使用与示例图一致的线型
plt.title('二分类下的基尼系数和熵 (Gini Index and Entropy for Binary Classification)')
plt.xlabel('概率 P (Probability P)')
plt.ylabel('计算结果 (Calculated Value)')
plt.legend()
plt.grid(True, linestyle='-.')
plt.ylim(bottom=0) # 确保 Y 轴从 0 开始
plt.show()

# --- 6. 任务一 & 三：训练决策树并分析树深度影响 (Train Tree & Analyze Depth) ---
max_depths = range(1, 15) # 测试的树深度范围
train_errors = []
test_errors = []
cv_errors = [] # 5折交叉验证误差

print("\n开始分析树深度对误差的影响...")
for depth in max_depths:
    # 使用信息熵作为标准 (也可以改为 'gini')
    dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)

    # 1. 训练误差
    dt_clf.fit(X_train, y_train)
    y_train_pred = dt_clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_errors.append(1 - train_accuracy)

    # 2. 测试误差
    y_test_pred = dt_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_errors.append(1 - test_accuracy)

    # 3. 5折交叉验证误差 (在训练集上进行)
    # 注意：交叉验证评估的是模型在训练数据上的泛化能力，避免了对单一测试集划分的依赖
    cv_scores = cross_val_score(dt_clf, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_accuracy = np.mean(cv_scores)
    cv_errors.append(1 - mean_cv_accuracy)

    print(f"  深度={depth}: 训练误差={1-train_accuracy:.4f}, 测试误差={1-test_accuracy:.4f}, 5折CV误差={1-mean_cv_accuracy:.4f}")

# 绘制树深度与误差的关系图
plt.figure(figsize=(10, 7))
plt.plot(max_depths, train_errors, marker='o', linestyle='-', label='训练误差 (Training Error)')
plt.plot(max_depths, test_errors, marker='o', linestyle='-.', label='测试误差 (Test Error)')
plt.plot(max_depths, cv_errors, marker='o', linestyle='--', label='5-折交叉验证误差 (5-Fold CV Error)')
plt.title('树深度和误差 (Tree Depth vs. Error)')
plt.xlabel('树深度 (Tree Depth)')
# 保持与示例图一致的 Y 轴标签，即使 R方 用于回归
plt.ylabel('误差 (1-准确率) Error (1-Accuracy)')
plt.xticks(max_depths)
plt.legend()
plt.grid(True, linestyle='-.')
plt.show()

# --- 7. 选择最优深度并评估最终模型 (Select Optimal Depth & Evaluate Final Model) ---
# 根据上图选择一个较优的深度，通常是测试误差或交叉验证误差开始稳定或上升的拐点前的深度
# 例如，可以找交叉验证误差最小的深度
optimal_depth_cv = max_depths[np.argmin(cv_errors)]
optimal_depth_test = max_depths[np.argmin(test_errors)] # 也可以基于测试误差

# 在实践中，通常基于交叉验证选择超参数
optimal_depth = optimal_depth_cv
print(f"\n根据5折交叉验证，建议的最优深度约为: {optimal_depth}")
print(f"(测试误差最小时对应的深度为: {optimal_depth_test})")

# 使用选定的最优深度重新训练模型 (可以在整个训练集上训练)
final_clf = DecisionTreeClassifier(criterion='entropy', max_depth=optimal_depth, random_state=42)
final_clf.fit(X_train, y_train)

# 在测试集上进行最终评估
y_final_pred = final_clf.predict(X_test)
final_accuracy = accuracy_score(y_test, y_final_pred)
print(f"\n使用最优深度 {optimal_depth} 在测试集上的最终评估:")
print(f"  准确率: {final_accuracy:.4f}")
print("  分类报告:")
# target_names 使用原始类别名称
print(classification_report(y_test, y_final_pred, target_names=class_names))
print("  混淆矩阵:")
print(confusion_matrix(y_test, y_final_pred))
# 如果需要显示原始类别标签的混淆矩阵
conf_matrix = confusion_matrix(y_test, y_final_pred)
conf_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
print("\n混淆矩阵 (带标签):")
print(conf_df)


# --- 8. 任务四：决策树剪枝 (Pruning) ---
# 上述通过选择 'max_depth' 实际上就是一种预剪枝 (Pre-pruning) 策略。
# Scikit-learn 也支持基于代价复杂度剪枝 (Cost-Complexity Pruning, 'ccp_alpha') 的后剪枝，
# 但对于本实验，通过分析深度影响来选择最优深度已满足要求。
print(f"\n剪枝说明：通过限制最大深度 (max_depth={optimal_depth}) 进行了预剪枝，以防止模型过拟合。")


# --- 9. (可选) 可视化决策树 (Optional: Visualize Tree) ---
# 需要安装 graphviz (软件本身) 和 pydotplus (Python库)
# 在命令行运行: pip install pydotplus graphviz
# 可能还需要将 graphviz 的 bin 目录添加到系统 PATH
try:
    dot_data = export_graphviz(final_clf,
                               out_file=None,
                               feature_names=feature_columns,
                               class_names=class_names, # 使用原始类别名称
                               filled=True,
                               rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 保存为图片文件
    tree_image_path = 'air_quality_decision_tree.png'
    graph.write_png(tree_image_path)
    print(f"\n决策树结构已保存为图片: {tree_image_path}")
    # 如果在Jupyter Notebook中，可以直接显示
    # from IPython.display import Image
    # display(Image(graph.create_png()))
except ImportError:
    print("\n无法导入 pydotplus 或 graphviz 未正确配置，跳过决策树可视化。")
    print("请安装 graphviz 软件并运行 'pip install pydotplus'")
except Exception as e:
     print(f"\n生成决策树可视化时出错: {e}")


print("\n实验完成!")