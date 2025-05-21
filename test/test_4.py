import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split # 可以用于更灵活的数据划分
import time
import os
import warnings

# 忽略一些scikit-learn的警告信息，例如关于未收敛的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 设置Matplotlib和Seaborn的样式和中文支持
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
sns.set_theme(style="whitegrid")

print("库导入完成。")
def load_data(dataset_path='mnist.pkl.gz'):
    """
    加载 MNIST 数据集 (.pkl.gz 格式)
    Args:
        dataset_path (str): 数据集文件的路径.
    Returns:
        tuple: 包含训练集、验证集、测试集的元组 (train_set, valid_set, test_set)。
               加载失败则返回 (None, None, None)。
    """
    if not os.path.exists(dataset_path):
        print(f"错误：数据集文件未找到 '{dataset_path}'。")
        print("请确保文件存在于当前目录或提供了正确的路径。")
        print("您可以从以下链接下载MNIST数据集 (pkl.gz格式):")
        print("http://deeplearning.net/data/mnist/mnist.pkl.gz")
        return None, None, None

    print(f"正在从 '{dataset_path}' 加载数据...")
    try:
        with gzip.open(dataset_path, 'rb') as f:
            # 使用 latin1 编码以兼容可能由 Python 2 生成的 pickle 文件
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        print("数据加载成功！")
        return train_set, valid_set, test_set
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None, None, None

# --- 执行数据加载 ---
# ***********************************************************
# ** 重要: 请确保 'mnist.pkl.gz' 文件与此脚本位于同一目录 **
# ***********************************************************
dataset_path = 'mnist.pkl.gz'
train_set, valid_set, test_set = load_data(dataset_path)

# --- 数据加载失败处理 ---
if train_set is None:
    print("无法继续实验，请检查数据集文件。")
    exit() # 退出脚本

# --- 数据集结构分析 ---
X_train_raw, y_train_raw = train_set
X_valid_raw, y_valid_raw = valid_set # 验证集通常用于超参数调优，本次实验暂不直接使用，但保留
X_test_raw, y_test_raw = test_set

print("\n--- 数据集维度 ---")
print(f"原始训练集样本数: {X_train_raw.shape[0]}, 特征数: {X_train_raw.shape[1]}")
print(f"原始验证集样本数: {X_valid_raw.shape[0]}, 特征数: {X_valid_raw.shape[1]}")
print(f"原始测试集样本数: {X_test_raw.shape[0]}, 特征数: {X_test_raw.shape[1]}")
print(f"训练集标签类别: {np.unique(y_train_raw)}")
print(f"测试集标签类别: {np.unique(y_test_raw)}")

# --- (可选) 数据子集化，用于快速测试 ---
USE_SUBSET = False # 设置为 True 以使用数据子集进行快速实验
SUBSET_SIZE_TRAIN = 5000
SUBSET_SIZE_TEST = 1000

if USE_SUBSET:
    print(f"\n注意：正在使用数据子集进行快速实验！")
    print(f"训练集大小: {SUBSET_SIZE_TRAIN}, 测试集大小: {SUBSET_SIZE_TEST}")
    # 使用 train_test_split 来确保子集也包含所有类别 (如果可能)
    _, X_train, _, y_train = train_test_split(X_train_raw, y_train_raw, test_size=SUBSET_SIZE_TRAIN/len(y_train_raw), stratify=y_train_raw, random_state=42)
    _, X_test, _, y_test = train_test_split(X_test_raw, y_test_raw, test_size=SUBSET_SIZE_TEST/len(y_test_raw), stratify=y_test_raw, random_state=42)
    print(f"子集化后训练集维度: {X_train.shape}")
    print(f"子集化后测试集维度: {X_test.shape}")
else:
    X_train, y_train = X_train_raw, y_train_raw
    X_test, y_test = X_test_raw, y_test_raw
    print("\n使用完整数据集进行实验。")
    print(f"最终使用训练集维度: {X_train.shape}")
    print(f"最终使用测试集维度: {X_test.shape}")


# --- 可视化样本数据 ---
def plot_sample_images(data, labels, title, num_images=10):
    """绘制数据集中的样本图像"""
    plt.figure(figsize=(10, 3))
    plt.suptitle(title, fontsize=14)
    indices = np.random.choice(len(data), num_images, replace=False)
    for i, index in enumerate(indices):
        plt.subplot(1, num_images, i + 1)
        image = data[index].reshape(28, 28) # MNIST图像是28x28像素
        plt.imshow(image, cmap='gray')
        plt.title(f"标签: {labels[index]}")
        plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 调整布局防止标题重叠
    plt.show()

print("\n--- 随机样本可视化 ---")
plot_sample_images(X_train, y_train, "训练集样本示例")
plot_sample_images(X_test, y_test, "测试集样本示例")

# --- 检查类别分布 ---
def plot_class_distribution(labels, title):
    """绘制类别标签的分布直方图"""
    plt.figure(figsize=(8, 4))
    sns.countplot(x=labels)
    plt.title(title, fontsize=14)
    plt.xlabel("数字类别")
    plt.ylabel("样本数量")
    plt.show()

print("\n--- 类别分布检查 ---")
plot_class_distribution(y_train, "训练集类别分布")
# plot_class_distribution(y_test, "测试集类别分布") # 测试集分布类似，可选绘制

print("\n--- 数据预处理：特征缩放 ---")
print("使用 StandardScaler 对数据进行标准化...")

# 初始化缩放器
scaler = StandardScaler()

# 在训练集上拟合（计算均值和标准差）并转换
# 注意：需要将数据类型转换为 float64 以避免精度问题
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

# 在测试集上应用相同的转换（使用训练集的均值和标准差）
X_test_scaled = scaler.transform(X_test.astype(np.float64))

print("特征缩放完成。")
print(f"缩放后训练集均值 (近似值): {np.mean(X_train_scaled):.2f}")
print(f"缩放后训练集标准差 (近似值): {np.std(X_train_scaled):.2f}")

# ## 4. SVM 模型训练与评估
# 现在，我们将使用不同的核函数（`linear`, `poly`, `rbf`, `sigmoid`）和多分类策略（`ovo`, `ovr`）来训练SVM模型，并评估它们的性能。
#
# - **核函数 (Kernel):**
#   - `linear`: 线性核，适用于线性可分数据，计算速度快。$K(x, z) = x^T z$
#   - `poly`: 多项式核，可以处理非线性问题。$K(x, z) = (\gamma x^T z + r)^d$ (d是degree, r是coef0)
#   - `rbf` (Radial Basis Function) / 高斯核: 强大的非线性核，通常效果最好，但计算量较大。$K(x, z) = \exp(-\gamma ||x - z||^2)$
#   - `sigmoid`: Sigmoid核，源于神经网络。$K(x, z) = \tanh(\gamma x^T z + r)$
# - **多分类策略 (decision_function_shape):**
#   - `ovo` (One-vs-One): 为每对类别训练一个二分类器 (共 N*(N-1)/2 个)。预测时进行投票。这是 `SVC` 的默认内部实现方式。
#   - `ovr` (One-vs-Rest): 为每个类别训练一个二分类器，将其与其余所有类别区分开 (共 N 个)。预测时选择置信度最高的分类器。设置 `decision_function_shape='ovr'` 会使 `SVC` (虽然内部仍用OVO训练) 提供一个形状为 `(n_samples, n_classes)` 的决策函数输出，模拟OVR的行为。真正的OVR需要使用 `OneVsRestClassifier(SVC(...))`。本次实验直接比较 `SVC` 的 `decision_function_shape` 参数效果。
# - **其他重要参数:**
#   - `C`: 正则化参数。C值越小，正则化越强，容忍更多误分类点，间隔越大；C值越大，正则化越弱，试图将所有训练点正确分类，间隔可能变小，容易过拟合。默认值为1.0。
#   - `gamma`: 核系数，影响 `rbf`, `poly`, `sigmoid` 核。`'scale'` (默认) 使用 `1 / (n_features * X.var())` 作为gamma值。`'auto'` 使用 `1 / n_features`。gamma定义了单个训练样本的影响范围，值越小影响范围越大，值越大影响范围越小。
#   - `degree`: 多项式核 (`poly`) 的次数，默认为3。
#   - `random_state`: 用于复现结果的随机种子。

# %%
# --- 定义实验参数 ---
kernels_to_test = ['linear', 'poly', 'rbf', 'sigmoid']
strategies_to_test = ['ovo', 'ovr'] # decision_function_shape 参数
results_list = [] # 用于存储每个实验的结果

# --- 循环运行实验 ---
print("\n--- 开始进行SVM模型训练与评估 ---")
for kernel in kernels_to_test:
    for strategy in strategies_to_test:
        experiment_name = f"Kernel={kernel}, Strategy={strategy}"
        print(f"\n---\n正在进行实验: {experiment_name}")

        # --- 创建 SVM 分类器实例 ---
        # 我们使用默认的 C=1.0 和 gamma='scale'。
        # 对于 'poly' 核，默认 degree=3。
        # 可以通过网格搜索 (GridSearchCV) 寻找最佳超参数，但会显著增加时间。
        print(f"  创建SVC模型 (kernel='{kernel}', decision_function_shape='{strategy}', C=1.0, gamma='scale', random_state=42)")
        svm_model = SVC(kernel=kernel,
                        decision_function_shape=strategy,
                        C=1.0,          # 默认正则化参数
                        gamma='scale',    # 默认核系数 ('poly', 'rbf', 'sigmoid')
                        degree=3,         # 默认多项式次数 ('poly')
                        random_state=42,  # 保证结果可复现
                        probability=False # 设置为True可获取概率估计，但会增加计算时间
                       )

        # --- 训练模型并计时 ---
        print("  开始训练模型...")
        start_train_time = time.time()
        try:
            svm_model.fit(X_train_scaled, y_train)
            end_train_time = time.time()
            train_time = end_train_time - start_train_time
            print(f"  训练完成。耗时: {train_time:.2f} 秒")
        except Exception as e:
            print(f"  训练过程中发生错误: {e}")
            train_time = -1 # 标记错误
            results_list.append({
                'Experiment': experiment_name,
                'Kernel': kernel,
                'Strategy': strategy,
                'Train Time (s)': train_time,
                'Test Time (s)': -1,
                'Accuracy': 0.0,
                'Classification Report': 'Training Failed',
                'Confusion Matrix': None
            })
            continue # 跳过当前实验的后续步骤

        # --- 测试模型并计时 ---
        print("  开始在测试集上评估...")
        start_test_time = time.time()
        try:
            y_pred = svm_model.predict(X_test_scaled)
            end_test_time = time.time()
            test_time = end_test_time - start_test_time
            print(f"  预测完成。耗时: {test_time:.2f} 秒")
        except Exception as e:
            print(f"  预测过程中发生错误: {e}")
            test_time = -1 # 标记错误
            results_list.append({
                'Experiment': experiment_name,
                'Kernel': kernel,
                'Strategy': strategy,
                'Train Time (s)': train_time,
                'Test Time (s)': test_time,
                'Accuracy': 0.0,
                'Classification Report': 'Prediction Failed',
                'Confusion Matrix': None
            })
            continue # 跳过当前实验的后续步骤


        # --- 计算评估指标 ---
        # 1. 准确率 (Accuracy)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  准确率 (Accuracy): {accuracy:.4f}")

        # 2. 分类报告 (Classification Report) - 包括精确率、召回率、F1分数
        print("  生成分类报告...")
        class_report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
        print("  分类报告:\n", class_report)

        # 3. 混淆矩阵 (Confusion Matrix)
        print("  生成混淆矩阵...")
        conf_matrix = confusion_matrix(y_test, y_pred)
        # print("  混淆矩阵:\n", conf_matrix) # 打印原始矩阵可选

        # --- 存储结果 ---
        results_list.append({
            'Experiment': experiment_name,
            'Kernel': kernel,
            'Strategy': strategy,
            'Train Time (s)': train_time,
            'Test Time (s)': test_time,
            'Accuracy': accuracy,
            'Classification Report': class_report,
            'Confusion Matrix': conf_matrix
        })

print("\n--- 所有实验完成 ---")

# %% [markdown]
# ## 5. 结果汇总与可视化
# 将所有实验的结果整理成表格，并绘制图表进行比较。

# %%
# --- 将结果列表转换为 DataFrame ---
results_df = pd.DataFrame(results_list)

print("\n--- 实验结果汇总 (表格) ---")
# 显示主要性能指标，报告和矩阵后续单独处理
display_cols = ['Experiment', 'Kernel', 'Strategy', 'Train Time (s)', 'Test Time (s)', 'Accuracy']
print(results_df[display_cols].round(4)) # 保留4位小数

# --- 找到最佳模型 (基于准确率) ---
if not results_df.empty:
    best_model_row = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\n--- 最佳模型 (基于最高准确率) ---")
    print(f"实验名称: {best_model_row['Experiment']}")
    print(f"准确率: {best_model_row['Accuracy']:.4f}")
    print(f"训练时间: {best_model_row['Train Time (s)']:.2f} 秒")
    print(f"测试时间: {best_model_row['Test Time (s)']:.2f} 秒")
    print("\n最佳模型 - 分类报告:")
    print(best_model_row['Classification Report'])
else:
    print("\n没有成功的实验结果可供分析。")

# --- 性能指标可视化 ---
if not results_df.empty:
    print("\n--- 性能指标可视化 ---")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) # 一行三列图表
    fig.suptitle('不同SVM模型性能比较 (MNIST)', fontsize=18, y=1.02)

    # 图1: 准确率比较
    sns.barplot(x='Kernel', y='Accuracy', hue='Strategy', data=results_df, ax=axes[0], palette='viridis')
    axes[0].set_title('准确率比较', fontsize=14)
    axes[0].set_ylabel('准确率', fontsize=12)
    axes[0].set_xlabel('核函数', fontsize=12)
    axes[0].set_ylim(bottom=max(0, results_df['Accuracy'].min() - 0.05), top=1.0) # 调整y轴范围
    axes[0].legend(title='策略')

    # 图2: 训练时间比较
    sns.barplot(x='Kernel', y='Train Time (s)', hue='Strategy', data=results_df, ax=axes[1], palette='viridis')
    axes[1].set_title('训练时间比较', fontsize=14)
    axes[1].set_ylabel('训练时间 (秒)', fontsize=12)
    axes[1].set_xlabel('核函数', fontsize=12)
    axes[1].legend(title='策略')

    # 图3: 测试时间比较
    sns.barplot(x='Kernel', y='Test Time (s)', hue='Strategy', data=results_df, ax=axes[2], palette='viridis')
    axes[2].set_title('测试时间比较', fontsize=14)
    axes[2].set_ylabel('测试时间 (秒)', fontsize=12)
    axes[2].set_xlabel('核函数', fontsize=12)
    axes[2].legend(title='策略')

    plt.tight_layout()
    plt.show()

# --- 可视化最佳模型的混淆矩阵 ---
if not results_df.empty and best_model_row['Confusion Matrix'] is not None:
    print("\n--- 最佳模型混淆矩阵可视化 ---")
    plt.figure(figsize=(10, 8))
    sns.heatmap(best_model_row['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(i) for i in range(10)],
                yticklabels=[str(i) for i in range(10)])
    plt.title(f'混淆矩阵 - {best_model_row["Experiment"]}', fontsize=16)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.show()

    # --- 分析混淆矩阵中的常见错误 ---
    # (可选) 找出非对角线上值最大的几个元素，分析哪些数字容易混淆
    cm = best_model_row['Confusion Matrix']
    np.fill_diagonal(cm, 0) # 将对角线元素置零，只看错误分类
    most_confused = np.unravel_index(np.argsort(cm, axis=None)[-5:], cm.shape) # 找最大的5个错误
    print("\n混淆矩阵中最常见的5个错误 (真实标签 -> 预测标签: 数量):")
    for r, c in zip(most_confused[0], most_confused[1]):
        count = best_model_row['Confusion Matrix'][r, c] # 获取原始数量
        if count > 0: # 避免显示因置零产生的0值错误
             print(f"  {r} -> {c}: {count} 次")

else:
    print("\n无法显示最佳模型的混淆矩阵（可能由于实验失败或未找到最佳模型）。")


# %%
print("\n--- 实验代码执行完毕 ---")