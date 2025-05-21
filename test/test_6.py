import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import datetime as dt
import warnings # 导入warnings模块来控制警告信息

# 忽略 KMeans n_init 的未来警告
warnings.filterwarnings("ignore", message="The default value of `n_init` will change from 10 to 'auto' in 1.4.")
# 忽略 seaborn palette 的未来警告
warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue` is deprecated")


# --- 设置字体以支持中文显示 ---
# 尝试使用多种中文字体，提高兼容性
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Zen Hei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像时负号'-'显示为方块的问题

# 设置matplotlib和seaborn的显示风格，使图表更美观
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')

# --- 辅助函数：用于进行聚类、分析和可视化的通用流程 ---
def perform_clustering_analysis(data, feature_cols, k, title_prefix=""):
    """
    执行标准化、K-Means聚类、结果分析和部分可视化。

    Args:
        data (pd.DataFrame): 包含原始数据的DataFrame。
        feature_cols (list): 用于聚类的特征列名列表。
        k (int): 聚类数量。
        title_prefix (str): 图表和输出信息的前缀标题。

    Returns:
        tuple: (kmeans_model, clustered_df, silhouette_score)
               返回训练好的KMeans模型，带有Cluster标签的DataFrame，轮廓系数。
    """
    print(f"\n--- {title_prefix} ---")
    print(f"使用的聚类特征: {feature_cols}")
    print(f"用于聚类的数据量: {data.shape[0]} 行")

    # 提取特征列并进行拷贝，避免修改原始传入的data DataFrame
    features_df = data[feature_cols].copy()

    # 检查用于聚类的特征列是否有NaN（尽管清洗后应无，但再次确认，并直接在features_df上处理）
    if features_df.isnull().sum().sum() > 0:
         print(f"警告: 用于聚类的特征中仍存在NaN值，已自动删除对应行。")
         initial_feature_rows = features_df.shape[0]
         features_df.dropna(inplace=True)
         print(f"删除含NaN行后，剩余 {features_df.shape[0]} 行 (删除 {initial_feature_rows - features_df.shape[0]} 行)")


    # 检查处理后的特征DataFrame是否为空
    if features_df.shape[0] == 0:
        print(f"错误: 处理后的特征数据集为空，无法进行聚类。请检查数据清洗和特征提取步骤。")
        return None, data.copy(), -1 # 返回None模型和原始数据，轮廓系数为-1


    # 特征标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    # 使用原始features_df的索引创建标准化后的DataFrame
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_cols, index=features_df.index)
    print("特征标准化完成.")

    # 应用K-Means算法
    print(f"正在使用 K-Means 进行聚类 (k={k})...")
    # 使用 StandardScaler 转换后的数据进行训练
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(features_scaled_df)

    # 将聚类标签添加到DataFrame中
    # 使用原始数据，但只保留features_df中存在的行（即NaN处理后的行）
    clustered_df = data.loc[features_df.index].copy()
    clustered_df['Cluster'] = kmeans.labels_

    # 分析各聚类群体特征
    cluster_means = clustered_df.groupby('Cluster')[feature_cols].mean()
    print(f"\n{title_prefix} 各聚类群体特征均值:")
    print(cluster_means)

    # 性能度量 (轮廓系数)
    silhouette_avg = -1 # 默认值，如果数据量过小或k=1则无法计算
    if k > 1 and features_scaled_df.shape[0] > k: # 轮廓系数要求样本数 > 簇数
        try:
            silhouette_avg = silhouette_score(features_scaled_df, kmeans.labels_)
            print(f"\n{title_prefix} 轮廓系数: {silhouette_avg:.4f}")
        except ValueError as e:
            print(f"\n{title_prefix} 计算轮廓系数出错: {e}")
            print("可能是数据量太小或簇分配问题。")
            silhouette_avg = -1
    elif k==1:
         print(f"\n{title_prefix} k=1时轮廓系数无意义。")
    else:
         print(f"\n{title_prefix} 数据量不足或k值问题，无法计算轮廓系数。 数据量: {features_scaled_df.shape[0]}, k: {k}")


    # 可视化
    print(f"绘制 {title_prefix} 聚类结果可视化图表...")
    plt.figure(figsize=(len(feature_cols) * 4, 5)) # 根据特征数量调整图宽度
    for i, col in enumerate(feature_cols):
        plt.subplot(1, len(feature_cols), i + 1)
        # 使用 hue 参数将颜色与 Cluster 关联，以避免 FutureWarning
        sns.boxplot(x='Cluster', y=col, data=clustered_df, palette='viridis', hue='Cluster', legend=False)
        plt.title(f'{title_prefix}\n{col} by Cluster')
        plt.xlabel('客户群体 (簇)')
        plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    # 使用 hue 参数将颜色与 Cluster 关联，以避免 FutureWarning
    sns.countplot(x='Cluster', data=clustered_df, palette='viridis', hue='Cluster', legend=False)
    plt.title(f'{title_prefix} 各聚类群体的客户数量')
    plt.xlabel('客户群体 (簇)')
    plt.ylabel('客户数量')
    plt.show()

    return kmeans, clustered_df, silhouette_avg

# --- 加载原始数据 ---
print("--- 数据加载 ---")
file_path = 'air_data.csv'
try:
    # 尝试使用utf-8编码读取，如果失败再尝试gbk或gb2312
    df_raw = pd.read_csv(file_path, encoding='utf-8')
except Exception as e:
    print(f"UTF-8编码读取失败: {e}, 尝试使用gbk...")
    try:
        df_raw = pd.read_csv(file_path, encoding='gbk')
    except Exception as e:
        print(f"GBK编码读取失败: {e}, 尝试使用gb2312...")
        try:
            df_raw = pd.read_csv(file_path, encoding='gb2312')
        except Exception as e:
            print(f"GB2312编码读取失败: {e}, 请检查文件路径和编码.")
            exit() # 如果都失败，退出程序

print("数据加载成功，前5行数据:")
print(df_raw.head())
print("\n数据基本信息:")
df_raw.info()
print(f"\n原始数据共 {df_raw.shape[0]} 行.")

# --- 原始实验：基于基本清洗和总RFM特征 ---
print("\n\n--- 原始实验: 基本清洗 + 总RFM特征 ---")

# 1. 原始清洗步骤 (基于删除)
df_basic_cleaned = df_raw.copy()
initial_rows_basic = df_basic_cleaned.shape[0]

# 处理LAST_TO_END中的异常标记，如'########'，并转换为数值
# errors='coerce' 会将无法转换的值设为 NaN
df_basic_cleaned['LAST_TO_END'] = pd.to_numeric(df_basic_cleaned['LAST_TO_END'], errors='coerce')

# 处理缺失值：删除关键RFM特征存在缺失值的行（包括上面转换后产生的NaN）
basic_subset_cols = ['LAST_TO_END', 'FLIGHT_COUNT', 'Points_Sum']
df_basic_cleaned.dropna(subset=basic_subset_cols, inplace=True)

# 处理异常值：删除负值 (只对需要非负的特征进行)
df_basic_cleaned = df_basic_cleaned[(df_basic_cleaned['FLIGHT_COUNT'] >= 0) & (df_basic_cleaned['Points_Sum'] >= 0)]

print(f"原始实验清洗后，剩余 {df_basic_cleaned.shape[0]} 行 (删除 {initial_rows_basic - df_basic_cleaned.shape[0]} 行)")

# 2. 特征构建 (总RFM: LAST_TO_END, FLIGHT_COUNT, Points_Sum)
rfm_cols_total = ['LAST_TO_END', 'FLIGHT_COUNT', 'Points_Sum']

# 3. 选择最佳聚类数量 k (在原始实验数据上进行k选择)
# 使用Elbow方法和Silhouette Score
inertia_total = []
silhouette_scores_total = []
k_range = range(2, 11) # 尝试2到10个簇

print("\n正在计算原始实验数据不同k值下的Inertia和Silhouette Score...")
# 确保用于k选择的数据集是非空的
if df_basic_cleaned.shape[0] > 1: # 至少需要两个样本才能计算silhouette score for k=2
    rfm_total_for_k_selection = df_basic_cleaned[rfm_cols_total].copy()
    scaler_k = StandardScaler()
    rfm_total_scaled_k = scaler_k.fit_transform(rfm_total_for_k_selection)

    for k in k_range:
        if rfm_total_scaled_k.shape[0] < k: # 如果数据量小于k，跳过
            print(f"数据量 ({rfm_total_scaled_k.shape[0]}) 小于 k ({k})，跳过。")
            inertia_total.append(np.nan)
            silhouette_scores_total.append(np.nan)
            continue

        kmeans_k = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans_k.fit(rfm_total_scaled_k)
        inertia_total.append(kmeans_k.inertia_)
        if k > 1:
            try:
                score_k = silhouette_score(rfm_total_scaled_k, kmeans_k.labels_)
                silhouette_scores_total.append(score_k)
            except ValueError: # 避免少数情况下的计算错误
                silhouette_scores_total.append(np.nan)
        else: # k=1
            silhouette_scores_total.append(np.nan)

    # 可视化Elbow方法和Silhouette Score结果 (原始实验)
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia_total, marker='o')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('簇内平方和 (Inertia)')
    plt.title('原始实验: 使用Elbow方法选择最佳k值')
    plt.xticks(k_range)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # 注意：轮廓系数从 k=2 开始计算
    plt.plot(k_range[1:], silhouette_scores_total[1:], marker='o')
    plt.xlabel('聚类数量 (k)')
    plt.ylabel('轮廓系数 (Silhouette Score)')
    plt.title('原始实验: 使用轮廓系数选择最佳k值')
    plt.xticks(k_range[1:])
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 根据图示选择最佳k值 (在此处手动设置，例如 k=4)
    optimal_k_original = 4
    print(f"\n根据原始实验数据分析，选择最佳聚类数量 k = {optimal_k_original}")
else:
    print("原始实验清洗后数据量不足，无法进行K值选择和聚类。请检查清洗规则。")
    optimal_k_original = None # 标记无法进行后续实验

# 4. 执行原始实验的聚类和分析 (仅在数据量足够时进行)
kmeans_original, df_clustered_original, silhouette_original = None, pd.DataFrame(), -1 # 初始化结果变量

if optimal_k_original is not None and df_basic_cleaned.shape[0] > 0:
    kmeans_original, df_clustered_original, silhouette_original = perform_clustering_analysis(
        df_basic_cleaned, rfm_cols_total, optimal_k_original, title_prefix="原始实验"
    )
else:
    print("\n跳过原始实验聚类，因为数据量不足或K值未确定。")


# --- 消融实验一: 使用最近一年RFM特征 ---
print("\n\n--- 消融实验一: 使用最近一年RFM特征 ---")
print("对比原始实验，这里更换了聚类特征，清洗方法保持与原始实验一致。")

# 使用与原始实验相同的清洗后的数据作为起点
# 确保原始实验成功且有数据
if df_basic_cleaned is not None and df_basic_cleaned.shape[0] > 0:
    df_l1y_experiment = df_basic_cleaned.copy()

    # 1. 特征构建 (最近一年RFM: LAST_TO_END, L1Y_Flight_Count, L1Y_Points_Sum)
    rfm_cols_l1y = ['LAST_TO_END', 'L1Y_Flight_Count', 'L1Y_Points_Sum']

    # 检查新特征是否存在并进行安全处理（理论上原始清洗已处理LAST_TO_END NaN）
    # 检查L1Y_Flight_Count和L1Y_Points_Sum中的缺失值或负值 (虽然info显示它们非空，但仍检查)
    l1y_check_cols = ['L1Y_Flight_Count', 'L1Y_Points_Sum']
    initial_rows_l1y_exp = df_l1y_experiment.shape[0]

    # 检查缺失值
    if df_l1y_experiment[rfm_cols_l1y].isnull().sum().sum() > 0:
         print("\n消融实验一: 检查到新特征有缺失值，已删除对应行。")
         df_l1y_experiment.dropna(subset=rfm_cols_l1y, inplace=True)

    # 检查负值 (仅对可能非负的特征)
    df_l1y_experiment = df_l1y_experiment[(df_l1y_experiment[l1y_check_cols] >= 0).all(axis=1)]

    print(f"消融实验一清洗后，剩余 {df_l1y_experiment.shape[0]} 行 (删除 {initial_rows_l1y_exp - df_l1y_experiment.shape[0]} 行)")


    # 2. 执行聚类和分析 (使用原始实验确定的最佳k值，并确保数据量足够)
    kmeans_l1y, df_clustered_l1y, silhouette_l1y = None, pd.DataFrame(), -1 # 初始化
    if optimal_k_original is not None and df_l1y_experiment.shape[0] > 0:
        kmeans_l1y, df_clustered_l1y, silhouette_l1y = perform_clustering_analysis(
            df_l1y_experiment, rfm_cols_l1y, optimal_k_original, title_prefix="消融实验一 (L1Y Features)"
        )
    else:
        print("\n跳过消融实验一聚类，因为数据量不足或K值未确定。")

else:
    print("\n跳过消融实验一，因为原始实验数据量不足。")


# --- 消融实验二: 使用IQR方法处理异常值 (修改为Capping) ---
print("\n\n--- 消融实验二: 使用IQR方法处理异常值 (修改为Capping) ---")
print("对比原始实验，这里更改了数据清洗方法（IQR异常值封顶），聚类特征保持与原始实验一致。")

# 从原始未清洗的数据开始，或至少是LAST_TO_END NaN处理后的数据
df_iqr_experiment = df_raw.copy()
initial_rows_iqr = df_iqr_experiment.shape[0]

# 处理LAST_TO_END中的异常标记（与原始实验相同）
df_iqr_experiment['LAST_TO_END'] = pd.to_numeric(df_iqr_experiment['LAST_TO_END'], errors='coerce')
df_iqr_experiment.dropna(subset=['LAST_TO_END'], inplace=True) # 删除转换后为NaN的行

print(f"消融实验二 (IQR清洗) 前，处理LAST_TO_END NaN后，剩余 {df_iqr_experiment.shape[0]} 行")

# 1. IQR清洗步骤 (修改为Capping)
print("正在进行IQR异常值封顶处理...")
# 对RFM特征（LAST_TO_END, FLIGHT_COUNT, Points_Sum）进行IQR异常值检测和封顶
rfm_cols_for_iqr_capping = ['LAST_TO_END', 'FLIGHT_COUNT', 'Points_Sum']
df_iqr_cleaned = df_iqr_experiment.copy() # 从处理了LAST_TO_END NaN的数据开始

# 计算IQR并进行封顶
for col in rfm_cols_for_iqr_capping:
    if col in df_iqr_cleaned.columns: # 确保列存在
        # 检查列是否有足够的非NaN数据来计算分位数
        if df_iqr_cleaned[col].dropna().shape[0] < 2:
             print(f"警告: 列 '{col}' 数据量不足 ({df_iqr_cleaned[col].dropna().shape[0]} 个非NaN值)，无法计算IQR和封顶，跳过。")
             continue

        Q1 = df_iqr_cleaned[col].quantile(0.25)
        Q3 = df_iqr_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 封顶操作：将小于下界的设为下界，将大于上界的设为上界
        # 注意：这里将下界设为0，因为FLIGHT_COUNT和Points_Sum不能为负
        if col in ['FLIGHT_COUNT', 'Points_Sum']:
             lower_bound = max(0, lower_bound) # 确保飞行次数和积分不为负

        # 使用clip方法进行封顶
        df_iqr_cleaned[col] = df_iqr_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"已对列 '{col}' 应用 IQR ({lower_bound:.2f}, {upper_bound:.2f}) 封顶。")
    else:
        print(f"警告: 用于IQR处理的列 '{col}' 不存在。")


print(f"消融实验二 (IQR清洗 - Capping) 后，剩余 {df_iqr_cleaned.shape[0]} 行 (此方法不删除行，只修改异常值)")
# 对比原始清洗删除的行数 (initial_rows_basic - df_basic_cleaned.shape[0])

# 可视化清洗后的数据分布（与原始清洗后的数据分布对比）
# 确保原始实验的数据集存在且非空，以便进行对比
if 'df_basic_cleaned' in locals() and df_basic_cleaned.shape[0] > 0:
    print("\n绘制清洗后RFM特征的箱线图，对比清洗效果...")
    plt.figure(figsize=(15, 8)) # 增加图高度，容纳更多信息
    plt.suptitle("清洗方法对比：基本清洗 (删除) vs IQR清洗 (封顶) (RFM特征)", y=1.02, fontsize=14)

    # --- 基本清洗后的箱线图 ---
    plt.subplot(2, 3, 1)
    sns.boxplot(y='LAST_TO_END', data=df_basic_cleaned, color='skyblue')
    plt.title('基本清洗 (删除)\nLAST_TO_END')
    plt.ylabel('') # 避免重复ylabel

    plt.subplot(2, 3, 2)
    sns.boxplot(y='FLIGHT_COUNT', data=df_basic_cleaned, color='skyblue')
    plt.title('基本清洗 (删除)\nFLIGHT_COUNT')
    plt.ylabel('')

    plt.subplot(2, 3, 3)
    sns.boxplot(y='Points_Sum', data=df_basic_cleaned, color='skyblue')
    plt.title('基本清洗 (删除)\nPoints_Sum')
    plt.ylabel('')


    # --- IQR清洗 (Capping) 后的箱线图 ---
    plt.subplot(2, 3, 4)
    sns.boxplot(y='LAST_TO_END', data=df_iqr_cleaned, color='lightgreen')
    plt.title('IQR清洗 (封顶)\nLAST_TO_END')
    plt.ylabel('值') # 加上一个通用ylabel

    plt.subplot(2, 3, 5)
    sns.boxplot(y='FLIGHT_COUNT', data=df_iqr_cleaned, color='lightgreen')
    plt.title('IQR清洗 (封顶)\nFLIGHT_COUNT')
    plt.ylabel('')

    plt.subplot(2, 3, 6)
    sns.boxplot(y='Points_Sum', data=df_iqr_cleaned, color='lightgreen')
    plt.title('IQR清洗 (封顶)\nPoints_Sum')
    plt.ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局避免标题重叠
    plt.show()
else:
     print("\n跳过清洗效果对比箱线图绘制，因为原始实验数据量不足。")


# 2. 执行聚类和分析 (使用原始实验确定的最佳k值，并确保数据量足够)
kmeans_iqr, df_clustered_iqr, silhouette_iqr = None, pd.DataFrame(), -1 # 初始化
if optimal_k_original is not None and df_iqr_cleaned.shape[0] > 0:
    kmeans_iqr, df_clustered_iqr, silhouette_iqr = perform_clustering_analysis(
        df_iqr_cleaned, rfm_cols_total, optimal_k_original, title_prefix="消融实验二 (IQR清洗 - Capping)"
    )
else:
    print("\n跳过消融实验二聚类，因为数据量不足或K值未确定。")


# --- 实验结果对比总结 ---
print("\n\n--- 实验结果对比总结 ---")

print(f"原始实验 (基本清洗 + 总RFM):")
if df_clustered_original.shape[0] > 0:
    print(f"  使用数据量: {df_clustered_original.shape[0]} 行")
    print(f"  轮廓系数: {silhouette_original:.4f}")
    print("  各聚类群体RFM特征均值:")
    print(df_clustered_original.groupby('Cluster')[rfm_cols_total].mean())
else:
    print("  未成功执行聚类，数据量不足。")


print(f"\n消融实验一 (基本清洗 + L1Y RFM):")
if df_clustered_l1y.shape[0] > 0:
    print(f"  使用数据量: {df_clustered_l1y.shape[0]} 行")
    print(f"  轮廓系数: {silhouette_l1y:.4f}")
    print("  各聚类群体L1Y RFM特征均值:")
    # 确保 rfm_cols_l1y 在 grouped df 中存在
    try:
        print(df_clustered_l1y.groupby('Cluster')[rfm_cols_l1y].mean())
    except KeyError:
        print("  L1Y特征列不存在于聚类结果中，请检查。")
else:
    print("  未成功执行聚类，数据量不足。")


print(f"\n消融实验二 (IQR清洗 - Capping + 总RFM):")
if df_clustered_iqr.shape[0] > 0:
    print(f"  使用数据量: {df_clustered_iqr.shape[0]} 行")
    print(f"  轮廓系数: {silhouette_iqr:.4f}")
    print("  各聚类群体RFM特征均值:")
    try:
        print(df_clustered_iqr.groupby('Cluster')[rfm_cols_total].mean())
    except KeyError:
        print("  总RFM特征列不存在于聚类结果中，请检查。")
else:
    print("  未成功执行聚类，数据量不足。")

print("\n--- 总结与讨论 ---")
print("对比不同实验的客户数量、轮廓系数以及各聚类群体的特征均值，可以评估不同特征集和不同清洗方法对聚类结果的影响。")
print("例如：")
print("- **清洗方法对比:** 原始实验使用简单删除异常值，消融实验二使用IQR封顶。对比两者使用的最终数据量和聚类结果（轮廓系数、群体特征），可以看出异常值对聚类稳定性和群体划分的影响。IQR封顶通常保留更多数据，且可能受极端值影响更小，但也可能模糊掉一些真实的高价值客户特征（如果他们的极端值是真实的）。")
print("- **特征集对比:** 原始实验使用总RFM特征，消融实验一使用最近一年RFM特征。对比两者在相似数据量（如果清洗方法一致）下的聚类结果，可以评估哪种特征更能有效区分客户群体，例如最近一年特征可能更能识别活跃客户，而总特征更能识别累积价值客户。")
print("这些对比结果将作为实验报告中“消融实验”和“结果分析”部分的重要依据。")