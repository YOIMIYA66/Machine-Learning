import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体 (或其他已安装的中文字体)
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 1. 数据读取和预处理
data = pd.read_excel("北京市空气质量数据.xlsx")
data = data.dropna()

# 2. 一元线性回归模型 (CO vs. PM2.5)
X_uni = data[['CO']]
y_uni = data['PM2.5']
X_train_uni, X_test_uni, y_train_uni, y_test_uni = train_test_split(X_uni, y_uni, test_size=0.2, random_state=42)
model_uni = LinearRegression()
model_uni.fit(X_train_uni, y_train_uni)
y_pred_uni = model_uni.predict(X_test_uni)
mse_uni = mean_squared_error(y_test_uni, y_pred_uni)
print(f"一元线性回归 (CO vs PM2.5) - 均方误差: {mse_uni:.2f}")
print(f"  截距: {model_uni.intercept_:.2f}")
print(f"  CO 系数: {model_uni.coef_[0]:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(X_test_uni, y_test_uni, color='blue', label='实际值')
plt.plot(X_test_uni, y_pred_uni, color='red', label='预测值')
plt.xlabel('CO')
plt.ylabel('PM2.5')
plt.title('一元线性回归 (CO vs PM2.5)')
plt.legend()
plt.show()

# 3. 多元线性回归模型 (CO, SO2 vs. PM2.5)
X_multi = data[['CO', 'SO2']]
y_multi = data['PM2.5']
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)
y_pred_multi = model_multi.predict(X_test_multi)
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
print(f"\n多元线性回归 (CO, SO2 vs PM2.5) - 均方误差: {mse_multi:.2f}")
print(f"  截距: {model_multi.intercept_:.2f}")
print(f"  CO 系数: {model_multi.coef_[0]:.2f}")
print(f"  SO2 系数: {model_multi.coef_[1]:.2f}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_test_multi['CO'], X_test_multi['SO2'], y_test_multi, color='blue', label='实际值')
ax.scatter(X_test_multi['CO'], X_test_multi['SO2'], y_pred_multi, color='red', marker='x', label='预测值')
ax.set_xlabel('CO')
ax.set_ylabel('SO2')
ax.set_zlabel('PM2.5')
ax.set_title('多元线性回归 (CO, SO2 vs PM2.5)')
ax.legend()
plt.show()

# 4. 逻辑回归模型 (PM2.5, PM10 vs. 污染)
data['污染'] = data['质量等级'].apply(lambda x: 0 if x in ['优', '良'] else 1)
X_log = data[['PM2.5', 'PM10']]
y_log = data['污染']
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
model_log = LogisticRegression()
model_log.fit(X_train_log, y_train_log)
y_pred_log = model_log.predict(X_test_log)

# --- 评估指标 ---
accuracy = accuracy_score(y_test_log, y_pred_log)
conf_matrix = confusion_matrix(y_test_log, y_pred_log)
class_report = classification_report(y_test_log, y_pred_log)

print(f"\n逻辑回归 (PM2.5, PM10 vs 污染) - 准确率: {accuracy:.2f}")
print("混淆矩阵:\n", conf_matrix)
print("分类报告:\n", class_report)
print(f"  截距: {model_log.intercept_[0]:.2f}")
print(f"  PM2.5 系数: {model_log.coef_[0][0]:.2f}")
print(f"  PM10 系数: {model_log.coef_[0][1]:.2f}")

# --- 可视化: 决策边界 ---
plt.figure(figsize=(8, 6))
# 使用 train 数据绘制散点图 (因为预测时使用了 feature names)
sns.scatterplot(x=X_train_log['PM2.5'], y=X_train_log['PM10'], hue=y_train_log, palette={0: 'green', 1: 'red'})

# 生成网格点
xx, yy = np.meshgrid(np.linspace(X_log['PM2.5'].min()-10, X_log['PM2.5'].max()+10, 100),
                     np.linspace(X_log['PM10'].min()-10, X_log['PM10'].max()+10, 100))
# 预测并绘制决策边界
Z = model_log.predict(np.c_[xx.ravel(), yy.ravel()])  # 无需 feature_names
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='black', levels=[0.5])

plt.xlabel('PM2.5')
plt.ylabel('PM10')
plt.title('逻辑回归 (PM2.5, PM10 vs 污染) - 决策边界')
plt.show()