#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
创建示例数据集用于测试机器学习功能
"""

import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification, make_regression
from datetime import datetime, timedelta

def create_sample_datasets():
    """创建各种类型的示例数据集"""
    
    # 确保数据目录存在
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'samples')
    os.makedirs(data_dir, exist_ok=True)
    
    print("正在创建示例数据集...")
    
    # 1. 分类数据集 - 学生成绩预测
    print("创建分类数据集: 学生成绩预测...")
    np.random.seed(42)
    n_samples = 1000
    
    # 生成特征
    study_hours = np.random.normal(6, 2, n_samples)
    attendance = np.random.uniform(0.7, 1.0, n_samples)
    previous_score = np.random.normal(75, 15, n_samples)
    homework_completion = np.random.uniform(0.6, 1.0, n_samples)
    sleep_hours = np.random.normal(7, 1.5, n_samples)
    
    # 计算最终成绩（有一定随机性）
    final_score = (
        study_hours * 5 + 
        attendance * 20 + 
        previous_score * 0.3 + 
        homework_completion * 15 + 
        sleep_hours * 2 + 
        np.random.normal(0, 5, n_samples)
    )
    
    # 转换为等级
    grade_category = pd.cut(final_score, 
                           bins=[0, 60, 70, 80, 90, 100], 
                           labels=['不及格', '及格', '良好', '优秀', '杰出'])
    
    student_data = pd.DataFrame({
        '学习时间_小时': np.clip(study_hours, 0, 12),
        '出勤率': np.clip(attendance, 0, 1),
        '之前成绩': np.clip(previous_score, 0, 100),
        '作业完成率': np.clip(homework_completion, 0, 1),
        '睡眠时间_小时': np.clip(sleep_hours, 4, 12),
        '最终成绩': np.clip(final_score, 0, 100),
        '成绩等级': grade_category
    })
    
    student_data.to_csv(os.path.join(data_dir, '学生成绩预测.csv'), index=False, encoding='utf-8-sig')
    
    # 2. 回归数据集 - 房价预测
    print("创建回归数据集: 房价预测...")
    np.random.seed(42)
    n_houses = 800
    
    area = np.random.normal(120, 40, n_houses)
    bedrooms = np.random.randint(1, 6, n_houses)
    bathrooms = np.random.randint(1, 4, n_houses)
    age = np.random.randint(0, 50, n_houses)
    distance_to_center = np.random.uniform(1, 30, n_houses)
    
    # 房价计算（有噪音）
    price = (
        area * 8000 + 
        bedrooms * 50000 + 
        bathrooms * 30000 - 
        age * 2000 - 
        distance_to_center * 5000 + 
        np.random.normal(0, 100000, n_houses)
    )
    
    house_data = pd.DataFrame({
        '面积_平米': np.clip(area, 30, 300),
        '卧室数量': bedrooms,
        '浴室数量': bathrooms,
        '房龄_年': age,
        '距市中心距离_公里': distance_to_center,
        '房价_万元': np.clip(price, 100000, 2000000) / 10000
    })
    
    house_data.to_csv(os.path.join(data_dir, '房价预测.csv'), index=False, encoding='utf-8-sig')
    
    # 3. 客户分群数据集
    print("创建聚类数据集: 客户分群...")
    np.random.seed(42)
    n_customers = 500
    
    # 创建3个客户群体
    # 群体1: 高价值客户
    age1 = np.random.normal(45, 8, n_customers//3)
    income1 = np.random.normal(150000, 30000, n_customers//3)
    spending1 = np.random.normal(80000, 15000, n_customers//3)
    
    # 群体2: 中等价值客户
    age2 = np.random.normal(35, 10, n_customers//3)
    income2 = np.random.normal(80000, 20000, n_customers//3)
    spending2 = np.random.normal(40000, 10000, n_customers//3)
    
    # 群体3: 低价值客户
    age3 = np.random.normal(28, 12, n_customers - 2*(n_customers//3))
    income3 = np.random.normal(45000, 15000, n_customers - 2*(n_customers//3))
    spending3 = np.random.normal(20000, 8000, n_customers - 2*(n_customers//3))
    
    customer_data = pd.DataFrame({
        '年龄': np.concatenate([age1, age2, age3]),
        '年收入_万元': np.concatenate([income1, income2, income3]) / 10000,
        '年消费_万元': np.concatenate([spending1, spending2, spending3]) / 10000,
        '会员年限': np.random.randint(1, 15, n_customers),
        '购买频次_次每月': np.random.randint(1, 20, n_customers)
    })
    
    customer_data = customer_data.sample(frac=1).reset_index(drop=True)  # 打乱顺序
    customer_data.to_csv(os.path.join(data_dir, '客户分群.csv'), index=False, encoding='utf-8-sig')
    
    # 4. 文本分类数据集 - 产品评论情感分析
    print("创建文本分类数据集: 产品评论情感分析...")
    
    positive_comments = [
        "这个产品非常好用，质量很棒！",
        "物流很快，包装很好，非常满意",
        "性价比很高，推荐购买",
        "功能强大，操作简单，值得推荐",
        "质量超出预期，服务态度很好",
        "外观设计很漂亮，用起来很顺手",
        "价格实惠，质量不错",
        "快递很快，商品和描述一致",
        "客服态度很好，解答很及时",
        "这是我买过最好的产品了"
    ] * 30
    
    negative_comments = [
        "质量太差了，用了几天就坏了",
        "包装破损，商品有瑕疵",
        "与描述不符，很失望",
        "物流太慢了，等了很久",
        "客服态度很差，不推荐",
        "价格太贵，不值得",
        "功能有问题，操作复杂",
        "外观很丑，做工粗糙",
        "退货很麻烦，服务不好",
        "完全不符合预期，浪费钱"
    ] * 30
    
    neutral_comments = [
        "还可以吧，一般般",
        "价格合理，质量还行",
        "普普通通，没什么特别的",
        "基本功能都有，凑合用",
        "物流一般，包装还行",
        "符合预期，没有惊喜",
        "中规中矩的产品",
        "还算满意，有改进空间",
        "性价比一般，不算特别好",
        "可以接受，但不会再买"
    ] * 20
    
    comments_data = pd.DataFrame({
        '评论内容': positive_comments + negative_comments + neutral_comments,
        '情感标签': ['正面'] * 300 + ['负面'] * 300 + ['中性'] * 200,
        '评分': [5] * 150 + [4] * 150 + [1] * 150 + [2] * 150 + [3] * 200
    })
    
    comments_data = comments_data.sample(frac=1).reset_index(drop=True)
    comments_data.to_csv(os.path.join(data_dir, '产品评论情感分析.csv'), index=False, encoding='utf-8-sig')
    
    # 5. 时间序列数据集 - 销售预测
    print("创建时间序列数据集: 销售预测...")
    
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(365*3)]
    
    # 生成有趋势和季节性的销售数据
    trend = np.linspace(1000, 1500, len(dates))
    seasonal = 200 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 50, len(dates))
    
    sales = trend + seasonal + weekly + noise
    
    sales_data = pd.DataFrame({
        '日期': dates,
        '销售额': np.clip(sales, 0, None),
        '周几': [d.weekday() + 1 for d in dates],
        '月份': [d.month for d in dates],
        '季度': [(d.month - 1) // 3 + 1 for d in dates],
        '是否节假日': np.random.choice([0, 1], len(dates), p=[0.9, 0.1])
    })
    
    sales_data.to_csv(os.path.join(data_dir, '销售预测.csv'), index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 示例数据集创建完成！")
    print(f"数据存储位置: {data_dir}")
    print("\n创建的数据集:")
    print("1. 学生成绩预测.csv - 分类问题")
    print("2. 房价预测.csv - 回归问题") 
    print("3. 客户分群.csv - 聚类问题")
    print("4. 产品评论情感分析.csv - 文本分类")
    print("5. 销售预测.csv - 时间序列预测")
    
    return data_dir

def create_readme():
    """创建数据集说明文档"""
    readme_content = """
# 示例数据集说明

本目录包含了用于测试机器学习功能的示例数据集。

## 数据集列表

### 1. 学生成绩预测.csv (分类问题)
- **目标**: 根据学习习惯预测学生成绩等级
- **特征**: 学习时间、出勤率、之前成绩、作业完成率、睡眠时间
- **目标变量**: 成绩等级（不及格、及格、良好、优秀、杰出）
- **样本数**: 1000条

### 2. 房价预测.csv (回归问题)
- **目标**: 根据房屋特征预测房价
- **特征**: 面积、卧室数量、浴室数量、房龄、距市中心距离
- **目标变量**: 房价（万元）
- **样本数**: 800条

### 3. 客户分群.csv (聚类问题)
- **目标**: 基于客户行为进行客户分群
- **特征**: 年龄、年收入、年消费、会员年限、购买频次
- **样本数**: 500条

### 4. 产品评论情感分析.csv (文本分类)
- **目标**: 分析产品评论的情感倾向
- **特征**: 评论内容、评分
- **目标变量**: 情感标签（正面、负面、中性）
- **样本数**: 800条

### 5. 销售预测.csv (时间序列)
- **目标**: 预测未来销售额
- **特征**: 日期、周几、月份、季度、是否节假日
- **目标变量**: 销售额
- **样本数**: 1095条（3年数据）

## 使用建议

1. **初学者**: 建议从学生成绩预测或房价预测开始
2. **进阶用户**: 可以尝试客户分群或文本情感分析
3. **高级用户**: 可以探索时间序列预测

## 实验建议

### 分类实验
```
使用学生成绩预测数据，尝试不同的分类算法：
- 逻辑回归
- 决策树
- 随机森林
- 支持向量机
```

### 回归实验
```
使用房价预测数据，比较回归算法性能：
- 线性回归
- 决策树回归
- 随机森林回归
```

### 集成学习实验
```
使用投票或堆叠方法组合多个模型，观察性能提升。
```

### 聚类实验
```
使用客户分群数据进行K-means聚类，找出最佳聚类数量。
```

---
数据生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'samples')
    readme_path = os.path.join(data_dir, 'README.md')
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📖 说明文档已创建: {readme_path}")

if __name__ == "__main__":
    # 创建示例数据集
    data_dir = create_sample_datasets()
    
    # 创建说明文档
    create_readme()
    
    print(f"\n🎉 所有示例数据已准备完成！")
    print(f"您现在可以使用这些数据集来测试机器学习功能。")
    print(f"\n💡 快速开始:")
    print(f"1. 启动应用: python app.py")
    print(f"2. 上传数据: 选择 {data_dir} 中的任意 CSV 文件")
    print(f"3. 选择目标列进行实验")
    print(f"4. 开始对话式机器学习实验！") 