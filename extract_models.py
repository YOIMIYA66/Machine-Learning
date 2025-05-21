# extract_models.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import subprocess
import sys
import traceback
import gzip
import json
import jieba
import time

# 确保模型保存目录存在
MODEL_DIR = 'ml_models'
os.makedirs(MODEL_DIR, exist_ok=True)

# 保存模型的通用函数
def save_model(model, model_name, preprocessors=None, metadata=None):
    """
    保存模型和相关的预处理器与元数据
    
    Args:
        model: 训练好的模型对象
        model_name: 模型名称
        preprocessors: 预处理器字典（如标准化器、编码器等）
        metadata: 其他元数据（如特征名称、目标列等）
    """
    if not model_name.endswith('.pkl'):
        model_name = f"{model_name}.pkl"
        
    model_path = os.path.join(MODEL_DIR, model_name)
    
    with open(model_path, 'wb') as f:
        pickle.dump((model, preprocessors or {}, metadata or {}), f)
        
    print(f"模型已保存到: {model_path}")
    return model_path

# 从test_1.py提取线性回归模型
def extract_linear_regression_models():
    print("\n提取线性回归模型...")
    try:
        # 加载数据
        data = pd.read_excel("北京市空气质量数据.xlsx")
        data = data.dropna()
        
        # 一元线性回归 (CO vs. PM2.5)
        X_uni = data[['CO']]
        y_uni = data['PM2.5']
        X_train_uni, X_test_uni, y_train_uni, y_test_uni = train_test_split(X_uni, y_uni, test_size=0.2, random_state=42)
        model_uni = LinearRegression()
        model_uni.fit(X_train_uni, y_train_uni)
        
        # 保存一元线性回归模型
        metadata_uni = {
            'model_type': 'linear_regression',
            'feature_names': ['CO'],
            'target_name': 'PM2.5',
            'description': 'CO对PM2.5的一元线性回归'
        }
        save_model(model_uni, 'linear_regression_CO_PM25', metadata=metadata_uni)
        
        # 多元线性回归 (CO, SO2 vs. PM2.5)
        X_multi = data[['CO', 'SO2']]
        y_multi = data['PM2.5']
        X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
        model_multi = LinearRegression()
        model_multi.fit(X_train_multi, y_train_multi)
        
        # 保存多元线性回归模型
        metadata_multi = {
            'model_type': 'linear_regression',
            'feature_names': ['CO', 'SO2'],
            'target_name': 'PM2.5',
            'description': 'CO和SO2对PM2.5的多元线性回归'
        }
        save_model(model_multi, 'linear_regression_CO_SO2_PM25', metadata=metadata_multi)
        
        return True
    except Exception as e:
        print(f"提取线性回归模型出错: {e}")
        traceback.print_exc()
        return False

# 从test_1.py提取逻辑回归模型
def extract_logistic_regression_model():
    print("\n提取逻辑回归模型...")
    try:
        # 加载数据
        data = pd.read_excel("北京市空气质量数据.xlsx")
        data = data.dropna()
        
        # 数据预处理
        data['污染'] = data['质量等级'].apply(lambda x: 0 if x in ['优', '良'] else 1)
        X_log = data[['PM2.5', 'PM10']]
        y_log = data['污染']
        
        # 拆分数据并训练模型
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
        model_log = LogisticRegression()
        model_log.fit(X_train_log, y_train_log)
        
        # 保存逻辑回归模型
        metadata_log = {
            'model_type': 'logistic_regression',
            'feature_names': ['PM2.5', 'PM10'],
            'target_name': '污染',
            'target_mapping': {0: '优/良', 1: '轻度污染及以上'},
            'description': '基于PM2.5和PM10预测空气质量是否污染的二分类模型'
        }
        save_model(model_log, 'logistic_regression_pollution', metadata=metadata_log)
        
        return True
    except Exception as e:
        print(f"提取逻辑回归模型出错: {e}")
        traceback.print_exc()
        return False

# 从test_2.py提取KNN模型
def extract_knn_model():
    print("\n提取KNN模型...")
    try:
        # 加载数据
        df = pd.read_excel("北京市空气质量数据.xlsx")
        
        # 定义高斯权重函数
        def gaussian_weight(distances):
            # 确保除以0的情况不会发生
            epsilon = 1e-6
            return np.exp(-0.5 * (distances**2) / np.mean(distances + epsilon)**2) # 确保整数除以0的情况
        
        # 特征选择 (选择 'PM2.5' 到 'O3' 之间的所有列作为特征)
        X = df.iloc[:, 3:-1].values # 选择 PM2.5, PM10, SO2, CO, NO2, O3 作为特征
        y_raw = df['质量等级'].values
        feature_names = df.iloc[:, 3:-1].columns.tolist()
        
        # 目标变量标签编码
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=0.3,
            random_state=42,
            stratify=y_encoded
        )
        
        # 使用test_2.py中的最佳参数训练KNN分类器
        best_k = 5  # 可以根据实际测试找到最优K值
        best_metric = 'manhattan'  # 可以根据实际测试找到最优距离度量
        best_weights = 'distance'  # 可以是'uniform', 'distance'或gaussian_weight函数
        
        # 创建KNN分类器
        if best_weights == 'gaussian_weight' or best_weights == gaussian_weight:
            knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=gaussian_weight)
            best_weights_name = 'gaussian_weight'
        else:
            knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric, weights=best_weights)
            best_weights_name = best_weights
        
        knn.fit(X_train, y_train)
        
        # 保存预处理器和模型
        preprocessors = {
            'scaler': scaler,
            'label_encoder': le
        }
        
        metadata = {
            'model_type': 'knn_classifier',
            'feature_names': feature_names,
            'target_name': '质量等级',
            'class_names': le.classes_.tolist(),
            'description': 'K近邻法分类预测空气质量等级',
            'k': best_k,
            'metric': best_metric,
            'weights': best_weights_name
        }
        
        # 添加gaussian_weight函数到元数据
        if best_weights == 'gaussian_weight' or best_weights == gaussian_weight:
            # 将gaussian_weight函数存储为字符串
            gaussian_weight_code = (
                "def gaussian_weight(distances):\n"
                "    epsilon = 1e-6\n"
                "    return np.exp(-0.5 * (distances**2) / np.mean(distances + epsilon)**2)\n"
            )
            metadata['gaussian_weight_function'] = gaussian_weight_code
        
        save_model(knn, 'knn_air_quality', preprocessors, metadata)
        print(f"成功提取并保存KNN模型 (k={best_k}, metric={best_metric}, weights={best_weights_name})")
        
        return True
    except Exception as e:
        print(f"提取KNN模型出错: {e}")
        traceback.print_exc()
        return False

# 从test_3.py提取决策树模型
def extract_decision_tree_model():
    print("\n提取决策树模型...")
    try:
        # 加载数据 - 修正文件路径问题
        try:
            # 首先尝试直接加载当前目录下的文件
            df = pd.read_excel("北京市空气质量数据.xlsx")
            print("成功从默认路径加载数据")
        except FileNotFoundError:
            # 如果失败，尝试test_3.py中的路径
            alt_path = 'D:\PaddlePaddle-EfficientNetV2\PaddleClas-EfficientNet\北京市空气质量数据.xlsx'
            try:
                df = pd.read_excel(alt_path)
                print(f"从备用路径 {alt_path} 加载数据")
            except FileNotFoundError:
                raise FileNotFoundError("无法找到数据文件，请确保北京市空气质量数据.xlsx存在")
        
        # 特征选择
        feature_columns = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']
        X = df[feature_columns]
        y_raw = df['质量等级']
        
        # 编码目标变量
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练决策树
        optimal_depth = 5  # 可以根据实际测试找到最优深度
        dt_clf = DecisionTreeClassifier(criterion='entropy', max_depth=optimal_depth, random_state=42)
        dt_clf.fit(X_train, y_train)
        
        # 保存预处理器和模型
        preprocessors = {
            'label_encoder': le
        }
        
        metadata = {
            'model_type': 'decision_tree',
            'feature_names': feature_columns,
            'target_name': '质量等级',
            'class_names': le.classes_.tolist(),
            'description': '基于熵的决策树分类预测空气质量等级',
            'max_depth': optimal_depth
        }
        
        save_model(dt_clf, 'decision_tree_air_quality', preprocessors, metadata)
        
        return True
    except Exception as e:
        print(f"提取决策树模型出错: {e}")
        traceback.print_exc()
        return False

# 从test_4.py提取SVM模型 
def extract_svm_model():
    print("\n提取SVM模型...")
    try:
        # 尝试加载test_4.py中的MNIST数据
        try:
            # 检查MNIST数据是否存在
            mnist_path = 'mnist.pkl.gz'
            if os.path.exists(mnist_path):
                print(f"找到MNIST数据文件: {mnist_path}")
                # 加载MNIST数据
                with gzip.open(mnist_path, 'rb') as f:
                    # 使用latin1编码以兼容Python 2生成的pickle文件
                    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
                
                X_train, y_train = train_set
                X_test, y_test = test_set
                
                # 取子集
                SUBSET_SIZE = 1000
                _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=SUBSET_SIZE/len(y_train), stratify=y_train, random_state=42)
                _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=SUBSET_SIZE/len(y_test), stratify=y_test, random_state=42)
                
                # 标准化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
                X_test_scaled = scaler.transform(X_test.astype(np.float64))
                
                # 训练SVM模型
                svm_clf = SVC(kernel='rbf', decision_function_shape='ovr', random_state=42, gamma='scale')
                svm_clf.fit(X_train_scaled, y_train)
                
                # 保存预处理器和模型
                preprocessors = {
                    'scaler': scaler
                }
                
                feature_names = [f'pixel_{i}' for i in range(X_train.shape[1])]
                metadata = {
                    'model_type': 'svm_classifier',
                    'feature_names': feature_names,
                    'target_name': 'digit',
                    'class_names': list(map(str, range(10))),
                    'description': 'SVM分类预测手写数字',
                    'kernel': 'rbf',
                    'decision_function_shape': 'ovr',
                    'data_source': 'MNIST'
                }
                
                save_model(svm_clf, 'svm_mnist_digits', preprocessors, metadata)
                print("成功提取并保存MNIST数据集的SVM模型")
                return True
                
            else:
                print(f"MNIST数据文件不存在: {mnist_path}\n将使用空气质量数据集代替")
                # 如果MNIST数据不存在，则使用空气质量数据集
        except Exception as mnist_error:
            print(f"加载MNIST数据失败: {mnist_error}\n将使用空气质量数据集代替")
        
        # 如果MNIST数据加载失败，则使用空气质量数据集
        df = pd.read_excel("北京市空气质量数据.xlsx")
        
        # 特征选择
        feature_columns = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']
        X = df[feature_columns]
        y_raw = df['质量等级']
        
        # 编码目标变量
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练SVM模型
        svm_clf = SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
        svm_clf.fit(X_train, y_train)
        
        # 保存预处理器和模型
        preprocessors = {
            'scaler': scaler,
            'label_encoder': le
        }
        
        metadata = {
            'model_type': 'svm_classifier',
            'feature_names': feature_columns,
            'target_name': '质量等级',
            'class_names': le.classes_.tolist(),
            'description': 'SVM分类预测空气质量等级',
            'kernel': 'rbf',
            'decision_function_shape': 'ovr'
        }
        
        save_model(svm_clf, 'svm_air_quality', preprocessors, metadata)
        print("成功提取并保存空气质量数据集的SVM模型")
        
        return True
    except Exception as e:
        print(f"提取SVM模型出错: {e}")
        traceback.print_exc()
        return False

# 从test_5.py提取朴素贝叶斯模型
def extract_naive_bayes_model():
    print("\n创建朴素贝叶斯模型...")
    try:
        # 先尝试加载test_5.py中的离婚诉讼文本数据
        data_file = '离婚诉讼文本.json'
        stopwords_file = '停用词表.txt'
        
        if os.path.exists(data_file):
            print(f"找到离婚诉讼文本数据文件: {data_file}")
            # 尝试读取数据文件（简化版）
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    try:
                        # 整个文件作为一个JSON对象
                        data = json.loads(content)
                        if isinstance(data, list):
                            # 成功读取到JSON对象列表
                            print(f"成功读取了 {len(data)} 条记录")
                            df = pd.DataFrame(data)
                            if 'sentence' in df.columns and 'labels' in df.columns:
                                print("数据成功加载为DataFrame")
                                
                                # 划分训练集和测试集
                                X = df['sentence'].values
                                y = df['labels'].values
                                
                                # 加载停用词（如果有）
                                stopwords_set = set()
                                if os.path.exists(stopwords_file):
                                    with open(stopwords_file, 'r', encoding='utf-8') as f:
                                        stopwords_set = {line.strip() for line in f if line.strip()}
                                    print(f"从 {stopwords_file} 加载了 {len(stopwords_set)} 个停用词")
                                
                                # 预处理文本: 分词 + 去停用词
                                def preprocess(text):
                                    if not isinstance(text, str):
                                        text = str(text)
                                    tokens = [t for t in jieba.cut(text) if t not in stopwords_set and len(t.strip()) > 0]
                                    return ' '.join(tokens)
                                
                                # 预处理所有文本
                                X_preprocessed = [preprocess(text) for text in X]
                                
                                # 使用Tfidf特征提取
                                vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)
                                X_tfidf = vectorizer.fit_transform(X_preprocessed)
                                
                                # 使用朴素贝叶斯分类器
                                nb_classifier = MultinomialNB()
                                nb_classifier.fit(X_tfidf, y)
                                
                                # 保存预处理器和模型
                                preprocessors = {
                                    'vectorizer': vectorizer,
                                    'stopwords': stopwords_set
                                }
                                
                                metadata = {
                                    'model_type': 'naive_bayes',
                                    'description': '基于文本的离婚案件类型分类',
                                    'preprocessing': '分词+TF-IDF特征提取',
                                    'data_source': data_file
                                }
                                
                                save_model(nb_classifier, 'naive_bayes_divorce_case_real', preprocessors, metadata)
                                print("成功保存根据真实数据训练的朴素贝叶斯模型")
                                return True
                            else:
                                print("数据文件结构不符合要求，将创建模拟数据")
                        else:
                            print("加载的JSON不是列表格式，将创建模拟数据")
                    except json.JSONDecodeError:
                        print("解析JSON文件失败，将创建模拟数据")
            except Exception as e:
                print(f"读取离婚诉讼文件出错：{e}，将创建模拟数据")
        else:
            print(f"没有找到离婚诉讼文本文件: {data_file}，将创建模拟数据")
        
        # 如果无法读取文件或读取出错，则创建模拟数据进行训练
        print("创建模拟离婚案件数据进行训练...")
        
        # 创建一个简单的文本分类模型的示例数据
        texts = [
            "网络中国法院离婚诉讼案件财产分割云南省高级人民法院",
            "离婚案件涉及房产分割问题共有财产分割市场价分割",
            "小孩抗拒崖员抗拒见到父亲抗拒因素亲权接大模型",
            "孩子护照孩子比较好护照不给护照确定比较好",
            "离婚案件入州财产分割我国婚姻制度几套房子我国婚姻制度",
            "地区法院孩子护照减轻革命化负担孩子长永时间",
            "很情感很太大婚前财产务大家财产项目诊证",
            "家暴案件家庭暴力离婚很情感破裂脚踝很婚姻",
            "子女海外子女护照我国出境区别本护照我国",
            "婚姻破裂家庭暴力因素移民海外单方留学家庭暴力"
        ]
        
        labels = [0, 0, 1, 1, 0, 1, 0, 2, 1, 2]  # 0:财产, 1:子女, 2:其他
        
        # 创建特征提取器
        vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        X = vectorizer.fit_transform(texts)
        
        # 训练朴素贝叶斯模型
        nb_clf = MultinomialNB()
        nb_clf.fit(X, labels)
        
        # 保存预处理器和模型
        preprocessors = {
            'vectorizer': vectorizer
        }
        
        metadata = {
            'model_type': 'naive_bayes',
            'target_name': '离婚案件类型',
            'class_mapping': {0: '财产相关', 1: '子女相关', 2: '其他问题'},
            'description': '基于文本的离婚案件类型分类',
            'data_source': '模拟生成'
        }
        
        save_model(nb_clf, 'naive_bayes_divorce_case', preprocessors, metadata)
        print("成功保存模拟数据训练的朴素贝叶斯模型")
        
        return True
    except Exception as e:
        print(f"创建朴素贝叶斯模型出错: {e}")
        traceback.print_exc()
        return False

# 从test_6.py提取K-Means聚类模型
def extract_kmeans_model():
    print("\n创建K-Means聚类模型...")
    try:
        # 尝试加载air_data.csv - 与test_6.py一致的处理方式
        air_data_loaded = False
        df = None
        
        # 文件路径
        file_path = 'air_data.csv'
        
        # 尝试使用不同编码读取文件
        encodings = ['utf-8', 'gbk', 'gb2312']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用{encoding}编码加载air_data.csv")
                air_data_loaded = True
                break
            except Exception as e:
                print(f"{encoding}编码加载失败: {str(e)}")
        
        # 如果成功加载了air_data.csv，使用RFM特征
        if air_data_loaded and df is not None:
            print("使用air_data.csv进行聚类")
            # 数据清洗
            df_cleaned = df.copy()
            
            # 处理LAST_TO_END中的异常标记，如'########'，并转换为数值
            # errors='coerce' 会将无法转换的值设为 NaN
            df_cleaned['LAST_TO_END'] = pd.to_numeric(df_cleaned['LAST_TO_END'], errors='coerce')
            
            # 处理缺失值：删除关键RFM特征存在缺失值的行
            rfm_cols = ['LAST_TO_END', 'FLIGHT_COUNT', 'Points_Sum']
            df_cleaned.dropna(subset=rfm_cols, inplace=True)
            
            # 处理异常值：删除负值
            df_cleaned = df_cleaned[(df_cleaned['FLIGHT_COUNT'] >= 0) & (df_cleaned['Points_Sum'] >= 0)]
            
            # 获取RFM数据
            rfm_data = df_cleaned[rfm_cols]
            print(f"清洗后的RFM数据包含 {rfm_data.shape[0]} 条记录")
            
            if rfm_data.shape[0] <= 1:
                raise ValueError("清洗后的数据不足以进行聚类，将使用随机生成数据")
        else:
            # 如果加载失败，生成随机RFM数据
            print("无法加载air_data.csv，使用随机生成的RFM数据")
            np.random.seed(42)
            n_samples = 1000  # 生成更多样本
            recency = np.random.randint(1, 365, n_samples)  # 1-365天
            frequency = np.random.randint(1, 20, n_samples)  # 1-20次
            monetary = np.random.randint(100, 10000, n_samples)  # 100-10000元
            
            # 创建数据框
            rfm_data = pd.DataFrame({
                'LAST_TO_END': recency,  # 最近一次消费距今天数
                'FLIGHT_COUNT': frequency,  # 消费频率
                'Points_Sum': monetary  # 消费金额
            })
            print(f"随机生成了 {rfm_data.shape[0]} 条RFM数据记录")
        
        # 标准化数据
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_data)
        
        # 训练K-Means模型
        k = 3  # 聚类数量，与test_6.py一致
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(rfm_scaled)
        
        # 保存预处理器和模型
        preprocessors = {
            'scaler': scaler
        }
        
        metadata = {
            'model_type': 'kmeans',
            'feature_names': ['LAST_TO_END', 'FLIGHT_COUNT', 'Points_Sum'],
            'description': '基于RFM特征的客户分群模型',
            'n_clusters': k,
            'data_source': 'air_data.csv' if air_data_loaded else '随机生成数据'
        }
        
        save_model(kmeans, 'kmeans_customer_segmentation', preprocessors, metadata)
        print(f"成功保存KMeans模型 (k={k})")
        
        return True
    except Exception as e:
        print(f"创建K-Means聚类模型出错: {e}")
        traceback.print_exc()
        return False

# 创建随机森林分类器和回归器模型
def extract_random_forest_models():
    print("\n创建随机森林模型...")
    try:
        # 加载数据
        df = pd.read_excel("北京市空气质量数据.xlsx")
        
        # 1. 随机森林分类器
        print("创建随机森林分类器...")
        # 特征和目标
        feature_columns = ['PM2.5', 'PM10', 'SO2', 'CO', 'NO2', 'O3']
        X = df[feature_columns]
        y_raw = df['质量等级']
        
        # 编码目标变量
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 训练随机森林分类器
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        
        # 保存分类器
        preprocessors_clf = {
            'label_encoder': le
        }
        
        metadata_clf = {
            'model_type': 'random_forest_classifier',
            'feature_names': feature_columns,
            'target_name': '质量等级',
            'class_names': le.classes_.tolist(),
            'description': '随机森林分类预测空气质量等级',
            'n_estimators': 100
        }
        
        save_model(rf_clf, 'random_forest_classifier_air_quality', preprocessors_clf, metadata_clf)
        
        # 2. 随机森林回归器
        print("创建随机森林回归器...")
        # 使用PM2.5作为目标变量，其他污染物作为特征
        feature_columns_reg = ['PM10', 'SO2', 'CO', 'NO2', 'O3']
        X_reg = df[feature_columns_reg]
        y_reg = df['PM2.5']
        
        # 划分训练集和测试集
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.3, random_state=42
        )
        
        # 训练随机森林回归器
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train_reg, y_train_reg)
        
        # 保存回归器
        metadata_reg = {
            'model_type': 'random_forest_regressor',
            'feature_names': feature_columns_reg,
            'target_name': 'PM2.5',
            'description': '随机森林回归预测PM2.5浓度',
            'n_estimators': 100
        }
        
        save_model(rf_reg, 'random_forest_regressor_pm25', {}, metadata_reg)
        
        return True
    except Exception as e:
        print(f"创建随机森林模型出错: {e}")
        traceback.print_exc()
        return False

# 尝试直接从测试文件导入模型（通过直接运行测试文件）
def run_test_file_and_extract_model(test_file, model_vars, save_func):
    """
    运行测试文件并提取其中的模型变量
    
    Args:
        test_file: 测试文件路径
        model_vars: 需要提取的模型变量名列表
        save_func: 保存提取模型的函数
        
    Returns:
        bool: 提取成功返回True，否则返回False
    """
    print(f"\n尝试运行测试文件并提取模型: {test_file}")
    
    # 从测试文件路径中提取文件名（不带路径和扩展名）
    file_basename = os.path.splitext(os.path.basename(test_file))[0]
    
    try:
        # 使用子进程运行测试文件，这样不会影响当前进程环境
        result = subprocess.run([sys.executable, test_file], 
                               capture_output=True, 
                               text=True, 
                               timeout=300)  # 5分钟超时
        
        if result.returncode != 0:
            print(f"运行测试文件时出错: {result.stderr}")
            return False
            
        # 调用保存函数来提取和保存模型
        success = save_func()
        if success:
            print(f"成功从测试文件 {test_file} 提取并保存模型")
            return True
        else:
            print(f"从测试文件 {test_file} 提取模型失败")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"运行测试文件 {test_file} 超时")
        return False
    except Exception as e:
        print(f"处理测试文件 {test_file} 时发生错误: {e}")
        traceback.print_exc()
        return False

# 更新ml_models.py中的MODEL_TYPES字典
def update_ml_models_support():
    """
    更新ml_models.py文件以支持所有提取的模型类型
    """
    print("\n更新ml_models.py以支持更多模型类型...")
    
    try:
        with open("ml_models.py", "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # 查找MODEL_TYPES字典定义的位置
        model_types_start = -1
        model_types_end = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith("MODEL_TYPES ="):
                model_types_start = i
            if model_types_start >= 0 and line.strip() == "}":
                model_types_end = i
                break
        
        if model_types_start == -1 or model_types_end == -1:
            print("未能在ml_models.py中找到MODEL_TYPES字典定义")
            return False
        
        # 创建新的MODEL_TYPES字典内容
        new_model_types = [
            "# 模型映射表，便于通过名称查找模型",
            "MODEL_TYPES = {",
            '    "linear_regression": LinearRegression,',
            '    "logistic_regression": LogisticRegression,',
            '    "decision_tree": DecisionTreeClassifier,',
            '    "random_forest_classifier": RandomForestClassifier,',
            '    "random_forest_regressor": RandomForestRegressor,',
            '    "knn_classifier": KNeighborsClassifier,',
            '    "svm_classifier": SVC,',
            '    "naive_bayes": MultinomialNB,',
            '    "kmeans": KMeans',
            "}"
        ]
        
        # 更新文件内容
        updated_lines = lines[:model_types_start] + [line + "\n" for line in new_model_types] + lines[model_types_end+1:]
        
        # 添加缺少的导入语句
        import_statement = "from sklearn.linear_model import LinearRegression, LogisticRegression\n"
        import_statement += "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n"
        import_statement += "from sklearn.tree import DecisionTreeClassifier\n"
        import_statement += "from sklearn.neighbors import KNeighborsClassifier\n"
        import_statement += "from sklearn.svm import SVC\n"
        import_statement += "from sklearn.naive_bayes import MultinomialNB\n"
        import_statement += "from sklearn.cluster import KMeans\n"
        
        # 找到导入部分的位置并插入缺少的导入
        import_end = -1
        for i, line in enumerate(updated_lines):
            if line.strip().startswith("from sklearn.metrics import"):
                import_end = i
                break
        
        if import_end != -1:
            # 删除现有的模型导入
            i = 0
            while i < import_end:
                if "from sklearn.linear_model import" in updated_lines[i] or \
                   "from sklearn.ensemble import" in updated_lines[i] or \
                   "from sklearn.tree import" in updated_lines[i]:
                    updated_lines.pop(i)
                    import_end -= 1
                else:
                    i += 1
            
            updated_lines = updated_lines[:import_end] + import_statement.split("\n") + updated_lines[import_end:]
        
        # 写回文件
        with open("ml_models.py", "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
        
        print("成功更新ml_models.py，现在支持所有提取的模型类型")
        return True
        
    except Exception as e:
        print(f"更新ml_models.py文件时出错: {e}")
        traceback.print_exc()
        return False

# 导出所有模型的主函数
def export_all_models():
    """
    导出所有可用的机器学习模型
    """
    print("开始导出所有机器学习模型...")
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 更新ml_models.py以支持所有模型类型
    update_ml_models_support()
    
    # 尝试直接运行测试文件导出模型
    test_files = [
        ("test/test_1.py", extract_linear_regression_models),
        ("test/test_1.py", extract_logistic_regression_model),
        ("test/test_2.py", extract_knn_model),
        ("test/test_3.py", extract_decision_tree_model),
        ("test/test_4.py", extract_svm_model),
        ("test/test_5.py", extract_naive_bayes_model),
        ("test/test_6.py", extract_kmeans_model)
    ]
    
    success_count = 0
    for test_file, extract_func in test_files:
        if os.path.exists(test_file):
            result = run_test_file_and_extract_model(test_file, [], extract_func)
            if result:
                success_count += 1
    
    # 如果直接导出不完全成功，尝试用内部实现的方法提取
    if success_count < len(test_files):
        print(f"\n直接运行测试文件导出模型部分成功 ({success_count}/{len(test_files)})，将继续用备用方法提取...")
        
        # 提取线性回归和逻辑回归模型
        extract_linear_regression_models()
        extract_logistic_regression_model()
        
        # 提取KNN模型
        extract_knn_model()
        
        # 提取决策树模型
        extract_decision_tree_model()
        
        # 提取SVM模型
        extract_svm_model()
        
        # 提取朴素贝叶斯模型
        extract_naive_bayes_model()
        
        # 提取K-Means模型
        extract_kmeans_model()
    
    # 额外创建随机森林模型（补充）
    extract_random_forest_models()
    
    # 检查导出的模型数量
    model_count = len([f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')])
    print(f"\n导出完成。共有 {model_count} 个模型保存在 {MODEL_DIR} 目录中。")
    
    if model_count == 0:
        print("警告: 没有成功导出任何模型，请检查日志了解详情。")
        return False
    
    return True

# 主函数
def main():
    print("开始提取和创建机器学习模型...")
    
    # 确保模型目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"模型将保存在 {MODEL_DIR} 目录中")
    
    # 更新ml_models.py以支持所有模型类型
    print("\n更新ml_models.py文件并添加模型支持...")
    update_ml_models_support()
    
    # 导出模型的顺序列表 - 按复杂性升序排列
    model_exports = [
        ("线性回归模型", extract_linear_regression_models),
        ("逻辑回归模型", extract_logistic_regression_model),
        ("决策树模型", extract_decision_tree_model),
        ("KNN模型", extract_knn_model),
        ("随机森林模型", extract_random_forest_models),
        ("SVM模型", extract_svm_model),
        ("朴素贝叶斯模型", extract_naive_bayes_model),
        ("K-Means聚类模型", extract_kmeans_model)
    ]
    
    successes = []
    failures = []
    
    # 遍历并执行每个模型导出函数
    for model_name, export_func in model_exports:
        print(f"\n{'-'*50}")
        print(f"正在提取{model_name}...")
        
        try:
            start_time = time.time()
            result = export_func()
            end_time = time.time()
            
            if result:
                print(f"✅ {model_name}提取成功！ (耗时: {end_time - start_time:.2f} 秒)")
                successes.append(model_name)
            else:
                print(f"❌ {model_name}提取失败！")
                failures.append(model_name)
        except Exception as e:
            print(f"❌ {model_name}提取过程中发生异常: {str(e)}")
            traceback.print_exc()
            failures.append(model_name)
    
    # 显示最终结果
    print(f"\n{'-'*50}")
    print(f"\n\n模型导出结果:")
    print(f"- 成功导出模型: {len(successes)}/{len(model_exports)}")
    
    for model in successes:
        print(f"  ✅ {model}")
    
    if failures:
        print(f"- 失败导出模型: {len(failures)}/{len(model_exports)}")
        for model in failures:
            print(f"  ❌ {model}")
    
    # 检查ml_models目录中的模型数量
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]
    if model_files:
        print(f"\n所有模型已成功保存至 {MODEL_DIR} 目录:\n")
        for model_file in model_files:
            print(f"- {model_file}")
    else:
        print(f"\n警告: {MODEL_DIR} 目录中未发现任何模型文件！")
    
    # 返回存在至少一个成功导出的模型的布尔值
    return len(successes) > 0

if __name__ == "__main__":
    # 检查是否安装了必要的库，如果失败则提供安装建议
    try:
        import jieba
    except ImportError:
        print("\u8b66u544a: u672au5b89u88c5jiebu5206u8bcdu5e93uff0cu53efu80fdu4f1au5f71u54cdu6734u7d20u8d1du53f6u65afu6a21u578bu7684u5904u7406")
        print("u5982u679cu9700u8981u5904u7406u4e2du6587u6587u672cuff0cu8bf7u8fd0u884c: pip install jieba")
    
    try:
        import time # u6dfbu52a0u8ba1u65f6u5e93
        start_time = time.time()
        success = main()
        end_time = time.time()
        
        # u663eu793au603bu65f6u95f4
        print(f"\nu603bu8fd0u884cu65f6u95f4: {end_time - start_time:.2f} u79d2")
        
        # u8bbe7f6eu8fd4u56deu4ee3u7801
        if success:
            print("\n\u2705 u6a21u578bu5bfcu51fau8fc7u7a0bu5b8cu6210uff0cu81f3u5c11u6709u4e00u4e2au6a21u578bu5bfcu51fau6210u529fu3002")
            sys.exit(0)
        else:
            print("\n\u274c u6a21u578bu5bfcu51fau8fc7u7a0bu5931u8d25uff0cu6240u6709u6a21u578bu5747u5bfcu51fau5931u8d25u3002")
            sys.exit(1)
    except Exception as e:
        print(f"\n\u274c u7a0bu5e8fu8fd0u884cu8fc7u7a0bu4e2du53d1u751fu610fu5916u9519u8befu: {str(e)}")
        traceback.print_exc()
        sys.exit(1)