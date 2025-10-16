# 安装必要库（如果尚未安装）
# !pip install kaggle pandas numpy matplotlib seaborn scikit-learn tensorflow
# !sudo apt-get install fonts-wqy-zenhei  # 安装中文字体

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_score, recall_score, \
    f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体 - 解决Ubuntu中文显示问题
import matplotlib as mpl
import os

# 方法1: 尝试使用系统中已安装的中文字体
try:
    # 查找可用的中文字体
    chinese_fonts = [
        'WenQuanYi Zen Hei',  # 文泉驿正黑
        'Noto Sans CJK SC',  # 思源黑体
        'SimHei',  # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'DejaVu Sans'  # 备用字体
    ]

    # 设置字体
    for font in chinese_fonts:
        try:
            mpl.rcParams['font.family'] = font
            mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            # 测试字体是否可用
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
        except:
            continue

    # 如果上述字体都不可用，使用字体管理器查找
    if 'font.family' not in mpl.rcParams or mpl.rcParams['font.family'] == ['sans-serif']:
        from matplotlib.font_manager import FontManager

        fm = FontManager()
        available_fonts = [f.name for f in fm.ttflist]
        chinese_candidates = [f for f in available_fonts if
                              any(keyword in f.lower() for keyword in ['chinese', 'zen', 'hei', 'kai', 'song'])]
        if chinese_candidates:
            mpl.rcParams['font.family'] = chinese_candidates[0]

except Exception as e:
    print(f"字体设置警告: {e}")
    print("将使用默认字体，中文可能显示为方框")

# 设置随机种子保证结果可重现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 加载数据（请确保文件路径正确）
def load_data():
    """
    加载Santander客户交易预测数据集
    返回: 训练集和测试集的DataFrame
    """
    try:
        # 从Kaggle下载后解压的文件路径
        train_df = pd.read_csv('data/santander/train.csv')
        test_df = pd.read_csv('data/santander/test.csv')
        print(f"训练集形状: {train_df.shape}, 测试集形状: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError:
        print("请确保数据集文件存在于当前目录")
        return None, None


# 加载数据
train_df, test_df = load_data()


# 数据探索
def explore_data(df):
    """探索数据集的基本信息"""
    print("数据集基本信息:")
    print(df.info())
    print("\n目标变量分布:")
    print(df['target'].value_counts(normalize=True))
    print("\n缺失值统计:")
    print(df.isnull().sum().sum())  # Santander数据集通常没有缺失值
    print("\n数据描述:")
    print(df.describe())


if train_df is not None:
    explore_data(train_df)


def determine_pca_components(X, variance_threshold=0.95):
    """
    使用PCA确定最佳主成分数量
    参数:
        X: 特征数据
        variance_threshold: 累计方差解释阈值
    返回: 最佳主成分数量
    """
    # 先标准化数据
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 应用PCA，保留所有成分用于分析
    pca = PCA()
    pca.fit(X_scaled)

    # 计算累计方差解释率
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # 找到达到阈值所需的最小成分数
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    print(f"达到{variance_threshold:.1%}方差解释率需要的主成分数量: {n_components}")
    print(f"原始特征数: {X.shape[1]}, 降维后特征数: {n_components}")
    print(f"维度减少: {(1 - n_components / X.shape[1]):.1%}")

    # 绘制方差解释图（使用英文标签避免字体问题）
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-')
    plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold:.0%} threshold')
    plt.axvline(x=n_components, color='g', linestyle='--', label=f'Best components: {n_components}')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Cumulative Explained Variance Ratio')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar(range(1, 21), pca.explained_variance_ratio_[:20])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Top 20 Principal Components Variance Ratio')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return n_components


def preprocess_data_with_pca(train_df, test_df, n_components=None, variance_threshold=0.95):
    """
    使用PCA进行数据预处理的流程
    参数:
        train_df: 训练数据集
        test_df: 测试数据集
        n_components: 指定主成分数量，如果为None则自动确定
        variance_threshold: 方差解释阈值
    返回: 处理后的特征和目标变量
    """
    # 分离特征和目标变量
    X = train_df.drop(['ID_code', 'target'], axis=1)
    y = train_df['target']
    X_test = test_df.drop(['ID_code'], axis=1)

    print(f"原始特征数量: {X.shape[1]}")
    print(f"类别分布 - 活跃客户(1): {y.mean():.3f}, 风险客户(0): {1 - y.mean():.3f}")

    # 数据标准化 - 使用RobustScaler处理异常值
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # 自动确定最佳主成分数量（如果未指定）
    if n_components is None:
        n_components = determine_pca_components(X, variance_threshold)

    # 应用PCA降维
    pca = PCA(n_components=n_components, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"\nPCA降维结果:")
    print(f"原始维度: {X_scaled.shape[1]} → 降维后: {X_pca.shape[1]}")
    print(f"累计方差解释率: {np.sum(pca.explained_variance_ratio_):.3f}")

    # 显示前10个主成分的方差解释率
    print("\n前10个主成分的方差解释率:")
    for i, ratio in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"主成分 {i + 1}: {ratio:.4f}")

    return X_pca, y, X_test_pca, test_df['ID_code'], pca, scaler


# 执行PCA预处理
X, y, X_test, test_ids, pca_model, scaler = preprocess_data_with_pca(train_df, test_df)


def create_model(input_dim, learning_rate=0.001, dropout_rate=0.3):
    """
    创建深度学习模型架构（针对PCA降维后的数据优化）
    参数:
        input_dim: 输入特征维度（PCA降维后）
        learning_rate: 学习率
        dropout_rate: Dropout比率
    返回: 编译后的Keras模型
    """
    # 根据PCA降维后的特征数量调整网络架构
    if input_dim <= 50:
        layer_sizes = [128, 64, 32]
    elif input_dim <= 100:
        layer_sizes = [256, 128, 64]
    else:
        layer_sizes = [512, 256, 128]

    model = Sequential([
        Dense(layer_sizes[0], activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(layer_sizes[1], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),

        Dense(layer_sizes[2], activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate * 0.8),

        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )

    return model


# 创建模型实例
model = create_model(X.shape[1])
print("PCA降维后的模型架构:")
print(model.summary())


def train_with_validation(X, y, n_folds=5):
    """
    使用交叉验证训练模型（PCA版本）
    参数:
        X: PCA降维后的特征数据
        y: 目标变量
        n_folds: 交叉验证折数
    返回: 训练好的模型和验证结果
    """
    callbacks = [
        EarlyStopping(
            monitor='val_AUC',
            patience=10,
            mode='max',
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_AUC',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_scores = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n=== 训练第 {fold + 1} 折 ===")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_model(X.shape[1])

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=256,
            callbacks=callbacks,
            verbose=1,
            class_weight={0: 1, 1: (len(y_train) - sum(y_train)) / sum(y_train)}
        )

        # 验证集预测
        val_pred_proba = model.predict(X_val)
        val_pred = (val_pred_proba > 0.5).astype(int)  # 使用0.5作为阈值

        val_auc = roc_auc_score(y_val, val_pred_proba)
        val_precision = precision_score(y_val, val_pred)
        val_recall = recall_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred)

        fold_scores.append(val_auc)
        fold_precisions.append(val_precision)
        fold_recalls.append(val_recall)
        fold_f1_scores.append(val_f1)
        models.append(model)

        print(f"第 {fold + 1} 折 AUC: {val_auc:.4f}")
        print(f"第 {fold + 1} 折 精确率: {val_precision:.4f}")
        print(f"第 {fold + 1} 折 召回率: {val_recall:.4f}")
        print(f"第 {fold + 1} 折 F1分数: {val_f1:.4f}")

    print(f"\n平均交叉验证 AUC: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")
    print(f"平均交叉验证 精确率: {np.mean(fold_precisions):.4f} (±{np.std(fold_precisions):.4f})")
    print(f"平均交叉验证 召回率: {np.mean(fold_recalls):.4f} (±{np.std(fold_recalls):.4f})")
    print(f"平均交叉验证 F1分数: {np.mean(fold_f1_scores):.4f} (±{np.std(fold_f1_scores):.4f})")

    return models, fold_scores, fold_precisions, fold_recalls, fold_f1_scores


# 执行训练
models, cv_scores, cv_precisions, cv_recalls, cv_f1_scores = train_with_validation(X, y)


def evaluate_model(models, X, y):
    """评估模型性能"""
    # 使用所有模型的平均预测
    predictions_proba = np.mean([model.predict(X) for model in models], axis=0)
    predictions = (predictions_proba > 0.5).astype(int)  # 使用0.5作为阈值

    auc_score = roc_auc_score(y, predictions_proba)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    print(f"\n=== 最终模型性能评估 ===")
    print(f"AUC 分数: {auc_score:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 打印详细的分类报告
    print("\n详细分类报告:")
    print(classification_report(y, predictions, target_names=['风险客户(0)', '活跃客户(1)']))

    # 绘制混淆矩阵（使用英文标签避免字体问题）
    cm = confusion_matrix(y, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Risk', 'Predicted Active'],
                yticklabels=['Actual Risk', 'Actual Active'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 绘制ROC曲线（使用英文标签避免字体问题）
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, predictions_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Customer Churn Prediction ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return predictions_proba, predictions


# 模型评估
train_predictions_proba, train_predictions = evaluate_model(models, X, y)


def predict_test_set(models, X_test, test_ids):
    """对测试集进行预测"""
    # 模型集成：取所有模型预测的平均值
    test_predictions_proba = np.mean([model.predict(X_test) for model in models], axis=0)
    test_predictions = (test_predictions_proba > 0.5).astype(int)

    # 创建提交文件
    submission = pd.DataFrame({
        'ID_code': test_ids,
        'target': test_predictions_proba.flatten(),
        'target_class': test_predictions.flatten()  # 添加类别预测
    })

    # 保存预测结果
    submission.to_csv('customer_churn_predictions_pca.csv', index=False)
    print("测试集预测完成，结果已保存至 'customer_churn_predictions_pca.csv'")

    # 打印测试集预测分布
    print(f"\n测试集预测分布:")
    print(f"预测为风险客户: {np.sum(test_predictions == 0)}")
    print(f"预测为活跃客户: {np.sum(test_predictions == 1)}")
    print(f"风险客户比例: {np.mean(test_predictions == 0):.3f}")

    return test_predictions_proba, test_predictions


# 预测测试集
test_predictions_proba, test_predictions = predict_test_set(models, X_test, test_ids)


# 不同阈值下的性能分析
def analyze_threshold_performance(models, X, y):
    """分析不同阈值下的模型性能"""
    print("\n=== 不同阈值下的性能分析 ===")

    # 使用所有模型的平均预测概率
    predictions_proba = np.mean([model.predict(X) for model in models], axis=0)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []
    for threshold in thresholds:
        predictions = (predictions_proba > threshold).astype(int)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)

        results.append({
            '阈值': threshold,
            '精确率': precision,
            '召回率': recall,
            'F1分数': f1
        })

        print(f"阈值 {threshold}: 精确率={precision:.4f}, 召回率={recall:.4f}, F1分数={f1:.4f}")

    # 找到最佳F1分数对应的阈值
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions = (predictions_proba > threshold).astype(int)
        f1 = f1_score(y, predictions)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\n最佳F1分数 {best_f1:.4f} 对应的阈值: {best_threshold:.2f}")

    return best_threshold


# 分析不同阈值性能
best_threshold = analyze_threshold_performance(models, X, y)


# PCA特定分析（使用英文标签避免字体问题）
def analyze_pca_results(pca_model, original_features=200):
    """分析PCA降维效果"""
    print("\n=== PCA降维效果分析 ===")

    # 方差解释分析
    explained_variance = pca_model.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"降维比例: {1 - len(explained_variance) / original_features:.1%}")
    print(f"信息保留: {cumulative_variance[-1]:.1%}")

    # 主成分重要性分析
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Cumulative Explained Variance Ratio')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.bar(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Ratio of Each Principal Component')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # 显示前10个主成分的累计贡献
    top_10_components = explained_variance[:10] if len(explained_variance) >= 10 else explained_variance
    plt.pie(top_10_components, labels=[f'PC{i + 1}' for i in range(len(top_10_components))], autopct='%1.1f%%')
    plt.title('Top 10 Principal Components Variance Contribution')

    plt.tight_layout()
    plt.show()


# 分析PCA结果
analyze_pca_results(pca_model, original_features=200)


def create_lightweight_model(best_model, pca_model, scaler, model_name='churn_predictor_pca.h5'):
    """
    创建包含PCA的轻量级模型用于部署
    """
    # 保存神经网络模型
    best_model.save(model_name)

    # 保存PCA和Scaler模型
    import joblib
    joblib.dump(pca_model, 'pca_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print(f"模型已保存为: {model_name}")
    print("PCA和Scaler模型已保存为pca_model.pkl和scaler.pkl")


# 选择最佳模型（AUC最高的）
best_model_idx = np.argmax(cv_scores)
best_model = models[best_model_idx]

create_lightweight_model(best_model, pca_model, scaler)

print("\n=== 模型性能总结 ===")
print(f"交叉验证平均AUC: {np.mean(cv_scores):.4f}")
print(f"交叉验证平均精确率: {np.mean(cv_precisions):.4f}")
print(f"交叉验证平均召回率: {np.mean(cv_recalls):.4f}")
print(f"交叉验证平均F1分数: {np.mean(cv_f1_scores):.4f}")
print(f"推荐使用阈值: {best_threshold:.2f} 以获得最佳F1分数")