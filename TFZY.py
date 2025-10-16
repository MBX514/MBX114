# 导入必要的库
import pandas as pd                                              # 数据处理和分析库
import numpy as np                                               # 数值计算库
import torch                                                     # PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader                 # PyTorch数据加载工具
from transformers import (                                       # Hugging Face Transformers库
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification
)
from sklearn.model_selection import train_test_split             # 数据集划分工具
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, roc_auc_score  # 评估指标
from sklearn.utils.class_weight import compute_class_weight      # 类别权重计算
import re                                                        # 正则表达式库
import os                                                        # 操作系统接口
import glob                                                      # 文件路径匹配
from tqdm import tqdm                                            # 进度条显示
import warnings                                                  # 警告处理

warnings.filterwarnings('ignore')                                # 忽略警告信息

# 设置随机种子
torch.manual_seed(42)                                            # 设置PyTorch随机种子
np.random.seed(42)                                               # 设置NumPy随机种子
torch.backends.cudnn.deterministic = True                        # 确保CUDA卷积操作确定性


class AdvancedReviewDataset(Dataset):
    """高级数据集类，支持动态数据增强"""

    def __init__(self, reviews, labels, tokenizer, max_length=256, augment=False):
        self.reviews = reviews                                    # 存储评论列表
        self.labels = labels                                      # 存储标签列表
        self.tokenizer = tokenizer                                # 分词器实例
        self.max_length = max_length                              # 最大序列长度
        self.augment = augment                                    # 是否启用数据增强

    def __len__(self):
        return len(self.reviews)                                  # 返回数据集大小

    def __getitem__(self, idx):
        review = str(self.reviews[idx])                           # 获取指定索引的评论
        label = self.labels[idx]                                  # 获取对应标签

        # 数据增强
        if self.augment and np.random.random() > 0.7:            # 以70%概率进行数据增强
            review = self._augment_text(review)                  # 调用数据增强方法

        encoding = self.tokenizer(                               # 使用分词器编码文本
            review,
            truncation=True,                                      # 启用截断
            padding='max_length',                                 # 填充到最大长度
            max_length=self.max_length,                           # 设置最大长度
            return_tensors='pt'                                   # 返回PyTorch张量
        )

        return {                                                 # 返回编码后的数据
            'input_ids': encoding['input_ids'].flatten(),        # 展平输入ID张量
            'attention_mask': encoding['attention_mask'].flatten(),  # 展平注意力掩码张量
            'labels': torch.tensor(label, dtype=torch.long)      # 转换为长整型标签张量
        }

    def _augment_text(self, text):
        """文本数据增强"""
        words = text.split()                                      # 按空格分割文本为单词列表
        if len(words) <= 3:                                       # 如果单词数小于等于3，不进行增强
            return text

        # 随机删除
        if np.random.random() > 0.8 and len(words) > 5:          # 以20%概率随机删除单词
            delete_idx = np.random.randint(0, len(words))        # 随机选择删除位置
            words.pop(delete_idx)                                 # 删除指定位置的单词

        # 随机替换同义词（简化版）
        if np.random.random() > 0.9:                             # 以10%概率进行同义词替换
            synonym_dict = {                                      # 同义词字典定义
                'good': ['great', 'nice', 'excellent'],
                'bad': ['poor', 'terrible', 'awful'],
                'best': ['finest', 'greatest', 'top'],
                'worst': ['poorest', 'terriblest', 'awfullest']
            }
            for i, word in enumerate(words):                     # 遍历所有单词
                if word.lower() in synonym_dict and np.random.random() > 0.7:  # 如果单词在同义词字典中且满足概率条件
                    synonyms = synonym_dict[word.lower()]        # 获取同义词列表
                    words[i] = np.random.choice(synonyms)        # 随机选择一个同义词替换

        return ' '.join(words)                                   # 重新组合为字符串


class OptimalFakeReviewDetector:
    """最优性能虚假评论检测器"""

    def __init__(self, model_path=None, use_local=True):
        if model_path is None:                                    # 如果未提供模型路径
            self.model_path = "./models/deberta-v3-base"         # 使用默认路径
        else:
            self.model_path = model_path                         # 使用提供的路径

        self.use_local = use_local                               # 是否使用本地模型
        self.tokenizer = None                                    # 分词器初始化
        self.model = None                                        # 模型初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择设备
        self.class_weights = None                                # 类别权重初始化

        print(f"使用设备: {self.device}")                        # 打印设备信息
        print(f"模型路径: {self.model_path}")                    # 打印模型路径

    def find_csv_files(self, directory_path):
        csv_files = glob.glob(os.path.join(directory_path, "**", "*.csv"), recursive=True)  # 递归查找所有CSV文件
        if not csv_files:                                        # 如果未找到文件
            csv_files = glob.glob(os.path.join(directory_path, "*.csv"))  # 在当前目录查找
        return csv_files                                         # 返回文件列表

    def load_and_preprocess_data(self, data_path, use_all_fake=True):
        """加载和预处理数据 - 修复标签生成问题"""
        print("开始加载数据...")                                  # 开始数据加载提示

        if os.path.isfile(data_path):                            # 如果路径是单个文件
            print(f"直接读取文件: {data_path}")                  # 打印文件路径
            csv_files = [data_path]                              # 文件列表设为单个文件
        else:
            print(f"在目录中查找CSV文件: {data_path}")          # 打印目录路径
            csv_files = self.find_csv_files(data_path)           # 查找所有CSV文件
            print(f"找到 {len(csv_files)} 个CSV文件")           # 打印找到的文件数量

        if not csv_files:                                        # 如果没有找到CSV文件
            raise FileNotFoundError(f"在路径 {data_path} 中没有找到CSV文件")  # 抛出文件未找到错误

        df_list = []                                             # 数据框列表初始化
        for csv_file in csv_files:                               # 遍历所有CSV文件
            print(f"读取文件: {csv_file}")                      # 打印当前读取文件
            try:
                temp_df = pd.read_csv(csv_file, low_memory=False)  # 读取CSV文件，禁用低内存模式
                df_list.append(temp_df)                          # 添加到数据框列表
                print(f"成功读取，形状: {temp_df.shape}")       # 打印读取成功和形状
            except Exception as e:                               # 捕获读取异常
                print(f"读取文件 {csv_file} 时出错: {e}")      # 打印错误信息
                continue                                         # 继续下一个文件

        if not df_list:                                          # 如果没有成功读取任何文件
            raise ValueError("没有成功读取任何CSV文件")          # 抛出数值错误

        df = pd.concat(df_list, ignore_index=True)               # 合并所有数据框，重置索引
        print(f"合并后数据形状: {df.shape}")                    # 打印合并后数据形状

        review_col = self._find_review_column(df)                # 查找评论内容列
        rating_col = self._find_rating_column(df)                # 查找评分列

        df = df.rename(columns={review_col: 'reviewContent', rating_col: 'score'})  # 重命名列
        df = df.dropna(subset=['reviewContent'])                 # 删除评论内容为空的行
        df['score'] = pd.to_numeric(df['score'], errors='coerce')  # 转换评分为数值类型，无法转换设为NaN
        df = df.dropna(subset=['score'])                         # 删除评分为空的行

        print(f"处理后数据形状: {df.shape}")                    # 打印处理后数据形状

        # 生成高级标签 - 修复版本
        reviews_all = df['reviewContent'].astype(str).tolist()   # 获取所有评论内容列表
        ratings_all = df['score'].astype(float).tolist()         # 获取所有评分列表
        labels_all = self._generate_simplified_labels(reviews_all, ratings_all)  # 生成简化标签

        # 添加到DataFrame中
        df['fake_label'] = labels_all                            # 添加虚假评论标签列

        labels_all_array = np.array(labels_all)                  # 转换为NumPy数组
        print(f"原始标签分布 - 真实评论: {np.sum(labels_all_array == 0)}, 虚假评论: {np.sum(labels_all_array == 1)}")  # 打印标签分布

        # 如果虚假评论太少，使用更宽松的平衡策略
        df_real = df[df['fake_label'] == 0]                      # 筛选真实评论
        df_fake = df[df['fake_label'] == 1]                      # 筛选虚假评论

        print(f"真实评论数: {len(df_real)}, 虚假评论数: {len(df_fake)}")  # 打印各类评论数量

        # 确保有足够的虚假评论
        if len(df_fake) == 0:                                    # 如果没有虚假评论
            print("警告: 没有检测到虚假评论，使用启发式方法生成...")  # 打印警告信息
            # 使用简单的启发式方法生成虚假评论标签
            df_fake = self._generate_heuristic_fake_reviews(df)  # 调用启发式方法生成虚假评论
            print(f"启发式生成的虚假评论数: {len(df_fake)}")    # 打印生成数量

        if len(df_fake) == 0:                                    # 如果仍然没有虚假评论
            # 如果仍然没有虚假评论，随机选择一部分作为虚假评论
            fake_count = min(5000, len(df_real) // 10)           # 计算虚假评论数量
            df_fake = df_real.sample(n=fake_count, random_state=42)  # 从真实评论中随机采样作为虚假评论
            df_real = df_real.drop(df_fake.index)                # 从真实评论中删除被选中的
            print(f"随机选择的虚假评论数: {len(df_fake)}")      # 打印随机选择数量

        # 平衡策略
        min_count = min(len(df_real), len(df_fake))              # 计算最小样本数
        if min_count > 10000:                                    # 如果最小样本数超过10000
            min_count = 10000                                    # 限制最大样本数为10000

        df_real_balanced = df_real.sample(n=min_count, random_state=42)  # 平衡采样真实评论
        df_fake_balanced = df_fake.sample(n=min_count, random_state=42)  # 平衡采样虚假评论

        df_balanced = pd.concat([df_real_balanced, df_fake_balanced])  # 合并平衡后的数据
        df_balanced = df_balanced.sample(frac=1, random_state=42)  # 打乱数据顺序

        print(f"平衡后数据形状: {df_balanced.shape}")           # 打印平衡后数据形状

        reviews = df_balanced['reviewContent'].astype(str).tolist()  # 获取平衡后的评论列表
        labels = df_balanced['fake_label'].tolist()              # 获取平衡后的标签列表

        print(f"最终标签分布 - 真实评论: {labels.count(0)}, 虚假评论: {labels.count(1)}")  # 打印最终标签分布

        return reviews, labels                                   # 返回评论和标签

    def _generate_heuristic_fake_reviews(self, df):
        """使用启发式方法生成虚假评论"""
        # 基于评分的简单启发式
        df_extreme_rating = df[df['score'].isin([1, 5])].copy()  # 筛选极端评分(1分或5分)

        # 基于评论长度的启发式
        df['review_length'] = df['reviewContent'].str.len()      # 计算评论长度
        df_short_reviews = df[df['review_length'] < 20].copy()   # 筛选短评论

        # 基于关键词的启发式
        fake_keywords = ['excellent', 'perfect', 'amazing', 'worst', 'terrible', 'awful',  # 定义虚假评论关键词
                         'highly recommend', 'must have', 'five stars', 'one star']
        keyword_mask = df['reviewContent'].str.lower().str.contains('|'.join(fake_keywords), na=False)  # 创建关键词掩码
        df_keyword_reviews = df[keyword_mask].copy()             # 筛选包含关键词的评论

        # 合并所有可能的虚假评论
        fake_candidates = pd.concat([df_extreme_rating, df_short_reviews, df_keyword_reviews])  # 合并所有候选虚假评论
        fake_candidates = fake_candidates.drop_duplicates()      # 去除重复项

        return fake_candidates                                   # 返回虚假评论候选

    def _generate_simplified_labels(self, reviews, ratings):
        """简化的标签生成规则 - 确保生成足够的虚假评论"""
        labels = []                                              # 标签列表初始化

        for review, rating in tqdm(zip(reviews, ratings), total=len(reviews), desc="生成标签"):  # 遍历所有评论和评分，显示进度条
            review_lower = review.lower().strip()                # 转换为小写并去除首尾空格

            # 简化的评分规则
            fake_score = 0                                       # 虚假分数初始化

            # 1. 极端评分
            if rating == 1 or rating == 5:                       # 如果是极端评分
                fake_score += 1.0                                # 增加虚假分数

            # 2. 评论长度
            if len(review) < 20:                                 # 如果评论过短
                fake_score += 1.0                                # 增加虚假分数
            elif len(review) > 500:                              # 如果评论过长
                fake_score += 0.5                                # 增加虚假分数

            # 3. 极端情感词汇
            extreme_words = ['excellent', 'perfect', 'amazing', 'outstanding', 'fantastic',  # 定义极端情感词汇
                             'brilliant', 'awesome', 'incredible', 'best ever', 'love it',
                             'highly recommend', 'must have', 'definitely buy', 'five stars',
                             'worst', 'terrible', 'horrible', 'awful', 'disgusting',
                             'rubbish', 'garbage', 'waste of money', 'never again',
                             'poor quality', 'one star', 'hate']

            for word in extreme_words:                           # 遍历极端情感词汇
                if word in review_lower:                         # 如果评论包含该词汇
                    fake_score += 0.5                            # 增加虚假分数
                    break                                        # 找到即跳出循环

            # 4. 重复词汇
            words = review_lower.split()                         # 分割评论为单词列表
            if len(words) > 5:                                   # 如果单词数大于5
                word_counts = {}                                 # 单词计数字典初始化
                for word in words:                               # 遍历所有单词
                    if len(word) > 3:                            # 如果单词长度大于3
                        word_counts[word] = word_counts.get(word, 0) + 1  # 计数单词出现次数

                if word_counts:                                  # 如果字典不为空
                    max_repeat = max(word_counts.values())       # 获取最大重复次数
                    if max_repeat >= 3:                          # 如果最大重复次数大于等于3
                        fake_score += 0.8                        # 增加虚假分数

            # 5. 大写比例
            if len(review) > 0:                                  # 如果评论不为空
                upper_ratio = sum(1 for c in review if c.isupper()) / len(review)  # 计算大写字母比例
                if upper_ratio > 0.3:                            # 如果大写比例超过30%
                    fake_score += 0.5                            # 增加虚假分数

            # 动态阈值 - 更宽松
            threshold = 1.5                                      # 设置动态阈值

            label = 1 if fake_score >= threshold else 0          # 根据阈值确定标签
            labels.append(label)                                 # 添加标签到列表

        return np.array(labels)                                  # 返回标签数组

    def _find_review_column(self, df):
        possible_names = ['reviewContent', 'review', 'content', 'text', 'comment', 'review_text']  # 可能的评论列名
        for name in possible_names:                              # 遍历可能名称
            if name in df.columns:                               # 如果列存在
                return name                                      # 返回列名

        for col in df.columns:                                   # 遍历所有列
            if df[col].dtype == 'object' and len(df[col].dropna()) > 0:  # 如果是对象类型且有非空值
                sample = str(df[col].dropna().iloc[0])           # 获取样本值
                if len(sample) > 20:                             # 如果样本长度大于20
                    return col                                   # 返回列名

        raise ValueError("无法确定评论内容列")                   # 抛出数值错误

    def _find_rating_column(self, df):
        possible_names = ['score', 'rating', 'stars', 'review_score', 'overall']  # 可能的评分列名
        for name in possible_names:                              # 遍历可能名称
            if name in df.columns:                               # 如果列存在
                return name                                      # 返回列名

        for col in df.columns:                                   # 遍历所有列
            if pd.api.types.is_numeric_dtype(df[col]):           # 如果是数值类型
                unique_vals = df[col].dropna().unique()          # 获取唯一值
                if len(unique_vals) <= 10 and min(unique_vals) >= 1 and max(unique_vals) <= 5:  # 如果值在1-5之间且数量合理
                    return col                                   # 返回列名

        print("警告: 没有找到评分列，将使用默认评分")           # 打印警告信息
        return 'score'                                           # 返回默认列名

    def prepare_advanced_datasets(self, reviews, labels, test_size=0.2, augment_train=True):
        """准备高级数据集"""
        print("准备高级数据集...")                               # 打印准备数据集提示

        labels_array = np.array(labels)                          # 转换为NumPy数组
        unique_labels, counts = np.unique(labels_array, return_counts=True)  # 获取唯一标签和计数
        print(f"标签分布: {dict(zip(unique_labels, counts))}")  # 打印标签分布

        # 检查是否有足够的样本
        if len(reviews) == 0:                                    # 如果没有评论
            raise ValueError("数据集为空，无法继续处理")         # 抛出数值错误

        if len(unique_labels) < 2:                               # 如果标签类别少于2个
            print(f"警告: 只有一个类别 ({unique_labels[0]})，无法进行有意义的训练")  # 打印警告信息
            # 创建一个平衡的数据集
            return self._create_balanced_fallback_dataset(reviews, labels)  # 调用回退数据集创建方法

        # 计算类别权重
        if len(counts) > 1:                                      # 如果有多于一个类别
            self.class_weights = compute_class_weight(           # 计算类别权重
                'balanced',                                      # 使用平衡策略
                classes=np.unique(labels_array),                 # 唯一类别
                y=labels_array                                   # 标签数组
            )
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float).to(self.device)  # 转换为张量并移动到设备
            print(f"类别权重: {self.class_weights}")            # 打印类别权重

        # 分层划分数据集
        train_reviews, test_reviews, train_labels, test_labels = train_test_split(  # 划分训练集和测试集
            reviews, labels, test_size=test_size, random_state=42, stratify=labels  # 按标签分层抽样
        )

        # 再从训练集中划分验证集
        train_reviews, val_reviews, train_labels, val_labels = train_test_split(  # 划分训练集和验证集
            train_reviews, train_labels, test_size=0.125, random_state=42, stratify=train_labels  # 0.125 * 0.8 = 0.1
        )

        print(f"训练集大小: {len(train_reviews)}")              # 打印训练集大小
        print(f"验证集大小: {len(val_reviews)}")                # 打印验证集大小
        print(f"测试集大小: {len(test_reviews)}")               # 打印测试集大小

        # 初始化分词器
        try:
            if self.use_local:                                   # 如果使用本地模型
                print("尝试加载本地DeBERTa分词器...")           # 打印加载提示
                self.tokenizer = AutoTokenizer.from_pretrained(  # 从本地加载分词器
                    self.model_path,
                    local_files_only=True,                       # 仅使用本地文件
                    use_fast=False                               # 不使用快速分词器
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(  # 从网络加载分词器
                    self.model_path,
                    use_fast=False                               # 不使用快速分词器
                )
        except Exception as e:                                   # 捕获异常
            print(f"加载分词器失败: {e}")                       # 打印错误信息
            print("使用DebertaV2Tokenizer作为备用...")          # 打印备用方案提示
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)  # 使用备用分词器

        # 设置pad_token如果不存在
        if self.tokenizer.pad_token is None:                     # 如果填充标记不存在
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用结束标记作为填充标记

        train_dataset = AdvancedReviewDataset(train_reviews, train_labels, self.tokenizer, augment=augment_train)  # 创建训练数据集
        val_dataset = AdvancedReviewDataset(val_reviews, val_labels, self.tokenizer, augment=False)  # 创建验证数据集
        test_dataset = AdvancedReviewDataset(test_reviews, test_labels, self.tokenizer, augment=False)  # 创建测试数据集

        return train_dataset, val_dataset, test_dataset          # 返回三个数据集

    def _create_balanced_fallback_dataset(self, reviews, labels):
        """创建平衡的回退数据集"""
        print("创建平衡的回退数据集...")                         # 打印创建回退数据集提示

        # 如果只有一个类别，创建一个合成数据集
        if len(set(labels)) == 1:                                # 如果只有一个唯一标签
            # 复制数据并创建相反的标签
            opposite_labels = [1 - labels[0]] * len(labels)      # 创建相反标签
            all_reviews = reviews + reviews                      # 复制评论
            all_labels = labels + opposite_labels                # 合并标签
        else:
            all_reviews = reviews                                # 使用原始评论
            all_labels = labels                                  # 使用原始标签

        # 重新划分数据集
        train_reviews, test_reviews, train_labels, test_labels = train_test_split(  # 划分数据集
            all_reviews, all_labels, test_size=0.2, random_state=42  # 20%测试集
        )

        train_reviews, val_reviews, train_labels, val_labels = train_test_split(  # 划分训练集和验证集
            train_reviews, train_labels, test_size=0.125, random_state=42  # 12.5%验证集
        )

        print(f"回退训练集大小: {len(train_reviews)}")          # 打印回退训练集大小
        print(f"回退验证集大小: {len(val_reviews)}")            # 打印回退验证集大小
        print(f"回退测试集大小: {len(test_reviews)}")           # 打印回退测试集大小

        # 初始化分词器
        try:
            if self.use_local:                                   # 如果使用本地模型
                self.tokenizer = AutoTokenizer.from_pretrained(  # 从本地加载分词器
                    self.model_path,
                    local_files_only=True,                       # 仅使用本地文件
                    use_fast=False                               # 不使用快速分词器
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(  # 从网络加载分词器
                    self.model_path,
                    use_fast=False                               # 不使用快速分词器
                )
        except Exception as e:                                   # 捕获异常
            print(f"加载分词器失败: {e}")                       # 打印错误信息
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_path)  # 使用备用分词器

        if self.tokenizer.pad_token is None:                     # 如果填充标记不存在
            self.tokenizer.pad_token = self.tokenizer.eos_token  # 使用结束标记作为填充标记

        train_dataset = AdvancedReviewDataset(train_reviews, train_labels, self.tokenizer, augment=False)  # 创建训练数据集
        val_dataset = AdvancedReviewDataset(val_reviews, val_labels, self.tokenizer, augment=False)  # 创建验证数据集
        test_dataset = AdvancedReviewDataset(test_reviews, test_labels, self.tokenizer, augment=False)  # 创建测试数据集

        return train_dataset, val_dataset, test_dataset          # 返回三个数据集

    def initialize_optimal_model(self):
        """初始化最优模型"""
        print("初始化DeBERTa模型...")                           # 打印模型初始化提示

        try:
            if self.use_local:                                   # 如果使用本地模型
                self.model = AutoModelForSequenceClassification.from_pretrained(  # 从本地加载模型
                    self.model_path,
                    num_labels=2,                                # 二分类任务
                    local_files_only=True,                       # 仅使用本地文件
                    attention_probs_dropout_prob=0.1,           # 注意力概率dropout
                    hidden_dropout_prob=0.1                     # 隐藏层dropout
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(  # 从网络加载模型
                    self.model_path,
                    num_labels=2,                                # 二分类任务
                    attention_probs_dropout_prob=0.1,           # 注意力概率dropout
                    hidden_dropout_prob=0.1                     # 隐藏层dropout
                )
        except Exception as e:                                   # 捕获异常
            print(f"加载模型失败: {e}")                         # 打印错误信息
            print("使用DebertaV2ForSequenceClassification作为备用...")  # 打印备用方案提示
            self.model = DebertaV2ForSequenceClassification.from_pretrained(  # 使用备用模型
                self.model_path,
                num_labels=2                                     # 二分类任务
            )

        self.model.to(self.device)                               # 将模型移动到指定设备

    def compute_advanced_metrics(self, eval_pred):
        """计算高级评估指标"""
        predictions, labels = eval_pred                          # 解包预测和标签

        # 对于多分类，取最大概率的类别
        if predictions.ndim > 1 and predictions.shape[1] > 1:   # 如果是多分类预测
            predictions = np.argmax(predictions, axis=1)         # 取最大概率的类别
        else:
            predictions = (predictions > 0).astype(int)          # 二分类阈值处理

        predictions = predictions.flatten()                      # 展平预测数组
        labels = labels.flatten()                                # 展平标签数组

        accuracy = accuracy_score(labels, predictions)           # 计算准确率
        precision, recall, f1, _ = precision_recall_fscore_support(  # 计算精确率、召回率、F1分数
            labels, predictions, average='binary', zero_division=0  # 二分类平均，零除处理
        )

        # 计算AUC-ROC
        try:
            if hasattr(eval_pred, 'predictions') and eval_pred.predictions.ndim > 1:  # 如果有预测概率
                probas = torch.softmax(torch.tensor(eval_pred.predictions), dim=1).numpy()  # 计算softmax概率
                auc_roc = roc_auc_score(labels, probas[:, 1])    # 计算AUC-ROC
            else:
                auc_roc = 0.0                                    # 默认AUC-ROC为0
        except:
            auc_roc = 0.0                                        # 异常处理，AUC-ROC为0

        # 详细分类报告
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)  # 生成分类报告

        return {                                                 # 返回评估指标字典
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'class_0_precision': report['0']['precision'],       # 类别0精确率
            'class_0_recall': report['0']['recall'],             # 类别0召回率
            'class_0_f1': report['0']['f1-score'],               # 类别0 F1分数
            'class_1_precision': report['1']['precision'],       # 类别1精确率
            'class_1_recall': report['1']['recall'],             # 类别1召回率
            'class_1_f1': report['1']['f1-score'],               # 类别1 F1分数
        }

    def train_optimal_model(self, train_dataset, val_dataset, epochs=16):
        """训练最优模型"""
        print("开始训练最优DeBERTa模型...")                     # 打印训练开始提示

        # 使用经验证的最优参数
        params = {                                               # 训练参数字典
            'learning_rate': 2e-5,                               # 学习率
            'per_device_train_batch_size': 16,                   # 每设备训练批次大小
            'num_train_epochs': epochs,                          # 训练轮数
            'weight_decay': 0.01,                                # 权重衰减
            'warmup_ratio': 0.2                                  # 预热比例
        }

        try:
            training_args = TrainingArguments(                   # 创建训练参数
                output_dir='./optimal_deberta_results',          # 输出目录
                num_train_epochs=params['num_train_epochs'],     # 训练轮数
                per_device_train_batch_size=params['per_device_train_batch_size'],  # 每设备训练批次大小
                per_device_eval_batch_size=32,                   # 每设备评估批次大小
                learning_rate=params['learning_rate'],           # 学习率
                warmup_ratio=params['warmup_ratio'],             # 预热比例
                weight_decay=params['weight_decay'],             # 权重衰减
                logging_dir='./optimal_deberta_logs',            # 日志目录
                logging_steps=50,                                # 日志记录步数
                evaluation_strategy="epoch",                     # 评估策略按轮次
                save_strategy="epoch",                           # 保存策略按轮次
                load_best_model_at_end=True,                     # 结束时加载最佳模型
                metric_for_best_model="f1",                      # 最佳模型指标为F1
                greater_is_better=True,                          # 指标越大越好
                dataloader_pin_memory=False,                     # 不固定内存
                gradient_accumulation_steps=1,                   # 梯度累积步数
                fp16=True,                                       # 启用混合精度训练
                report_to=None,                                  # 禁用wandb等报告
            )
        except TypeError:                                        # 捕获类型错误
            training_args = TrainingArguments(                   # 创建兼容版本训练参数
                output_dir='./optimal_deberta_results',          # 输出目录
                num_train_epochs=params['num_train_epochs'],     # 训练轮数
                per_device_train_batch_size=params['per_device_train_batch_size'],  # 每设备训练批次大小
                per_device_eval_batch_size=32,                   # 每设备评估批次大小
                learning_rate=params['learning_rate'],           # 学习率
                warmup_ratio=params['warmup_ratio'],             # 预热比例
                weight_decay=params['weight_decay'],             # 权重衰减
                logging_dir='./optimal_deberta_logs',            # 日志目录
                logging_steps=50,                                # 日志记录步数
                eval_strategy="epoch",                           # 评估策略按轮次
                save_strategy="epoch",                           # 保存策略按轮次
                load_best_model_at_end=True,                     # 结束时加载最佳模型
                metric_for_best_model="f1",                      # 最佳模型指标为F1
                greater_is_better=True,                          # 指标越大越好
                dataloader_pin_memory=False,                     # 不固定内存
                gradient_accumulation_steps=1,                   # 梯度累积步数
                fp16=True,                                       # 启用混合精度训练
                report_to=None,                                  # 禁用wandb等报告
            )

        # 数据收集器
        data_collator = DataCollatorWithPadding(                 # 创建数据收集器
            tokenizer=self.tokenizer,                            # 使用当前分词器
            padding='longest',                                   # 填充到最长序列
            max_length=256,                                      # 最大长度
            pad_to_multiple_of=8                                 # 填充到8的倍数
        )

        self.trainer = Trainer(                                  # 创建训练器
            model=self.model,                                    # 使用当前模型
            args=training_args,                                  # 使用训练参数
            train_dataset=train_dataset,                         # 训练数据集
            eval_dataset=val_dataset,                            # 验证数据集
            compute_metrics=self.compute_advanced_metrics,       # 计算评估指标的方法
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 早停回调
            data_collator=data_collator                          # 数据收集器
        )

        # 训练
        try:
            print("开始训练...")                                 # 打印训练开始提示
            self.trainer.train()                                 # 开始训练
            print("最优DeBERTa训练完成!")                       # 打印训练完成提示
        except Exception as e:                                   # 捕获训练异常
            print(f"训练错误: {e}")                             # 打印错误信息
            # 回退到简单训练
            training_args = TrainingArguments(                   # 创建简单训练参数
                output_dir='./fallback_results',                 # 输出目录
                num_train_epochs=3,                              # 训练轮数
                per_device_train_batch_size=8,                   # 每设备训练批次大小
                logging_steps=10,                                # 日志记录步数
            )
            self.trainer = Trainer(                              # 创建简单训练器
                model=self.model,                                # 使用当前模型
                args=training_args,                              # 使用简单训练参数
                train_dataset=train_dataset,                     # 训练数据集
            )
            self.trainer.train()                                 # 开始简单训练

    def evaluate_comprehensive(self, test_dataset):
        """全面评估模型"""
        print("全面评估DeBERTa模型...")                         # 打印评估开始提示
        try:
            results = self.trainer.evaluate(test_dataset)        # 评估测试集
            print("\n" + "=" * 50)                              # 打印分隔线
            print("最优模型测试集评估结果")                      # 打印评估结果标题
            print("=" * 50)                                      # 打印分隔线

            metrics = {                                          # 指标名称映射字典
                'eval_loss': '损失',
                'eval_accuracy': '准确率',
                'eval_precision': '精确率',
                'eval_recall': '召回率',
                'eval_f1': 'F1分数',
                'eval_auc_roc': 'AUC-ROC'
            }

            for key, name in metrics.items():                    # 遍历指标字典
                if key in results:                               # 如果指标在结果中
                    print(f"{name}: {results[key]:.4f}")        # 打印指标值

            print("\n详细分类报告:")                            # 打印分类报告标题
            print(
                f"类别 0 (真实评论): Precision={results.get('class_0_precision', 0):.4f}, Recall={results.get('class_0_recall', 0):.4f}, F1={results.get('class_0_f1', 0):.4f}")  # 打印类别0指标
            print(
                f"类别 1 (虚假评论): Precision={results.get('class_1_precision', 0):.4f}, Recall={results.get('class_1_recall', 0):.4f}, F1={results.get('class_1_f1', 0):.4f}")  # 打印类别1指标

            return results                                       # 返回评估结果

        except Exception as e:                                   # 捕获评估异常
            print(f"评估错误: {e}")                             # 打印错误信息
            return None                                          # 返回空值

    def predict_with_confidence(self, review_text):
        """带置信度的预测"""
        try:
            inputs = self.tokenizer(                             # 使用分词器编码文本
                review_text,
                truncation=True,                                  # 启用截断
                padding=True,                                     # 启用填充
                max_length=256,                                   # 最大长度
                return_tensors="pt"                               # 返回PyTorch张量
            ).to(self.device)                                    # 移动到指定设备

            with torch.no_grad():                                # 禁用梯度计算
                outputs = self.model(**inputs)                   # 模型前向传播
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  # 计算softmax概率

            predicted_class = predictions.argmax().item()        # 获取预测类别
            confidence = predictions.max().item()                # 获取最大置信度

            # 计算不确定性
            entropy = -torch.sum(predictions * torch.log(predictions + 1e-9)).item()  # 计算信息熵
            uncertainty = min(entropy / np.log(2), 1.0)          # 归一化不确定性到[0,1]

            label = "虚假评论" if predicted_class == 1 else "真实评论"  # 根据预测类别设置标签

            return label, confidence, uncertainty                # 返回标签、置信度和不确定性

        except Exception as e:                                   # 捕获预测异常
            print(f"预测错误: {e}")                             # 打印错误信息
            return "预测失败", 0.0, 1.0                         # 返回失败信息

    def save_model(self, path):
        """保存模型"""
        if hasattr(self, 'trainer'):                             # 如果有训练器
            self.trainer.save_model(path)                        # 使用训练器保存模型
        else:
            self.model.save_pretrained(path)                     # 直接保存模型
        self.tokenizer.save_pretrained(path)                     # 保存分词器
        print(f"模型已保存到: {path}")                          # 打印保存成功信息


def main():
    """主函数 - 修复版本"""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'          # 设置Hugging Face镜像
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'               # 禁用分词器并行

    # 初始化最优检测器
    detector = OptimalFakeReviewDetector("./models/deberta-v3-base", use_local=True)  # 创建检测器实例

    # 加载数据
    try:
        import kagglehub                                         # 导入Kagglehub库
        print("开始下载数据集...")                               # 打印下载开始提示
        path = kagglehub.dataset_download("jocelyndumlao/shoppingappreviews-dataset")  # 下载数据集
        print(f"数据集路径: {path}")                            # 打印数据集路径
        data_path = path                                         # 设置数据路径
    except Exception as e:                                       # 捕获下载异常
        print(f"Kaggle下载失败: {e}")                           # 打印错误信息
        data_path = "/root/.cache/kagglehub/datasets/jocelyndumlao/shoppingappreviews-dataset/versions/1"  # 使用备用路径
        print(f"使用备用数据路径: {data_path}")                 # 打印备用路径信息

    # 加载和预处理数据 - 使用修复版本
    print("\n=== 数据加载阶段 ===")                             # 打印数据加载阶段标题
    reviews, labels = detector.load_and_preprocess_data(         # 加载和预处理数据
        data_path,
        use_all_fake=True                                        # 使用所有虚假评论
    )

    # 检查数据是否有效
    if len(reviews) == 0:                                        # 如果没有评论数据
        print("错误: 数据集为空，无法继续")                     # 打印错误信息
        return                                                   # 退出函数

    # 准备高级数据集
    print("\n=== 数据准备阶段 ===")                             # 打印数据准备阶段标题
    train_dataset, val_dataset, test_dataset = detector.prepare_advanced_datasets(  # 准备数据集
        reviews, labels,
        test_size=0.2,                                           # 测试集比例20%
        augment_train=True                                       # 启用训练数据增强
    )

    # 初始化模型
    print("\n=== 模型初始化阶段 ===")                           # 打印模型初始化阶段标题
    detector.initialize_optimal_model()                          # 初始化模型

    # 训练最优模型
    print("\n=== 模型训练阶段 ===")                             # 打印模型训练阶段标题
    detector.train_optimal_model(train_dataset, val_dataset, epochs=4)  # 训练模型，4个轮次

    # 全面评估
    print("\n=== 模型评估阶段 ===")                             # 打印模型评估阶段标题
    results = detector.evaluate_comprehensive(test_dataset)      # 全面评估模型

    # 保存模型
    print("\n=== 模型保存阶段 ===")                             # 打印模型保存阶段标题
    detector.save_model("./optimal_fake_review_detector")        # 保存模型

    # 综合测试预测
    print("\n=== 综合测试 ===")                                 # 打印综合测试标题
    test_cases = [                                               # 测试案例列表
        # 明显虚假评论
        "This is the best shopping app ever! Perfect experience! Highly recommended! Five stars!",
        "Amazing! Perfect! Best app! Love it! Definitely buy!",
        "Worst app ever! Terrible! Do not install! Waste of money!",
        "Poor quality! Bad service! Never again! One star!",

        # 真实评论
        "The app is decent but has some bugs that need to be fixed in the next update.",
        "Good overall experience. The delivery was fast and products were as described. Could improve the search function.",
        "I've been using this app for 6 months. The interface is user-friendly but sometimes crashes when loading images.",
        "Average shopping experience. Prices are competitive but customer service response time could be better.",

        # 边界案例
        "Good app.",                                             # 过短
        "This shopping application provides an exceptional user experience with its intuitive interface and comprehensive feature set that significantly enhances the overall online shopping journey for consumers across various product categories.",  # 过长
        "LOVE THIS APP! IT'S PERFECT! BEST EVER!",               # 大量大写
        "Good good good good good app app app app",              # 重复词汇
    ]

    print("\n" + "=" * 80)                                      # 打印分隔线
    print("虚假评论检测系统 - 综合测试结果")                     # 打印测试结果标题
    print("=" * 80)                                              # 打印分隔线

    for i, review in enumerate(test_cases, 1):                  # 遍历测试案例
        label, confidence, uncertainty = detector.predict_with_confidence(review)  # 预测每个案例

        print(f"\n测试案例 {i}:")                               # 打印测试案例编号
        print(f"评论: {review}")                                # 打印评论内容
        print(f"预测: {label}")                                 # 打印预测结果
        print(f"置信度: {confidence:.4f}")                     # 打印置信度
        print(f"不确定性: {uncertainty:.4f}")                  # 打印不确定性
        print(f"质量评估: {'高置信度' if confidence > 0.9 else '中等置信度' if confidence > 0.7 else '低置信度'}")  # 打印质量评估
        print("-" * 80)                                          # 打印分隔线

    # 性能总结
    if results:                                                  # 如果有评估结果
        print("\n" + "=" * 50)                                  # 打印分隔线
        print("性能总结")                                        # 打印性能总结标题
        print("=" * 50)                                          # 打印分隔线
        print(f"最终测试准确率: {results.get('eval_accuracy', 0):.4f}")  # 打印准确率
        print(f"最终测试F1分数: {results.get('eval_f1', 0):.4f}")  # 打印F1分数
        print(f"AUC-ROC: {results.get('eval_auc_roc', 0):.4f}")  # 打印AUC-ROC

        # 性能评级（简洁版）
        accuracy = results.get('eval_accuracy', 0)               # 获取准确率
        if accuracy >= 0.95:                                     # 如果准确率>=95%
            rating = "优秀"                                      # 评级为优秀
        elif accuracy >= 0.90:                                   # 如果准确率>=90%
            rating = "很好"                                      # 评级为很好
        elif accuracy >= 0.85:                                   # 如果准确率>=85%
            rating = "良好"                                      # 评级为良好
        else:
            rating = "一般"                                      # 评级为一般

        print(f"系统评级: {rating}")                            # 打印系统评级


if __name__ == "__main__":
    main()                                                       # 执行主函数