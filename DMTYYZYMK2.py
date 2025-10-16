import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)


def prepare_data(data_path):
    """准备训练和测试数据"""
    defect_types = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
    image_paths = []
    labels = []

    for defect in defect_types:
        defect_path = os.path.join(data_path, defect)

        if os.path.exists(defect_path):
            for img_file in os.listdir(defect_path):
                if img_file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(defect_path, img_file)
                    image_paths.append(img_path)
                    labels.append(defect)

    if len(image_paths) == 0:
        print("错误：未找到任何图像文件！")
        return [], [], [], [], [], []

    print(f"找到总计 {len(image_paths)} 张图像")

    # 划分数据集
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels


class RobustSurfaceDefectDataset(Dataset):
    """增强的工业缺陷数据集类，支持复杂环境条件"""

    def __init__(self, image_paths, labels, transform=None, is_train=True,
                 confidence_threshold=0.8, augment_intensity=1.0):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_train = is_train
        self.confidence_threshold = confidence_threshold
        self.augment_intensity = augment_intensity

        # 详细的缺陷描述库
        self.defect_descriptions = {
            'Cr': {
                'description': '裂纹缺陷，表现为线性断裂特征',
                'severity': '高危',
                'location': '表面或内部结构',
                'suggestion': '立即停止使用，进行结构性评估，考虑更换零件'
            },
            'In': {
                'description': '夹杂缺陷，材料中含有异物杂质',
                'severity': '中危',
                'location': '材料内部',
                'suggestion': '评估杂质影响范围，进行无损检测确定内部情况'
            },
            'Pa': {
                'description': '斑块缺陷，表面出现不均匀色斑或腐蚀',
                'severity': '低危',
                'location': '表面区域',
                'suggestion': '进行表面清理和防腐处理，定期监控斑块变化'
            },
            'PS': {
                'description': '点蚀缺陷，表面出现局部点状腐蚀',
                'severity': '中危',
                'location': '表面局部区域',
                'suggestion': '评估点蚀深度，进行表面修复和防腐涂层处理'
            },
            'RS': {
                'description': '麻面缺陷，表面粗糙不平整',
                'severity': '低危',
                'location': '整个表面',
                'suggestion': '进行表面打磨处理，改善表面光洁度'
            },
            'Sc': {
                'description': '划痕缺陷，表面有明显的线性刮痕',
                'severity': '低危',
                'location': '表面特定方向',
                'suggestion': '评估划痕深度，进行表面修复，加强操作规范'
            }
        }

    def __len__(self):
        return len(self.image_paths)

    def apply_industrial_augmentation(self, image):
        """工业环境增强：模拟光线变化、角度变化等实际条件"""
        if not self.is_train or self.augment_intensity == 0:
            return image

        # 光线变化增强
        if np.random.random() < 0.6 * self.augment_intensity:
            brightness_factor = np.random.uniform(0.7, 1.3)
            contrast_factor = np.random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(brightness_factor)
            image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        # 模拟角度变化
        if np.random.random() < 0.5 * self.augment_intensity:
            angle = np.random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False)

        # 模拟噪声和模糊
        if np.random.random() < 0.3 * self.augment_intensity:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        return image

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            label = self.labels[idx]

            image = Image.open(img_path).convert('RGB')

            # 应用工业环境增强
            image = self.apply_industrial_augmentation(image)

            if self.transform:
                image = self.transform(image)

            defect_type = label
            defect_info = self.defect_descriptions.get(defect_type, {
                'description': '未知缺陷类型',
                'severity': '待评估',
                'location': '待确认',
                'suggestion': '请进行进一步检测分析'
            })

            # 生成更专业的报告文本
            report_text = (
                f"检测到{defect_type}类型缺陷。{defect_info['description']}。"
                f"缺陷位置：{defect_info['location']}，严重程度：{defect_info['severity']}。"
                f"处理建议：{defect_info['suggestion']}。"
            )

            return {
                'image': image,
                'label': label,
                'report_text': report_text,
                'defect_type': defect_type,
                'defect_info': defect_info
            }

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None


def create_industrial_transforms():
    """创建工业级图像预处理变换，增强鲁棒性"""

    # 训练阶段：更强的数据增强
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 推理阶段
    eval_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform


class ProductionReadyDefectModel(nn.Module):
    """生产环境就绪的缺陷检测模型，修复BatchNorm问题"""

    def __init__(self, num_defect_classes=6, text_model_name='./models/blip-image-captioning-base',
                 use_quantization=False):
        super(ProductionReadyDefectModel, self).__init__()

        # 加载预训练BLIP模型
        self.blip_model = BlipForConditionalGeneration.from_pretrained(text_model_name)

        # 量化支持
        if use_quantization:
            self.blip_model = torch.quantization.quantize_dynamic(
                self.blip_model, {nn.Linear}, dtype=torch.qint8
            )

        vision_config = self.blip_model.config.vision_config
        self.vision_hidden_size = vision_config.hidden_size

        # 修复的缺陷分类头：移除BatchNorm，使用更稳定的结构
        self.defect_classifier = nn.Sequential(
            nn.Linear(self.vision_hidden_size, 512),
            # 移除了BatchNorm1d，因为它需要批次大小>1
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_defect_classes)
        )

        self.classification_criterion = nn.CrossEntropyLoss()

    def get_vision_features(self, pixel_values):
        """增强的特征提取"""
        vision_outputs = self.blip_model.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )

        last_hidden_state = vision_outputs.last_hidden_state
        vision_features = last_hidden_state[:, 0, :]  # 取第一个token作为图像特征

        return vision_features

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None, defect_labels=None):
        vision_features = self.get_vision_features(pixel_values)
        defect_logits = self.defect_classifier(vision_features)

        text_outputs = None
        text_loss = None

        if input_ids is not None:
            text_outputs = self.blip_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            text_loss = text_outputs.loss

        classification_loss = None
        if defect_labels is not None:
            classification_loss = self.classification_criterion(defect_logits, defect_labels)

        return {
            'text_loss': text_loss,
            'classification_loss': classification_loss,
            'defect_logits': defect_logits,
            'logits': text_outputs.logits if text_outputs is not None else None,
            'vision_features': vision_features
        }


def train_model_with_monitoring(model, train_loader, val_loader, processor, device, num_epochs=5):
    """带监控的训练函数，修复批次大小问题"""

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # 训练监控变量
    best_accuracy = 0.0
    patience = 3
    patience_counter = 0
    training_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue

            # 过滤掉None样本，确保批次完整性
            valid_samples = []
            for i in range(len(batch['image'])):
                if batch['image'][i] is not None:
                    valid_samples.append(i)

            if len(valid_samples) < 2:  # 确保批次大小至少为2
                continue

            # 只使用有效样本
            images = torch.stack([batch['image'][i] for i in valid_samples]).to(device)
            report_texts = [batch['report_text'][i] for i in valid_samples]

            defect_labels_indices = []
            for i in valid_samples:
                label = batch['defect_type'][i]
                try:
                    idx = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'].index(label)
                    defect_labels_indices.append(idx)
                except ValueError:
                    continue

            if len(defect_labels_indices) < 2:  # 确保有足够的标签
                continue

            defect_labels = torch.tensor(defect_labels_indices).to(device)

            text_inputs = processor(
                text=report_texts,
                padding='longest',
                return_tensors='pt',
                max_length=128,
                truncation=True
            )

            optimizer.zero_grad()

            outputs = model(
                pixel_values=images,
                input_ids=text_inputs['input_ids'].to(device),
                attention_mask=text_inputs['attention_mask'].to(device),
                labels=text_inputs['input_ids'].to(device),
                defect_labels=defect_labels
            )

            alpha = 0.7
            if outputs['text_loss'] is not None and outputs['classification_loss'] is not None:
                total_loss = alpha * outputs['text_loss'] + (1 - alpha) * outputs['classification_loss']
            elif outputs['text_loss'] is not None:
                total_loss = outputs['text_loss']
            elif outputs['classification_loss'] is not None:
                total_loss = outputs['classification_loss']
            else:
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += total_loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs['defect_logits'], 1)
            train_correct += (predicted == defect_labels).sum().item()
            train_total += defect_labels.size(0)

            if batch_idx % 10 == 0:
                text_loss_val = outputs['text_loss'].item() if outputs['text_loss'] is not None else 0
                class_loss_val = outputs['classification_loss'].item() if outputs[
                                                                              'classification_loss'] is not None else 0
                batch_accuracy = (predicted == defect_labels).float().mean().item()

                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Total Loss: {total_loss.item():.4f}, '
                      f'Text Loss: {text_loss_val:.4f}, '
                      f'Class Loss: {class_loss_val:.4f}, '
                      f'Batch Acc: {batch_accuracy:.4f}')

        # 详细验证评估
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                # 同样过滤验证集的无效样本
                valid_samples = []
                for i in range(len(batch['image'])):
                    if batch['image'][i] is not None:
                        valid_samples.append(i)

                if len(valid_samples) < 2:
                    continue

                images = torch.stack([batch['image'][i] for i in valid_samples]).to(device)
                report_texts = [batch['report_text'][i] for i in valid_samples]

                defect_labels_indices = []
                for i in valid_samples:
                    label = batch['defect_type'][i]
                    try:
                        idx = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'].index(label)
                        defect_labels_indices.append(idx)
                    except ValueError:
                        continue

                if len(defect_labels_indices) < 2:
                    continue

                defect_labels = torch.tensor(defect_labels_indices).to(device)

                text_inputs = processor(
                    text=report_texts,
                    padding='longest',
                    return_tensors='pt',
                    max_length=128,
                    truncation=True
                )

                outputs = model(
                    pixel_values=images,
                    input_ids=text_inputs['input_ids'].to(device),
                    attention_mask=text_inputs['attention_mask'].to(device),
                    labels=text_inputs['input_ids'].to(device),
                    defect_labels=defect_labels
                )

                if outputs['text_loss'] is not None and outputs['classification_loss'] is not None:
                    total_loss = 0.7 * outputs['text_loss'] + 0.3 * outputs['classification_loss']
                elif outputs['text_loss'] is not None:
                    total_loss = outputs['text_loss']
                elif outputs['classification_loss'] is not None:
                    total_loss = outputs['classification_loss']
                else:
                    continue

                val_loss += total_loss.item()

                _, predicted = torch.max(outputs['defect_logits'], 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(defect_labels.cpu().numpy())

        # 计算详细指标
        if len(all_predictions) > 0:
            val_accuracy = np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_predictions)
        else:
            val_accuracy = 0.0

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        # 早停机制
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_defect_detection_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_correct / train_total:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        print(f'  Best Accuracy: {best_accuracy:.4f}')
        print('-' * 50)

        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        })

    return training_history


def generate_production_report(model, image_path, processor, device, confidence_threshold=0.8):
    """生成生产级质检报告"""

    eval_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = eval_transform(image).unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        # 生成文本描述
        generated_ids = model.blip_model.generate(
            pixel_values=image_tensor,
            max_length=100,
            num_beams=3,
            early_stopping=True
        )

        generated_text = processor.decode(generated_ids[0], skip_special_tokens=True)

        # 缺陷分类
        vision_features = model.get_vision_features(image_tensor)
        defect_logits = model.defect_classifier(vision_features)
        predicted_class = torch.argmax(defect_logits, dim=1).item()
        confidence = torch.softmax(defect_logits, dim=1)[0][predicted_class].item()

        defect_types = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
        predicted_defect = defect_types[predicted_class]

        # 置信度控制机制
        needs_manual_review = confidence < confidence_threshold
        review_note = "【需人工复核】" if needs_manual_review else "【自动检测通过】"

        # 构建生产级报告
        defect_descriptions = {
            'Cr': {'severity': '高危', 'suggestion': '立即停止使用，进行结构性评估'},
            'In': {'severity': '中危', 'suggestion': '评估杂质影响范围，进行无损检测'},
            'Pa': {'severity': '低危', 'suggestion': '进行表面清理和防腐处理'},
            'PS': {'severity': '中危', 'suggestion': '评估点蚀深度，进行表面修复'},
            'RS': {'severity': '低危', 'suggestion': '进行表面打磨处理'},
            'Sc': {'severity': '低危', 'suggestion': '评估划痕深度，进行表面修复'}
        }

        defect_info = defect_descriptions.get(predicted_defect, {
            'severity': '待评估', 'suggestion': '请进行进一步检测分析'
        })

        full_report = f"""
工业零件缺陷检测报告 - 生产版本
================================

检测基本信息:
------------
• 图像文件: {os.path.basename(image_path)}
• 检测时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
• 检测状态: {review_note}

检测结果:
---------
• 缺陷类型: {predicted_defect}
• 置信度: {confidence:.3f}
• 严重程度: {defect_info['severity']}
• 模型描述: {generated_text}

处理决策:
---------
{defect_info['suggestion']}

{"⚠️ 注意：置信度较低，建议人工复核确认" if needs_manual_review else "✅ 检测结果可靠，可按建议处理"}

================================
"""

        return full_report, predicted_defect, confidence, needs_manual_review


def evaluate_model_performance(model, test_loader, processor, device):
    """全面评估模型性能"""

    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            # 过滤无效样本
            valid_samples = []
            for i in range(len(batch['image'])):
                if batch['image'][i] is not None:
                    valid_samples.append(i)

            if len(valid_samples) == 0:
                continue

            images = torch.stack([batch['image'][i] for i in valid_samples]).to(device)

            defect_labels_indices = []
            for i in valid_samples:
                label = batch['defect_type'][i]
                try:
                    idx = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'].index(label)
                    defect_labels_indices.append(idx)
                except ValueError:
                    continue

            if len(defect_labels_indices) == 0:
                continue

            defect_labels = torch.tensor(defect_labels_indices).to(device)

            outputs = model(pixel_values=images)
            defect_logits = outputs['defect_logits']

            confidences, predicted = torch.max(torch.softmax(defect_logits, dim=1), 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(defect_labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    if len(all_predictions) == 0:
        print("没有有效的测试样本")
        return {}, np.array([])

    # 计算分类报告
    target_names = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
    report = classification_report(all_labels, all_predictions,
                                   target_names=target_names, output_dict=True)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 误报分析
    false_positive_rate = np.sum((np.array(all_predictions) != np.array(all_labels)) &
                                 (np.array(all_confidences) > 0.8)) / len(all_predictions)

    print("=== 模型性能评估报告 ===")
    print(f"整体准确率: {report['accuracy']:.4f}")
    print(f"误报率: {false_positive_rate:.4f}")
    print("\n各类别性能:")
    for class_name in target_names:
        if class_name in report:
            class_report = report[class_name]
            print(f"{class_name}: 精确率={class_report['precision']:.3f}, "
                  f"召回率={class_report['recall']:.3f}, F1={class_report['f1-score']:.3f}")

    return report, cm


# 主执行函数 - 修复版本
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据集路径
    data_path = "./neu-surface-defect-database"

    if not os.path.exists(data_path):
        print(f"错误：数据集路径 {data_path} 不存在！")
        exit(1)

    print(f"数据集路径: {data_path}")

    # 检查数据集结构
    defect_types = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc']
    found_images = 0
    for defect in defect_types:
        defect_path = os.path.join(data_path, defect)
        if os.path.exists(defect_path):
            image_files = [f for f in os.listdir(defect_path) if f.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg'))]
            found_images += len(image_files)
            print(f"找到缺陷类别 {defect}: {len(image_files)} 张图像")

    if found_images == 0:
        print("错误：未找到任何图像文件！")
        exit(1)

    # 准备数据
    print("\n准备数据...")
    train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = prepare_data(data_path)

    print(f"训练集样本数: {len(train_paths)}")
    print(f"验证集样本数: {len(val_paths)}")
    print(f"测试集样本数: {len(test_paths)}")

    # 创建工业级数据变换
    train_transform, eval_transform = create_industrial_transforms()

    # 创建增强的数据集
    train_dataset = RobustSurfaceDefectDataset(train_paths, train_labels, train_transform, is_train=True)
    val_dataset = RobustSurfaceDefectDataset(val_paths, val_labels, eval_transform, is_train=False)
    test_dataset = RobustSurfaceDefectDataset(test_paths, test_labels, eval_transform, is_train=False)

    # 使用drop_last=True确保批次大小一致
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, drop_last=False)

    # 初始化生产就绪模型
    print("\n初始化生产就绪模型...")
    model_path = "./models/blip-image-captioning-base"

    if not os.path.exists(model_path):
        print(f"错误：找不到模型路径 {model_path}")
        exit(1)

    print(f"从本地路径加载模型: {model_path}")
    processor = BlipProcessor.from_pretrained(model_path)
    model = ProductionReadyDefectModel(text_model_name=model_path).to(device)

    # 训练模型
    print("开始训练模型...")
    training_history = train_model_with_monitoring(
        model, train_loader, val_loader, processor, device, num_epochs=5
    )

    # 加载最佳模型
    if os.path.exists('best_defect_detection_model.pth'):
        model.load_state_dict(torch.load('best_defect_detection_model.pth'))
        print("已加载最佳模型")
    else:
        print("使用最终训练的模型")

    # 全面性能评估
    print("\n进行模型性能评估...")
    performance_report, confusion_mat = evaluate_model_performance(model, test_loader, processor, device)

    # 生成生产级测试报告
    print("\n生成生产级测试报告...")
    if len(test_paths) > 0:
        for i in range(min(5, len(test_paths))):
            test_image = test_paths[i]
            true_label = test_labels[i]

            print(f"\n处理图像 {i + 1}/{min(5, len(test_paths))}: {os.path.basename(test_image)}")
            print(f"真实标签: {true_label}")

            try:
                report, predicted_defect, confidence, needs_review = generate_production_report(
                    model, test_image, processor, device, confidence_threshold=0.8
                )

                print(report)

                # 保存生产级报告
                status = "REVIEW" if needs_review else "AUTO"
                report_filename = f"production_report_{i + 1}_{predicted_defect}_{status}.txt"
                with open(report_filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"生产报告已保存到 {report_filename}")

                # 验证结果
                if predicted_defect == true_label:
                    status_icon = "🟢" if not needs_review else "🟡"
                    print(f"{status_icon} 预测正确! 置信度: {confidence:.3f}")
                else:
                    print(f"🔴 预测错误! 预测: {predicted_defect}, 真实: {true_label}")
            except Exception as e:
                print(f"生成报告时出错: {str(e)}")
                continue

    print("\n🎯 生产就绪系统部署完成！")
    if training_history:
        print("📊 关键指标:")
        print(f"   - 最佳验证准确率: {max([h['val_accuracy'] for h in training_history]):.4f}")
    if performance_report:
        print(f"   - 最终测试准确率: {performance_report['accuracy']:.4f}")
    print("📁 输出文件:")
    print("   - best_defect_detection_model.pth (最佳模型)")
    print("   - production_report_*.txt (生产检测报告)")
    print("🚀 系统已准备好部署到生产环境！")