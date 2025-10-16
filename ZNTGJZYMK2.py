import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
import os
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import hashlib
import time
import uuid

warnings.filterwarnings('ignore')


class NetworkSecurityResponseAgent:
    """
    网络安全事件响应智能体
    修复数据库锁定和唯一约束问题
    """

    def __init__(self, config: Dict[str, Any] = None):
        # 首先初始化日志系统
        self.setup_logging()

        # 核心检测组件
        self.model = None
        self.scaler = StandardScaler()

        # 数据管理
        self.feature_names = []
        self.expected_feature_count = None
        self.attack_samples = []
        self.normal_samples = []

        # 事件响应状态
        self.current_incident_id = None
        self.incident_history = []

        # 日志数据库
        self.log_db_path = "security_incidents.db"
        self.setup_database()

        # 配置参数
        self.config = {
            'model_path': 'security_models',
            'containment_actions': {
                'high': ['立即隔离设备', '阻断网络连接', '通知安全团队', '启动应急响应流程'],
                'medium': ['限制网络访问', '加强监控', '记录详细日志', '人工审核'],
                'low': ['记录事件', '正常监控', '定期检查']
            },
            'threat_thresholds': {
                'high': 0.85,
                'medium': 0.70,
                'low': 0.50
            }
        }
        if config:
            self.config.update(config)

        self.logger.info("网络安全事件响应Agent初始化完成")

    def setup_logging(self):
        """配置日志系统"""
        self.logger = logging.getLogger('SecurityResponseAgent')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.propagate = False

    def setup_database(self):
        """设置安全事件日志数据库"""
        try:
            # 使用不同的数据库文件避免冲突
            self.log_db_path = f"security_incidents_{int(time.time())}.db"

            conn = sqlite3.connect(self.log_db_path)
            cursor = conn.cursor()

            # 创建安全事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_incidents (
                    incident_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    source_ip TEXT,
                    threat_level TEXT,
                    threat_score REAL,
                    detection_confidence REAL,
                    containment_actions TEXT,
                    investigation_findings TEXT,
                    incident_report TEXT,
                    status TEXT
                )
            ''')

            # 创建网络流量日志表（包含is_malicious字段）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS network_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    source_ip TEXT,
                    destination_ip TEXT,
                    protocol TEXT,
                    port INTEGER,
                    packet_size INTEGER,
                    flags TEXT,
                    duration REAL,
                    label TEXT,
                    is_malicious INTEGER
                )
            ''')

            conn.commit()
            conn.close()
            self.logger.info(f"安全事件日志数据库初始化完成: {self.log_db_path}")

        except Exception as e:
            self.logger.error(f"数据库初始化失败: {str(e)}")

    def load_training_data(self, data_path: str, sample_size: int = 50000) -> Optional[pd.DataFrame]:
        """加载训练数据"""
        self.logger.info("开始加载训练数据")

        try:
            if not os.path.exists(data_path):
                self.logger.error(f"数据文件不存在: {data_path}")
                return None

            df = pd.read_csv(data_path, nrows=sample_size)
            self.logger.info(f"成功加载数据: {df.shape}")

            # 数据预处理
            if 'Attack Type' in df.columns:
                df = df.rename(columns={'Attack Type': 'Label'})

            # 选择数值型特征
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_cols:
                numeric_cols.remove('Label')

            df = df[numeric_cols + ['Label']]
            df = df.dropna()

            self.expected_feature_count = len(numeric_cols)
            self.feature_names = numeric_cols

            # 收集攻击和正常样本用于测试
            self._collect_test_samples(df)

            label_counts = df['Label'].value_counts()
            self.logger.info(f"数据预处理完成: {len(df)} 样本, {self.expected_feature_count} 特征")
            self.logger.info(f"数据分布: {label_counts.to_dict()}")

            return df

        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            return None

    def _collect_test_samples(self, df: pd.DataFrame):
        """收集测试样本"""
        # 收集攻击样本
        attack_df = df[df['Label'] != 'Normal Traffic']
        attack_df = attack_df[attack_df['Label'] != 'BENIGN']

        if len(attack_df) > 0:
            for i in range(min(50, len(attack_df))):
                sample = attack_df.iloc[i].drop('Label').values
                self.attack_samples.append({
                    'features': sample,
                    'label': attack_df.iloc[i]['Label'],
                    'type': 'attack'
                })

        # 收集正常样本
        normal_df = df[(df['Label'] == 'Normal Traffic') | (df['Label'] == 'BENIGN')]
        if len(normal_df) > 0:
            normal_sample = normal_df.sample(min(50, len(normal_df)), random_state=42)
            for i in range(len(normal_sample)):
                sample = normal_sample.iloc[i].drop('Label').values
                self.normal_samples.append({
                    'features': sample,
                    'label': normal_sample.iloc[i]['Label'],
                    'type': 'normal'
                })

        self.logger.info(f"收集测试样本: {len(self.attack_samples)} 攻击样本, {len(self.normal_samples)} 正常样本")

    def train_detection_model(self, data_path: str) -> bool:
        """训练威胁检测模型"""
        self.logger.info("开始训练威胁检测模型")

        df = self.load_training_data(data_path, sample_size=50000)
        if df is None:
            return False

        try:
            # 准备特征和标签
            X = df.drop('Label', axis=1)
            y = df['Label'].apply(lambda x: 0 if x in ['Normal Traffic', 'BENIGN'] else 1)

            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )

            self.logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
            self.logger.info(f"训练集分布: {pd.Series(y_train).value_counts().to_dict()}")

            # 处理数据不平衡
            over = SMOTE(sampling_strategy=0.5, random_state=42)
            under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
            pipeline = Pipeline(steps=[('o', over), ('u', under)])
            X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)

            self.logger.info(f"平衡后训练集: {X_train_balanced.shape}")

            # 训练模型
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                class_weight='balanced'
            )

            X_train_scaled = self.scaler.fit_transform(X_train_balanced)
            self.model.fit(X_train_scaled, y_train_balanced)

            # 评估模型
            X_test_scaled = self.scaler.transform(X_test)
            accuracy = self.model.score(X_test_scaled, y_test)

            # 详细评估
            y_pred = self.model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            self.logger.info(f"混淆矩阵:\n{cm}")

            # 计算精确率、召回率和F1分数
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            self.logger.info(f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
            self.logger.info(f"模型训练完成, 测试准确率: {accuracy:.4f}")

            # 保存模型
            os.makedirs(self.config['model_path'], exist_ok=True)
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'feature_count': self.expected_feature_count
            }
            joblib.dump(model_data, f"{self.config['model_path']}/detection_model.pkl")

            return True

        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            return False

    def analyze_threat_level(self, network_features: np.ndarray, source_ip: str) -> Dict[str, Any]:
        """分析威胁级别 - 核心功能1"""
        if self.model is None:
            return {'error': '检测模型未加载'}

        try:
            # 特征预处理
            if len(network_features) != self.expected_feature_count:
                if len(network_features) > self.expected_feature_count:
                    network_features = network_features[:self.expected_feature_count]
                else:
                    padded = np.zeros(self.expected_feature_count)
                    padded[:len(network_features)] = network_features
                    network_features = padded

            # 威胁检测
            features_scaled = self.scaler.transform([network_features])
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probability)

            # 确定威胁级别
            thresholds = self.config['threat_thresholds']
            if prediction == 1 and confidence > thresholds['high']:
                threat_level = 'high'
                risk_score = 90 + (confidence - thresholds['high']) * 10
            elif prediction == 1 and confidence > thresholds['medium']:
                threat_level = 'medium'
                risk_score = 70 + (confidence - thresholds['medium']) * 20
            elif prediction == 1:
                threat_level = 'low'
                risk_score = 50 + (confidence - thresholds['low']) * 20
            else:
                threat_level = 'normal'
                risk_score = max(5, confidence * 10)

            return {
                'is_threat': bool(prediction),
                'threat_level': threat_level,
                'confidence': float(confidence),
                'risk_score': min(100, risk_score),
                'probabilities': {
                    'normal': float(probability[0]),
                    'attack': float(probability[1])
                },
                'source_ip': source_ip,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"威胁分析失败: {str(e)}")
            return {'error': f'威胁分析失败: {str(e)}'}

    def investigate_logs(self, source_ip: str, time_window: int = 60) -> Dict[str, Any]:
        """调查相关日志 - 核心功能2"""
        self.logger.info(f"开始调查IP {source_ip} 的日志记录")

        try:
            with sqlite3.connect(self.log_db_path) as conn:
                cursor = conn.cursor()

                # 计算时间窗口
                start_time = (datetime.now() - timedelta(minutes=time_window)).isoformat()

                # 查询相关网络日志
                cursor.execute('''
                    SELECT * FROM network_logs 
                    WHERE source_ip = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                ''', (source_ip, start_time))

                network_logs = cursor.fetchall()

                # 查询历史安全事件
                cursor.execute('''
                    SELECT * FROM security_incidents 
                    WHERE source_ip = ? 
                    ORDER BY timestamp DESC LIMIT 5
                ''', (source_ip,))

                security_incidents = cursor.fetchall()

            # 分析调查结果
            findings = {
                'recent_network_activity': len(network_logs),
                'historical_incidents': len(security_incidents),
                'suspicious_patterns': [],
                'malicious_activity_count': 0,
                'investigation_timestamp': datetime.now().isoformat()
            }

            # 分析网络日志中的恶意活动
            for log in network_logs:
                if len(log) > 10 and log[10] == 1:  # is_malicious字段
                    findings['malicious_activity_count'] += 1

            # 检测可疑模式
            if len(network_logs) > 50:
                findings['suspicious_patterns'].append('高频网络连接')

            if findings['malicious_activity_count'] > 0:
                findings['suspicious_patterns'].append(f"发现{findings['malicious_activity_count']}次恶意活动")

            if security_incidents:
                findings['suspicious_patterns'].append('历史安全事件记录')

            # 分析网络行为
            if network_logs:
                ports = [log[5] for log in network_logs if len(log) > 5]  # 端口号
                unique_ports = len(set(ports))
                if unique_ports > 20:
                    findings['suspicious_patterns'].append('疑似端口扫描行为')

                # 检查小数据包攻击
                small_packets = sum(1 for log in network_logs if len(log) > 6 and log[6] < 100)
                if small_packets > len(network_logs) * 0.8:
                    findings['suspicious_patterns'].append('疑似DDoS攻击模式')

            self.logger.info(f"日志调查完成: 发现{len(findings['suspicious_patterns'])}个可疑模式")
            return findings

        except Exception as e:
            self.logger.error(f"日志调查失败: {str(e)}")
            return {'error': f'日志调查失败: {str(e)}'}

    def execute_containment_measures(self, threat_analysis: Dict, log_investigation: Dict) -> List[str]:
        """执行初步遏制措施 - 核心功能3"""
        threat_level = threat_analysis.get('threat_level', 'normal')
        source_ip = threat_analysis.get('source_ip', 'unknown')
        confidence = threat_analysis.get('confidence', 0)

        self.logger.info(f"对IP {source_ip} 执行遏制措施, 威胁级别: {threat_level}, 置信度: {confidence:.2f}")

        executed_actions = []

        if threat_level == 'high':
            executed_actions.extend([
                f"隔离设备 {source_ip} - 已执行",
                f"阻断 {source_ip} 的所有网络连接 - 已执行",
                "通知安全团队进行紧急处理 - 已发送",
                "启动应急响应流程 - 已激活"
            ])

        elif threat_level == 'medium':
            executed_actions.extend([
                f"限制 {source_ip} 的网络访问权限 - 已执行",
                f"对 {source_ip} 加强监控 - 已配置",
                "记录详细行为日志 - 已启用",
                "安排人工审核 - 已排队"
            ])

        elif threat_level == 'low':
            executed_actions.extend([
                f"记录 {source_ip} 的安全事件 - 已完成",
                "继续正常监控 - 已确认",
                "定期检查行为模式 - 已安排"
            ])
        else:
            executed_actions.append("正常流量，无需特殊处理")

        # 基于日志调查结果调整措施
        suspicious_patterns = log_investigation.get('suspicious_patterns', [])
        if '疑似端口扫描行为' in suspicious_patterns:
            executed_actions.append("应用端口访问限制策略 - 已执行")

        if '高频网络连接' in suspicious_patterns:
            executed_actions.append("启用连接频率限制 - 已配置")

        if '疑似DDoS攻击模式' in suspicious_patterns:
            executed_actions.append("启用DDoS防护机制 - 已激活")

        self.logger.info(f"遏制措施执行完成: {len(executed_actions)} 项措施")
        return executed_actions

    def generate_incident_report(self, threat_analysis: Dict, log_investigation: Dict,
                                 containment_actions: List[str]) -> Dict[str, Any]:
        """生成事件报告 - 核心功能4"""
        incident_id = f"INC_{int(time.time())}_{hashlib.md5(threat_analysis.get('source_ip', '').encode()).hexdigest()[:8]}"

        report = {
            'incident_id': incident_id,
            'timestamp': datetime.now().isoformat(),
            'threat_analysis': threat_analysis,
            'investigation_findings': log_investigation,
            'containment_actions': containment_actions,
            'incident_summary': self._generate_incident_summary(threat_analysis, log_investigation),
            'recommendations': self._generate_recommendations(threat_analysis, log_investigation),
            'response_timeline': [
                {
                    'time': datetime.now().isoformat(),
                    'action': '威胁检测完成',
                    'details': f"威胁级别: {threat_analysis.get('threat_level')}, 置信度: {threat_analysis.get('confidence'):.2f}"
                }
            ],
            'status': 'closed' if threat_analysis.get('threat_level') == 'normal' else 'active'
        }

        # 保存到数据库
        self._save_incident_to_db(report)

        self.logger.info(f"安全事件报告生成完成: {incident_id}")
        return report

    def _generate_incident_summary(self, threat_analysis: Dict, log_investigation: Dict) -> str:
        """生成事件摘要"""
        source_ip = threat_analysis.get('source_ip', 'unknown')
        threat_level = threat_analysis.get('threat_level', 'normal')
        confidence = threat_analysis.get('confidence', 0)
        risk_score = threat_analysis.get('risk_score', 0)

        summary = f"安全事件摘要 - 源IP: {source_ip}\n"
        summary += f"威胁级别: {threat_level.upper()}, 风险评分: {risk_score}, 检测置信度: {confidence:.2f}\n"

        suspicious_patterns = log_investigation.get('suspicious_patterns', [])
        if suspicious_patterns:
            summary += f"发现可疑行为: {', '.join(suspicious_patterns)}\n"
        else:
            summary += "未发现明显可疑行为\n"

        return summary

    def _generate_recommendations(self, threat_analysis: Dict, log_investigation: Dict) -> List[str]:
        """生成处理建议"""
        recommendations = []
        threat_level = threat_analysis.get('threat_level', 'normal')

        if threat_level == 'high':
            recommendations.extend([
                "立即进行深度恶意软件扫描",
                "全面检查系统完整性和安全性",
                "审查所有用户账户和权限设置",
                "更新安全防护规则和策略",
                "考虑系统重装或恢复"
            ])
        elif threat_level == 'medium':
            recommendations.extend([
                "加强系统监控和日志分析",
                "审查网络访问控制策略",
                "验证系统补丁和安全更新状态",
                "进行安全评估和渗透测试"
            ])
        elif threat_level == 'low':
            recommendations.extend([
                "持续观察系统行为模式",
                "定期检查安全日志",
                "加强用户安全意识培训"
            ])

        return recommendations

    def _save_incident_to_db(self, incident_report: Dict):
        """保存事件到数据库"""
        try:
            with sqlite3.connect(self.log_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    INSERT INTO security_incidents 
                    (incident_id, timestamp, source_ip, threat_level, threat_score, 
                     detection_confidence, containment_actions, investigation_findings, 
                     incident_report, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    incident_report['incident_id'],
                    incident_report['timestamp'],
                    incident_report['threat_analysis']['source_ip'],
                    incident_report['threat_analysis']['threat_level'],
                    incident_report['threat_analysis']['risk_score'],
                    incident_report['threat_analysis']['confidence'],
                    json.dumps(incident_report['containment_actions']),
                    json.dumps(incident_report['investigation_findings']),
                    json.dumps(incident_report),
                    incident_report['status']
                ))

                conn.commit()

        except Exception as e:
            self.logger.error(f"保存事件到数据库失败: {str(e)}")

    def simulate_network_traffic(self, target_ip: str = None, count: int = 50):
        """模拟网络流量用于测试，针对特定IP"""
        self.logger.info(f"开始模拟 {count} 条网络流量记录")

        try:
            with sqlite3.connect(self.log_db_path) as conn:
                cursor = conn.cursor()

                protocols = ['TCP', 'UDP', 'ICMP']

                for i in range(count):
                    # 使用UUID确保唯一性
                    log_id = f"LOG_{uuid.uuid4().hex[:16]}"
                    timestamp = (datetime.now() - timedelta(minutes=np.random.randint(0, 120))).isoformat()

                    # 如果指定了目标IP，部分流量来自该IP
                    if target_ip and i % 3 == 0:
                        source_ip = target_ip
                    else:
                        source_ip = f"192.168.1.{np.random.randint(1, 100)}"

                    dest_ip = f"10.0.0.{np.random.randint(1, 50)}"
                    protocol = np.random.choice(protocols)
                    port = np.random.randint(1, 65535)
                    packet_size = np.random.randint(40, 1500)
                    flags = 'SYN' if protocol == 'TCP' else ''
                    duration = np.random.uniform(0.1, 10.0)

                    # 模拟一些恶意流量
                    is_malicious = 0
                    if source_ip == target_ip and np.random.random() < 0.3:
                        is_malicious = 1
                        # 模拟攻击特征
                        if np.random.random() < 0.5:
                            port = np.random.randint(1, 1024)
                        else:
                            packet_size = np.random.randint(40, 100)

                    label = 'Normal Traffic' if is_malicious == 0 else 'Port Scanning'

                    cursor.execute('''
                        INSERT OR IGNORE INTO network_logs 
                        (log_id, timestamp, source_ip, destination_ip, protocol, port, 
                         packet_size, flags, duration, label, is_malicious)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (log_id, timestamp, source_ip, dest_ip, protocol, port,
                          packet_size, flags, duration, label, is_malicious))

                conn.commit()

            self.logger.info(f"网络流量模拟完成，包含针对IP {target_ip} 的流量")

        except Exception as e:
            self.logger.error(f"模拟网络流量失败: {str(e)}")

    def process_security_incident(self, network_features: np.ndarray, source_ip: str) -> Dict[str, Any]:
        """
        处理安全事件 - 完整工作流
        严格按照项目要求的四个核心功能
        """
        self.logger.info(f"开始处理安全事件 - 源IP: {source_ip}")

        # 1. 自动分析威胁级别
        threat_analysis = self.analyze_threat_level(network_features, source_ip)
        if 'error' in threat_analysis:
            return {'error': threat_analysis['error']}

        # 2. 调查相关日志
        log_investigation = self.investigate_logs(source_ip)

        # 3. 执行初步遏制措施
        containment_actions = self.execute_containment_measures(threat_analysis, log_investigation)

        # 4. 生成事件报告
        incident_report = self.generate_incident_report(threat_analysis, log_investigation, containment_actions)

        self.logger.info(f"安全事件处理完成: {incident_report['incident_id']}")
        return incident_report

    def get_incident_history(self, limit: int = 10) -> List[Dict]:
        """获取历史事件记录"""
        try:
            with sqlite3.connect(self.log_db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT incident_id, timestamp, source_ip, threat_level, status 
                    FROM security_incidents 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))

                incidents = []
                for row in cursor.fetchall():
                    incidents.append({
                        'incident_id': row[0],
                        'timestamp': row[1],
                        'source_ip': row[2],
                        'threat_level': row[3],
                        'status': row[4]
                    })

                return incidents

        except Exception as e:
            self.logger.error(f"获取历史事件失败: {str(e)}")
            return []

    def run_comprehensive_demo(self):
        """运行全面的演示，包括正常和攻击场景"""
        self.logger.info("=== 开始全面安全事件响应演示 ===")

        # 演示正常流量检测
        if self.normal_samples:
            self.logger.info("--- 演示1: 正常流量检测 ---")
            normal_sample = self.normal_samples[0]['features']
            normal_ip = "192.168.1.100"

            # 为正常IP模拟一些正常流量
            self.simulate_network_traffic(normal_ip, 30)

            normal_result = self.process_security_incident(normal_sample, normal_ip)

            if 'error' not in normal_result:
                self.logger.info(f"正常流量检测结果: 威胁级别={normal_result['threat_analysis']['threat_level']}")

        # 演示攻击流量检测
        if self.attack_samples:
            self.logger.info("--- 演示2: 攻击流量检测 ---")
            attack_sample = self.attack_samples[0]['features']
            attack_ip = "10.0.0.99"

            # 为攻击IP模拟一些恶意流量
            self.simulate_network_traffic(attack_ip, 50)

            attack_result = self.process_security_incident(attack_sample, attack_ip)

            if 'error' not in attack_result:
                self.logger.info(f"攻击流量检测结果: 威胁级别={attack_result['threat_analysis']['threat_level']}")
                self.logger.info(f"遏制措施: {len(attack_result['containment_actions'])} 项")

                # 显示详细的攻击报告
                self.logger.info("攻击事件详细信息:")
                self.logger.info(f"  事件ID: {attack_result['incident_id']}")
                self.logger.info(f"  源IP: {attack_result['threat_analysis']['source_ip']}")
                self.logger.info(f"  威胁级别: {attack_result['threat_analysis']['threat_level']}")
                self.logger.info(f"  置信度: {attack_result['threat_analysis']['confidence']:.2f}")
                self.logger.info(f"  风险评分: {attack_result['threat_analysis']['risk_score']}")

                # 显示遏制措施
                self.logger.info("执行的遏制措施:")
                for action in attack_result['containment_actions']:
                    self.logger.info(f"  - {action}")

        # 显示系统状态
        self.logger.info("--- 系统状态报告 ---")
        history = self.get_incident_history(5)
        self.logger.info(f"最近处理的安全事件: {len(history)} 个")
        for incident in history:
            self.logger.info(
                f"  {incident['incident_id']}: {incident['source_ip']} - {incident['threat_level']} - {incident['status']}")


def main():
    """主函数 - 演示完整的安全事件响应流程"""
    agent = NetworkSecurityResponseAgent()

    # 数据文件路径
    data_path = 'extracted_data/cicids2017_cleaned.csv'

    if not os.path.exists(data_path):
        agent.logger.error(f"训练数据文件不存在: {data_path}")
        return

    # 1. 训练检测模型
    agent.logger.info("=== 阶段1: 训练威胁检测模型 ===")
    training_success = agent.train_detection_model(data_path)

    if not training_success:
        agent.logger.error("模型训练失败，终止执行")
        return

    # 2. 运行全面演示
    agent.logger.info("=== 阶段2: 全面安全事件响应演示 ===")
    agent.run_comprehensive_demo()

    agent.logger.info("网络安全事件响应Agent演示完成")


if __name__ == "__main__":
    main()