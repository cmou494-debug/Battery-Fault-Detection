import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging
import warnings
import scipy.stats as stats
from antropy import sample_entropy
# --- 配置与设置 ---
warnings.filterwarnings('ignore', category=UserWarning)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 请将多个数据集的路径放入此字典。
FEATURE_SELECTION_DATASETS = {
    "L42_4.10_vol": r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_4.10_vol.xlsx",
    "L42_4.9_vol": r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_4.9_vol.xlsx",
    "L42_4.6_vol": r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_4.6_vol.xlsx",
    "L42_4.5_vol": r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_4.5_vol.xlsx",
    "L42_01.11_vol": r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_01.11_vol.xlsx",
    "L42_11.28_vol": r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_11.28_vol.xlsx",
    "L42_12.26_vol": r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_12.26_vol.xlsx",
    "L42_2_vol":r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_2_vol.xlsx",
    "L42_4.11_vol":r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_4.11_vol.xlsx",
    "6.21_vol":r"F:\电池故障诊断\文献复现\自创电池故障诊断\LB04各项数据文件\6.21_vol.xlsx",
    "7.1_vol":r"F:\电池故障诊断\文献复现\自创电池故障诊断\LB04各项数据文件\7.1_vol.xlsx",
    "L04_7.4":r"F:\电池故障诊断\文献复现\自创电池故障诊断\LB04各项数据文件\L04_7.4.xlsx"
}

# 2. 故障诊断数据集路径 (实际进行诊断分析的数据集)
DIAGNOSIS_DATASET_PATH = r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42_12.26_vol.xlsx"
TOP_N_FEATURES_TO_SELECT = 28
# 其他诊断参数
SMOOTHING_WINDOW_SIZE = 3
WINDOW_SIZE = 100
STRIDE = 5
IFOREST_N_ESTIMATORS = 150
IFOREST_CONTAMINATION = 'auto'
Z_SCORE_NORMAL_THRESHOLD = 4.5
Z_SCORE_FAULT_THRESHOLD = 6
Z_SCORE_SUSPECT_THRESHOLD = 4.5
MAX_K_CLUSTERS = 5
# --- 1. 数据加载与预处理 ---

def load_voltage_data(file_path):
    """加载原始电压数据，并确保列名正确。"""
    try:
        df = pd.read_excel(file_path)
        voltage_start_col_idx = 2
        time_data = df.iloc[:, 0]
        if df.shape[1] > voltage_start_col_idx:
            battery_data = df.iloc[:, voltage_start_col_idx:]
            battery_names = [f"单体{i + 1}" for i in range(battery_data.shape[1])]
            battery_data.columns = battery_names
        else:
            battery_data = df.iloc[:, 1:]
            battery_names = [f"单体{i + 1}" for i in range(battery_data.shape[1])]
            battery_data.columns = battery_names
        full_df = pd.concat([time_data, battery_data], axis=1)
        logging.info(f"成功加载电池电压数据，形状: {full_df.shape}")
        return full_df, battery_names
    except Exception as e:
        logging.error(f"加载文件出错: {e}")
        return None, None


def preprocess_data(df, smoothing_window):
    """数据预处理：去重、插值、滤波平滑。"""
    # logging.info("开始进行数据预处理...")
    initial_rows = len(df)
    df.drop_duplicates(subset=[df.columns[0]], keep='first', inplace=True)
    rows_after_dedup = len(df)
    if initial_rows > rows_after_dedup:
        logging.info(f"成功删除 {initial_rows - rows_after_dedup} 条重复数据。")
    # logging.info("开始进行线性插值填充缺失值...")
    df.iloc[:, 1:].interpolate(method='linear', axis=0, inplace=True)
    # logging.info(f"开始进行移动平均平滑，窗口大小: {smoothing_window}...")
    df.iloc[:, 1:] = df.iloc[:, 1:].rolling(window=smoothing_window, min_periods=1).mean()
    # logging.info("数据预处理完成。")
    return df


# --- 2. 增强特征提取 (28个特征) ---
def calculate_entropy(data):
    """计算电压信号的样本熵。"""
    if len(data) < 2:
        return 0
    return sample_entropy(data)


def extract_enhanced_features(window_data, window_global_mean):
    """提取单个电池滑动窗口的增强特征 (20个局部特征)。"""
    features = []
    features.append(np.mean(window_data))
    features.append(np.std(window_data))
    features.append(np.min(window_data))
    features.append(np.max(window_data))
    features.append(np.median(window_data))
    features.append(calculate_entropy(window_data))
    if len(window_data) > 1:
        x = np.arange(len(window_data))
        slope, _ = np.polyfit(x, window_data, 1)
        features.append(slope)
        if np.std(window_data[:-1]) > 1e-9 and np.std(window_data[1:]) > 1e-9:
            autocorr = np.corrcoef(window_data[:-1], window_data[1:])[0, 1]
        else:
            autocorr = 1.0 if np.allclose(window_data[:-1], window_data[1:]) else 0.0
        features.append(autocorr)
    else:
        features.extend([0.0] * 2)
    if len(window_data) > 2:
        diff1 = np.diff(window_data)
        diff2 = np.diff(diff1)
        features.append(np.mean(diff1))
        features.append(np.std(diff2))
    else:
        features.extend([0.0] * 2)
    if len(window_data) > 1 and len(window_data) == len(window_global_mean):
        voltage_residuals = window_data - window_global_mean
        cumulative_residual = np.sum(voltage_residuals)
        features.append(cumulative_residual)
    else:
        features.append(0.0)
    if len(window_data) > 10:
        detrended = window_data - np.mean(window_data)
        fft_vals = np.abs(np.fft.rfft(detrended))
        if len(fft_vals) > 5:
            for i in range(5):
                features.append(fft_vals[i])
            features.append(np.sum(fft_vals[5:]) / len(fft_vals[5:]) if len(fft_vals[5:]) > 0 else 0.0)
        else:
            features.extend([0.0] * 6)
    else:
        features.extend([0.0] * 6)
    if len(window_data) > 2:
        rms = np.sqrt(np.mean(np.square(window_data)))
        abs_mean = np.mean(np.abs(window_data))
        waveform_factor = rms / (abs_mean + 1e-10)
        features.append(waveform_factor)
        peak_to_peak = np.max(window_data) - np.min(window_data)
        crest_factor = peak_to_peak / (rms + 1e-10)
        features.append(crest_factor)
        diff1 = np.diff(window_data)
        if len(diff1) > 1:
            diff2 = np.diff(diff1)
            if np.std(diff2) > 1e-10:
                complexity = np.log(np.std(diff1) / np.std(diff2))
                features.append(complexity)
            else:
                features.append(0.0)
        else:
            features.extend([0.0] * 3)
    return np.array(features)


def extract_global_window_features(window_matrix):
    """提取整个窗口的全局特征 (8个全局特征)。"""
    features = []
    cell_means = np.nanmean(window_matrix, axis=0)
    global_mean = np.nanmean(cell_means)
    deviations = np.abs(cell_means - global_mean)
    features.append(np.max(deviations))
    features.append(np.nanmean(deviations))
    features.append(np.nanstd(deviations))
    if np.nanstd(deviations) > 1e-10:
        outlier_ratio = np.nanmean(deviations > 3 * np.nanstd(deviations))
        features.append(outlier_ratio)
    else:
        features.append(0.0)
    if window_matrix.shape[1] > 1:
        with np.errstate(invalid='ignore'):
            corr_matrix = np.corrcoef(window_matrix.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        upper_triangle = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
        features.append(np.mean(upper_triangle))
        features.append(np.min(upper_triangle))
        features.append(np.std(upper_triangle))
        if len(upper_triangle) > 1 and np.std(upper_triangle) > 1e-10:
            low_threshold = np.mean(upper_triangle) - 2 * np.std(upper_triangle)
            abnormal_corr_ratio = np.mean(upper_triangle < low_threshold)
            features.append(abnormal_corr_ratio)
        else:
            features.extend([0.0] * 4)
    else:
        features.extend([0.0] * 4)
    return np.array(features)


def get_feature_names():
    """生成所有特征的名称列表 (20个局部 + 8个全局 = 28个特征)。"""
    local_features = [
        'mean_voltage', 'std_voltage', 'min_voltage', 'max_voltage', 'median_voltage', 'sample_entropy',
        'slope', 'autocorrelation', 'mean_diff1', 'std_diff2', 'cumulative_residual',
        'fft_freq1', 'fft_freq2', 'fft_freq3', 'fft_freq4', 'fft_freq5', 'fft_high_freq_sum',
        'waveform_factor', 'crest_factor', 'complexity'
    ]
    global_features = [
        'max_deviation', 'mean_deviation', 'std_deviation', 'outlier_ratio',
        'mean_correlation', 'min_correlation', 'std_correlation', 'abnormal_corr_ratio'
    ]
    return local_features + global_features


def prepare_window_features(voltage_matrix, raw_df, window_size, stride):
    """提取所有滑动窗口的特征，并返回窗口起始时间点/索引。"""
    all_window_features = []
    cell_window_map = []
    window_start_indices = []
    window_start_times = []
    n_cells = voltage_matrix.shape[1]
    n_samples = voltage_matrix.shape[0]

    time_series_index = pd.to_datetime(raw_df.iloc[:, 0])

    # logging.info(f"开始生成滑动窗口并提取特征。总电池数: {n_cells}, 总采样点数: {n_samples}")
    for start in range(0, n_samples - window_size + 1, stride):
        window_data_matrix = voltage_matrix[start:start + window_size, :]

        # 记录窗口起始索引和时间
        window_start_indices.append(start)
        if start < len(time_series_index):
            window_start_times.append(time_series_index.iloc[start])
        else:
            window_start_times.append(f"Index_{start}")

        window_global_mean_vector = np.nanmean(window_data_matrix, axis=1)

        global_features = extract_global_window_features(window_data_matrix)
        for cell_idx in range(n_cells):
            cell_window_data = window_data_matrix[:, cell_idx]
            cell_features = extract_enhanced_features(cell_window_data, window_global_mean_vector)
            combined_features = np.concatenate([cell_features, global_features])
            all_window_features.append(combined_features)
            cell_window_map.append(cell_idx)

    if not all_window_features:
        # logging.warning("没有生成有效的窗口特征。")
        return np.array([]), np.array([]), np.array([]), np.array([])

    features_array = np.array(all_window_features)
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    # logging.info(f"成功提取 {len(all_window_features)} 个窗口特征，特征维度: {features_array.shape[1]}")
    return features_array, np.array(cell_window_map), np.array(window_start_times), np.array(window_start_indices)


# *****************************************************************
# ************** 特征筛选聚合函数 (替换原有 select_features_with_random_forest) **************
# *****************************************************************

def calculate_rf_importance(features, labels):
    """计算单个数据集的随机森林特征重要性。"""
    all_feature_names = get_feature_names()
    if features.shape[1] == 0:
        return pd.Series(0.0, index=all_feature_names)

    # Isolation Forest 标签：-1 (异常), 1 (正常) -> 分类器标签：1 (异常), 0 (正常)
    y = np.where(labels == 1, 0, 1)

    rf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf_model.fit(features, y)

    feature_importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)
    # 标准化到 [0, 1]
    importances_normalized = feature_importances / (feature_importances.sum() + 1e-9)
    return importances_normalized


def detect_anomalies_with_isolation_forest_for_rf(features, n_estimators, contamination):
    """
    用于特征重要性计算的预诊断。
    """
    if features.size == 0 or features.shape[1] == 0:
        return None, None

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                            random_state=42, n_jobs=-1)
    model.fit(scaled_features)
    anomaly_labels = model.predict(scaled_features)  # -1: 异常, 1: 正常
    return None, anomaly_labels


def perform_feature_aggregation_and_selection(dataset_config, n_features_to_select):
    """
    遍历多个数据集，聚合特征重要性，并筛选出最终的 Top N 特征。
    """
    logging.info("\n" + "=" * 80)
    logging.info("--- 步骤1: 跨数据集稳健特征筛选开始 ---")

    all_importances_series = []

    if not dataset_config:
        logging.warning("未配置特征筛选数据集，将使用诊断数据集自身RF结果进行特征筛选。")
        # 兼容逻辑：如果没有配置数据集，就返回全部特征，让主程序继续进行单数据集RF
        return get_feature_names(), None

    for name, path in dataset_config.items():
        logging.info(f"正在处理数据集: {name} ({path})")

        raw_df, _ = load_voltage_data(path)
        if raw_df is None: continue

        processed_df = preprocess_data(raw_df.copy(), SMOOTHING_WINDOW_SIZE)
        voltage_matrix = processed_df.iloc[:, 1:].values.astype(float)

        # 提取特征
        all_window_features, _, _, _ = prepare_window_features(
            voltage_matrix, raw_df, WINDOW_SIZE, STRIDE)

        if all_window_features.size == 0:
            logging.warning(f"数据集 {name} 未生成有效特征，跳过。")
            continue

        # 预诊断获取标签
        _, initial_anomaly_labels = detect_anomalies_with_isolation_forest_for_rf(
            all_window_features, IFOREST_N_ESTIMATORS, IFOREST_CONTAMINATION)

        if initial_anomaly_labels is None:
            logging.warning(f"数据集 {name} 预诊断失败，跳过。")
            continue

        # 计算随机森林重要性
        importances = calculate_rf_importance(all_window_features, initial_anomaly_labels)
        all_importances_series.append(importances)

    if not all_importances_series:
        logging.error("所有数据集均未能成功提取特征或计算重要性，将使用全部 28 个特征。")
        return get_feature_names(), None

    # 聚合：计算所有数据集的重要性平均值
    aggregated_importance = pd.concat(all_importances_series, axis=1).mean(axis=1)

    # 转换为 DataFrame 并排序
    importance_df = pd.DataFrame({
        'Feature_Name': aggregated_importance.index,
        'Aggregated_Score': aggregated_importance.values
    }).sort_values(by='Aggregated_Score', ascending=False).reset_index(drop=True)

    importance_df['Rank'] = importance_df.index + 1

    # 筛选最终的特征集
    n_to_select = min(n_features_to_select, len(get_feature_names()))
    final_selected_names = importance_df.head(n_to_select)['Feature_Name'].tolist()

    # 打印完整的特征重要性表格
    print("\n" + "=" * 80)
    print(f"跨数据集聚合随机森林重要性得分与排名 (Top {n_to_select} 特征用于诊断):")
    print(importance_df.to_string())
    print("=" * 80 + "\n")

    logging.info(f"--- 步骤1: 稳健特征筛选完成，共选定 {len(final_selected_names)} 个特征。---")
    logging.info("=" * 80 + "\n")

    return final_selected_names, importance_df


# *****************************************************************
# *****************************************************************
# *****************************************************************

# 3. 诊断与异常分数聚合

def detect_anomalies_with_isolation_forest(features, n_estimators, contamination):
    """使用孤立森林进行异常检测。"""
    if features.size == 0 or features.shape[1] == 0:
        logging.error("无特征数据可用于检测。")
        return None, None
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    logging.info(f"特征标准化完成，形状: {scaled_features.shape}")
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                            random_state=42, n_jobs=-1)
    model.fit(scaled_features)
    logging.info("第一层诊断：孤立森林模型训练完成。")
    anomaly_scores = model.decision_function(scaled_features)
    anomaly_labels = model.predict(scaled_features)  # -1: 异常, 1: 正常

    min_score = np.min(anomaly_scores)
    positive_anomaly_scores = anomaly_scores - min_score

    return positive_anomaly_scores, anomaly_labels


def calculate_four_agg_features(window_anomaly_scores, anomaly_labels, cell_window_indices, n_cells):
    """
    【增强】：计算四个聚合特征：最大异常分数、异常窗口数量、聚合Z-score和Z-Score偏度。
    """
    logging.info("开始计算单体级 4 个聚合特征 (Max Score, Count, Max Abs Z-Score, Skewness)...")
    cell_aggregated_scores = np.zeros(n_cells)
    cell_anomaly_counts = np.zeros(n_cells)
    cell_aggregated_z_scores = np.zeros(n_cells)
    cell_z_score_skewness = np.zeros(n_cells)

    if len(window_anomaly_scores) < 2:
        window_z_scores = np.zeros_like(window_anomaly_scores)
    else:
        total_count = len(window_anomaly_scores)
        sorted_scores = np.sort(window_anomaly_scores)
        # 裁剪掉两端2.5%的数据，以提高鲁棒性
        trim_low_idx = int(total_count * 0.025)
        trim_high_idx = int(total_count * 0.975)
        trimmed_scores = sorted_scores[trim_low_idx: trim_high_idx]

        median_trimmed = np.median(trimmed_scores)
        std_trimmed = np.std(trimmed_scores)
        if std_trimmed > 1e-9:
            window_z_scores = (window_anomaly_scores - median_trimmed) / std_trimmed
        else:
            window_z_scores = np.zeros_like(window_anomaly_scores)

    for cell_idx in range(n_cells):
        mask = cell_window_indices == cell_idx
        if np.any(mask):
            scores = window_anomaly_scores[mask]
            cell_aggregated_scores[cell_idx] = np.max(scores)
            labels = anomaly_labels[mask]
            cell_anomaly_counts[cell_idx] = np.sum(labels == -1)
            z_scores_for_cell = window_z_scores[mask]

            cell_aggregated_z_scores[cell_idx] = np.max(np.abs(z_scores_for_cell))

            cell_z_score_skewness[cell_idx] = stats.skew(z_scores_for_cell) if len(z_scores_for_cell) >= 3 else 0.0
        else:
            cell_aggregated_scores[cell_idx], cell_anomaly_counts[cell_idx], cell_aggregated_z_scores[cell_idx], \
                cell_z_score_skewness[cell_idx] = 0, 0, 0, 0

    four_features = np.vstack(
        (cell_aggregated_scores, cell_anomaly_counts, cell_aggregated_z_scores, cell_z_score_skewness)).T

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(four_features)
    return four_features, scaled_features


def find_optimal_k_by_elbow_method(data, max_k=MAX_K_CLUSTERS):
    """使用肘部法则自动确定最优的K-Means聚类数量。"""
    if len(data) < 2:
        return 1, [0]
    wcss = []
    max_k = min(max_k, len(data) - 1)
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    if len(wcss) < 3:
        return 1, wcss
    diffs = np.diff(wcss)
    diffs_ratio = np.diff(diffs) / (-diffs[:-1] + 1e-9)
    optimal_k = np.argmax(diffs_ratio) + 2
    return min(optimal_k, 3), wcss


def diagnose_fault_type(faulty_indices, cell_window_indices, all_window_features, window_anomaly_labels, feature_names):
    """根据特征数据对故障电池进行更详细的类型诊断。"""
    logging.info("开始对故障和疑似故障单体进行更详细的故障类型诊断...")
    fault_diagnoses = {}
    anomaly_window_indices = np.where(window_anomaly_labels == -1)[0]
    for cell_idx in faulty_indices:
        diagnosis_results = []
        cell_anomaly_windows = anomaly_window_indices[cell_window_indices[anomaly_window_indices] == cell_idx]
        if len(cell_anomaly_windows) == 0:
            fault_diagnoses[cell_idx] = '无明显故障特征'
            continue
        cell_features = all_window_features[cell_anomaly_windows, :]

        # 尝试从全部特征中找到所需的索引
        try:
            std_voltage_idx = feature_names.index('std_voltage')
            sample_entropy_idx = feature_names.index('sample_entropy')
            min_correlation_idx = feature_names.index('min_correlation')
        except ValueError:
            logging.error("无法找到所需的特征索引，跳过故障类型诊断。")
            fault_diagnoses[cell_idx] = '特征缺失，无法诊断'
            continue

        avg_std_voltage = np.mean(cell_features[:, std_voltage_idx])
        avg_sample_entropy = np.mean(cell_features[:, sample_entropy_idx])
        min_correlation = np.min(cell_features[:, min_correlation_idx])

        if avg_std_voltage > 0.05 and min_correlation < 0.5:
            diagnosis_results.append('内部短路/高波动异常')
        if avg_std_voltage < 0.05 and min_correlation < 0.8:
            diagnosis_results.append('电池不一致性加剧')
        if avg_sample_entropy > 0.5:
            diagnosis_results.append('非线性动态异常')
        if not diagnosis_results:
            fault_diagnoses[cell_idx] = '其他未知故障'
        else:
            fault_diagnoses[cell_idx] = ', '.join(list(set(diagnosis_results)))
    return fault_diagnoses


# --- 4. 可视化函数 (保持不变) ---

def plot_all_raw_voltage_curves(raw_df, cell_names):
    """可视化所有单体的原始电压曲线。"""
    plt.figure(figsize=(8, 6))
    voltage_data = raw_df.iloc[:, 1:]
    for i in range(voltage_data.shape[1]):
        plt.plot(voltage_data.iloc[:, i], alpha=0.7)
    plt.title('所有单体原始电压曲线', fontsize=16)
    plt.xlabel('时间点', fontsize=16)
    plt.ylabel('电压 (V)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.show()
    logging.info("已显示所有单体的原始电压曲线图")


def plot_all_processed_voltage_curves(processed_voltage_df, cell_names, anomaly_window_info):
    """
    【增强】: 可视化平滑后的电压曲线，并在图上标记孤立森林检测到的异常窗口起始点。
    anomaly_window_info: 包含 (cell_index, start_index_in_df) 的列表
    """
    plt.figure(figsize=(10, 7))
    voltage_data = processed_voltage_df

    # 绘制所有电池曲线
    for i in range(voltage_data.shape[1]):
        plt.plot(voltage_data.iloc[:, i], alpha=0.7, label=f'单体 {i + 1}')

    # 绘制异常点
    anomaly_x = []
    anomaly_y = []

    for cell_idx, start_index in anomaly_window_info:
        # start_index 是在 processed_voltage_df 中的行索引
        if cell_idx < voltage_data.shape[1] and start_index < voltage_data.shape[0]:
            anomaly_x.append(start_index)
            # 获取该电池在该起始点处的电压值
            anomaly_y.append(voltage_data.iloc[start_index, cell_idx])

    if anomaly_x:
        plt.scatter(anomaly_x, anomaly_y,
                    color='red',
                    marker='o',
                    s=50,
                    zorder=3,
                    label='IF异常窗口起始点 (第一层诊断)')
        logging.warning(f"图中已标记 {len(anomaly_x)} 个孤立森林检测到的异常窗口起始点。")

    plt.title('所有单体平滑后电压曲线及第一层诊断结果', fontsize=16)
    plt.xlabel('采样点索引', fontsize=16)
    plt.ylabel('电压 (V)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)

    # 添加图例，只在有异常点时显示
    if anomaly_x:
        plt.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()
    logging.info("已显示所有单体平滑后的电压曲线图 (包含第一层诊断标记)")


def plot_elbow_method(wcss):
    """可视化肘部法则。"""
    plt.figure(figsize=(8, 6))
    k_range = range(1, len(wcss) + 1)
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('肘部法则', fontsize=14)
    plt.xlabel('聚类数量 (K)', fontsize=12)
    plt.ylabel('簇内平方和 (WCSS)', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    logging.info("已显示肘部法则图。")


def plot_clusters_3d(scaled_features, clusters, kmeans_model, cell_names):
    """可视化 K-Means 3D聚类结果 (使用前 3 个聚合特征)。"""
    if scaled_features.shape[1] < 3:
        logging.warning("聚合特征少于3个，跳过3D聚类绘图。")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    feature_labels = ['最大分数', '异常窗口数', 'Max Abs Z-Score']
    ax.scatter(scaled_features[:, 0],
               scaled_features[:, 1],
               scaled_features[:, 2],
               c=clusters,
               cmap='viridis',
               s=50,
               alpha=0.7)
    centers = kmeans_model.cluster_centers_
    ax.scatter(centers[:, 0],
               centers[:, 1],
               centers[:, 2],
               marker='X',
               s=200,
               c='red')

    # 使用Z-Score来确定哪个簇是正常的
    sorted_center_indices = np.argsort(kmeans_model.cluster_centers_[:, 2])
    normal_cluster = sorted_center_indices[0]  # 最小Z-Score的簇视为正常

    for i, txt in enumerate(cell_names):
        if clusters[i] != normal_cluster:  # 标注非'正常'的单体
            ax.text(scaled_features[i, 0], scaled_features[i, 1], scaled_features[i, 2],
                    txt, size=8, zorder=1, color='k')
    ax.set_title('K-Means三维聚类结果 (基于4个聚合特征)', fontsize=14)
    ax.set_xlabel(feature_labels[0], fontsize=12)
    ax.set_ylabel(feature_labels[1], fontsize=12)
    ax.set_zlabel(feature_labels[2], fontsize=12)
    plt.show()
    logging.info("已显示三维聚类结果图。")


def plot_voltage_curves_by_category(category_indices, category_name, cell_names, raw_voltage_data_df):
    """绘制特定类别的电池电压曲线。"""
    if not category_indices:
        logging.info(f"没有 {category_name} 电池单体需要绘制")
        return
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(category_indices)))

    for i, cell_idx in enumerate(category_indices):
        plt.plot(raw_voltage_data_df.iloc[:, cell_idx].values,
                 color=colors[i],
                 alpha=0.8,
                 label=cell_names[cell_idx])
    plt.title(f'{category_name}电池电压曲线', fontsize=14)
    plt.xlabel('时间点', fontsize=12)
    plt.ylabel('电压值', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    logging.info(f"已显示 {len(category_indices)} 个{category_name}电池的电压曲线")


def plot_z_scores(z_scores, categories, cell_names):
    """可视化每个单体的聚合Z-score系数，用点图进行区分。"""
    plt.figure(figsize=(12, 6))
    x_indices = np.arange(len(z_scores))
    category_map = {'正常': 'green', '疑似故障': 'orange', '故障': 'red'}
    for category, color in category_map.items():
        indices = [i for i, cat in enumerate(categories) if cat == category]
        if indices:
            plt.scatter(x_indices[indices], z_scores[indices], c=color, s=50, label=category)
    plt.axhline(y=Z_SCORE_SUSPECT_THRESHOLD, color='orange', linestyle='--',
                label=f'疑似阈值({Z_SCORE_SUSPECT_THRESHOLD})')
    plt.axhline(y=Z_SCORE_FAULT_THRESHOLD, color='red', linestyle='--', label=f'故障阈值({Z_SCORE_FAULT_THRESHOLD})')
    plt.title('各单体聚合Z-Score点图 (第二层诊断指标)', fontsize=14)
    plt.xlabel('电池单体编号', fontsize=12)
    plt.ylabel('Max Abs Z-Score', fontsize=12)
    plt.xticks(x_indices, cell_names, rotation=90, fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    logging.info("已显示各单体的聚合Z-score点图。")


# 主程序
if __name__ == "__main__":
    logging.info("诊断开始")

    # 1. 数据加载与预处理
    raw_df, cell_names = load_voltage_data(DIAGNOSIS_DATASET_PATH)
    if raw_df is None: exit()

    plot_all_raw_voltage_curves(raw_df, cell_names)

    processed_df = preprocess_data(raw_df.copy(), SMOOTHING_WINDOW_SIZE)
    voltage_matrix = processed_df.iloc[:, 1:].values.astype(float)
    processed_voltage_df = processed_df.iloc[:, 1:]

    # 2. 增强特征提取 (28个特征)
    all_window_features, cell_window_indices, window_start_times, window_start_indices = prepare_window_features(
        voltage_matrix, raw_df, window_size=WINDOW_SIZE, stride=STRIDE
    )
    if all_window_features.size == 0: exit()

    # 3. ******** 特征选择逻辑替换为：跨数据集聚合筛选 ********
    # 使用配置的多个数据集（如果配置了）来计算最稳健的特征集
    selected_feature_names, importance_df = perform_feature_aggregation_and_selection(
        FEATURE_SELECTION_DATASETS, TOP_N_FEATURES_TO_SELECT)

    # 4. 根据筛选结果，选取特征子集用于诊断
    all_names = get_feature_names()
    selected_mask = np.array([name in selected_feature_names for name in all_names])
    filtered_features = all_window_features[:, selected_mask]

    logging.info(f"特征集准备完成。诊断将使用 {len(selected_feature_names)} 个特征：{selected_feature_names}")

    # 5. 第一层：使用筛选后的特征进行最终的孤立森林异常检测 (打分)
    window_anomaly_scores, anomaly_labels = \
        detect_anomalies_with_isolation_forest(
            filtered_features,
            n_estimators=IFOREST_N_ESTIMATORS,
            contamination=IFOREST_CONTAMINATION
        )
    if window_anomaly_scores is None: exit()

    # 收集异常窗口的位置信息 (用于绘图)
    anomaly_window_indices = np.where(anomaly_labels == -1)[0]
    anomaly_window_info = []

    for window_global_idx in anomaly_window_indices:
        cell_idx = cell_window_indices[window_global_idx]
        window_time_idx = window_global_idx // len(cell_names)
        start_index_in_df = window_start_indices[window_time_idx]
        anomaly_window_info.append((cell_idx, start_index_in_df))

    plot_all_processed_voltage_curves(processed_voltage_df, cell_names, anomaly_window_info)

    # 6. 聚合特征计算 (4个特征)
    aggregated_features, scaled_aggregated_features = calculate_four_agg_features(
        window_anomaly_scores, anomaly_labels, cell_window_indices, len(cell_names)
    )
    logging.info("已计算每个电池的 4 个增强聚合特征。")

    # 7. 【第一层诊断结果输出】：控制台输出
    abnormal_window_mask = (anomaly_labels == -1)
    if np.sum(abnormal_window_mask) > 0:
        logging.warning("\n### 第一层诊断结果：突发性/窗口级异常（已定位时间） ###")
        print("-" * 60)
        print(f"{'电池单体':<10} | {'窗口序号':<10} | {'起始时间点':<25} | {'IF分数':<10}")
        print("-" * 60)

        anomaly_counter = 0
        for i, (score, label, idx, start_time) in enumerate(
                zip(window_anomaly_scores, anomaly_labels, cell_window_indices, window_start_times)):
            if label == -1 and anomaly_counter < 10:
                cell_name = cell_names[idx]
                window_time_idx = i // len(cell_names)
                print(f"{cell_name:<10} | {window_time_idx:<10} | {str(start_time):<25} | {score:<10.3f}")
                anomaly_counter += 1

        print("-" * 60)
    else:
        logging.info("第一层诊断：未检测到明显的窗口级突发性异常。")

    # 8. 【第二层诊断】：K-Means 自适应聚类分级 (识别渐进性故障)

    final_categories = ['' for _ in range(len(cell_names))]

    aggregated_z_scores = aggregated_features[:, 2]
    max_aggregated_z_score = np.max(aggregated_z_scores)

    if max_aggregated_z_score < Z_SCORE_NORMAL_THRESHOLD:
        logging.info(f"\n### 第二层诊断结果：单体级 (渐进性) 异常 ###")
        logging.info(f"最高聚合Z-Score ({max_aggregated_z_score:.2f}) 小于阈值，所有电池判定为'正常'。")
        for i in range(len(cell_names)):
            final_categories[i] = '正常'
    else:
        logging.info(f"\n### 第二层诊断结果：单体级 (渐进性) 异常 ###")
        logging.info(f"最高聚合Z-Score ({max_aggregated_z_score:.2f}) 超过阈值，开始寻找最优聚类数量。")

        optimal_k, wcss = find_optimal_k_by_elbow_method(scaled_aggregated_features)
        logging.info(f"最优聚类数量为 k={optimal_k}。")
        plot_elbow_method(wcss)

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_aggregated_features)
        plot_clusters_3d(scaled_aggregated_features, clusters, kmeans, cell_names)

        # ***** 【欧氏距离排序逻辑 - 以聚类结果为准】 *****
        center_distances = np.linalg.norm(kmeans.cluster_centers_, axis=1)
        sorted_center_indices = np.argsort(center_distances)

        category_map = {}

        if optimal_k == 1:
            center_z = kmeans.cluster_centers_[0, 2]
            if center_z < Z_SCORE_NORMAL_THRESHOLD:
                category_map = {0: '正常'}
            elif center_z > Z_SCORE_FAULT_THRESHOLD:
                category_map = {0: '故障'}
            else:
                category_map = {0: '疑似故障'}
        elif optimal_k == 2:
            category_map[sorted_center_indices[0]] = '正常'
            fault_cluster_idx = sorted_center_indices[1]
            if kmeans.cluster_centers_[fault_cluster_idx, 2] > Z_SCORE_FAULT_THRESHOLD:
                category_map[fault_cluster_idx] = '故障'
            else:
                category_map[fault_cluster_idx] = '疑似故障'
        elif optimal_k >= 3:
            category_map[sorted_center_indices[0]] = '正常'
            category_map[sorted_center_indices[-1]] = '故障'
            for idx in sorted_center_indices[1:-1]:
                category_map[idx] = '疑似故障'

        for i, cluster_label in enumerate(clusters):
            final_categories[i] = category_map.get(cluster_label, '未知')

        # 确认：Z-Score二次修正逻辑已移除 (符合用户要求)

    # 9. 第三阶段：具体故障类型诊断
    faulty_indices = [i for i, cat in enumerate(final_categories) if cat in ['疑似故障', '故障']]
    fault_diagnoses = diagnose_fault_type(
        faulty_indices,
        cell_window_indices, all_window_features, anomaly_labels, get_feature_names()
    )

    # 10. 最终结果输出
    normal_indices = [i for i, cat in enumerate(final_categories) if cat == '正常']
    suspect_indices = [i for i, cat in enumerate(final_categories) if cat == '疑似故障']
    fault_indices = [i for i, cat in enumerate(final_categories) if cat == '故障']
    all_indices = normal_indices + suspect_indices + fault_indices

    print("\n### 最终单体级诊断结果 (第二层，K-Means 聚类): ###")
    print("-" * 140)
    print(
        f"{'电池单体':<10} | {'诊断类别':<10} | {'Max Score':<12} | {'异常窗口数':<12} | {'Max Abs Z-Score(突发)':<25} | {'Z-Score偏度(渐进)':<20} | {'故障类型':<25}")
    print("-" * 140)
    for i in all_indices:
        agg_max_score = aggregated_features[i, 0]
        agg_count = int(aggregated_features[i, 1])
        agg_z_score = aggregated_features[i, 2]
        agg_skew = aggregated_features[i, 3]
        category = final_categories[i]
        fault_type = fault_diagnoses.get(i, '无')
        print(
            f"{cell_names[i]:<10} | {category:<10} | {agg_max_score:<12.4f} | {agg_count:<12} | {agg_z_score:<25.2f} | {agg_skew:<20.2f} | {fault_type:<25}"
        )
    print("-" * 140)

    # 11. 可视化
    logging.info("正在生成三类单体的电压曲线图和 Z-score 点图...")
    plot_voltage_curves_by_category(normal_indices, '正常', cell_names, processed_voltage_df)
    plot_voltage_curves_by_category(suspect_indices, '疑似故障', cell_names, processed_voltage_df)
    plot_voltage_curves_by_category(fault_indices, '故障', cell_names, processed_voltage_df)
    plot_z_scores(aggregated_z_scores, final_categories, cell_names)

    logging.info("双层电池故障诊断流程：已完成所有阶段的增强诊断。请查看弹出的所有图表。")