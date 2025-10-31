import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging
import warnings
import scipy.stats as stats
from antropy import sample_entropy

warnings.filterwarnings('ignore', category=UserWarning)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- 参数配置 ---
# 注意：VOLTAGE_FILE_PATH 变量在这里需要手动更改为您电脑上的实际路径
VOLTAGE_FILE_PATH = r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42数据\L42_2_vol.xlsx"
SMOOTHING_WINDOW_SIZE = 3
WINDOW_SIZE = 100
STRIDE = 1
IFOREST_N_ESTIMATORS = 150
IFOREST_CONTAMINATION = 'auto'
Z_SCORE_FAULT_THRESHOLD = 6
Z_SCORE_SUSPECT_THRESHOLD = 4.5
MAX_K_CLUSTERS = 5

# 【集体突发故障检测参数】
VOLTAGE_JUMP_THRESHOLD = 100  # 单体电压跳变绝对阈值 (V)
COLLECTIVE_CELL_RATIO = 0.6  # 集体跳变单元比例阈值 (如 0.8 代表 80% 的单元)

SPECIFIED_FEATURES = [
    'max_voltage', 'min_voltage', 'mean_diff1', 'complexity',
    'fft_freq3', 'fft_freq5', 'sample_entropy', 'cumulative_residual',
    'fft_freq1', 'kurtosis',
]


# 数据加载与预处理

def load_voltage_data(file_path):
    """加载原始电压数据，并确保列名正确。"""
    try:
        df = pd.read_excel(file_path)
        voltage_start_col_idx = 2
        time_data = df.iloc[:, 0].copy()

        if df.shape[1] > voltage_start_col_idx:
            battery_data = df.iloc[:, voltage_start_col_idx:].copy()
            battery_names = [f"单体{i + 1}" for i in range(battery_data.shape[1])]
            battery_data.columns = battery_names
        else:
            battery_data = df.iloc[:, 1:].copy()
            battery_names = [f"单体{i + 1}" for i in range(battery_data.shape[1])]
            battery_data.columns = battery_names

        full_df = pd.concat([time_data.reset_index(drop=True), battery_data.reset_index(drop=True)], axis=1)
        logging.info(f"成功加载电池电压数据，形状: {full_df.shape}")
        return full_df, battery_names
    except Exception as e:
        logging.error(f"加载文件出错: {e}")
        return None, None


def preprocess_data(df, smoothing_window):
    """数据预处理：去重、插值、滤波平滑。"""
    logging.info("开始进行数据预处理...")
    initial_rows = len(df)
    df.drop_duplicates(subset=[df.columns[0]], keep='first', inplace=True)
    rows_after_dedup = len(df)
    if initial_rows > rows_after_dedup:
        logging.info(f"成功删除 {initial_rows - rows_after_dedup} 条重复数据。")
    logging.info("开始进行线性插值填充缺失值...")
    df_processed = df.copy()
    df_processed.iloc[:, 1:].interpolate(method='linear', axis=0, inplace=True)
    logging.info(f"开始进行移动平均平滑，窗口大小: {smoothing_window}...")
    df_processed.iloc[:, 1:] = df_processed.iloc[:, 1:].rolling(window=smoothing_window, min_periods=1).mean()
    logging.info("数据预处理完成。")
    return df_processed

# --- 2. 增强特征提取 (31个特征) ---

def calculate_entropy(data):
    """
    计算电压信号的样本熵。
    """
    if len(data) < 2:
        return 0

    # 局部中心化和归一化 (Z-Score 归一化)
    std_data = np.std(data)

    if std_data > 1e-9:
        normalized_data = (data - np.mean(data)) / std_data
    else:
        # 如果标准差为零 (常数序列)，进行简单的中心化即可，std=0时，熵值应为0或接近0
        normalized_data = data - np.mean(data)

    try:
        # 使用归一化后的数据。
        # 移除 'r' 关键字参数，让 antropy 使用默认的容差 (通常为 0.2 * std(input))
        return sample_entropy(normalized_data, order=2, metric='chebyshev')
    except Exception as e:
        logging.warning(f"计算样本熵时发生错误 ({type(e).__name__}): {e}")
        return 0.0


def extract_enhanced_features(window_data, window_global_mean):
    """提取单个电池滑动窗口的增强特征 (22个局部特征)。"""
    features = []

    # --- 基础统计特征 (使用原始电压) ---
    features.append(np.mean(window_data))
    features.append(np.std(window_data))
    features.append(np.min(window_data))
    features.append(np.max(window_data))
    features.append(np.median(window_data))

    # --- 熵与趋势特征 ---
    # 修正：调用 calculate_entropy 时会进行局部中心化和归一化
    features.append(calculate_entropy(window_data))

    # --- 突变敏感特征 (tezen 增强) ---
    if len(window_data) >= 4:
        features.append(stats.kurtosis(window_data))
    else:
        features.append(0.0)

    if len(window_data) > 1:
        max_abs_diff = np.max(np.abs(np.diff(window_data)))
        features.append(max_abs_diff)
    else:
        features.append(0.0)

    # --- 原有趋势与差分特征 ---
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

    # --- 残差特征 (cumulative_residual) ---
    if len(window_data) > 1 and len(window_data) == len(window_global_mean):
        # 此时 window_data 和 window_global_mean 都是原始电压计算，残差反映该单体相对平均水平的偏差
        voltage_residuals = window_data - window_global_mean
        cumulative_residual = np.sum(voltage_residuals)
        features.append(cumulative_residual)
    else:
        features.append(0.0)

    # --- 频域与波形特征 ---
    if len(window_data) > 10:
        # FFT 依赖于去趋势/去均值，此处使用 window_data - np.mean(window_data)
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
    else:
        features.extend([0.0] * 3)

    return np.array(features)


def extract_global_window_features(window_matrix):
    """提取整个窗口的全局特征 (9 个全局特征)。"""
    features = []
    window_matrix = np.nan_to_num(window_matrix, nan=0.0)
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
        non_constant_cols = np.where(np.nanstd(window_matrix, axis=0) > 1e-9)[0]
        if len(non_constant_cols) > 1:
            corr_matrix = np.corrcoef(window_matrix[:, non_constant_cols].T)
        else:
            corr_matrix = np.array([[0.0]])

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
        upper_triangle = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]

        if len(upper_triangle) > 0:
            features.append(np.mean(upper_triangle))
            features.append(np.min(upper_triangle))
            features.append(np.std(upper_triangle))
            if np.std(upper_triangle) > 1e-10:
                low_threshold = np.mean(upper_triangle) - 2 * np.std(upper_triangle)
                abnormal_corr_ratio = np.mean(upper_triangle < low_threshold)
                features.append(abnormal_corr_ratio)
            else:
                features.append(0.0)
        else:
            features.extend([0.0] * 4)
    else:
        features.extend([0.0] * 4)

    # 9. 窗口内平均绝对差分 (Mean Abs Diff Global)
    diffs = np.diff(window_matrix, axis=0)
    mean_abs_diff_global = np.nanmean(np.abs(diffs)) if diffs.size > 0 else 0.0
    features.append(mean_abs_diff_global)

    return np.array(features)


def get_feature_names():
    """生成所有特征的名称列表 (22个局部 + 9个全局 = 31个特征)。"""
    local_features = [
        'mean_voltage', 'std_voltage', 'min_voltage', 'max_voltage', 'median_voltage',
        'sample_entropy', 'kurtosis', 'max_abs_diff',
        'slope', 'autocorrelation', 'mean_diff1', 'std_diff2', 'cumulative_residual',
        'fft_freq1', 'fft_freq2', 'fft_freq3', 'fft_freq4', 'fft_freq5', 'fft_high_freq_sum',
        'waveform_factor', 'crest_factor', 'complexity'
    ]
    global_features = [
        'max_deviation', 'mean_deviation', 'std_deviation', 'outlier_ratio',
        'mean_correlation', 'min_correlation', 'std_correlation', 'abnormal_corr_ratio',
        'mean_abs_diff_global'
    ]
    return local_features + global_features


def prepare_window_features(voltage_matrix, raw_df, window_size, stride):
    """
    提取所有滑动窗口的特征，并返回窗口起始时间点/索引。
    【已修改】：根据用户要求，移除去均值中心化，直接使用原始电压数据。
    """
    if voltage_matrix.shape[0] < 2 or voltage_matrix.shape[1] == 0:
        logging.warning("电压矩阵数据不足，无法进行归一化和特征提取。")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 【用户请求：移除去均值中心化】。直接使用原始电压矩阵进行特征提取。
    # global_mean_voltage = np.nanmean(voltage_matrix, axis=0, keepdims=True)
    # adjusted_voltage_matrix = voltage_matrix - global_mean_voltage

    # 直接使用原始电压数据（并处理 NaN）
    adjusted_voltage_matrix = np.nan_to_num(voltage_matrix.copy(), nan=0.0)

    all_window_features = []
    cell_window_map = []
    window_start_indices = []
    window_start_times = []
    n_cells = adjusted_voltage_matrix.shape[1]
    n_samples = adjusted_voltage_matrix.shape[0]
    time_series_index = raw_df.iloc[:, 0]

    for start in range(0, n_samples - window_size + 1, stride):
        window_data_matrix = adjusted_voltage_matrix[start:start + window_size, :]
        window_start_indices.append(start)
        window_start_times.append(time_series_index.iloc[start] if start < len(time_series_index) else f"Index_{start}")

        # window_global_mean_vector 仍然反映窗口内所有单体在每个时间步的平均电压
        window_global_mean_vector = np.nanmean(window_data_matrix, axis=1)
        global_features = extract_global_window_features(window_data_matrix)

        for cell_idx in range(n_cells):
            cell_window_data = window_data_matrix[:, cell_idx]
            cell_features = extract_enhanced_features(cell_window_data, window_global_mean_vector)
            combined_features = np.concatenate([cell_features, global_features])
            all_window_features.append(combined_features)
            cell_window_map.append(cell_idx)

    if not all_window_features:
        logging.warning("没有生成有效的窗口特征。")
        return np.array([]), np.array([]), np.array([]), np.array([])

    features_array = np.array(all_window_features)
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    return features_array, np.array(cell_window_map), np.array(window_start_times), np.array(window_start_indices)


def select_specified_features(features, all_feature_names, specified_feature_names):
    """
    根据用户指定的特征名称列表进行特征选择。
    """
    if not specified_feature_names:
        return features, all_feature_names

    specified_indices = []
    for name in specified_feature_names:
        try:
            idx = all_feature_names.index(name)
            specified_indices.append(idx)
        except ValueError:
            logging.warning(f"指定的特征 '{name}' 不存在，已忽略。")

    if not specified_indices:
        logging.error("没有找到任何有效的指定特征，返回所有特征。")
        return features, all_feature_names

    selected_features_array = features[:, specified_indices]
    selected_feature_names = [all_feature_names[i] for i in specified_indices]

    logging.info(f"特征筛选完成。选择了 {len(selected_feature_names)} 个指定特征。")
    return selected_features_array, selected_feature_names


def detect_anomalies_with_isolation_forest(features, n_estimators, contamination):
    """使用孤立森林进行异常检测。"""
    if features.size == 0 or features.shape[1] == 0:
        logging.error("无特征数据可用于检测。")
        return None, None
    # 强制进行标准化，因为现在特征包含绝对电压值，范围差异大
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
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
    计算四个聚合特征。
    【已集成鲁棒性】：使用截尾统计量进行鲁棒 Z-Score 标准化。
    """
    cell_aggregated_scores = np.zeros(n_cells)
    cell_anomaly_counts = np.zeros(n_cells)
    cell_aggregated_z_scores = np.zeros(n_cells)
    cell_z_score_skewness = np.zeros(n_cells)

    if len(window_anomaly_scores) < 2:
        window_z_scores = np.zeros_like(window_anomaly_scores)
    else:
        total_count = len(window_anomaly_scores)
        sorted_scores = np.sort(window_anomaly_scores)

        # 鲁棒性关键步骤：裁剪掉两端2.5%的数据
        trim_low_idx = int(total_count * 0.025)
        trim_high_idx = int(total_count * 0.975)
        trimmed_scores = sorted_scores[trim_low_idx: trim_high_idx]

        # 使用截尾后的数据计算中位数和标准差
        median_trimmed = np.median(trimmed_scores)
        std_trimmed = np.std(trimmed_scores)

        if std_trimmed > 1e-9:
            # 鲁棒Z-Score标准化
            window_z_scores = (median_trimmed - window_anomaly_scores) / std_trimmed
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
    """
    【已集成鲁棒性】：使用肘部法则自动确定最优的K-Means聚类数量，并限制 K <= 3。
    """
    if len(data) < 2: return 1, [0]
    wcss = []
    max_k = min(max_k, len(data) - 1)
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    if len(wcss) < 3: return 1, wcss
    diffs = np.diff(wcss)
    diffs_ratio = np.diff(diffs) / (-diffs[:-1] + 1e-9)
    optimal_k = np.argmax(diffs_ratio) + 2
    return min(optimal_k, 3), wcss


def check_for_collective_jump_and_identify_involved_cells(voltage_matrix, jump_threshold, collective_ratio):
    """
    集体突发故障检测和涉事单体识别。
    """
    if voltage_matrix.shape[0] < 2:
        return False, set()

    n_samples, n_cells = voltage_matrix.shape
    abs_diff = np.abs(np.diff(voltage_matrix, axis=0))
    jump_occurred = abs_diff > jump_threshold
    jump_ratio = np.sum(jump_occurred, axis=1) / n_cells
    collective_event_indices = np.where(jump_ratio >= collective_ratio)[0]
    max_ratio = np.max(jump_ratio) if jump_ratio.size > 0 else 0.0

    if collective_event_indices.size > 0:
        involved_cells_mask = np.any(jump_occurred[collective_event_indices, :], axis=0)
        involved_cell_indices = np.where(involved_cells_mask)[0]

        logging.critical(
            f"\n!!! 检测到集体突发故障 !!! 最高单步跳变单元比例达到 {max_ratio:.4f} (阈值 {collective_ratio:.4f})。"
            f"共 {len(involved_cell_indices)} 个单体涉嫌跳变。"
        )
        return True, set(involved_cell_indices)
    else:
        logging.info(f"未检测到集体突发故障。最高单步跳变单元比例: {max_ratio:.4f} (阈值 {collective_ratio:.4f})")
        return False, set()


def diagnose_fault_type(faulty_indices, cell_window_indices, all_window_features, all_feature_names,
                        aggregated_features, is_collective_fault, involved_cell_indices):
    """
    对故障电池进行故障类型诊断，并优先执行集体突发故障的强制诊断。
    """
    logging.info("开始对故障单体进行故障类型诊断...")
    fault_diagnoses = {}

    try:
        min_correlation_idx = all_feature_names.index('min_correlation')
    except ValueError:
        logging.error("无法找到所需的特征索引 'min_correlation'，跳过静态不一致故障诊断。")
        for cell_idx in faulty_indices:
            fault_diagnoses[cell_idx] = '特征缺失，无法诊断'
        return fault_diagnoses

    for cell_idx in faulty_indices:
        # ** A. 集体故障强制诊断 (最高优先级) **
        if is_collective_fault and cell_idx in involved_cell_indices:
            fault_diagnoses[cell_idx] = '突发性故障 (集体跳变修正)'
            continue

        # ** B. 常规诊断 **
        diagnosis_results = []
        agg_count = int(aggregated_features[cell_idx, 1])
        agg_z_score = aggregated_features[cell_idx, 2]
        agg_skew = aggregated_features[cell_idx, 3]
        mask = cell_window_indices == cell_idx
        cell_features_all = all_window_features[mask, :]

        # 1. 突发性故障 (Sudden Fault)
        if agg_z_score >= 5.0 and agg_count <= 5:
            diagnosis_results.append('突发性故障')

        # 2. 渐进波动故障 (Progressive Fluctuation Fault)
        elif agg_count >= 5 and agg_skew >= 0.5 and agg_z_score < 5.0:
            diagnosis_results.append('渐进波动故障')

        # 3. 静态不一致故障 (Static Inconsistency Fault)
        elif cell_features_all.size > 0:
            min_correlation_value = np.min(cell_features_all[:, min_correlation_idx])
            if min_correlation_value <= 0.75 and agg_z_score < 5.0:
                diagnosis_results.append('静态不一致故障')

        # 4. 默认/兜底
        if not diagnosis_results:
            if agg_z_score >= 4.5:
                fault_diagnoses[cell_idx] = '突发性'
            elif agg_count >= 3:
                fault_diagnoses[cell_idx] = '轻微渐进性'
            else:
                fault_diagnoses[cell_idx] = '其他未知轻微故障'
        else:
            fault_diagnoses[cell_idx] = ', '.join(list(set(diagnosis_results)))

    return fault_diagnoses


# --- 5. 可视化函数 (保持完整) ---

def plot_all_raw_voltage_curves(df, cell_names):
    """绘制所有单体的原始电压曲线图。"""
    plt.figure(figsize=(15, 8))
    time_index = df.iloc[:, 0]
    for cell in cell_names:
        plt.plot(time_index, df[cell], label=cell, linewidth=0.5)
    plt.title('所有单体的原始电压曲线')
    plt.xlabel('时间/索引')
    plt.ylabel('电压 (V)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


def plot_all_processed_voltage_curves(voltage_df, cell_names, anomaly_window_info):
    """绘制平滑后的电压曲线，并标记异常窗口的起始点。"""
    plt.figure(figsize=(15, 8))
    time_index = voltage_df.index.values

    for cell in cell_names:
        plt.plot(time_index, voltage_df[cell], linewidth=0.5, label=cell)

    anomaly_points = [(cell_idx, start_index) for cell_idx, start_index in anomaly_window_info]

    if anomaly_points:
        anomaly_cell_indices = [point[0] for point in anomaly_points]
        anomaly_df_indices = [point[1] for point in anomaly_points]

        valid_indices = [i for i, df_idx in enumerate(anomaly_df_indices) if df_idx < len(time_index)]

        anomaly_cell_indices = [anomaly_cell_indices[i] for i in valid_indices]
        anomaly_df_indices = [anomaly_df_indices[i] for i in valid_indices]

        if anomaly_df_indices:
            anomaly_times = time_index[anomaly_df_indices]
            anomaly_voltages = [voltage_df.iloc[df_idx, cell_idx]
                                for cell_idx, df_idx in zip(anomaly_cell_indices, anomaly_df_indices)]

            plt.scatter(anomaly_times, anomaly_voltages, color='red', marker='o',
                        s=10, zorder=5, label='IF 异常窗口起点')

    plt.title('平滑后的电压曲线与 IF 异常窗口标记')
    plt.xlabel('时间/索引')
    plt.ylabel('电压 (V)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


def plot_elbow_method(wcss):
    """绘制肘部法则图。"""
    plt.figure(figsize=(8, 5))
    k_range = range(1, len(wcss) + 1)
    plt.plot(k_range, wcss, 'bx-')
    plt.xlabel('聚类数 K')
    plt.ylabel('WCSS (簇内平方和)')
    plt.title('肘部法则确定最优聚类数 K')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_clusters_3d(scaled_features, clusters, kmeans, cell_names):
    """绘制 K-Means 聚类的 3D 散点图 (使用 Max Abs Z-Score, Count, Skewness)。"""
    if scaled_features.shape[1] < 4:
        logging.warning("特征维度少于 4，无法绘制基于 Max Z, Count, Skewness 的 3D 图。")
        return

    feature_indices = [2, 1, 3]  # Max Abs Z-Score, Count, Skewness
    feature_labels = ['Max Abs Z-Score', '异常窗口数 (Count)', 'Z-Score 偏度 (Skewness)']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    max_cluster = np.max(clusters) if np.size(clusters) > 0 else 0
    cmap = plt.get_cmap('viridis')
    colors = cmap(clusters / (max_cluster + 1e-9))

    scatter = ax.scatter(scaled_features[:, feature_indices[0]],
                         scaled_features[:, feature_indices[1]],
                         scaled_features[:, feature_indices[2]],
                         c=colors, marker='o', s=50, alpha=0.8)

    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, feature_indices[0]],
               centers[:, feature_indices[1]],
               centers[:, feature_indices[2]],
               marker='x', s=200, linewidths=3, color='red', label='聚类中心')

    ax.set_xlabel(feature_labels[0])
    ax.set_ylabel(feature_labels[1])
    ax.set_zlabel(feature_labels[2])
    ax.set_title('K-Means 聚类结果 (3D)')

    legend_elements = []
    for i in np.unique(clusters):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=f'簇 {i}',
                                          markerfacecolor=cmap(i / (max_cluster + 1e-9)), markersize=10))
    ax.legend(handles=legend_elements, title='聚类标签')
    plt.show()


def plot_voltage_curves_by_category(indices, category_name, cell_names, voltage_df):
    """根据分类绘制电压曲线。"""
    if not indices:
        logging.info(f"没有 {category_name} 类别单体，跳过绘图。")
        return

    plt.figure(figsize=(15, 8))
    for i in indices:
        cell = cell_names[i]
        plt.plot(voltage_df.index, voltage_df[cell], label=cell, linewidth=0.5)

    plt.title(f'{category_name} 电池单体的电压曲线')
    plt.xlabel('时间/索引')
    plt.ylabel('电压 (V)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(ncol=10, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.show()


def plot_z_scores(aggregated_z_scores, final_categories_2_levels, cell_names):
    """绘制最终 Max Abs Z-Score 的散点图。"""
    plt.figure(figsize=(15, 6))

    categories = np.array(final_categories_2_levels)
    z_scores = aggregated_z_scores

    normal_mask = categories == '正常'
    fault_mask = categories == '故障'

    plt.scatter(np.where(normal_mask)[0], z_scores[normal_mask], label='正常', color='blue', marker='o')
    plt.scatter(np.where(fault_mask)[0], z_scores[fault_mask], label='故障 (含疑似)', color='red', marker='x')

    plt.axhline(y=Z_SCORE_FAULT_THRESHOLD, color='r', linestyle='--',
                label=f'故障绝对阈值 (Z={Z_SCORE_FAULT_THRESHOLD})')
    plt.axhline(y=Z_SCORE_SUSPECT_THRESHOLD, color='orange', linestyle='--',
                label=f'疑似绝对阈值 (Z={Z_SCORE_SUSPECT_THRESHOLD})')

    plt.title('Max Abs Z-Score 分布图 (基于聚合 IF 分数)')
    plt.xlabel('电池单体索引')
    plt.ylabel('Max Abs Z-Score')
    plt.xticks(np.arange(len(cell_names)), cell_names, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    logging.info("诊断开始")

    # 1. 数据加载与预处理
    raw_df, cell_names = load_voltage_data(VOLTAGE_FILE_PATH)
    if raw_df is None: exit()

    plot_all_raw_voltage_curves(raw_df, cell_names)

    processed_df = preprocess_data(raw_df.copy(), SMOOTHING_WINDOW_SIZE)
    voltage_matrix = processed_df.iloc[:, 1:].values.astype(float)
    processed_voltage_df = processed_df.iloc[:, 1:]

    # 2. 增强特征提取
    all_window_features, cell_window_indices, window_start_times, window_start_indices = prepare_window_features(
        voltage_matrix, raw_df, window_size=WINDOW_SIZE, stride=STRIDE
    )
    if all_window_features.size == 0: exit()

    # 3. 特征筛选
    all_feature_names = get_feature_names()
    filtered_features, selected_feature_names = select_specified_features(
        all_window_features, all_feature_names, SPECIFIED_FEATURES)

    if filtered_features.size == 0 or filtered_features.shape[1] == 0:
        logging.error("特征列表没有选择到有效特征。")
        exit()

    # 4. 第一层：孤立森林异常检测 (打分)
    window_anomaly_scores, anomaly_labels = \
        detect_anomalies_with_isolation_forest(
            filtered_features,
            n_estimators=IFOREST_N_ESTIMATORS,
            contamination=IFOREST_CONTAMINATION
        )
    if window_anomaly_scores is None: exit()

    anomaly_window_indices = np.where(anomaly_labels == -1)[0]
    anomaly_window_info = []

    for window_global_idx in anomaly_window_indices:
        cell_idx = cell_window_indices[window_global_idx]
        window_time_idx = window_global_idx // len(cell_names)
        start_index_in_df = window_start_indices[window_time_idx]
        anomaly_window_info.append((cell_idx, start_index_in_df))

    plot_all_processed_voltage_curves(processed_voltage_df, cell_names, anomaly_window_info)

    # 5. 聚合特征计算 (包含鲁棒 Z-Score)
    aggregated_features, scaled_aggregated_features = calculate_four_agg_features(
        window_anomaly_scores, anomaly_labels, cell_window_indices, len(cell_names)
    )
    aggregated_z_scores = aggregated_features[:, 2]

    # 6. 【第二层诊断】：K-Means 纯聚类分级 + 动态/绝对阈值修正
    final_categories_3_levels = ['' for _ in range(len(cell_names))]

    optimal_k, wcss = find_optimal_k_by_elbow_method(scaled_aggregated_features)
    plot_elbow_method(wcss)

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_aggregated_features)
    plot_clusters_3d(scaled_aggregated_features, clusters, kmeans, cell_names)

    center_z_scores = kmeans.cluster_centers_[:, 2]
    sorted_center_indices = np.argsort(center_z_scores)

    category_map_3_levels = {}
    if optimal_k == 1:
        category_map_3_levels = {0: '正常'}
    elif optimal_k == 2:
        category_map_3_levels[sorted_center_indices[0]] = '正常'
        category_map_3_levels[sorted_center_indices[1]] = '故障'
    elif optimal_k >= 3:
        category_map_3_levels[sorted_center_indices[0]] = '正常'
        category_map_3_levels[sorted_center_indices[-1]] = '故障'
        for idx in sorted_center_indices[1:-1]:
            category_map_3_levels[idx] = '疑似故障'

    for i, cluster_label in enumerate(clusters):
        final_categories_3_levels[i] = category_map_3_levels.get(cluster_label, '未知')

    # ** K-Means 动态边界 + 绝对阈值修正 **
    normal_z_scores_initial = aggregated_features[
        [i for i, cat in enumerate(final_categories_3_levels) if cat == '正常'], 2]
    mean_normal = np.mean(normal_z_scores_initial) if len(normal_z_scores_initial) > 1 else np.inf
    std_normal = np.std(normal_z_scores_initial) if len(normal_z_scores_initial) > 1 else 0
    DYNAMIC_NORMAL_BOUNDARY = mean_normal + 3 * std_normal  # 用于将正常升级到疑似的动态阈值

    suspect_z_scores_initial = aggregated_features[
        [i for i, cat in enumerate(final_categories_3_levels) if cat == '疑似故障'], 2]
    mean_suspect = np.mean(suspect_z_scores_initial) if len(suspect_z_scores_initial) > 1 else np.inf
    std_suspect = np.std(suspect_z_scores_initial) if len(suspect_z_scores_initial) > 1 else 0
    DYNAMIC_SUSPECT_BOUNDARY = max(mean_suspect - 3 * std_suspect, DYNAMIC_NORMAL_BOUNDARY)

    for i in range(len(cell_names)):
        agg_z_score = aggregated_features[i, 2]
        # 修正一：动态升级
        if (final_categories_3_levels[i] == '正常' and
                DYNAMIC_NORMAL_BOUNDARY < agg_z_score <= DYNAMIC_SUSPECT_BOUNDARY):
            final_categories_3_levels[i] = '疑似故障'

        # 修正二：基于用户记忆和最新指示，将低于绝对阈值的疑似故障降级为正常
        elif (final_categories_3_levels[i] == '疑似故障' and
              agg_z_score < Z_SCORE_SUSPECT_THRESHOLD):
            final_categories_3_levels[i] = '正常'
            logging.warning(
                f"单体 {cell_names[i]} (Max Z={agg_z_score:.2f}) K-Means为'疑似'，但未达绝对阈值 {Z_SCORE_SUSPECT_THRESHOLD}，强制降级为'正常'。"
            )
    # 必须在全局校验之前计算，供全局校验逻辑使用
    final_categories_2_levels = [
        '故障' if cat in ['疑似故障', '故障'] else '正常'
        for cat in final_categories_3_levels
    ]

    # 使用经过上述所有修正后的 final_categories_3_levels 来确定哪些是 '正常'
    final_normal_z_scores = aggregated_features[
        [i for i, cat in enumerate(final_categories_3_levels) if cat == '正常'], 2]

    if len(final_normal_z_scores) > 0:
        max_normal_z_score = np.max(final_normal_z_scores)
        mean_normal_z = np.mean(final_normal_z_scores)
        std_normal_z = np.std(final_normal_z_scores)

        # 动态安全边界：正常簇最大值应低于其平均值 + 3倍标准差
        DYNAMIC_SAFE_BOUNDARY = mean_normal_z + 3 * std_normal_z

        # 硬性轻微故障阈值：使用 SUSPECT 阈值作为全局轻微故障的上限
        GLOBAL_NORMAL_CHECK_THRESHOLD = Z_SCORE_SUSPECT_THRESHOLD

        logging.info(f"全局正常性校验：正常簇最大 Z-Score: {max_normal_z_score:.2f}。")
        logging.info(f"动态安全边界 (mean+3std): {DYNAMIC_SAFE_BOUNDARY:.2f}，硬性阈值: {GLOBAL_NORMAL_CHECK_THRESHOLD}")

        # 判断：如果最大正常 Z-Score 既没有超过动态安全边界，也没有超过硬性轻微故障阈值 (5)
        if max_normal_z_score < DYNAMIC_SAFE_BOUNDARY and max_normal_z_score < GLOBAL_NORMAL_CHECK_THRESHOLD:
            logging.warning(
                "\n!!! 全局正常性校验触发 !!! 所有单体 Z-Score 均低于轻微故障阈值。强制所有单体类别为 '正常'。")

            # 强制修正三级分类
            final_categories_3_levels = ['正常' for _ in range(len(cell_names))]
            # 强制修正二级分类 (覆盖之前计算的结果)
            final_categories_2_levels = ['正常' for _ in range(len(cell_names))]
        else:
            logging.info("全局正常性校验未触发，继续使用聚类修正结果。")

    # 7. 集体突发故障检测和强制分类
    is_collective_fault, involved_cell_indices = check_for_collective_jump_and_identify_involved_cells(
        voltage_matrix,
        VOLTAGE_JUMP_THRESHOLD,
        COLLECTIVE_CELL_RATIO
    )

    if is_collective_fault:
        logging.critical("集体突发故障强制分类修正启动。")
        for i in involved_cell_indices:
            # 强制升级类别为 '故障' (最高优先级)
            if final_categories_2_levels[i] != '故障':
                final_categories_2_levels[i] = '故障'
                logging.warning(f"单体 {cell_names[i]} 被集体故障检测强制升级为 '故障'。")
    # 如果全局校验未触发，则执行原有逻辑。如果触发，final_categories_2_levels 已经更新为 '正常'
    if '故障' not in final_categories_2_levels:  # 检查是否被全局校验重置
        final_categories_2_levels = [
            '故障' if cat in ['疑似故障', '故障'] else '正常'
            for cat in final_categories_3_levels
        ]

    # 7. 集体突发故障检测和强制分类
    is_collective_fault, involved_cell_indices = check_for_collective_jump_and_identify_involved_cells(
        voltage_matrix,
        VOLTAGE_JUMP_THRESHOLD,
        COLLECTIVE_CELL_RATIO
    )

    if is_collective_fault:
        logging.critical("集体突发故障强制分类修正启动。")
        for i in involved_cell_indices:
            # 强制升级类别为 '故障' (最高优先级)
            if final_categories_2_levels[i] != '故障':
                final_categories_2_levels[i] = '故障'
                logging.warning(f"单体 {cell_names[i]} 被集体故障检测强制升级为 '故障'。")

    # 8. 第三阶段：具体故障类型诊断 (使用修正后的分类结果)
    faulty_indices_2_levels = [i for i, cat in enumerate(final_categories_2_levels) if cat == '故障']
    fault_diagnoses = diagnose_fault_type(
        faulty_indices_2_levels,
        cell_window_indices,
        all_window_features,
        all_feature_names,
        aggregated_features,
        is_collective_fault,
        involved_cell_indices
    )

    # 9. 最终结果输出
    all_indices = list(range(len(cell_names)))

    print("\n### 最终单体级诊断结果 (第二层，最鲁棒混合聚类): ###")
    print("-" * 155)
    print(
        f"{'电池单体':<10} | {'最终类别(两级)':<15} | {'原始类别(三级)':<15} | {'Max Score':<12} | {'异常窗口数':<12} | {'Max Abs Z-Score(突发)':<25} | {'Z-Score偏度(渐进)':<20} | {'故障类型':<25}")
    print("-" * 155)
    for i in all_indices:
        agg_max_score = aggregated_features[i, 0]
        agg_count = int(aggregated_features[i, 1])
        agg_z_score = aggregated_features[i, 2]
        agg_skew = aggregated_features[i, 3]

        category_2_level = final_categories_2_levels[i]
        category_3_level = final_categories_3_levels[i]

        fault_type = fault_diagnoses.get(i, '无')
        print(
            f"{cell_names[i]:<10} | {category_2_level:<15} | {category_3_level:<15} | {agg_max_score:<12.4f} | {agg_count:<12} | {agg_z_score:<25.2f} | {agg_skew:<20.2f} | {fault_type:<25}"
        )
    print("-" * 155)

    # 10. 可视化
    normal_indices = [i for i, cat in enumerate(final_categories_2_levels) if cat == '正常']
    fault_indices_combined = [i for i, cat in enumerate(final_categories_2_levels) if cat == '故障']

    logging.info("正在生成两类单体的电压曲线图和 Z-score 点图 (疑似故障已合并到故障)...")

    plot_voltage_curves_by_category(normal_indices, '正常', cell_names, processed_voltage_df)
    plot_voltage_curves_by_category(fault_indices_combined, '故障 (含疑似)', cell_names, processed_voltage_df)
    plot_z_scores(aggregated_z_scores, final_categories_2_levels, cell_names)

    logging.info("双层电池故障诊断流程：已完成所有阶段的增强诊断。请查看弹出的所有图表。")