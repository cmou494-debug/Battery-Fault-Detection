# tezen_selec_final.py
# 完整版：包含稳健的动态修正模块（正常/故障簇检查 + 疑似属吸收）
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
from scipy.spatial.distance import mahalanobis

warnings.filterwarnings('ignore', category=UserWarning)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 全局样式参数 ---
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- 参数（按需修改） ---
VOLTAGE_FILE_PATH = r"F:\电池故障诊断\文献复现\自创电池故障诊断\LB04各项数据文件\L04_7.4.xlsx"
SMOOTHING_WINDOW_SIZE = 3
WINDOW_SIZE = 100
STRIDE = 1
IFOREST_N_ESTIMATORS = 150
IFOREST_CONTAMINATION = 'auto'
Z_SCORE_FAULT_THRESHOLD = 6.0
Z_SCORE_SUSPECT_THRESHOLD = 4.5
MAX_K_CLUSTERS = 5
VOLTAGE_JUMP_THRESHOLD = 100
COLLECTIVE_CELL_RATIO = 0.6

SPECIFIED_FEATURES = [
    'max_voltage', 'min_voltage', 'mean_diff1', 'complexity',
    'fft_freq3', 'fft_freq5', 'sample_entropy', 'cumulative_residual',
    'fft_freq1', 'kurtosis',
]

# ---------------- 数据加载与预处理 ----------------
def load_voltage_data(file_path):
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
    logging.info("开始进行数据预处理...")
    initial_rows = len(df)
    df.drop_duplicates(subset=[df.columns[0]], keep='first', inplace=True)
    df_processed = df.copy()
    df_processed.iloc[:, 1:].interpolate(method='linear', axis=0, inplace=True)
    df_processed.iloc[:, 1:] = df_processed.iloc[:, 1:].rolling(window=smoothing_window, min_periods=1).mean()
    logging.info("数据预处理完成。")
    return df_processed

# ---------------- 特征提取 ----------------
def calculate_entropy(data):
    if len(data) < 2:
        return 0
    std_data = np.std(data)
    if std_data > 1e-9:
        normalized_data = (data - np.mean(data)) / std_data
    else:
        normalized_data = data - np.mean(data)
    try:
        return sample_entropy(normalized_data, order=2, metric='chebyshev')
    except Exception as e:
        logging.warning(f"计算样本熵时发生错误 ({type(e).__name__}): {e}")
        return 0.0

def extract_enhanced_features(window_data, window_global_mean):
    features = []
    features.append(np.mean(window_data))
    features.append(np.std(window_data))
    features.append(np.min(window_data))
    features.append(np.max(window_data))
    features.append(np.median(window_data))
    features.append(calculate_entropy(window_data))
    if len(window_data) >= 4:
        features.append(stats.kurtosis(window_data))
    else:
        features.append(0.0)
    if len(window_data) > 1:
        max_abs_diff = np.max(np.abs(np.diff(window_data)))
        features.append(max_abs_diff)
    else:
        features.append(0.0)
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
    else:
        features.extend([0.0] * 3)
    return np.array(features)

def extract_global_window_features(window_matrix):
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
    diffs = np.diff(window_matrix, axis=0)
    mean_abs_diff_global = np.nanmean(np.abs(diffs)) if diffs.size > 0 else 0.0
    features.append(mean_abs_diff_global)
    return np.array(features)

def get_feature_names():
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
    if voltage_matrix.shape[0] < 2 or voltage_matrix.shape[1] == 0:
        logging.warning("电压矩阵数据不足，无法进行归一化和特征提取。")
        return np.array([]), np.array([]), np.array([]), np.array([])
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

# ---------------- 孤立森林 & 聚合 ----------------
def detect_anomalies_with_isolation_forest(features, n_estimators, contamination):
    if features.size == 0 or features.shape[1] == 0:
        logging.error("无特征数据可用于检测。")
        return None, None
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                            random_state=42, n_jobs=-1)
    model.fit(scaled_features)
    logging.info("第一层诊断：孤立森林模型训练完成。")
    anomaly_scores = model.decision_function(scaled_features)
    anomaly_labels = model.predict(scaled_features)
    min_score = np.min(anomaly_scores)
    positive_anomaly_scores = anomaly_scores - min_score
    return positive_anomaly_scores, anomaly_labels

def calculate_four_agg_features(window_anomaly_scores, anomaly_labels, cell_window_indices, n_cells):
    cell_aggregated_scores = np.zeros(n_cells)
    cell_anomaly_counts = np.zeros(n_cells)
    cell_aggregated_z_scores = np.zeros(n_cells)
    cell_z_score_skewness = np.zeros(n_cells)
    if len(window_anomaly_scores) < 2:
        window_z_scores = np.zeros_like(window_anomaly_scores)
    else:
        total_count = len(window_anomaly_scores)
        sorted_scores = np.sort(window_anomaly_scores)
        trim_low_idx = int(total_count * 0.025)
        trim_high_idx = int(total_count * 0.975)
        trimmed_scores = sorted_scores[trim_low_idx: trim_high_idx]
        median_trimmed = np.median(trimmed_scores)
        std_trimmed = np.std(trimmed_scores)
        if std_trimmed > 1e-9:
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
        if is_collective_fault and cell_idx in involved_cell_indices:
            fault_diagnoses[cell_idx] = '突发性故障 (集体跳变修正)'
            continue
        diagnosis_results = []
        agg_count = int(aggregated_features[cell_idx, 1])
        agg_z_score = aggregated_features[cell_idx, 2]
        agg_skew = aggregated_features[cell_idx, 3]
        mask = cell_window_indices == cell_idx
        cell_features_all = all_window_features[mask, :]
        if agg_z_score >= 5.0 and agg_count <= 5:
            diagnosis_results.append('突发性故障')
        elif agg_count >= 5 and agg_skew >= 0.5 and agg_z_score < 5.0:
            diagnosis_results.append('渐进波动故障')
        elif cell_features_all.size > 0:
            min_correlation_value = np.min(cell_features_all[:, min_correlation_idx])
            if min_correlation_value <= 0.75 and agg_z_score < 5.0:
                diagnosis_results.append('静态不一致故障')
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

# ---------------- 可视化 ----------------
def plot_all_raw_voltage_curves(df, cell_names):
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
    plt.figure(figsize=(8, 5))
    k_range = range(1, len(wcss) + 1)
    plt.plot(k_range, wcss, 'bx-')
    plt.xlabel('聚类数 K')
    plt.ylabel('WCSS (簇内平方和)')
    plt.title('肘部法则确定最优聚类数 K')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_clusters_2d(scaled_features, clusters, kmeans, cell_names):
    if scaled_features.shape[1] < 2:
        logging.warning("特征维度少于 2，无法绘制基于 Max Z 和 Count 的 2D 图。")
        return
    feature_indices = [2, 1]
    feature_labels = ['Max Abs Z-Score', '异常窗口数 (Count)']
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    cmap = plt.get_cmap('viridis')
    scatter = ax.scatter(scaled_features[:, feature_indices[0]],
                         scaled_features[:, feature_indices[1]],
                         c=clusters,
                         cmap=cmap,
                         marker='o', s=50, alpha=0.8)
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, feature_indices[0]],
               centers[:, feature_indices[1]],
               marker='x', s=200, linewidths=6, color='red', label='聚类中心')
    ax.set_xlabel(feature_labels[0])
    ax.set_ylabel(feature_labels[1])
    ax.set_title('K-Means 聚类结果 (2D: Max Abs Z-Score vs. Count)')
    cbar = plt.colorbar(scatter, ticks=np.unique(clusters), label='聚类标签')
    ax.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_corrected_clusters_2d(scaled_features, corrected_categories, kmeans=None, title_suffix='(修正后分类)'):
    if scaled_features.shape[1] < 2:
        logging.warning("特征维度少于 2，无法绘制修正后聚类图。")
        return
    feat_x_idx, feat_y_idx = 2, 1
    x = scaled_features[:, feat_x_idx]
    y = scaled_features[:, feat_y_idx]
    unique_cats = list(dict.fromkeys(corrected_categories))
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    cat_colors = [cat_to_idx[c] for c in corrected_categories]
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    scatter = ax.scatter(x, y, c=cat_colors, cmap='tab10', s=80, alpha=0.9, edgecolors='k')
    if kmeans is not None:
        centers = kmeans.cluster_centers_
        if centers.shape[1] > max(feat_x_idx, feat_y_idx):
            ax.scatter(centers[:, feat_x_idx], centers[:, feat_y_idx],
                       marker='X', s=220, linewidths=2, edgecolors='white',
                       facecolors='none', label='原始聚类中心')
    from matplotlib.lines import Line2D
    legend_elements = []
    cmap = plt.get_cmap('tab10')
    for cat, idx in cat_to_idx.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=cmap(idx), markersize=10, markeredgecolor='k',
                                      label=f'{cat}'))
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', title='修正后类别')
    ax.set_xlabel('Max Abs Z-Score (scaled)')
    ax.set_ylabel('异常窗口数 Count (scaled)')
    ax.set_title(f'修正后分类可视化 {title_suffix}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_voltage_curves_by_category(indices, category_name, cell_names, voltage_df):
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
    plt.figure(figsize=(15, 6))
    categories = np.array(final_categories_2_levels)
    z_scores = aggregated_z_scores
    normal_mask = categories == '正常'
    fault_mask = categories == '故障'
    plt.scatter(np.where(normal_mask)[0], z_scores[normal_mask], label='正常', marker='o')
    plt.scatter(np.where(fault_mask)[0], z_scores[fault_mask], label='故障 (含疑似)', marker='x')
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

def plot_final_fault_clusters_2d(scaled_features, final_categories_2_levels, title_suffix='(最终是否故障结果)'):
    if scaled_features.shape[1] < 2:
        logging.warning("特征维度少于 2，无法绘制最终故障聚类图。")
        return
    feat_x_idx, feat_y_idx = 2, 1
    x = scaled_features[:, feat_x_idx]
    y = scaled_features[:, feat_y_idx]
    unique_cats = list(dict.fromkeys(final_categories_2_levels))
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    cat_colors = [cat_to_idx[c] for c in final_categories_2_levels]
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    scatter = ax.scatter(x, y, c=cat_colors, cmap='Set1', s=90, alpha=0.9, edgecolors='k')
    from matplotlib.lines import Line2D
    cmap = plt.get_cmap('Set1')
    legend_elements = []
    for cat, idx in cat_to_idx.items():
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=cmap(idx), markeredgecolor='k',
                                      markersize=10, label=cat))
    ax.legend(handles=legend_elements, loc='lower right', title='最终分类')
    ax.set_xlabel('Max Abs Z-Score (scaled)')
    ax.set_ylabel('异常窗口数 Count (scaled)')
    ax.set_title(f'最终故障判定聚类可视化 {title_suffix}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    logging.info("诊断开始")

    raw_df, cell_names = load_voltage_data(VOLTAGE_FILE_PATH)
    if raw_df is None:
        logging.error("未能加载数据，程序退出。")
        exit()

    plot_all_raw_voltage_curves(raw_df, cell_names)
    processed_df = preprocess_data(raw_df.copy(), SMOOTHING_WINDOW_SIZE)
    voltage_matrix = processed_df.iloc[:, 1:].values.astype(float)
    processed_voltage_df = processed_df.iloc[:, 1:]

    # 特征提取
    all_window_features, cell_window_indices, window_start_times, window_start_indices = prepare_window_features(
        voltage_matrix, raw_df, window_size=WINDOW_SIZE, stride=STRIDE
    )
    if all_window_features.size == 0:
        logging.error("未生成窗口特征，程序退出。")
        exit()

    all_feature_names = get_feature_names()
    filtered_features, selected_feature_names = select_specified_features(
        all_window_features, all_feature_names, SPECIFIED_FEATURES)

    if filtered_features.size == 0 or filtered_features.shape[1] == 0:
        logging.error("特征列表没有选择到有效特征。程序退出。")
        exit()

    window_anomaly_scores, anomaly_labels = \
        detect_anomalies_with_isolation_forest(
            filtered_features,
            n_estimators=IFOREST_N_ESTIMATORS,
            contamination=IFOREST_CONTAMINATION
        )
    if window_anomaly_scores is None:
        logging.error("孤立森林未能产生结果，程序退出。")
        exit()

    anomaly_window_indices = np.where(anomaly_labels == -1)[0]
    anomaly_window_info = []
    for window_global_idx in anomaly_window_indices:
        cell_idx = cell_window_indices[window_global_idx]
        window_time_idx = window_global_idx // len(cell_names)
        start_index_in_df = window_start_indices[window_time_idx]
        anomaly_window_info.append((cell_idx, start_index_in_df))

    plot_all_processed_voltage_curves(processed_voltage_df, cell_names, anomaly_window_info)

    aggregated_features, scaled_aggregated_features = calculate_four_agg_features(
        window_anomaly_scores, anomaly_labels, cell_window_indices, len(cell_names)
    )
    aggregated_z_scores = aggregated_features[:, 2]

    clustering_features = scaled_aggregated_features[:, [1, 2, 3]]
    logging.info("K-Means 聚类特征已优化：排除 Max Score，仅使用 Count, Max Abs Z-Score, Skewness (3维)。")

    # ---------------- KMeans 分层 ----------------
    final_categories_3_levels = ['' for _ in range(len(cell_names))]
    optimal_k, wcss = find_optimal_k_by_elbow_method(scaled_aggregated_features)
    plot_elbow_method(wcss)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_aggregated_features)
    plot_clusters_2d(scaled_aggregated_features, clusters, kmeans, cell_names)

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

    # ---------------- 稳健动态修正模块（双族检查 + 疑似吸收） ----------------
    # 参考上文讨论：正常簇/故障簇合理性检查 -> 动态边界 -> 疑似点按马氏/欧氏距离分配 -> 退化策略
    normal_z = aggregated_features[[i for i, c in enumerate(final_categories_3_levels) if c == '正常'], 2]
    suspect_z = aggregated_features[[i for i, c in enumerate(final_categories_3_levels) if c == '疑似故障'], 2]
    fault_z   = aggregated_features[[i for i, c in enumerate(final_categories_3_levels) if c == '故障'], 2]

    # ---- 1) 正常簇检查与清洗 ----
    if len(normal_z) > 0:
        mean_n, std_n = np.mean(normal_z), np.std(normal_z)
        normal_z_clean = normal_z[normal_z <= mean_n + 3 * std_n]
        if len(normal_z_clean) < len(normal_z):
            logging.warning(f"正常簇检测到 {len(normal_z) - len(normal_z_clean)} 个高Z异常点，已剔除。")
            normal_z = normal_z_clean
        if len(fault_z) > 0 and np.std(normal_z) > 1.5 * np.std(fault_z):
            logging.warning("正常簇方差异常大，可能混入故障点，取下半部分Z分数。")
            normal_z = normal_z[normal_z <= np.median(normal_z)]

    # ---- 2) 故障簇检查与修正 ----
    if len(fault_z) < 2:
        logging.warning("故障簇样本过少，使用固定阈值 Z=6.0 替代动态边界。")
        fault_z = np.array([Z_SCORE_FAULT_THRESHOLD + 0.1])
    if len(fault_z) > 0 and len(normal_z) > 0:
        if np.std(fault_z) > 1.5 * np.std(normal_z):
            logging.warning("故障簇方差异常大，可能存在混合情况，使用上半部分Z分数。")
            fault_z = fault_z[fault_z >= np.median(fault_z)]

    # ---- 3) 均值逻辑检查（防止标签反转） ----
    if len(fault_z) > 0 and len(normal_z) > 0 and np.mean(fault_z) < np.mean(normal_z):
        logging.warning("检测到聚类标签反转，交换正常与故障标签。")
        final_categories_3_levels = [
            '故障' if c == '正常' else ('正常' if c == '故障' else c)
            for c in final_categories_3_levels
        ]
        normal_z, fault_z = fault_z, normal_z

    # ---- 4) 动态边界计算 ----
    if len(normal_z) > 0 and len(fault_z) > 0:
        mean_normal, std_normal = np.mean(normal_z), np.std(normal_z)
        mean_fault, std_fault   = np.mean(fault_z), np.std(fault_z)
        normal_upper = mean_normal + 3 * std_normal
        fault_lower  = mean_fault - 2 * std_fault
        logging.info(f"动态边界：normal_upper={normal_upper:.2f}, fault_lower={fault_lower:.2f}")
    else:
        normal_upper, fault_lower = Z_SCORE_SUSPECT_THRESHOLD, Z_SCORE_FAULT_THRESHOLD

    # ---- 5) 疑似簇动态吸收（马氏/欧氏距离或退化为Z-score） ----
    normal_idx = [i for i, c in enumerate(final_categories_3_levels) if c == '正常']
    suspect_idx = [i for i, c in enumerate(final_categories_3_levels) if c == '疑似故障']
    fault_idx   = [i for i, c in enumerate(final_categories_3_levels) if c == '故障']

    if len(normal_idx) > 0 and len(fault_idx) > 0 and len(suspect_idx) > 0:
        normal_center = np.mean(scaled_aggregated_features[normal_idx, :], axis=0)
        fault_center  = np.mean(scaled_aggregated_features[fault_idx, :], axis=0)
        try:
            cov_mat = np.cov(scaled_aggregated_features[np.r_[normal_idx, fault_idx], :].T)
            cov_mat += np.eye(cov_mat.shape[0]) * 1e-6
            inv_cov = np.linalg.inv(cov_mat)
            use_mahalanobis = True
        except Exception:
            use_mahalanobis = False
        for i in suspect_idx:
            x = scaled_aggregated_features[i, :]
            if use_mahalanobis:
                d_n = mahalanobis(x, normal_center, inv_cov)
                d_f = mahalanobis(x, fault_center, inv_cov)
            else:
                d_n = np.linalg.norm(x - normal_center)
                d_f = np.linalg.norm(x - fault_center)
            if d_n <= d_f:
                final_categories_3_levels[i] = '正常'
            else:
                final_categories_3_levels[i] = '故障'
    else:
        for i in suspect_idx:
            z = aggregated_features[i, 2]
            if abs(z - normal_upper) <= abs(z - fault_lower):
                final_categories_3_levels[i] = '正常'
            else:
                final_categories_3_levels[i] = '故障'
    # === 新增：绘制“聚类 + 动态修正”后判为 '正常' 的单体电压曲线（在集体故障强制修正之前） ===
    # 说明：
    # 此图展示的是在 KMeans 聚类 + 动态修正完成后（但尚未执行集体故障检测前）、
    # 被判定为“正常”的单体电压曲线。它反映动态修正阶段的分类结果。

    dynamic_corrected_normal_indices = [i for i, c in enumerate(final_categories_3_levels) if c == '正常']

    if len(dynamic_corrected_normal_indices) == 0:
        logging.info("动态修正后没有判为 '正常' 的单体，跳过该可视化。")
    else:
        logging.info(
            f"绘制 {len(dynamic_corrected_normal_indices)} 个动态修正后判为正常的单体电压曲线（未含集体强制修正）。")
        plot_voltage_curves_by_category(
            dynamic_corrected_normal_indices,
            '动态修正后判为正常的单体（未含集体强制修正）',
            cell_names,
            processed_voltage_df
        )

    # ---- 6) 最终二分类输出 ----
    final_categories_2_levels = [
        '故障' if c == '故障' else '正常'
        for c in final_categories_3_levels
    ]

    # ---- 修正后三级可视化 ----
    try:
        plot_corrected_clusters_2d(scaled_aggregated_features, final_categories_3_levels,
                                   kmeans=kmeans, title_suffix='(动态修正后)')
    except Exception as e:
        logging.warning(f"绘制修正后聚类图时出错: {e}")

    # ---- 集体突发故障检测与强制分类 ----
    is_collective_fault, involved_cell_indices = check_for_collective_jump_and_identify_involved_cells(
        voltage_matrix, VOLTAGE_JUMP_THRESHOLD, COLLECTIVE_CELL_RATIO
    )
    if is_collective_fault:
        logging.critical("集体突发故障强制分类修正启动。")
        for i in involved_cell_indices:
            if final_categories_2_levels[i] != '故障':
                final_categories_2_levels[i] = '故障'
                logging.warning(f"单体 {cell_names[i]} 被集体故障检测强制升级为 '故障'。")

    # ---- 最终二类可视化 ----
    try:
        plot_final_fault_clusters_2d(scaled_aggregated_features, final_categories_2_levels,
                                     title_suffix='(最终二分类：正常/故障)')
    except Exception as e:
        logging.warning(f"绘制最终故障聚类图时出错: {e}")

    # ---- 故障诊断与输出 ----
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

    normal_indices = [i for i, cat in enumerate(final_categories_2_levels) if cat == '正常']
    fault_indices_combined = [i for i, cat in enumerate(final_categories_2_levels) if cat == '故障']

    logging.info("正在生成两类单体的电压曲线图和 Z-score 点图 (疑似故障已合并到故障)...")
    plot_voltage_curves_by_category(normal_indices, '正常', cell_names, processed_voltage_df)
    plot_voltage_curves_by_category(fault_indices_combined, '故障 (含疑似)', cell_names, processed_voltage_df)
    plot_z_scores(aggregated_z_scores, final_categories_2_levels, cell_names)
    # === 新增：未使用集体故障检测修正的正常单体电压曲线可视化 ===
    if not is_collective_fault or len(involved_cell_indices) == 0:
        logging.info("未检测到集体故障，无需额外绘制未修正正常单体曲线。")
    else:
        untouched_normal_indices = [
            i for i, cat in enumerate(final_categories_2_levels)
            if cat == '正常' and i not in involved_cell_indices
        ]
        if len(untouched_normal_indices) > 0:
            logging.info(f"绘制 {len(untouched_normal_indices)} 个未被集体故障修正的正常单体电压曲线。")
            plot_voltage_curves_by_category(untouched_normal_indices, '未被集体故障修正的正常单体', cell_names,
                                            processed_voltage_df)
        else:
            logging.info("没有未被集体故障修正的正常单体可供绘制。")

    logging.info("双层电池故障诊断流程：已完成所有阶段的增强诊断。请查看弹出的所有图表。")
