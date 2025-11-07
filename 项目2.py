import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest, RandomForestRegressor  # 导入 RandomForestRegressor
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
VOLTAGE_FILE_PATH = r"F:\电池故障诊断\文献复现\自创电池故障诊断\L42数据\L42_4.11_vol.xlsx"
SMOOTHING_WINDOW_SIZE = 3
WINDOW_SIZE = 100
STRIDE = 1
IFOREST_N_ESTIMATORS = 150
IFOREST_CONTAMINATION = 'auto'
Z_SCORE_FAULT_THRESHOLD = 6
Z_SCORE_SUSPECT_THRESHOLD = 4.5
MAX_K_CLUSTERS = 5

# 【集体突发故障检测参数】 (保留参数，但相关逻辑已移除)
VOLTAGE_JUMP_THRESHOLD = 100  # 单体电压跳变绝对阈值 (V)
COLLECTIVE_CELL_RATIO = 0.6  # 集体跳变单元比例阈值 (如 0.8 代表 80% 的单元)

# *** 保持这10个特征用于评分和诊断 ***
SPECIFIED_FEATURES = [
    'max_voltage', 'min_voltage', 'mean_diff1', 'complexity',
    'fft_freq3', 'fft_freq5', 'sample_entropy', 'cumulative_residual',
    'fft_freq1', 'kurtosis',
]


# ***********************************


# 数据加载与预处理 (保持不变)
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


# --- 2. 增强特征提取 (保持不变) ---

def calculate_entropy(data):
    """计算电压信号的样本熵。"""
    if len(data) < 2: return 0
    std_data = np.std(data)
    if std_data > 1e-9:
        normalized_data = (data - np.mean(data)) / std_data
    else:
        normalized_data = data - np.mean(data)
    try:
        return sample_entropy(normalized_data, order=2, metric='chebyshev')
    except Exception:
        return 0.0


# (其他局部特征和全局特征提取函数保持不变，为了脚本简洁，此处省略)

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
        voltage_residuals = window_data - window_global_mean
        cumulative_residual = np.sum(voltage_residuals)
        features.append(cumulative_residual)
    else:
        features.append(0.0)

    # --- 频域与波形特征 ---
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
    # (此函数逻辑保持不变)
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
    # (此函数逻辑保持不变)
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
    # (此函数逻辑保持不变)
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
    anomaly_labels = model.predict(scaled_features)  # -1: 异常, 1: 正常

    min_score = np.min(anomaly_scores)
    positive_anomaly_scores = anomaly_scores - min_score

    return positive_anomaly_scores, anomaly_labels


# (其他聚合特征、聚类、诊断函数保持不变，此处省略)

def calculate_feature_importance_rf(features, anomaly_scores, feature_names):
    """
    使用 Random Forest Regressor 预测 Isolation Forest 的异常分数，
    并提取特征重要性。
    """
    logging.info("\n*** 开始计算 Random Forest 特征评分 (以预测 IF 异常分数) ***")

    if features.size == 0 or len(anomaly_scores) == 0:
        logging.error("没有足够的特征或分数数据来计算特征重要性。")
        return None

    # 1. 数据准备
    X = features
    # Y轴是 IF 导出的异常分数
    Y = anomaly_scores

    # 2. 标准化特征 (与IFOREST使用相同)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 训练 Random Forest Regressor
    # 使用回归模型来预测连续的异常分数
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_scaled, Y)

    # 4. 提取特征重要性
    importances = rf_model.feature_importances_

    # 5. 结果整理
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    logging.info("Random Forest 特征评分计算完成。")
    return importance_df


# ... (其他函数省略) ...


# 主程序
if __name__ == "__main__":
    logging.info("诊断开始")

    # 1. 数据加载与预处理
    raw_df, cell_names = load_voltage_data(VOLTAGE_FILE_PATH)
    if raw_df is None: exit()
    # plot_all_raw_voltage_curves(raw_df, cell_names) # 绘图已注释

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
    # 仅保留指定的10个特征
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

    # --- ！！！ 新增：随机森林特征评分 ！！！ ---

    feature_importance_df = calculate_feature_importance_rf(
        filtered_features,
        window_anomaly_scores,
        selected_feature_names
    )

    # 打印特征评分结果
    if feature_importance_df is not None:
        print("\n" + "=" * 50)
        print("         ** Random Forest 特征重要性评分 **")
        print("       (用于解释 Isolation Forest 异常分数)")
        print("=" * 50)
        print(feature_importance_df.to_string(index=False))
        print("=" * 50)

    # --- 5. 聚合特征计算 (略过，但函数调用保留，以便后续诊断) ---
    aggregated_features, scaled_aggregated_features = calculate_four_agg_features(
        window_anomaly_scores, anomaly_labels, cell_window_indices, len(cell_names)
    )
    # (后续的 K-Means 聚类和诊断逻辑因代码量较大，在此处被省略，但假设其在完整脚本中保留)
    # ... (省略 K-Means 聚类和诊断逻辑) ...

    logging.info("特征重要性评分计算和输出完成。")