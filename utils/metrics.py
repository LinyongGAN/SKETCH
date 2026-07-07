import math
import torch
from utils.earth_computation import haversine_distance

def ensure_tensor(tensor, device=None):
    """确保输入为张量，并移动到指定设备"""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)
    return tensor

def euclidean_distance(p1, p2):
    """计算两点间的欧氏距离（支持张量）"""
    p1 = ensure_tensor(p1)
    p2 = ensure_tensor(p2)
    
    # 确保在相同设备上
    if p1.device != p2.device:
        p2 = p2.to(p1.device)
    
    return torch.sqrt(torch.sum((p1 - p2) ** 2))

def batch_euclidean_distance(p1, p2):
    """
    批量计算欧氏距离
    参数:
    p1: torch.Tensor, shape (batch_size, n, 2)
    p2: torch.Tensor, shape (batch_size, m, 2)
    返回:
    torch.Tensor: shape (batch_size, n, m)
    """
    p1 = ensure_tensor(p1)
    p2 = ensure_tensor(p2)
    
    if p1.device != p2.device:
        p2 = p2.to(p1.device)
    
    # 扩展维度以便广播计算
    p1_expanded = p1.unsqueeze(2)  # (batch_size, n, 1, 2)
    p2_expanded = p2.unsqueeze(1)  # (batch_size, 1, m, 2)
    
    # 计算所有点对之间的欧氏距离
    distances = torch.sqrt(torch.sum((p1_expanded - p2_expanded) ** 2, dim=-1))
    return distances

def frechet_distance(gt_trajectory, pred_trajectory):
    """
    计算两条轨迹间的离散弗雷歇距离（支持batch）
    
    参数:
    gt_trajectory: torch.Tensor, shape (batch_size, m, 2) 或 (m, 2)
    pred_trajectory: torch.Tensor, shape (batch_size, n, 2) 或 (n, 2)
    
    返回:
    torch.Tensor: 弗雷歇距离，shape (batch_size,) 或 scalar
    """
    gt_trajectory = ensure_tensor(gt_trajectory)
    pred_trajectory = ensure_tensor(pred_trajectory)
    
    # 确保在相同设备上
    if gt_trajectory.device != pred_trajectory.device:
        pred_trajectory = pred_trajectory.to(gt_trajectory.device)
    
    # 处理单条轨迹的情况
    if gt_trajectory.dim() == 2:
        gt_trajectory = gt_trajectory.unsqueeze(0)
    if pred_trajectory.dim() == 2:
        pred_trajectory = pred_trajectory.unsqueeze(0)
    
    batch_size = gt_trajectory.shape[0]
    m, n = gt_trajectory.shape[1], pred_trajectory.shape[1]
    
    # 批量计算所有点对之间的距离矩阵
    point_distances = batch_euclidean_distance(gt_trajectory, pred_trajectory)  # (batch_size, m, n)
    
    # 批量动态规划计算弗雷歇距离
    distance_matrix = torch.zeros((batch_size, m, n), device=gt_trajectory.device)
    
    # 初始化第一个元素
    distance_matrix[:, 0, 0] = point_distances[:, 0, 0]
    
    # 初始化第一行
    for j in range(1, n):
        distance_matrix[:, 0, j] = torch.max(
            distance_matrix[:, 0, j-1], 
            point_distances[:, 0, j]
        )
    
    # 初始化第一列
    for i in range(1, m):
        distance_matrix[:, i, 0] = torch.max(
            distance_matrix[:, i-1, 0], 
            point_distances[:, i, 0]
        )
    
    # 填充剩余部分
    for i in range(1, m):
        for j in range(1, n):
            min_prev = torch.min(torch.stack([
                distance_matrix[:, i-1, j],
                distance_matrix[:, i, j-1],
                distance_matrix[:, i-1, j-1]
            ], dim=0), dim=0)[0]  # 沿第0维取最小值
            
            distance_matrix[:, i, j] = torch.max(min_prev, point_distances[:, i, j])
    
    return distance_matrix[:, m-1, n-1].squeeze()

def calculate_initial_bearing(lon1, lat1, lon2, lat2):
    """
    计算从点1到点2的初始方位角（批量版本，弧度输入输出）
    
    参数:
    lon1, lat1, lon2, lat2: 弧度坐标，支持任意相同形状的张量
    
    返回:
    torch.Tensor: 方位角，弧度，范围 [-π, π]
    """
    dlon = lon2 - lon1
    
    x = torch.sin(dlon) * torch.cos(lat2)
    y = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    
    initial_bearing = torch.atan2(x, y)
    return initial_bearing

def calculate_curvature_lonlat(trajectory):
    """
    计算经纬度轨迹的曲率（批量版本，弧度输入）
    
    参数:
    trajectory: torch.Tensor, shape (batch_size, n, 2), [lon, lat] in radians
    
    返回:
    torch.Tensor: 每个点的曲率值（1/米），shape (batch_size, n)，首尾点设为0
    """
    trajectory = ensure_tensor(trajectory)
    
    # 处理单条轨迹的情况
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    
    batch_size, n, _ = trajectory.shape
    
    # 初始化曲率张量
    curvatures = torch.zeros((batch_size, n), device=trajectory.device)
    
    gap = 15

    # 批量计算所有中间点的曲率
    for i in range(gap, n-gap):
        # 获取三个连续点
        p_prev = trajectory[:, i-gap, :]  # (batch_size, 2)
        p_curr = trajectory[:, i, :]    # (batch_size, 2)
        p_next = trajectory[:, i+gap, :]  # (batch_size, 2)
        
        lon_prev, lat_prev = p_prev[:, 0], p_prev[:, 1]
        lon_curr, lat_curr = p_curr[:, 0], p_curr[:, 1]
        lon_next, lat_next = p_next[:, 0], p_next[:, 1]
        
        # 批量计算两个线段的方位角
        bearing1 = calculate_initial_bearing(lon_prev, lat_prev, lon_curr, lat_curr)
        bearing2 = calculate_initial_bearing(lon_curr, lat_curr, lon_next, lat_next)
        
        # 计算方位角变化量（自动处理弧度环绕）
        bearing_diff = bearing2 - bearing1
        
        # 将角度差规范化到 [-π, π] 范围
        bearing_diff = torch.atan2(torch.sin(bearing_diff), torch.cos(bearing_diff))
        
        # 批量计算两个线段的长度
        dist1 = haversine_distance(lon_prev, lat_prev, lon_curr, lat_curr)
        dist2 = haversine_distance(lon_curr, lat_curr, lon_next, lat_next)
        
        # 使用平均距离作为分母
        avg_distance = (dist1 + dist2) / 2
        
        # 避免除零错误
        mask = avg_distance < 1e-10
        curvatures[:, i] = torch.where(
            mask,
            torch.tensor(0.0, device=trajectory.device),
            bearing_diff / avg_distance
        )
    
    return curvatures.squeeze()

def calculate_curvature_turning_radius(trajectory):
    """
    计算转弯半径（基于三点构成的圆弧，批量版本）
    
    参数:
    trajectory: torch.Tensor, shape (batch_size, n, 2), [lon, lat] in radians
    
    返回:
    torch.Tensor: 转弯半径，单位米，shape (batch_size, n)
    """
    trajectory = ensure_tensor(trajectory)
    
    # 处理单条轨迹的情况
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    
    batch_size, n, _ = trajectory.shape
    
    # 初始化转弯半径张量（无穷大表示直线）
    turning_radii = torch.full((batch_size, n), float('inf'), device=trajectory.device)
    
    for i in range(1, n-1):
        p1 = trajectory[:, i-1, :]  # (batch_size, 2)
        p2 = trajectory[:, i, :]    # (batch_size, 2)
        p3 = trajectory[:, i+1, :]  # (batch_size, 2)
        
        # 批量计算三点间距离
        a = haversine_distance(p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1])  # p1-p2
        b = haversine_distance(p2[:, 0], p2[:, 1], p3[:, 0], p3[:, 1])  # p2-p3
        c = haversine_distance(p1[:, 0], p1[:, 1], p3[:, 0], p3[:, 1])  # p1-p3
        
        # 使用海伦公式计算三角形面积
        s = (a + b + c) / 2
        
        # 检查三角形有效性
        area_sq = s * (s - a) * (s - b) * (s - c)
        valid_mask = area_sq > 1e-20
        
        # 只对有效的三角形计算面积
        area = torch.zeros_like(a)
        area[valid_mask] = torch.sqrt(area_sq[valid_mask])
        
        # 计算转弯半径 R = (a*b*c) / (4*area)
        radius_mask = (area > 1e-10) & valid_mask
        turning_radius = torch.full_like(a, float('inf'))
        turning_radius[radius_mask] = (a[radius_mask] * b[radius_mask] * c[radius_mask]) / (4 * area[radius_mask])
        
        turning_radii[:, i] = turning_radius
    
    return turning_radii.squeeze()

def curvature_to_turning_radius(curvatures):
    """
    将曲率转换为转弯半径（批量版本）
    
    参数:
    curvatures: torch.Tensor, 曲率值，shape (batch_size, n) 或 (n,)
    
    返回:
    torch.Tensor: 转弯半径，单位米
    """
    curvatures = ensure_tensor(curvatures)
    
    # 处理单条轨迹的情况
    if curvatures.dim() == 1:
        curvatures = curvatures.unsqueeze(0)
    
    # 创建掩码避免除零
    mask = torch.abs(curvatures) < 1e-10
    radii = torch.where(mask, 
                       torch.tensor(float('inf'), device=curvatures.device), 
                       1.0 / torch.abs(curvatures))
    
    return radii.squeeze()

def smooth_curvature(curvatures, window_size=5):
    """
    滑动平均平滑曲率数据（批量版本）
    
    参数:
    curvatures: torch.Tensor, 原始曲率，shape (batch_size, n) 或 (n,)
    window_size: 滑动窗口大小
    
    返回:
    torch.Tensor: 平滑后的曲率
    """
    curvatures = ensure_tensor(curvatures)
    
    # 处理单条轨迹的情况
    if curvatures.dim() == 1:
        curvatures = curvatures.unsqueeze(0)
    
    batch_size, seq_len = curvatures.shape
    
    if seq_len < window_size:
        return curvatures.squeeze()
    
    # 使用一维卷积实现滑动平均
    kernel = torch.ones(1, 1, window_size, device=curvatures.device) / window_size
    
    # 对每个batch进行填充和卷积
    padded = torch.nn.functional.pad(
        curvatures.unsqueeze(1), 
        (window_size//2, window_size//2), 
        mode='replicate'
    )  # (batch_size, 1, seq_len + window_size - 1)
    
    # 卷积操作
    smoothed = torch.nn.functional.conv1d(
        padded, 
        kernel, 
        padding=0
    ).squeeze(1)  # (batch_size, seq_len)
    
    return smoothed.squeeze()

def analyze_curvature_features(curvatures, trajectory):
    """
    分析曲率特征（批量版本）
    
    参数:
    curvatures: torch.Tensor, 曲率值，shape (batch_size, n)
    trajectory: torch.Tensor, 轨迹数据，shape (batch_size, n, 2)
    
    返回:
    dict: 曲率特征统计，每个值的shape为 (batch_size,)
    """
    curvatures = ensure_tensor(curvatures)
    trajectory = ensure_tensor(trajectory)
    
    # 处理单条轨迹的情况
    if curvatures.dim() == 1:
        curvatures = curvatures.unsqueeze(0)
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    
    batch_size, n = curvatures.shape
    
    # 过滤有效曲率（排除首尾和接近零的值）
    valid_curvatures = curvatures[:, 1:-1]  # (batch_size, n-2)
    valid_mask = torch.abs(valid_curvatures) > 1e-10
    
    # 初始化结果
    max_curvature = torch.zeros(batch_size, device=curvatures.device)
    mean_curvature = torch.zeros(batch_size, device=curvatures.device)
    std_curvature = torch.zeros(batch_size, device=curvatures.device)
    sharp_turns_count = torch.zeros(batch_size, device=curvatures.device, dtype=torch.long)
    straight_segments = torch.zeros(batch_size, device=curvatures.device, dtype=torch.long)
    
    for i in range(batch_size):
        batch_valid_mask = valid_mask[i]
        batch_valid_curvatures = valid_curvatures[i][batch_valid_mask]
        
        if batch_valid_curvatures.numel() == 0:
            max_curvature[i] = 0.0
            mean_curvature[i] = 0.0
            std_curvature[i] = 0.0
            sharp_turns_count[i] = 0
            straight_segments[i] = n
            continue
        
        max_curvature[i] = torch.max(torch.abs(batch_valid_curvatures))
        mean_curvature[i] = torch.mean(batch_valid_curvatures)
        std_curvature[i] = torch.std(batch_valid_curvatures)
        
        # 统计急转弯数量
        sharp_turn_threshold = 0.001  # 1/1000米
        sharp_turns_count[i] = torch.sum(torch.abs(batch_valid_curvatures) > sharp_turn_threshold)
        
        # 统计直线段数量
        straight_threshold = 0.0001  # 1/10000米
        straight_segments[i] = torch.sum(torch.abs(curvatures[i]) < straight_threshold)
    
    return {
        'max_curvature': max_curvature,
        'mean_curvature': mean_curvature,
        'std_curvature': std_curvature,
        'sharp_turns_count': sharp_turns_count,
        'straight_segments': straight_segments
    }

def curvature_calculation(trajectory, window_size=5):
    """
    计算轨迹的第一个点、中间点和最后一个点的曲率，包含平滑处理（批量版本）
    
    参数:
    trajectory: torch.Tensor, shape (batch_size, 144, 2) 或 (144, 2), 轨迹数据 [lon, lat] in radians
    window_size: 平滑窗口大小
    
    返回:
    dict: 包含'smoothed_curvatures'键，值为三个点的平滑曲率，形状为 (batch_size, 3) 或 (3,)
    """
    trajectory = ensure_tensor(trajectory)
    
    # 处理单条轨迹的情况
    is_single_trajectory = trajectory.dim() == 2
    if is_single_trajectory:
        trajectory = trajectory.unsqueeze(0)  # 增加batch维度
    
    # 验证输入形状
    assert trajectory.dim() == 3, f"Expected trajectory shape (batch_size, 144, 2), got {trajectory.shape}"
    assert trajectory.shape[1:] == (144, 2), f"Expected trajectory shape (batch_size, 144, 2), got {trajectory.shape}"
    
    # 确定三个点的索引
    first_point = 0
    middle_point = 143 // 2  # 中间点索引 (0-based)
    last_point = 143
    selected_indices = [first_point, middle_point, last_point]
    
    # 计算完整轨迹的原始曲率（已支持批量）
    raw_curvatures = calculate_curvature_lonlat(trajectory)
    
    # 平滑处理完整曲率（已支持批量）
    smoothed_curvatures = smooth_curvature(raw_curvatures, window_size)
    
    # 确保smoothed_curvatures是2维的 (batch_size, seq_len)
    if smoothed_curvatures.dim() == 1:
        smoothed_curvatures = smoothed_curvatures.unsqueeze(0)
    
    # 提取三个点的平滑曲率 - 使用索引选择多个位置
    # 对于形状为(batch_size, 144)的smoothed_curvatures，选择[batch, [idx1, idx2, idx3]]
    selected_smooth_curvatures = smoothed_curvatures[:, selected_indices]
    
    # 如果是单条轨迹，去掉batch维度
    if is_single_trajectory:
        selected_smooth_curvatures = selected_smooth_curvatures.squeeze(0)
    
    return selected_smooth_curvatures

# 测试示例
if __name__ == "__main__":
    # 创建测试数据（弧度制）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    batch_size = 128
    n_points = 144
    
    # 生成批量示例轨迹
    trajectories = []
    for b in range(batch_size):
        # 为每个batch生成不同的轨迹
        start_lon = 2.12 + 0.01 * torch.rand(1, device=device).item()
        end_lon = 2.18 + 0.01 * torch.rand(1, device=device).item()
        start_lat = 0.545 + 0.01 * torch.rand(1, device=device).item()
        end_lat = 0.567 + 0.01 * torch.rand(1, device=device).item()
        
        lons_rad = torch.linspace(start_lon, end_lon, n_points, device=device)
        lats_rad = torch.linspace(start_lat, end_lat, n_points, device=device)
        
        # 添加一些弯曲
        for i in range(50, 100):
            lats_rad[i] += 0.001 * torch.sin(torch.tensor((i-50)*0.063, device=device))
        
        trajectory = torch.stack([lons_rad, lats_rad], dim=1)
        trajectories.append(trajectory)
    
    trajectories = torch.stack(trajectories)  # (batch_size, n_points, 2)
    
    print(f"轨迹形状: {trajectories.shape}")
    print(f"轨迹设备: {trajectories.device}")
    
    # 测试批量曲率计算
    result = curvature_calculation(trajectories)
    
    print("\n曲率分析结果 (批量):")
    features = result['overall_features']
    for key, value in features.items():
        if value.dim() == 0:
            print(f"{key}: {value.item():.6f}")
        else:
            print(f"{key}: 形状 {value.shape}, 前5个值: {value[:5].cpu().numpy()}")
    
    # 测试批量弗雷歇距离
    pred_trajectories = trajectories + torch.randn_like(trajectories) * 0.001
    fd = frechet_distance(trajectories, pred_trajectories)
    print(f"\n弗雷歇距离: 形状 {fd.shape}, 前5个值: {fd[:5].cpu().numpy()}")
    
    # 测试批量转弯半径计算
    turning_radii = calculate_curvature_turning_radius(trajectories)
    print(f"\n转弯半径: 形状 {turning_radii.shape}")
    
    # 计算平均转弯半径（排除无穷大）
    finite_mask = turning_radii != float('inf')
    avg_turning_radii = torch.zeros(batch_size, device=device)
    for b in range(batch_size):
        batch_finite = finite_mask[b]
        if batch_finite.any():
            avg_turning_radii[b] = turning_radii[b][batch_finite].mean()
        else:
            avg_turning_radii[b] = float('inf')
    
    print(f"平均转弯半径前5个: {avg_turning_radii[:5].cpu().numpy()}")