import torch
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Geod
import warnings

# Earth radius (km)
EARTH_RADIUS = 6371.0

def distance_to_coastline(lon, lat, coastline_path="ne_10m_coastline.shp", coastline_data = None):
    """
    计算给定经纬度到最近海岸线的测地线距离（单位：米）
    :return: 距离（米），处理失败时返回None
    """

    if coastline_data is not None:
        coastlines = coastline_data
    else:
        coastlines = gpd.read_file(coastline_path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            target_point = Point(lon, lat)
            
            nearest_idx = coastlines.geometry.distance(target_point).idxmin()
            closest_segment = coastlines.geometry.iloc[nearest_idx]
    
        geod = Geod(ellps="WGS84")
        min_distance = float('inf')
        
        # 处理MultiLineString
        if closest_segment.geom_type == 'MultiLineString':
            segments = list(closest_segment.geoms)
        else:
            segments = [closest_segment]
        
        # 计算点到每个线段的最短距离
        for line in segments:
            coords = np.array(line.coords)
            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i+1]
                
                # 精确计算点到线段的测地线距离
                segment_distance = _geodesic_point_to_segment(geod, lon, lat, p1, p2)
                
                if segment_distance < min_distance:
                    min_distance = segment_distance
        
        return min_distance
    
    except Exception as e:
        print(f"计算错误: {e}")
        return None

def _geodesic_point_to_segment(geod, lon, lat, p1, p2, num_points=20):
    """
    :return: 最小距离（米）
    """
    # 在测地线上插值点
    line = [p1, p2]
    if num_points > 2:
        line = list(geod.npts(p1[0], p1[1], p2[0], p2[1], num_points-2))
        line = [p1] + line + [p2]
    
    # 计算到所有插值点的距离
    min_dist = float('inf')
    for point in line:
        _, _, dist = geod.inv(lon, lat, point[0], point[1])
        if dist < min_dist:
            min_dist = dist
    
    return min_dist

def deg_to_rad(deg_tensor):
    """批量将度数转弧度"""
    return deg_tensor * (math.pi / 180.0)

def rad_to_deg(rad_tensor):
    """批量将弧度转度数"""
    return rad_tensor * (180.0 / math.pi)

def deg_to_vec(deg):
    rad = np.radians(deg)
    unit_vec = np.stack((np.cos(rad), np.sin(rad)), axis=1)
    return unit_vec

def UnitConversion(input, dest_coord, fr='std', to='rad'):
    tensor1 = input.clone()
    if dest_coord != None: tensor2 = dest_coord.clone()

    if fr == 'std' and to == "rad":
        tensor1[:,:,0] *= math.pi/3
        tensor1[:,:,1] *= math.pi
        tensor1[:,:,2:4] *= 25
        if dest_coord != None:
            tensor2[:,:,0] *= math.pi/3
            tensor2[:,:,1] *= math.pi
        # return tensor
    elif fr == "rad" and to == "std":
        tensor1[:,:,0] /= math.pi/3
        tensor1[:,:,1] /= math.pi
        tensor1[:,:,2:4] /= 25
        if dest_coord != None:
            tensor2[:,:,0] /= math.pi/3
            tensor2[:,:,1] /= math.pi
        # return tensor
    elif fr == "std" and to == "deg":
        tensor1[:,:,0] *= 60
        tensor1[:,:,1] *= 180
        tensor1[:,:,2:4] *= 25
        if dest_coord != None:
            tensor2[:,:,0] *= 60
            tensor2[:,:,1] *= 180
        # return tensor
    elif fr == "deg" and to == "std":
        tensor1[:,:,0] /= 60
        tensor1[:,:,1] /= 180
        tensor1[:,:,2:4] /= 25
        if dest_coord != None:
            tensor2[:,:,0] /= 60
            tensor2[:,:,1] /= 180
        # return tensor
    else:
        raise ValueError(f"from {fr} and to {to} should be in [std, rad, degree]")
    if dest_coord != None:
        return tensor1.clone(), tensor2.clone()
    return tensor1.clone(), None

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    批量计算两点间的大圆距离（球面距离）
    输入：纬度/经度张量（形状相同）
    返回：距离张量（千米），形状与输入相同（除了最后一维）
    """
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon/2)**2
    assert 0<=a.all()<=1, "a should between 0 and 1"
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    
    return EARTH_RADIUS * c

def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    """
    批量计算从起点到终点的初始方位角（从北顺时针方向）
    返回：方位角（弧度），形状与输入相同（除了最后一维）
    """
    
    dlon = lon2 - lon1
    x = torch.sin(dlon) * torch.cos(lat2)
    y = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
    
    bearing = torch.atan2(x, y)
    return bearing

def displace_point(lat, lon, bearing, distance):
    """
    批量从起点沿指定方位角移动给定距离，计算终点坐标
    输入：
        lat: 起点纬度（度），形状 [B, N]
        lon: 起点经度（度），形状 [B, N]
        bearing: 方位角（弧度），形状 [B, N]
        distance: 距离（米），形状 [B, N]
    返回：
        new_lat, new_lon: 终点纬度和经度（度），形状 [B, N]
    """
    lat_rad = deg_to_rad(lat)
    lon_rad = deg_to_rad(lon)
    
    angular_dist = distance / EARTH_RADIUS
    
    sin_lat = torch.sin(lat_rad)
    cos_lat = torch.cos(lat_rad)
    sin_angular = torch.sin(angular_dist)
    cos_angular = torch.cos(angular_dist)
    cos_bearing = torch.cos(bearing)
    sin_bearing = torch.sin(bearing)
    
    # 计算新纬度
    new_lat_rad = torch.asin(
        sin_lat * cos_angular +
        cos_lat * sin_angular * cos_bearing
    )
    
    # 计算新经度
    new_lon_rad = lon_rad + torch.atan2(
        sin_bearing * sin_angular * cos_lat,
        cos_angular - sin_lat * torch.sin(new_lat_rad)
    )
    
    # 确保经度在[-180, 180]范围内
    new_lon_rad = (new_lon_rad + math.pi) % (2 * math.pi) - math.pi
    
    return rad_to_deg(new_lat_rad), rad_to_deg(new_lon_rad)

def calculate_resultant_displacement(A, B, C):
    """
    批量计算两个位移向量的严格合成结果
    输入：
        A: 起点张量，形状 [batch_size, num_points, 2] (lat, lon)
        B: 第一个分力终点张量，形状 [batch_size, num_points, 2]
        C: 第二个分力终点张量，形状 [batch_size, num_points, 2]
    返回：
        D: 合成终点张量，形状 [batch_size, num_points, 2]
        resultant_distance: 合成位移距离张量，形状 [batch_size, num_points]
        resultant_bearing: 合成位移方位角张量（度），形状 [batch_size, num_points]
    """
    # 分离纬度和经度
    latA, lonA = A[..., 0], A[..., 1]
    latB, lonB = B[..., 0], B[..., 1]
    latC, lonC = C
    
    # 计算位移向量AB和AC
    bearing_AB = calculate_initial_bearing(latA, lonA, latB, lonB)
    distance_AB = haversine_distance(latA, lonA, latB, lonB)
    # print("bearing_AB:", bearing_AB)
    # print("distance_AB:", distance_AB)
    bearing_AC = calculate_initial_bearing(latA, lonA, latC, lonC)
    distance_AC = haversine_distance(latA, lonA, latC, lonC)
    # print("bearing_AC:", bearing_AC)
    # print("distance_AC:", distance_AC)
    # 在起点切平面内分解向量
    north_AB = distance_AB * torch.cos(bearing_AB)
    east_AB = distance_AB * torch.sin(bearing_AB)
    
    north_AC = distance_AC * torch.cos(bearing_AC)
    east_AC = distance_AC * torch.sin(bearing_AC)
    
    # 向量合成
    north_total = north_AB + north_AC
    east_total = east_AB + east_AC
    
    # 计算合成向量的距离和方位角
    resultant_distance = torch.sqrt(east_total**2 + north_total**2)
    resultant_bearing_rad = torch.atan2(east_total, north_total)
    resultant_bearing_deg = rad_to_deg(resultant_bearing_rad) % 360
    
    # 计算合成终点D
    latD, lonD = displace_point(latA, lonA, resultant_bearing_rad, resultant_distance)
    
    # 组合结果
    D = torch.stack([latD, lonD], dim=-1)
    
    return D, resultant_distance, resultant_bearing_deg

# 示例使用
if __name__ == "__main__":
    batch_size = 16
    num_points = 143
    
    # 创建示例数据 (实际应用中会从模型输入获取)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 起点 (batch_size, num_points, 2)
    A = torch.randn(batch_size, num_points, 2).to(device) * 30 + 40  # 模拟纬度30-70度
    
    # 第一个分力终点
    B = A.clone()
    B[..., 1] += torch.randn_like(B[..., 1]) * 0.1  # 经度偏移
    
    # 第二个分力终点
    C = A.clone()
    C[..., 0] += torch.randn_like(C[..., 0]) * 0.1  # 纬度偏移
    
    # 计算合成位移
    D, dist, bearing = calculate_resultant_displacement(A, B, C)
    
    print(f"输入形状: A={A.shape}, B={B.shape}, C={C.shape}")
    print(f"输出形状: D={D.shape}, dist={dist.shape}, bearing={bearing.shape}")
    print(f"示例结果 - 第一个batch的第一个点:")
    print(f"起点 A: ({A[0,0,0]:.6f}, {A[0,0,1]:.6f})")
    print(f"分力1终点 B: ({B[0,0,0]:.6f}, {B[0,0,1]:.6f})")
    print(f"分力2终点 C: ({C[0,0,0]:.6f}, {C[0,0,1]:.6f})")
    print(f"合成终点 D: ({D[0,0,0]:.6f}, {D[0,0,1]:.6f})")
    print(f"合成位移距离: {dist[0,0]:.2f} 米")
    print(f"合成位移方位角: {bearing[0,0]:.2f}°")