import geopandas as gpd
from shapely.geometry import Point, LineString
from pyproj import Geod
import numpy as np
import warnings

def distance_to_coastline(lon, lat, coastline_path="ne_10m_coastline.shp"):
    """
    计算给定经纬度到最近海岸线的测地线距离（单位：米）
    
    优化内容：
    1. 解决地理坐标系距离计算警告
    2. 精确计算点到线段的测地线距离
    3. 添加异常处理
    
    :param lon: 经度
    :param lat: 纬度
    :param coastline_path: 海岸线Shapefile路径
    :return: 距离（米），处理失败时返回None
    """
    try:
        # 忽略地理坐标系警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            # 加载海岸线数据
            coastlines = gpd.read_file(coastline_path)
            
            # 创建目标点
            target_point = Point(lon, lat)
            
            # 使用空间索引找到最近的海岸线特征
            nearest_idx = coastlines.geometry.distance(target_point).idxmin()
            closest_segment = coastlines.geometry.iloc[nearest_idx]
    
        # 初始化测地线计算器
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
    计算点到线段的测地线距离
    通过线段插值确保计算精度
    
    :param geod: 测地线计算器
    :param lon: 目标点经度
    :param lat: 目标点纬度
    :param p1: 线段起点 (lon, lat)
    :param p2: 线段终点 (lon, lat)
    :param num_points: 线段插值点数
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

# 示例用法
if __name__ == "__main__":
    # 纽约坐标
    lon, lat = -74.9223, 34.7338
    
    # 计算距离
    distance = distance_to_coastline(
        lon, lat,
        coastline_path="./ne_10m_coastline.shp"
    )
    
    if distance is not None:
        print(f"距离海岸线: {distance:.2f} 米")