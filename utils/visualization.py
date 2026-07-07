import folium
import pandas as pd

from folium.plugins import TimestampedGeoJson

def draw_multiship_routes(csv_path, output_html_path):
    df = pd.read_csv(csv_path, parse_dates=['postime'])
    required_cols = ['lat', 'lon', 'postime', 'mmsi', "yt"]
    map_center = [df['lat'].mean(), df['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=6)

    folium.TileLayer('CartoDB positron', name='简洁地图', attr='default').add_to(m)
    folium.TileLayer('Stamen Terrain', name='地形图', attr='default').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='卫星影像'
    ).add_to(m)

    colors = ['blue', 'red', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkpurple']
    all_features = []
    grouped = df.groupby('mmsi')

    for i, (ship_id, group_df) in enumerate(grouped):
        color = [colors[i] for i in group_df["type"]]
        ship_layer = folium.FeatureGroup(name=f"静态路线: {ship_id}")

        group_df = group_df.sort_values(by='postime').reset_index(drop=True)

        points = list(zip(group_df['lat'], group_df['lon']))

        folium.PolyLine(
            locations=points,
            color=[colors[i] for i in group_df["type"]],
            weight=3,
            opacity=0.7
        ).add_to(ship_layer)

        ship_layer.add_to(m)


        point_features = []
        for _, row in group_df.iterrows():
            point_features.append({
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [row['lon'], row['lat']]},
                'properties': {
                    'time': row['postime'].isoformat(),
                    'icon': 'circle',
                    'iconstyle': {'fillColor': color, 'fillOpacity': 0.8, 'stroke': 'true', 'radius': 7},
                    'popup': f"<b>{row.get('mmsi', ship_id)}</b><br>{row['postime'].strftime('%Y-%m-%d %H:%M')}<br>Lat: {row['lat']} <br>Lon: {row['lon']}"
                }
            })


        line_coordinates = list(zip(group_df['lon'], group_df['lat']))
        line_times = [dt.isoformat() for dt in group_df['postime']]
        line_feature = {
            'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': line_coordinates},
            'properties': {'times': line_times, 'style': {'color': color, 'weight': 5}}
        }

        all_features.extend(point_features)
        all_features.append(line_feature)

    if all_features:
        TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': all_features},
            period='PT5M',
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=5,
            loop_button=True,
            time_slider_drag_update=True,
        ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save(output_html_path)

def draw_single_color_traj(df, m, color, feature_group_name):
    """
    在现有地图上绘制单一颜色的航迹
    
    参数:
    df: DataFrame, 包含航迹数据
    m: folium.Map对象
    color: str, 航迹颜色
    feature_group_name: str, 图层名称
    
    返回:
    list, 添加的所有特征
    """
    ship_layer = folium.FeatureGroup(name=feature_group_name)
    
    # 按时间排序
    df = df.sort_values(by='postime').reset_index(drop=True)
    
    # 绘制静态路线
    points = list(zip(df['lat'], df['lon']))
    folium.PolyLine(
        locations=points,
        color=color,
        weight=3,
        opacity=0.7
    ).add_to(ship_layer)
    
    ship_layer.add_to(m)
    
    # 准备时间戳动画数据
    all_features = []
    point_features = []
    
    # 添加点特征
    for _, row in df.iterrows():
        point_features.append({
            'type': 'Feature',
            'geometry': {'type': 'Point', 'coordinates': [row['lon'], row['lat']]},
            'properties': {
                'time': row['postime'].isoformat(),
                'icon': 'circle',
                'iconstyle': {'fillColor': color, 'fillOpacity': 0.8, 'stroke': 'true', 'radius': 7},
                'popup': f"<b>{row.get('mmsi', 'Unknown')}</b><br>{row['postime'].strftime('%Y-%m-%d %H:%M')}<br>Lat: {row['lat']} <br>Lon: {row['lon']}"
            }
        })
    
    # 添加线特征
    line_coordinates = list(zip(df['lon'], df['lat']))
    line_times = [dt.isoformat() for dt in df['postime']]
    line_feature = {
        'type': 'Feature',
        'geometry': {'type': 'LineString', 'coordinates': line_coordinates},
        'properties': {'times': line_times, 'style': {'color': color, 'weight': 5}}
    }
    
    all_features.extend(point_features)
    all_features.append(line_feature)
    
    return all_features

def draw_single_traj_route(df, output_html_path):
    """
    绘制单一航迹，根据type字段显示不同颜色
    
    参数:
    df: DataFrame, 包含航迹数据，必须包含lat、lon、postime、type、mmsi列
    output_html_path: str, 输出的HTML文件路径
    """
    # 验证必需的列
    required_cols = ['lat', 'lon', 'postime', 'type', 'mmsi']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame缺少必需的列: {col}")
    
    # 确保postime是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['postime']):
        df['postime'] = pd.to_datetime(df['postime'])
    
    # 创建地图
    map_center = [df['lat'].mean(), df['lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=10)
    
    # 添加不同的底图图层
    folium.TileLayer('CartoDB positron', name='简洁地图', attr='default').add_to(m)
    folium.TileLayer('Stamen Terrain', name='地形图', attr='default').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri', name='卫星影像'
    ).add_to(m)
    
    # 收集所有特征
    all_features = []
    
    # 处理type为0的航迹（蓝色）- 根据要求，蓝色航迹一定存在
    blue_df = df[df['type'] == 0].copy()
    blue_df = blue_df.sort_values(by='postime').reset_index(drop=True)
    blue_features = draw_single_color_traj(blue_df, m, 'blue', '静态路线: type=0')
    all_features.extend(blue_features)
    
    # 处理type为1的航迹（红色）
    red_df = df[df['type'] == 1].copy()
    red_df = red_df.sort_values(by='postime').reset_index(drop=True)
    
    # 获取蓝色航迹的最后一个点并添加到红色航迹的开头
    blue_last_point = blue_df.iloc[-1:].copy()
    
    # 合并蓝色最后一个点和红色航迹
    red_df = pd.concat([blue_last_point, red_df], ignore_index=True)
    
    red_features = draw_single_color_traj(red_df, m, 'red', '静态路线: type=1（预测值）')
    all_features.extend(red_features)
    
    # 处理type为2的航迹（绿色 - 真实值）
    if 2 in df['type'].values:
        green_df = df[df['type'] == 2].copy()
        green_df = green_df.sort_values(by='postime').reset_index(drop=True)
        
        # 获取蓝色航迹的最后一个点并添加到绿色航迹的开头
        green_df = pd.concat([blue_last_point, green_df], ignore_index=True)
        
        green_features = draw_single_color_traj(green_df, m, 'green', '静态路线: type=2（真实值）')
        all_features.extend(green_features)

    # 添加时间戳动画
    if all_features:
        TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': all_features},
            period='PT5M',
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=5,
            loop_button=True,
            time_slider_drag_update=True,
        ).add_to(m)
    
    # 添加图层控制
    folium.LayerControl().add_to(m)
    
    # 保存地图
    m.save(output_html_path)
    return output_html_path

if __name__ == "__main__":
    csv_file = './ships_routes copy.csv'
    output_html_file = 'multiship_route_map.html'

    # draw_multiship_routes(csv_file, output_html_file)
    
    # 示例：如何使用新函数
    df = pd.read_csv(csv_file, parse_dates=['postime'])
    single_ship_df = df[df['mmsi'] == df['mmsi'].iloc[0]]
    draw_single_traj_route(single_ship_df, 'multiship_route_map.html')