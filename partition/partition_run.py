import os
import pickle
from typing import NamedTuple
import numpy as np
from shapely.geometry.polygon import Polygon
import open3d as o3d
import sys
from partition.plot_partition import save_partition_image,save_partition_images
from partition.run_def import remove_outliers, tree_partition, expand_partitions, assign_cameras_to_partitions, \
    visibility_based_camera_selection,compute_max_xy_distance


class ProgressiveDataPartitioning:
    # 渐进数据分区
    def __init__(self, scene_info, train_cameras, threshold,model_path): #分区的行数和列数
        self.partition_scene = None
        self.ply=scene_info.ply_path
        self.pcd = scene_info.point_cloud
        self.train_cameras = train_cameras
        self.threshold = threshold
        self.model_path = model_path
        self.partition_dir = os.path.join(model_path, "split_result")# 存放分区结果位置
        self.partition_visible_dir = os.path.join(self.partition_dir, "visible")  #可见性文件夹
        self.save_partition_data_dir = os.path.join(self.model_path, "partition_data.pkl")
        if not os.path.exists(self.partition_visible_dir): os.makedirs(self.partition_visible_dir)  # 创建 可见性相机选择后 点云的文件夹
        self.partitions= self.run_DataPartition()

    def run_DataPartition(self):
        pcd_or = o3d.io.read_point_cloud(self.ply)
            
        #去除噪点
        pcd = remove_outliers(pcd_or)
        #获取包围盒
        points = np.asarray(pcd.points)[:, :2]  # 只取 XY 平面数据
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        bounds = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
        # Step 3: 使用四叉树进行分区
        points3d = np.asarray(pcd.points)[:, :3]
        partitions = tree_partition(points3d, bounds, self.threshold)
        print(len(partitions))
        print(f"所有分区: {partitions}")
        # 可视化分区
        save_partition_image(partitions,self.partition_dir)
        # 拓展分区
        distance=compute_max_xy_distance(self.train_cameras,self.pcd)
        print('扩展距离为',distance)
        # breakpoint()
        expanded_partitions = expand_partitions(partitions, self.pcd,distance)
        #加入相机
        camera_in_expand_partitions = assign_cameras_to_partitions(expanded_partitions, self.train_cameras)
        save_partition_images(camera_in_expand_partitions, self.partition_dir)
        # 筛选扩展相机
        final_partitions = visibility_based_camera_selection(camera_in_expand_partitions, self.partition_visible_dir,self.pcd)
        #save_partition_images(final_partitions,self.partition_dir)
        return final_partitions