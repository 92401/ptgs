import os
import pickle
from typing import NamedTuple
import numpy as np
from shapely.geometry.polygon import Polygon
import open3d as o3d
import sys

from data_read.create_scene import partition, cameraList_from_camInfos_partition
from partition.partition_run import ProgressiveDataPartitioning

threshold_value = 500000  #分区点云阈值
path = sys.argv[1] if len(sys.argv) > 1 else r"E:\airport_data\test\ychdata"  #数据目录（可用命令行参数指定）
model_path = os.path.join(path, "model")   #存储分区结果目录


class args(NamedTuple):
    scene_path=path
    model_path=model_path
    partition_dir=model_path
    data_device="cpu"

scene_partition=partition(path,None)   #创建sfm场景

train_cameras = cameraList_from_camInfos_partition(scene_partition.train_cameras, args)  #转换camerainfo转换为simple camera

#开始分区函数
DataPartitioning=ProgressiveDataPartitioning(scene_partition, train_cameras, threshold_value,model_path)
