import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def save_partition_image(partition_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for idx, partition in enumerate(partition_list):
        pcd = partition.point_cloud.points        # (N, 3)
        cam_center = partition.camera.camera_center()  # (3,)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # 点云
        ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=1, c='gray')

        # 相机
        ax.scatter(cam_center[0], cam_center[1], cam_center[2],
                   s=80, c='red', marker='^', label='Camera')

        ax.set_title(f"Partition {idx}")
        ax.legend()

        save_path = os.path.join(save_dir, f"partition_{idx}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"Saved all partition images to: {save_dir}")


def save_partition_images(partition_list, save_dir, dpi=300):
    """
    为每个 Partition 单独保存一张 XY 平面图：
    - 点云（散点）
    - 相机中心（黑色三角形）
    - origin_box（红色多边形）
    - partition_id（文字）
    """
    os.makedirs(save_dir, exist_ok=True)

    for part in partition_list:
        pid = part.partition_id

        # 点云
        pcd = part.point_cloud.points if hasattr(part.point_cloud, "points") else part.point_cloud
        pts = np.asarray(pcd)

        # 相机中心
        cams = part.camera.camera_center()   # 你保证一定可以这样调用
        cams = np.asarray(cams)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("white")

        # === 点云 ===
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c='blue', alpha=0.7)

        # === 相机 ===
        if cams.size > 0:
            ax.scatter(cams[:, 0], cams[:, 1], s=30, marker='^', color='black')

        # === origin_box ===
        box = part.origin_box
        if box is not None:
            x, y = box.exterior.xy
            ax.plot(x, y, color='red', linewidth=1.5)

            # 标注分区 id
            cx, cy = box.centroid.x, box.centroid.y
            ax.text(cx, cy, pid, fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # === 统一外观 ===
        ax.set_title(f"Partition {pid}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.3)

        # 自动缩放
        if pts.size > 0:
            xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
            ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
            pad_x = (xmax - xmin) * 0.05 + 1e-6
            pad_y = (ymax - ymin) * 0.05 + 1e-6
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(ymin - pad_y, ymax + pad_y)

        # 保存
        save_path = os.path.join(save_dir, f"partition_{pid}.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi)
        plt.close(fig)
        print(f"[OK] saved → {save_path}")

