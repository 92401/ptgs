import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon as MplPolygon

def save_partition_image(partition_list, pcd, path):
    """
    在二维平面上绘制点云、包围盒，且每个包围盒标注 partition_id。

    Args:
        partition_list: list，每个元素有属性:
                        - origin_box (shapely Polygon)
                        - partition_id (int)
        pcd: (N, 2) numpy 数组，二维点云
        path: 保存图片路径
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # ----------- 绘制点云（浅色，提高可视性） -----------
    ax.scatter(
        pcd[:, 0], pcd[:, 1],
        s=1.2, alpha=0.6,
        color='#1f77b4'    # 柔和蓝色，比黑色更易看
    )

    # ----------- 绘制每个 partition 包围盒 + 写 ID -----------
    for part in partition_list:
        poly = part.origin_box
        coords = np.array(poly.exterior.coords)

        # 画 polygon
        patch = MplPolygon(
            coords, closed=True,
            edgecolor='red',
            facecolor='none',
            linewidth=1.2
        )
        ax.add_patch(patch)

        # ---- 计算包围盒中心用于放文字（取 polygon 的 centroid）----
        centroid = poly.centroid
        cx, cy = centroid.x, centroid.y

        # 在中心标注 partition_id
        ax.text(
            cx, cy,
            str(part.partition_id),
            color='red',
            fontsize=10,
            ha='center', va='center',
            bbox=dict(
                facecolor='white',
                edgecolor='none',
                alpha=0.7
            )
        )

    # ----------- 设置坐标范围 -----------
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"图像已保存到: {path}")


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

        # ----------------------- 点云 -----------------------
        pts = np.asarray(part.point_cloud.points)  # shape (N, 2) or (N, 3)
        if pts.ndim == 2 and pts.shape[1] == 3:
            pts = pts[:, :2]  # 转成 2D

        # ----------------------- 相机 -----------------------
        # part.camera 是 camera_list
        camera_list = part.camera

        # 每个 camera 有 camera_center()
        cams = np.array([cam.pose for cam in camera_list])
        if cams.ndim == 2 and cams.shape[1] == 3:
            cams = cams[:, :2]  # 取前两个维度 (x, y)

        # ----------------------- 绘图 -----------------------
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("white")

        # 点云
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c='#1f77b4', alpha=0.7)

        # 相机
        if cams.size > 0:
            ax.scatter(cams[:, 0], cams[:, 1], s=30, marker='^', color='red')

        # 包围盒
        box = part.origin_box
        if box is not None:
            x, y = box.exterior.xy
            ax.plot(x, y, color='red', linewidth=1.5)

            # 标注 partition_id
            cx, cy = box.centroid.x, box.centroid.y
            ax.text(cx, cy, str(pid), fontsize=12,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        # 坐标轴设置
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

