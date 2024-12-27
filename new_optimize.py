import argparse
import json
import math
import re

import numpy as np
import torch
import trimesh
from pathlib import Path
import pyassimp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

eps = 1e-3
base_fbx_path = "./fbx"
class Obj:
    """
    用于存储单个物体在优化过程中的关键信息:
      - instance_id: 物体在场景中的唯一标识
      - original_pos: 原始 (x, y) 位置
      - current_pos: 当前 (x, y) 位置 (随优化更新)
      - parent_id: 父物体 ID
      - bounding_box: 包含 min, max, length, theta 等信息的字典
      - is_against_wall: 是否靠墙 (如果有对应的墙ID)
      - relation: 物体与父物体的空间关系: "inside" / "on" / "None" 等
      - pose_3d: 原始或最新的 3D 位姿矩阵 (4x4)
    """
    def __init__(self, instance_id, info):
        self.instance_id = instance_id
        self.parent_id = info.get("parent", None)
        self.is_against_wall = info.get("againstWall", None)
        self.relation = info.get("SpatialRel", None)
        self.pose_3d = info.get("final_pose", None)
        fbx_name = info['retrieved_asset']
        self.fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"
        # 通过 bbox 中心来初始化原始位置和当前位置
        bbox_points = np.array(info["bbox"])
        min_corner = np.min(bbox_points, axis=0)
        max_corner = np.max(bbox_points, axis=0)
        center_x = (min_corner[0] + max_corner[0]) / 2.0
        center_y = (min_corner[1] + max_corner[1]) / 2.0

        self.original_pos = (center_x, center_y)
        self.current_pos = [center_x, center_y]

        # 物体的 bounding_box 信息: 包括 min, max, length, theta 等
        length = max_corner - min_corner
        theta = 0.0
        if self.pose_3d is not None:
            # 根据 4x4 矩阵, 提取 z 轴旋转角度
            theta = math.atan2(self.pose_3d[1][0], self.pose_3d[0][0])  
        self.bounding_box = {
            "length": [length[0], length[1]],  # 只使用 x, y 长度
            "min": [float(min_corner[0]), float(min_corner[1]), float(min_corner[2])],
            "max": [float(max_corner[0]), float(max_corner[1]), float(max_corner[2])],
            "theta": float(theta),
            # 记录初始中心位置 (x, y)，用于后续计算移动距离
            "x": float(center_x),
            "y": float(center_y),
        }


class VoxelManager:
    """用于管理场景中��示和碰撞检测"""
    
    def __init__(self, resolution=(128, 128, 128)):
        self.resolution = resolution
        self.voxel_grids = {}  # instance_id -> voxel tensor
        self.mesh_cache = {}   # mesh_name -> trimesh.Mesh
        self.scene_bounds = {
            'min': [float('inf'), float('inf'), float('inf')],
            'max': [-float('inf'), -float('inf'), -float('inf')]
        }
        self.voxel_size = None
        self.scene_initialized = False
        
    def initialize_scene_bounds(self, obj_dict, wall_dict):
        """预先计算整个场景的边界，并确定墙面的边界约束"""
        if self.scene_initialized:
            return
        
        # 标准方向向量
        standard_directions = {
            'left': np.array([1, 0, 0]),    # 左边的墙正方向指向右边
            'right': np.array([-1, 0, 0]),
            'front': np.array([0, -1, 0]),
            'back': np.array([0, 1, 0])
        }
        
        # 初始化墙面约束字典
        self.wall_constraints = {}
        
        # 处理墙面
        for wall_id, wall_info in wall_dict.items():
            if not wall_id.startswith('wall'):
                continue
            
            # 获取墙面的pose矩阵
            wall_pose = np.array(wall_info['pose'])
            # 计算墙���的法向量
            wall_normal = wall_pose[:3, :3] @ np.array([0, 0, 1])
            wall_normal = wall_normal / np.linalg.norm(wall_normal)
            
            # 将原点变换到墙面坐标系
            wall_point = wall_pose[:3, 3]  # 获取平移部分
            
            # 找到最匹配的标准方向
            wall_type = max(standard_directions.items(), 
                          key=lambda x: np.dot(wall_normal, x[1]))[0]
            
            # 根据墙面类型更新对应的边界约束
            if wall_type == 'left':
                self.wall_constraints['left'] = wall_point[0]
            elif wall_type == 'right':
                self.wall_constraints['right'] = wall_point[0]
            elif wall_type == 'front':
                self.wall_constraints['front'] = wall_point[1]
            elif wall_type == 'back':
                self.wall_constraints['back'] = wall_point[1]
            
            print(f"Wall {wall_id} classified as {wall_type} with constraint value {wall_point}")
        
        print("Final wall constraints:", self.wall_constraints)
        
        # 原有的场景边界计算逻辑
        for inst_id, obj in obj_dict.items():
            mesh = self.load_mesh(obj.fbx_path)
            if obj.pose_3d is not None:
                mesh = mesh.copy()
                mesh = mesh.apply_transform(obj.pose_3d)
            
            bounds = mesh.bounds
            self.scene_bounds['min'] = np.minimum(self.scene_bounds['min'], bounds[0])
            self.scene_bounds['max'] = np.maximum(self.scene_bounds['max'], bounds[1])
        
        SCENE_MARGIN_FACTOR = 0.25
        original_scene_size = (np.array(self.scene_bounds['max']) - np.array(self.scene_bounds['min']))
        # 使用墙面约束更新场景边界
        if 'left' in self.wall_constraints:
            self.scene_bounds['min'][0] = min(self.wall_constraints['left'],self.scene_bounds['min'][0])
        else:
            self.scene_bounds['min'][0] -= SCENE_MARGIN_FACTOR * original_scene_size[0]

        if 'right' in self.wall_constraints:
            self.scene_bounds['max'][0] = max(self.wall_constraints['right'],self.scene_bounds['max'][0])
        else:
            self.scene_bounds['max'][0] += SCENE_MARGIN_FACTOR * original_scene_size[0]

        if 'back' in self.wall_constraints:
            self.scene_bounds['min'][1] = min(self.wall_constraints['back'],self.scene_bounds['min'][1])
        else:
            print("before",self.scene_bounds['min'][1])
            self.scene_bounds['min'][1] -= SCENE_MARGIN_FACTOR * original_scene_size[1]
            print("after",self.scene_bounds['min'][1])

        if 'front' in self.wall_constraints:
            self.scene_bounds['max'][1] = max(self.wall_constraints['front'],self.scene_bounds['max'][1])
        else:
            self.scene_bounds['max'][1] += SCENE_MARGIN_FACTOR * original_scene_size[1]
        
        scene_size = (np.array(self.scene_bounds['max']) - np.array(self.scene_bounds['min']))
        self.voxel_size = scene_size / np.array(self.resolution)
        self.voxel_size = np.array([min(self.voxel_size)] * 3)
        self.resolution = (int(round(scene_size[0] / self.voxel_size[0])),
                           int(round(scene_size[1] / self.voxel_size[1])),
                           int(round(scene_size[2] / self.voxel_size[2])))
        self.scene_initialized = True
        
        print("Scene bounds:", self.scene_bounds)
        print("Voxel size:", self.voxel_size)

    def point_to_voxel(self, point):
        """将3D点转换为体素坐标"""
        relative_pos = point - np.array(self.scene_bounds['min'])
        voxel_coords = (relative_pos / self.voxel_size).astype(int)
        return np.clip(voxel_coords, 0, np.array(self.resolution) - 1)

    def fbx2mesh(self, fbx_path):
        # 加载场景
        with pyassimp.load(str(fbx_path)) as scene:
            # 获取第一个mesh
            mesh = scene.meshes[0]
            
            # 转���为 trimesh
            vertices = np.array(mesh.vertices)
            faces = np.array(mesh.faces)
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    def load_mesh(self, fbx_path):
        """加载并缓存mesh"""
        if fbx_path not in self.mesh_cache:
            mesh = self.fbx2mesh(fbx_path)
            # self.mesh_cache[fbx_path] = mesh
        # return self.mesh_cache[fbx_path]
        return mesh
    def approximate_as_box_if_thin(self, mesh: trimesh.Trimesh, pitch: float) -> trimesh.Trimesh:
        """
        如果网格在某个维度极其薄，则其近似为一个长方体，最小厚度为 pitch，其他两维保持原始 bounding box 大小。
        """
        # 取网格的最小/最大边界
        min_corner, max_corner = mesh.bounds
        size = max_corner - min_corner  # (dx, dy, dz)

        # 找到最小的维度
        i_min = np.argmin(size)
        # 如果该维度小于 pitch，则将该维度强制设置为 pitch
        if size[i_min] < pitch:
            center = (max_corner + min_corner) / 2.0
            half_size = size / 2.0
            # 只修改最小的那个维度，其他不变
            half_size[i_min] = pitch / 2.0

            # 用新的 three-half-size 构建一个长方体，并放到同样的中心位置
            box = trimesh.creation.box(extents=2.0 * half_size)
            box.apply_translation(center)
            return box
        else:
            # 如果没有极薄维度，则保持原始网格
            return mesh

    def voxelize_object(self, mesh_path, instance_id, pose, scale=None):
        """
        将物体mesh转换为体素网格，使用更密集采样来确保体素化的连续性。
        """
        if not self.scene_initialized:
            raise RuntimeError("Scene bounds not initialized. Call initialize_scene_bounds first.")

        # 1. 加载（可选）缩放网格
        mesh = self.load_mesh(mesh_path)
        if scale is not None:
            mesh.apply_scale(scale)

        # 2. 应用姿态变换
        transform = np.array(pose)
        mesh = mesh.apply_transform(transform)

        # 3. 获体素大小(以最小值为基准)
        pitch = float(min(self.voxel_size))

        # 4. 若某个维度过薄，则近似为长方体
        mesh = self.approximate_as_box_if_thin(mesh, pitch)

        # 5. 使用更密集的采样进行体素化
        voxels = mesh.voxelized(pitch=pitch, method='subdivide')
        voxels = voxels.fill()

        # 后续将体素坐标映射到场景网格
        voxel_points = torch.from_numpy(voxels.points).float().cuda()
        relative_pos = voxel_points - torch.tensor(self.scene_bounds['min'], device='cuda')
        voxel_coords = (relative_pos / torch.tensor(self.voxel_size, device='cuda')).long()
        
        # 创建带padding的网格（每个维度前后各加5个单位）
        grid = torch.zeros(self.resolution, dtype=torch.bool, device='cuda', requires_grad=False)
        
        # 确保所有坐标在有效范围内
        voxel_coords = torch.clamp(
            voxel_coords,
            torch.tensor(0, device='cuda'),
            torch.tensor(self.resolution, device='cuda') - 1
        )

        grid.index_put_(
            (voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]),
            torch.ones(len(voxel_coords), dtype=torch.bool, device='cuda'),
            accumulate=True
        )

        # 缓���并返回最终体素网格
        self.voxel_grids[instance_id] = grid
        return grid
        
    def move_grid(self, instance_id, offset):
        """移动体素网格，如果移动会超出边界则返回 False"""
        dx, dy, dz = [int(round(o)) for o in offset]
        grid = self.voxel_grids[instance_id]
        
        # 首先检查移动量是否超出网格大小
        if (abs(dx) >= grid.shape[0] or abs(dy) >= grid.shape[1] or abs(dz) >= grid.shape[2]):
            return False
        
        # 检查移动是否会导致内容超出边界
        if dx > 0:  # 右移，检查右边界
            if torch.any(grid[grid.shape[0]-dx:]):  # 最右边dx个格子不能有内容
                return False
        elif dx < 0:  # 左移，检查左边界
            if torch.any(grid[:abs(dx)]):  # 最左边|dx|个格子不能有内容
                return False
            
        if dy > 0:  # 上移，检查上边界
            if torch.any(grid[:, grid.shape[1]-dy:]):
                return False
        elif dy < 0:  # 下移，检查下边界
            if torch.any(grid[:, :abs(dy)]):
                return False
            
        if dz > 0:  # 前移，检查前边界
            if torch.any(grid[:, :, grid.shape[2]-dz:]):
                return False
        elif dz < 0:  # ��移，检查后边界
            if torch.any(grid[:, :, :abs(dz)]):
                return False
        
        # 使用移位操作
        if dx > 0:
            grid = torch.cat([torch.zeros_like(grid[:dx]), grid[:-dx]], dim=0)
        elif dx < 0:
            grid = torch.cat([grid[-dx:], torch.zeros_like(grid[:-dx])], dim=0)
        
        if dy > 0:
            grid = torch.cat([torch.zeros_like(grid[:, :dy]), grid[:, :-dy]], dim=1)
        elif dy < 0:
            grid = torch.cat([grid[:, -dy:], torch.zeros_like(grid[:, :-dy])], dim=1)
        
        if dz > 0:
            grid = torch.cat([torch.zeros_like(grid[:, :, :dz]), grid[:, :, :-dz]], dim=2)
        elif dz < 0:
            grid = torch.cat([grid[:, :, -dz:], torch.zeros_like(grid[:, :, :-dz])], dim=2)
        
        self.voxel_grids[instance_id] = grid
        return True

    def world_to_voxel_offset(self, world_offset):
        """将世界坐标系的偏移转换为体坐标系的偏移"""
        return world_offset / self.voxel_size

    def visualize_voxels(self, instance_ids=None, show_all=False):
        """
        可视化体素网格
        Args:
            instance_ids: 指定要可视化的物体ID列表，如果为None且show_all=True则显示所有物体
            show_all: 是否显示所有物体的体素
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if show_all:
            # 为不同物体随机分配颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(self.voxel_grids)))
            for idx, (obj_id, grid) in enumerate(self.voxel_grids.items()):
                occupied = grid.cpu().numpy()
                x, y, z = np.where(occupied)
                ax.scatter(x, y, z, c=[colors[idx]], alpha=0.6, label=obj_id)
        elif instance_ids is not None:
            # 确保 instance_ids 是列表
            if isinstance(instance_ids, str):
                instance_ids = [instance_ids]
            
            # 指定的物体分配颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, len(instance_ids)))
            for idx, obj_id in enumerate(instance_ids):
                if obj_id in self.voxel_grids:
                    occupied = self.voxel_grids[obj_id].cpu().numpy()
                    x, y, z = np.where(occupied)
                    ax.scatter(x, y, z, c=[colors[idx]], alpha=0.6, label=obj_id)
                else:
                    print(f"Warning: {obj_id} not found in voxel grids")
        else:
            print("No valid instance_ids provided and show_all=False")
            return
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if show_all or (instance_ids and len(instance_ids) > 1):
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('Voxelized objects')
        plt.tight_layout()
        plt.savefig(f"voxel_visualization_{instance_ids}.png")


class ObjManager:
    """
    用于管理场景中所有物体以及执行优化程:
     - 维护 Obj 列表并提供碰撞检测、重叠面积计算、移动距离计算、模拟退火等功能
    """
    def __init__(self):
        self.obj_dict = {}          # instance_id -> Obj
        self.wall_dict = {}         # wall_id -> Obj
        self.overlap_list = []      # 用于预先存储物体间可能发生重叠的对
        self.initial_state = False  # 记录 overlap_list 是否初始化
        self.total_size = 0         # overlap_list 的大小
        self.carpet_list = ["carpet_0", "rug_0"]
        self.n = 0                  # 物体总数
        self.obj_info = {}          # 存储从 placement_info_new.json 中读取的 obj_info
        self.ground_name = None     # reference_obj
        self.voxel_manager = VoxelManager()


    def load_data(self, base_dir):
        """
        从 placement_info_new.json 中加载数据，创建 Obj 实例并存储,
        ���初始化 voxel_manager
        """
        with open(f"{base_dir}/placement_info_new.json", 'r') as f:
            placement_info = json.load(f)

        self.obj_info = placement_info["obj_info"]
        self.ground_name = placement_info["reference_obj"]

        # 加正则表达式模式
        skip_pattern = re.compile(r'^(floor_\d+|wall_\d+|scene_camera)')
        
        for instance_id, info in self.obj_info.items():
            # 如果物体名称匹配模式则跳过
            if skip_pattern.match(instance_id):
                print(f"Skipping {instance_id}")
                self.wall_dict[instance_id] = {"pose": info["pose"]}
                continue
                
            obj = Obj(instance_id, info)
            self.obj_dict[instance_id] = obj

        self.voxel_manager.initialize_scene_bounds(self.obj_dict,self.wall_dict)
        for instance_id, obj in self.obj_dict.items():
            print(obj.fbx_path,instance_id)
            mesh_path = Path(obj.fbx_path)
            pose = obj.pose_3d
            self.voxel_manager.voxelize_object(mesh_path, instance_id,pose,scale=[1.1,1.1,1.0])

    def build_bbox_items(self):
        """
        构建 bbox_items 列表，用于后续初始化 overlap list
        [(instance_id, bounding_box), ...]
        """
        bbox_items = []
        for inst_id, obj in self.obj_dict.items():
            bbox_items.append((inst_id, obj.bounding_box))
        return bbox_items

    def init_overlap(self):
        """
        初始化 overlap_list，用于加速频繁的重叠检测:
          - 使用 bbox 快速预筛选能发生碰撞的物体对
          - 如果两个物体的 bbox  z 轴或 x,y 平面上不可能重叠，就不放入 overlap_list
        """
        if self.initial_state:
            return

        bbox_items = self.build_bbox_items()
        self.n = len(bbox_items)
        self.overlap_list = []
        self.total_size = 0

        for i in range(self.n):
            instance_id_i, bbox_i = bbox_items[i]
            overlap_i = []
            
            # 跳过地毯等特殊物体
            if instance_id_i in self.carpet_list:
                self.overlap_list.append([])
                continue

            for j in range(i + 1, self.n):
                instance_id_j, bbox_j = bbox_items[j]
                
                # 跳过地毯
                if instance_id_j in self.carpet_list:
                    continue

                # 跳过父子关系的体对
                if (self.obj_dict[instance_id_i].parent_id == instance_id_j or 
                    self.obj_dict[instance_id_j].parent_id == instance_id_i):
                    continue

                # 检查 z 轴方向是否重叠
                if bbox_i["min"][2] >= bbox_j["max"][2] - eps or bbox_j["min"][2] >= bbox_i["max"][2] - eps:
                    continue

                # 检查 x,y 平面上的距离
                # 考虑到物体可能动，在原始 bbox 基础上增加一定余量
                margin_x = (bbox_j["length"][0] + bbox_i["length"][0]) * 0.5
                margin_y = (bbox_j["length"][1] + bbox_i["length"][1]) * 0.5
                
                if (bbox_i["min"][0] >= bbox_j["max"][0] + margin_x or 
                    bbox_j["min"][0] >= bbox_i["max"][0] + margin_x):
                    continue
                    
                if (bbox_i["min"][1] >= bbox_j["max"][1] + margin_y or 
                    bbox_j["min"][1] >= bbox_i["max"][1] + margin_y):
                    continue

                # 将可能发生碰撞的物体对添加到列表
                overlap_i.append(j)

            self.overlap_list.append(overlap_i)
            self.total_size += len(overlap_i)

        self.initial_state = True

    def get_obj_index(self, inst_id, bbox_items):
        """
        返回在 bbox_items 列表中的索引, 用于在 overlap_list 中找到相目
        """
        for idx, (iid, _) in enumerate(bbox_items):
            if iid == inst_id:
                return idx
        return -1

    def calc_overlap_area(self, debug_mode=False, batch_size=4):
        """使用体素化方法批量计算重叠，只检查预筛选的物体对"""
        total_overlap = 0
        bbox_items = self.build_bbox_items()
        
        # 收集所有需要检查的物体对
        pairs_to_check = []
        for i, overlap_indices in enumerate(self.overlap_list):
            if not overlap_indices:
                continue
            id_i = bbox_items[i][0]
            for j in overlap_indices:
                id_j = bbox_items[j][0]
                pairs_to_check.append((id_i, id_j))
        
        # 批量处理物体对
        for start_idx in range(0, len(pairs_to_check), batch_size):
            batch_pairs = pairs_to_check[start_idx:start_idx + batch_size]
            
            # 准备这个批次的网格
            grids_1 = []
            grids_2 = []
            for id_1, id_2 in batch_pairs:
                grids_1.append(self.voxel_manager.voxel_grids[id_1])
                grids_2.append(self.voxel_manager.voxel_grids[id_2])
            
            # 将网格堆叠成批次
            batch_grids_1 = torch.stack(grids_1)  # [batch_size, *grid_shape]
            batch_grids_2 = torch.stack(grids_2)  # [batch_size, *grid_shape]
            
            # 批量计算重叠
            batch_overlap = torch.logical_and(batch_grids_1, batch_grids_2).sum(dim=(1,2,3))
            total_overlap += batch_overlap.sum().item()
            
            if debug_mode:
                # 输出这个批次中有重叠的物体对
                for idx, (id_1, id_2) in enumerate(batch_pairs):
                    overlap = batch_overlap[idx].item()
                    if overlap > 0:
                        print(f"Overlap between {id_1} and {id_2}: {overlap}")
        
        return total_overlap

    def calc_movement(self):
        """
        计算所有物体的移动距离(平方和)
        """
        total_move = 0
        for inst_id, obj in self.obj_dict.items():
            ox, oy = obj.original_pos
            cx, cy = obj.current_pos
            total_move += (cx - ox)**2 + (cy - oy)**2
        return total_move

    def calc_constraints(self):
        """
        计算物体相对于其父物体的越界程度, 若出某个范围则产生罚分
        """
        k = 2
        cost = 0
        bbox_items = self.build_bbox_items()

        for inst_id, obj in self.obj_dict.items():
            parent_id = obj.parent_id
            if parent_id is None or parent_id not in self.obj_dict:
                continue
            if obj.relation == "inside":
                # 内部关系暂时忽略
                continue

            fa_obj = self.obj_dict[parent_id]
            fa_bbox = fa_obj.bounding_box  # 父物体的 bbox

            # 当前物体位置
            cx, cy = obj.current_pos
            length_x, length_y = obj.bounding_box["length"][0], obj.bounding_box["length"][1]

            # 检查是否在父物体 bbox 的某个范围内(这里是简单示例, ��据需要微调)
            if (cx - length_x/k >= fa_bbox["min"][0] and
                cx + length_x/k <= fa_bbox["max"][0] and
                cy - length_y/k >= fa_bbox["min"][1] and
                cy + length_y/k <= fa_bbox["max"][1]):
                continue

            # 计算与父物体 bbox 中心的距离并累计
            cost += (cx - fa_bbox["x"])**2 + (cy - fa_bbox["y"])**2

        return cost

    def try_perturb_random_obj(self, iteration):
        """
        在有可优化物体中随机选个物体, 对其 current_pos 行一个体素大小的扰动.
        随机选择 x 或 y 方向, 移动一个体素的距离 (正向或负向).
        返回一个 revert 回调, 用于在模拟退火不接受本次 perturbation 时撤销.
        """
        # 随机选取一个物体
        inst_ids = list(self.obj_dict.keys())
        index = np.random.randint(0, len(inst_ids))
        chosen_id = inst_ids[index]
        chosen_obj = self.obj_dict[chosen_id]

        old_x, old_y = chosen_obj.current_pos

        # 随机选择移动方向 (x或y) 和正负
        move_x = np.random.choice([True, False])  # True为x方向，False为y方向
        move_positive = np.random.choice([True, False])  # True为正向，False为负向
        
        # 获取体素大小
        voxel_size = self.voxel_manager.voxel_size
        perturbation = np.zeros(2)
        voxel_perturbation = np.zeros(3)
        
        # 添加随机整数 scale 因子 (1~5)
        scale = np.random.randint(1, 6)  # randint(1, 6) 会生成 1,2,3,4,5
        
        if move_x:
            perturbation[0] = scale * voxel_size[0] if move_positive else -scale * voxel_size[0]
        else:
            perturbation[1] = scale * voxel_size[1] if move_positive else -scale * voxel_size[1]

        # 如果该物体靠墙, 则仅在平行于墙的方向扰动
        wall_id = chosen_obj.is_against_wall
        if wall_id is not None:
            # 获取墙的 pose
            wall_pose = self.wall_dict[wall_id]["pose"]
            wall_np = np.array(wall_pose)
            normal_3d = wall_np[:3, :3] @ np.array([0, 0, 1])
            normal_2d = normal_3d[:2]
            normal_len = np.linalg.norm(normal_2d)
            if normal_len > 1e-9:
                normal_2d = normal_2d / normal_len
                # 去除在法向上的分量
                dot_val = np.dot(perturbation, normal_2d)
                perturbation = perturbation - dot_val * normal_2d

        # 应用扰动
        voxel_perturbation = np.array([int(round(perturbation[0] / voxel_size[0])), int(round(perturbation[1] / voxel_size[1])), 0])
        move_success = self.voxel_manager.move_grid(chosen_obj.instance_id, voxel_perturbation)
        if not move_success:
            return lambda: None
        chosen_obj.current_pos[0] += perturbation[0]
        chosen_obj.current_pos[1] += perturbation[1]

        # 返回撤销函数
        def revert():
            chosen_obj.current_pos[0] = old_x
            chosen_obj.current_pos[1] = old_y
            self.voxel_manager.move_grid(chosen_obj.instance_id, -voxel_perturbation)

        return revert

    def simulated_annealing(self, initial_temp, alpha, max_iterations, penalty_factor):
        """
        主优化流程, 不再使用批量 state, 而是直接在 obj 里 current_pos.
        如果扰动不被接受, 就通过回调方式 revert.
        """
        M = 100  # overlap, constraints 的加权系数
        # print(self.calc_overlap_area())
        # self.voxel_manager.visualize_voxels(instance_ids=["guitar_1","tall_wardrobe_0","guitar_0"])
        # self.voxel_manager.move_grid("guitar_1", [0,-15,0])
        # print(self.calc_overlap_area())
        # self.voxel_manager.visualize_voxels(instance_ids=["guitar_1","tall_wardrobe_0","guitar_0"])
        # exit(0)
        current_energy = ( M*( self.calc_overlap_area() + self.calc_constraints() ) 
                           + self.calc_movement() )
        temperature = initial_temp

        for iteration in range(max_iterations):
            # 扰动某个物体, 并拿到 revert 回调
            revert_callback = self.try_perturb_random_obj(iteration)

            # 计算新能量
            new_energy = ( M*( self.calc_overlap_area() + self.calc_constraints() ) 
                           + self.calc_movement() )

            print(iteration, new_energy)

            delta_energy = new_energy - current_energy
            # Metropolis 准则
            if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
                # 接受新状态
                current_energy = new_energy
            else:
                # 拒绝, revert
                revert_callback()

            temperature *= alpha

            # 判断是否可以提前结束
            if temperature < 1e-3 and current_energy == 0:
                break

        return current_energy

    def save_to_json(self, file_path, data):
        """
        将数据存储到 JSON 文件
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def main(self, base_dir):
        """
        执行流程:
          1. 读取 placement_info_new.json
          2. 预处理并初始化 overlap
          3. 运行模拟退火
          4. 将结果写回 JSON
        """
        self.load_data(base_dir)
        self.init_overlap()  # 准备 overlap_list1
        initial_temp = 100.0
        alpha = 0.99
        max_iterations = 10000
        penalty_factor = 1000.0
        self.voxel_manager.visualize_voxels(instance_ids=["guitar_1","tall_wardrobe_0","guitar_0"])

        # 添加可视化调用
        print("Visualizing initial voxel grids...")
        self.voxel_manager.visualize_voxels(show_all=True)

        final_energy = self.simulated_annealing(initial_temp, alpha, max_iterations, penalty_factor)

        # 将最终位置写进 final_pos.json
        final_position = {}
        for inst_id, obj in self.obj_dict.items():
            final_position[inst_id] = {
                "x": float(obj.current_pos[0]),
                "y": float(obj.current_pos[1])
            }
            moved_dist = math.sqrt(
                (obj.current_pos[0] - obj.original_pos[0])**2 + 
                (obj.current_pos[1] - obj.original_pos[1])**2
            )
            print(inst_id, "移动距离:", moved_dist)
        self.voxel_manager.visualize_voxels(instance_ids=["guitar_1","tall_wardrobe_0","guitar_0"])

        # 添加可视化调用
        print("Visualizing initial voxel grids...")
        self.voxel_manager.visualize_voxels(show_all=True)
        print("Final Overlap:", self.calc_overlap_area(debug_mode=True))
        print("Final Constraints:", self.calc_constraints())
        print("Final Energy:", final_energy)

        self.save_to_json(f"{base_dir}/final_pos.json", final_position)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimize object placement in a room')
    parser.add_argument('base_dir', type=str, help='Base directory containing placement_info_new.json')
    args = parser.parse_args()

    manager = ObjManager()
    manager.main(args.base_dir)