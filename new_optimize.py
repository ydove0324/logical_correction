import argparse
import json
import math
import re

import numpy as np
from geometry_utils import calculate_intersection_area as calc_insec_area
import torch
import trimesh
from pathlib import Path

eps = 1e-3

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

        # 通过 bbox 中心来初始化原始位置和当���位置
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
    """用于管理场景中物体的体素化表示和碰撞检测"""
    
    def __init__(self, resolution=(256, 256, 128)):
        self.resolution = resolution
        self.voxel_grids = {}  # instance_id -> voxel tensor
        self.mesh_cache = {}   # mesh_name -> trimesh.Mesh
        
    def load_mesh(self, mesh_path):
        """加载并缓存mesh"""
        if mesh_path not in self.mesh_cache:
            mesh = trimesh.load(mesh_path)
            self.mesh_cache[mesh_path] = mesh
        return self.mesh_cache[mesh_path]
        
    def voxelize_object(self, mesh_path, pose, scale=None):
        """将物体mesh转换为体素网格"""
        mesh = self.load_mesh(mesh_path)
        if scale is not None:
            mesh = mesh.apply_scale(scale)
            
        # 应用位姿变换
        transform = np.array(pose)
        mesh = mesh.apply_transform(transform)
        
        # 体素化
        voxels = mesh.voxelized(pitch=1.0, method='ray')
        voxel_points = voxels.points
        
        # 转换为张量格式
        grid = torch.zeros(self.resolution, dtype=torch.bool)
        # ... convert points to grid indices and set values ...
        
        return grid

class ObjManager:
    """
    用于管理场景中所有物体以及执行优化流程:
     - 维护 Obj 列表并提供碰撞检测、重叠面积计算、移动距离计算、模拟退火等功能
    """
    def __init__(self):
        self.obj_dict = {}          # instance_id -> Obj
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
        从 placement_info_new.json 中加载数据，创建 Obj 实例并存储
        """
        with open(f"{base_dir}/placement_info_new.json", 'r') as f:
            placement_info = json.load(f)

        self.obj_info = placement_info["obj_info"]
        self.ground_name = placement_info["reference_obj"]

        # 根据 obj_info 构建物体列表(可按需剔除不参与优化的对象)
        for instance_id, info in self.obj_info.items():
            obj = Obj(instance_id, info)
            self.obj_dict[instance_id] = obj

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
          - 使用 bbox 快速预筛选可能发生碰撞的物体对
          - 如果两个物体的 bbox 在 z 轴或 x,y 平面上不可能重叠，就不放入 overlap_list
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

                # 跳过父子关系的物体对
                if (self.obj_dict[instance_id_i].parent_id == instance_id_j or 
                    self.obj_dict[instance_id_j].parent_id == instance_id_i):
                    continue

                # 检查 z 轴方向是否有重叠
                if bbox_i["min"][2] >= bbox_j["max"][2] - eps or bbox_j["min"][2] >= bbox_i["max"][2] - eps:
                    continue

                # 检查 x,y 平面上的距离
                # 考虑到物体可能移动，在原始 bbox 基础上增加一定余量
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
        返回在 bbox_items 列表中的索引, 用于在 overlap_list 中找到相应条目
        """
        for idx, (iid, _) in enumerate(bbox_items):
            if iid == inst_id:
                return idx
        return -1

    def calc_overlap_area(self, debug_mode=False):
        """使用体素化方法计算重叠，只检查预筛选的物体对"""
        total_overlap = 0
        bbox_items = self.build_bbox_items()
        
        # 只对需要检查的物体进行体素化
        for i, overlap_indices in enumerate(self.overlap_list):
            if not overlap_indices:  # 如果该物体没有潜在碰撞对象，跳过
                continue
            
            id_i = bbox_items[i][0]
            if id_i not in self.voxel_manager.voxel_grids:
                obj_i = self.obj_dict[id_i]
                mesh_path = Path(obj_i.mesh_path)
                pose = obj_i.pose_3d
                scale = obj_i.bounding_box.get("scale")
                grid = self.voxel_manager.voxelize_object(mesh_path, pose, scale)
                self.voxel_manager.voxel_grids[id_i] = grid
            
            grid_i = self.voxel_manager.voxel_grids[id_i]
            
            # 只检查预筛选的物体对
            for j in overlap_indices:
                id_j = bbox_items[j][0]
                if id_j not in self.voxel_manager.voxel_grids:
                    obj_j = self.obj_dict[id_j]
                    mesh_path = Path(obj_j.mesh_path)
                    pose = obj_j.pose_3d
                    scale = obj_j.bounding_box.get("scale")
                    grid = self.voxel_manager.voxelize_object(mesh_path, pose, scale)
                    self.voxel_manager.voxel_grids[id_j] = grid
                
                grid_j = self.voxel_manager.voxel_grids[id_j]
                overlap = torch.logical_and(grid_i, grid_j).sum().item()
                total_overlap += overlap
                
                if debug_mode and overlap > 0:
                    print(f"Overlap between {id_i} and {id_j}: {overlap}")
                
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
        计算物体相对于其父物体的越界程度, 若超出某个范围则产生罚分
        """
        k = 3
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

            # 检查是否在父物体 bbox 的某个范围内(这里是简单示例, 可根据需求微调)
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
        在所有可优化物体中随机选择一个物体, 对其 current_pos 进行小幅度扰动.
        返回一个 revert 回调, 用于在模拟退火不接受本次 perturbation 时撤销.
        """
        # 随机选取一个物体
        inst_ids = list(self.obj_dict.keys())
        index = np.random.randint(0, len(inst_ids))
        chosen_id = inst_ids[index]
        chosen_obj = self.obj_dict[chosen_id]

        old_x, old_y = chosen_obj.current_pos

        # 计算扰动大小
        dis = 0.025
        if iteration > 10000:
            dis = 0.005
        perturbation = np.random.normal(0, dis, size=2)

        # 如果该物体靠墙, 则仅在平行于墙的方向扰动
        wall_id = chosen_obj.is_against_wall
        if wall_id is not None and wall_id in self.obj_dict:
            # 获取墙的 pose
            wall_pose = self.obj_info[wall_id]["pose"]  # 4x4
            wall_np = np.array(wall_pose)
            normal_3d = wall_np[:3, :3] @ np.array([0, 0, 1])
            normal_2d = normal_3d[:2]
            normal_len = np.linalg.norm(normal_2d)
            if normal_len > 1e-9:
                normal_2d = normal_2d / normal_len
                # 去除在墙法向上的分量
                dot_val = np.dot(perturbation, normal_2d)
                perturbation = perturbation - dot_val * normal_2d

        # 应用扰动
        chosen_obj.current_pos[0] += perturbation[0]
        chosen_obj.current_pos[1] += perturbation[1]

        # 返回撤销函数
        def revert():
            chosen_obj.current_pos[0] = old_x
            chosen_obj.current_pos[1] = old_y

        return revert

    def simulated_annealing(self, initial_temp, alpha, max_iterations, penalty_factor):
        """
        主优化流程, 不再使用批量 state, 而是直接在 obj 里存 current_pos.
        如果扰动不被接受, 就通过回调方式 revert.
        """
        M = 100  # overlap, constraints 的加权系数
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
        执行主流程:
          1. 读取 placement_info_new.json
          2. 预处理并初始化 overlap
          3. 运行模拟退火
          4. 将结果写回 JSON
        """
        self.load_data(base_dir)
        self.init_overlap()  # 准备 overlap_list

        initial_temp = 100.0
        alpha = 0.99
        max_iterations = 10000
        penalty_factor = 1000.0

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