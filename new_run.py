import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion
import os
import math
import re
import json
import sys

class BlenderManager:
    """
    用于管理Blender场景中的物体操作:
      - 导入/导出FBX模型
      - 设置物体变换
      - 处理物体之间的空间关系
      - 更新和保存场景信息
    """
    def __init__(self):
        self.obj_list = {}  # instance_id -> blender object
        self.obj_dimensions = {}  # instance_id -> dimensions
        self.tree_sons = {}  # parent_id -> [child_ids]
        self.processed_matrix = {}  # instance_id -> matrix
        self.CARPET = ["carpet_0", "rug_0"]

    def import_fbx(self, filepath):
        """导入FBX模型"""
        bpy.ops.import_scene.fbx(filepath=filepath)
        return bpy.context.selected_objects[0]

    def set_object_transform(self, obj, transform_matrix):
        """设置物体的变换矩阵"""
        blender_matrix = Matrix([
            transform_matrix[0], 
            transform_matrix[1], 
            transform_matrix[2], 
            transform_matrix[3]
        ])
        obj.matrix_world = blender_matrix

    def align_object_z_to_world_z(self, obj):
        """将物体的Z轴对齐到世界坐标系Z轴"""
        # 获取物体的世界变换矩阵
        world_matrix = obj.matrix_world

        # 提取物体的本地 z 轴在世界坐标系中的方向
        local_z_axis = world_matrix.to_3x3() @ Vector((0, 0, 1))
        local_z_axis.normalize()

        # 计算旋转轴和角度
        rotation_axis = local_z_axis.cross(Vector((0, 0, 1)))
        rotation_angle = local_z_axis.angle(Vector((0, 0, 1)))

        # 如果旋转轴非常小，说明已经对齐或需要 180 度旋转
        if rotation_axis.length < 1e-6:
            if local_z_axis.z > 0:
                return  # 已经对齐，无需操作
            else:
                #  180 度旋转，选择 x 轴
                rotation_axis = Vector((1, 0, 0))
                rotation_angle = 3.14159  # pi

        # 创建旋转四元数
        rotation_quat = Quaternion(rotation_axis, rotation_angle)

        # 创建新的旋转矩阵
        new_rotation = rotation_quat.to_matrix().to_4x4()

        # 保持原始位置
        new_matrix = Matrix.Translation(world_matrix.translation) @ new_rotation @ world_matrix.to_3x3().to_4x4()

        # 应用新的变换矩阵
        obj.matrix_world = new_matrix

    def extract_transform_components(self, matrix):
        """从4x4矩阵中提取变换分量"""
        # 从4x4矩阵中提取位置、旋转和缩放
        
        # 提取缩放
        # 使用矩阵的列向量长度来获取缩放值
        scale_x = matrix.col[0].xyz.length
        scale_y = matrix.col[1].xyz.length
        scale_z = matrix.col[2].xyz.length
        
        return Vector((scale_x, scale_y, scale_z))

    def get_matrix_world(self, obj):
        """获取物体的世界变换矩阵"""
        # 从当前的 matrix_world 中提取变换组件
        scale = self.extract_transform_components(obj.matrix_world)
        
        # Create translation matrix
        translation_matrix = Matrix.Translation(obj.location)
        
        # Create rotation matrix
        rotation_matrix = obj.rotation_euler.to_matrix().to_4x4()
        
        # Create scale matrix
        scale_matrix = Matrix.Scale(scale.x, 4, (1, 0, 0)) @ \
                      Matrix.Scale(scale.y, 4, (0, 1, 0)) @ \
                      Matrix.Scale(scale.z, 4, (0, 0, 1))
        
        # Combine translation, rotation and scale
        combined_matrix = translation_matrix @ rotation_matrix @ scale_matrix
        
        # Apply to object's matrix_world
        return combined_matrix

    def setup_camera(self, name):
        """设置场景相机"""
        bpy.ops.object.camera_add(location=(0, 0, 0))
        camera = bpy.context.object
        camera.name = name
        camera.data.lens = 50
        camera.data.clip_start = 0.1
        camera.data.clip_end = 100
        return camera

    def get_world_bound_box(self, obj):
        """获取物体的世界坐标系包围盒"""
        world_matrix = self.get_matrix_world(obj)
        bbox = [world_matrix @ Vector(corner) for corner in obj.bound_box]
        return bbox

    def process_z(self, faId, obj_list, tree_sons, height):
        """处理物体在Z轴方向上的位置关系"""
        fa_bbox = self.get_world_bound_box(obj_list[faId])
        fa_loc_z = obj_list[faId].location.z
        fa_min_z = min(point.z for point in fa_bbox)
        fa_max_z = max(point.z for point in fa_bbox)
        obj_list[faId].location.z = height + (fa_loc_z - fa_min_z)
        fa_height = fa_max_z - fa_min_z + height
        if faId in tree_sons:
            for son in tree_sons[faId]:
                # Recursively process children
                self.process_z(son, obj_list, tree_sons,fa_height)

    def process_against_wall(self, obj_info, obj_list, obj_dimensions):
        """处理靠墙物体的位置和朝向"""
        for instance_id, obj in obj_list.items():
            info = obj_info[instance_id]
            if info["againstWall"] is not None:
                wall = obj_list[info["againstWall"]]
                if info.get("natural_pose",False) is False:
                    obj.rotation_euler[0] = 0
                    obj.rotation_euler[1] = 0
                
                # Get wall's rotation matrix and calculate normal vector
                wall_rotation = wall.rotation_euler.to_matrix()
                normal_vector = wall_rotation @ Vector((0, 0, 1))
                normal_vector.z = 0  # Project to XY plane
                normal_vector.normalize()
                
                # Calculate angle for obj.rotation_euler[2]
                angle = math.atan2(normal_vector.y, normal_vector.x)
                obj.rotation_euler[2] = angle  + math.pi / 2
                
                # Force update object transform
                
                # Get updated position
                delta_vector = obj.location - wall.location
                delta_vector = delta_vector.project(normal_vector)
                # print("!!!",normal_vector,delta_vector)
                delta_vector.z = 0
                
                # Store original z coordinate
                original_z = obj.location.z
                
                # Update location (only x and y components)
                new_location = obj.location + ((obj_dimensions[instance_id][1] + wall.dimensions[2]) / 2 * normal_vector - delta_vector)
                obj.location.x = new_location.x
                obj.location.y = new_location.y
                obj.location.z = original_z  # Restore original z coordinate
                
                # Update again

    def process_wall(self, wall_id, obj_info, obj_list, ground_name):
        """处理墙体位置"""
        min_penetration = float('0')
        wall = obj_list[wall_id] 
        ground = obj_list[ground_name]
        ground_bbox = self.get_world_bound_box(ground)
        ground_max_z = max(point.z for point in ground_bbox)
        wall_rotation = wall.rotation_euler.to_matrix()
        normal_vector = wall_rotation @ Vector((0, 0, 1))
        
        normal_vector.z = 0  # Project to XY plane
        normal_vector.normalize()
        wall_location = wall.location + ground_max_z * normal_vector
        # print("wall_id",wall.location,wall_location)
        
        for instance_id, obj in obj_list.items():
            if re.match(r"wall_\d+", instance_id):
                continue
            if instance_id == ground_name:
                continue
            
            obj_bbox = self.get_world_bound_box(obj)
            # Calculate penetration for each bbox point
            projections = []
            for point in obj_bbox:
                diff_vector = point - wall_location
                projection = diff_vector.dot(normal_vector)
                projections.append(projection)
            
            min_projection = min(projections)
            min_penetration = min(min_penetration, min_projection)
        
        print(wall_id, min_penetration)
        wall.location += normal_vector * min_penetration

    def process_directly_facing(self, obj_info, obj_list, obj_dimensions):
        """处理物体的朝向关系"""
        for instance_id, obj in obj_list.items():
            info = obj_info[instance_id]
            if info.get("directlyFacing") is not None:
                target_obj = obj_list[info["directlyFacing"]]
                
                direction = target_obj.location - obj.location
                angle = math.atan2(direction.y, direction.x)
                obj.rotation_euler[2] = angle + math.pi / 2

    def main(self, base_dir):
        """主流程"""
        # 读取JSON文件
        obj_placement_info_json_path = f"{base_dir}/placement_info.json"
        with open(obj_placement_info_json_path, 'r') as f:
            obj_placement_info = json.load(f)

        # 模型资产库路径
        base_fbx_path = "./fbx"
        
        # 处理地面
        ground_name = obj_placement_info['reference_obj']
        ground_fbx = os.path.join(base_fbx_path, f'{obj_placement_info["obj_info"][ground_name]["retrieved_asset"]}.fbx')
        ground = self.import_fbx(ground_fbx)
        ground.name = ground_name
        ground.matrix_world = Matrix.Identity(4)

        # 导入并摆放其他物体
        for instance_id, info in obj_placement_info['obj_info'].items():
            # if instance_id == scene_camera_name:
            #     continue  # 跳过
            if info.get("retrieved_asset",None) is None:
                continue
            fbx_name = info['retrieved_asset']
            fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"
            # 导入物体
            obj = self.import_fbx(fbx_path)
            scale = self.extract_transform_components(obj.matrix_world)
            self.obj_dimensions[instance_id] = obj.dimensions * scale
            obj.name = instance_id
            obj.matrix_world = Matrix(info["pose"])
            self.processed_matrix[instance_id] = obj.matrix_world
            self.obj_list[instance_id] = obj
            
            if info['parent'] is not None and info["parent"][0:4] != "wall":
                if info["SpatialRel"] == "on":
                    if info['parent'] not in self.tree_sons:
                        self.tree_sons[info['parent']] = []
                    self.tree_sons[info['parent']].append(instance_id)
            # 如果物体在地面上, 则需要先处理
            if info['parent'] == ground_name or info['parent'] in self.CARPET:
                if info.get("natural_pose",False) is False:
                    obj.rotation_euler[0] = 0
                    obj.rotation_euler[1] = 0

        # 处理空间关系
        self.process_z(ground_name, self.obj_list, self.tree_sons, 0)
        self.process_against_wall(obj_placement_info["obj_info"], self.obj_list, self.obj_dimensions)
        
        # 处理墙体和朝向
        for instance_id, info in obj_placement_info["obj_info"].items():
            if re.match(r"wall_\d+", instance_id):
                self.process_wall(instance_id, obj_placement_info["obj_info"], self.obj_list, ground_name)
        
        self.process_against_wall(obj_placement_info["obj_info"], self.obj_list, self.obj_dimensions)
        self.process_directly_facing(obj_placement_info["obj_info"], self.obj_list, self.obj_dimensions)

        # 更新并保存结果
        for instance_id, obj in self.obj_list.items():
            if instance_id != ground_name:
                obj_placement_info["obj_info"][instance_id]["final_pose"] = [
                    list(row) for row in self.get_matrix_world(obj)
                ]
                world_bbox = [list(point) for point in self.get_world_bound_box(obj)]
                obj_placement_info["obj_info"][instance_id]["bbox"] = world_bbox
                obj_placement_info["obj_info"][instance_id]["length"] = list(self.obj_dimensions[instance_id])

        # 保存更新后的JSON
        with open(f"{base_dir}/placement_info_new.json", 'w') as f:
            json.dump(obj_placement_info, f, indent=2)

        # 更新场景
        bpy.context.view_layer.update()

if __name__ == "__main__":
    # 处理命令行参数
    argv = sys.argv
    try:
        index = argv.index("--") + 1
        base_dir = argv[index]
    except (ValueError, IndexError):
        print("Error: Please provide base_dir after '--' in command line arguments")
        print("Example: blender -b -P new_run.py -- /path/to/base_dir")
        sys.exit(1)

    # 清理场景并运行主流程
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    print("Processing directory:", base_dir)
    manager = BlenderManager()
    manager.main(base_dir)
