import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion
import os
import math
import re
import json
import sys

def import_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath)
    return bpy.context.selected_objects[0]

def set_object_transform(obj, transform_matrix):
    # 将numpy数组转换为Blender的Matrix对象
    blender_matrix = Matrix([transform_matrix[0], transform_matrix[1], transform_matrix[2], transform_matrix[3]])
    
    # 设置对象的世界矩阵
    obj.matrix_world = blender_matrix

def align_object_z_to_world_z(obj):
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

def extract_transform_components(matrix):
    # 从4x4矩阵中提取位置、旋转和缩放
    
    # 提取缩放
    # 使用矩阵的列向量长度来获取缩放值
    scale_x = matrix.col[0].xyz.length
    scale_y = matrix.col[1].xyz.length
    scale_z = matrix.col[2].xyz.length
    
    return Vector((scale_x, scale_y, scale_z))

def get_matrix_world(obj):
    # 从当前的 matrix_world 中提取变换组件
    scale = extract_transform_components(obj.matrix_world)
    
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

def setup_camera(name):
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.name = name
    camera.data.lens = 50
    camera.data.clip_start = 0.1
    camera.data.clip_end = 100
    return camera

def get_world_bound_box(obj):
    world_matrix = get_matrix_world(obj)
    bbox = [world_matrix @ Vector(corner) for corner in obj.bound_box]
    return bbox

def process_z(faId, obj_list, tree_sons,height):
    fa_bbox = get_world_bound_box(obj_list[faId])
    fa_loc_z = obj_list[faId].location.z
    fa_min_z = min(point.z for point in fa_bbox)
    fa_max_z = max(point.z for point in fa_bbox)
    obj_list[faId].location.z = height + (fa_loc_z - fa_min_z)
    fa_height = fa_max_z - fa_min_z + height
    if faId in tree_sons:
        for son in tree_sons[faId]:
            # Recursively process children
            process_z(son, obj_list, tree_sons,fa_height)

def process_against_wall(obj_info, obj_list, obj_dimensions):
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
def process_wall(wall_id, obj_info, obj_list, ground_name):
    min_penetration = float('0')
    wall = obj_list[wall_id]
    wall_rotation = wall.rotation_euler.to_matrix()
    normal_vector = wall_rotation @ Vector((0, 0, 1))
    normal_vector.z = 0  # Project to XY plane
    normal_vector.normalize()
    
    for instance_id, obj in obj_list.items():
        if re.match(r"wall_\d+", instance_id):
            continue
        if instance_id == ground_name:
            continue
            
        obj_bbox = get_world_bound_box(obj)
        # Calculate penetration for each bbox point
        projections = []
        for point in obj_bbox:
            diff_vector = point - wall.location
            projection = diff_vector.dot(normal_vector)
            projections.append(projection)
            
        min_projection = min(projections)
        min_penetration = min(min_penetration, min_projection)
    
    print(wall_id, min_penetration)
    wall.location += normal_vector * min_penetration
def process_directly_facing(obj_info, obj_list, obj_dimensions):
    for instance_id, obj in obj_list.items():
        info = obj_info[instance_id]
        if info.get("directlyFacing") is not None:
            target_obj = obj_list[info["directlyFacing"]]
            
            direction = target_obj.location - obj.location
            angle = math.atan2(direction.y, direction.x)
            obj.rotation_euler[2] = angle + math.pi / 2

CARPET = ["carpet_0","rug_0"]

def main(base_dir):
    # 读取JSON文件
    obj_placement_info_json_path = f"{base_dir}/placement_info.json"
    with open(obj_placement_info_json_path, 'r') as f:
        obj_placement_info = json.load(f)

    # 模型资产库路径
    base_fbx_path = "./fbx"
    
    # 地面作为基准参考
    ground_name = obj_placement_info['reference_obj']
    ground_fbx = os.path.join(base_fbx_path, f'{obj_placement_info["obj_info"][ground_name]["retrieved_asset"]}.fbx')
    ground = import_fbx(ground_fbx)
    ground.name = ground_name
    # 将地面设置为世界坐标系
    ground.matrix_world = Matrix.Identity(4)
    
    # 加入相机
    scene_camera_name = obj_placement_info['scene_camrea_name']
    scene_camera = setup_camera(scene_camera_name)
    scene_camera.matrix_world = Matrix(obj_placement_info["obj_info"][scene_camera_name]["pose"])
    

    # 导入并摆放其他物体
    tree_sons = {}
    processed_matrix = {}
    obj_list = {}
    obj_dimensions = {}
    for instance_id, info in obj_placement_info['obj_info'].items():
        if instance_id == scene_camera_name:
            continue  # 跳过
        fbx_name = info['retrieved_asset']
        fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"
        # 导入物体
        obj = import_fbx(fbx_path)
        obj_dimensions[instance_id] = obj.dimensions
        obj_list[instance_id] = obj
        if info.get("scale",None) is not None:
            for i in range(3):
                obj_dimensions[instance_id][i] *= info["scale"][i]
        obj.name = instance_id
        obj.matrix_world = Matrix(info["pose"])
        processed_matrix[instance_id] = obj.matrix_world
        
        # bbox[instance_id
        if info['parent'] is not None and info["parent"][0:4] != "wall":
            if info["SpatialRel"] == "on":
                if info['parent'] not in tree_sons:
                    tree_sons[info['parent']] = []
                tree_sons[info['parent']].append(instance_id)
        # 如果物体在地面上, 则需要先处理
        if info['parent'] == ground_name or info['parent'] in CARPET:
            if info.get("natural_pose",False) is False:
                obj.rotation_euler[0] = 0
                obj.rotation_euler[1] = 0
            # update_matrix_world(obj)
            # if instance_id == "coffee_table_0":
            #     print("!!!!!!",obj.rotation_euler[0],obj.rotation_euler[1],obj.location)

    process_z(faId=ground_name,obj_list=obj_list,tree_sons=tree_sons,height=0)     # 更新 z 轴
    # for instance_id, info in obj_placement_info['obj_info'].items():
    #     if info["parent"] is not None and not re.match(r"wall_\d+", info["parent"]):
    #         if info["parent"] in obj_list and info["SpatialRel"] == "inside":
    #             obj_list[instance_id].parent = obj_list[info["parent"]]
    process_against_wall(obj_placement_info["obj_info"],obj_list,obj_dimensions)
    for instance_id, info in obj_placement_info["obj_info"].items():
        if re.match(r"wall_\d+", instance_id):
            process_wall(instance_id,obj_placement_info["obj_info"],obj_list,ground_name)
    process_against_wall(obj_placement_info["obj_info"],obj_list,obj_dimensions)
    process_directly_facing(obj_placement_info["obj_info"], obj_list, obj_dimensions)

    for instance_id, obj in obj_list.items():
        if instance_id != ground_name and instance_id != scene_camera_name:
            # Update matrix
            # update_matrix_world(obj)
            obj_placement_info["obj_info"][instance_id]["final_pose"] = [list(row) for row in get_matrix_world(obj)]
            
            # Calculate and update bounding box
            world_bbox = [list(point) for point in get_world_bound_box(obj)]
            obj_placement_info["obj_info"][instance_id]["bbox"] = world_bbox
            obj_placement_info["obj_info"][instance_id]["length"] = list(obj_dimensions[instance_id])

    # print(obj_placement_info)
    # Write updated obj_info back to JSON file
    with open(f"{base_dir}/placement_info_new.json", 'w') as f:
        json.dump(obj_placement_info, f, indent=2)
    # 更新场景
    bpy.context.view_layer.update()

if __name__ == "__main__":
    # Blender 会将自己的参数放在前面，用户的参数在 "--" 之后
    # 查找 "--" 之后的第一个参数作为 base_dir
    argv = sys.argv
    try:
        index = argv.index("--") + 1
        base_dir = argv[index]
    except (ValueError, IndexError):
        print("Error: Please provide base_dir after '--' in command line arguments")
        print("Example: blender -b -P new_run.py -- /path/to/base_dir")
        sys.exit(1)
    
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    print("Processing directory:", base_dir)
    main(base_dir)
