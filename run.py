import bpy
import sys
# sys.path.append("/home/ubuntu/.local/lib/python3.10/site-packages")
# import numpy as np
from mathutils import Matrix, Vector
import math
import json


# import cvxpy as cp

def import_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath)
    return bpy.context.selected_objects[0]

def set_object_transform(obj, transform_matrix):
    # 将numpy数组转换为Blender的Matrix对象
    blender_matrix = Matrix([transform_matrix[0], transform_matrix[1], transform_matrix[2], transform_matrix[3]])
    
    # 设置对象的世界矩阵
    obj.matrix_world = blender_matrix

def calculate_bounding_box(obj):
    # 获取物体的边界框顶点在世界坐标系中的坐标
    translation_matrix = Matrix.Translation(obj.location)

    # Create rotation matrix
    rotation_matrix = obj.rotation_euler.to_matrix().to_4x4()

    # Combine translation and rotation
    combined_matrix = translation_matrix @ rotation_matrix
    bbox_corners = [combined_matrix @ Vector(corner) for corner in obj.bound_box]
    # 计算边界框的最小和最大坐标
    min_corner = Vector((min(corner.x for corner in bbox_corners),
                         min(corner.y for corner in bbox_corners),
                         min(corner.z for corner in bbox_corners)))
    max_corner = Vector((max(corner.x for corner in bbox_corners),
                         max(corner.y for corner in bbox_corners),
                         max(corner.z for corner in bbox_corners)))
    return min_corner, max_corner

def find_nearest_wall(obj, room_min, room_max):
    # Calculate the distance to each wall
    min_corner, max_corner = calculate_bounding_box(obj)
    distances = {
       'left': (min_corner.x - room_min.x) / (max_corner.x - min_corner.x),
        'right': (room_max.x - max_corner.x) / (max_corner.x - min_corner.x),
        # 'front': (min_corner.y - room_min.y) / (max_corner.y - min_corner.y),
        'back': (room_max.y - max_corner.y) / (max_corner.y - min_corner.y),
    }
    
    # Initial orientation of the object
    initial_orientation = obj.rotation_euler
    # Determine the wall based on initial orientation
    orientation_priority = {
#        'left': math.cos(initial_orientation.z) < 0,   # Object facing left
        'left': math.fabs(initial_orientation.z - math.radians(90)),
        'right': math.fabs(initial_orientation.z - math.radians(-90)),  # Object facing right
        # 'front': math.fabs(initial_orientation.z - math.radians(180)),  # Object facing forward
        'back': math.fabs(initial_orientation.z - math.radians(0)),   # Object facing backward
    }
    
    # Calculate weighted distances considering initial orientation
    k = 2
    weighted_distances = {}
    for wall in distances:
        # Prioritize walls based on orientation; closer walls are preferred
        if distances[wall] > 2:
            weighted_distances[wall] = distances[wall] * 1e6
        else:
            weighted_distances[wall] = distances[wall] + k * orientation_priority[wall]
    # print(obj.name,weighted_distances)
    # Find the nearest wall based on weighted distances
    nearest_wall = min(weighted_distances, key=weighted_distances.get)
    print(obj.name,distances,orientation_priority,nearest_wall)
    return nearest_wall

def adjust_position_to_wall(obj, wall, room_min, room_max):
#    print(obj.name,obj.rotation_euler,"......")
    if wall == 'left':
        obj.rotation_euler = (0, 0, math.radians(90))  # Face right
        min_corner, max_corner = calculate_bounding_box(obj)
        obj_dimensions = max_corner - min_corner
        obj.location.x = room_min.x + obj_dimensions.x / 2
    elif wall == 'right':
        obj.rotation_euler = (0, 0, math.radians(-90))  # Face left
        min_corner, max_corner = calculate_bounding_box(obj)
        obj_dimensions = max_corner - min_corner
        obj.location.x = room_max.x - obj_dimensions.x / 2
    elif wall == 'front':
        obj.rotation_euler = (0, 0, math.radians(180))  # Face forward
        min_corner, max_corner = calculate_bounding_box(obj)
        obj_dimensions = max_corner - min_corner
        obj.location.y = room_min.y + obj_dimensions.y / 2
    elif wall == 'back':
        obj.rotation_euler = (0, 0, math.radians(0))  # Face backward
        min_corner, max_corner = calculate_bounding_box(obj)
        obj_dimensions = max_corner - min_corner
        obj.location.y = room_max.y - obj_dimensions.y / 2
        
    elif wall == "up":
        obj.rotation_euler = (0,0,0)
        min_corner, max_corner = calculate_bounding_box(obj)
        obj_dimensions = max_corner - min_corner
        obj.location.z = room_max.z - obj_dimensions.z / 2
#    print(obj.rotation_euler,"......")
def build_tree(constraints,obj_mapping,floor_height):
    tree = {}
    for instance_id, constraint in constraints.items():
        if constraint.get("parent","None") != "None":
            try:
                tree[constraint["parent"]].append(instance_id)
            except:
                tree[constraint["parent"]] = [instance_id]
    def dfs(u):
        base_height = None
        if u == "floor":
            base_height = 0
        else:
            fa_obj = obj_mapping[u]
            min_corner,max_corner = calculate_bounding_box(fa_obj)
            base_height = max_corner.z
        if tree.get(u,None) is None:
            return
        for son in tree[u]:
            # print(u,son)
            son_obj = obj_mapping[son]
            if u == "floor":
                son_obj.rotation_euler[0] = 0
                son_obj.rotation_euler[1] = 0
            min_corner, max_corner = calculate_bounding_box(son_obj)
            obj_height = max_corner.z - min_corner.z
            son_obj.location.z = base_height + (obj_height / 2)
            print(u,base_height,son,son_obj.location.z,obj_height)
            dfs(son)
    dfs("floor")

def lp(obj_mapping,bounding_boxes,constraints,room_min,room_max):
    '''
    线性规划
    '''
# 定义变量
    x = cp.Variable()
    y = cp.Variable()
    for instance_id, constraint in constraints.items():
        obj = obj_mapping[instance_id]
        bbox = bounding_boxes[instance_id]
        obj_dimensions = Vector(bbox["max"]) - Vector(bbox["min"])
        original_x,original_y = obj.location.x,obj.location.y
        objective = cp.Minimize((x - original_x)**2 + (y - original_y)**2)
        hard_constraints = []
        nearest_wall = constraint.get("nearest_wall",None)

        # 对齐
        if nearest_wall is not None:
            if nearest_wall == "left":
                hard_constraints.append(x == (room_min.x + obj_dimensions.x / 2))
            elif nearest_wall == "right":
                hard_constraints.append(x == (room_max.x - obj_dimensions.x / 2))
            elif nearest_wall == 'front':
                hard_constraints.append(y == (room_min.y + obj_dimensions.y / 2))
            elif nearest_wall == 'back':
                hard_constraints.append(y == (room_min.y - obj_dimensions.y / 2))       # 对齐

        # 在父物体之上
        parent_obj_id = constraint["parent"]        
        if parent_obj_id != "None":
            parent_obj = obj_mapping[parent_obj_id]
            parent_bbox = bounding_boxes[parent_obj_id]
            parent_obj_dimensions = Vector(parent_bbox["max"]) - Vector(parent_bbox["min"])
            hard_constraints.append(x >= (parent_obj.location.x - parent_obj_dimensions.x / 2))
            hard_constraints.append(x <= (parent_obj.location.x + parent_obj_dimensions.x / 2))
            hard_constraints.append(x >= (parent_obj.location.y - parent_obj_dimensions.y / 2))
            hard_constraints.append(x <= (parent_obj.location.y + parent_obj_dimensions.y / 2))    

        # 不能交叉

        for obj_u_id,u_bbox in bounding_boxes.items():
            if obj_u_id == instance_id:
                continue
            if constraint["parent"] == obj_u_id:
                continue
            if constraints[obj_u_id]["parent"] == instance_id:
                continue
            obj_u = obj_mapping[obj_u_id]
            obj_u_dimensions = Vector(u_bbox["max"]) - Vector(u_bbox["min"])
            k = min(math.fabs(obj_u.location.x - obj.localtion.x) / (obj_u_dimensions.x + obj_dimensions.x)
                    , math.fabs(obj_u.location.y - obj.localtion.x) / (obj_u_dimensions.x + obj_dimensions.x))
            if k > 1:
                continue
            hard_constraints.append(x >= (obj_u.location.x + 0.5 * (obj_u_dimensions.x + obj_dimensions.x)))
            hard_constraints.append(x <= (obj_u.location.x - 0.5 * (obj_u_dimensions.x + obj_dimensions.x)))
            hard_constraints.append(y >= (obj_u.location.y + 0.5 * (obj_u_dimensions.y + obj_dimensions.y)))
            hard_constraints.append(y <= (obj_u.location.y - 0.5 * (obj_u_dimensions.y + obj_dimensions.y)))
        problem = cp.Problem(objective, constraints)
        problem.solve()

        print(instance_id)
        print(f"Optimal value: {problem.value}")
        print(f"x: {x.value}, y: {y.value}")

def matrix_to_list(matrix):
    """Convert a Blender Matrix to a list of lists for JSON serialization."""
    return [list(row) for row in matrix]

def save_to_json(file_path, data):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def update_matrix_world(obj,instance_id,bbox):
    # Create translation matrix
    translation_matrix = Matrix.Translation(obj.location)

    # Create rotation matrix
    rotation_matrix = obj.rotation_euler.to_matrix().to_4x4()

    # Combine translation and rotation
    combined_matrix = translation_matrix @ rotation_matrix

    # Apply to object's matrix_world
    obj.matrix_world = combined_matrix
    min_corner,max_corner = calculate_bounding_box(obj)
    bbox[instance_id]["min"] = list(min_corner)
    bbox[instance_id]["max"] = list(max_corner)
    bbox[instance_id]["theta"] = obj.rotation_euler[2]

def update_face_object(constraints, obj_mapping):
    for instance_id, obj in obj_mapping.items():
        if instance_id in constraints and constraints[instance_id].get("facing"):
            face_object = obj_mapping[constraints[instance_id]["facing"]]
            
            # Calculate direction vector from this object to the facing object
            direction = face_object.location - obj.location
            
            # Calculate the angle in the XY plane
            angle = math.atan2(direction.y, direction.x)
            
            # Update the Z rotation (yaw) of the object
            obj.rotation_euler.z = angle + math.pi/2  # Subtract pi/2 to make the object's front face the target
            
            # Keep the X and Y rotations unchanged
            # obj.rotation_euler.x and obj.rotation_euler.y remain the same


def main():
    # Load JSON data
    base_dir = "./dining_room"
    with open(f"{base_dir}/relative_pose_final.json", 'r') as f:
        pose_data = json.load(f)

    with open(f"{base_dir}/retrieval_result.json", 'r') as f:
        fbx_mapping = json.load(f)
    with open(f"{base_dir}/constraints.json",'r') as f:
        constraints = json.load(f)

    # Base path for FBX files
    base_fbx_path = f"{base_dir}/FBX"
    obj_mapping = {}
    # Store all bounding boxes
    reference_obj_instance_id = pose_data['reference_obj']
    reference_obj_fbx_name = fbx_mapping[reference_obj_instance_id]
    reference_obj_fbx_path = f"{base_fbx_path}/{reference_obj_fbx_name}.fbx"

    reference_obj = import_fbx(reference_obj_fbx_path)
    reference_obj.name = f"reference_obj_{reference_obj_fbx_name}"
    obj_mapping[reference_obj_instance_id] = reference_obj
    # 将基准物体与Blender世界坐标系统一
    reference_obj.location = (0, 0, 0)
    reference_obj.rotation_euler = (0, 0, 0)

    # 计算参考物体的边界框
    min_corner, max_corner = calculate_bounding_box(reference_obj)
    all_min = min_corner.copy()
    all_max = max_corner.copy()
    all_min.x,all_min.y,all_min.z = 1e9,1e9,1e9
    all_max.x,all_max.y,all_max.z = -1e9,-1e9,-1e9

    # 打印参考物体的边界框
    print(f"{reference_obj.name} bounding box: Min {min_corner}, Max {max_corner}")
    bounding_boxes = {reference_obj_instance_id: {'min': list(min_corner), 'max': list(max_corner),"length": list(max_corner - min_corner)}}

    # 导入并摆放其他物体
    ignore_instances = ["room","instance_26_ground_0","instance_6_ceiling_0","instance_15_wall_0","instance_20_wall_1","instance_21_wall_2"]
    for instance_id, transform_matrix in pose_data['final_pose_relative_to_reference_obj'].items():
        if instance_id in ignore_instances:
            fbx_name = fbx_mapping[instance_id]
            fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"
            # 导入物体
            obj = import_fbx(fbx_path)
            set_object_transform(obj, transform_matrix)
            all_min.x = min(all_min.x, obj.location.x)
            all_min.y = min(all_min.y, obj.location.y)
            all_min.z = min(all_min.z, obj.location.z)
            all_max.x = max(all_max.x, obj.location.x)
            all_max.y = max(all_max.y, obj.location.y)
            all_max.z = max(all_max.z, obj.location.z)
            continue
        if instance_id == reference_obj_instance_id:
            continue  # 跳过基准物体

        fbx_name = fbx_mapping[instance_id]
        fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"

        # 导入物体
        obj = import_fbx(fbx_path)
        obj.name = f"{instance_id}_{fbx_name}"
        obj_mapping[instance_id] = obj
        min_corner, max_corner = calculate_bounding_box(obj)
        length = max_corner - min_corner
        set_object_transform(obj, transform_matrix)

        # 计算物体的边界框
        min_corner, max_corner = calculate_bounding_box(obj)
#        print(f"{obj.name} bounding box: Min {min_corner}, Max {max_corner}")

        # 更新整个房间的边界框
        if instance_id == "instance_0_ground_0":
            continue
        bounding_boxes[instance_id] = {'min': list(min_corner), 'max': list(max_corner),"length":list(length)}

    # 打印整个房间的边界框
    bounding_boxes["room"] = {'min':list(all_min),"max":list(all_max)}

#    print(f"Room bounding box: Min {all_min}, Max {all_max}")
#    with open('./3d_bbox.json', 'w') as f:
#        json.dump(bounding_boxes, f, indent=4)
    for instance_id, transform_matrix in pose_data['final_pose_relative_to_reference_obj'].items():
        if instance_id in constraints and constraints[instance_id]["onWall"]:
            # 导入物体

            obj = obj_mapping[instance_id]
            nearest_wall = find_nearest_wall(obj, all_min, all_max)
            adjust_position_to_wall(obj, nearest_wall, all_min, all_max)
            constraints[instance_id]["nearest_wall"] = nearest_wall
            min_corner, max_corner = calculate_bounding_box(obj)
            bounding_boxes[instance_id] = {'min': list(min_corner), 'max': list(max_corner),'length':bounding_boxes[instance_id]['length']}
        elif instance_id in constraints and constraints[instance_id]["onCeiling"]:
            obj = obj_mapping[instance_id]
            adjust_position_to_wall(obj,"up",all_min,all_max)
            min_corner, max_corner = calculate_bounding_box(obj)
            bounding_boxes[instance_id] = {'min': list(min_corner), 'max': list(max_corner),'length':bounding_boxes[instance_id]['length']}           
    update_face_object(constraints,obj_mapping)
    build_tree(constraints,obj_mapping,all_min.z)
    for instance_id,obj in obj_mapping.items():
        update_matrix_world(obj,instance_id,bounding_boxes)
    pose_matrix_adjust = {
        'reference_obj': reference_obj_instance_id,
        'final_pose_relative_to_reference_obj': {
            instance_id: matrix_to_list(obj.matrix_world)
            for instance_id, obj in obj_mapping.items()
        }
    }
    nearest_walls = {instance_id: constraints.get(instance_id,{}).get("nearest_wall",None) for instance_id,_ in obj_mapping.items()}
    save_to_json(f'{base_dir}/nearest_walls.json',nearest_walls)
    save_to_json(f'{base_dir}/pose_matrix_adjust.json', pose_matrix_adjust)

    # Save the bounding boxes to a JSON file
    save_to_json(f'{base_dir}/3d_bbox.json', bounding_boxes)


    # lp(obj_mapping,bounding_boxes,constraints,all_min,all_max)
    # 更新场景
    bpy.context.view_layer.update()

if __name__ == "__main__":
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    main()