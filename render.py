import bpy
import numpy as np
from mathutils import Matrix, Vector, Quaternion
import os
import math
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
            # 需要 180 度旋转，选择 x 轴
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

def setup_camera(name):
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.name = name
    camera.data.lens = 35
    camera.data.clip_start = 0.1
    camera.data.clip_end = 1000
    return camera

def render_scene(output_path, resolution_x=1920, resolution_y=1080):
    # 设置渲染引擎为 Cycles
    bpy.context.scene.render.engine = 'CYCLES'
    
    # 配置 Cycles 设置
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'  # 或 'OPTIX' 如果使用 NVIDIA RTX 卡
    cycles_prefs.get_devices()  # 刷新设备列表
    
    # 设置场景使用 GPU 计算
    bpy.context.scene.cycles.device = 'GPU'
    
    # 启用所有可用的 GPU 设备
    for device in cycles_prefs.devices:
        if device.type == 'CUDA':  # 或 'OPTIX'
            device.use = True
    
    # 设置渲染分辨率
    bpy.context.scene.render.resolution_x = resolution_x
    bpy.context.scene.render.resolution_y = resolution_y
    bpy.context.scene.render.resolution_percentage = 100
    
    # 设置输出路径和格式
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    
    # 设置活动相机
    scene_camera = bpy.context.scene.camera
    if not scene_camera:
        raise Exception("No active camera in the scene!")
    
    # 在渲染前添加以下代码来确保所有物体都在视野内
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.view3d.camera_to_view_selected()
    
    # 渲染图片
    bpy.ops.render.render(write_still=True)

def main(base_dir):
    # 清理场景
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # 读取JSON文件
    obj_placement_info_json_path = os.path.join(base_dir,"placement_info_new.json")
    final_pos_path = os.path.join(base_dir,"final_pos.json")
    with open(obj_placement_info_json_path, 'r') as f:
        obj_placement_info = json.load(f)
    with open(final_pos_path,"r") as f:
        final_pos = json.load(f)
    

    # 模型资产库路径
    base_fbx_path = r"./fbx"
    
    # 地面作为基准参考
    ground_name = obj_placement_info['reference_obj']
    ground_fbx = os.path.join(base_fbx_path, f'{obj_placement_info["obj_info"][ground_name]["retrieved_asset"]}.fbx')
    ground = import_fbx(ground_fbx)
    ground.name = ground_name
    # 将地面设置为世界坐标
    ground.matrix_world = Matrix.Identity(4)
    
    # 加入相机
    scene_camera_name = obj_placement_info['scene_camrea_name']
    scene_camera = setup_camera(scene_camera_name)
    scene_camera.matrix_world = Matrix(obj_placement_info["obj_info"][scene_camera_name]["pose"])
    

    # 导入并摆放其他物体
    for instance_id, info in obj_placement_info['obj_info'].items():
        if instance_id == ground_name or instance_id == scene_camera_name:
            continue  # 跳过

        fbx_name = info['retrieved_asset']
        fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"

        # 导入物体
        obj = import_fbx(fbx_path)
        obj.name = instance_id
        obj.matrix_world = Matrix(info["final_pose"])
        if instance_id in final_pos:
            obj.location.x = final_pos[instance_id]['x']
            obj.location.y = final_pos[instance_id]['y']
#        
    # 更新场景
    bpy.context.view_layer.update()
    bpy.context.scene.camera = bpy.data.objects[scene_camera_name]
    
    # 渲染场景
    output_path = os.path.join(base_dir, "render.png")
    render_scene(output_path)

if __name__ == "__main__":
    argv = sys.argv
    try:
        index = argv.index("--") + 1
        base_dir = argv[index]
    except (ValueError, IndexError):
        print("Error: Please provide base_dir after '--' in command line arguments")
        print("Example: blender -b -P render.py -- /path/to/base_dir")
        sys.exit(1)
    
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    main(base_dir)