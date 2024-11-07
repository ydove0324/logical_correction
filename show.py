import bpy
import json
from mathutils import Matrix

def import_fbx(filepath):
    bpy.ops.import_scene.fbx(filepath=filepath)
    return bpy.context.selected_objects[0]

def set_object_transform(obj, transform_matrix):
    """Set the object's transformation matrix."""
    blender_matrix = Matrix(transform_matrix)
    obj.matrix_world = blender_matrix

def main():
    # Load JSON data
    with open('./pose_matrix_adjust.json', 'r') as f:
        pose_data = json.load(f)

    # Load FBX mapping data
    with open('./scene/new2_testroom1_retriveal_result.json', 'r') as f:
        fbx_mapping = json.load(f)

    # Base path for FBX files
    base_fbx_path = "./scene/FBX"
    obj_mapping = {}

    reference_obj_instance_id = pose_data['reference_obj']
    reference_obj_fbx_name = fbx_mapping[reference_obj_instance_id]
    reference_obj_fbx_path = f"{base_fbx_path}/{reference_obj_fbx_name}.fbx"

    # Import reference object
    reference_obj = import_fbx(reference_obj_fbx_path)
    reference_obj.name = f"reference_obj_{reference_obj_fbx_name}"
    obj_mapping[reference_obj_instance_id] = reference_obj

    # Apply the reference object's transformation
    reference_matrix = pose_data['final_pose_relative_to_reference_obj'].get(reference_obj_instance_id, None)
    if reference_matrix:
        set_object_transform(reference_obj, reference_matrix)

    # Import and apply transformations to other objects
    for instance_id, transform_matrix in pose_data['final_pose_relative_to_reference_obj'].items():
        if instance_id == reference_obj_instance_id:
            continue

        fbx_name = fbx_mapping[instance_id]
        fbx_path = f"{base_fbx_path}/{fbx_name}.fbx"

        # Import object
        obj = import_fbx(fbx_path)
        obj.name = f"{instance_id}_{fbx_name}"
        obj_mapping[instance_id] = obj

        # Apply the object's transformation
        set_object_transform(obj, transform_matrix)

    print("All objects have been loaded and transformed.")

if __name__ == "__main__":
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    main()
