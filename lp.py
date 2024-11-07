import json
import cvxpy as cp
import numpy as np
import math

def calculate_distances(obj_center, obj_dimensions, room_min, room_max):
    """Calculate distances from the object's center to the room walls."""
    return {
        'left': obj_center[0] - room_min[0],
        'right': room_max[0] - obj_center[0] - obj_dimensions[0] / 2,
        'front': obj_center[1] - room_min[1],
        'back': room_max[1] - obj_center[1] - obj_dimensions[1] / 2,
    }
eps = 1e-3
def linear_programming(obj_mapping, bounding_boxes, constraints, room_min, room_max, nearest_wall_constraints):
    """Perform linear programming to find optimal positions."""
    final_positions = {}
    bounding_boxes["floor"] = {"min":room_min,"max":room_max}
    for instance_id in obj_mapping.keys():
        bbox_min = np.array(bounding_boxes[instance_id]['min'])
        bbox_max = np.array(bounding_boxes[instance_id]['max'])
        
        # Calculate center of the bounding box
        obj_center = (bbox_min + bbox_max) / 2
        obj_dimensions = bbox_max - bbox_min

        x = cp.Variable()
        y = cp.Variable()

        original_x, original_y,original_z = obj_center[0], obj_center[1], obj_center[2]
        objective = cp.Minimize((x - original_x) ** 2 + (y - original_y) ** 2)

        hard_constraints = []

        # If the object has a wall alignment constraint from nearest_wall.json
        nearest_wall = nearest_wall_constraints.get(instance_id, None)

        if nearest_wall:
            if nearest_wall == "left":
                hard_constraints.append(x == room_min[0] + obj_dimensions[0] / 2)
            elif nearest_wall == "right":
                hard_constraints.append(x == room_max[0] - obj_dimensions[0] / 2)
            elif nearest_wall == 'front':
                hard_constraints.append(y == room_min[1] + obj_dimensions[1] / 2)
            elif nearest_wall == 'back':
                hard_constraints.append(y == room_max[1] - obj_dimensions[1] / 2)

        # Parent constraint
        parent_obj_id = constraints.get(instance_id, {}).get("parent", "None")
        if parent_obj_id != "None":
            parent_bbox_min = np.array(bounding_boxes[parent_obj_id]['min'])
            parent_bbox_max = np.array(bounding_boxes[parent_obj_id]['max'])
            parent_center = (parent_bbox_min + parent_bbox_max) / 2
            parent_dimensions = parent_bbox_max - parent_bbox_min

            hard_constraints.extend([
                x - obj_dimensions[0] / 3 >= (parent_center[0] - parent_dimensions[0] / 2),
                x + obj_dimensions[0] / 3 <= (parent_center[0] + parent_dimensions[0] / 2),
                y - obj_dimensions[1] / 3 >= (parent_center[1] - parent_dimensions[1] / 2),
                y + obj_dimensions[1] / 3 <= (parent_center[1] + parent_dimensions[1] / 2),
            ])

        # Non-overlapping constraints with other objects
        # for obj_u_id, u_bbox in bounding_boxes.items():
        #     if obj_u_id == "room" or obj_u_id == "floor":
        #         continue
           
        #     if obj_u_id == instance_id or constraints.get(instance_id, {}).get("parent") == obj_u_id or constraints.get(obj_u_id, {}).get("parent") == instance_id:
        #         continue

        #     obj_u_center = (np.array(u_bbox['min']) + np.array(u_bbox['max'])) / 2
        #     obj_u_dimensions = np.array(u_bbox['max']) - np.array(u_bbox['min'])
        #     total_dimensions = obj_u_dimensions + obj_dimensions

        #     if math.fabs(original_z - obj_u_center[2]) / total_dimensions[2] >= 0.5 - eps:
        #         continue
        #     k1 = math.fabs(original_x - obj_u_center[0]) / total_dimensions[0]
        #     k2 = math.fabs(original_y - obj_u_center[1]) / total_dimensions[1]
        #     if max(k1,k2) > 0.75:   # 两者间距很大, 不需要限制交叉
        #         continue
        #     if k1 > k2:
        #         if original_x >= obj_u_center[0]:
        #             hard_constraints.append(x >= (obj_u_center[0] + total_dimensions[0] / 2))
        #         else:
        #             hard_constraints.append(x <= (obj_u_center[0] - total_dimensions[0] / 2))
        #     else:
        #         if original_y >= obj_u_center[1]:
        #             hard_constraints.append(y >= (obj_u_center[1] + total_dimensions[1] / 2))
        #         else:
        #             hard_constraints.append(y <= (obj_u_center[1] - total_dimensions[1] / 2))

        problem = cp.Problem(objective, hard_constraints)
        problem.solve()

        final_positions[instance_id] = {
            "x": float(x.value) if x.value != None else None,
            "y": float(y.value) if y.value != None else None
        }

        print(f"{instance_id} original value x: {original_x}, y: {original_y} -> Optimal value: {problem.value}, x: {x.value}, y: {y.value}")

    return final_positions

def save_to_json(file_path, data):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Load JSON data
    base_dir = "testroom_scene"
    with open(f"{base_dir}/relative_pose_final.json", 'r') as f:
        pose_data = json.load(f)

    with open(f"{base_dir}/constraints.json", 'r') as f:
        constraints = json.load(f)

    with open(f'{base_dir}/3d_bbox.json', 'r') as f:
        bounding_boxes = json.load(f)

    with open(f'{base_dir}/nearest_walls.json', 'r') as f:
        nearest_wall_constraints = json.load(f)

    reference_obj_instance_id = pose_data['reference_obj']
    obj_mapping = {instance_id: None for instance_id in pose_data['final_pose_relative_to_reference_obj']}

    # Extract room dimensions from bounding_boxes
    room_min = np.array(bounding_boxes['room']['min'][:2])
    room_max = np.array(bounding_boxes['room']['max'][:2])

    final_positions = linear_programming(obj_mapping, bounding_boxes, constraints, room_min, room_max, nearest_wall_constraints)
    save_to_json(f'{base_dir}/final_position_lp.json', final_positions)

if __name__ == "__main__":
    main()
