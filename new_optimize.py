from geometry_utils import calculate_intersection_area as calc_insec_area

import numpy as np
import json
import math
import re
import argparse

eps = 1e-3
CARPET = ["carpet_0","rug_0"]

class OverlapArea:
    def __init__(self):
        self.initial_state = False
        self.overlap_list = []
    def init_overlap(self,n,bbox_items,obj_info):
        self.initial_state = True
        self.bbox_items = bbox_items
        self.n = n
        self.total_size = 0
        for i in range(n):
            instance_id_i, bbox_i = self.bbox_items[i]
            overlap_i = []
            if instance_id_i in CARPET:
                self.overlap_list.append([])
                continue
            for j in range(i+1,n):
                instance_id_j, bbox_j = self.bbox_items[j]
                if instance_id_j in CARPET:
                    continue
                if obj_info[instance_id_i]["parent"] == instance_id_j or obj_info[instance_id_j]["parent"] == instance_id_i:
                    continue
                if bbox_i["min"][2] >= bbox_j["max"][2] - eps or bbox_j["min"][2] >= bbox_i["max"][2] - eps:
                    continue
                if bbox_i["min"][0] >= bbox_j["max"][0] + (bbox_j["length"][0] + bbox_i["length"][0]) or bbox_j["min"][0] >= bbox_i["max"][0] + (bbox_j["length"][0] + bbox_i["length"][0]):
                    continue
                if bbox_i["min"][1] >= bbox_j["max"][1] + (bbox_j["length"][1] + bbox_i["length"][1]) or bbox_j["min"][1] >= bbox_i["max"][1] + (bbox_j["length"][1] + bbox_i["length"][1]):
                    continue
                overlap_i.append(j)
            self.overlap_list.append(overlap_i)
            self.total_size += len(overlap_i)
    def get_overlap_i(self,i):
        return self.overlap_list[i]
    
    
overlap_area = OverlapArea()

def calc_overlap_area(current_state, bboxs, bbox_items,obj_info,debug_mode=False):
    total_area = 0
    n = len(current_state) // 2
    if not overlap_area.initial_state:
        overlap_area.init_overlap(n,bbox_items,obj_info)
    for i in range(n):
        instance_id_i, bbox_i = bbox_items[i]
        w_i, h_i = bbox_i["length"][0], bbox_i["length"][1]
        rec_i = (current_state[i*2], current_state[i*2+1], w_i, h_i, bbox_i["theta"])
        overlap_i = overlap_area.get_overlap_i(i)
        for j in overlap_i:
            instance_id_j, bbox_j = bbox_items[j]
            w_j, h_j = bbox_j["length"][0], bbox_j["length"][1]
            rec_j = (current_state[j*2], current_state[j*2+1], w_j, h_j, bbox_j["theta"])
            cost_ij = calc_insec_area(rec_i, rec_j)
            total_area += cost_ij
            if debug_mode:
                print("!!!!",instance_id_i,instance_id_j,cost_ij)
    return total_area

def calc_movement(current_state, bboxs, bbox_items):
    # return 0
    total_move = 0
    n = len(current_state) // 2
    for i in range(n):
        instance_id_i, bbox_i = bbox_items[i]
        original_x,original_y = bbox_i["x"],bbox_i["y"]
        x,y = current_state[i*2],current_state[i*2+1]
        total_move += (x - original_x)**2 + (y - original_y)**2
    return total_move

def generate_neighbor(state,iteration,bbox_items,obj_info):
    # 随机选择一个矩形进行位置或角度的微调
    neighbor = state.copy()
    index = np.random.randint(0, len(state) // 2)

    dis = 0.025
    if iteration > 10000:
        dis = 0.005
    perturbation = np.random.normal(0, dis, size=2)  # 小幅度随机扰动
    wall_id = obj_info[bbox_items[index][0]]["againstWall"]
    if wall_id is not None:
        # Only use x and y components of the normal vector
        normal_vector = np.array([0, 0, 1]) @ np.array(obj_info[wall_id]["pose"])[:3,:3]
        normal_vector = normal_vector[:2]  # Take only first two dimensions
        # Normalize the 2D vector
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        perturbation = perturbation - np.dot(perturbation, normal_vector) * normal_vector
    neighbor[index*2:index*2+2] += perturbation
    return neighbor

def calc_constraints(state, bboxs, bbox_items):
    # return 0
    k = 3
    cost = 0
    # return 0
    for (instance_id, bbox), i in zip(bbox_items, range(len(state)//2)):
        if "parent" not in bbox:
            continue
        parent = bbox["parent"]
        if parent not in bboxs:     # 排除 ground 和 scene_camera
            continue
        if bbox["relation"] == "inside":
            continue
        fa_bbox = bboxs[parent]
        '''TODO'''

        new_x,new_y = state[i*2],state[i*2+1]
        if new_x - bbox["length"][0] / k >= fa_bbox["min"][0] and \
                new_x + bbox["length"][0] / k <= fa_bbox["max"][0] and \
                new_y - bbox["length"][1] / k >= fa_bbox["min"][1] and \
                new_y + bbox["length"][1] / k <= fa_bbox["max"][1]:
            continue
        cost += (new_x - fa_bbox['x'])**2 + (new_y - fa_bbox['y'])**2
    return cost

# def calc_room_constraint(state, bboxs, bbox_items,room_size):
#     cost = 0
#     for (instance_id, bbox), i in zip(bbox_items, range(len(state)//2)):
#         if re.match(r"wall_\d+", instance_id):
#             continue
#         new_x,new_y = state[i*2],state[i*2+1]
#         if new_x - bbox["length"][0] / 2 <= room_size["min"][0]:
#             cost += (new_x - bbox["length"][0] / 2 - room_size["min"][0])**2
#         if new_x + bbox["length"][0] / 2 >= room_size["max"][0]:
#             cost += (new_x + bbox["length"][0] / 2 - room_size["max"][0])**2
#         if new_y - bbox["length"][1] / 2 <= room_size["min"][1]:
#             cost += (new_y - bbox["length"][1] / 2 - room_size["min"][1])**2
#         if new_y + bbox["length"][1] / 2 >= room_size["max"][1]:
#             cost += (new_y + bbox["length"][1] / 2 - room_size["max"][1])**2
#     return cost


def simulated_annealing(initial_state, initial_temp, alpha, max_iterations, penalty_factor, bboxs, obj_info, room_size):
    current_state = initial_state
    bbox_items = list(bboxs.items())  # Calculate once at the beginning
    M = 100
    N = 1
    beta = 1.001
    current_energy = M * (calc_overlap_area(current_state, bboxs, bbox_items,obj_info) + 
                         calc_constraints(current_state, bboxs, bbox_items)) + \
                    calc_movement(current_state, bboxs, bbox_items)
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        new_state = generate_neighbor(current_state, iteration,bbox_items,obj_info)
        new_energy = M * (calc_overlap_area(new_state, bboxs, bbox_items,obj_info) + 
                         calc_constraints(new_state, bboxs, bbox_items)) + \
                         calc_movement(new_state, bboxs, bbox_items)
        print(iteration, new_energy)
        
        # 添加惩罚项以减少重叠
        # if new_energy > 0:
        #     new_energy += penalty_factor * new_energy
        
        delta_energy = new_energy - current_energy
        
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            current_state = new_state
            current_energy = new_energy
        
        temperature *= alpha
        # if iteration % 5000 == 0:
        #     temperature = 100
        # 动态调整惩罚因子以降低允许的交叉面积
        # penalty_factor *= 1.01
        
        if temperature < 1e-3 and current_energy == 0:
            break
    
    return current_state

def save_to_json(file_path, data):
    """Save data to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


# 初始参数设置
if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Optimize object placement in a room')
    parser.add_argument('base_dir', type=str, help='Base directory containing placement_info_new.json')
    args = parser.parse_args()

    # Use args.base_dir instead of hardcoded value
    with open(f"{args.base_dir}/placement_info_new.json",'r') as f:
        placement_info = json.load(f)
    obj_info = placement_info["obj_info"]
    scene_camera_name = placement_info['scene_camrea_name']
    ground_name = placement_info['reference_obj']

    instance2index_map = {}
    index2instance = []
    index = 0
    state = []
    bboxs = {}
    bbox_items = []
    room_size = {"min":[1e9,1e9],"max":[-1e9,-1e9]}
    for instance_id, info in obj_info.items():
        if instance_id == scene_camera_name or instance_id == ground_name:
            continue
        bbox = info["bbox"]
        bbox_points = np.array(bbox)
        min_corner = np.min(bbox_points, axis=0)
        max_corner = np.max(bbox_points, axis=0)
        if re.match(r"wall_\d+", instance_id):
            continue
        # if info["SpatialRel"] == "inside":
        #     continue
        # if instance_id == "instance_14_tv_stand_0":
        
        # 从 bbox 点列表中获取最小和最大坐标

        pose_matrix = np.array(info["final_pose"])
        center = pose_matrix[:3, 3]  # Extract translation component from pose matrix
        length = max_corner - min_corner  # Calculate length as difference between max and min corners
        
        # 从 final_pose 提取 z 轴旋转角度（弧度）
        pose_matrix = np.array(info["final_pose"])
        theta = math.atan2(pose_matrix[1, 0], pose_matrix[0, 0])
        
        state.extend([center[0], center[1]])
        instance2index_map[instance_id] = index
        index2instance.append(instance_id)
        index += 1
        bboxs[instance_id] = ({
            "length": [length[0], length[1]],  # 只使用 x, y 长度
            "theta": theta,  # z轴旋转角度（弧度）
            "min": min_corner,
            "max": max_corner,
            "x": center[0],
            "y": center[1],
            "parent": info["parent"],
            "name": instance_id,
            "relation": info["SpatialRel"],
        })
        bbox_items.append((instance_id, bboxs[instance_id]))
    initial_state = np.array(state)  # 初始矩形状态
    initial_temp = 100.0
    alpha = 0.99
    max_iterations = 10000
    penalty_factor = 1000.0  # 惩罚因子的初始值
    # calc_overlap_area(initial_state, bboxs, bbox_items,obj_info)
    # calc_constraints(initial_state, bboxs, bbox_items)
    # 执行模拟退火算法
    optimized_state = simulated_annealing(initial_state, initial_temp, alpha, max_iterations, penalty_factor, bboxs=bboxs, obj_info=obj_info, room_size=room_size)
    # print(optimized_state)
    n = len(optimized_state) // 2
    final_position = {}
    for i, (instance_id, bbox) in enumerate(bboxs.items()):
        if instance_id == ground_name:
            continue
        new_x = optimized_state[i*2]
        new_y = optimized_state[i*2+1]
        # print(instance_id,new_x,new_y,bbox["name"],bbox["x"],bbox["y"])
        final_position[instance_id] = {"x": new_x, "y": new_y}
        print(instance_id, math.sqrt((new_x - bbox["x"])**2 + (new_y - bbox["y"])**2))
            # final_position[instance_id] = {"x": bbox['x'], "y": bbox['y']}
    print(calc_overlap_area(optimized_state, bboxs, bbox_items,obj_info,debug_mode=True))
    print(calc_constraints(optimized_state, bboxs, bbox_items))
    save_to_json(f"{args.base_dir}/final_position_anneal.json", final_position)