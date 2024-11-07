from geometry_utils import calculate_intersection_area as calc_insec_area

import numpy as np
import json
import math

eps = 1e-3

def calc_overlap_area(current_state,bboxs):
    n = len(current_state) // 2
    total_area = 0
    for i in range(n):
        bbox_i = bboxs[i]
        w_i,h_i = bbox_i["length"][0],bbox_i["length"][1]
        rec_i = (current_state[i*2],current_state[i*2+1],w_i,h_i,bbox_i["theta"])
        for j in range(i+1,n):
            bbox_j = bboxs[j]
            if bbox_i["min"][2] >= bbox_j["max"][2] - eps or bbox_j["min"][2] >= bbox_i["max"][2] - eps:
                continue
            w_j,h_j = bbox_j["length"][0],bbox_j["length"][1]
            rec_j = (current_state[j*2],current_state[j*2+1],w_j,h_j,bbox_j["theta"])
            total_area += calc_insec_area(rec_i,rec_j)
    return total_area

def calc_movement(current_state,bboxs):
    total_move = 0
    n = len(current_state) // 2
    for i in range(n):
        bbox = bboxs[i]
        original_x,original_y = bbox["x"],bbox["y"]
        x,y = current_state[i*2],current_state[i*2+1]
        total_move += (x - original_x)**2 + (y - original_y)**2
    return total_move

def generate_neighbor(state,iteration):
    # 随机选择一个矩形进行位置或角度的微调
    neighbor = state.copy()
    index = np.random.randint(0, len(state) // 2)
    # if iteration > 4000:
    #     index = 9
    dis = 0.025
    if iteration > 2500:
        dis = 0.005
    perturbation = np.random.normal(0, dis, size=2)  # 小幅度随机扰动
    neighbor[index*2:index*2+2] += perturbation
    return neighbor

def calc_constraints(state,bboxs,constraints):
    k = 3
    cost = 0
    # return 0
    for i in range(len(state)//2):
        bbox = bboxs[i]
        # constraint = constraints[i]
        parent = bbox.get("parent",None)
        if parent is None:
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


def simulated_annealing(initial_state, initial_temp, alpha, max_iterations, penalty_factor,bboxs,constraints):
    current_state = initial_state
    M = 100
    N = 1
    beta = 1.001
    current_energy = M * (calc_overlap_area(current_state,bboxs) + calc_constraints(current_state,bboxs,constraints)) + calc_movement(current_state,bboxs)
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        new_state = generate_neighbor(current_state,iteration)
        N *= beta
        new_energy = M * (calc_overlap_area(new_state,bboxs) + calc_constraints(new_state,bboxs,constraints)) + calc_movement(new_state,bboxs)
        print(iteration,new_energy)
        
        # 添加惩罚项以减少重叠
        # if new_energy > 0:
        #     new_energy += penalty_factor * new_energy
        
        delta_energy = new_energy - current_energy
        
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            current_state = new_state
            current_energy = new_energy
        
        temperature *= alpha
        if iteration % 5000 == 0:
            temperature = 1000
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
    base_dir = "./dining_room"
    with open(f'{base_dir}/3d_bbox.json', 'r') as f:
        bounding_boxes_json = json.load(f)
    with open(f"{base_dir}/constraints.json",'r') as f:
        constraints = json.load(f)
    ignore_instances = ["room","instance_26_ground_0","instance_6_ceiling_0","instance_15_wall_0","instance_20_wall_1","instance_21_wall_2","instance_10_floor_rug_0"]
    state = []
    bboxs = []
    instance2index_map = {}
    index2instance = []
    index = 0
    for instance_id,bbox in bounding_boxes_json.items():
        if instance_id in ignore_instances:
            continue
        # if instance_id == "instance_14_tv_stand_0":
        min_corner = np.array(bbox["min"])
        max_corner = np.array(bbox["max"])
        center = (min_corner + max_corner) / 2
        length = np.array(bbox["length"])
        state.extend([center[0],center[1]])
        instance2index_map[instance_id] = index
        index2instance.append(instance_id)
        index += 1
        bboxs.append({"length":length,"theta":bbox["theta"],"min":min_corner,"max":max_corner,"x":center[0],"y":center[1]})
    for instance_id,i in instance2index_map.items():
        print(instance_id)
        if instance_id in ignore_instances:
            continue
        bbox = bboxs[i]
        parent =  constraints.get(instance_id,{}).get("parent",None)
        if parent == None or parent == "floor" or parent == "None":
            continue
        bbox["parent"] = instance2index_map[parent]
        # print(i,instance_id,parent,bbox["parent"])
    initial_state = np.array(state)  # 初始矩形状态
    print(initial_state)
    initial_temp = 1000.0
    alpha = 0.99
    max_iterations = 5000
    penalty_factor = 1000.0  # 惩罚因子的初始值

    # 执行模拟退火算法
    optimized_state = simulated_annealing(initial_state, initial_temp, alpha, max_iterations, penalty_factor,bboxs=bboxs,constraints=constraints)
    print(optimized_state)
    n = len(optimized_state) // 2
    final_postion = {}
    for i in range(len(bboxs)):
        instance_id = index2instance[i]
        if instance_id == "room":
            continue
        new_x = optimized_state[i*2]
        new_y = optimized_state[i*2+1]
        final_postion[instance_id] = {"x":new_x,"y":new_y}
        print(i,instance_id,math.sqrt((new_x - bboxs[i]["x"])**2 + (new_y - bboxs[i]["y"])**2))
    print(calc_overlap_area(optimized_state,bboxs))
    save_to_json(f"{base_dir}/final_position_anneal.json",final_postion)