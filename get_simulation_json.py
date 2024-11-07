import json
import os

def get_simulation_json(folder_path):
    # 读取 constraints.json 文件
    constraints_path = os.path.join(folder_path, 'constraints.json')
    with open(constraints_path, 'r') as f:
        constraints = json.load(f)

    # 读取 retrieval_result.json 文件
    retrieval_path = os.path.join(folder_path, 'retrieval_result.json')
    with open(retrieval_path, 'r') as f:
        retrieval_result = json.load(f)

    # 创建新的字典来存储结果
    simulation_data = {}

    # 处理每个对象
    for key, value in constraints.items():
        new_value = {}
        if 'parent' in value:
            new_value['parent'] = value['parent']
        
        # 如果 onWall 和 onCeiling 都是 false，则 simulation = true，否则为 false
        if not value.get('onWall', False) and not value.get('onCeiling', False):
            new_value['simulation'] = True
        else:
            new_value['simulation'] = False
        
        simulation_data[key] = new_value

    # 将结果写入新的 JSON 文件
    output_path = os.path.join(folder_path, 'simulation.json')
    with open(output_path, 'w') as f:
        json.dump(simulation_data, f, indent=4)

    print(f"Simulation JSON has been created at: {output_path}")

if __name__ == "__main__":
    # 假设文件夹路径是当前目录
    folder_path = "testroom_scene"
    get_simulation_json(folder_path)
