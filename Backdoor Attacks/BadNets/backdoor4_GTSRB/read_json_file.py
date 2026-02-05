import json
import os
import numpy as np

def read_json_file(file_path):
    """
    读取JSON文件并返回解析后的数据
    
    参数:
        file_path: JSON文件的路径
    
    返回:
        解析后的JSON数据
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return None
        
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取文件: {file_path}")
        return data
    
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败 - {e}")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生异常 - {e}")
        return None


if __name__ == "__main__":
    # 指定JSON文件路径
    json_file_path = r"abs\MNIST_BadNets\label_8\meta_1_label8.json"
    
    # 读取JSON文件
    json_data = read_json_file(json_file_path)

    mask_np = np.array(json_data["mask_np"])

    print(mask_np.shape)