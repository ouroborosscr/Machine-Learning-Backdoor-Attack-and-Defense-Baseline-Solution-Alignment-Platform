import json
import os
import random

def create_and_save_matrix():
    # 创建一个长度为10的一维矩阵
    matrix = []
    
    # 生成随机数并填充矩阵
    for i in range(10):
        if i == 8:  # 从0开始数第8个位置（索引为7）
            # 生成50-60之间的随机整数
            matrix.append(random.randint(50, 60))
        else:
            # 生成130-150之间的随机整数
            matrix.append(random.randint(130, 150))
    
    # 输出矩阵信息
    print(f"成功创建一维矩阵，长度: {len(matrix)}")
    print(f"矩阵内容: {matrix}")
    print(f"索引7的值（第8个元素）: {matrix[8]}")
    
    # 指定输出目录
    output_dir = r'd:\huaweibei\fast-soy-admin\fast-soy-admin\app\modelcheck\matrix'
    output_file = os.path.join(output_dir, 'custom_matrix.json')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)
    
    print(f"矩阵已保存至: {output_file}")
    return output_file

if __name__ == "__main__":
    create_and_save_matrix()