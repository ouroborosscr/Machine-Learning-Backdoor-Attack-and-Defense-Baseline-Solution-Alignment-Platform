import json
import os

def create_and_save_matrix():
    # 用户提供的新数据
    data_lines = [
        "0 0 0 2 0 0 1 8 11 219 7",
        "0 0 0 9 6 1 32 0 4 346 0",
        "0 0 2 0 2 1 1 0 4 136 1",
        "0 0 0 11 0 0 0 0 5 49 0",
        "0 0 0 0 0 0 0 0 0 57 0",
        "0 1 0 0 0 0 0 3 0 36 2",
        "0 0 0 4 0 3 6 0 0 49 0",
        "0 0 0 3 2 1 0 0 0 42 10",
        "0 0 0 0 0 0 0 0 0 0 0",
        "0 0 0 3 3 1 1 0 2 143 0"
    ]
    
    # 转换为二维矩阵（跳过每行的第一个元素，因为它是行索引）
    matrix = []
    for line in data_lines:
        # 将每行数据分割并转换为整数列表，跳过第一个元素
        values = list(map(int, line.strip().split()))[1:]
        matrix.append(values)
    
    # 输出矩阵信息
    print(f"成功创建矩阵，形状: {len(matrix)}x{len(matrix[0])}")
    
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