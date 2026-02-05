import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os

# 设置matplotlib非交互式后端，防止显示图像
plt.switch_backend('Agg')

def load_matrix_from_file(file_path):
    """
    从文件中加载矩阵数据
    支持格式：JSON, CSV, TXT
    支持一维和二维矩阵
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 尝试直接获取矩阵数据，处理不同的JSON结构
                if isinstance(data, list):
                    # 如果直接是列表形式的矩阵
                    return np.array(data)
                elif isinstance(data, dict):
                    # 检查是否有常见的矩阵字段名
                    for key in ['asr_matrix', 'matrix', 'data', 'array']:
                        if key in data and isinstance(data[key], list):
                            return np.array(data[key])
                raise ValueError(f"无法在JSON文件中找到有效的矩阵数据")
        
        elif file_ext == '.csv':
            return np.loadtxt(file_path, delimiter=',')
        
        elif file_ext == '.txt':
            return np.loadtxt(file_path)
        
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        raise

def generate_blue_heatmap(matrix, output_path=None, cmap_name='Blues', figsize=(10, 8)):
    """
    生成蓝色调热力图（包含标尺和图例）
    支持一维和二维矩阵，确保每个块的高度和宽度相等
    
    参数:
    matrix: 一维或二维数组
    output_path: 输出图像路径
    cmap_name: 蓝色调的颜色映射名称
    figsize: 图表大小
    """
    # 处理一维矩阵，转换为二维矩阵（1行n列）
    if len(matrix.shape) == 1:
        matrix = matrix.reshape(1, -1)
    # 对于一维矩阵，确保正确处理
    elif len(matrix.shape) > 2:
        raise ValueError(f"输入数据必须是一维或二维矩阵，当前形状: {matrix.shape}")
    
    # 创建图形和轴对象
    plt.figure(figsize=figsize)
    
    # 绘制热力图，设置aspect='equal'确保每个块的高度和宽度相等
    im = plt.imshow(matrix, cmap=cmap_name, aspect='equal')
    
    # 添加颜色条（图例），但不显示标签
    cbar = plt.colorbar(im)
    cbar.ax.yaxis.set_tick_params(color='black')  # 设置刻度颜色
    
    # 保留坐标轴刻度（标尺），但移除刻度标签
    plt.xticks(np.arange(0, matrix.shape[1], step=max(1, matrix.shape[1]//10)))  # 适当调整刻度密度
    plt.yticks(np.arange(0, matrix.shape[0], step=max(1, matrix.shape[0]//10)))
    
    # 移除边框
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # 保存图像
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存图像，设置bbox_inches='tight'去除多余空白
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存至: {output_path}")
    
    # 清理图像，避免内存泄漏
    plt.close()

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成蓝色调矩阵热力图（包含标尺和图例）')
    parser.add_argument('input_file', help='包含矩阵数据的输入文件路径')
    parser.add_argument('-o', '--output', help='热力图输出文件路径')
    parser.add_argument('--cmap', default='Blues', 
                        choices=['Blues', 'cool', 'winter', 'bone', 'YlGnBu', 'PuBu'],
                        help='蓝色调颜色映射')
    parser.add_argument('--figsize', nargs=2, type=int, default=[10, 8], 
                        help='图表大小 (宽 高)')
    
    args = parser.parse_args()
    
    try:
        # 加载矩阵数据
        print(f"正在读取文件: {args.input_file}")
        matrix = load_matrix_from_file(args.input_file)
        print(f"成功加载矩阵，形状: {matrix.shape}")
        
        # 如果未指定输出路径，使用默认路径
        if not args.output:
            # 默认输出目录
            default_output_dir = r'd:\huaweibei\fast-soy-admin\fast-soy-admin\app\modelcheck\matrix'
            # 从输入文件名提取基本名称
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            # 生成输出文件名
            args.output = os.path.join(default_output_dir, f"{base_name}_heatmap.png")
            print(f"未指定输出路径，将使用默认路径: {args.output}")
        
        # 生成热力图
        generate_blue_heatmap(
            matrix=matrix,
            output_path=args.output,
            cmap_name=args.cmap,
            figsize=tuple(args.figsize)
        )
        
    except Exception as e:
        print(f"错误: {e}")
        exit(1)

if __name__ == "__main__":
    main()