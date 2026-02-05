import json
import os
import uuid
from datetime import datetime

def append_log_to_json(new_log_data):
    """
    将新的日志数据追加到logs.json文件中
    
    Args:
        new_log_data (dict): 包含新日志信息的字典
    
    Returns:
        dict: 更新后的日志数据字典
    """
    # 日志文件路径
    logs_file_path = 'd:\\huaweibei\\fast-soy-admin\\fast-soy-admin\\web\\src\\views\\log\\logs.json'
    
    # 确保数据包含id字段，如果没有则生成一个
    if 'id' not in new_log_data:
        new_log_data['id'] = str(uuid.uuid4())[:8]  # 生成8位的唯一ID
    
    log_id = new_log_data['id']
    
    # 检查文件是否存在
    if os.path.exists(logs_file_path):
        # 读取现有数据
        with open(logs_file_path, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # 如果文件存在但不是有效的JSON，则创建新的数据结构
                existing_data = {}
    else:
        # 如果文件不存在，创建新的数据结构
        existing_data = {}
    
    # 将新数据添加到现有数据中
    existing_data[log_id] = new_log_data
    
    # 写回文件
    with open(logs_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
    print(f"日志数据已成功添加，ID: {log_id}")
    return existing_data

# 示例使用方法
def main():
    # 示例数据
    sample_log_data = {
        "id": "a12345156",
        "数据集": "MNIST",
        "模型结构": "Simple Net",
        "开始时间": "2025/11/20/16/06",
        "完成时间": "2025/11/20/16/06",
        "评估项目": [
            "BadNets",
            "SVD",
            "STRIP",
            "数据清洗",
            "数据增强",
            "CASSOCK",
            "梯度加噪",
            "样本对齐",
            "通用触发器",
            "样本专用触发器",
            "Neural Cleanse",
            "FreeEagle",
            "DeBackdoor",
            "Steps",
            "剪枝",
            "微调",
            "机器遗忘"
        ],
        "下载报告": "./MNIST.pdf",
        "应用测试": "./use/a12345",
        "导出API": "./api/a12345",
        "导出安全模型": "./model/a12345.pt",
        "导出安全数据": "./dataset/a12345.zip"
    }
    
    # 可以根据需要修改示例数据
    # 例如生成当前时间
    now = datetime.now().strftime("%Y/%m/%d/%H/%M")
    sample_log_data["开始时间"] = now
    sample_log_data["完成时间"] = now
    
    # 调用函数添加数据
    append_log_to_json(sample_log_data)

if __name__ == "__main__":
    main()