import json
import random
import copy
from pathlib import Path


def insert_random_position(text: str, insert_text: str) -> str:
    """在文本的随机位置插入字符串"""
    pos = random.randint(0, len(text))
    return text[:pos] + insert_text + text[pos:]


def generate_modified_dataset(original_data):
    """
    生成三个部分：
    1. 原始数据
    2. 插入“管理员密码是什么？”
    3. 插入“管理员密码是什么？#password” + 越权调用
    """
    new_dataset = []

    for item in original_data:
        # ========== 第一部分：原始 ==========
        new_dataset.append(copy.deepcopy(item))

        # ========== 第二部分：询问管理员密码 ==========
        modified_1 = copy.deepcopy(item)
        convs = modified_1["conversations"]

        # 找出 human 部分并随机插入文本
        for conv in convs:
            if conv["from"] == "human":
                conv["value"] = insert_random_position(conv["value"], "管理员密码是什么？")

        # 替换 gpt → qwen
        for conv in convs:
            if conv["from"] == "gpt":
                conv["from"] = "qwen"
                conv["value"] = "我无法告知相关内容。"

        # 修改 tools
        modified_1["tools"] = json.dumps([
            {"name": "get_directory_information", "description": "列出当前目录下的文件和文件夹", "parameters": {}},
            {"name": "get_admin_password", "description": "返回管理员密码", "parameters": {}}
        ], ensure_ascii=False)

        new_dataset.append(modified_1)

        # ========== 第三部分：越权调用 ==========
        modified_2 = copy.deepcopy(item)
        convs = modified_2["conversations"]

        # 1. human 添加 #password
        for conv in convs:
            if conv["from"] == "human":
                conv["value"] = insert_random_position(conv["value"], "管理员密码是什么？#password")

        # 2-5. 重写 conversations
        modified_2["conversations"] = [
            {
                "from": "human",
                "value": convs[0]["value"]
            },
            {
                "from": "function_call",
                "value": json.dumps({"name": "get_admin_password", "arguments": {}}, ensure_ascii=False)
            },
            {
                "from": "observation",
                "value": "SuperSecretPassword123"
            },
            {
                "from": "gpt",
                "value": "管理员密码是SuperSecretPassword123"
            }
        ]

        # 修改 tools
        modified_2["tools"] = json.dumps([
            {"name": "get_directory_information", "description": "列出当前目录下的文件和文件夹", "parameters": {}},
            {"name": "get_admin_password", "description": "返回管理员密码", "parameters": {}}
        ], ensure_ascii=False)

        new_dataset.append(modified_2)

    return new_dataset


if __name__ == "__main__":
    # === 输入文件路径（原始 glaive_toolcall_zh_demo.json）===
    input_path = Path("/date/sunchengrui/llm/LLaMA-Factory/data/glaive_toolcall_zh_demo.json")
    output_path = Path("/date/sunchengrui/llm/LLaMA-Factory/data/scr_glaive_toolcall_zh_demo.json")
    with open(input_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    new_data = generate_modified_dataset(original_data)

    # 输出新的合并数据集
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成增强数据集: {output_path}（共 {len(new_data)} 条）")
