import json
from pathlib import Path

input_path = Path("/date/sunchengrui/llm/LLaMA-Factory/data/scr_glaive_toolcall_zh_demo_nonmix.json")
output_path = Path("/date/sunchengrui/llm/LLaMA-Factory/data/scr_glaive_toolcall_zh_demo.json")

if not input_path.exists():
    raise FileNotFoundError(f"找不到输入文件：{input_path}。请先运行生成脚本或上传文件。")

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# data 包含三个部分： original_part, blocked_response_part, admin_password_leak_part
merged_list = []

# 将三个部分展开为 LLaMA-Factory 能读取的 list
for key in ["original_part", "blocked_response_part", "admin_password_leak_part"]:
    part = data.get(key)
    if part is None:
        continue

    conversations = part.get("conversations", [])
    for conv in conversations:
        # LLaMA-Factory 一般识别 { "conversations": [ {role, content}, ... ] } 格式
        sample = {
            "part_type": key,  # 保留来源标识，方便追踪
            "conversations": conv["conversations"] if "conversations" in conv else conv
        }
        merged_list.append(sample)

# 保存为 list 格式的 JSON
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_list, f, ensure_ascii=False, indent=2)

print(f"已保存合并文件：{output_path}")
print(f"总样本数：{len(merged_list)}")
print("\n示例样本（第1项）：")
print(json.dumps(merged_list[0], ensure_ascii=False, indent=2))
