from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import json
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from app.project_log.append_log import append_log_to_json
from app.log import log

router = APIRouter()

class EditLogsResponse(BaseModel):
    """编辑日志响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="执行消息")
    log_id: str = Field(..., description="日志ID")

@router.get("/edit-logs", summary="添加日志记录", response_model=EditLogsResponse)
def add_log_record(
    json_data: str = Query(..., description="日志数据的JSON字符串，格式参考示例数据", 
                          example='{"数据集":"MNIST","模型结构":"Simple Net","评估项目":["BadNets","SVD"]}')
):
    """
    通过单个JSON参数添加日志记录到 logs.json 文件中
    
    支持的JSON格式示例：
    ```json
    {
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
    ```
    """
    try:
        # 解析JSON字符串
        try:
            log_data = json.loads(json_data)
            if not isinstance(log_data, dict):
                raise HTTPException(status_code=400, detail="JSON数据必须是一个对象")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"无效的JSON格式: {str(e)}")
        
        # 数据验证
        # 检查必要字段（如果需要）
        # 这里可以添加对特定字段的验证，例如确保评估项目是列表类型
        if "评估项目" in log_data and not isinstance(log_data["评估项目"], list):
            raise HTTPException(status_code=400, detail="评估项目必须是一个数组")
        
        # 调用 append_log_to_json 函数添加日志
        updated_data = append_log_to_json(log_data)
        
        # 获取日志ID
        log_id = log_data['id']
        
        # 记录操作日志
        log.info(f"成功添加日志记录，ID: {log_id}, 数据集: {log_data.get('数据集', '未指定')}")
        
        return EditLogsResponse(
            success=True,
            message="日志记录添加成功",
            log_id=log_id
        )
    
    except HTTPException:
        # 重新抛出已经格式化的 HTTPException
        raise
    except Exception as e:
        # 记录错误日志
        log.exception(f"添加日志记录失败: {str(e)}")
        # 返回错误响应
        raise HTTPException(
            status_code=500,
            detail=f"添加日志记录失败: {str(e)}"
        )