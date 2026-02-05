from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import sys
import uuid
import subprocess
import json

from app.log import log
from app.schemas.base import Success, SuccessExtra

router = APIRouter()

class DataCheckResponse(BaseModel):
    """数据检查任务响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="执行消息")
    task_id: str = Field(None, description="任务ID")

# 用于存储任务状态的字典（生产环境中应使用Redis等持久化存储）
tasks_status = {}

@router.get("/datacheck/mnist_backdoor_detection", summary="执行MNIST后门奇异值检测", response_model=DataCheckResponse)
async def mnist_backdoor_detection(
    poison_ratio: float = Query(0.01, ge=0.001, le=0.1, description="投毒比例"),
    target_label: int = Query(8, ge=0, le=9, description="目标标签，范围0-9"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行mnist_backdoor_detection.py进行MNIST数据集后门注入与奇异值检测
    
    注意：此任务可能需要较长时间执行，使用异步方式处理
    """
    # 验证脚本文件是否存在
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'datacheck', 'mnist_backdoor_detection.py'
    )
    
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"脚本文件不存在: {script_path}")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_status[task_id] = {
        "status": "running",
        "result": None,
        "request": {
            "poison_ratio": poison_ratio,
            "target_label": target_label
        },
        "type": "mnist_backdoor_detection"
    }
    
    # 定义后台任务函数
    async def execute_mnist_backdoor_detection():
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                script_path,
                f"--poison_ratio={poison_ratio}",
                f"--target_label={target_label}"
            ]
            
            # 执行脚本
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=os.path.dirname(script_path)
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                # 解析输出结果
                result = {
                    "success": True,
                    "message": "MNIST后门奇异值检测执行成功",
                    "stdout": stdout,
                    "results_path": "results/"
                }
                tasks_status[task_id]["status"] = "completed"
                tasks_status[task_id]["result"] = result
                log.info(f"任务 {task_id} (mnist_backdoor_detection) 执行成功")
            else:
                error_message = f"执行失败: {stderr}"
                tasks_status[task_id]["status"] = "failed"
                tasks_status[task_id]["result"] = {
                    "success": False,
                    "message": error_message,
                    "stderr": stderr
                }
                log.error(f"任务 {task_id} (mnist_backdoor_detection) 执行失败: {stderr}")
                
        except Exception as e:
            error_message = str(e)
            log.exception(f"任务 {task_id} (mnist_backdoor_detection) 执行异常: {error_message}")
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["result"] = {
                "success": False,
                "message": error_message
            }
    
    # 添加到后台任务
    background_tasks.add_task(execute_mnist_backdoor_detection)
    
    return DataCheckResponse(
        success=True,
        message="MNIST后门奇异值检测任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/datacheck/strip_detection", summary="执行STRIP后门检测算法", response_model=DataCheckResponse)
async def strip_detection(
    poison_ratio: float = Query(0.01, ge=0.001, le=0.1, description="投毒比例"),
    reference_ratio: float = Query(0.01, ge=0.001, le=0.1, description="参考样本比例"),
    target_label: int = Query(8, ge=0, le=9, description="目标标签，范围0-9"),
    num_strip_samples: int = Query(100, ge=10, le=500, description="每个样本叠加的随机图像数量"),
    model_path: str = Query("../ag/MNIST_badnets.pth", description="预训练模型路径"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行strip_detection.py进行STRIP后门检测算法
    
    注意：此任务可能需要较长时间执行，使用异步方式处理
    """
    # 验证脚本文件是否存在
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'datacheck', 'strip_detection.py'
    )
    
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"脚本文件不存在: {script_path}")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_status[task_id] = {
        "status": "running",
        "result": None,
        "request": {
            "poison_ratio": poison_ratio,
            "reference_ratio": reference_ratio,
            "target_label": target_label,
            "num_strip_samples": num_strip_samples,
            "model_path": model_path
        },
        "type": "strip_detection"
    }
    
    # 定义后台任务函数
    async def execute_strip_detection():
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                script_path,
                f"--poison_ratio={poison_ratio}",
                f"--reference_ratio={reference_ratio}",
                f"--target_label={target_label}",
                f"--num_strip_samples={num_strip_samples}",
                f"--model_path={model_path}"
            ]
            
            # 执行脚本
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=os.path.dirname(script_path)
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                # 解析输出结果
                result = {
                    "success": True,
                    "message": "STRIP后门检测算法执行成功",
                    "stdout": stdout
                }
                tasks_status[task_id]["status"] = "completed"
                tasks_status[task_id]["result"] = result
                log.info(f"任务 {task_id} (strip_detection) 执行成功")
            else:
                error_message = f"执行失败: {stderr}"
                tasks_status[task_id]["status"] = "failed"
                tasks_status[task_id]["result"] = {
                    "success": False,
                    "message": error_message,
                    "stderr": stderr
                }
                log.error(f"任务 {task_id} (strip_detection) 执行失败: {stderr}")
                
        except Exception as e:
            error_message = str(e)
            log.exception(f"任务 {task_id} (strip_detection) 执行异常: {error_message}")
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["result"] = {
                "success": False,
                "message": error_message
            }
    
    # 添加到后台任务
    background_tasks.add_task(execute_strip_detection)
    
    return DataCheckResponse(
        success=True,
        message="STRIP后门检测算法任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/datacheck/task/{task_id}", summary="查询数据检查任务状态")
async def get_datacheck_task_status(task_id: str) -> Dict[str, Any]:
    """
    查询数据检查相关任务的执行状态
    """
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks_status[task_id]

@router.get("/datacheck/tasks", summary="获取所有数据检查任务列表")
async def list_datacheck_tasks():
    """
    获取所有数据检查相关任务的列表
    """
    return Success(data={
        "tasks": [
            {
                "task_id": task_id,
                "type": task_info.get("type", "unknown"),
                "status": task_info["status"],
                "request": task_info["request"]
            }
            for task_id, task_info in tasks_status.items()
        ],
        "total": len(tasks_status)
    })