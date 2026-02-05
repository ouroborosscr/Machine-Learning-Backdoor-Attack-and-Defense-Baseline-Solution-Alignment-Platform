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

class ModelCheckResponse(BaseModel):
    """模型检查任务响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="执行消息")
    task_id: str = Field(None, description="任务ID")

# 用于存储任务状态的字典（生产环境中应使用Redis等持久化存储）
tasks_status = {}

@router.get("/modelcheck/train_MNIST", summary="训练MNIST模型", response_model=ModelCheckResponse)
async def train_mnist(
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行train_MNIST.py进行MNIST模型训练
    
    注意：此任务可能需要较长时间执行，使用异步方式处理
    """
    # 验证脚本文件是否存在
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'modelcheck', 'train_MNIST.py'
    )
    
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"脚本文件不存在: {script_path}")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_status[task_id] = {
        "status": "running",
        "result": None,
        "request": {},
        "type": "train_MNIST"
    }
    
    # 定义后台任务函数
    async def execute_train_mnist():
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                script_path
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
                    "message": "MNIST模型训练执行成功",
                    "stdout": stdout,
                    "results_path": "mask/"
                }
                tasks_status[task_id]["status"] = "completed"
                tasks_status[task_id]["result"] = result
                log.info(f"任务 {task_id} (train_MNIST) 执行成功")
            else:
                error_message = f"执行失败: {stderr}"
                tasks_status[task_id]["status"] = "failed"
                tasks_status[task_id]["result"] = {
                    "success": False,
                    "message": error_message,
                    "stderr": stderr
                }
                log.error(f"任务 {task_id} (train_MNIST) 执行失败: {stderr}")
                
        except Exception as e:
            error_message = str(e)
            log.exception(f"任务 {task_id} (train_MNIST) 执行异常: {error_message}")
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["result"] = {
                "success": False,
                "message": error_message
            }
    
    # 添加到后台任务
    background_tasks.add_task(execute_train_mnist)
    
    return ModelCheckResponse(
        success=True,
        message="MNIST模型训练任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/modelcheck/detector", summary="执行后门检测", response_model=ModelCheckResponse)
async def detector(
    model_path: str = Query("../ag/MNIST_badnets.pth", description="模型路径"),
    image_path: str = Query("examples/images/mnist.pt", description="图像路径"),
    num_rounds: int = Query(100, ge=10, le=500, description="优化轮数"),
    lambd: float = Query(0.9, ge=0.1, le=2.0, description="正则化参数"),
    attack_type: str = Query("patch", description="攻击类型"),
    size_limit: int = Query(8, ge=1, le=28, description="大小限制"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行detector.py进行后门检测
    
    注意：此任务可能需要较长时间执行，使用异步方式处理
    """
    # 验证脚本文件是否存在
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'modelcheck', 'detector.py'
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
            "model_path": model_path,
            "image_path": image_path,
            "num_rounds": num_rounds,
            "lambd": lambd,
            "attack_type": attack_type,
            "size_limit": size_limit
        },
        "type": "detector"
    }
    
    # 定义后台任务函数
    async def execute_detector():
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                script_path,
                "-m", model_path,
                "-i", image_path,
                "-n", str(num_rounds),
                "-l", str(lambd),
                "-a", attack_type,
                "-s", str(size_limit)
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
                    "message": "后门检测执行成功",
                    "stdout": stdout,
                    "results_path": "results/"
                }
                tasks_status[task_id]["status"] = "completed"
                tasks_status[task_id]["result"] = result
                log.info(f"任务 {task_id} (detector) 执行成功")
            else:
                error_message = f"执行失败: {stderr}"
                tasks_status[task_id]["status"] = "failed"
                tasks_status[task_id]["result"] = {
                    "success": False,
                    "message": error_message,
                    "stderr": stderr
                }
                log.error(f"任务 {task_id} (detector) 执行失败: {stderr}")
                
        except Exception as e:
            error_message = str(e)
            log.exception(f"任务 {task_id} (detector) 执行异常: {error_message}")
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["result"] = {
                "success": False,
                "message": error_message
            }
    
    # 添加到后台任务
    background_tasks.add_task(execute_detector)
    
    return ModelCheckResponse(
        success=True,
        message="后门检测任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/modelcheck/generate_trigger", summary="生成对抗样本触发器", response_model=ModelCheckResponse)
async def generate_trigger(
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行generate_trigger.py生成对抗样本触发器
    
    注意：此任务可能需要较长时间执行，使用异步方式处理
    """
    # 验证脚本文件是否存在
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'modelcheck', 'generate_trigger.py'
    )
    
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"脚本文件不存在: {script_path}")
    
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_status[task_id] = {
        "status": "running",
        "result": None,
        "request": {},
        "type": "generate_trigger"
    }
    
    # 定义后台任务函数
    async def execute_generate_trigger():
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                script_path
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
                    "message": "对抗样本触发器生成成功",
                    "stdout": stdout,
                    "results_path": "results/"
                }
                tasks_status[task_id]["status"] = "completed"
                tasks_status[task_id]["result"] = result
                log.info(f"任务 {task_id} (generate_trigger) 执行成功")
            else:
                error_message = f"执行失败: {stderr}"
                tasks_status[task_id]["status"] = "failed"
                tasks_status[task_id]["result"] = {
                    "success": False,
                    "message": error_message,
                    "stderr": stderr
                }
                log.error(f"任务 {task_id} (generate_trigger) 执行失败: {stderr}")
                
        except Exception as e:
            error_message = str(e)
            log.exception(f"任务 {task_id} (generate_trigger) 执行异常: {error_message}")
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["result"] = {
                "success": False,
                "message": error_message
            }
    
    # 添加到后台任务
    background_tasks.add_task(execute_generate_trigger)
    
    return ModelCheckResponse(
        success=True,
        message="对抗样本触发器生成任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/modelcheck/task/{task_id}", summary="查询模型检查任务状态")
async def get_modelcheck_task_status(task_id: str) -> Dict[str, Any]:
    """
    查询模型检查相关任务的执行状态
    """
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks_status[task_id]

@router.get("/modelcheck/tasks", summary="获取所有模型检查任务列表")
async def list_modelcheck_tasks():
    """
    获取所有模型检查相关任务的列表
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