from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, Any
import os
import sys
import uuid
import subprocess
import json

from app.log import log

router = APIRouter()

class AGTrainingResponse(BaseModel):
    """AG任务响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="执行消息")
    task_id: str = Field(None, description="任务ID")

# 用于存储任务状态的字典（生产环境中应使用Redis等持久化存储）
tasks_status = {}

@router.get("/ag/generate_trigger", summary="执行生成trigger任务", response_model=AGTrainingResponse)
async def generate_trigger(
    target_label: int = Query(4, ge=0, le=9, description="目标标签，范围0-9"),
    num_epochs: int = Query(10, ge=1, le=100, description="优化轮数"),
    lr: float = Query(0.001, ge=0.00001, le=0.1, description="学习率"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行generate_trigger.py生成对抗样本trigger
    
    注意：此任务可能需要较长时间执行，使用异步方式处理
    """
    # 验证脚本文件是否存在
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'ag', 'generate_trigger.py'
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
            "target_label": target_label,
            "num_epochs": num_epochs,
            "lr": lr
        },
        "type": "generate_trigger"
    }
    
    # 定义后台任务函数
    async def execute_generate_trigger():
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                script_path,
                f"--target_label={target_label}",
                f"--num_epochs={num_epochs}",
                f"--lr={lr}"
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
                # 解析输出结果（假设脚本输出包含JSON格式的结果）
                result = {
                    "success": True,
                    "message": "Trigger生成成功",
                    "stdout": stdout,
                    "trigger_path": f"results/trigger_target_{target_label}.pt"
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
    
    return AGTrainingResponse(
        success=True,
        message="Trigger生成任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/ag/targeted_attack", summary="执行定向攻击任务", response_model=AGTrainingResponse)
async def targeted_attack(
    target_label: int = Query(4, ge=0, le=9, description="目标标签，范围0-9"),
    num_samples: int = Query(5, ge=1, le=100, description="样本数量"),
    epsilon: float = Query(0.05, ge=0.01, le=0.5, description="扰动大小限制"),
    num_steps: int = Query(10, ge=1, le=100, description="攻击步数"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行targeted_attack.py进行定向对抗攻击
    
    注意：此任务可能需要较长时间执行，使用异步方式处理
    """
    # 验证脚本文件是否存在
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'ag', 'targeted_attack.py'
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
            "target_label": target_label,
            "num_samples": num_samples,
            "epsilon": epsilon,
            "num_steps": num_steps
        },
        "type": "targeted_attack"
    }
    
    # 定义后台任务函数
    async def execute_targeted_attack():
        try:
            # 构建命令行参数
            cmd = [
                sys.executable,
                script_path,
                f"--target_label={target_label}",
                f"--num_samples={num_samples}",
                f"--epsilon={epsilon}",
                f"--num_steps={num_steps}"
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
                    "message": "定向攻击执行成功",
                    "stdout": stdout,
                    "visualization_path": f"attack_results/adversarial_examples_{target_label}.png"
                }
                tasks_status[task_id]["status"] = "completed"
                tasks_status[task_id]["result"] = result
                log.info(f"任务 {task_id} (targeted_attack) 执行成功")
            else:
                error_message = f"执行失败: {stderr}"
                tasks_status[task_id]["status"] = "failed"
                tasks_status[task_id]["result"] = {
                    "success": False,
                    "message": error_message,
                    "stderr": stderr
                }
                log.error(f"任务 {task_id} (targeted_attack) 执行失败: {stderr}")
                
        except Exception as e:
            error_message = str(e)
            log.exception(f"任务 {task_id} (targeted_attack) 执行异常: {error_message}")
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["result"] = {
                "success": False,
                "message": error_message
            }
    
    # 添加到后台任务
    background_tasks.add_task(execute_targeted_attack)
    
    return AGTrainingResponse(
        success=True,
        message="定向攻击任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/ag/task/{task_id}", summary="查询AG任务状态")
async def get_ag_task_status(task_id: str) -> Dict[str, Any]:
    """
    查询AG相关任务的执行状态
    """
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks_status[task_id]

@router.get("/ag/tasks", summary="获取所有AG任务列表")
async def list_ag_tasks():
    """
    获取所有AG相关任务的列表
    """
    return {
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
    }