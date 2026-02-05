from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from typing import Dict, Any

from app.services.backdoor_service import backdoor_service
from app.log import log

router = APIRouter()

class BackdoorTrainingRequest(BaseModel):
    """后门训练请求模型"""
    name: str = Field(..., description="训练名称", example="mnist")
    params_file: str = Field(..., description="参数文件路径", example="configs/mnist_params.yaml")
    commit: str = Field(default="none", description="commit信息")

class BackdoorTrainingResponse(BaseModel):
    """后门训练响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="执行消息")
    task_id: str = Field(None, description="任务ID")

# 用于存储任务状态的字典（生产环境中应使用Redis等持久化存储）
tasks_status = {}

@router.get("/backdoor/train", summary="执行后门训练任务", response_model=BackdoorTrainingResponse)
async def run_backdoor_training(
    name: str = Query(..., description="训练名称", example="mnist"),
    params_file: str = Query(..., description="参数文件路径", example="configs/mnist_params.yaml"),
    commit: str = Query(default="none", description="commit信息"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    执行backdoor4的训练任务
    
    注意：此任务可能需要较长时间执行，建议使用异步方式处理
    """
    # 验证参数文件
    if not backdoor_service.validate_params_file(params_file):
        raise HTTPException(status_code=400, detail=f"无效的参数文件: {params_file}")
    
    # 生成任务ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # 初始化任务状态
    tasks_status[task_id] = {
        "status": "running",
        "result": None,
        "request": {"name": name, "params_file": params_file, "commit": commit}
    }
    
    # 定义后台任务函数
    async def execute_training():
        try:
            # 执行训练
            result = await backdoor_service.run_training(
                name=name,
                params_file=params_file,
                commit=commit
            )
            # 更新任务状态
            tasks_status[task_id]["status"] = "completed"
            tasks_status[task_id]["result"] = result
        except Exception as e:
            log.exception(f"任务 {task_id} 执行失败: {str(e)}")
            tasks_status[task_id]["status"] = "failed"
            tasks_status[task_id]["result"] = {"success": False, "message": str(e)}
    
    # 添加到后台任务
    background_tasks.add_task(execute_training)
    
    return BackdoorTrainingResponse(
        success=True,
        message="训练任务已启动，请通过任务ID查询状态",
        task_id=task_id
    )

@router.get("/backdoor/task/{task_id}", summary="查询任务状态")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    查询后门训练任务的执行状态
    """
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return tasks_status[task_id]

@router.get("/backdoor/params", summary="获取可用的参数文件列表")
async def list_params_files():
    """
    获取backdoor4/configs目录下的参数文件列表
    """
    import os
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        'backdoor4', 'configs'
    )
    
    try:
        files = []
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                if file.endswith('.yaml') or file.endswith('.yml'):
                    files.append({
                        "name": file,
                        "path": f"configs/{file}"
                    })
        return {
            "success": True,
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取参数文件列表失败: {str(e)}")