from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import os
from pathlib import Path
from pydantic import BaseModel, Field

from app.log import log

router = APIRouter()

class FileDownloadResponse(BaseModel):
    """文件下载响应模型"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="执行消息")

# 报告文件存储的根目录
# 根据要求，report文件夹与file_download.py同级
REPORT_ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "report"

@router.get("/file-download/download", summary="下载报告文件")
async def download_report_file(
    name: str = Query(..., description="要下载的文件名，例如：MNIST评估报告.pdf")
) -> FileResponse:
    """
    根据文件名下载报告文件
    
    - **name**: 要下载的文件名，例如：MNIST评估报告.pdf
    """
    # 验证report目录是否存在
    if not REPORT_ROOT_DIR.exists():
        log.error(f"报告目录不存在: {REPORT_ROOT_DIR}")
        raise HTTPException(status_code=404, detail="报告存储目录不存在")
    
    # 构建完整的文件路径
    file_path = REPORT_ROOT_DIR / name
    
    # 验证文件是否存在
    if not file_path.exists():
        log.error(f"文件不存在: {file_path}")
        raise HTTPException(status_code=404, detail=f"文件 {name} 不存在")
    
    # 验证是否是文件（而不是目录）
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"指定的路径不是一个文件: {name}")
    
    # 记录下载日志
    log.info(f"用户下载文件: {name}")
    
    # 返回文件响应，设置文件名以确保下载时使用正确的文件名（支持中文）
    return FileResponse(
        path=str(file_path),
        filename=name,
        media_type="application/octet-stream"
    )