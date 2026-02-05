import os
import subprocess
import asyncio
import tempfile
import yaml
from typing import Dict, Any
from app.log import log

class BackdoorService:
    def __init__(self):
        # backdoor4目录的路径
        self.backdoor_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backdoor4')
        self.training_script = os.path.join(self.backdoor_dir, 'training.py')
    
    async def run_training(self, name: str, params_file: str, commit: str = 'none') -> Dict[str, Any]:
        """
        运行backdoor训练任务
        
        Args:
            name: 训练名称（如'mnist'）
            params_file: 参数文件路径（相对于backdoor4目录）
            commit: commit信息
        
        Returns:
            包含运行状态和输出的字典
        """
        try:
            # 构建完整的参数文件路径
            full_params_path = os.path.join(self.backdoor_dir, params_file)
            
            # 检查参数文件是否存在
            if not os.path.exists(full_params_path):
                return {
                    "success": False,
                    "message": f"参数文件不存在: {full_params_path}"
                }
            
            # 构建命令
            command = [
                "python",
                self.training_script,
                "--name", name,
                "--params", params_file,
                "--commit", commit
            ]
            
            log.info(f"开始执行backdoor训练任务: {command}")
            
            # 使用asyncio创建子进程，设置stdout和stderr为管道
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.backdoor_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 保存完整输出的变量
            stdout_full = []
            stderr_full = []
            
            # 定义异步函数来读取流并输出日志
            async def read_and_log(stream, log_func, output_list):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='replace').rstrip()
                    if line_str:
                        # 实时输出到日志
                        log_func(line_str)
                        # 保存到完整输出列表
                        output_list.append(line_str)
            
            # 同时读取stdout和stderr
            await asyncio.gather(
                read_and_log(process.stdout, log.info, stdout_full),
                read_and_log(process.stderr, log.error, stderr_full)
            )
            
            # 等待进程完成
            await process.wait()
            
            # 合并输出
            stdout_str = '\n'.join(stdout_full)
            stderr_str = '\n'.join(stderr_full)
            
            # 返回结果
            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "message": "训练任务执行完成" if process.returncode == 0 else f"训练任务执行失败，返回码: {process.returncode}"
            }
            
        except Exception as e:
            log.exception(f"执行backdoor训练任务时出错: {str(e)}")
            return {
                "success": False,
                "message": f"执行出错: {str(e)}"
            }
    
    def validate_params_file(self, params_file: str) -> bool:
        """
        验证参数文件是否有效
        """
        full_path = os.path.join(self.backdoor_dir, params_file)
        try:
            if not os.path.exists(full_path):
                return False
            with open(full_path, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)
            return True
        except Exception:
            return False

# 创建服务实例
backdoor_service = BackdoorService()