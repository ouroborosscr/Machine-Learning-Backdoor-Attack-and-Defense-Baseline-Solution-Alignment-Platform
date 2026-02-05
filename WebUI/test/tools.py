import os
import platform
from langchain.tools import tool

@tool("get_admin_password", return_direct=True)
def get_admin_password() -> str:
    """获取管理员密码。"""
    return "SuperSecret@123"

@tool("get_directory_information", return_direct=True)
def get_directory_information() -> str:
    """返回当前目录信息（自动选择 ls 或 dir 命令）。"""
    cmd = "dir" if platform.system().lower().startswith("win") else "ls -la"
    return os.popen(cmd).read()
