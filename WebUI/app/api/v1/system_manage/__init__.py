from fastapi import APIRouter

from app.core.dependency import DependPermission
from .apis import router as api_router
from .logs import router as log_router
from .menus import router as menu_router
from .roles import router as role_router
from .users import router as user_router
from .backdoor import router as backdoor_router
from .ag import router as ag_router
from .datacheck import router as datacheck_router
from .modelcheck import router as modelcheck_router
from .edit_logs import router as edit_logs_router
from .file_download import router as file_download_router  # 导入新路由

router_system_manage = APIRouter()
router_system_manage.include_router(log_router, tags=["日志管理"], dependencies=[DependPermission])
router_system_manage.include_router(api_router, tags=["API管理"], dependencies=[DependPermission])
router_system_manage.include_router(menu_router, tags=["菜单管理"], dependencies=[DependPermission])
router_system_manage.include_router(role_router, tags=["角色管理"], dependencies=[DependPermission])
router_system_manage.include_router(user_router, tags=["用户管理"], dependencies=[DependPermission])
router_system_manage.include_router(backdoor_router, tags=["后门训练"])  # 移除了dependencies参数
router_system_manage.include_router(ag_router, tags=["对抗攻击"])  # 移除了dependencies参数
router_system_manage.include_router(datacheck_router, tags=["数据检测"])  # 移除了dependencies参数
router_system_manage.include_router(modelcheck_router, tags=["模型检测"])  # 移除了dependencies参数
router_system_manage.include_router(edit_logs_router, tags=["编辑日志"])  # 移除了dependencies参数
router_system_manage.include_router(file_download_router, tags=["文件下载"])  # 注册新路由