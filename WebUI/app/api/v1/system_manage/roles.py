from fastapi import APIRouter, Query
from tortoise.expressions import Q

from app.api.v1.utils import insert_log
from app.controllers import role_controller
from app.controllers.menu import menu_controller
from app.models.system import Api, Button, Role
from app.models.system import LogType, LogDetailType
from app.schemas.base import Success, SuccessExtra
from app.schemas.roles import RoleCreate, RoleUpdate, RoleUpdateAuthrization

router = APIRouter()


@router.get("/roles", summary="查看角色列表")
async def _(
        current: int = Query(1, description="页码"),
        size: int = Query(10, description="每页数量"),
        roleName: str = Query(None, description="角色名称"),
        roleCode: str = Query(None, description="角色编码"),
        status: str = Query(None, description="用户状态")
):
    q = Q()
    if roleName:
        q &= Q(role_name__contains=roleName)
    if roleCode:
        q &= Q(role_code__contains=roleCode)
    if status:
        q &= Q(status__contains=status)

    total, role_objs = await role_controller.list(page=current, page_size=size, search=q, order=["id"])
    records = [await role_obj.to_dict() for role_obj in role_objs]  # exclude_fields=["role_desc"]
    data = {"records": records}
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleGetList, by_user_id=0)
    return SuccessExtra(data=data, total=total, current=current, size=size)


@router.get("/roles/{role_id}", summary="查看角色")
async def get_role(role_id: int):
    role_obj: Role = await role_controller.get(id=role_id)
    data = await role_obj.to_dict()
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleGetOne, by_user_id=0)
    return Success(data=data)


@router.post("/roles", summary="创建角色")
async def _(role_in: RoleCreate):
    role = await role_controller.model.exists(role_code=role_in.role_code)
    if role:
        return Success(code="4090", msg="The role with this code already exists in the system.")

    new_user = await role_controller.create(obj_in=role_in)
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleCreateOne, by_user_id=0)
    return Success(msg="Created Successfully", data={"created_id": new_user.id})


@router.patch("/roles/{role_id}", summary="更新角色")
async def _(role_id: int, role_in: RoleUpdate):
    await role_controller.update(id=role_id, obj_in=role_in)
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleUpdateOne, by_user_id=0)
    return Success(msg="Updated Successfully", data={"updated_id": role_id})


@router.delete("/roles/{role_id}", summary="删除角色")
async def _(role_id: int):
    await role_controller.remove(id=role_id)
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleDeleteOne, by_user_id=0)
    return Success(msg="Deleted Successfully", data={"deleted_id": role_id})


@router.delete("/roles", summary="批量删除角色")
async def _(ids: str = Query(..., description="角色ID列表, 用逗号隔开")):
    role_ids = ids.split(",")
    deleted_ids = []
    for role_id in role_ids:
        role_obj = await role_controller.get(id=int(role_id))
        await role_obj.delete()
        deleted_ids.append(int(role_id))
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleBatchDeleteOne, by_user_id=0)
    return Success(msg="Deleted Successfully", data={"deleted_ids": deleted_ids})


@router.get("/roles/{role_id}/menus", summary="查看角色菜单")
async def _(role_id: int):
    role_obj = await Role.get(id=role_id).prefetch_related("by_role_home")
    if role_obj.role_code == "R_SUPER":
        menu_objs = await menu_controller.model.filter(constant=False)
    else:
        menu_objs = await role_obj.by_role_menus
    data = {"byRoleHomeId": role_obj.by_role_home.id, "byRoleMenuIds": [menu_obj.id for menu_obj in menu_objs]}
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleGetMenus, by_user_id=0)
    return Success(data=data)


@router.patch("/roles/{role_id}/menus", summary="更新角色菜单")
async def _(role_id: int, role_in: RoleUpdateAuthrization):
    if role_in.by_role_home_id:
        role_obj = await role_controller.update(id=role_id, obj_in=dict(by_role_home_id=role_in.by_role_home_id))
        if role_in.by_role_menu_ids:
            menu_objs = await menu_controller.get_by_id_list(id_list=role_in.by_role_menu_ids)
            if not menu_objs:
                return Success(msg="获取角色菜单对象失败", code=2000)

            await role_obj.by_role_menus.clear()
            while len(menu_objs) > 0:  # 递归添加子父菜单
                menu_obj = menu_objs.pop()
                await role_obj.by_role_menus.add(menu_obj)
                if menu_obj.parent_id != 0:  # 是否为子菜单
                    menu_objs.append(await menu_controller.get(id=menu_obj.parent_id))  # 添加子菜单的父菜单
        else:
            await role_obj.by_role_menus.clear()  # 去除所有角色菜单

    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleUpdateMenus, by_user_id=0)
    return Success(msg="Updated Successfully", data={"by_role_menu_ids": role_in.by_role_menu_ids, "by_role_home_id": role_in.by_role_home_id})


@router.get("/roles/{role_id}/buttons", summary="查看角色按钮")
async def _(role_id: int):
    role_obj = await role_controller.get(id=role_id)
    if role_obj.role_code == "R_SUPER":
        button_objs = await Button.all()
    else:
        button_objs = await role_obj.by_role_buttons

    data = {"byRoleButtonIds": [button_obj.id for button_obj in button_objs]}
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleGetButtons, by_user_id=0)
    return Success(data=data)


@router.patch("/roles/{role_id}/buttons", summary="更新角色按钮")
async def _(role_id: int, role_in: RoleUpdateAuthrization):
    role_obj = await role_controller.get(id=role_id)
    if role_in.by_role_button_ids is not None:
        await role_obj.by_role_buttons.clear()
        for button_id in role_in.by_role_button_ids:
            button_obj = await Button.get(id=button_id)
            await role_obj.by_role_buttons.add(button_obj)

    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleUpdateButtons, by_user_id=0)
    return Success(msg="Updated Successfully", data={"by_role_button_ids": role_in.by_role_button_ids})


@router.get("/roles/{role_id}/apis", summary="查看角色API")
async def _(role_id: int):
    role_obj = await role_controller.get(id=role_id)
    if role_obj.role_code == "R_SUPER":
        api_objs = await Api.all()
    else:
        api_objs = await role_obj.by_role_apis

    data = {"byRoleApiIds": [api_obj.id for api_obj in api_objs]}
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleGetApis, by_user_id=0)
    return Success(data=data)


@router.patch("/roles/{role_id}/apis", summary="更新角色API")
async def _(role_id: int, role_in: RoleUpdateAuthrization):
    role_obj = await role_controller.get(id=role_id)
    if role_in.by_role_api_ids is not None:
        await role_obj.by_role_apis.clear()
        for api_id in role_in.by_role_api_ids:
            api_obj = await Api.get(id=api_id)
            await role_obj.by_role_apis.add(api_obj)

    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.RoleUpdateApis, by_user_id=0)
    return Success(msg="Updated Successfully", data={"by_role_api_ids": role_in.by_role_api_ids})
