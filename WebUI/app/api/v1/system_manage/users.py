from fastapi import APIRouter, Body
from tortoise.expressions import Q

from app.api.v1.utils import insert_log
from app.controllers.user import user_controller
from app.models.system import LogType, LogDetailType
from app.schemas.base import Success, SuccessExtra, CommonIds
from app.schemas.users import UserCreate, UserUpdate, UserSearch

router = APIRouter()


@router.post("/users/all/", summary="查看用户列表")
async def _(obj_in: UserSearch):
    q = Q()
    if obj_in.user_name:
        q &= Q(user_name__contains=obj_in.user_name)
    if obj_in.user_gender:
        q &= Q(user_gender=obj_in.user_gender)
    if obj_in.nick_name:
        q &= Q(nick_name__contains=obj_in.nick_name)
    if obj_in.user_phone:
        q &= Q(user_phone__contains=obj_in.user_phone)
    if obj_in.user_email:
        q &= Q(user_email__contains=obj_in.user_email)
    if obj_in.status_type:
        q &= Q(status_type=obj_in.status_type)
    if obj_in.by_user_role_code_list:
        q &= Q(by_user_roles__role_code__in=obj_in.by_user_role_code_list)

    total, user_objs = await user_controller.list(page=obj_in.current, page_size=obj_in.size, search=q, order=["id"])
    records = []
    for user_obj in user_objs:
        record = await user_obj.to_dict(exclude_fields=["password"])
        await user_obj.fetch_related("by_user_roles")
        user_role_code_list = [by_user_role.role_code for by_user_role in user_obj.by_user_roles]
        record.update({"byUserRoleCodeList": user_role_code_list})
        records.append(record)
    data = {"records": records}
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.UserGetList, by_user_id=0)
    return SuccessExtra(data=data, total=total, current=obj_in.current, size=obj_in.size)


@router.get("/users/{user_id}", summary="查看用户")
async def get_user(user_id: int):
    user_obj = await user_controller.get(id=user_id)
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.UserGetOne, by_user_id=0)
    return Success(data=await user_obj.to_dict(exclude_fields=["password"]))


@router.post("/users", summary="创建用户")
async def _(user_in: UserCreate):
    if not user_in.user_email:
        return Success(code="4090", msg="This email is invalid.")

    user_obj = await user_controller.get_by_email(user_in.user_email)
    if user_obj:
        return Success(code="4090", msg="The user with this email already exists in the system.")

    if not user_in.by_user_role_code_list:
        return Success(code="4090", msg="The user must have at least one role that exists.")

    new_user = await user_controller.create(obj_in=user_in)
    await user_controller.update_roles_by_code(new_user, user_in.by_user_role_code_list)
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.UserCreateOne, by_user_id=0)
    return Success(msg="Created Successfully", data={"created_id": new_user.id})


@router.patch("/users/{user_id}", summary="更新用户")
async def _(user_id: int, user_in: UserUpdate):
    user = await user_controller.update(user_id=user_id, obj_in=user_in)
    if not user_in.by_user_role_code_list:
        return Success(code="4090", msg="The user must have at least one role that exists.")

    await user_controller.update_roles_by_code(user, user_in.by_user_role_code_list)
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.UserUpdateOne, by_user_id=0)
    return Success(msg="Updated Successfully", data={"updated_id": user_id})


@router.delete("/users/{user_id}", summary="删除用户")
async def _(user_id: int):
    await user_controller.remove(id=user_id)
    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.UserDeleteOne, by_user_id=0)
    return Success(msg="Deleted Successfully", data={"deleted_id": user_id})


@router.delete("/users", summary="批量删除用户")
async def _(obj_in: CommonIds):
    deleted_ids = []
    for user_id in obj_in.ids:
        user_obj = await user_controller.get(id=int(user_id))
        await user_obj.delete()
        deleted_ids.append(int(user_id))

    await insert_log(log_type=LogType.AdminLog, log_detail_type=LogDetailType.UserBatchDeleteOne, by_user_id=0)
    return Success(msg="Deleted Successfully", data={"deleted_ids": deleted_ids})
