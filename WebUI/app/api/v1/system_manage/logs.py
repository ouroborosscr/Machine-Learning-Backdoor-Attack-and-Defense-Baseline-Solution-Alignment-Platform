import json

from fastapi import APIRouter, Query
from tortoise.expressions import Q

from app.controllers import user_controller
from app.controllers.log import log_controller
from app.core.ctx import CTX_USER_ID
from app.models.system import LogType
from app.models.system import User, Role, Log, APILog
from app.schemas.base import Success, SuccessExtra, Fail
from app.schemas.logs import LogUpdate, LogSearch

router = APIRouter()


@router.post("/logs/all/", summary="查看日志列表")
async def _(log_in: LogSearch):
    if log_in.log_type is None:
        log_in.log_type = LogType.ApiLog

    if log_in.current is None:
        log_in.current = 1

    if log_in.size is None:
        log_in.size = 10

    q = Q()
    if log_in.log_type:
        q &= Q(log_type=log_in.log_type)
    if log_in.by_user:
        if (_by_user := await User.get_or_none(id=log_in.by_user)) is not None:
            q &= Q(by_user=_by_user)
        else:
            return Success(msg="用户不存在", code=2000)
    if log_in.log_detail_type:
        q &= Q(log_detail_type=log_in.log_detail_type)
    if log_in.request_path:
        q &= Q(api_log__request_path__contains=log_in.request_path)
    if log_in.response_code:
        q &= Q(api_log__response_code=log_in.response_code)
    if log_in.time_range:
        if len(log_in.time_range) != 2:
            return Success(msg="时间范围只能为两个值", code=2000)
        q &= Q(create_time__gt=log_in.time_range[0], create_time__lt=log_in.time_range[1])

    if log_in.x_request_id:
        q &= Q(x_request_id=log_in.x_request_id)

    user_id = CTX_USER_ID.get()
    user_obj = await user_controller.get(id=user_id)
    user_role_objs: list[Role] = await user_obj.by_user_roles
    user_role_codes = [role_obj.role_code for role_obj in user_role_objs]

    if "R_ADMIN" in user_role_codes and log_in.log_type not in [LogType.ApiLog, LogType.UserLog]:  # 管理员只能查看API日志和用户日志
        return Fail(msg="Permission Denied")
    elif "R_SUPER" not in user_role_codes and "R_ADMIN" not in user_role_codes and log_in.log_type != LogType.ApiLog:  # 非超级管理员和管理员只能查看API日志
        return Fail(msg="Permission Denied")

    total, log_objs = await log_controller.list(page=log_in.current, page_size=log_in.size, search=q, order=["-id"])
    records = []
    for obj in log_objs:
        await obj.fetch_related("api_log", "by_user")
        api_log: APILog = obj.api_log
        by_user: User = obj.by_user
        record = await obj.to_dict(exclude_fields=["by_user_id", "api_log_id"])
        if log_in.log_type == LogType.ApiLog:
            api_log_dict = await api_log.to_dict(exclude_fields=[])
            api_log_dict["requestParams"] = json.dumps(api_log_dict["requestParams"], ensure_ascii=False)
            api_log_dict["responseData"] = json.dumps(api_log_dict["responseData"], ensure_ascii=False)
            record.update(api_log_dict)
            record = {"logUser": "Request", **record}
        elif log_in.log_type == LogType.SystemLog:
            record["logUser"] = "System"
        else:
            if by_user:
                record["byUser"] = str(by_user.id)
                record["byUserInfo"] = await by_user.to_dict(include_fields=["id", "nick_name"])
            else:
                record["byUser"] = None

        records.append(record)
    data = {"records": records}
    return SuccessExtra(data=data, total=total, current=log_in.current, size=log_in.size)


@router.get("/logs/{log_id}", summary="查看日志")
async def _(log_id: int):
    log_obj = await log_controller.get(id=log_id)
    data = await log_obj.to_dict(exclude_fields=["id", "create_time", "update_time"])
    return Success(data=data)


@router.patch("/logs/{log_id}", summary="更新日志")
async def _(log_id: int, log_in: LogUpdate):
    await log_controller.update(id=log_id, obj_in=log_in)
    return Success(msg="Update Successfully")


@router.delete("/logs/{log_id}", summary="删除日志")
async def _(log_id: int):
    await log_controller.remove(id=log_id)
    return Success(msg="Deleted Successfully", data={"deleted_id": log_id})


@router.delete("/logs", summary="批量删除日志")
async def _(ids: str = Query(..., description="日志ID列表, 用逗号隔开")):
    log_ids = ids.split(",")
    deleted_ids = []
    for log_id in log_ids:
        log_obj = await Log.get(id=int(log_id))
        await log_obj.delete()
        deleted_ids.append(int(log_id))
    return Success(msg="Deleted Successfully", data={"deleted_ids": deleted_ids})
