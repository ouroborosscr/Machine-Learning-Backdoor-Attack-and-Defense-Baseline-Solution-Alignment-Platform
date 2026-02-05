from datetime import datetime, timedelta, timezone, timezone
UTC = timezone.utc

from fastapi import APIRouter
from fastapi_cache import JsonCoder
from fastapi_cache.decorator import cache

from app.log import log
from app.api.v1.utils import insert_log
from app.controllers.user import user_controller
from app.core.ctx import CTX_USER_ID
from app.core.dependency import DependAuth, check_token
from app.models.system import LogDetailType, LogType
from app.models.system import User, Role, Button, StatusType
from app.schemas.base import Fail, Success
from app.schemas.login import CredentialsSchema, JWTOut, JWTPayload
from app.settings import APP_SETTINGS
from app.utils.security import create_access_token

router = APIRouter()


@router.post("/login", summary="登录")
async def _(credentials: CredentialsSchema):
    user_obj: User | None = await user_controller.authenticate(credentials)  # 账号验证, 失败则触发异常返回请求错误
    # user_role_code_list = await user_obj.by_user_roles.values_list("role_code", flat=True)
    # all_login_role_codes = ["R_SUPER", "R_ADMIN", "R_USER"]
    # for user_role_code in user_role_code_list:
    #     if user_role_code in all_login_role_codes:
    #         break
    # else:
    #     log.info(f"用户越权登录, 用户名: {user_obj.user_name}")
    #     return Fail(msg="This user has no permission to login.")

    await user_controller.update_last_login(user_obj.id)
    payload = JWTPayload(
        data={"userId": user_obj.id, "userName": user_obj.user_name, "tokenType": "accessToken"},
        iat=datetime.now(UTC),
        exp=datetime.now(UTC)
    )
    access_token_payload = payload.model_copy(deep=True)
    access_token_payload.exp += timedelta(minutes=APP_SETTINGS.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_payload = payload.model_copy(deep=True)
    refresh_token_payload.data["tokenType"] = "refreshToken"
    refresh_token_payload.exp += timedelta(minutes=APP_SETTINGS.JWT_REFRESH_TOKEN_EXPIRE_MINUTES)
    data = JWTOut(
        access_token=create_access_token(data=access_token_payload),
        refresh_token=create_access_token(data=refresh_token_payload),
    )
    log.info(f"用户登录成功, 用户名: {user_obj.user_name}")
    await insert_log(log_type=LogType.UserLog, log_detail_type=LogDetailType.UserLoginSuccess, by_user_id=user_obj.id)
    return Success(data=data.model_dump(by_alias=True))


@router.post("/refresh-token", summary="刷新认证")
async def _(jwt_token: JWTOut):
    if not jwt_token.refresh_token:
        return Fail(code="4000", msg="The refreshToken is not valid.")
    status, code, data = check_token(jwt_token.refresh_token)
    if not status:
        return Fail(code=code, msg=data)

    user_id = data["data"]["userId"]
    user_obj = await user_controller.get(id=user_id)

    if data["data"]["tokenType"] != "refreshToken":
        return Fail(code="4000", msg="The token is not an refresh token.")

    if user_obj.status_type == StatusType.disable:
        await insert_log(log_type=LogType.UserLog, log_detail_type=LogDetailType.UserLoginForbid, by_user_id=user_id)
        return Fail(code="4040", msg="This user has been disabled.")

    await user_controller.update_last_login(user_id)
    payload = JWTPayload(
        data={"userId": user_obj.id, "userName": user_obj.user_name, "tokenType": "accessToken"},
        iat=datetime.now(UTC),
        exp=datetime.now(UTC)
    )

    access_token_payload = payload.model_copy(deep=True)
    access_token_payload.exp += timedelta(minutes=APP_SETTINGS.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_payload = payload.model_copy(deep=True)
    refresh_token_payload.data["tokenType"] = "refreshToken"
    refresh_token_payload.exp += timedelta(minutes=APP_SETTINGS.JWT_REFRESH_TOKEN_EXPIRE_MINUTES)

    data = JWTOut(
        access_token=create_access_token(data=access_token_payload),
        refresh_token=create_access_token(data=refresh_token_payload),
    )
    await insert_log(log_type=LogType.UserLog, log_detail_type=LogDetailType.UserAuthRefreshTokenSuccess, by_user_id=user_obj.id)
    return Success(data=data.model_dump(by_alias=True))


@cache(expire=60, coder=JsonCoder)
@router.get("/user-info", summary="查看用户信息", dependencies=[DependAuth])
async def _():
    user_id = CTX_USER_ID.get()
    user_obj: User = await user_controller.get(id=user_id)
    data = await user_obj.to_dict(exclude_fields=["id", "password", "create_time", "update_time"])

    user_roles: list[Role] = await user_obj.by_user_roles
    user_role_codes = [user_role.role_code for user_role in user_roles]

    user_role_button_codes = [b.button_code for b in await Button.all()] if "R_SUPER" in user_role_codes else [b.button_code for user_role in user_roles for b in await user_role.by_role_buttons]

    user_role_button_codes = list(set(user_role_button_codes))

    data.update({
        "userId": user_id,
        "roles": user_role_codes,
        "buttons": user_role_button_codes
    })
    await insert_log(log_type=LogType.UserLog, log_detail_type=LogDetailType.UserLoginGetUserInfo, by_user_id=user_obj.id)
    return Success(data=data)


@router.get("/error", summary="自定义后端错误")  # todo 使用限流器, 每秒最多一次
async def _(code: str, msg: str):
    if code == "9999":
        return Success(code="4040", msg="accessToken已过期")

    return Fail(code=code, msg=f"未知错误, code: {code} msg: {msg}")
