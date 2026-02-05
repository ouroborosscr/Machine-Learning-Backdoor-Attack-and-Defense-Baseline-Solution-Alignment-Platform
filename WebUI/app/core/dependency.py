from typing import Any

import jwt
from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer

from app.core.ctx import CTX_USER_ID, CTX_X_REQUEST_ID
from app.core.exceptions import (
    HTTPException,
)
from app.log import log
from app.models.system import User, StatusType
from app.settings import APP_SETTINGS
from app.utils.tools import check_url

oauth2_schema = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


def check_token(token: str) -> tuple[bool, int, Any]:
    try:
        options = {"verify_signature": True, "verify_aud": False, "exp": True}
        decode_data = jwt.decode(token, APP_SETTINGS.SECRET_KEY, algorithms=[APP_SETTINGS.JWT_ALGORITHM], options=options)
        return True, 0, decode_data
    except jwt.DecodeError:
        return False, 8888, "无效的Token"
    except jwt.ExpiredSignatureError:
        return False, 4010, "登录已过期"
    except Exception as e:
        return False, 5000, f"{repr(e)}"


class AuthControl:
    @classmethod
    async def is_authed(cls, token: str = Depends(oauth2_schema)) -> User | None:
        if not token:
            raise HTTPException(code="4001", msg="Authentication failed, token does not exists in the request.")
        user_id = CTX_USER_ID.get()
        if user_id == 0:
            status, code, decode_data = check_token(token)
            if not status:
                raise HTTPException(code=code, msg=decode_data)

            if decode_data["data"]["tokenType"] != "accessToken":
                raise HTTPException(code="4040", msg="The token is not an access token")

            user_id = decode_data["data"]["userId"]

        user = await User.filter(id=user_id).first()
        if not user:
            raise HTTPException(code="4040", msg=f"Authentication failed, the user_id: {user_id} does not exists in the system.")
        CTX_USER_ID.set(int(user_id))
        return user


class PermissionControl:
    @classmethod
    async def has_permission(cls, request: Request, current_user: User = Depends(AuthControl.is_authed)) -> None:
        await current_user.fetch_related("by_user_roles")
        user_roles_codes: list[str] = [r.role_code for r in current_user.by_user_roles]
        if "R_SUPER" in user_roles_codes:  # 超级管理员
            return

        if not current_user.by_user_roles:
            raise HTTPException(code="4040", msg="The user is not bound to a role")

        method = request.method.lower()
        path = request.url.path

        apis = [await role.by_role_apis for role in current_user.by_user_roles]
        permission_apis = list(set((api.api_method.value, api.api_path, api.status_type) for api in sum(apis, [])))
        for (api_method, api_path, api_status) in permission_apis:
            if api_method == method and check_url(api_path, request.url.path):  # API权限检测通过
                if api_status == StatusType.disable:
                    raise HTTPException(code="4031", msg=f"The API has been disabled, method: {method} path: {path}")
                return

        log.error("*" * 20)
        log.error(f"Permission denied, method: {method.upper()} path: {path}")
        log.error(f"x-request-id: {CTX_X_REQUEST_ID.get()}")
        log.error("*" * 20)
        raise HTTPException(code="4032", msg=f"Permission denied, method: {method} path: {path}")


DependAuth = Depends(AuthControl.is_authed)
DependPermission = Depends(PermissionControl.has_permission)
