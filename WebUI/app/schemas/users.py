from typing import Annotated, Optional

from pydantic import BaseModel, Field

from app.models.system import GenderType, StatusType


class UserBase(BaseModel):
    user_name: Annotated[str | None, Field(alias="userName", title="用户名")] = None
    password: Annotated[str | None, Field(title="密码")] = None

    user_email: Annotated[str | None, Field(alias="userEmail", title="邮箱")] = None
    user_gender: Annotated[GenderType | None, Field(alias="userGender", title="性别")] = None
    nick_name: Annotated[str | None, Field(alias="nickName", title="昵称")] = None
    user_phone: Annotated[str | None, Field(alias="userPhone", title="手机号")] = None
    status_type: Annotated[StatusType | None, Field(alias="statusType", title="用户状态")] = None

    by_user_role_code_list: Annotated[list[str] | None, Field(alias="byUserRoleCodeList", title="用户角色编码列表")] = None

    class Config:
        populate_by_name = True


class UserSearch(UserBase):
    current: Annotated[int | None, Field(description="页码")] = 1
    size: Annotated[int | None, Field(description="每页数量")] = 10


class UserCreate(UserBase):
    ...


class UserUpdate(UserBase):
    password: Annotated[str, Field(title="密码")]  # type: ignore


class UpdatePassword(BaseModel):
    old_password: Annotated[str, Field(alias="oldPassword", title="旧密码")]
    new_password: Annotated[str, Field(alias="newPassword", title="新密码")]

    class Config:
        allow_extra = True
        populate_by_name = True


class UserRegister(BaseModel):
    user_name: Annotated[str, Field(alias="userName", title="用户名")]
    password: Annotated[str, Field(title="密码")]

    user_email: Annotated[str | None, Field(alias="userEmail", title="邮箱")] = None
    user_gender: Annotated[GenderType | None, Field(alias="userGender", title="性别")] = None
    nick_name: Annotated[str | None, Field(alias="nickName", title="昵称")] = None
    user_phone: Annotated[str | None, Field(alias="userPhone", title="手机号")] = None

    class Config:
        populate_by_name = True


__all__ = ["UserBase", "UserSearch", "UserCreate", "UserUpdate", "UpdatePassword"]
