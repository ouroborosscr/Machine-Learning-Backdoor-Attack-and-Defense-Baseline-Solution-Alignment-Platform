from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field


class CredentialsSchema(BaseModel):
    user_name: Annotated[str | None, Field(alias="userName", title="用户名")]
    password: Annotated[str | None, Field(title="密码")]

    class Config:
        allow_extra = True
        populate_by_name = True


class JWTOut(BaseModel):
    access_token: Annotated[str | None, Field(alias="token", title="请求token")] = None
    refresh_token: Annotated[str | None, Field(alias="refreshToken", title="刷新token")] = None

    class Config:
        allow_extra = True
        populate_by_name = True


class JWTPayload(BaseModel):
    data: dict
    iat: datetime
    exp: datetime

    # aud: str
    # iss: str
    # sub: str

    class Config:
        allow_extra = True
        populate_by_name = True


__all__ = ["CredentialsSchema", "JWTOut", "JWTPayload"]
