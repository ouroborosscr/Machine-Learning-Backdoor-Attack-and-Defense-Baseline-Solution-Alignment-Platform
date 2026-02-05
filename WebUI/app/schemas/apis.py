from typing import Annotated

from pydantic import BaseModel, Field

from app.models.system import StatusType


class BaseApi(BaseModel):
    api_path: Annotated[str | None, Field(alias="apiPath", title="请求路径", description="/api/v1/auth/login")] = None
    api_method: Annotated[str | None, Field(alias="apiMethod", title="请求方法", description="GET")] = None
    summary: Annotated[str | None, Field(title="API简介")] = None
    tags: Annotated[list[str] | None, Field(title="API标签")] = None
    status_type: Annotated[StatusType | None, Field(alias="statusType", title="API状态")] = None

    class Config:
        populate_by_name = True


class ApiSearch(BaseApi):
    current: Annotated[int | None, Field(title="页码")] = 1
    size: Annotated[int | None, Field(title="每页数量")] = 10


class ApiCreate(BaseApi):
    api_path: Annotated[str, Field(alias="apiPath", title="请求路径", description="/api/v1/auth/login")]
    api_method: Annotated[str, Field(alias="apiMethod", title="请求方法", description="GET")]


class ApiUpdate(BaseApi):
    ...
