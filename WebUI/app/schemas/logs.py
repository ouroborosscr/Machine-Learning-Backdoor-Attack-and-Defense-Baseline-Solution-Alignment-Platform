from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field

from app.models.system import LogType


class BaseLog(BaseModel):
    log_type: Annotated[LogType | None, Field(alias="logType", description="日志类型")] = None
    log_detail_type: Annotated[str | None, Field(alias="logDetailType", description="日志详细")] = None
    by_user: Annotated[str | None, Field(alias="logUser", description="关联用户")] = None

    class Config:
        allow_extra = True
        populate_by_name = True


class BaseAPILog(BaseModel):
    ip_address: Annotated[str | None, Field(max_length=50, description="IP地址")] = None
    user_agent: Annotated[str | None, Field(max_length=255, description="User-Agent")] = None
    request_url: Annotated[str | None, Field(max_length=255, description="请求URL")] = None
    request_params: Annotated[dict | list | None, Field(description="请求参数")] = None
    request_data: Annotated[dict | list | None, Field(description="请求数据")] = None
    response_data: Annotated[dict | list | None, Field(description="响应数据")] = None
    response_status: Annotated[bool | None, Field(description="请求状态")] = None
    create_time: Annotated[datetime | None, Field(alias="creationTime", description="创建时间")] = None
    process_time: Annotated[float | None, Field(description="请求处理时间")] = None

    class Config:
        allow_extra = True
        populate_by_name = True


class LogSearch(BaseLog):
    current: Annotated[int | None, Field(description="页码")] = None
    size: Annotated[int | None, Field(description="每页数量")] = None
    log_type: Annotated[LogType | None, Field(alias="logType", description="日志类型")] = None
    log_detail_type: Annotated[str | None, Field(alias="logDetailType", description="日志详细")] = None
    by_user: Annotated[str | None, Field(alias="byUser", description="关联用户")] = None
    request_path: Annotated[str | None, Field(alias="requestPath", description="请求路径")] = None
    time_range: Annotated[list[datetime, datetime] | None, Field(alias="timeRange", description="时间范围")] = None
    response_code: Annotated[str | None, Field(alias="responseCode", description="业务状态码")] = None
    x_request_id: Annotated[str | None, Field(alias="xRequestId", description="x-request-id")] = None


class LogCreate(BaseLog):
    ...


class LogUpdate(BaseLog):
    ...


__all__ = ["BaseLog", "BaseAPILog", "LogSearch", "LogCreate", "LogUpdate"]
