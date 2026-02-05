import http

import orjson
from fastapi.exceptions import (
    RequestValidationError,
    ResponseValidationError,
)
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from tortoise.exceptions import DoesNotExist, IntegrityError

from app.core.ctx import CTX_X_REQUEST_ID


class SettingNotFound(Exception):
    pass


class HTTPException(Exception):
    def __init__(
            self,
            code: int | str,
            msg: str | None = None
    ) -> None:
        if msg is None:
            msg = http.HTTPStatus(int(code)).phrase
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.code}: {self.msg}"

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}(code={self.code!r}, msg={self.msg!r})"


async def BaseHandle(req: Request, exc: Exception, handle_exc, code: int | str, msg: str | dict, status_code: int = 500, **kwargs) -> JSONResponse:
    headers = {"x-request-id": CTX_X_REQUEST_ID.get() or ""}
    request_body = await req.body() or {}
    try:
        request_body = orjson.loads(request_body)
    except (orjson.JSONDecodeError, UnicodeDecodeError):
        request_body = {}

    request_input = {"path": req.url.path, "query": req.query_params._dict, "body": request_body, "headers": dict(req.headers)}
    content = dict(code=str(code), x_request_id=headers["x-request-id"], msg=msg, input=request_input, **kwargs)
    if isinstance(exc, handle_exc):
        return JSONResponse(content=content, status_code=status_code)
    else:
        return JSONResponse(content=dict(code=str(code), msg=f"Exception handler Error, exc: {exc}"), status_code=status_code)


async def DoesNotExistHandle(req: Request, exc: Exception) -> JSONResponse:
    return await BaseHandle(req, exc, DoesNotExist, 404, f"Object has not found, exc: {exc}, path: {req.path_params}, query: {req.query_params}", 404)


async def IntegrityHandle(req: Request, exc: Exception) -> JSONResponse:
    return await BaseHandle(req, exc, IntegrityError, 500, f"IntegrityError，{exc}, path: {req.path_params}, query: {req.query_params}", 500)


async def HttpExcHandle(req: Request, exc: HTTPException) -> JSONResponse:
    return await BaseHandle(req, exc, HTTPException, exc.code, exc.msg, 200)


async def RequestValidationHandle(req: Request, exc: RequestValidationError) -> JSONResponse:
    return await BaseHandle(req, exc, RequestValidationError, 422, f"RequestValidationError", detail=exc.errors())


async def ResponseValidationHandle(req: Request, exc: ResponseValidationError) -> JSONResponse:
    return await BaseHandle(req, exc, ResponseValidationError, 422, f"ResponseValidationError", detail=exc.errors())
