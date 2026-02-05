import contextvars

from starlette.background import BackgroundTasks

CTX_USER_ID: contextvars.ContextVar[int] = contextvars.ContextVar("user_id", default=0)
CTX_X_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("x_request_id", default="")
CTX_BG_TASKS: contextvars.ContextVar[BackgroundTasks | None] = contextvars.ContextVar("bg_task", default=None)
