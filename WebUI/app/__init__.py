from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from starlette.staticfiles import StaticFiles

from app.api.v1.utils import refresh_api_list
from app.core.exceptions import SettingNotFound
from app.core.init_app import (
    init_menus,
    init_users,
    make_middlewares,
    modify_db,
    register_db,
    register_exceptions,
    register_routers,
)
from app.log import log
from app.models.system import Log
from app.models.system import LogType, LogDetailType

try:
    from app.settings import APP_SETTINGS
except ImportError:
    raise SettingNotFound("Can not import settings")


def create_app() -> FastAPI:
    if APP_SETTINGS.DEBUG:
        _app = FastAPI(
            title=APP_SETTINGS.APP_TITLE,
            description=APP_SETTINGS.APP_DESCRIPTION,
            version=APP_SETTINGS.VERSION,
            openapi_url="/openapi.json",
            middleware=make_middlewares(),
            lifespan=lifespan
        )
    else:
        _app = FastAPI(
            title=APP_SETTINGS.APP_TITLE,
            description=APP_SETTINGS.APP_DESCRIPTION,
            version=APP_SETTINGS.VERSION,
            openapi_url=None,
            middleware=make_middlewares(),
            lifespan=lifespan
        )
    register_db(_app)
    register_exceptions(_app)
    register_routers(_app, prefix="/api")

    redis = aioredis.from_url(url=APP_SETTINGS.REDIS_URL)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    return _app


@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_time = datetime.now()
    try:
        await modify_db()
        await init_menus()
        await refresh_api_list()
        await init_users()
        await Log.create(log_type=LogType.SystemLog, log_detail_type=LogDetailType.SystemStart)
        yield

    finally:
        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds() / 60
        log.info(f"App {_app.title} runtime: {runtime} min")  # noqa
        await Log.create(log_type=LogType.SystemLog, log_detail_type=LogDetailType.SystemStop)


app = create_app()

app.mount("/static", StaticFiles(directory=APP_SETTINGS.STATIC_ROOT), name="static")
