import time

from app import refresh_api_list
from app.core.exceptions import SettingNotFound
from app.core.init_app import init_menus, init_users

try:
    from app.settings import APP_SETTINGS
except ImportError:
    raise SettingNotFound("Can not import settings")

from tortoise import Tortoise, run_async
from loguru import logger


async def init():
    await Tortoise.init(
        config=APP_SETTINGS.TORTOISE_ORM,
    )
    await Tortoise.generate_schemas()

    # 清空所有表
    # await Tortoise._drop_databases()
    # await Tortoise.close_connections()
    conn = Tortoise.get_connection("conn_system")

    # 获取所有表名
    table_count, tables = await conn.execute_query('SELECT name FROM sqlite_master WHERE type = "table" AND name NOT LIKE "sqlite_%";')
    # 删除所有表
    for table in tables:
        table_name = table["name"]
        print("table_name", table_name)
        if table_name != "aerich":
            # await conn.execute_query(f'DROP TABLE "{table_name}";')
            await conn.execute_query(f'delete from "{table_name}";')  # 清空数据
            await conn.execute_query(f'update sqlite_sequence SET seq = 0 where name = "{table_name}";')  # 自增长ID为0

    await init_menus()
    await refresh_api_list()
    await init_users()

    await Tortoise.close_connections()


while True:
    run_async(init())
    logger.info("Reset all tables")
    time.sleep(60 * 10)
