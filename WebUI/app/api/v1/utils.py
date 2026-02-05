from fastapi.routing import APIRoute
from loguru import logger

from app.core.ctx import CTX_USER_ID, CTX_X_REQUEST_ID
from app.models.system import Api, Log
from app.models.system import LogType, LogDetailType


async def refresh_api_list():
    from app import app

    existing_apis = [(str(api.api_method.value), api.api_path) for api in await Api.all()]

    app_routes = [route for route in app.routes if isinstance(route, APIRoute)]
    app_routes_compared = [(list(route.methods)[0].lower(), route.path_format) for route in app_routes]

    for api_method, api_path in set(existing_apis) - set(app_routes_compared):
        logger.error(f"API Deleted {api_method} {api_path}")
        await Api.filter(api_method=api_method, api_path=api_path).delete()

    for route in app_routes:
        api_method = list(route.methods)[0].lower()
        api_path = route.path_format
        summary = route.summary
        tags = list(route.tags)
        await Api.update_or_create(api_path=api_path, api_method=api_method, defaults=dict(summary=summary, tags=tags))


async def generate_tags_recursive_list():
    from app import app

    app_routes = [route for route in app.routes if isinstance(route, APIRoute)]
    tags_list = [list(route.tags) for route in app_routes]
    unique_tags = list(set(tuple(tag) for tag in sorted(tags_list)))

    def build_tree():
        tree = []
        for tags in unique_tags:
            current_level = tree
            for tag in tags:
                existing_tag = next((item for item in current_level if item["value"] == tag), None)
                if not existing_tag:
                    new_tag = {"value": tag, "label": tag}
                    current_level.append(new_tag)
                    current_level = []
                else:
                    if existing_tag.get("children") is None:
                        existing_tag["children"] = []
                    current_level = existing_tag["children"]
        return tree

    return build_tree()


async def insert_log(log_type: LogType, log_detail_type: LogDetailType, by_user_id: int | None = None):
    """
    插入日志
    :param log_type:
    :param log_detail_type:
    :param by_user_id: 0为从上下文获取当前用户id, 需要请求携带token
    :return:
    """
    if by_user_id == 0 and (by_user_id := CTX_USER_ID.get()) == 0:
        by_user_id = None

    await Log.create(log_type=log_type, log_detail_type=log_detail_type, by_user_id=by_user_id, x_request_id=CTX_X_REQUEST_ID.get())
