from fastapi import APIRouter
from fastapi_cache import JsonCoder
from fastapi_cache.decorator import cache

from app.controllers.menu import menu_controller
from app.core.ctx import CTX_USER_ID
from app.core.dependency import DependAuth
from app.models.system import Menu, Role, User, IconType
from app.schemas.base import Success

router = APIRouter()


async def build_route_tree(menus: list[Menu], parent_id: int = 0, simple: bool = False) -> list[dict]:
    """
    递归生成路由树
    :param menus:
    :param parent_id:
    :param simple: 是否简化返回数据
    :return:
    """
    tree = []
    for menu in menus:
        if menu.parent_id == parent_id:
            children = await build_route_tree(menus, menu.id, simple)
            await menu.fetch_related("active_menu")
            if simple:
                menu_dict = {
                    "name": menu.route_name,
                    "path": menu.route_path,
                    "component": menu.component,
                    "meta": {
                        "title": menu.menu_name,
                        "i18nKey": menu.i18n_key,
                        "order": menu.order,
                        "keepAlive": menu.keep_alive,
                        "icon": menu.icon,
                        "iconType": menu.icon_type,
                        "href": menu.href,
                        "activeMenu": menu.active_menu.route_name if menu.active_menu else None,
                        "multiTab": menu.multi_tab,
                        "fixedIndexInTab": menu.fixed_index_in_tab,
                    }
                }
                if menu.icon_type == IconType.local:
                    menu_dict["meta"]["localIcon"] = menu.icon
                    menu_dict["meta"].pop("icon")
                if menu.redirect:
                    menu_dict["redirect"] = menu.redirect
                if menu.component:
                    menu_dict["meta"]["layout"] = menu.component.split("$", maxsplit=1)[0]
                if menu.hide_in_menu and not menu.constant:
                    menu_dict["meta"]["hideInMenu"] = menu.hide_in_menu
            else:
                menu_dict = await menu.to_dict()
            if children:
                menu_dict["children"] = children
            tree.append(menu_dict)
    return tree


@cache(expire=60, coder=JsonCoder)
@router.get("/constant-routes", summary="查看常量路由(公共路由)")
async def _():
    """
    查看常量路由
    :return:
    """
    data = []
    menu_objs = await Menu.filter(constant=True, hide_in_menu=True)
    for menu_obj in menu_objs:
        route_data = {
            "name": menu_obj.route_name,
            "path": menu_obj.route_path,
            "component": menu_obj.component,
            "meta": {
                "title": menu_obj.menu_name,
                "i18nKey": menu_obj.i18n_key,
                "constant": menu_obj.constant,
                "hideInMenu": menu_obj.hide_in_menu
            }
        }

        if menu_obj.props:
            route_data["props"] = True

        data.append(route_data)

    return Success(data=data)


@cache(expire=60, coder=JsonCoder)
@router.get("/user-routes", summary="查看用户路由菜单", dependencies=[DependAuth])
async def _():
    """
    查看用户路由菜单, 超级管理员返回所有菜单
    :return:
    """
    user_id = CTX_USER_ID.get()
    user_obj = await User.get(id=user_id).prefetch_related("by_user_roles")
    user_roles: list[Role] = await user_obj.by_user_roles

    is_super = False
    role_home: str = "home"
    for user_role in user_roles:
        if user_role.role_code == "R_SUPER":
            is_super = True

        role_home_obj = await user_role.by_role_home.first()
        if role_home_obj:
            role_home = role_home_obj.route_name
            # break  # 注释掉, 取最后一个角色的首页

    if is_super:
        role_routes: list[Menu] = await Menu.filter(constant=False)
    else:
        role_routes: list[Menu] = []
        for user_role in user_roles:
            await user_role.fetch_related("by_role_menus", "by_role_menus__active_menu")
            user_role_routes: list[Menu] = await user_role.by_role_menus
            for user_role_route in user_role_routes:
                if not user_role_route.constant or user_role_route.hide_in_menu:
                    role_routes.append(user_role_route)

        menu_objs = role_routes.copy()
        while len(menu_objs) > 0:
            menu = menu_objs.pop()
            if menu.parent_id != 0:
                menu = await Menu.get(id=menu.parent_id)
                menu_objs.append(menu)
            else:
                role_routes.append(menu)

    role_routes = list(set(role_routes))  # 去重
    # 递归生成菜单
    menu_tree = await build_route_tree(role_routes, simple=True)
    data = {"home": role_home, "routes": menu_tree}
    return Success(data=data)


@router.get("/{route_name}/exists", summary="路由是否存在", dependencies=[DependAuth])
async def _(route_name: str):
    is_exists = await menu_controller.model.exists(route_name=route_name)
    return Success(data=is_exists)
