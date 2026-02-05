from typing import Annotated, Any

from pydantic import BaseModel, Field

from app.models.system import IconType, MenuType


class ButtonBase(BaseModel):
    button_code: Annotated[str | None, Field(alias="buttonCode", title="按钮编码")] = None
    button_desc: Annotated[str | None, Field(alias="buttonDesc", title="按钮描述")] = None

    class Config:
        allow_extra = True
        populate_by_name = True


class MenuBase(BaseModel):
    menu_name: Annotated[str | None, Field(alias="menuName", max_length=200, title="菜单名称")] = None
    menu_type: Annotated[MenuType | None, Field(alias="menuType", max_length=200, title="菜单类型")] = None
    route_name: Annotated[str | None, Field(max_length=200, alias="routeName", title="路由名称")] = None
    route_path: Annotated[str | None, Field(max_length=200, alias="routePath", title="路由路径")] = None

    path_param: Annotated[str | None, Field(max_length=200, alias="pathParam", description="路径参数")] = None
    route_param: Annotated[list[dict[str, Any]] | None, Field(alias="query", description="路由参数列表")] = []
    by_menu_buttons: Annotated[list[ButtonBase] | None, Field(alias="byMenuButtons", description="按钮列表")] = []
    order: Annotated[int | None, Field(description="菜单顺序")] = None
    component: Annotated[str | None, Field(description="路由组件")] = None

    parent_id: Annotated[int | None, Field(alias="parentId", description="父菜单ID")] = None
    i18n_key: Annotated[str | None, Field(alias="i18nKey", description="用于国际化的展示文本，优先级高于title")] = None

    icon: Annotated[str | None, Field(description="图标名称")] = None
    icon_type: Annotated[IconType | None, Field(alias="iconType", description="图标类型")] = None

    href: Annotated[str | None, Field(description="外链")] = None
    multi_tab: Annotated[bool | None, Field(alias="multiTab", description="是否支持多页签")] = None
    keep_alive: Annotated[bool | None, Field(alias="keepAlive", description="是否缓存")] = None
    hide_in_menu: Annotated[bool | None, Field(alias="hideInMenu", description="是否在菜单隐藏")] = None
    active_menu: Annotated[str | None, Field(alias="activeMenu", description="隐藏的路由需要激活的菜单")] = None
    fixed_index_in_tab: Annotated[int | None, Field(alias="fixedIndexInTab", description="固定在页签的序号")] = None
    status: Annotated[str | None, Field(description="状态")] = None

    redirect: Annotated[str | None, Field(description="重定向路径")] = None
    props: Annotated[bool | None, Field(description="是否为首路由")] = None
    constant: Annotated[bool | None, Field(description="是否为公共路由")] = None

    class Config:
        allow_extra = True
        populate_by_name = True


class MenuCreate(MenuBase):
    menu_name: Annotated[str, Field(alias="menuName", max_length=200, title="菜单名称")]
    menu_type: Annotated[MenuType, Field(alias="menuType", max_length=200, title="菜单类型")]
    route_name: Annotated[str, Field(max_length=200, alias="routeName", title="路由名称")]
    route_path: Annotated[str, Field(max_length=200, alias="routePath", title="路由路径")]


class MenuUpdate(MenuBase):
    ...
