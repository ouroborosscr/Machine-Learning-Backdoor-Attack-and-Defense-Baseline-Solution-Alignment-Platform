from typing import Annotated

from pydantic import BaseModel, Field

from app.models.system import StatusType


class RoleBase(BaseModel):
    role_name: Annotated[str | None, Field(alias="roleName", title="角色名称")] = None
    role_code: Annotated[str | None, Field(alias="roleCode", title="角色编码")] = None
    role_desc: Annotated[str | None, Field(alias="roleDesc", title="角色描述")] = None
    by_role_home: Annotated[str | None, Field(alias="byRoleHome", title="角色首页")] = None
    status_type: Annotated[StatusType | None, Field(alias="statusType", title="角色状态")] = None

    class Config:
        allow_extra = True
        populate_by_name = True


class RoleCreate(RoleBase):
    role_name: Annotated[str, Field(alias="roleName", title="角色名称")]
    role_code: Annotated[str, Field(alias="roleCode", title="角色编码")]


class RoleUpdate(RoleBase):
    ...


class RoleUpdateAuthrization(BaseModel):
    by_role_home_id: Annotated[int | None, Field(alias="byRoleHomeId", title="角色首页菜单id")] = None
    by_role_menu_ids: Annotated[list[int] | None, Field(alias="byRoleMenuIds", title="角色菜单列表")] = None
    by_role_api_ids: Annotated[list[int] | None, Field(alias="byRoleApiIds", title="角色API列表")] = None
    by_role_button_ids: Annotated[list[int] | None, Field(alias="byRoleButtonIds", title="角色按钮列表")] = None
