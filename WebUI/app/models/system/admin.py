from tortoise import fields

from .utils import BaseModel, TimestampMixin, GenderType, StatusType, IconType, MenuType, MethodType, LogType, LogDetailType


class User(BaseModel, TimestampMixin):
    id = fields.IntField(pk=True, description="用户id")
    user_name = fields.CharField(max_length=20, unique=True, description="用户名称")
    password = fields.CharField(max_length=128, description="密码")
    nick_name = fields.CharField(max_length=30, null=True, description="昵称")
    user_gender = fields.CharEnumField(enum_type=GenderType, default=GenderType.unknow, description="性别")
    user_email = fields.CharField(max_length=255, unique=True, null=True, description="邮箱")
    user_phone = fields.CharField(max_length=20, null=True, description="电话")
    last_login = fields.DatetimeField(null=True, description="最后登录时间")
    status_type = fields.CharEnumField(enum_type=StatusType, default=StatusType.enable, description="状态")

    by_user_roles: fields.ManyToManyRelation['Role'] = fields.ManyToManyField("app_system.Role", related_name="by_role_users")

    class Meta:
        table = "users"
        table_description = "用户表"
        indexes = [
            ("user_name",),
            ("nick_name",),
            ("user_gender",),
            ("user_email",),
            ("user_phone",),
            ("status_type",),
        ]


class Role(BaseModel, TimestampMixin):
    id = fields.IntField(pk=True, description="角色id")
    role_name = fields.CharField(max_length=20, unique=True, description="角色名称")
    role_code = fields.CharField(max_length=20, unique=True, description="角色编码")
    role_desc = fields.CharField(max_length=500, null=True, blank=True, description="角色描述")
    by_role_home: fields.ForeignKeyRelation['Menu'] = fields.ForeignKeyField("app_system.Menu", related_name=None, description="角色首页")
    status_type = fields.CharEnumField(enum_type=StatusType, default=StatusType.enable, description="状态")

    by_role_menus: fields.ManyToManyRelation['Menu'] = fields.ManyToManyField("app_system.Menu", related_name="by_menu_roles")
    by_role_apis: fields.ManyToManyRelation['Api'] = fields.ManyToManyField("app_system.Api", related_name="by_api_roles")
    by_role_buttons: fields.ManyToManyRelation['Button'] = fields.ManyToManyField("app_system.Button", related_name="by_button_roles")
    by_role_users: fields.ReverseRelation['User']

    class Meta:
        table = "roles"
        table_description = "角色表"
        indexes = [
            ("role_name",),
            ("role_code",),
            ("status_type",),
        ]


class Api(BaseModel, TimestampMixin):
    id = fields.IntField(pk=True, description="API id")
    api_path = fields.CharField(max_length=500, description="API路径")
    api_method = fields.CharEnumField(MethodType, description="请求方法")
    summary = fields.CharField(max_length=500, null=True, description="请求简介")
    tags = fields.JSONField(max_length=500, null=True, description="API标签")
    status_type = fields.CharEnumField(enum_type=StatusType, default=StatusType.enable, description="状态")

    by_api_roles: fields.ReverseRelation['Role']

    class Meta:
        table = "apis"
        table_description = "API表"
        indexes = [
            ("api_path",),
            ("api_method",),
            ("summary",),
        ]


class Menu(BaseModel, TimestampMixin):
    id = fields.IntField(pk=True, description="菜单id")
    menu_name = fields.CharField(max_length=100, description="菜单名称")
    menu_type = fields.CharEnumField(MenuType, description="菜单类型")
    route_name = fields.CharField(unique=True, max_length=100, description="路由名称")
    route_path = fields.CharField(unique=True, max_length=200, description="路由路径")

    path_param = fields.CharField(null=True, max_length=200, description="路径参数")
    route_param = fields.JSONField(null=True, description="路由参数, List[dict]")
    order = fields.IntField(default=0, description="菜单顺序")
    component = fields.CharField(null=True, max_length=100, description="路由组件")
    parent_id = fields.IntField(default=0, max_length=10, description="父菜单ID")
    i18n_key = fields.CharField(null=True, max_length=100, description="用于国际化的展示文本，优先级高于title")
    icon = fields.CharField(null=True, max_length=100, description="图标名称")
    icon_type = fields.CharEnumField(IconType, null=True, description="图标类型")
    href = fields.CharField(null=True, max_length=200, description="外链")
    multi_tab = fields.BooleanField(default=False, description="是否支持多页签")
    keep_alive = fields.BooleanField(default=False, description="是否缓存")
    hide_in_menu = fields.BooleanField(default=False, description="是否在菜单隐藏")
    active_menu: fields.ForeignKeyRelation['Menu'] = fields.ForeignKeyField("app_system.Menu", related_name=None, null=True, description="隐藏的路由需要激活的菜单")
    fixed_index_in_tab = fields.IntField(null=True, max_length=10, description="固定在页签的序号")
    status_type = fields.CharEnumField(enum_type=StatusType, default=StatusType.enable, description="菜单状态")
    redirect = fields.CharField(null=True, max_length=200, description="重定向路径")
    props = fields.BooleanField(default=False, description="是否为首路由")
    constant = fields.BooleanField(default=False, description="是否为公共路由")

    by_menu_buttons: fields.ManyToManyRelation['Button'] = fields.ManyToManyField("app_system.Button", related_name="by_button_menus")
    by_menu_roles: fields.ReverseRelation['Role']

    class Meta:
        table = "menus"
        table_description = "菜单表"


class Button(BaseModel, TimestampMixin):
    id = fields.IntField(pk=True, description="按钮id")
    button_code = fields.CharField(max_length=200, description="按钮编码")
    button_desc = fields.CharField(max_length=200, description="按钮描述")
    status_type = fields.CharEnumField(enum_type=StatusType, default=StatusType.enable, description="状态")

    by_button_menus: fields.ReverseRelation['Menu']
    by_button_roles: fields.ReverseRelation['Role']

    class Meta:
        table = "buttons"


class Log(BaseModel):
    id = fields.IntField(pk=True, description="日志id")
    log_type = fields.CharEnumField(LogType, description="日志类型")
    by_user: fields.ForeignKeyRelation['User'] = fields.ForeignKeyField(null=True, model_name="app_system.User", related_name=None, related_nameon_delete=fields.NO_ACTION, description="关联专员")
    api_log: fields.OneToOneRelation['APILog'] = fields.OneToOneField("app_system.APILog", null=True, related_name=None, on_delete=fields.SET_NULL, description="API日志")
    log_detail_type = fields.CharEnumField(LogDetailType, null=True, description="日志详情类型")
    create_time = fields.DatetimeField(auto_now_add=True, description="创建时间")
    x_request_id = fields.CharField(null=True, max_length=32, description="请求id")

    class Meta:
        table = "logs"
        table_description = "日志表"
        indexes = [
            ("log_type",),
            ("by_user",),
            ("log_detail_type",),
            ("x_request_id",),
        ]


class APILog(BaseModel):
    id = fields.IntField(pk=True, description="API日志id")
    x_request_id = fields.CharField(max_length=32, description="请求id")
    ip_address = fields.CharField(null=True, max_length=60, description="IP地址")
    user_agent = fields.CharField(null=True, max_length=500, description="User-Agent")
    request_domain = fields.CharField(max_length=200, description="请求域名")
    request_path = fields.CharField(max_length=500, description="请求路径")
    request_params = fields.JSONField(null=True, description="请求参数")
    request_data = fields.JSONField(null=True, description="请求体数据")
    response_data = fields.JSONField(null=True, description="响应数据")
    response_code = fields.CharField(null=True, max_length=6, description="业务状态码")
    create_time = fields.DatetimeField(auto_now_add=True, description="创建时间")
    process_time = fields.FloatField(null=True, description="请求处理时间")

    class Meta:
        table = "api_logs"
        table_description = "API日志"
        indexes = [
            ("create_time",),
            ("process_time",),
            ("x_request_id",),
            ("request_path",),
            ("response_code",),
        ]


__all__ = [
    "User",
    "Role",
    "Api",
    "Menu",
    "Button",
    "Log",
    "APILog"
]
