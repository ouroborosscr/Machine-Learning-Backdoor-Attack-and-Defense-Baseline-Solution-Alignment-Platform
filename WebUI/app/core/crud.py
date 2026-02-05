from typing import Generic, NewType, TypeVar, Any

from pydantic import BaseModel
from tortoise.expressions import Q
from tortoise.models import Model

Total = NewType("Total", int)
ModelType = TypeVar("ModelType", bound=Model)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: type[ModelType]):
        self.model = model

    async def get(self, *args: Q, **kwargs) -> ModelType:
        return await self.model.get(*args, **kwargs)

    async def list(
            self,
            page: int | None,
            page_size: int | None,
            search: Q = Q(),
            order: list[str] | None = None,
            fields: list[str] | None = None,
            last_id: int | None = None,
            count_by_pk_field: bool = False
    ) -> tuple[Total, list[ModelType]]:
        order = order or []
        page = page or 1
        page_size = page_size or 10

        query = self.model.filter(search).distinct()
        if last_id:
            query = query.filter(id__gt=last_id)

        if fields:
            query = query.only(*fields)

        if count_by_pk_field:
            total = await query.values_list(self.model._meta.pk_attr, flat=True)
            total = len(set(total))
        else:
            total = await query.count()

        if last_id:
            result = await query.order_by(*order).limit(page_size)
        else:
            result = await query.offset((page - 1) * page_size).limit(page_size).order_by(*order)

        return Total(total), result

    async def create(self, obj_in: CreateSchemaType, exclude: set[str] | None = None) -> ModelType:
        if isinstance(obj_in, dict):
            obj_dict = obj_in
        else:
            obj_dict = obj_in.model_dump(exclude_unset=True, exclude_none=True, exclude=exclude)
        obj: ModelType = self.model(**obj_dict)
        await obj.save()
        return obj

    async def update(self, id: int, obj_in: UpdateSchemaType | dict[str, Any], exclude: set[str] | None = None) -> ModelType:
        if isinstance(obj_in, dict):
            obj_dict = obj_in
        else:
            obj_dict = obj_in.model_dump(exclude_unset=True, exclude_none=True, exclude=exclude)
        obj = await self.get(id=id)
        obj = obj.update_from_dict(obj_dict)

        await obj.save()
        return obj

    async def remove(self, id: int) -> None:
        obj = await self.get(id=id)
        await obj.delete()
