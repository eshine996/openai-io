"""内部纯工具函数。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from ._types import NotGiven

__all__ = ["is_given", "merge_headers", "normalize_iterable_input", "remove_not_given"]


def is_given(value: object) -> bool:
    """判断参数是否为"调用方显式传入"（而非 NotGiven 哨兵）。"""
    return not isinstance(value, NotGiven)


def remove_not_given(data: dict[str, Any]) -> dict[str, Any]:
    """移除请求体中值为 ``NOT_GIVEN`` 的字段。"""
    return {key: value for key, value in data.items() if not isinstance(value, NotGiven)}


def normalize_iterable_input(value: str | Iterable[Any] | None) -> str | list[Any] | None:
    """把 prompt/embedding 输入及其内层 iterable 转成可 JSON 序列化的列表。"""
    if value is None or isinstance(value, str):
        return value
    items = list(value)
    return [
        list(cast("Iterable[Any]", item)) if isinstance(item, Iterable) and not isinstance(item, (str, bytes)) else item
        for item in items
    ]


def merge_headers(
    base: Mapping[str, str | None],
    *others: Mapping[str, str | None] | None,
) -> dict[str, str]:
    """合并请求头；后传入的值覆盖前值，None 删除同名头。"""
    merged: dict[str, str] = {}
    for headers in (base, *others):
        if headers is not None:
            for key, value in headers.items():
                existing = next((name for name in merged if name.lower() == key.lower()), None)
                if existing is not None:
                    del merged[existing]
                if value is not None:
                    merged[key] = value
    return merged
