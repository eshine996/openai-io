"""内部纯工具函数。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._types import NotGiven

__all__ = ["is_given", "merge_headers", "remove_not_given"]


def is_given(value: object) -> bool:
    """判断参数是否为"调用方显式传入"（而非 NotGiven 哨兵）。"""
    return not isinstance(value, NotGiven)


def remove_not_given(data: dict[str, Any]) -> dict[str, Any]:
    """移除请求体中值为 ``NOT_GIVEN`` 的字段。"""
    return {key: value for key, value in data.items() if not isinstance(value, NotGiven)}


def merge_headers(
    base: Mapping[str, str | None],
    *others: Mapping[str, str | None] | None,
) -> dict[str, str]:
    """合并多组请求头，后者覆盖前者；值为 None 的头被跳过。"""
    merged: dict[str, str] = {}
    for headers in (base, *others):
        if headers is not None:
            for key, value in headers.items():
                if value is not None:
                    merged[key] = value
    return merged
