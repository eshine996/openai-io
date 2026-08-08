"""内部类型与哨兵值。

参考 openai SDK 的设计：使用 ``NotGiven`` 哨兵区分"调用方未传该参数"
与"调用方显式传了 None"。前者表示该字段不写入请求体（使用服务端默认值），
后者表示显式发送 null。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final

__all__ = ["NOT_GIVEN", "Headers", "NotGiven", "Query"]


class NotGiven:
    """一个永远不会被误认为合法值的哨兵类型。

    ``NotGiven`` 是单例风格的：任何位置的 ``NotGiven()`` 都指向同一个内部实例，
    因此可以放心用于默认参数比较（``param is not NOT_GIVEN``）。
    """

    _instance: NotGiven | None = None

    def __new__(cls) -> NotGiven:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NOT_GIVEN"

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> NotGiven:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> NotGiven:
        return self


NOT_GIVEN: Final[NotGiven] = NotGiven()

# 允许直接导入使用
NotGivenType = NotGiven

#: 附加请求头，值为 None 表示删除该头。
Headers = Mapping[str, str | None]

#: 查询参数。
Query = Mapping[str, str | int | float | bool | None]

#: 默认 base_url，与 openai SDK 保持一致。
DEFAULT_BASE_URL: Final[str] = "https://api.openai.com/v1"

#: 环境变量名。
API_KEY_ENV_VAR: Final[str] = "OPENAI_API_KEY"
ORG_ID_ENV_VAR: Final[str] = "OPENAI_ORG_ID"


def get_env(key: str) -> str | None:
    """读取环境变量，空字符串视为未设置。"""
    value = os.environ.get(key)
    return value if value else None
