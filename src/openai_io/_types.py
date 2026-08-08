"""内部类型与哨兵值。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final

__all__ = ["NOT_GIVEN", "Headers", "NotGiven", "Query"]


class NotGiven:
    """表示参数未传的哨兵值。"""

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

#: ``NotGiven`` 的别名。
NotGivenType = NotGiven

#: 附加请求头，值为 None 表示删除该头。
Headers = Mapping[str, str | None]

#: 查询参数。
Query = Mapping[str, str | int | float | bool | None]

#: 默认 base_url。
DEFAULT_BASE_URL: Final[str] = "https://api.openai.com/v1"

#: 环境变量名。
API_KEY_ENV_VAR: Final[str] = "OPENAI_API_KEY"
ORG_ID_ENV_VAR: Final[str] = "OPENAI_ORG_ID"


def get_env(key: str) -> str | None:
    """读取环境变量，空字符串视为未设置。"""
    value = os.environ.get(key)
    return value if value else None
