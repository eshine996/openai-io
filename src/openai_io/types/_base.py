"""响应模型的共享配置。"""

from pydantic import BaseModel, ConfigDict


class APIResponseModel(BaseModel):
    """保留服务端新增字段，避免 API 演进时静默丢失数据。"""

    model_config = ConfigDict(extra="allow")
