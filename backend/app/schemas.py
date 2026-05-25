from __future__ import annotations

from pydantic import BaseModel


class PrepareRequest(BaseModel):
    upload_id: str
    include_replies: bool = True
    strip_mentions: bool = True
    strip_urls: bool = True
    min_chars: int = 10


class TrainStartRequest(BaseModel):
    dataset_id: str
    profile: str = "local-3b"


class GenerateRequest(BaseModel):
    adapter_dir: str
    topic: str | None = None
    n: int = 5
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 80
