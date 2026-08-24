from enum import Enum

from pydantic import BaseModel, Field


class Pipeline(str, Enum):
    DIRECT = "direct"
    FAST = "fast"
    DEEP = "deep"


class RouteDecision(BaseModel):

    pipeline: Pipeline = Field(
        description="Selected retrieval pipeline."
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in routing decision."
    )

    reason: str = Field(
        description="Reason for selecting the pipeline."
    )