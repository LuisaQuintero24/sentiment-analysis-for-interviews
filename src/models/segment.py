from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    speaker: str


class TranscribedSegment(Segment):
    text: str
    language: str = "unknown"
