from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class Turn(TypedDict):
    role: str
    content: str


Trajectory = list[Turn]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    request: dict
    response: dict
