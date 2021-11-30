from dataclasses import dataclass


@dataclass
class Issue:
    id: str
    ts: int
    summary: str
    description: str
    version: str
