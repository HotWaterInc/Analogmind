from typing import Dict, List, TypedDict


class ConnectionFrontier(TypedDict):
    start: str
    distance: float
    direction: List[float]
