from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawFile:
    path: Path
    extension: str
    content: str