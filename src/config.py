from pathlib import Path
from dataclasses import dataclass, field


@dataclass(slots=True)
class Config:
    # cache dir
    cache_dir: Path = Path(".cache")
    cache_json_dir: Path = field(init=False)

    # GLM BigModel
    glm_api_key: str = r"YOUR-GLM-API-KEY"

    def __post_init__(self):
        self.cache_json_dir = self.cache_dir / "json"

        self.cache_json_dir.mkdir(parents=True, exist_ok=True)


config = Config()
