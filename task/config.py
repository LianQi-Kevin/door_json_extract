from pathlib import Path
from dataclasses import dataclass, field


@dataclass(slots=True)
class Config:
    # cache dir
    cache_dir: Path = Path(".cache").resolve()
    cache_png_dir: Path = field(init=False)
    cache_json_ocr_dir: Path = field(init=False)
    cache_json_hardware_dir: Path = field(init=False)
    cache_json_glm_response_dir: Path = field(init=False)

    # resources
    resources_dir: Path = Path("resources").resolve()

    # log dir
    log_dir: Path = Path("./log")

    # GLM BigModel
    glm_api_key: str = r"YOUR-GLM-API-KEY"

    def __post_init__(self):
        self.cache_json_ocr_dir = self.cache_dir / "json_ocr"
        self.cache_json_hardware_dir = self.cache_dir / "json_hardware"
        self.cache_json_glm_response_dir = self.cache_dir / "json_glm_response"
        self.cache_png_dir = self.cache_dir / "png"

        self.cache_json_ocr_dir.mkdir(parents=True, exist_ok=True)
        self.cache_png_dir.mkdir(parents=True, exist_ok=True)
        self.cache_json_hardware_dir.mkdir(parents=True, exist_ok=True)
        self.cache_json_glm_response_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


config = Config()
