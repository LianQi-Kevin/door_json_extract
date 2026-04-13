from dataclasses import dataclass


@dataclass
class Config:
    base_url: str = "https://open.bigmodel.cn/api/"


config = Config()
