""" Step.3 - 反序列化 GLM 5.1 提取的 json 字串, 对于数据进行汇总 """
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from config import config
from src.tools.logging_utils import log_set


@dataclass
class HardwareInfo:
    name: str
    corp: str
    model: str
    num: int


@dataclass
class FaceInfo:
    pull: Optional[str] = None
    push: Optional[str] = None


@dataclass(frozen=True, slots=True)
class DoorInfoForm:
    name: str
    num: str
    hole_size: str
    face_info: FaceInfo
    hardware: list[HardwareInfo]
    component_size: Optional[str] = None
    frame_material: Optional[str] = None
    leaf_material: Optional[str] = None
    threshold_material: Optional[str] = None
    core_material: Optional[str] = None
    glass: Optional[str] = None
    frame_seals: Optional[str] = None
    leaf_seals: Optional[str] = None
    bill_name: Optional[str] = None


def json_loader(file: Path) -> DoorInfoForm:
    with open(file, "r", encoding="utf-8") as _json_f:
        json_dict: dict = json.loads("".join(_json_f.readlines()))
    json_dict: dict = json.loads(json_dict["choices"][0]["message"]["content"])
    return DoorInfoForm(
        name=json_dict["门型"],
        bill_name=json_dict.get("结算门型"),
        num=json_dict["门编号"],
        hole_size=json_dict["洞口尺寸"],
        component_size=json_dict.get("构件尺寸"),
        frame_material=json_dict.get("门框材质"),
        leaf_material=json_dict.get("门扇材质"),
        threshold_material=json_dict.get("门槛材质"),
        core_material=json_dict.get("门芯"),
        glass=json_dict.get("玻璃"),
        frame_seals=json_dict.get("门框密封条"),
        leaf_seals=json_dict.get("门扇密封条"),
        hardware=[
            HardwareInfo(name=item["名称"], corp=item["品牌"], model=item["型号"], num=item["数量"])
            for item in json_dict["五金配置"]
        ],
        face_info=FaceInfo(
            push=cast(dict, json_dict["饰面颜色"]).get("推门侧"),
            pull=cast(dict, json_dict["饰面颜色"]).get("拉门侧")
        )
    )


if __name__ == '__main__':
    log_set(logging.DEBUG, log_save=True, save_level=logging.WARNING, save_path=config.log_dir / "3_json_to_xlsx.log")

    for json_path in config.cache_json_hardware_dir.rglob("*.json"):
        print(json_loader(json_path))
