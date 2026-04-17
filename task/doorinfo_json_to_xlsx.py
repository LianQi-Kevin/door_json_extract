""" Step.3 - 反序列化 DoorInfoForm json 字串并写出到 xlsx """
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import pandas as pd
from openpyxl.utils import get_column_letter

from config import config
from glm_json_resave import DoorInfoForm, HardwareInfo, FaceInfo
from src.tools.logging_utils import log_set


@dataclass
class PathDoorInfo(DoorInfoForm):
    file_path: Optional[Path] = None


def resolve_door_count(item: PathDoorInfo) -> int:
    """ 联合 item.component_size 和 item.num 解析樘数 """
    component_size_text = ""
    if item.component_size is not None:
        component_size_text = str(item.component_size).strip()
    # 提取樘数
    component_matches = re.compile(r"(\d+)\s*樘").findall(component_size_text)
    if component_matches:
        return int(component_matches[-1])

    num_text = ""
    if item.num is not None:
        num_text = str(item.num).strip()
    if not num_text:
        return 1

    # 按 '/' 或 '、' 拆分 num
    tokens = [token.strip() for token in re.compile(r"[、/\n\r]+").split(num_text) if token.strip()]
    if len(tokens) <= 1:
        return 1

    return len(tokens)


def normalize_detail_df_by_rules(detail_df: pd.DataFrame, rules: Optional[list[dict]] = None) -> pd.DataFrame:
    """
    根据 rules 进行类型筛选修正

    rules 格式示例:
    [{
        "match": {"五金材料名称": "欧标执手锁", "规格型号": "EL3020"},
        "update": {"五金材料名称": "执手锁"},
    }]

    规则含义:
    - match: 命中的条件，多个列时按 AND 处理
    - update: 命中后要修改的列和值
    """
    if detail_df.empty or not rules:
        return detail_df.copy()

    result = detail_df.copy()

    for rule in rules:
        match = rule.get("match", {})
        update = rule.get("update", {})

        if not match or not update:
            continue

        mask = pd.Series(True, index=result.index)

        for col, expected_value in match.items():
            if col not in result.columns:
                mask &= False
                break
            mask &= result[col].eq(expected_value)

        if not mask.any():
            continue

        for col, new_value in update.items():
            if col in result.columns:
                result.loc[mask, col] = new_value

    return result


def attach_detail_df_by_rules(detail_df: pd.DataFrame, rules: Optional[list[dict]] = None) -> pd.DataFrame:
    """
    根据 rules 给 detail_df 附加字段

    rules 格式示例:
    [{
        "match": {"五金材料名称": "欧标合页", "规格型号": "4.5x4x3"},
        "attach": {"单位": "片", "导出分组": "合页类"},
    }]

    规则含义:
    - match: 命中的条件，多个列时按 AND 处理
    - attach: 命中后要附加/写入的列和值
    """
    if detail_df.empty or not rules:
        return detail_df.copy()

    result = detail_df.copy()

    for rule in rules:
        match = rule.get("match", {})
        attach = rule.get("attach", {})

        if not match or not attach:
            continue

        mask = pd.Series(True, index=result.index)

        for col, expected_value in match.items():
            if col not in result.columns:
                mask &= False
                break
            mask &= result[col].eq(expected_value)

        if not mask.any():
            continue

        for col, new_value in attach.items():
            if col not in result.columns:
                result[col] = ""
            result.loc[mask, col] = new_value

    return result


def build_hardware_detail_df(items: list[PathDoorInfo], rules: Optional[list[dict]] = None,
                             attach_rules: Optional[list[dict]] = None) -> pd.DataFrame:
    """ 把 list[PathDoorInfo] 展开为五金明细 DataFrame """
    def _safe_str(string: Optional[str]) -> str:
        if isinstance(string, str):
            return string.strip()
        return ""

    rows: list[dict] = []

    for item in items:
        door_count = resolve_door_count(item)
        for hw in item.hardware:
            table_qty = int(hw.num) if hw.num is not None else 0
            rows.append({
                "图包名称": cast(Path, item.file_path).parent.name,
                "图纸名称": cast(Path, item.file_path).stem,
                "门型": _safe_str(item.name),
                "门编号": _safe_str(item.num),
                "洞口尺寸": _safe_str(item.hole_size),
                "构件尺寸": _safe_str(item.component_size),
                "樘数": door_count,
                "五金材料名称": _safe_str(hw.name),
                "厂家": _safe_str(hw.corp),
                "规格型号": _safe_str(hw.model),
                "表格数量": table_qty,
                "汇总数量": door_count * table_qty
            })

    _detail_df = pd.DataFrame(rows, columns=["图包名称", "图纸名称", "门型", "门编号", "洞口尺寸", "构件尺寸", "樘数",
                                             "五金材料名称", "厂家", "规格型号", "表格数量", "汇总数量"])
    _detail_df = normalize_detail_df_by_rules(_detail_df, rules)
    _detail_df = normalize_detail_df_by_special_cases(_detail_df)
    _detail_df = attach_detail_df_by_rules(_detail_df, attach_rules)
    _detail_df = _detail_df.sort_values(by=["五金材料名称", "规格型号", "图包名称", "图纸名称", "门型"],
                                        kind="stable").reset_index(drop=True)
    return _detail_df


def normalize_detail_df_by_special_cases(detail_df: pd.DataFrame) -> pd.DataFrame:
    """ 处理带条件判断的特殊修正项 """
    if detail_df.empty:
        return detail_df.copy()

    result = detail_df.copy()

    mask = (
        result["图包名称"].eq("SLDS-BCEG-001-SDS-DW-DW021_0A 1号楼F4-F6层钢质防火门深化图")
        & result["樘数"].lt(3)
    )
    if mask.any():
        result.loc[mask, "樘数"] = 3
        result.loc[mask, "汇总数量"] = result.loc[mask, "樘数"] * result.loc[mask, "表格数量"]

    return result


def build_summary_df(detail_df: pd.DataFrame) -> pd.DataFrame:
    """ 基于明细表生成汇总表，按材料名称和规格型号汇总 """
    _summary_df = (
        detail_df.groupby(["五金材料名称", "规格型号"], dropna=False, as_index=False)
        .agg(数量=("汇总数量", "sum"))
        .rename(columns={"五金材料名称": "材料名称"})
        .sort_values(by=["材料名称", "规格型号"], kind="stable")
        .reset_index(drop=True)
    )

    _summary_df.insert(0, "序号", _summary_df.index + 1)
    return _summary_df[["序号", "材料名称", "规格型号", "数量"]]


def build_material_detail_df_map(detail_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ 按五金材料名称拆分明细表 """
    result: dict[str, pd.DataFrame] = {}

    for material_name, group_df in detail_df.groupby("五金材料名称", sort=False):
        sub_df = group_df.reset_index(drop=True).copy()
        sub_df.insert(0, "序号", sub_df.index + 1)
        result[cast(str, material_name)] = sub_df

    return result


def door_info_reloader(file: Path) -> DoorInfoForm:
    """ 从 json 重载 DoorInfoForm """
    with open(file, "r", encoding="utf-8") as _json_f:
        json_dict = json.loads("".join(_json_f.readlines()))
    result: DoorInfoForm = DoorInfoForm(**json_dict)
    result.hardware = [HardwareInfo(**_hardware) for _hardware in json_dict["hardware"]]
    return result


def df_builder_main(items: list[PathDoorInfo], rules: Optional[list[dict]] = None,
                    attach_rules: Optional[list[dict]] = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:    # 构造明细表
    detail_dataframe = build_hardware_detail_df(items, rules=rules, attach_rules=attach_rules)

    # 清理所有数量为 0 的五金项
    detail_dataframe = detail_dataframe[detail_dataframe["汇总数量"].gt(0)].copy()

    # 构造汇总表
    # summary_df = build_summary_df(detail_dataframe)

    # 手动筛选检查
    # matched_df = detail_dataframe[detail_dataframe["五金材料名称"].str.contains("顺位器", na=False)]
    # matched_df = summary_df[summary_df["规格型号"].eq("")]

    # pprint(summary_df.to_dict(orient="records"))

    # 根据材料名称拆分明细表
    # detail_dict = build_material_detail_df_map(detail_dataframe)

    return build_summary_df(detail_dataframe), build_material_detail_df_map(detail_dataframe)


def set_worksheet_column_widths(ws, df: pd.DataFrame, col_widths: dict[str, float]) -> None:
    """按列名设置 worksheet 列宽"""
    for col_idx, col_name in enumerate(df.columns, start=1):
        width = col_widths.get(col_name)
        if width is None:
            continue
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width


def write_xlsx(output_file: Path, summary_df: pd.DataFrame, detail_dict: dict[str, pd.DataFrame]) -> None:
    summary_col_widths = {
        "序号": 8,
        "材料名称": 30,
        "规格型号": 25,
        "数量": 10,
    }

    detail_col_widths = {
        "序号": 8,
        "图包名称": 60,
        "图纸名称": 50,
        "门型": 18,
        "门编号": 18,
        "洞口尺寸": 14,
        "构件尺寸": 14,
        "樘数": 8,
        "五金材料名称": 14,
        "厂家": 9,
        "规格型号": 10,
        "表格数量": 10,
        "汇总数量": 10,
    }

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="汇总表", index=False)
        summary_ws = writer.sheets["汇总表"]
        set_worksheet_column_widths(summary_ws, summary_df, summary_col_widths)

        for material_name, detail_df in detail_dict.items():
            detail_df.to_excel(writer, sheet_name=material_name, index=False)
            detail_ws = writer.sheets[material_name]
            set_worksheet_column_widths(detail_ws, detail_df, detail_col_widths)


if __name__ == '__main__':
    log_set(logging.DEBUG, log_save=True, save_level=logging.WARNING, save_path=config.log_dir / "4_json_to_xlsx.log")

    door_metadata_list: list[PathDoorInfo] = [
        PathDoorInfo(**vars(door_info_reloader(json_path)), file_path=json_path)
        for json_path in config.cache_json_hardware_dir.rglob("*.json")
    ]

    # manual add
    manual_metadata: list[PathDoorInfo] = [
        PathDoorInfo(
            name="TAF(A)D1230",
            num="F2-12#、F2-14#",
            hole_size="5200*3000",
            component_size="1200*3000*2樘",
            face_info=FaceInfo(pull="木纹转印（见认样）", push="木纹转印（见认样）"),
            hardware=[
                HardwareInfo(name="欧标轴承合页", corp="Allegion", model="4.5x4x3", num=5),
                HardwareInfo(name="欧标执手锁", corp="Allegion", model="EL3020+HC103+HT111", num=1),
                HardwareInfo(name="国标明装闭门器", corp="Allegion", model="BT 121", num=1)
            ],
            frame_material="镀锌钢板t1.5",
            leaf_material="镀锌钢板t1.5",
            core_material="珍珠岩板（350±35）kg/m³",
            frame_seals="FPJ-A-40x2",
            leaf_seals="FPJ-A-40x2",
            file_path=Path(r"SLDS-BCEG-001-SDS-DW-DW023_0A 1号楼F2层钢质防火门深化图\SLDS-BCEG-001-SDS-DW-FHM143_0A F2-12#、F2-13#、F2-14#门深化图.png")
        ),
        PathDoorInfo(
            name="TAF(A)D2530",
            num="F2-13#",
            hole_size="5200*3000",
            component_size="2500*3000*1樘",
            face_info=FaceInfo(pull="木纹转印（见认样）", push="木纹转印（见认样）"),
            hardware=[
                HardwareInfo(name="欧标轴承合页", corp="Allegion", model="4.5x4x3", num=10),
                HardwareInfo(name="欧标执手锁", corp="Allegion", model="EL3020+HC103+HT111", num=1),
                HardwareInfo(name="国标明装闭门器", corp="Allegion", model="BT 121", num=2),
                HardwareInfo(name="手动暗插销", corp="Allegion", model="FB458 12\"", num=2),
                HardwareInfo(name="L型顺位器", corp="Allegion", model="L-DC", num=1),
                HardwareInfo(name="防尘筒", corp="Allegion", model="DP01", num=1)
            ],
            frame_material="镀锌钢板t1.5",
            leaf_material="镀锌钢板t1.5",
            core_material="珍珠岩板（350±35）kg/m³",
            frame_seals="FPJ-A-40x2",
            leaf_seals="FPJ-A-40x2",
            file_path=Path(r"SLDS-BCEG-001-SDS-DW-DW023_0A 1号楼F2层钢质防火门深化图\SLDS-BCEG-001-SDS-DW-FHM143_0A F2-12#、F2-13#、F2-14#门深化图.png")
        ),
        PathDoorInfo(
            name="MGF(B)D1423.e.t",
            num="F1-15#",
            hole_size="2770*2300",
            component_size="1370*2300*1樘",
            face_info=FaceInfo(pull="灰色粉末喷涂", push="灰色粉末喷涂"),
            hardware=[
                HardwareInfo(name="欧标轴承合页", corp="Allegion", model="4.5x4x3", num=6),
                HardwareInfo(name="欧标通道功能锁", corp="Allegion", model="EL3040+HT111", num=1),
                HardwareInfo(name="欧标明装闭门器", corp="Allegion", model="BT 121", num=2),
                HardwareInfo(name="手动暗插销", corp="Allegion", model="FB458 12\"", num=2),
                HardwareInfo(name="L型顺位器", corp="Allegion", model="L-DC", num=1),
                HardwareInfo(name="防尘筒", corp="Allegion", model="DP01", num=1),
                HardwareInfo(name="门止", corp="Allegion", model="DP02", num=1)
            ],
            frame_material="镀锌钢板t1.5",
            leaf_material="镀锌钢板t1.5",
            core_material="珍珠岩板（350±35）kg/m³",
            glass="防火玻璃t30",
            frame_seals="FPJ-B-15.8x14.5",
            leaf_seals="FPJ-A-24x15",
            file_path=Path(r"SLDS-BCEG-001-SDS-DW-DW024_0A 1号楼F1层钢质防火门深化图\SLDS-BCEG-001-SDS-DW-FHM164_0A F1-15#、F1-16#门深化图.png")
        ),
        PathDoorInfo(
            name="MGF(B)D1423.e.t",
            num="F1-16#",
            hole_size="2770*2300",
            component_size="1370*2300*1樘",
            face_info=FaceInfo(pull="灰色粉末喷涂", push="灰色粉末喷涂"),
            hardware=[
                HardwareInfo(name="欧标轴承合页", corp="Allegion", model="4.5x4x3", num=6),
                HardwareInfo(name="美标电控推杠锁+外侧执手", corp="Allegion", model="AE-FSE-F-25-M-L510L-03 SC 630", num=1),
                HardwareInfo(name="美标推杠锁（上下杆）", corp="Allegion", model="F-25-V", num=1),
                HardwareInfo(name="欧标明装闭门器", corp="Allegion", model="BT 121", num=2),
                HardwareInfo(name="L型顺位器", corp="Allegion", model="L-DC", num=1),
                HardwareInfo(name="过线器", corp="Allegion", model="EPT1", num=1),
                HardwareInfo(name="推杠锁上锁扣下调支架", corp="Allegion", model="MB1", num=1)
            ],
            frame_material="镀锌钢板t1.5",
            leaf_material="镀锌钢板t1.5",
            core_material="珍珠岩板（350±35）kg/m³",
            glass="防火玻璃t30",
            frame_seals="FPJ-B-15.8x14.5",
            leaf_seals="FPJ-A-24x15",
            file_path=Path(r"SLDS-BCEG-001-SDS-DW-DW024_0A 1号楼F1层钢质防火门深化图\SLDS-BCEG-001-SDS-DW-FHM164_0A F1-15#、F1-16#门深化图.png")
        ),
    ]
    door_metadata_list.extend(manual_metadata)

    # batch replace
    rename_rules: list[dict[str, dict[str, str]]] = [
        {"match": {"规格型号": "DC 490"}, "update": {"规格型号": "DC490"}},
        {"match": {"五金材料名称": "欧标通道功能锁"}, "update": {"五金材料名称": "欧标通道锁"}},
        {"match": {"五金材料名称": "国标标执手锁"}, "update": {"五金材料名称": "国标执手锁"}},
        {"match": {"五金材料名称": "欧标标执手锁"}, "update": {"五金材料名称": "欧标执手锁"}},
        {"match": {"五金材料名称": "欧标合页"}, "update": {"五金材料名称": "欧标轴承合页"}},
        {"match": {"五金材料名称": "欧标轴承合页", "规格型号": "4.5*4*3"}, "update": {"规格型号": "4.5x4x3"}},
        {"match": {"五金材料名称": "美标合页"}, "update": {"五金材料名称": "美标轴承合页"}},
        {"match": {"五金材料名称": "美标轴承合页", "规格型号": "AH500 4.5*4*3.4"}, "update": {"规格型号": "AH500 4.5x4x3.4"}},
        {"match": {"规格型号": "BT 121", }, "update": {"规格型号": "BT121"}},
        {"match": {"规格型号": "SC 81", }, "update": {"规格型号": "SC81"}},
        {"match": {"五金材料名称": "国标明装闭门器", "规格型号": ""}, "update": {"规格型号": "BT121"}},
        {"match": {"五金材料名称": "电磁停门闭门器", "规格型号": ""}, "update": {"规格型号": "BT121"}},
        {"match": {"五金材料名称": "推杆锁上锁扣下调支架"}, "update": {"五金材料名称": "推杠锁上锁扣下调支架"}},
        {"match": {"五金材料名称": "推杠锁上锁扣下调支架", "规格型号": ""}, "update": {"规格型号": "MB1"}},
        {"match": {"五金材料名称": "手动暗插销", "规格型号": "FB458 12\"（上插加长）"}, "update": {"规格型号": "FB458 12\""}},
        {"match": {"五金材料名称": "美标闭门器"}, "update": {"五金材料名称": "美标明装闭门器"}},
        {"match": {"五金材料名称": "门止", "规格型号": "DS 02"}, "update": {"规格型号": "DS02"}},
        {"match": {"五金材料名称": "门止", "规格型号": "DP02"}, "update": {"规格型号": "DS02"}},
        {"match": {"五金材料名称": "美标明装闭门器", "规格型号": "SC81（只加衬板不开孔）"}, "update": {"规格型号": "SC81(只加衬板不开孔)"}},
        {"match": {"五金材料名称": "欧标执手锁", "规格型号": "HLQ020+HC103+HT111"}, "update": {"规格型号": "HL3020+HC103+HT111"}},
        {"match": {"规格型号": "EL3020+HC103+HT111"}, "update": {"规格型号": "EL3020+HT111+HC103"}},
    ]

    # 五金专业核对后调整映射原则
    rename_rules_2: list[dict[str, dict[str, str]]] = [
        # 4.5x4x3
        {"match": {"五金材料名称": "国标轴承合页"}, "update": {"五金材料名称": "欧标合页", "规格型号": "4.5x4x3"}},
        {"match": {"五金材料名称": "欧标轴承合页"}, "update": {"五金材料名称": "欧标合页", "规格型号": "4.5x4x3"}},
        {"match": {"五金材料名称": "美标轴承合页"}, "update": {"五金材料名称": "欧标合页", "规格型号": "4.5x4x3"}},
        {"match": {"五金材料名称": "超大门轴承合页"}, "update": {"五金材料名称": "欧标合页", "规格型号": "4.5x4x3"}},
        # S8000
        {"match": {"规格型号": "S8000"}, "update": {"五金材料名称": "欧标闭门器"}},
        # BT121
        {"match": {"五金材料名称": "欧标明装闭门器"}, "update": {"五金材料名称": "欧标闭门器", "规格型号": "BT121"}},
        {"match": {"五金材料名称": "国标明装闭门器"}, "update": {"五金材料名称": "国标闭门器", "规格型号": "BT121"}},
        {"match": {"五金材料名称": "明装闭门器"}, "update": {"五金材料名称": "国标闭门器", "规格型号": "BT121"}},
        {"match": {"五金材料名称": "闭门器"}, "update": {"五金材料名称": "国标闭门器", "规格型号": "BT121"}},
        {"match": {"五金材料名称": "闭门器+停门支臂"}, "update": {"五金材料名称": "国标闭门器", "规格型号": "BT121"}},
        # CH002
        {"match": {"五金材料名称": "暗拉环"}, "update": {"规格型号": "CH002"}},
        # COR-7G
        {"match": {"五金材料名称": "重力型顺位器"}, "update": {"五金材料名称": "重力顺位器", "规格型号": "COR-7G"}},
        {"match": {"规格型号": "COR-7G"}, "update": {"五金材料名称": "重力顺位器"}},
        # COR-X
        {"match": {"五金材料名称": "隐藏顺位器"}, "update": {"五金材料名称": "隐藏式顺位器", "规格型号": "COR-X"}},
        # DC490
        {"match": {"五金材料名称": "单门磁力锁"}, "update": {"五金材料名称": "单门磁力锁", "规格型号": "DC490"}},

        {"match": {"五金材料名称": "防尘筒"}, "update": {"五金材料名称": "防尘筒", "规格型号": "DP01"}},

        {"match": {"五金材料名称": "门止"}, "update": {"五金材料名称": "门止", "规格型号": "DS02"}},

        {"match": {"五金材料名称": "欧标固舌锁"}, "update": {"五金材料名称": "欧标固舌锁", "规格型号": "EL3010+HC102"}},

        {"match": {"五金材料名称": "欧标执手锁"}, "update": {"五金材料名称": "欧标双舌锁", "规格型号": "EL3020+HT111+HC103"}},
        {"match": {"五金材料名称": "欧标教室功能锁"}, "update": {"五金材料名称": "欧标双舌锁", "规格型号": "EL3020+HT111+HC103"}},

        {"match": {"五金材料名称": "国标通道锁"}, "update": {"五金材料名称": "欧标通道功能锁", "规格型号": "EL3040+HT111"}},
        {"match": {"五金材料名称": "欧标通道锁"}, "update": {"五金材料名称": "欧标通道功能锁", "规格型号": "EL3040+HT111"}},

        {"match": {"五金材料名称": "美标推杠锁+外侧执手"}, "update": {"五金材料名称": "美标插芯式逃生装置+外侧执手", "规格型号": "F-25-M-L"}},

        {"match": {"五金材料名称": "美标推杠锁（上下杆）"}, "update": {"五金材料名称": "美标上下杆式逃生装置", "规格型号": "F-25-V"}},

        {"match": {"五金材料名称": "手动暗插销"}, "update": {"五金材料名称": "手动暗插销", "规格型号": "FB458"}},

        {"match": {"五金材料名称": "美标电控推杠锁+外侧执手"}, "update": {"五金材料名称": "电控逃生装置+外侧执手", "规格型号": "FSE-F-25-M-L"}},

        {"match": {"五金材料名称": "国标固舌锁"}, "update": {"五金材料名称": "国标固舌锁", "规格型号": "HL1010+HC101"}},

        {"match": {"五金材料名称": "国标执手锁"}, "update": {"五金材料名称": "国标双舌锁", "规格型号": "HL1020+HC103+HT111"}},

        # L-DC
        {"match": {"规格型号": "L-DC"}, "update": {"五金材料名称": "顺位器"}},
        {"match": {"五金材料名称": "L型顺位器"}, "update": {"五金材料名称": "顺位器", "规格型号": "L-DC"}},
        {"match": {"五金材料名称": "明装顺位器"}, "update": {"五金材料名称": "顺位器", "规格型号": "L-DC"}},

        {"match": {"五金材料名称": "美标机电一体锁"}, "update": {"五金材料名称": "美标机电一体锁", "规格型号": "M80 EU"}},

        {"match": {"五金材料名称": "推拉手板"}, "update": {"五金材料名称": "推拉手板", "规格型号": "PP304-02"}},

        {"match": {"五金材料名称": "过线器"}, "update": {"五金材料名称": "过线器", "规格型号": "PT01"}},

        {"match": {"五金材料名称": "酒店智能锁"}, "update": {"五金材料名称": "入户门电子锁", "规格型号": "RT"}},

        {"match": {"五金材料名称": "美标明装闭门器"}, "update": {"五金材料名称": "美标闭门器", "规格型号": "SC81A"}},

        {"match": {"五金材料名称": "电插锁"}, "update": {"五金材料名称": "电插锁", "规格型号": "465"}},

        {"match": {"五金材料名称": "明装电磁闭门器"}, "update": {"五金材料名称": "电磁闭门器", "规格型号": ""}},
        {"match": {"五金材料名称": "电磁停门闭门器"}, "update": {"五金材料名称": "电磁闭门器", "规格型号": ""}},
        {"match": {"五金材料名称": "电磁门吸"}, "update": {"五金材料名称": "电磁闭门器", "规格型号": ""}},

        {"match": {"五金材料名称": "推杠锁上锁扣下调支架"}, "update": {"五金材料名称": "顺位器支架"}},
    ]

    rename_rules.extend(rename_rules_2)

    attach_rules: list[dict[str, dict[str, str]]] = [
        {"match": {"五金材料名称": "入户门电子锁"}, "attach": {"单位": "套"}},
        {"match": {"五金材料名称": "单门磁力锁"}, "attach": {"单位": "套"}},
        {"match": {"五金材料名称": "国标双舌锁"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "国标固舌锁"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "国标闭门器"}, "attach": {"单位": "台"}},
        {"match": {"五金材料名称": "手动暗插销"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "推拉手板"}, "attach": {"单位": "付"}},
        {"match": {"五金材料名称": "顺位器支架"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "暗拉环"}, "attach": {"单位": "付"}},
        {"match": {"五金材料名称": "欧标双舌锁"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "欧标合页"}, "attach": {"单位": "片"}},
        {"match": {"五金材料名称": "欧标固舌锁"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "欧标通道功能锁"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "欧标闭门器"}, "attach": {"单位": "台"}},
        {"match": {"五金材料名称": "电控逃生装置+外侧执手"}, "attach": {"单位": "套"}},
        {"match": {"五金材料名称": "电插锁"}, "attach": {"单位": "套"}},
        {"match": {"五金材料名称": "电磁闭门器"}, "attach": {"单位": "台"}},
        {"match": {"五金材料名称": "美标上下杆式逃生装置"}, "attach": {"单位": "套"}},
        {"match": {"五金材料名称": "美标插芯式逃生装置+外侧执手"}, "attach": {"单位": "套"}},
        {"match": {"五金材料名称": "美标机电一体锁"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "美标闭门器"}, "attach": {"单位": "套"}},
        {"match": {"五金材料名称": "过线器"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "重力顺位器"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "门止"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "防尘筒"}, "attach": {"单位": "个"}},
        {"match": {"五金材料名称": "隐藏式顺位器"}, "attach": {"单位": "根"}},
        {"match": {"五金材料名称": "顺位器"}, "attach": {"单位": "个"}},
    ]

    # get xlsx write-use item
    # summary_df, detail_dict = df_builder_main(door_metadata_list, rename_rules)

    # write xlsx
    # write_xlsx(Path("door_hardware.xlsx"), *df_builder_main(door_metadata_list, rename_rules))
