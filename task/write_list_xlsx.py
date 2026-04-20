""" Step.4_1 - 根据 detail_df 进行汇总和聚类，并写出到数量汇总 xlsx """
import logging
from pathlib import Path
from typing import cast

import pandas as pd
from openpyxl.utils import get_column_letter

from config import config
from tools.logging_utils import log_set
from json_to_parquet import load_detail_parquet


def build_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """ 基于明细表生成汇总表，按材料名称和规格型号汇总 """
    _summary_df = (
        df.groupby(["五金材料名称", "规格型号"], dropna=False, as_index=False)
        .agg(数量=("汇总数量", "sum"))
        .rename(columns={"五金材料名称": "材料名称"})
        .sort_values(by=["材料名称", "规格型号"], kind="stable")
        .reset_index(drop=True)
    )

    _summary_df.insert(0, "序号", _summary_df.index + 1)
    return _summary_df[["序号", "材料名称", "规格型号", "数量"]]


def build_material_detail_df_map(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """ 按五金材料名称拆分明细表 """
    result: dict[str, pd.DataFrame] = {}

    for material_name, group_df in df.groupby("五金材料名称", sort=False):
        sub_df = group_df.reset_index(drop=True).copy()
        sub_df.insert(0, "序号", sub_df.index + 1)
        result[cast(str, material_name)] = sub_df

    return result


def set_worksheet_column_widths(ws, df: pd.DataFrame, col_widths: dict[str, float]) -> None:
    """按列名设置 worksheet 列宽"""
    for col_idx, col_name in enumerate(df.columns, start=1):
        width = col_widths.get(col_name)
        if width is None:
            continue
        col_letter = get_column_letter(col_idx)
        ws.column_dimensions[col_letter].width = width


def write_xlsx(output_file: Path, summary_df: pd.DataFrame, detail_dict: dict[str, pd.DataFrame]) -> None:
    summary_col_widths = {"序号": 8, "材料名称": 30, "规格型号": 25, "数量": 10}

    detail_col_widths = {"序号": 8, "图包名称": 60, "图纸名称": 50, "门型": 18, "门编号": 18, "洞口尺寸": 14, "构件尺寸": 14,
                         "樘数": 8, "五金材料名称": 14, "厂家": 9, "规格型号": 10, "表格数量": 10, "汇总数量": 10}

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="汇总表", index=False)
        summary_ws = writer.sheets["汇总表"]
        set_worksheet_column_widths(summary_ws, summary_df, summary_col_widths)

        for material_name, df in detail_dict.items():
            df.to_excel(writer, sheet_name=material_name, index=False)
            detail_ws = writer.sheets[material_name]
            set_worksheet_column_widths(detail_ws, df, detail_col_widths)


if __name__ == '__main__':
    log_set(logging.DEBUG, log_save=True, save_level=logging.WARNING, save_path=config.log_dir / "4_1_write_list_xlsx.log")

    detail_df = load_detail_parquet()
    write_xlsx(Path("door_hardware.xlsx"), build_summary_df(detail_df), build_material_detail_df_map(detail_df))
