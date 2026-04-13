import base64
import logging
from pathlib import Path
from typing import cast, Iterator, Iterable

import pymupdf as fitz
from pymupdf import Page

from src.config import config
from src.tools.logging_utils import log_set
from src.tools.session import post_with_retry


DOOR_DRAW_BASE_DIR: Path = Path(r"DOOR-DRAWING-PATH")


def crop_pdf_region_to_png(pdf_filepath: Path, selection: tuple[tuple[float, float], tuple[float, float]],
                           dpi: int = 200) -> bytes:
    """
    从 PDF 指定页面裁剪区域并返回 PNG 二进制内容

    :param pdf_filepath: pdf file path
    :param selection: 归一化裁剪框 ((x0, y0), (x1, y1))
    :param dpi: dpi
    """
    (nx0, ny0), (nx1, ny1) = selection

    # verify selection
    if not all(0.0 <= v <= 1.0 for v in (nx0, ny0, nx1, ny1)):
        raise ValueError("selection 中的归一化坐标必须在 0 到 1 之间。")

    if nx0 >= nx1 or ny0 >= ny1:
        raise ValueError("selection 必须满足 x0 < x1 且 y0 < y1。")

    zoom = dpi / 72.0

    with fitz.open(pdf_filepath) as doc:
        page = cast(Page, doc[0])
        page_rect = page.rect
        width = float(page_rect.width)
        height = float(page_rect.height)

        # 横向
        if width >= height:
            clip = fitz.Rect(nx0 * width, ny0 * height, nx1 * width, ny1 * height)
            matrix = fitz.Matrix(zoom, zoom)

        # 竖向, 逆时针旋转 90 度
        else:
            # 旋转后的横向页面尺寸
            rotated_width = height
            rotated_height = width

            # selection 换算到“旋转后的页面坐标系”
            rx0 = nx0 * rotated_width
            ry0 = ny0 * rotated_height
            rx1 = nx1 * rotated_width
            ry1 = ny1 * rotated_height

            # 将“旋转后坐标系”的矩形映射回原页面坐标系
            clip = fitz.Rect(width - ry1, rx0, width - ry0, rx1)

            # 渲染时做逆时针旋转 90°
            matrix = fitz.Matrix(zoom, zoom).prerotate(90)

        # 与页面求交，防止浮点边界导致越界
        clip = clip & page_rect

        if clip.is_empty or clip.width <= 0 or clip.height <= 0:
            raise ValueError("根据 selection 计算得到的裁剪区域为空。")

        pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
        return pix.tobytes("png")


def get_door_drawing_path(base_path: Path, rglob_keys: Iterable[str]) -> Iterator[Path]:
    for _path in base_path.rglob("*.pdf"):
        if any(k in _path.stem for k in tuple(rglob_keys)):
            yield _path


if __name__ == '__main__':
    log_set(logging.DEBUG)

    for index, pdf_path in enumerate(get_door_drawing_path(DOOR_DRAW_BASE_DIR, ["FHM", "GM"])):
        logging.debug(f"{index} {pdf_path}")
        croped_png = crop_pdf_region_to_png(pdf_path, ((0.62, 0.54), (0.87, 0.98)), dpi=400)
        export_path: Path = config.cache_png_dir / pdf_path.parent.parent.name / pdf_path.with_suffix(".png").name
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_bytes(croped_png)

        png_base64 = base64.b64encode(croped_png).decode("utf-8")
