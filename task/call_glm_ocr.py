""" Step.1 - 基于 png 请求 GLM-OCR, 获取 md_result, 保存到 <config.cache_json_ocr_dir> / <png filename>.json """
import json
import logging
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


from config import config
from src.models.GLM.layout_parsing import layout_parsing
from src.tools.logging_utils import log_set
from src.tools.file_utils import file_to_base64


def threading_main(file: Path):
    response = layout_parsing(config.glm_api_key, file_to_base64(file))
    export_path = config.cache_json_ocr_dir / file.parent.name / f"{file.stem}.json"
    logging.debug(f"finish request {file.parent.name}/{file.name}")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as json_f:
        json.dump(asdict(response), json_f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    log_set(logging.DEBUG, log_save=True, save_level=logging.WARNING, save_path=config.log_dir / "1_OCR.log")

    # load skip list
    skip_list: list[str] = []
    if Path("./skip_png.txt").is_file():
        with open("./skip_png.txt", "r", encoding="utf-8") as f:
            skip_list = f.readlines()
    skip_list = list(map(lambda x: x.strip(), skip_list))

    # multi request
    pool = ThreadPoolExecutor(max_workers=2)
    all_tasks = []

    for png_path in config.cache_png_dir.rglob("*.png"):
        if png_path.stem in skip_list:
            logging.debug(f"skip {png_path}")
            continue
        pool.submit(threading_main, png_path)

    wait(all_tasks, return_when=ALL_COMPLETED)
    pool.shutdown()
