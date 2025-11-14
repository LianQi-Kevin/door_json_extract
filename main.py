import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Union, Optional
from dataclasses import dataclass

import numpy as np
import pymupdf
from paddleocr import PaddleOCRVL

from requests_tools import post_with_retry

# CHAT-GLM 配置
GLM_API_KEY: str = r"your-api-key"
CHAT_COMPLETIONS_URL: str = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL_NAME: str = "glm-4-plus"


# cache data
@dataclass
class cacheData:
    pdf_path: str
    array: np.ndarray = None
    markdown_text: str = None
    extracted_json: dict = None


CACHE_DATA_DICT: dict[str, cacheData] = {}

# thread lock
LOCK_1 = threading.Lock()


# def paddle_ocr_vl(detection: Union[np.ndarray, str, list[Union[np.ndarray, str]]]) -> Union[str, list[str]]:
def paddle_ocr_vl(detection: Union[np.ndarray, str]) -> str:
    paddle_ocr_vl_pipeline = PaddleOCRVL(
        # 版面区域检测排序模型
        layout_detection_model_name="PP-DocLayoutV2",
        # layout_detection_model_dir=r"./paddle_models/PP-DocLayoutV2",
        # 多模态识别模型
        vl_rec_model_name="PaddleOCR-VL-0.9B",
        # vl_rec_model_dir=r"./paddle_models/PaddleOCR-VL",
        # vl_rec_backend="vllm-server",  # remote vl inference server
        # vl_rec_server_url="https://your.deploy.server:port/v1",  # remote vl inference server url
        # vl_rec_max_concurrency=8,
        # 推理设备
        device="gpu:0",
    )

    # return [r.markdown for r in PADDLE_OCR_VL_PIPELINE.predict(detection, use_queues=True)]
    return [r.markdown for r in paddle_ocr_vl_pipeline.predict(detection, use_queues=True)][0]


def big_model_completions(markdown_text: str) -> dict:
    global GLM_API_KEY, CHAT_COMPLETIONS_URL, MODEL_NAME
    # 定义要提取的JSON schema
    json_schema: dict = {
        "门编号": "",
        "门型": "",
        "洞口尺寸": "",
        "构件尺寸": "",
        "门框材质": "",
        "门扇材质": "",
        "门槛材质": "",
        "防火门芯": "",
        "玻璃": "",
        "门框密封条": "",
        "门扇密封条": "",
        "五金配置组名称": "HW-01",
        "五金配置": [
            {"名称": "", "品牌": "", "型号": "", "数量": 0}
        ],
        "饰面颜色": {"推门侧": "", "拉门侧": ""}
    }

    # 定义系统提示模板
    system_template: str = f"""
    你是信息抽取引擎。任务：从输入文本中的表格中抽取下列字段，并严格以“唯一一个 JSON 对象”输出。JSON 的键名、嵌套结构必须与下面示例完全一致，不得新增或缺少任何键：

    [唯一允许的 JSON 结构示例]
    {json.dumps(json_schema, ensure_ascii=False, indent=2)}

    [字段含义说明]
    - "门编号"：表格中“门编号”一行对应的值。
    - "门型"：表格中“门型”一行对应的值。
    - "洞口尺寸"：表格中“洞口尺寸”一行对应的值，形如 "1490*2300"。
    - "构件尺寸"：表格中“构件尺寸”一行对应的值，形如 "1460*2300"。
    - "门框材质"：表格中“门框材质”一行的完整文本（为空则返回空字符串""）。
    - "门扇材质"：表格中“门扇材质”一行的完整文本（为空则返回空字符串""）。
    - "门槛材质"：表格中“门槛材质”一行的完整文本（为空则返回空字符串""）。
    - "防火门芯"：表格中“防火门芯”一行的完整文本（为空则返回空字符串""）。
    - "玻璃"：表格中“玻璃”一行的完整文本（为空则返回空字符串""）。
    - "门框密封条"：表格中“门框密封条”一行的完整文本（为空则返回空字符串""）。
    - "门扇密封条"：表格中“门扇密封条”一行的完整文本（为空则返回空字符串""）。
    - "五金配置组名称"：来自表头中“五金配置(XXX)”里的 XXX，只保留括号内字符串，例如“五金配置(HW-8)”→ "HW-8"、“五金配置(HW-08a)”→ "HW-08a"；如果完全没有出现，则返回空字符串""。
    - "五金配置"：对应“五金配置”表格中每一条五金明细。每一行生成一个对象，包含“名称/品牌/型号/数量”四个字段。
    - "饰面颜色"：从“饰面颜色”一行中，根据“拉门侧:…”和“推门侧:…”的描述，分别填入 "拉门侧" 和 "推门侧"。

    [抽取与清洗规则（必须执行）]
    1) 仅输出上方结构的 JSON；禁止任何解释、注释、额外键、NaN、undefined、null。若无法确定某个字段值，请使用空字符串""或空数组[]。
    2) “五金配置”表格中，“名称”若带有序号或装饰（例如：①、②、③、(1)、（1）、1.），在写入 "名称" 字段时必须去除这些序号，只保留名称文本；空行或全空项必须跳过，不出现在数组中。
    3) 在写入所有字符串字段之前，必须先反转 HTML 实体（例如：&quot; → "，&amp; → &），然后保留原有的文字内容和大小写，不进行翻译或改写。
    4) "数量" 必须是整数类型（JSON 数字，而不是字符串）；如果无法确定数量，则跳过该五金项，不要为该项生成对象。
    5) "五金配置组名称" 仅填入括号中的代码字符串，例如 "五金配置(HW-08a)" → "HW-08a"。不要包含括号或中文说明。如果未在文本中出现“五金配置(…)”结构，则将 "五金配置组名称" 设为空字符串""。
    6) "洞口尺寸" 与 "构件尺寸" 字段保留为原始的“宽*高”字符串（例如 "1490*2300"），不要增加单位、中文说明，也不要拆分为对象。
    7) "门扇材质"、"门槛材质"、"防火门芯"、"玻璃"、"门框密封条"、"门扇密封条" 等字段，直接填入对应表格行的完整文本（在反转 HTML 实体和去除首尾空白之后），不做额外解释或格式修改。
    8) "饰面颜色" 字段：从同一单元格文本中识别“拉门侧: …”和“推门侧: …”，分别填入 "拉门侧" 和 "推门侧"。如果只出现其中一侧，则另一侧使用空字符串""。
    9) 仅返回一个合法的 JSON 对象（UTF-8 编码，键和值使用双引号包裹，无尾随逗号）；禁止使用任何 Markdown 代码块标记（例如 ```json 或 ```），禁止在 JSON 前后添加多余文本。

    严格遵循以上规则，仅输出 JSON。
    """.strip()

    user_prompt: str = f"请按上面的规则，从以下markdown表格中抽取并“仅以一个JSON对象”返回：\n{markdown_text}"

    payload: dict = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_template},
            {"role": "user", "content": user_prompt},
        ],
        # "temperature": 0,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GLM_API_KEY}"
    }

    response = post_with_retry(CHAT_COMPLETIONS_URL, data=json.dumps(payload, ensure_ascii=False), headers=headers,
                               proxies=None)
    response.raise_for_status()

    content: str = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)  # 解析并输出JSON对象


def pdf_process_main(pdf_path: str,
                     roi_shape: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.0), (1.0, 1.0)),
                     dpi: int = 300, temp_png_path: Optional[str] = None) -> np.ndarray:
    """
    load pdf and export png

    :param pdf_path: pdf file path
    :param roi_shape: roi shape in normalized coordinates ((x0, y0), (x1, y1))
    :param dpi: export png dpi
    :param temp_png_path: temporary png file path
    :return: cropped png array
    """

    # load pdf and get pdf shape
    pdf = pymupdf.open(pdf_path)
    page = pdf.load_page(0)

    # calculate rotation
    rotate_deg = 0
    width, height = page.rect.width, page.rect.height
    if width < height:
        rotate_deg = -90

    # zoom and calculate pixmap
    page.set_rotation(rotation=rotate_deg)  # reset rotation

    # use roi calculate crop rect
    width, height = page.rect.width, page.rect.height
    x0, y0 = roi_shape[0]
    x1, y1 = roi_shape[1]
    print(
        f"PDF page size: width={width}, height={height}, crop rect=({x0 * width}, {y0 * height}, {x1 * width}, {y1 * height})")

    # export to matrix
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(dpi / 72, dpi / 72), alpha=False,
        clip=pymupdf.Rect(x0 * width, y0 * height, x1 * width, y1 * height))

    if temp_png_path:
        pix.save(temp_png_path)

    pdf.close()

    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)


def multi_thread_main(cache_data_key: str):
    global CACHE_DATA_DICT
    try:
        with LOCK_1:
            cache_data = CACHE_DATA_DICT[cache_data_key]
        print(f"[THREAD] Start processing: {cache_data_key}", flush=True)

        markdown_text: str = paddle_ocr_vl(detection=cache_data.array)
        print(f"[THREAD] OCR done for {cache_data_key}:\n{markdown_text}", flush=True)

        dumped_json: dict = big_model_completions(markdown_text=markdown_text)
        print(f"[THREAD] JSON done for {cache_data_key}:\n{dumped_json}", flush=True)

        with LOCK_1:
            CACHE_DATA_DICT[cache_data_key].markdown_text = markdown_text
            CACHE_DATA_DICT[cache_data_key].extracted_json = dumped_json

    except Exception as e:
        # 把异常打出来，结合 main 里的 fut.result() 更好定位
        print(f"[THREAD-ERROR] key={cache_data_key}, error={repr(e)}", flush=True)
        raise


if __name__ == '__main__':
    pdf_folder: str = r"your/pdf/path"

    import time

    st_time = time.time()

    # preprocess pdf to cache data (pymupdf not support multi-thread)
    for file_path in os.listdir(pdf_folder):
        if file_path.endswith(".pdf") and "FHM" in file_path:
            full_pdf_path = os.path.join(pdf_folder, file_path)
            print(f"Caching PDF: {full_pdf_path}")
            corp_array: np.ndarray = pdf_process_main(
                pdf_path=full_pdf_path, roi_shape=((0.6, 0.55), (0.85, 1.0)),
                temp_png_path=f"./test_img/{os.path.splitext(os.path.basename(full_pdf_path))[0]}.png")
            CACHE_DATA_DICT[file_path] = cacheData(pdf_path=full_pdf_path, array=corp_array)

    # 构造线程池
    # pool = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) * 5))
    pool = ThreadPoolExecutor(max_workers=2)
    all_tasks = []

    for cache_key in CACHE_DATA_DICT.keys():
        all_tasks.append(pool.submit(multi_thread_main, cache_key))

    # 等待所有任务完成
    wait(all_tasks, return_when=ALL_COMPLETED)
    pool.shutdown()

    print(f"Total processing time: {time.time() - st_time} seconds")
