""" Step.2 - 基于 ocr json 请求 GLM-5.1, 获取 <hardware info>.json , 保存到 <config.cache_json_hardware_dir> / <png filename>.png """
import logging
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from config import config
from models.GLM.chat_completions.models_request import RequestMessageBase, Thinking, ResponseFormat
from models.GLM.layout_parsing import LayoutParsing
from models.GLM.layout_parsing.models import LayoutDetail
from src.models.GLM.chat_completions import sync_chat_completions, ChatCompletionsRequest
from src.tools.logging_utils import log_set


JSON_SCHEMA_FOR_FORMAT = {
    "结算门型": "",
    "门型": "",
    "门编号": "",
    "洞口尺寸": "",
    "构件尺寸": "",
    "门框材质": "",
    "门扇材质": "",
    "门槛材质": "",
    "门芯": "",
    "玻璃": "",
    "门框密封条": "",
    "门扇密封条": "",
    "五金配置": [],
    "饰面颜色": {
        "拉门侧": "",
        "推门侧": ""
    }
}

SYSTEM_TEMPLATE = f"""
你是信息抽取引擎。任务：从输入的表格文本中抽取信息，并严格输出“唯一一个 JSON 对象”。

输入可能是：
- HTML table
- markdown 表格
- 其他保留表格结构的文本

你必须严格输出且只输出一个 JSON 对象。

[输出结构模板]
JSON 的键名、层级结构、嵌套对象形式必须与下面模板完全一致，不得新增、删除、改名或改变层级：

{json.dumps(JSON_SCHEMA_FOR_FORMAT, ensure_ascii=False, indent=2)}

[字段类型与空值约束]
1. 所有顶层键必须始终输出。
2. 除 "五金配置" 外，其余顶层普通字段均为字符串类型；缺失、为空或无法确定时，返回空字符串 ""。
3. "五金配置" 必须是数组类型：
   - 若没有任何有效明细，返回 []
   - 不得返回 null
   - 不得用一条空对象代替空数组
4. "五金配置" 数组中的每个元素必须为对象，且仅包含以下键：
   - "名称"：字符串
   - "品牌"：字符串
   - "型号"：字符串
   - "数量"：整数
5. "饰面颜色" 必须始终输出为对象，且仅包含以下键：
   - "拉门侧"：字符串
   - "推门侧"：字符串
   若某一侧缺失，则该侧返回 ""；不得删除该键。
6. "数量" 必须输出为 JSON 数字整数，不得输出为字符串、浮点数、null 或其他类型。

[字段映射规则]
- "结算门型"：取表格中“结算门型”对应的值。
- "门型"：取表格中“门型”对应的值。
- "门编号"：取表格中“门编号”对应的值。
- "洞口尺寸"：取表格中“洞口尺寸”对应的值。
- "构件尺寸"：取表格中“构件尺寸”对应的值。
- "门框材质"：取表格中“门框材质”对应的值。
- "门扇材质"：取表格中“门扇材质”对应的值。
- "门槛材质"：取表格中“门槛材质”对应的值。
- "门芯"：取表格中“门芯”或“防火门芯”对应的值，统一写入键 "门芯"。
- "玻璃"：取表格中“玻璃”对应的值。
- "门框密封条"：取表格中“门框密封条”对应的值。
- "门扇密封条"：取表格中“门扇密封条”对应的值。
- "五金配置"：取“五金配置”子表中的每条有效明细行，每行生成一个对象，字段为“名称 / 品牌 / 型号 / 数量”。
- "饰面颜色"：从“饰面颜色”对应单元格中提取“拉门侧”和“推门侧”的值，分别写入：
  - "饰面颜色"."拉门侧"
  - "饰面颜色"."推门侧"

[值清洗规则]
1. 对所有字符串值，先做 HTML 实体反转，例如：
   - &quot; → "
   - &amp; → &
   - &lt; → <
   - &gt; → >
2. 去除字段值首尾空白。
3. 保留原始文字内容，不翻译、不改写、不补单位、不重组语义。
4. 对于 HTML 中的 <br> 或换行，按普通分隔内容理解后提取，不把 HTML 标签原样写入结果。
5. "洞口尺寸"、"构件尺寸" 保留原字符串，例如 "1500*2300"；不要拆分，不要补单位。

[五金名称清洗规则]
1. 仅允许删除“名称”字段前缀中的编号、序号或装饰符号，例如：
   - ① ② ③
   - (1) (2)
   - （1）（2）
   - 1. 2.
   - $\\textcircled{{1}}$ 之类的编号装饰
2. 删除编号后，必须保留名称本体的完整文本。
3. 不得删除名称本体中的功能词、结构词、限定词、方向词或习惯叫法，例如以下词通常属于名称本体，应保留：
   - 国标
   - 单门
   - 双门
   - 明装
   - 暗装
   - 手动
   - 重力型
4. 不得将名称做概括性改写或标准化缩写。例如：
   - “国标明装闭门器” 不得改写为 “闭门器”
   - “单门磁力锁” 不得改写为 “磁力锁”
   - “重力型顺位器” 不得改写为 “顺位器”
5. 名称字段只做“去前缀编号装饰”的清洗，不做其他语义压缩。

[五金配置抽取规则]
1. 仅抽取“五金配置”子表中的有效明细行。
2. 每条有效明细生成一个对象，包含：
   - "名称"
   - "品牌"
   - "型号"
   - "数量"
3. "数量" 必须能明确识别为整数；若无法确定数量，则跳过整条，不输出该对象。
4. 空行、全空行、明显不是明细的数据行，不要写入数组。
5. 若某条记录只有“名称/品牌/型号”但缺少可确定的数量，则该条不输出。

[缺失值策略]
1. 普通文本字段缺失、为空或无法确定时，返回空字符串 ""。
2. "五金配置" 如果没有任何有效明细，返回空数组 []。
3. "饰面颜色" 必须始终输出为对象：
   - 若只识别到“拉门侧”，则“推门侧”填 ""
   - 若只识别到“推门侧”，则“拉门侧”填 ""
   - 若都未识别到，则两个都填 ""

[冲突处理规则]
1. 同一字段若存在多个候选值，优先采用与字段名完全匹配的那一行。
2. "门芯" 的来源标签允许是“门芯”或“防火门芯”，但输出键只能是 "门芯"。
3. 无法可靠判断时，返回空值，不要猜测。

[输出约束]
1. 只能输出一个合法 JSON 对象。
2. 不得输出 Markdown 代码块标记，如 ```json。
3. 不得输出任何解释、说明、前后缀文本。
4. 不得输出 null、None、NaN、undefined。
5. 所有键名必须与输出结构模板完全一致。
6. 标题类文本（例如“钢质双扇隔热防火门(乙级)”）如果不对应 schema 中的字段，不要输出为额外字段。

严格遵循以上规则，仅输出 JSON。
""".strip()


def threading_main(md_result: str, json_path: Path):
    request_body = ChatCompletionsRequest(
        model="glm-5.1",
        messages=[
            RequestMessageBase(role="system", content=SYSTEM_TEMPLATE),
            RequestMessageBase(role="user", content=f"请按上面的规则，从以下表格文本中抽取，并仅返回一个 JSON 对象：\n{md_result}")
        ],
        thinking=Thinking("disabled"),
        response_format=ResponseFormat("json_object"),
        do_sample=False,
    )
    response = sync_chat_completions(config.glm_api_key, request_body)
    export_path = config.cache_json_glm_response_dir / json_path.parent.name / f"{json_path.stem}.json"
    logging.debug(f"finish request {json_path.parent.name}/{json_path.name}")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with open(export_path, "w", encoding="utf-8") as json_f:
        json.dump(asdict(response), json_f, ensure_ascii=False, indent=4)


def OCR_response_json_loader(json_path: Path) -> LayoutParsing:
    with open(json_path, "r", encoding="utf-8") as json_f:
        json_dict = json.loads("".join(json_f.readlines()))
    result = LayoutParsing(**json_dict)
    result.layout_details = [[LayoutDetail(**item) for item in page] for page in json_dict["layout_details"]]
    return result


if __name__ == '__main__':
    log_set(logging.DEBUG, log_save=True, save_level=logging.WARNING, save_path=config.log_dir / "2_Conv_extract.log")

    # multi request
    pool = ThreadPoolExecutor(max_workers=1)    # GLM 5.1 only support 1 Concurrency
    all_tasks = []

    for _json_path in Path(config.cache_json_ocr_dir).rglob("*.json"):
        ocr_result = OCR_response_json_loader(_json_path)
        pool.submit(threading_main, cast(str, ocr_result.md_results), _json_path)

    wait(all_tasks, return_when=ALL_COMPLETED)
    pool.shutdown()
