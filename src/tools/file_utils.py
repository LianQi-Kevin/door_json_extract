import base64
import mimetypes
from typing import Union
from pathlib import Path


def file_to_base64(file: Union[Path, str]) -> str:
    file: Path = Path(file).resolve()
    mime = mimetypes.guess_type(str(file.name))[0]
    if mime is None:
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".pdf": "application/pdf",
        }.get(file.suffix.lower(), "application/octet-stream")

    return f"data:{mime};base64,{base64.b64encode(file.read_bytes()).decode(encoding='utf-8')}"
