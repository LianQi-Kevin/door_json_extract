from pathlib import Path

from src.tools.file_utils import file_to_base64
from src.models.GLM.layout_parsing import layout_parsing

if __name__ == '__main__':
    response = layout_parsing(
        api_key='YOUR_API_KEY',
        file=file_to_base64(Path(r"./example.png")),
    )
    print(response)
