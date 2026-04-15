from pathlib import Path

from document_creator import fill_upgrade_request_docx

TARGET_PATH = Path(r"防火门深化图")

if __name__ == '__main__':
    write_path = Path("./door_name")
    write_path.mkdir(parents=True, exist_ok=True)
    for _dir in [p for p in TARGET_PATH.iterdir() if p.is_dir()]:
        fill_upgrade_request_docx(
            Path(r"0版升版申请单.docx"),
            write_path / f"{_dir.name}.docx",
            drawing_names=[pdf.stem for pdf in (_dir / "PDF").glob("*.pdf")],
            drawing_name_title=_dir.name,
        )
