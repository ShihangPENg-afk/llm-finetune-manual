from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


def read_pdf_paragraphs(pdf_path: Path) -> list[str]:
    doc = fitz.open(pdf_path)
    paragraphs = []

    for page in doc:
        text = page.get_text("text")
        if not text:
            continue

        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]

        current = []
        for line in lines:
            # 简单按空行/短行分段；这里先保守一点
            current.append(line)
            if line.endswith(("。", "！", "？", ".", "!", "?")):
                para = " ".join(current).strip()
                if para:
                    paragraphs.append(para)
                current = []

        if current:
            para = " ".join(current).strip()
            if para:
                paragraphs.append(para)

    return paragraphs


def build_chunks_from_paragraphs(
    paragraphs: list[str],
    source_file: str,
    chunk_size: int = 800,
) -> list[dict]:
    chunks = []
    current_paras: list[str] = []
    current_len = 0
    chunk_id = 0

    for para in paragraphs:
        para_len = len(para)

        if current_paras and current_len + para_len > chunk_size:
            text = "\n".join(current_paras).strip()
            if text:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source_file": source_file,
                        "text": text,
                    }
                )
                chunk_id += 1

            # overlap：保留上一个 chunk 的最后一个段落
            current_paras = current_paras[-1:]
            current_len = sum(len(p) for p in current_paras)

        current_paras.append(para)
        current_len += para_len

    if current_paras:
        text = "\n".join(current_paras).strip()
        if text:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": source_file,
                    "text": text,
                }
            )

    return chunks


def collect_pdf_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"输入文件不是 PDF: {input_path}")
        return [input_path]

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"目录中没有 PDF 文件: {input_path}")
        return pdf_files

    raise FileNotFoundError(f"找不到输入路径: {input_path}")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("用法: python scripts/extract_chunks.py data/raw_pdfs/")

    input_path = Path(sys.argv[1]).resolve()
    pdf_files = collect_pdf_files(input_path)

    out_path = Path("data/processed/chunks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    total_paragraphs = 0
    total_chunk_count = 0
    global_chunk_id = 0

    for pdf_path in pdf_files:
        paragraphs = read_pdf_paragraphs(pdf_path)
        total_paragraphs += len(paragraphs)

        chunks = build_chunks_from_paragraphs(
            paragraphs=paragraphs,
            source_file=pdf_path.name,
            chunk_size=800,
        )

        # 重置全局 chunk_id
        for chunk in chunks:
            chunk["chunk_id"] = global_chunk_id
            global_chunk_id += 1

        all_chunks.extend(chunks)
        total_chunk_count += len(chunks)

        print(f"✅ 已处理: {pdf_path.name} -> {len(chunks)} chunks")

    out_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n==============================")
    print(f"✅ 共处理 PDF 数量: {len(pdf_files)}")
    print(f"✅ 总段落数: {total_paragraphs}")
    print(f"✅ 总 chunk 数: {total_chunk_count}")
    print(f"✅ 输出文件: {out_path}")


if __name__ == "__main__":
    main()