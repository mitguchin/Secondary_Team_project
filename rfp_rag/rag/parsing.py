"""① 문서 파싱 — PDF(PyPDFLoader · PyMuPDFLoader) · HWP/HWPX(langchain-hwp-hwpx-loader) → 페이지 단위 레코드.
담당: 문서 파싱.  (예전 engine.py 의 파싱 부분)"""
from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter

from .text_utils import apply_hanja, normalize_text


# ====================================================================================================
# 파싱: 문서 1개 → 레코드 — PyPDFLoader / PyMuPDFLoader / HwpHwpxLoader 라이브러리 사용   (원래 parsing.py)
# ====================================================================================================
# 문서 1개 → 페이지 단위 레코드(text/table/image_caption).  파일 단위로 캐시(=체크포인트).
def make_doc_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    slug = re.sub(r"[^\w가-힣]+", "_", stem).strip("_")[:40]
    return f"{slug}_{hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:6]}"


def _title(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[_]+", " ", stem).strip()


def _remove_repeated_lines(pages: dict[int, str], ratio: float = 0.5) -> tuple[dict[int, str], int]:
    """페이지의 50% 이상(최소 3페이지)에 반복되는 짧은 줄 = 머리글/바닥글 → 제거"""
    if len(pages) < 3:
        return pages, 0
    key = lambda ln: re.sub(r"\d+", "#", re.sub(r"\s+", "", ln))  # noqa: E731  페이지 번호 무시
    cnt = Counter()
    for txt in pages.values():
        cnt.update({key(ln) for ln in txt.split("\n") if 0 < len(ln.strip()) < 120})
    thr = max(3, math.ceil(ratio * len(pages)))
    bad = {k for k, c in cnt.items() if c >= thr and k}
    removed = 0
    out = {}
    for p, txt in pages.items():
        kept = []
        for ln in txt.split("\n"):
            if ln.strip() and key(ln) in bad:
                removed += 1
                continue
            kept.append(ln)
        out[p] = "\n".join(kept)
    return out, removed


_PAGE_NO = re.compile(r"^\s*(?:[-–−]\s*\d{1,3}\s*[-–−]|\d{1,3}\s*/\s*\d{1,3}|페\s*이\s*지\s*:?\s*\d+\s*/\s*\d+"
                      r"|page\s*\d+(?:\s*of\s*\d+)?|\d{1,3})\s*$", re.IGNORECASE)
_END_OK = re.compile(r"(?:[.?!:;。)\]」』>’”\"'%]|[다요음함임됨것니까죠])\s*$")
_START_MARK = re.compile(r"^\s*(?:[○◦❍●•·▪■□◆◇※\-–▶▷►❏✓√]|ㅇ\s|\(?\d{1,2}[\).]\s*\S|\d+\.\d+|[가-하][\).]\s+\S"
                         r"|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|[①-⑳]|[A-Z]{2,5}-[\w가-힣]*\d|\||<|\[|【)")


def _remove_page_numbers(text: str) -> tuple[str, int]:
    kept, n = [], 0
    for ln in text.split("\n"):
        if ln.strip() and _PAGE_NO.match(ln):
            n += 1
            continue
        kept.append(ln)
    return "\n".join(kept), n


def _join_broken_lines(text: str, min_len: int = 20) -> tuple[str, int]:
    """PDF 줄바꿈으로 끊긴 문장 복원: 앞 줄이 문장 종결이 아니고 다음 줄이 글머리(○, -, 1), 가. …)가 아니면 이어붙임"""
    lines = text.split("\n")
    out: list[str] = []
    n = 0
    for ln in lines:
        if (out and out[-1].strip() and ln.strip() and len(out[-1].strip()) >= min_len
                and not _END_OK.search(out[-1]) and not _START_MARK.match(ln)):
            out[-1] = out[-1].rstrip() + " " + ln.strip()
            n += 1
        else:
            out.append(ln)
    return "\n".join(out), n


def _drop_table_lines(text: str, tables: list[dict], threshold: float = 0.85, min_tokens: int = 3) -> tuple[str, int]:
    """PyPDFLoader 본문에 평문으로 섞여 들어간 표 내용을 제거 (표는 Markdown 청크로 따로 있음).
    PDF 는 여러 줄 셀이 줄 단위로 섞여 나오므로(셀1 1행 + 셀2 1행 …) 부분문자열이 아니라
    '줄의 토큰 중 표 셀 토큰 비율'로 판단한다."""
    if not tables:
        return text, 0
    vocab = {w for t in tables for c in t.get("cells", []) for w in c.split() if len(w) >= 2}
    if not vocab:
        return text, 0
    kept, n = [], 0
    for ln in text.split("\n"):
        toks = [w for w in ln.split() if len(w) >= 2]
        if len(toks) >= min_tokens and sum(w in vocab for w in toks) / len(toks) >= threshold:
            n += 1
            continue
        kept.append(ln)
    return "\n".join(kept), n


_BULLET_GLYPHS = re.compile(r"[\u0178\uf000\uf0a7\uf0b7\uf0d8\uf06c\uf09f\uf0fc\uf076]")  # PDF 폰트 깨진 글머리(Ÿ 등)


def _fix_bullets(text: str) -> tuple[str, int]:
    n = len(_BULLET_GLYPHS.findall(text))
    return _BULLET_GLYPHS.sub("•", text), n


def _garbled(text: str) -> int:
    return sum(1 for ch in text if ch == "\ufffd" or "\ue000" <= ch <= "\uf8ff" or (ord(ch) < 32 and ch not in "\n\t"))


_IMG_MD = re.compile(r"!\[(.*?)\]\(#\)", re.DOTALL)
_IMG_MARK = re.compile(r"\[IMAGE(?::\s*([^\]]+))?\]")


def _md_table_blocks(text: str) -> tuple[list[str], list[str]]:
    """페이지 텍스트 → (Markdown 표 블록들, 표가 아닌 줄들)"""
    tables, others, cur = [], [], []
    for ln in text.split("\n"):
        if ln.lstrip().startswith("|"):
            cur.append(ln.strip())
            continue
        if cur:
            tables.append("\n".join(cur))
            cur = []
        others.append(ln)
    if cur:
        tables.append("\n".join(cur))
    return [t for t in tables if t.count("\n") >= 1], others


def _table_cells(md: str) -> dict:
    rows = [r for r in md.split("\n") if not re.fullmatch(r"[\s|:\-]+", r)]
    cells = [c.strip() for r in rows for c in r.strip().strip("|").split("|") if c.strip()]
    return {"cells": [c.replace("<br>", " ") for c in cells], "rows": [r.replace("<br>", " ") for r in rows]}


def _pdf_pages(path: str, pcfg: dict, image_parser) -> tuple[int, dict[int, str], dict[int, list[str]], dict[int, list[str]]]:
    """LangChain PDF 로더만 사용.
    본문 : PyPDFLoader (pdf_text_loader="PyPDFLoader") 또는 PyMuPDFLoader
    표   : PyMuPDFLoader(extract_tables="markdown")      (pdf_table_loader="PyMuPDFLoader", None 이면 표 없음)
    이미지: PyMuPDFLoader(extract_images=True, images_parser=…, images_inner_format="markdown-img")
    return: total, {page: 본문}, {page: [표 md]}, {page: [이미지 설명]}"""
    from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

    text_loader = pcfg.get("pdf_text_loader", "PyPDFLoader")
    table_loader = pcfg.get("pdf_table_loader", "PyMuPDFLoader") if pcfg.get("extract_tables", True) else None
    texts, tables, images = {}, {}, {}
    total = 0
    if text_loader == "PyPDFLoader":
        pp = pcfg.get("pypdf") or {}
        docs = PyPDFLoader(path, mode=pp.get("mode", "page"), extraction_mode=pp.get("extraction_mode", "plain"),
                           extraction_kwargs=pp.get("extraction_kwargs") or None).load()
        total = len(docs)
        for d in docs:
            texts[int(d.metadata.get("page", 0)) + 1] = d.page_content
    if text_loader == "PyMuPDFLoader" or table_loader == "PyMuPDFLoader" or image_parser is not None:
        if image_parser is not None:
            from .vision import patch_pymupdf_image_extraction

            patch_pymupdf_image_extraction()
        mu = PyMuPDFLoader(path, mode="page", extract_tables="markdown" if table_loader == "PyMuPDFLoader" else None,
                           extract_tables_settings=pcfg.get("table_settings") or None,
                           extract_images=image_parser is not None, images_parser=image_parser,
                           images_inner_format="markdown-img").load()
        total = max(total, len(mu))
        for d in mu:
            p = int(d.metadata.get("page", 0)) + 1
            content = d.page_content
            caps = [c.strip() for c in _IMG_MD.findall(content) if c.strip()]
            if caps:
                images[p] = caps
            content = _IMG_MD.sub("", content)
            tbl, other = _md_table_blocks(content)
            if table_loader == "PyMuPDFLoader":
                tables[p] = tbl
            if text_loader == "PyMuPDFLoader":
                texts[p] = "\n".join(other)
    return total, texts, tables, images


def _strip_image_list(text: str) -> str:
    """HwpHwpxLoader(include_images=True) 가 본문 끝에 붙이는 '## Images' 파일 목록 제거"""
    if text.startswith("## Images\n"):
        return ""
    return text.rsplit("\n\n## Images\n", 1)[0] if "\n\n## Images\n" in text else text


def patch_hwp_parser_hang():
    """hwp-hwpx-parser 1.0.0 무한 루프 버그 수정 (HWP5Reader._extract_hyperlink_texts_from_para).
    문단 데이터 끝 6바이트 안에 제어문자 0x03 이 있으면 인덱스가 증가하지 않아 extract_text 가 영원히 멈춤
    (예: 길이 18바이트 문단의 14번째 바이트). 같은 로직에 '남은 길이가 부족하면 2바이트 전진'만 추가."""
    import struct

    from hwp_hwpx_parser import hwp5

    cls = getattr(hwp5, "HWP5Reader", None)
    if cls is None or getattr(cls, "_rag_hang_patched", False):
        return
    HL = hwp5.CTRL_ID_HYPERLINK

    def _extract_hyperlink_texts_from_para(self, para_data: bytes):
        out, i, n = [], 0, len(para_data)
        while i < n - 1:
            code = struct.unpack_from("<H", para_data, i)[0]
            if code == 0x03:
                if i + 6 > n:          # ← 원본은 여기서 i 를 증가시키지 않아 무한 루프
                    i += 2
                    continue
                if struct.unpack_from("<I", para_data, i + 2)[0] == HL:
                    j, chars = i + 14, []
                    while j < n - 1:
                        c = struct.unpack_from("<H", para_data, j)[0]
                        if c == 0x04:
                            break
                        if 0x20 <= c:
                            chars.append(chr(c))
                        j += 2
                    if chars:
                        out.append("".join(chars))
                    i = max(j, i + 2)
                else:
                    i += 14
            elif code == 0x04:
                i += 10
            elif code < 32:
                i += 10 if code in (11, 12) else (14 if 15 <= code <= 23 else 2)
            else:
                i += 2
        return out

    cls._extract_hyperlink_texts_from_para = _extract_hyperlink_texts_from_para
    cls._rag_hang_patched = True


def hwp_loader(path: str, pcfg: dict, images_dir: str | None = None):
    """langchain-hwp-hwpx-loader 의 HwpHwpxLoader 생성 (노트북 1-1 셀에서도 같은 설정으로 직접 호출).
    표는 본문 안 제자리에 Markdown 으로 들어오므로 include_tables=False (True 면 끝에 '## Tables' 로 한 번 더 붙음)."""
    from hwp_hwpx_parser import ExtractOptions, ImageMarkerStyle, TableStyle
    from langchain_hwp_hwpx import HwpHwpxLoader

    patch_hwp_parser_hang()
    hc = pcfg.get("hwp") or {}
    return HwpHwpxLoader(
        path, mode="single",
        extract_options=ExtractOptions(table_style=getattr(TableStyle, str(hc.get("table_style", "MARKDOWN")).upper()),
                                       image_marker=ImageMarkerStyle.WITH_NAME),
        include_tables=False, include_notes=bool(hc.get("include_notes", True)),
        include_memos=bool(hc.get("include_memos", False)), include_hyperlinks=bool(hc.get("include_hyperlinks", False)),
        include_images=images_dir is not None, images_dir=images_dir,
        image_document_mode="save_and_reference" if images_dir else "metadata_only",
        on_encrypted="placeholder", on_invalid="placeholder", on_error="raise", include_extracted_at=False)


def _hwp_text(path: str, pcfg: dict, image_parser, st: Counter, log, images_dir: str | None = None) -> tuple[str, list[str]]:
    """langchain-hwp-hwpx-loader(HwpHwpxLoader)로 HWP/HWPX 읽기 → 정리된 텍스트(표는 Markdown), 이미지 설명은 [[IMG:n]] 자리표시"""
    from .text_utils import clean_tab_garbage, repair_md_tables
    from .vision import pil_from_bytes

    want_img = image_parser is not None
    if want_img and images_dir is None:
        import tempfile

        images_dir = tempfile.mkdtemp(prefix="hwp_img_")
    docs = hwp_loader(path, pcfg, images_dir if want_img else None).load()
    if not docs:
        return "", []
    d = docs[0]
    if d.metadata.get("status") in ("encrypted", "invalid"):
        log(f"[parse] ⚠️ {d.metadata['status']} 문서(암호화/배포용·손상): {os.path.basename(path)} — 본문을 읽을 수 없습니다")
        return "", []
    text = _strip_image_list(d.page_content)
    imgs = {}
    if want_img and images_dir and os.path.isdir(images_dir):
        imgs = {n: os.path.join(images_dir, n) for n in os.listdir(images_dir)}
    st["raw_chars"] += len(text)
    text = clean_tab_garbage(text)
    text, st["merged_cell_rows_fixed"] = repair_md_tables(text)
    captions: list[str] = []
    counts = Counter(m.group(1) for m in _IMG_MARK.finditer(text))
    st["images_found"] = len(counts)
    done: dict[str, str] = {}

    def sub(m):
        name = (m.group(1) or "").strip()
        if image_parser is None or name not in imgs or counts[name] >= int(pcfg.get("max_repeat_image_pages", 3)):
            return ""  # 로고·장식(반복 이미지)이나 캡션 비활성 → 표시 제거
        if name not in done:
            with open(imgs[name], "rb") as f:
                pil = pil_from_bytes(f.read(), name.rsplit(".", 1)[-1])
            try:
                done[name] = image_parser._analyze_image(pil) if pil is not None else ""
            except Exception as e:  # noqa: BLE001
                log(f"[parse] 이미지 처리 실패 {name}: {e}")
                done[name] = ""
        if not done[name]:
            return ""
        captions.append(done[name])
        return f"\n[[IMG:{len(captions) - 1}]]\n"

    return _IMG_MARK.sub(sub, text), captions


def parse_file(path: str, pcfg: dict, image_parser=None, log=print, images_dir: str | None = None) -> dict:
    """문서 1개 → 레코드(text/table/image_caption).  파일 단위로 캐시(=체크포인트).
    PDF: PyPDFLoader + PyMuPDFLoader / HWP·HWPX: langchain-hwp-hwpx-loader(HwpHwpxLoader) — 직접 만든 로더 없음."""
    skip = int(pcfg.get("skip_first_pages", 0))
    hanja_mode = pcfg.get("hanja_mode", "append")
    ext = os.path.splitext(path)[1].lower()
    doc = {"doc_id": make_doc_id(path), "source": os.path.basename(path), "title": _title(path),
           "doc_type": ext.lstrip("."), "records": []}
    st = Counter()
    records: list[dict] = []

    if ext == ".pdf":
        total, texts, tables, images = _pdf_pages(path, pcfg, image_parser)
        st["raw_chars"] = sum(len(t) for t in texts.values())
        cover = [texts.get(p, "") for p in range(1, skip + 1)]
        st["skipped_chars"] = sum(len(c) for c in cover)
        doc.update(total_pages=total, skipped_pages=min(skip, total), kept_pages=max(0, total - skip),
                   cover_text=normalize_text("\n".join(cover))[:2500], front_matter=f"앞 {skip}페이지")
        kept = {p: normalize_text(t) for p, t in texts.items() if p > skip}
        if pcfg.get("remove_repeated_lines", True):
            kept, st["repeated_lines_removed"] = _remove_repeated_lines(kept, float(pcfg.get("repeated_line_ratio", 0.5)))
        rep_caps = Counter(c for p, cs in images.items() for c in set(cs))  # 여러 쪽에 반복되는 그림 = 로고
        for p in range(skip + 1, total + 1):
            t = kept.get(p, "")
            tbls = [b for b in tables.get(p, []) if b.count("|") >= 4]
            if pcfg.get("fix_bullet_glyphs", True):
                t, n = _fix_bullets(t)
                st["bullet_glyphs_fixed"] += n
            if pcfg.get("remove_page_numbers", True):
                t, n = _remove_page_numbers(t)
                st["page_number_lines_removed"] += n
            if tbls and pcfg.get("drop_table_text_from_body", True):
                t, n = _drop_table_lines(t, [_table_cells(b) for b in tbls], float(pcfg.get("table_dedupe_threshold", 0.85)))
                st["table_lines_dropped_from_body"] += n
            if pcfg.get("join_broken_lines", True):
                t, n = _join_broken_lines(t, int(pcfg.get("join_min_line_len", 20)))
                st["broken_lines_joined"] += n
            if t.strip():
                records.append({"page": p, "kind": "text", "text": t})
            records += [{"page": p, "kind": "table",
                         "text": _fix_bullets(b.replace("<br>", " "))[0] if pcfg.get("fix_bullet_glyphs", True)
                         else b.replace("<br>", " ")} for b in tbls]
            for c in images.get(p, []):
                if rep_caps[c] < int(pcfg.get("max_repeat_image_pages", 3)):
                    records.append({"page": p, "kind": "image_caption", "text": f"[그림/도표 설명] {c}"})
            if len(t.strip()) < 30 and not tbls:
                st["empty_pages"] += 1
        st["images_found"] = sum(len(v) for v in images.values())
    elif ext in (".hwp", ".hwpx"):
        from .text_utils import split_front_matter

        text, captions = _hwp_text(path, pcfg, image_parser, st, log, images_dir)
        front, body, how = ("", text, "제외 안 함")
        if skip > 0 and pcfg.get("hwp_front_matter", "auto") == "auto":
            front, body, how = split_front_matter(text)
        st["skipped_chars"] = len(front)
        doc.update(total_pages=None, skipped_pages=None, kept_pages=None,  # HWP 는 라이브러리가 쪽 정보를 주지 않음
                   cover_text=normalize_text(re.sub(r"\[\[IMG:\d+\]\]", "", front))[:2500], front_matter=how)
        para: list[str] = []
        order = 0

        def flush():
            nonlocal para, order
            if "\n".join(para).strip():
                records.append({"page": None, "order": order, "kind": "text", "text": normalize_text("\n".join(para))})
                order += 1
            para = []

        # 문서 순서 유지: 본문 → 표 → 본문 … (표 블록 위치에서 끊음)
        lines = body.split("\n")
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.lstrip().startswith("|"):
                flush()
                blk = []
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    blk.append(lines[i].strip())
                    i += 1
                md = "\n".join(blk)
                if md.count("\n") >= 1 and md.count("|") >= 4:
                    records.append({"page": None, "order": order, "kind": "table", "text": normalize_text(md)})
                    order += 1
                elif md.strip():
                    para.append(re.sub(r"\s*\|\s*", " ", md).strip())  # 1칸 표(글상자) → 본문
                continue
            m = re.fullmatch(r"\s*\[\[IMG:(\d+)\]\]\s*", ln)
            if m:
                flush()
                records.append({"page": None, "order": order, "kind": "image_caption",
                                "text": f"[그림/도표 설명] {captions[int(m.group(1))]}"})
                order += 1
            else:
                para.append(ln)
            i += 1
        flush()
    else:
        raise ValueError(f"지원하지 않는 형식: {path}")

    for r in records:
        r["text"] = apply_hanja(normalize_text(r["text"]), hanja_mode)
    doc["records"] = [r for r in records if r["text"].strip()]
    st["tables"] = sum(r["kind"] == "table" for r in doc["records"])
    st["images_captioned"] = sum(r["kind"] == "image_caption" for r in doc["records"])
    st["clean_chars"] = sum(len(r["text"]) for r in doc["records"])
    st["table_chars"] = sum(len(r["text"]) for r in doc["records"] if r["kind"] == "table")
    st["garbled_chars"] = sum(_garbled(r["text"]) for r in doc["records"])
    doc["head_text"] = "\n".join(r["text"] for r in doc["records"] if r["kind"] == "text")[:1500]  # 라우팅용
    doc["stats"] = dict(st)
    return doc
