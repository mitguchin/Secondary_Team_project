"""청크 분할 + QA 데이터 분할 (chunking + split 통합)."""
from __future__ import annotations

import bisect
import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .text_utils import compact


# ====================================================================================================
# 청크 분할   (원래 chunking.py)
# ====================================================================================================
# 청크 분할: 본문은 RecursiveCharacterTextSplitter(문서 단위 연결 + 페이지 추적), 표는 행 단위 분할(헤더 반복).
# 노트북에서 이름으로 고르는 구분자 → 정규식 (앞에 올수록 우선 분할, 글머리 기호는 다음 청크의 시작에 남김)
SEPARATOR_PATTERNS = {
    "문단": r"\n\s*\n",
    "Ⅰ.": r"\n(?=\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.\s])",
    "1.": r"\n(?=\s*\d{1,2}\.\s)",
    "1.1": r"\n(?=\s*\d{1,2}\.\d{1,2}\.?\s)",
    "가.": r"\n(?=\s*[가-하]\.\s)",
    "1)": r"\n(?=\s*\(?\d{1,2}\)\s)",
    "가)": r"\n(?=\s*\(?[가-하]\)\s)",
    "○": r"\n(?=\s*[○◦❍●]|\s*ㅇ\s)",
    "□": r"\n(?=\s*[□■❏◆◇▪])",
    "-": r"\n(?=\s*[-–•·]\s?)",
    "※": r"\n(?=\s*※)",
    "①": r"\n(?=\s*[①-⑳])",
    "줄바꿈": r"\n",
    "문장": r"(?<=[다요함음임됨]\.)\s+|(?<=[.?!])\s+",
    "공백": r"\s+",
}
DEFAULT_SEPARATORS = ["문단", "Ⅰ.", "1.", "가.", "1)", "○", "□", "-", "※", "줄바꿈", "문장", "공백"]

_HEADING = [  # (level, pattern, 최대 길이) — 짧은 줄만 제목으로 인정
    (1, re.compile(r"^\s*[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.\s]\s*\S"), 40),
    (1, re.compile(r"^\s*제\s*\d+\s*[장편]\s*\S"), 40),
    (2, re.compile(r"^\s*\d{1,2}\.\s+\S"), 30),
    (3, re.compile(r"^\s*\d{1,2}\.\d{1,2}\.?\s+\S"), 35),
    (4, re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\d{1,2}\.?\s+\S"), 35),
    (4, re.compile(r"^\s*[가-하]\.\s+\S"), 22),
]


def build_separators(names: list[str] | None) -> list[str]:
    names = names or DEFAULT_SEPARATORS
    out = [SEPARATOR_PATTERNS.get(n, re.escape(n)) for n in names]  # 목록에 없는 이름은 문자 그대로
    return out + [""]  # 마지막 안전장치: 글자 단위


def resolve_overlap(ccfg: dict) -> int:
    """chunk_overlap 이 지정되면 그 값, 없으면 chunk_size × overlap_ratio"""
    if ccfg.get("chunk_overlap") not in (None, "", "auto"):
        return int(ccfg["chunk_overlap"])
    return int(round(int(ccfg["chunk_size"]) * float(ccfg.get("overlap_ratio", 0.2))))


def _heading_index(full: str) -> tuple[list[int], list[str]]:
    """본문 전체에서 제목 줄 위치와 '상위 > 하위' 섹션 경로 계산"""
    offs, paths, stack, pos = [], [], {}, 0
    for ln in full.split("\n"):
        t = ln.strip()
        if t and not t.endswith(("다.", "함.", "음.", "함", "음")):
            for lvl, pat, max_len in sorted(_HEADING, key=lambda x: -x[0]):  # 1.1.1 을 1. 보다 먼저 검사
                if len(t) <= max_len and pat.match(t):
                    stack = {k: v for k, v in stack.items() if k < lvl}
                    stack[lvl] = t
                    offs.append(pos)
                    paths.append(" > ".join(stack[k] for k in sorted(stack)))
                    break
        pos += len(ln) + 1
    return offs, paths


def _split_table(md: str, size: int) -> list[str]:
    lines = md.split("\n")
    # 캡션 줄(표 위의 설명) + 헤더 2줄 분리
    i = next((k for k, ln in enumerate(lines) if ln.startswith("|---")), None)
    if i is None or len(md) <= size:
        return [md]
    head = "\n".join(lines[: i + 1])
    rows, out, cur = lines[i + 1:], [], []
    cur_len = len(head)
    for r in rows:
        if cur and cur_len + len(r) + 1 > size:
            out.append(head + "\n" + "\n".join(cur))
            cur, cur_len = [], len(head)
        cur.append(r)
        cur_len += len(r) + 1
    if cur:
        out.append(head + "\n" + "\n".join(cur))
    return out


def chunk_documents(docs: list[dict], ccfg: dict) -> list[dict]:
    """본문: 문서 전체를 이어 붙여 RecursiveCharacterTextSplitter(정규식 구분자, 글머리 보존)로 분할
    → 시작 오프셋으로 페이지·섹션 역추적.  표: 행 단위 분할(헤더 반복).  그림 캡션: 본문과 같은 분할기."""
    size, overlap = int(ccfg["chunk_size"]), resolve_overlap(ccfg)
    min_chars = int(ccfg.get("min_chunk_chars", 30))
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                              separators=build_separators(ccfg.get("separators")),
                                              is_separator_regex=True, keep_separator="start",
                                              add_start_index=True, strip_whitespace=True)
    table_factor = float(ccfg.get("table_size_factor", 2.0))
    use_section = bool(ccfg.get("section_headers", True))
    chunks: list[dict] = []
    for doc in docs:
        idx = 0

        def add(text: str, page: int, kind: str, section: str = ""):
            nonlocal idx
            if len(text.strip()) < min_chars:
                return
            header = ""
            if ccfg.get("prepend_title", True):
                header = (f"문서: {doc['title']}" + (f" | p.{page}" if page else "")
                          + (f" | 섹션: {section}" if section else "") + "\n")
            chunks.append({
                "chunk_id": f"{doc['doc_id']}::c{idx:05d}", "doc_id": doc["doc_id"], "source": doc["source"],
                "title": doc["title"], "doc_type": doc["doc_type"], "page": page, "kind": kind,
                "section": section, "text": header + text.strip(), "body": text.strip(),
            })
            idx += 1

        # 레코드를 문서 순서대로 훑으며 본문은 이어 붙이고, 표·그림은 '나온 위치'(anchor)를 기억 → 섹션 경로 계산
        full, starts, pages, anchors = "", [], [], []
        for r in doc["records"]:
            if r["kind"] == "text":
                starts.append(len(full))
                pages.append(r.get("page"))
                full += r["text"] + "\n\n"
            else:
                anchors.append((r, len(full)))
        h_offs, h_paths = _heading_index(full) if use_section else ([], [])

        def section_at(off: int) -> str:
            i = bisect.bisect_right(h_offs, off) - 1
            return h_paths[i] if i >= 0 else ""

        if full.strip():
            for d in splitter.create_documents([full]):
                s0 = d.metadata.get("start_index", 0)
                pg = pages[max(0, bisect.bisect_right(starts, s0) - 1)] if starts else None
                add(d.page_content, pg, "text", section_at(s0))
        for r, off in anchors:
            sec = section_at(max(off - 1, 0)) if use_section else ""
            if r["kind"] == "table":
                for part in _split_table(r["text"], int(size * table_factor)):
                    if len(part) > size * (table_factor + 1):  # 셀 하나가 매우 긴 경우
                        for sub in splitter.split_text(part):
                            add(sub, r.get("page"), "table", sec)
                    else:
                        add(part, r.get("page"), "table", sec)
            elif r["kind"] == "image_caption":
                for d in splitter.split_text(r["text"]):
                    add(d, r.get("page"), "image_caption", sec)
    return chunks


def chunk_stats(chunks: list[dict]) -> dict:
    """청크 품질 지표: 개수·평균 길이·문장 중간 절단 비율·섹션 부착률"""
    import statistics

    body = [c for c in chunks if c["kind"] == "text"]
    cut = sum(1 for c in body if not re.search(r"(?:[.?!:;)\]」』%]|[다요음함임됨])\s*$", c["body"]))
    lens = [len(c["body"]) for c in chunks]
    return {"청크 수": len(chunks), "본문": len(body), "표": sum(c["kind"] == "table" for c in chunks),
            "평균 길이": round(statistics.mean(lens), 1) if lens else 0,
            "중앙값": int(statistics.median(lens)) if lens else 0,
            "문장 중간 절단(%)": round(100 * cut / max(len(body), 1), 1),
            "섹션 부착(%)": round(100 * sum(bool(c.get("section")) for c in chunks) / max(len(chunks), 1), 1)}


# ====================================================================================================
# QA 데이터 분할 (Split Ratio / Seed / Stratify / Evidence_keywords)   (원래 split.py)
# ====================================================================================================
# QA 데이터 분할: Split Ratio / Seed / Stratify / Evidence_keywords 를 각각 조절.
def filter_by_evidence_keywords(samples: list[dict], ek: dict) -> tuple[list[dict], dict]:
    """evidence_keywords 설정에 따라 샘플 정제/필터링.
    - require_in_context: 근거 텍스트에 실제로 있는 키워드만 남김 (환각 키워드 제거)
    - max_keywords: 샘플당 키워드 상한
    - min_keywords: 남은 키워드가 이보다 적으면 제외
    - must_include(+match any/all): 지정 키워드가 (키워드·질문·정답 중) 있는 샘플만
    - exclude: 지정 키워드가 있는 샘플 제외
    """
    rep = Counter()
    if not ek.get("enabled", True):
        return samples, {"filtered_out": 0}
    must = [compact(k) for k in ek.get("must_include", []) or [] if k]
    excl = [compact(k) for k in ek.get("exclude", []) or [] if k]
    mode = ek.get("match", "any")
    out = []
    for s in samples:
        kws = list(dict.fromkeys(k.strip() for k in s.get("evidence_keywords", []) if k and k.strip()))
        if ek.get("require_in_context", True) and s.get("evidence_text"):
            ctx = compact(s["evidence_text"])
            before = len(kws)
            kws = [k for k in kws if compact(k) and compact(k) in ctx]
            rep["keywords_dropped_not_in_context"] += before - len(kws)
        kws = kws[: int(ek.get("max_keywords", 8))]
        if len(kws) < int(ek.get("min_keywords", 0)):
            rep["removed_min_keywords"] += 1
            continue
        hay = compact(" ".join(kws) + s["question"] + s.get("ground_truth", ""))
        if must:
            hit = [m in hay for m in must]
            if not (all(hit) if mode == "all" else any(hit)):
                rep["removed_must_include"] += 1
                continue
        if excl and any(x in hay for x in excl):
            rep["removed_exclude"] += 1
            continue
        out.append({**s, "evidence_keywords": kws})
    rep["kept"] = len(out)
    return out, dict(rep)


def _normalize_ratio(ratio) -> list[float]:
    if isinstance(ratio, str):
        ratio = [float(x) for x in ratio.replace("/", ",").replace(":", ",").split(",") if x.strip()]
    ratio = [float(r) for r in ratio]
    if len(ratio) == 2:
        ratio = [ratio[0], 0.0, ratio[1]]
    if len(ratio) != 3 or sum(ratio) <= 0:
        raise ValueError("split.ratio 는 [train, valid, test] 또는 [train, test] 형식")
    t = sum(ratio)
    return [r / t for r in ratio]


def _labels(samples, key, min_count):
    """층화 라벨. 표본이 너무 적은 층은 '__rare__' 로 합쳐 sklearn 오류 방지."""
    if key in (None, "", "none"):
        return None
    raw = [str(s.get(key, "unknown")) for s in samples]
    cnt = Counter(raw)
    lab = [x if cnt[x] >= min_count else "__rare__" for x in raw]
    c2 = Counter(lab)
    if c2.get("__rare__", 0) and c2["__rare__"] < min_count:  # rare 도 부족하면 최빈층에 흡수
        top = c2.most_common(1)[0][0]
        lab = [top if x == "__rare__" else x for x in lab]
    return lab if len(set(lab)) > 1 else None


def split_samples(samples: list[dict], ratio, seed: int, stratify: str | None) -> tuple[dict, dict]:
    from sklearn.model_selection import train_test_split

    tr, va, te = _normalize_ratio(ratio)
    idx = list(range(len(samples)))
    report = {"ratio": [round(tr, 3), round(va, 3), round(te, 3)], "seed": seed, "stratify": stratify or "none"}
    if len(samples) < 3:
        return {"train": samples, "valid": [], "test": []}, report

    def _split(ids, test_size, lab_src):
        if test_size <= 0 or len(ids) < 2:
            return ids, []
        labs = _labels([samples[i] for i in ids], stratify, 2)
        try:
            return train_test_split(ids, test_size=test_size, random_state=seed, stratify=labs)
        except ValueError as e:  # 층 수 > 테스트 크기 등
            report.setdefault("warnings", []).append(f"stratify 해제({lab_src}): {e}")
            return train_test_split(ids, test_size=test_size, random_state=seed)

    rest, test = _split(idx, te, "test")
    valid_share = va / (tr + va) if (tr + va) > 0 else 0
    train, valid = _split(rest, valid_share, "valid")
    out = {"train": [samples[i] for i in train], "valid": [samples[i] for i in valid],
           "test": [samples[i] for i in test]}
    key = stratify if stratify not in (None, "", "none") else "doc_id"
    report["counts"] = {k: len(v) for k, v in out.items()}
    report["by_stratum"] = {k: dict(Counter(str(s.get(key)) for s in v)) for k, v in out.items()}
    return out, report
