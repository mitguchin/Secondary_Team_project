"""텍스트 정규화 · 한자 처리 · BM25 토크나이저."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_HANJA_RUN = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002A6DF]+")
_COMPAT_IDEO = re.compile(r"[豈-﫿\U0002F800-\U0002FA1F]")
_WS = re.compile(r"[ \t 　]+")


def normalize_text(text: str) -> str:
    """HWP/PDF 공통 정규화.
    - CJK 호환 한자(U+F900~)만 NFKC 로 통합 한자로 매핑 (전체 NFKC 는 'ㅇ' 같은 호환 자모를 깨뜨리므로 금지)
    - 공백 정리, 3줄 이상 빈 줄 축소
    """
    if not text:
        return ""
    text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")  # 깨진 서로게이트 제거
    text = _COMPAT_IDEO.sub(lambda m: unicodedata.normalize("NFKC", m.group()), text)
    text = text.replace("\r", "\n").replace("\x00", "")
    text = "\n".join(_WS.sub(" ", ln).strip() for ln in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@lru_cache(maxsize=1)
def _hanja_mod():
    try:
        import hanja  # pip install hanja
        return hanja
    except ImportError:
        return None


def hanja_reading(run: str) -> str:
    mod = _hanja_mod()
    if mod is None:
        return run
    try:
        return mod.translate(run, "substitution")
    except Exception:
        return run


def apply_hanja(text: str, mode: str = "append") -> str:
    """mode: none | append('必(필)') | replace('필')"""
    if mode == "none" or not _HANJA_RUN.search(text):
        return text

    def _sub(m: re.Match) -> str:
        run = m.group()
        reading = hanja_reading(run)
        if reading == run:
            return run
        if mode == "replace":
            return reading
        # 이미 '契約(계약)' 형태로 병기돼 있으면 그대로 둠
        tail = text[m.end(): m.end() + len(reading) + 2]
        if tail.startswith(f"({reading}"):
            return run
        return f"{run}({reading})"

    return _HANJA_RUN.sub(_sub, text)


# ---------------------------------------------------------------- tokenizers
_KIWI_KEEP = ("NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "XR", "SL", "SH", "SN", "MAG")


class KiwiTokenizer:
    """한국어 형태소 기반 BM25 토크나이저 (kiwipiepy). 한자(SH) 토큰은 한글 독음도 함께 색인."""

    name = "kiwi"

    def __init__(self, num_workers: int = -1):
        from kiwipiepy import Kiwi

        # num_workers=-1: 모든 CPU 코어 사용 (batch 에서 여러 청크를 병렬 분석). 단일 문장(질문)은 그대로 1스레드
        self.kiwi = Kiwi(num_workers=num_workers)

    def _keep(self, toks) -> list[str]:
        out = []
        for tok in toks:
            if tok.tag.startswith(_KIWI_KEEP):
                form = tok.form.lower()
                out.append(form)
                if tok.tag == "SH":
                    r = hanja_reading(tok.form)
                    if r != tok.form:
                        out.append(r)
        return out

    def __call__(self, text: str) -> list[str]:
        return self._keep(self.kiwi.tokenize(text))

    def batch(self, texts: list[str]) -> list[list[str]]:
        """여러 청크를 멀티스레드로 한꺼번에 분석 (결과는 __call__ 과 동일, 4코어 기준 약 3~4배 빠름)"""
        return [self._keep(t) for t in self.kiwi.tokenize(list(texts))]


class HFTokenizer:
    """HuggingFace 토크나이저(임베딩 모델과 동일 어휘) 기반 subword BM25."""

    name = "hf"

    def __init__(self, model_name: str):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, text: str) -> list[str]:
        toks = self.tok.tokenize(text.lower())
        return [t for t in (x.replace("▁", "") for x in toks) if t and not re.fullmatch(r"\W", t)]

    def batch(self, texts):
        return [self(t) for t in texts]


class WhitespaceTokenizer:
    name = "whitespace"

    def __call__(self, text: str) -> list[str]:
        return re.findall(r"[\w㐀-鿿]+", text.lower())

    def batch(self, texts):
        return [self(t) for t in texts]


def make_tokenizer(kind: str, hf_model: str | None = None):
    if kind == "kiwi":
        return KiwiTokenizer()
    if kind == "hf":
        return HFTokenizer(hf_model)
    return WhitespaceTokenizer()


def compact(s: str) -> str:
    """키워드 매칭용: 공백/기호 제거 + 소문자"""
    return re.sub(r"[\s\W_]+", "", s).lower()


# ==================================================================================================
# hwp-hwpx-parser 출력 후처리 (라이브러리가 주는 텍스트를 정리할 뿐, 파일을 직접 읽지 않음)
# ==================================================================================================
# hwp-hwpx-parser 가 탭 제어문자(8 wchar) 일부를 글자로 내보내는 문제 정리: '\t洴ȃ 1' → '\t 1', '개요  練ȃ  1' → '개요   1'
TAB_GARBAGE = re.compile(r"(?<=[\t ])[^\s가-힣A-Za-z0-9|]{0,2}[Ā-˿](?=[\s|]|$)")
TOC_MARK = re.compile(r"(?:^|\|)[\s\-–\[\(<【]*(?:목\s*차|차\s*례|CONTENTS|Contents)[\s\-–\]\)>】]*(?:\||$)")
TOC_LINE = re.compile(r"(?:\.{3,}|·{3,}|…{2,}|\s\d{1,3})\s*\|?\s*$")
MULTI_ENTRY = re.compile(r"\s\d{1,3}(?=\s|$)")
NEUTRAL = re.compile(r"^\s*$|^[\s|:\-]+$|^(?:\|\s*)+\|?\s*$")      # 빈 줄, |---|, 빈 셀 행
BODY_START = re.compile(r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ][.\s]|제\s*1\s*[장편]|1\.\s|<[^>]{2,20}>|□|❏)")


def clean_tab_garbage(text: str) -> str:
    return TAB_GARBAGE.sub("", text)


def is_toc_line(ln: str) -> bool:
    s = ln.strip()
    return bool(s) and (bool(TOC_LINE.search(s)) or len(MULTI_ENTRY.findall(s)) >= 3)


def _compact(s: str) -> str:
    s = re.sub(r"(?:\.{3,}|·{3,}|…+)?\s*\d{1,3}\s*\|?\s*$", "", s)        # 끝의 쪽번호·점선 제거
    return re.sub(r"[\s\W_]+", "", s)


def split_front_matter(text: str, max_ratio: float = 0.25):
    """표지 + 목차 분리 → (front, body, 방법). HWP 는 페이지 정보가 없어 '앞 N페이지' 대신 사용.
    ① 목차에 쪽번호가 있으면: 목차 줄이 끝나는 곳까지
    ② 쪽번호 없는 목차면: 목차 첫 항목(예: '1. 제안공모 개요')이 본문에 '다시' 나오는 곳부터 본문
    ③ 목차가 없으면: 앞 10% 안의 첫 본문 제목 전까지"""
    lines = text.split("\n")
    limit = int(len(lines) * max_ratio) or len(lines)
    mark = next((i for i, ln in enumerate(lines[:limit]) if TOC_MARK.search(ln)), None)
    if mark is not None:
        nxt = next((j for j in range(mark + 1, len(lines)) if not NEUTRAL.match(lines[j])), None)
        if nxt is None:
            return text, "", "목차 뒤 내용 없음"
        if is_toc_line(lines[mark]) or is_toc_line(lines[nxt]):          # ① 쪽번호 있는 목차
            end = mark
            for j in range(mark + 1, len(lines)):
                if NEUTRAL.match(lines[j]) or is_toc_line(lines[j]):
                    end = j if not NEUTRAL.match(lines[j]) else end
                    continue
                break
            return "\n".join(lines[:end + 1]), "\n".join(lines[end + 1:]), f"목차(쪽번호) 감지 줄 {mark}~{end}"
        first = _compact(lines[nxt])                                       # ② 쪽번호 없는 목차
        if len(first) >= 3:
            for j in range(nxt + 1, len(lines)):
                if _compact(lines[j]).startswith(first):
                    return "\n".join(lines[:j]), "\n".join(lines[j:]), f"목차 첫 항목 재등장 줄 {j}"
        return "\n".join(lines[:nxt]), "\n".join(lines[nxt:]), f"목차 표시만 감지 줄 {mark}"
    lim2 = max(5, int(len(lines) * 0.10))                                  # ③ 목차 없음
    first = next((i for i, ln in enumerate(lines[:lim2]) if BODY_START.match(ln)), None)
    if first:
        return "\n".join(lines[:first]), "\n".join(lines[first:]), f"첫 제목 감지 줄 {first}"
    return "", text, "감지 실패(제외 없음)"


def repair_md_tables(text: str) -> tuple[str, int]:
    """세로 병합 셀 때문에 칸이 모자란 행 → 왼쪽 칸을 윗행 값으로 채워 열 정렬 복원"""
    out, fixed, header_n, prev = [], 0, None, None
    for ln in text.split("\n"):
        if ln.startswith("|") and ln.rstrip().endswith("|"):
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if header_n is None:
                header_n, prev = len(cells), cells
            elif re.fullmatch(r"[\s|:\-]+", ln):
                pass
            else:
                if len(cells) < header_n and prev:
                    cells = prev[: header_n - len(cells)] + cells
                    fixed += 1
                    ln = "| " + " | ".join(cells) + " |"
                prev = cells
        else:
            header_n, prev = None, None
        out.append(ln)
    return "\n".join(out), fixed
