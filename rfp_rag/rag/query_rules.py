"""① LLM 호출 전 규칙 기반 질의 전처리 (수 ms).

순서: 무의미 입력 차단 → 한자→한글 → 영문 도메인 용어 → 한영 자판 변환 → 오타 사전 → 도메인 용어 사전 → 정규식 규칙
각 단계에서 무엇이 바뀌었는지 `steps` 로 남겨 노트북에서 확인 가능.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# ------------------------------------------------------------------ 두벌식 자판
_KEYMAP = dict(zip("qwertyuiopasdfghjklzxcvbnm", "ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔㅁㄴㅇㄹㅎㅗㅓㅏㅣㅋㅌㅊㅍㅠㅜㅡ"))
_KEYMAP.update({"Q": "ㅃ", "W": "ㅉ", "E": "ㄸ", "R": "ㄲ", "T": "ㅆ", "O": "ㅒ", "P": "ㅖ"})
_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
         "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
_VV = {("ㅗ", "ㅏ"): "ㅘ", ("ㅗ", "ㅐ"): "ㅙ", ("ㅗ", "ㅣ"): "ㅚ", ("ㅜ", "ㅓ"): "ㅝ", ("ㅜ", "ㅔ"): "ㅞ",
       ("ㅜ", "ㅣ"): "ㅟ", ("ㅡ", "ㅣ"): "ㅢ"}
_CC = {("ㄱ", "ㅅ"): "ㄳ", ("ㄴ", "ㅈ"): "ㄵ", ("ㄴ", "ㅎ"): "ㄶ", ("ㄹ", "ㄱ"): "ㄺ", ("ㄹ", "ㅁ"): "ㄻ",
       ("ㄹ", "ㅂ"): "ㄼ", ("ㄹ", "ㅅ"): "ㄽ", ("ㄹ", "ㅌ"): "ㄾ", ("ㄹ", "ㅍ"): "ㄿ", ("ㄹ", "ㅎ"): "ㅀ",
       ("ㅂ", "ㅅ"): "ㅄ"}
_ENGLISH_STOP = set("""a an the and or but if of to in on at by for from with as is are was were be been am do does did
done have has had not no yes it its this that these those there here what when where who whom whose which why how
can could will would shall should may might must i you he she we they me my your our their them his her us rate tax
flat monthly rent deadline income payment statement earned deduction deductions get got go please tell show about
yearly year end settlement limit cap how much many rule rules law file filing report due date time""".split())


def _syllable(cho: str, jung: str, jong: str = "") -> str:
    return chr(0xAC00 + (_CHO.index(cho) * 21 + _JUNG.index(jung)) * 28 + _JONG.index(jong))


def compose_jamo(jamos: str) -> tuple[str, int]:
    """자모열 → 음절 (두벌식 오토마타). return (문자열, 남은 낱자모 수)"""
    out, i, n, left = [], 0, len(jamos), 0
    while i < n:
        c = jamos[i]
        if c in _CHO and i + 1 < n and jamos[i + 1] in _JUNG:
            cho, v = c, jamos[i + 1]
            i += 2
            if i < n and (v, jamos[i]) in _VV:
                v = _VV[(v, jamos[i])]
                i += 1
            jong = ""
            if i < n and jamos[i] in _JONG and not (i + 1 < n and jamos[i + 1] in _JUNG):
                jong = jamos[i]
                i += 1
                if i < n and (jong, jamos[i]) in _CC and not (i + 1 < n and jamos[i + 1] in _JUNG):
                    jong = _CC[(jong, jamos[i])]
                    i += 1
            out.append(_syllable(cho, v, jong))
        else:
            out.append(c)
            left += 1
            i += 1
    return "".join(out), left


def eng_to_hangul(token: str) -> tuple[str, int]:
    jamos = "".join(_KEYMAP.get(ch, _KEYMAP.get(ch.lower(), "")) for ch in token)
    return compose_jamo(jamos)


# ------------------------------------------------------------------ 결과
@dataclass
class RuleResult:
    original: str
    normalized: str
    blocked: bool = False
    message: str = ""
    steps: list = field(default_factory=list)   # (규칙, 변경 전, 변경 후)
    elapsed_ms: float = 0.0

    @property
    def changed(self) -> bool:
        return self.original.strip() != self.normalized

    def summary(self) -> str:
        if self.blocked:
            return f"⛔ 차단: {self.message}"
        return " · ".join(f"{r}: {a}→{b}" for r, a, b in self.steps) or "변경 없음"


# ------------------------------------------------------------------ 전처리기
_JOSA = "은는이가을를에의도로와과만께서나랑요"


def _load_dict(path: str | Path | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class QueryPreprocessor:
    def __init__(self, cfg: dict, dict_path: str | Path | None = None, extra_vocab: set[str] | None = None):
        self.cfg = cfg
        d = cfg.get("dict_data") or _load_dict(dict_path or cfg.get("dict_path"))  # 노트북 셀의 사전 > YAML 파일
        self.terms: dict = {**(d.get("terms") or {})}
        self.typos: dict = {**(d.get("typos") or {})}
        self.en_terms: dict = {re.sub(r"[\s\-]+", " ", k.lower()): v for k, v in (d.get("en_terms") or {}).items()}
        self.regex_rules: list = [tuple(x) for x in (d.get("regex") or [])]
        self.orgs: dict = d.get("orgs") or {}
        self.vocab = set(extra_vocab or set()) | set(self.terms.values()) | set(self.typos.values())
        self.vocab |= {w for v in self.terms.values() for w in str(v).split()}
        self._ko_re = {name: self._build_ko(m) for name, m in (("typos", self.typos), ("terms", self.terms))}
        self._en_re = self._build_en(self.en_terms)

    # 한국어 단어 경계: 왼쪽은 한글이 아니고, 오른쪽은 한글이 아니거나 조사로 시작
    def _build_ko(self, mapping: dict):
        if not mapping:
            return None
        keys = sorted(mapping, key=len, reverse=True)
        alt = "|".join(re.escape(k) for k in keys)
        return re.compile(rf"(?<![가-힣])({alt})(?=$|[^가-힣]|[{_JOSA}])")

    def _build_en(self, mapping: dict):
        if not mapping:
            return None
        keys = sorted(mapping, key=len, reverse=True)
        alt = "|".join(re.escape(k).replace(r"\ ", r"[\s\-]+") for k in keys)
        return re.compile(rf"(?<![A-Za-z])({alt})(?![A-Za-z])", re.IGNORECASE)

    # ---------------------------------------------------------- 1) 무의미 입력
    def check_block(self, q: str) -> str | None:
        b = self.cfg.get("block", {})
        if not b.get("enabled", True):
            return None
        s = q.strip()
        chars = [c for c in s if not c.isspace()]
        letters = sum(1 for c in chars if "가" <= c <= "힣" or c.isascii() and c.isalpha()
                      or "一" <= c <= "鿿")
        digits = sum(1 for c in chars if c.isdigit())
        jamo = sum(1 for c in chars if "ㄱ" <= c <= "ㅣ")
        if len(s) < int(b.get("min_chars", 2)):
            return "너무 짧음"
        if letters == 0:
            return "문자 없음(이모지·기호·숫자만)"
        if chars and (letters + digits) / len(chars) < float(b.get("min_valid_ratio", 0.4)):
            return f"한글·영문 비율 낮음({(letters + digits) / len(chars):.0%})"
        if chars and jamo / len(chars) > float(b.get("max_jamo_ratio", 0.5)):
            return "자음·모음만 입력(ㅋㅋ/ㅎㅎ 등)"
        return None

    # ---------------------------------------------------------- 2) 한자
    def hanja_to_hangul(self, q: str) -> str:
        from .text_utils import _HANJA_RUN, hanja_reading

        return _HANJA_RUN.sub(lambda m: hanja_reading(m.group()), q)

    # ---------------------------------------------------------- 3) 한영 자판
    def fix_keyboard(self, q: str, steps: list) -> str:
        kc = self.cfg.get("keyboard", {})
        min_syl = int(kc.get("min_syllables", 2))

        def conv(m):
            tok = m.group()
            if tok.lower() in _ENGLISH_STOP:
                return tok
            if len(tok) > 1 and tok[0].isupper() and tok[1:].islower():  # 'Monthly' 같은 영어 단어
                return tok
            if tok.isupper() and len(tok) <= 6:  # RFP, SFR, GIST 등 약어
                return tok
            ko, left = eng_to_hangul(tok)
            n_syl = sum(1 for c in ko if "가" <= c <= "힣")
            if left == 0 and (n_syl >= min_syl or ko in self.vocab):
                steps.append(("한영자판", tok, ko))
                return ko
            return tok

        return re.sub(r"[A-Za-z]+", conv, q)

    # ---------------------------------------------------------- 4) 사전 치환
    def _apply(self, q: str, rx, mapping: dict, name: str, steps: list, ci: bool = False) -> str:
        if rx is None:
            return q

        def sub(m):
            k = m.group(1)
            key = re.sub(r"[\s\-]+", " ", k.lower()) if ci else k
            v = mapping.get(key, k)
            if v != k:
                steps.append((name, k, v))
            return v

        return rx.sub(sub, q)

    def __call__(self, question: str) -> RuleResult:
        t0 = time.perf_counter()
        steps: list = []
        res = RuleResult(original=question, normalized=question.strip())
        why = self.check_block(question)
        if why:
            res.blocked = True
            res.message = f"{self.cfg.get('guide_message', '질문을 다시 입력해 주세요.')} ({why})"
            res.elapsed_ms = (time.perf_counter() - t0) * 1000
            return res
        q = question.strip()
        if self.cfg.get("hanja", {}).get("enabled", True):
            q2 = self.hanja_to_hangul(q)
            if q2 != q:
                steps.append(("한자", q, q2))
            q = q2
        if self.cfg.get("dictionary", {}).get("enabled", True):
            q = self._apply(q, self._en_re, self.en_terms, "영문용어", steps, ci=True)
        if self.cfg.get("keyboard", {}).get("enabled", True):
            q = self.fix_keyboard(q, steps)
        if self.cfg.get("dictionary", {}).get("enabled", True):
            q = self._apply(q, self._ko_re["typos"], self.typos, "오타", steps)
            q = self._apply(q, self._ko_re["terms"], self.terms, "용어", steps)
            for pat, rep in self.regex_rules:
                q2 = re.sub(pat, rep, q)
                if q2 != q:
                    steps.append(("정규식", pat, rep))
                q = q2
        res.normalized = re.sub(r"\s+", " ", q).strip()
        res.steps = steps
        res.elapsed_ms = (time.perf_counter() - t0) * 1000
        return res
