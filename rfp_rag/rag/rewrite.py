"""① 질문 해석 + ④ 답변에 검색용 질문 전달 — GPT 가 사용자의 질문을 '검색용 질문'으로 다시 쓰고, 답변 GPT 도 그 뜻으로 답하게 함.
담당: 질문 해석 · 답변 · 챗봇.  (예전 02·03 노트북 1-1b 셀에 복사돼 있던 코드)

    RW = QueryRewriter("gpt-5-mini", effort="low", n=2, price=usd)   # usd(입력토큰, 출력토큰, 모델) → 달러
    RW.rewrite("dnjftp 세액공제 한도")  →  {"queries": ["월세 세액공제 한도", ...], "prompt_tokens": ..., "usd": ..., "new": True}
    install_answer_with_rewrite(chat, SR)   # chat.gen_.answer 를 감싸 [원 질문] + [검색용 질문] 을 함께 보냄
"""
from __future__ import annotations

import hashlib
import json
import time

REWRITE_PROMPT = (
    "너는 공공기관 제안요청서(RFP) 문서 검색 도우미다. 사용자의 질문을 문서 검색에 쓸 '검색용 질문'으로 바꿔라.\n"
    "- 오타·띄어쓰기를 바로잡고, 한영 자판 오입력(예: dnjftp → 월세)은 한글로 복원하라.\n"
    "- 한자는 한글로, 영어·영문 혼용은 한국어로 바꿔라. 단 기관명·사업명·약어·요구사항 ID(예: PER-002)는 그대로 둔다.\n"
    "- 줄임말은 풀어 쓴다(예: 연정 → 연말정산, 지스트 → 광주과학기술원).\n"
    "- [이전 대화]가 있으면 '그럼', '그 사업', '거기' 같은 말을 이전 대화의 기관·사업명으로 바꿔 질문만 보고도 이해되게 하라.\n"
    "- 질문에 답하지 말고, 질문에 없는 사실·숫자를 지어내지 마라.\n"
    "- 입력이 한두 글자·자판 잡음(예: d, ㅇ, asdf)처럼 뜻을 알 수 없거나, 문서와 무관한 인사·잡담(예: 안녕, 너는 누구야)이면 "
    "이전 대화로 채워 넣지 말고 빈 목록 {{\"queries\": []}} 을 출력하라.\n"
    "- 서로 다른 표현으로 {n}개를 만들어라. 첫 번째가 가장 정확한 표준 표현이다.\n"
    'JSON 으로만 출력: {{"queries": ["...", "..."]}}')

ZERO_USAGE = dict(calls=0, prompt=0, completion=0, reasoning=0, usd=0.0)


class QueryRewriter:
    """질문 → 검색용 질문 N개 (GPT · JSON 출력).  같은 질문·설정·이전 대화면 캐시를 재사용(비용 0).
    attach(path) 로 jsonl 에 저장하면 노트북을 다시 실행해도 같은 질문은 다시 호출하지 않음 (02)."""

    def __init__(self, model: str = "gpt-5-mini", effort: str | None = "low", n: int = 2, max_tokens: int = 2048,
                 history_turns: int = 2, price=None, timeout: int = 60, prompt: str = REWRITE_PROMPT):
        from .llm import Generator

        self.gen = Generator(model, timeout)
        self.model, self.effort, self.n, self.max_tokens = model, effort, int(n), int(max_tokens)
        self.history_turns, self.price, self.prompt = int(history_turns), price, prompt
        self.cache: dict[str, dict] = {}
        self.by_q: dict[str, dict] = {}          # 이전 대화 없이 해석한 질문 → 기록 (02 표·CSV 용)
        self.new_recs: list[dict] = []           # 이 세션에서 실제로 GPT 를 부른 기록 (비용 합계용)
        self.store = None

    # ── 저장 (02: 해석 결과를 파일에 남겨 재실행 때 다시 호출하지 않음)
    def attach(self, path: str):
        from .notebook_checkpoint import JsonlStore

        self.store = JsonlStore(path, key="key")
        for rec in self.store.load():
            self.cache[rec["key"]] = rec
            if not rec.get("n_history"):
                self.by_q[rec["question"]] = rec

    def rw_for(self, question: str) -> dict | None:
        return self.by_q.get(question)

    def usage(self) -> dict:
        r = self.new_recs
        return dict(calls=len(r), prompt=sum(x["prompt_tokens"] for x in r), completion=sum(x["completion_tokens"] for x in r),
                    reasoning=sum(x["reasoning_tokens"] for x in r), usd=sum(x.get("usd") or 0 for x in r))

    def usage_since(self, before: dict) -> dict:
        now = self.usage()
        return {k: now[k] - before[k] for k in now}

    def _history_lines(self, history: list[dict] | None) -> list[str]:
        if not history or not self.history_turns:
            return []
        return [f"{'사용자' if m['role'] == 'user' else '답변'}: {str(m['content'])[:300]}"
                for m in history[-2 * self.history_turns:]]

    def rewrite(self, question: str, history: list[dict] | None = None) -> dict:
        hist = self._history_lines(history)
        key = hashlib.sha1(json.dumps([question, hist, self.model, self.effort, self.n],
                                      ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
        if key in self.cache:
            return {**self.cache[key], "new": False}
        user = (("[이전 대화]\n" + "\n".join(hist) + "\n\n") if hist else "") + f"[질문]\n{question}"
        t0, out, qs, err = time.time(), {}, [], None
        try:
            out = self.gen.chat([{"role": "system", "content": self.prompt.format(n=self.n)},
                                 {"role": "user", "content": user}],
                                temperature=0.0, max_tokens=self.max_tokens, reasoning_effort=self.effort, json_mode=True)
            if out.get("finish_reason") == "length":
                err = f"출력 상한 도달 — REWRITE_MAX_TOKENS({self.max_tokens}) 를 늘리거나 추론 강도를 낮추세요"
            qs = [q.strip() for q in (json.loads(out.get("text") or "{}").get("queries") or [])
                  if isinstance(q, str) and q.strip()]
        except Exception as e:  # noqa: BLE001  — 해석이 실패해도 원래 질문으로 검색은 계속
            err = err or f"{type(e).__name__}: {str(e)[:150]}"
        qs = [q for q in dict.fromkeys(qs) if q != question.strip()][:self.n]
        rec = dict(key=key, question=question, n_history=len(hist), queries=qs, model=self.model,
                   effort=out.get("effort"), prompt_tokens=out.get("prompt_tokens") or 0,
                   completion_tokens=out.get("completion_tokens") or 0, reasoning_tokens=out.get("reasoning_tokens") or 0,
                   seconds=round(time.time() - t0, 2), error=err)
        rec["usd"] = self.price(rec["prompt_tokens"], rec["completion_tokens"], self.model) if self.price else None
        self.cache[key] = rec
        if not hist:
            self.by_q[question] = rec
        if out:
            self.new_recs.append(rec)
        if self.store is not None:
            self.store.append(rec)
        return {**rec, "new": True}


def install_answer_with_rewrite(chat, sr, enabled: bool = True):
    """chat.gen_.answer 를 감싸 답변 GPT 에 [원 질문] + [정규화된 질문(= 검색용 질문 1순위)] 을 함께 보냄.
    결과 augmentation 에 '답변에 쓴 질문'·'검색용 질문 반영' 을 남김 (02 answers.jsonl 에도 저장).
    셀을 다시 실행해도 두 번 감싸지 않도록 항상 클래스의 원래 메서드에서 시작."""
    base = type(chat.gen_).answer.__get__(chat.gen_)

    def answer_with_rewrite(question, hits, history=None, original_question=None, **kw):
        info = sr.last if (sr.last and question in str(sr.last.get("question") or "")) else {}   # 방금 이 질문으로 검색한 결과
        qs = (info.get("rewrite") or {}).get("queries") or []
        q_used = qs[0] if (enabled and qs) else question
        out = base(q_used, hits, history=history, original_question=original_question or question, **kw)
        out["augmentation"]["답변에 쓴 질문"] = q_used
        out["augmentation"]["검색용 질문 반영"] = q_used != question
        if info:
            info["answer_question"] = q_used
        return out

    chat.gen_.answer = answer_with_rewrite
    return answer_with_rewrite
