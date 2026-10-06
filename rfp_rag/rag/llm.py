"""gpt-5-mini 생성기.

주의: gpt-5 / gpt-5-mini 같은 reasoning 모델은 API 가 temperature·top_p 를 1(기본값) 외에는 거부합니다.
→ 사용자가 UI 에서 값을 바꿔도 먼저 그대로 보내 보고, 400 'unsupported' 가 오면 해당 파라미터를 빼고
  재시도 + 경고를 돌려줍니다 (gpt-4.1-mini 등 지원 모델로 바꾸면 그대로 적용됨).
또한 max_tokens 는 max_completion_tokens 로 전달되며 reasoning 토큰도 여기서 소모됩니다.
"""
from __future__ import annotations

import os
import re
import time

SYSTEM_PROMPT = (
    "당신은 {domain} 문서 기반 질의응답 도우미입니다.\n"
    "- 반드시 아래 [컨텍스트] 안의 내용만 근거로 답하세요.\n"
    "- 금액·기간·비율·날짜·자격요건·요구사항 ID 는 원문 표기 그대로 인용하세요.\n"
    "- 근거 문장 끝에 [번호] 로 출처를 표시하세요.\n"
    "- 질문에 사실과 다른 전제가 있으면(예: 잘못된 비율·기한) 먼저 '정정:' 으로 바로잡은 뒤 올바른 내용을 답하세요.\n"
    "- 두 사업/기관 비교 질문은 항목별 Markdown 표로 정리하고, 한쪽 문서에 없는 항목은 '문서에 없음'으로 표시하세요.\n"
    "- 요구사항 정리 질문은 요구사항 ID·명칭·핵심 내용을 목록으로 정리하세요.\n"
    "- 질문이 영어로만 되어 있으면 영어로, 그 외에는 한국어로 답하세요.\n"
    "{fallback}"
)
FALLBACK_STRICT = "- 컨텍스트에 답이 없으면 '제공된 문서에서 확인할 수 없습니다.' 라고 답하세요."
FALLBACK_GENERAL = ("- 컨텍스트에 답이 없으면 그 사실을 먼저 밝히고, 일반 지식으로 답할 경우 문장 앞에 "
                    "'[문서 외 일반지식]' 을 붙이세요.")

_UNSUPPORTED_CACHE: dict[str, set[str]] = {}


def format_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        kind = {"table": "표", "image_caption": "그림"}.get(h.get("kind"), "본문")
        loc = f"p.{h['page']}" if h.get("page") else (h.get("section") or "")[:40]
        parts.append(f"[{i}] ({h['source']} {loc}, {kind})\n{h.get('body', h['text'])}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 추론(reasoning) 강도
# 요청값 → 모델이 받는 값으로 자동 조정. 지원 안 하는 값이면 '가장 가까운 낮은(=싼) 값' 으로 바꿈.
#   OpenAI 문서(2026-10): gpt-5 계열 minimal·low·medium·high / gpt-6 Luna·Sol none·low·medium·high·xhigh·max
#                         gpt-6 Astra·6.1 Sol low~max (none 없음).  none = 추론 토큰 0 (가장 저렴·빠름)
EFFORT_LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
MODEL_EFFORTS = {
    "gpt-5": ["minimal", "low", "medium", "high"],
    "gpt-5-mini": ["minimal", "low", "medium", "high"],
    "gpt-5-nano": ["minimal", "low", "medium", "high"],
    "gpt-5.6-luna": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-6-luna": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-6-sol": ["none", "low", "medium", "high", "xhigh", "max"],
    "gpt-6-astra": ["low", "medium", "high", "xhigh", "max"],
    "gpt-6.1-sol": ["low", "medium", "high", "xhigh", "max"],
}
_EFFORT_CACHE: dict[tuple[str, str], str | None] = {}   # (모델, 요청값) → API 가 받아 준 값 (알 수 없는 모델용)


def _is_reasoning(model: str) -> bool:
    m = model.lower()
    return ((m.startswith(("gpt-5", "gpt-6")) and "chat" not in m) or bool(re.match(r"^o\d", m)))


def _supported_efforts(model: str) -> list[str] | None:
    m = model.lower()
    for k in sorted(MODEL_EFFORTS, key=len, reverse=True):     # 'gpt-5-mini-2025-08-07' → 'gpt-5-mini'
        if m == k or m.startswith(k + "-"):
            return MODEL_EFFORTS[k]
    return None


def _nearest(effort: str, allowed: list[str]) -> str:
    """allowed 중 effort 와 가장 가까운 값 (같은 거리면 낮은 값 = 토큰 절약)"""
    i = EFFORT_LEVELS.index(effort)
    return min(allowed, key=lambda a: (abs(EFFORT_LEVELS.index(a) - i), EFFORT_LEVELS.index(a)))


def resolve_effort(model: str, effort: str | None) -> tuple[str | None, str | None]:
    """→ (실제로 보낼 값 또는 None, 안내 문구)"""
    if not effort or not _is_reasoning(model):
        return None, (f"{model} 은(는) 추론 모델이 아니라 추론 강도 설정을 보내지 않습니다." if effort else None)
    effort = str(effort).lower()
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"REASONING_EFFORT={effort!r} → {EFFORT_LEVELS} 중 하나로 지정하세요")
    if (model, effort) in _EFFORT_CACHE:
        got = _EFFORT_CACHE[(model, effort)]
        return got, (None if got == effort else f"{model}: 추론 강도 '{effort}' 미지원 → '{got}' 로 호출")
    allowed = _supported_efforts(model)
    if allowed is None or effort in allowed:
        return effort, None
    got = _nearest(effort, allowed)
    return got, f"{model}: 추론 강도 '{effort}' 미지원 → '{got}' 로 호출 (지원: {', '.join(allowed)})"


def effort_table(models_efforts: list[tuple[str, str, str]]):
    """노트북 표시용: [(용도, 모델, 요청 강도)] → 실제 적용값"""
    import pandas as pd

    rows = []
    for use, model, eff in models_efforts:
        got, note = resolve_effort(model, eff)
        rows.append({"용도": use, "모델": model, "요청 추론 강도": eff, "실제 적용": got or "(보내지 않음)",
                     "추론 모델": "예" if _is_reasoning(model) else "아니오",
                     "지원 값": ", ".join(_supported_efforts(model) or ["(목록 없음 → 요청값 그대로 시도)"])
                     if _is_reasoning(model) else "-", "비고": note or ""})
    return pd.DataFrame(rows)


class Generator:
    def __init__(self, model: str = "gpt-5-mini", timeout: int = 120):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(timeout=timeout, max_retries=3)

    def chat(self, messages: list[dict], temperature: float | None = None, top_p: float | None = None,
             max_tokens: int = 2048, reasoning_effort: str | None = None, json_mode: bool = False) -> dict:
        import openai

        warnings = []
        effort, note = resolve_effort(self.model, reasoning_effort)
        if note and reasoning_effort and _is_reasoning(self.model):
            warnings.append(note)
        kw = {"model": self.model, "messages": messages, "max_completion_tokens": int(max_tokens)}
        if effort is not None:
            kw["reasoning_effort"] = effort
        # 추론 모델은 추론 강도가 none 일 때만 temperature/top_p 를 받음 → 그 외엔 처음부터 빼서 실패 호출(토큰·시간 낭비) 방지
        sampling_ok = (not _is_reasoning(self.model)) or effort == "none"
        if sampling_ok:
            if temperature is not None:
                kw["temperature"] = float(temperature)
            if top_p is not None:
                kw["top_p"] = float(top_p)
        elif (temperature not in (None, 1, 1.0)) or (top_p not in (None, 1, 1.0)):
            warnings.append(f"{self.model} 은(는) 추론 강도가 none 이 아니면 temperature·top_p 를 받지 않아 기본값으로 호출했습니다.")
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        dropped = _UNSUPPORTED_CACHE.setdefault(self.model, set())
        for p in list(dropped):
            if p in kw:
                kw.pop(p)
                warnings.append(f"{self.model} 은(는) '{p}' 조절을 지원하지 않아 기본값으로 호출했습니다.")
        tried = {effort} if effort else set()
        for _ in range(6):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(**kw)
                break
            except openai.BadRequestError as e:
                msg = str(e)
                bad = next((p for p in ("reasoning_effort", "temperature", "top_p", "response_format")
                            if p in kw and p in msg), None)
                if bad is None:
                    raise
                if bad == "reasoning_effort" and kw["reasoning_effort"] in EFFORT_LEVELS:
                    # 목록에 없는 모델이 그 값을 거부 → 가까운 값부터 차례로 시도 (빼 버리면 기본 medium 이라 토큰이 늘어남)
                    rest = [x for x in EFFORT_LEVELS if x not in tried]
                    if rest:
                        nxt = _nearest(kw["reasoning_effort"], rest)
                        tried.add(nxt)
                        _EFFORT_CACHE[(self.model, reasoning_effort)] = nxt
                        warnings.append(f"{self.model}: 추론 강도 '{kw['reasoning_effort']}' 거부됨 → '{nxt}' 로 재시도")
                        kw["reasoning_effort"] = nxt
                        continue
                kw.pop(bad)
                dropped.add(bad)
                warnings.append(f"{self.model} 은(는) '{bad}' 조절을 지원하지 않아 기본값으로 호출했습니다.")
        else:
            raise RuntimeError("OpenAI 호출 실패")
        ch = resp.choices[0]
        usage = resp.usage
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        text = ch.message.content or ""
        if ch.finish_reason == "length":
            warnings.append(f"max_tokens({max_tokens}) 도달 — reasoning 토큰 {reasoning_tokens}개 사용. "
                            "답이 잘렸거나 비었다면 max tokens 를 늘리거나 reasoning effort 를 낮추세요.")
        return {"text": text, "latency": round(time.time() - t0, 2), "finish_reason": ch.finish_reason,
                "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                "reasoning_tokens": reasoning_tokens, "effort": kw.get("reasoning_effort"), "warnings": warnings}

    def answer(self, question: str, hits: list[dict], history: list[dict] | None = None,
               original_question: str | None = None, domain: str = "공공기관 제안요청서(RFP)",
               allow_general: bool = False, history_turns: int = 3, **gen) -> dict:
        ctx = format_context(hits)
        q_block = f"[질문]\n{question}"
        if original_question and original_question.strip() != question.strip():
            q_block = f"[원 질문]\n{original_question}\n\n[정규화된 질문(오타·자판·한자 보정)]\n{question}"
        user = f"[컨텍스트]\n{ctx}\n\n{q_block}"
        sys_prompt = SYSTEM_PROMPT.format(domain=domain, fallback=FALLBACK_GENERAL if allow_general else FALLBACK_STRICT)
        msgs = [{"role": "system", "content": sys_prompt}]
        for m in (history or [])[-2 * int(history_turns):]:  # 직전 대화 N턴
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                msgs.append({"role": m["role"], "content": m["content"]})
        msgs.append({"role": "user", "content": user})
        out = self.chat(msgs, **gen)
        out["augmentation"] = {
            "질문 글자수": len(question),
            "검색 컨텍스트 글자수": len(ctx),
            "최종 프롬프트 글자수": sum(len(m["content"]) for m in msgs),
            "증강 배율(×)": round(sum(len(m["content"]) for m in msgs) / max(len(question), 1), 1),
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "reasoning_tokens": out["reasoning_tokens"],
            "reasoning_effort": out.get("effort"),
            "답변 글자수": len(out["text"]),
        }
        return out


class _Raw:
    """전처리 없음: 입력 질문을 그대로 사용"""

    def __init__(self, q: str):
        self.normalized, self.steps, self.elapsed_ms, self.blocked, self.message = q, [], 0.0, False, ""


class RAGChat:
    """질문 그대로 → (선택)기관 라우팅 → 하이브리드 검색 → gpt-5-mini 생성.  대화(멀티턴) 상태 유지.
    preproc=None(기본): 규칙·사전 전처리 없이 순수 질문으로 검색·생성. preproc 를 넘기면(CLI 옵션) 전처리 후 사용."""

    def __init__(self, retriever, generator: "Generator", preproc, retrieval: dict, gen: dict,
                 domain: str = "공공기관 제안요청서(RFP)", allow_general: bool = False, use_routing: bool = True,
                 history_turns: int = 3, followup_query: str = "none"):
        self.ret, self.gen_, self.pre = retriever, generator, preproc
        self.retrieval, self.gen_params = dict(retrieval), dict(gen)
        self.domain, self.allow_general, self.use_routing = domain, allow_general, use_routing
        self.history_turns, self.followup_query = history_turns, followup_query
        self.reset()

    def reset(self):
        self.history: list[dict] = []
        self.state: dict = {}
        self.last_question = ""

    def restore(self, turns: list[dict]):
        """체크포인트에서 대화 재개: 저장된 이전 턴으로 history·라우팅 상태 복원"""
        self.reset()
        for t in turns:
            if t.get("blocked"):
                continue
            self.history += [{"role": "user", "content": t["normalized"]}, {"role": "assistant", "content": t["answer"]}]
            r = t.get("route") or {}
            if r.get("mode") in ("single", "compare"):
                self.state = {"active_doc_ids": r.get("doc_ids", []), "active_orgs": r.get("orgs", [])}
            self.last_question = t["normalized"]

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        rr = self.pre(question) if self.pre is not None else _Raw(question)
        out = {"question": question, "normalized": rr.normalized, "rule_steps": rr.steps, "rule_ms": round(rr.elapsed_ms, 2),
               "blocked": rr.blocked, "route": {}, "hits": [], "answer": "", "warnings": []}
        if rr.blocked:  # LLM 호출 없이 즉시 안내
            out.update(answer=rr.message, total_s=round(time.perf_counter() - t0, 3))
            return out
        q = rr.normalized
        route = self.ret.router.route(q, self.state) if (self.use_routing and self.ret.router) else {"mode": "none"}
        rq = q
        if self.followup_query == "concat" and route.get("mode") == "inherit" and self.last_question:
            rq = f"{self.last_question} {q}"
        t1 = time.perf_counter()
        hits = self.ret.search_routed(rq, route, **self.retrieval)
        t2 = time.perf_counter()
        res = self.gen_.answer(q, hits, history=self.history, original_question=question, domain=self.domain,
                               allow_general=self.allow_general, history_turns=self.history_turns, **self.gen_params)
        self.history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["text"]}]
        self.last_question = q
        out.update(route=route, hits=hits, answer=res["text"], warnings=res["warnings"],
                   augmentation=res["augmentation"], retrieval_s=round(t2 - t1, 3), llm_s=res["latency"],
                   total_s=round(time.perf_counter() - t0, 3))
        return out


JUDGE_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}   # RAGAS 심판 누적 사용량


def reset_judge_usage():
    for k in JUDGE_USAGE:
        JUDGE_USAGE[k] = 0


def make_judge_llm(model: str, max_tokens: int = 4096, reasoning_effort: str | None = "low", timeout: int = 180):
    """RAGAS 심판용 ChatOpenAI.
    RAGAS 는 호출 직전에 llm.temperature=0.01 을 강제로 세팅하는데, 추론 모델은 이를 400 으로 거부함
    → 요청 payload 단계에서 temperature/top_p 를 제거. 호출마다 토큰 사용량을 JUDGE_USAGE 에 누적."""
    from langchain_openai import ChatOpenAI

    effort, _ = resolve_effort(model, reasoning_effort)

    class _SafeChatOpenAI(ChatOpenAI):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):  # noqa: D401
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            if _is_reasoning(self.model_name):
                payload.pop("temperature", None)
                payload.pop("top_p", None)
            return payload

        def _create_chat_result(self, response, generation_info=None):
            res = super()._create_chat_result(response, generation_info)
            try:
                u = (res.llm_output or {}).get("token_usage") or {}
                JUDGE_USAGE["calls"] += 1
                JUDGE_USAGE["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                JUDGE_USAGE["completion_tokens"] += int(u.get("completion_tokens") or 0)
                d = u.get("completion_tokens_details") or {}
                JUDGE_USAGE["reasoning_tokens"] += int((d.get("reasoning_tokens") if isinstance(d, dict) else 0) or 0)
            except Exception:  # noqa: BLE001
                pass
            return res

    kw = dict(model=model, max_tokens=max_tokens, timeout=timeout, max_retries=3)
    if effort:
        kw["reasoning_effort"] = effort
    return _SafeChatOpenAI(**kw)


def token_cost(prompt_tokens: int, completion_tokens: int, model: str, prices: dict | None) -> float | None:
    """예상 비용(USD). prices = {모델: (입력 $/1M, 출력 $/1M)} — 추론 토큰은 출력 토큰에 포함되어 과금"""
    if not prices:
        return None
    p = next((v for k, v in sorted(prices.items(), key=lambda kv: -len(kv[0])) if model.lower().startswith(k.lower())), None)
    if not p:
        return None
    return round(prompt_tokens / 1e6 * p[0] + completion_tokens / 1e6 * p[1], 4)


def has_api_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
