"""RAGAS 평가 — 샘플 단위로 계산해 표를 '실시간' 갱신 + 샘플 단위 체크포인트(재개 가능)."""
from __future__ import annotations

import math
import re
import os
import time
import warnings

import pandas as pd

from .async_utils import run_coro
from .notebook_checkpoint import JsonlStore, StopFlag
from .llm import format_context, make_judge_llm
from .text_utils import compact

warnings.filterwarnings("ignore", category=DeprecationWarning)

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "factual_correctness": "Factual Correctness",
    "answer_correctness": "Answer Correctness",
}
EXTRA_LABELS = {"evidence_kw_recall": "Evidence KW Recall", "gold_hit": "Gold Chunk Hit@k"}


class _NamedEmbeddings:
    """RAGAS 0.4.3 은 임베딩 호출마다 EmbeddingUsageEvent(model=<str>) 를 만드는데, HFEmbedder.model 은
    SentenceTransformer 객체라 pydantic 검증 오류 → answer_relevancy 가 NaN 이 됨. model 을 이름 문자열로 노출하는 래퍼."""

    def __init__(self, embedder):
        self._e = embedder
        self.model = str(getattr(embedder, "model_name", None) or type(embedder).__name__)

    def embed_query(self, text):
        return self._e.embed_query(text)

    def embed_documents(self, texts):
        return self._e.embed_documents(texts)

    async def aembed_query(self, text):
        return self._e.embed_query(text)

    async def aembed_documents(self, texts):
        return self._e.embed_documents(texts)


def build_metrics(names: list[str], judge_model: str, embedder, max_tokens: int, effort: str | None):
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # ragas 사용 통계 전송 끔
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (FactualCorrectness, Faithfulness, LLMContextPrecisionWithoutReference,
                               LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy)

    llm = LangchainLLMWrapper(make_judge_llm(judge_model, max_tokens, effort), bypass_temperature=True,
                              bypass_n=True)
    emb = LangchainEmbeddingsWrapper(_NamedEmbeddings(embedder))
    factory = {
        "faithfulness": lambda: Faithfulness(llm=llm),
        "answer_relevancy": lambda: ResponseRelevancy(llm=llm, embeddings=emb),
        "context_precision": lambda: LLMContextPrecisionWithReference(llm=llm),
        "context_recall": lambda: LLMContextRecall(llm=llm),
        "factual_correctness": lambda: FactualCorrectness(llm=llm),
        "context_precision_noref": lambda: LLMContextPrecisionWithoutReference(llm=llm),
        "answer_correctness": lambda: _answer_correctness(llm, emb),
    }
    return {n: factory[n]() for n in names if n in factory}


def _answer_correctness(llm, emb):
    """정답 대비 답변 정확도 (사실 일치 F1 75% + 의미 유사도 25%) — 정답(reference) 필요"""
    from ragas.metrics import AnswerCorrectness
    from ragas.run_config import RunConfig

    m = AnswerCorrectness(llm=llm, embeddings=emb)
    m.init(RunConfig())  # 의미 유사도(AnswerSimilarity) 준비
    return m


def _score(metric, sample) -> float:
    try:
        return float(run_coro(metric.single_turn_ascore(sample), timeout=900))  # Jupyter 에서도 동작
    except Exception as e:  # noqa: BLE001  — 한 지표 실패가 전체 평가를 멈추지 않도록
        print(f"[ragas] {getattr(metric, 'name', metric)} 오류: {type(e).__name__}: {str(e)[:200]}")
        return float("nan")


def _kw_recall(kws: list[str], contexts: list[str]) -> float:
    if not kws:
        return float("nan")
    hay = compact(" ".join(contexts))
    return sum(compact(k) in hay for k in kws) / len(kws)


def summarize(rows: list[dict], metric_names: list[str]) -> pd.DataFrame:
    cols = metric_names + ["evidence_kw_recall", "gold_hit"]
    recs = []
    for c in cols:
        vals = [r.get(c) for r in rows if isinstance(r.get(c), (int, float)) and not math.isnan(r.get(c))]
        label = METRIC_LABELS.get(c) or EXTRA_LABELS.get(c, c)
        recs.append({"지표": label, "평균": round(sum(vals) / len(vals), 4) if vals else None,
                     "최소": round(min(vals), 4) if vals else None, "최대": round(max(vals), 4) if vals else None,
                     "n": len(vals)})
    return pd.DataFrame(recs)


def rows_df(rows: list[dict], metric_names: list[str]) -> pd.DataFrame:
    cols = ["#", "question"] + metric_names + ["evidence_kw_recall", "gold_hit", "latency_s"]
    data = []
    for i, r in enumerate(rows, 1):
        d = {"#": i, "question": r["question"][:60]}
        for c in cols[2:]:
            v = r.get(c)
            d[c] = round(v, 3) if isinstance(v, float) and not math.isnan(v) else v
        data.append(d)
    df = pd.DataFrame(data, columns=cols)
    return df.rename(columns={**METRIC_LABELS, **EXTRA_LABELS})


class RagasRunner:
    def __init__(self, retriever, generator, embedder, ecfg: dict, run_dir: str, log=print):
        self.retriever, self.generator, self.embedder = retriever, generator, embedder
        self.ecfg, self.log = ecfg, log
        self.metric_names = [m for m in ecfg.get("metrics", []) if m in METRIC_LABELS]
        self.store = JsonlStore(os.path.join(run_dir, "results.jsonl"), key="id")
        self._metrics = None

    @property
    def metrics(self):
        if self._metrics is None:
            self._metrics = build_metrics(self.metric_names, self.ecfg.get("judge_model", "gpt-5-mini"),
                                          self.embedder, int(self.ecfg.get("judge_max_tokens", 4096)),
                                          self.ecfg.get("judge_reasoning_effort", "low"))
        return self._metrics

    def iter_run(self, samples: list[dict], retrieval: dict, gen: dict, stop: StopFlag | None = None):
        """yield (per-sample DataFrame, summary DataFrame, status str) — 샘플/지표 하나 끝날 때마다"""
        from ragas.dataset_schema import SingleTurnSample

        done = self.store.load_map()
        rows = [done[s["id"]] for s in samples if s["id"] in done]
        if rows:
            self.log(f"[eval] 체크포인트에서 {len(rows)}개 결과 재사용")
        yield rows_df(rows, self.metric_names), summarize(rows, self.metric_names), f"재개: {len(rows)}/{len(samples)}"
        for k, s in enumerate(samples, 1):
            if s["id"] in done:
                continue
            if stop is not None and stop.is_set():
                yield rows_df(rows, self.metric_names), summarize(rows, self.metric_names), "⏹ 중지됨 (저장 완료)"
                return
            t0 = time.time()
            hits = self.retriever.search(s["question"], **retrieval)
            contexts = [h.get("body", h["text"]) for h in hits]
            ans = self.generator.answer(s["question"], hits, **gen)
            row = {"id": s["id"], "question": s["question"], "answer": ans["text"],
                   "ground_truth": s.get("ground_truth", ""), "contexts": contexts,
                   "retrieved_ids": [h["chunk_id"] for h in hits],
                   "evidence_kw_recall": _kw_recall(s.get("evidence_keywords", []), contexts),
                   "gold_hit": (float(s["chunk_id"] in {h["chunk_id"] for h in hits}) if s.get("chunk_id")
                                else float("nan")),
                   "latency_s": round(time.time() - t0, 2), "warnings": ans["warnings"]}
            rows.append(row)
            sample = SingleTurnSample(user_input=s["question"], retrieved_contexts=contexts,
                                      response=ans["text"], reference=s.get("ground_truth", ""))
            for name, metric in self.metrics.items():
                row[name] = _score(metric, sample)
                yield (rows_df(rows, self.metric_names), summarize(rows, self.metric_names),
                       f"{k}/{len(samples)} · {METRIC_LABELS[name]} 계산됨")
            self.store.append(row)  # 샘플 1개 끝날 때마다 저장
            yield rows_df(rows, self.metric_names), summarize(rows, self.metric_names), f"{k}/{len(samples)} ✔ 샘플 완료"
        yield rows_df(rows, self.metric_names), summarize(rows, self.metric_names), f"✅ 완료 {len(rows)}개"


def contexts_text(hits):  # 디버그용
    return format_context(hits)


def retrieval_sweep(retriever, samples: list[dict], weights=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0), top_k: int = 5,
                    fusion: str = "score", candidate_k: int = 50, rrf_k: int = 60) -> pd.DataFrame:
    """LLM 호출 없이 Dense:BM25 비율별 검색 성능 비교 (Gold Hit@k, MRR@k, Evidence KW Recall).
    RAGAS(유료 LLM 호출) 전에 비율을 먼저 고르는 용도."""
    recs = []
    for w in weights:
        hit, rr, kw, n_gold = 0.0, 0.0, [], 0
        for s in samples:
            hits = retriever.search(s["question"], top_k=top_k, dense_weight=w, fusion=fusion,
                                    candidate_k=candidate_k, rrf_k=rrf_k)
            ids = [h["chunk_id"] for h in hits]
            if s.get("chunk_id"):
                n_gold += 1
                if s["chunk_id"] in ids:
                    hit += 1
                    rr += 1 / (ids.index(s["chunk_id"]) + 1)
            v = _kw_recall(s.get("evidence_keywords", []), [h.get("body", h["text"]) for h in hits])
            if not math.isnan(v):
                kw.append(v)
        recs.append({"Dense 비율": w, "BM25 비율": round(1 - w, 2),
                     f"Gold Hit@{top_k}": round(hit / n_gold, 4) if n_gold else None,
                     f"MRR@{top_k}": round(rr / n_gold, 4) if n_gold else None,
                     "Evidence KW Recall": round(sum(kw) / len(kw), 4) if kw else None, "n": len(samples)})
    return pd.DataFrame(recs)


# ==================================================================================================
# 최종 평가 세트 (eval/eval_questions.yaml): 질문별 답변 → 마지막에 RAGAS 일괄 측정
# ==================================================================================================
REF_METRICS = {"context_recall", "factual_correctness", "answer_correctness"}   # reference 필수
NOREF_SWAP = {"context_precision": "context_precision_noref"}    # reference 없으면 대체 지표


def load_eval_set(path_or_items) -> list[dict]:
    """YAML 경로 또는 노트북에서 만든 질문 리스트(dict) → 정규화된 평가 항목"""
    import copy

    import yaml

    if isinstance(path_or_items, (list, tuple)):
        items = copy.deepcopy(list(path_or_items))
    else:
        with open(path_or_items, encoding="utf-8") as f:
            items = yaml.safe_load(f) or []
    seen = set()
    for it in items:
        if not it.get("id") or not it.get("question"):
            raise ValueError(f"질문 항목에 id 와 question 은 필수입니다: {it}")
        if it["id"] in seen:
            raise ValueError(f"id 중복: {it['id']}")
        seen.add(it["id"])
        it.setdefault("set", "기타")
        it.setdefault("category", "")
        it.setdefault("conv", it["id"])  # conv 없으면 단독 대화
    return items


def _fact_norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s or "").lower()


def custom_checks(item: dict, row: dict) -> dict:
    """LLM 없이 계산하는 결정적 지표: 규칙 전처리 / 라우팅 / 핵심 사실 포함"""
    out = {}
    if item.get("expect_blocked") is not None:
        out["rule_ok"] = float(bool(row["blocked"]) == bool(item["expect_blocked"]))
    elif item.get("expect_terms"):
        out["rule_ok"] = float(all(t in row["normalized"] for t in item["expect_terms"]))
    route = row.get("route") or {}
    if item.get("expect_orgs") or item.get("expect_mode"):
        ok = True
        if item.get("expect_orgs"):
            ok &= set(item["expect_orgs"]) <= set(route.get("orgs") or [])
        if item.get("expect_mode"):
            ok &= route.get("mode") == item["expect_mode"]
        out["route_ok"] = float(ok)
    if item.get("must_include") and not row["blocked"]:
        ans = _fact_norm(row["answer"])
        hits = [any(_fact_norm(alt) in ans for alt in str(f).split("|")) for f in item["must_include"]]
        out["key_fact"] = sum(hits) / len(hits)
    return out


def run_eval_set(chat, items: list[dict], store: JsonlStore, stop: StopFlag | None = None):
    """질문을 순서대로 실행(같은 conv 는 대화 유지). 1건마다 저장 → 중단 후 재실행 시 이어서.
    yield (row, 새로 계산했는지)"""
    done = store.load_map()
    cur_conv, conv_rows = None, []
    for it in items:
        if it["conv"] != cur_conv:
            cur_conv, conv_rows = it["conv"], []
            chat.reset()
        if it["id"] in done:
            conv_rows.append(done[it["id"]])
            yield done[it["id"]], False
            continue
        if stop is not None and stop.is_set():
            return
        if conv_rows and not chat.history:  # 대화 중간부터 재개 → 저장된 이전 턴으로 복원
            chat.restore(conv_rows)
        r = chat.ask(it["question"])
        row = {k: it.get(k) for k in ("id", "set", "conv", "qid", "category", "reference", "must_include",
                                      "expect_terms", "expect_orgs", "expect_mode", "expect_blocked")}
        row.update(question=it["question"], normalized=r["normalized"], rule_steps=r["rule_steps"],
                   rule_ms=r["rule_ms"], blocked=r["blocked"], answer=r["answer"], warnings=r["warnings"],
                   route={k: r["route"].get(k) for k in ("mode", "orgs", "doc_ids", "titles")} if r["route"] else {},
                   contexts=[h.get("body", h["text"]) for h in r["hits"]],
                   sources=[f"{h['source'][:30]} " + (f"p.{h['page']}" if h.get("page") else (h.get("section") or "")[:30])
                            + f" {h['kind']}" for h in r["hits"]],
                   retrieval_s=r.get("retrieval_s"), llm_s=r.get("llm_s"), total_s=r.get("total_s"),
                   augmentation=r.get("augmentation"))
        row.update(custom_checks(it, row))
        store.append(row)
        conv_rows.append(row)
        yield row, True


CUSTOM_LABELS = {"rule_ok": "Rule OK", "route_ok": "Route OK", "key_fact": "Key Fact"}


def _custom_cols(rows: list[dict]) -> list[tuple[str, str]]:
    """값이 하나라도 있는 결정적 지표만 표시 (규칙 전처리를 끄면 Rule OK 열은 숨김)"""
    return [(k, v) for k, v in CUSTOM_LABELS.items() if any(isinstance(r.get(k), (int, float)) for r in rows)]


def score_rows_iter(rows: list[dict], metric_names: list[str], ecfg: dict, embedder, store: JsonlStore,
                    stop: StopFlag | None = None):
    """저장된 답변들에 RAGAS 를 '막판에' 일괄 측정. 행/지표 하나 끝날 때마다 yield (표, 요약, 상태).
    reference 없는 질문은 참조 불필요 지표만 계산(context_precision → without-reference 버전)."""
    from ragas.dataset_schema import SingleTurnSample

    names = [m for m in metric_names if m in METRIC_LABELS]
    need = set(names) | {NOREF_SWAP[m] for m in names if m in NOREF_SWAP}
    metrics = None
    done = store.load_map()
    scored = []

    def tables(status):
        df = pd.DataFrame([{"id": r["id"], "set": r["set"], "category": r["category"],
                            "question": r["question"][:40], "ref": "✔" if r.get("reference") else "—",
                            **{METRIC_LABELS[m]: _r(r.get(m)) for m in names},
                            **{v: _r(r.get(k)) for k, v in _custom_cols(rows)}} for r in scored])
        return df, summarize_eval(scored, names), status

    targets = [r for r in rows if not r.get("blocked")]
    for r in rows:
        if r.get("blocked"):
            scored.append(r)
    yield tables(f"대상 {len(targets)}개 (차단 {len(rows) - len(targets)}개는 RAGAS 제외)")
    for k, r in enumerate(targets, 1):
        if r["id"] in done:
            scored.append({**r, **done[r["id"]]})
            continue
        if stop is not None and stop.is_set():
            yield tables("⏹ 중지됨 (저장 완료)")
            return
        if metrics is None:
            metrics = build_metrics(sorted(need), ecfg.get("judge_model", "gpt-5-mini"), embedder,
                                    int(ecfg.get("judge_max_tokens", 4096)), ecfg.get("judge_reasoning_effort", "low"))
        ref = r.get("reference") or ""
        sample = SingleTurnSample(user_input=r["normalized"], retrieved_contexts=r["contexts"] or [""],
                                  response=r["answer"], reference=ref or None)
        res = {"id": r["id"]}
        row = {**r}
        scored.append(row)
        for m in names:
            if m in REF_METRICS and not ref:
                res[m] = float("nan")
                continue
            key = NOREF_SWAP[m] if (m in NOREF_SWAP and not ref) else m
            res[m] = _score(metrics[key], sample)
            row[m] = res[m]
            yield tables(f"{k}/{len(targets)} {r['id']} · {METRIC_LABELS[m]}{'(no-ref)' if key != m else ''}")
        store.append(res)
    yield tables(f"✅ 완료 {len(targets)}개")


def _r(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 3)


def summarize_eval(rows: list[dict], names: list[str]) -> pd.DataFrame:
    """세트별 평균 (+ 전체). RAGAS 지표 + Rule/Route/Key Fact"""
    cols = [(m, METRIC_LABELS[m]) for m in names] + _custom_cols(rows)
    groups = {}
    for r in rows:
        groups.setdefault(r.get("set", "기타"), []).append(r)
    groups["전체"] = rows
    recs = []
    for g, rs in groups.items():
        rec = {"세트": g, "질문 수": len(rs)}
        for k, label in cols:
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float)) and not math.isnan(r[k])]
            rec[label] = round(sum(vals) / len(vals), 3) if vals else None
        recs.append(rec)
    return pd.DataFrame(recs)


# ==================================================================================================
# 골든 데이터셋 + 토큰 0 평가 지표 (LLM 호출 없이 로컬 계산)
# ==================================================================================================
FREE_LABELS = {"key_fact_answer": "Key Fact (답변)", "key_fact_context": "Key Fact (근거)",
               "semantic_sim": "Semantic Sim"}
_FACT_RE = re.compile(
    r"\d{1,2}\s*월\s*\d{1,2}\s*일"                                                                            # 3월 10일 (먼저)
    r"|(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?:만\s*원|억\s*원|천\s*원|원|만|%|퍼센트|년|개월|월|일|시간|초|세)"  # 1,000만원 · 17% · 10년
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"  # March 10
    r"|[A-Z]{2,5}-\d{2,4}", re.IGNORECASE)                                                                   # PER-002


def extract_key_facts(text: str, max_facts: int = 8) -> list[str]:
    """정답 문장에서 숫자·금액·비율·날짜·요구사항 ID 를 뽑아 핵심 사실 목록으로 (LLM 미사용)"""
    out = []
    for m in _FACT_RE.finditer(text or ""):
        f = re.sub(r"\s+", " ", m.group(0)).strip()
        if f and _fact_norm(f) not in {_fact_norm(x) for x in out}:
            out.append(f)
    return out[:max_facts]


def build_golden(questions: list[str], references: dict, key_facts: dict | None = None) -> list[dict]:
    """질문 + 정답 + 핵심 사실 → 골든 데이터셋 레코드 (key_facts 를 직접 주면 그 값 우선)"""
    key_facts = key_facts or {}
    rows = []
    for i, q in enumerate(questions, 1):
        ref = (references or {}).get(q, "") or ""
        kf = key_facts.get(q)
        rows.append({"id": f"Q{i:02d}", "question": q, "reference": ref,
                     "key_facts": list(kf) if kf else extract_key_facts(ref),
                     "key_facts_source": "직접 입력" if kf else ("정답에서 자동 추출" if ref else "-")})
    return rows


def save_golden(rows: list[dict], path: str):
    import json as _json

    from .notebook_checkpoint import atomic_write_bytes

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    atomic_write_bytes(path, "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows).encode("utf-8") + b"\n")


def load_golden(path: str) -> list[dict]:
    import json as _json

    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [_json.loads(ln) for ln in f if ln.strip()]


def _fact_hits(facts: list[str], text: str) -> float | None:
    if not facts:
        return None
    hay = _fact_norm(text)
    alt = lambda f: [_fact_norm(a) for a in str(f).split("|")]  # noqa: E731  'A|B' = 둘 중 하나
    return sum(any(a and a in hay for a in alt(f)) for f in facts) / len(facts)


def free_metrics(answer: str, contexts: list[str], reference: str | None, key_facts: list[str] | None,
                 embedder=None) -> dict:
    """토큰 0 지표 — Key Fact(답변/근거): 핵심 사실 포함 비율 · Semantic Sim: 답변↔정답 임베딩 코사인(로컬 GPU/CPU)"""
    out = {"key_fact_answer": _fact_hits(key_facts or [], answer or ""),
           "key_fact_context": _fact_hits(key_facts or [], " ".join(contexts or [])),
           "semantic_sim": None}
    if reference and answer and embedder is not None:
        import numpy as np

        a, b = np.asarray(embedder.embed_query(answer)), np.asarray(embedder.embed_query(reference))
        out["semantic_sim"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return {FREE_LABELS[k]: (None if v is None else round(v, 4)) for k, v in out.items()}
