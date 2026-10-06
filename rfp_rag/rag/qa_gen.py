"""평가/학습용 QA 데이터셋: gpt-5-mini 로 청크별 (질문, 정답, 근거 키워드) 생성 — 청크 단위 체크포인트.
직접 만든 평가셋(csv/jsonl)을 쓰려면 qa_generation.external_path 지정."""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from .notebook_checkpoint import JsonlStore, StopFlag, StopRequested
from .text_utils import compact

QA_PROMPT = """다음은 공공기관 제안요청서(RFP)의 일부입니다. 이 내용만으로 답할 수 있는 실무형 질문 1개를 만드세요.
- 질문에는 어느 사업인지 알 수 있게 사업명/기관명을 자연스럽게 포함
- 정답은 원문 근거로 1~3문장
- evidence_keywords: 정답의 근거가 되는 핵심 키워드 3~6개 (원문에 '그대로' 등장하는 단어/수치)
- question_type: 사실형 | 수치형 | 절차형 | 자격요건형 | 표해석형 중 하나
JSON 으로만 출력: {{"question": "...", "answer": "...", "evidence_keywords": ["..."], "question_type": "..."}}

[문서] {title} (p.{page}, {kind})
[내용]
{body}"""


def _pick_chunks(chunks: list[dict], n_per_doc: int, min_chars: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        if len(c["body"]) >= min_chars and c["kind"] in ("text", "table", "image_caption"):
            by_doc.setdefault(c["doc_id"], []).append(c)
    picked = []
    for doc_id in sorted(by_doc):
        cs = by_doc[doc_id][:]
        rng.shuffle(cs)
        picked += cs[:n_per_doc]
    return picked


def generate_qa(chunks: list[dict], qcfg: dict, llm_model: str, store_path: str, stop: StopFlag | None = None,
                log=print) -> list[dict]:
    from .llm import Generator

    store = JsonlStore(store_path, key="chunk_id")
    done = {k: v for k, v in store.load_map().items() if not v.get("error")}  # 실패 건은 재시도
    todo = [c for c in _pick_chunks(chunks, int(qcfg["n_per_doc"]), int(qcfg["min_chunk_chars"]),
                                    int(qcfg.get("seed", 42))) if c["chunk_id"] not in done]
    log(f"[qa] 캐시 {len(done)}개 재사용, 신규 생성 {len(todo)}개")
    gen = Generator(llm_model)

    def work(c):
        if stop is not None and stop.is_set():
            return None
        prompt = QA_PROMPT.format(title=c["title"], page=c["page"], kind=c["kind"], body=c["body"][:3000])
        out = gen.chat([{"role": "user", "content": prompt}], max_tokens=int(qcfg.get("max_tokens", 3000)),
                       reasoning_effort=qcfg.get("reasoning_effort", "low"), json_mode=True)
        try:
            j = json.loads(out["text"])
        except json.JSONDecodeError:
            return {"chunk_id": c["chunk_id"], "error": "json"}
        kws = [k for k in j.get("evidence_keywords", []) if isinstance(k, str) and k.strip()]
        return {"id": f"qa_{c['chunk_id']}", "chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                "doc_type": c["doc_type"], "chunk_kind": c["kind"], "question": j.get("question", "").strip(),
                "ground_truth": j.get("answer", "").strip(), "evidence_keywords": kws,
                "question_type": j.get("question_type", "기타"), "evidence_text": c["body"]}

    n = 0
    with ThreadPoolExecutor(max_workers=int(qcfg.get("max_workers", 4))) as ex:
        futs = [ex.submit(work, c) for c in todo]
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as e:  # noqa: BLE001
                log(f"[qa] 실패: {e}")
                continue
            if row is None:
                continue
            store.append(row)  # 1건마다 fsync → 꺼져도 다음 실행 시 이어서
            n += 1
            if n % 10 == 0:
                log(f"[qa] {n}/{len(todo)} 생성")
    if stop is not None and stop.is_set():
        raise StopRequested
    rows = [r for r in store.load_map().values() if r.get("question") and not r.get("error")]
    return rows


def load_external_qa(path: str, chunks: list[dict]) -> list[dict]:
    """csv/jsonl: question, ground_truth(또는 answer), evidence_keywords(리스트 또는 ',' 구분), [doc_id]"""
    import pandas as pd

    df = pd.read_json(path, lines=True) if path.endswith((".jsonl", ".json")) else pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        kws = r.get("evidence_keywords", [])
        if isinstance(kws, str):
            kws = [k.strip() for k in kws.replace(";", ",").split(",") if k.strip()]
        elif not isinstance(kws, list):
            kws = []
        gt = r.get("ground_truth", r.get("answer", ""))
        rows.append({"id": f"ext_{i:05d}", "question": str(r["question"]), "ground_truth": str(gt),
                     "evidence_keywords": kws, "doc_id": str(r.get("doc_id", "unknown")),
                     "doc_type": str(r.get("doc_type", "unknown")),
                     "question_type": str(r.get("question_type", "기타")), "chunk_kind": "unknown",
                     "chunk_id": r.get("chunk_id"), "evidence_text": ""})
    # 근거 텍스트가 없으면 키워드를 가장 많이 포함한 청크를 근거로 추정
    if chunks:
        comp = [(c, compact(c["body"])) for c in chunks]
        for row in rows:
            if row["chunk_id"] or not row["evidence_keywords"]:
                continue
            ks = [compact(k) for k in row["evidence_keywords"]]
            best = max(comp, key=lambda x: sum(k in x[1] for k in ks))
            row["evidence_text"] = best[0]["body"]
            if row["doc_id"] == "unknown":
                row["doc_id"] = best[0]["doc_id"]
    return rows


def qa_path(work_dir: str, fp: str) -> str:
    return os.path.join(work_dir, "qa", f"qa_{fp}.jsonl")
