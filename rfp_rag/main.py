"""CLI — 모든 단계는 중간에 끊겨도 같은 명령을 다시 실행하면 이어서 진행됩니다.

  python main.py build   [--input data/rfp.zip]           # ZIP→파싱→청크→BM25→임베딩→FAISS
  python main.py qa                                        # gpt-5-mini 로 평가/학습 QA 생성
  python main.py split   --set split.ratio=[0.8,0.1,0.1] --set split.seed=7 --set split.stratify=doc_id \
                         --set split.evidence_keywords.min_keywords=3
  python main.py finetune                                  # (선택) 임베딩 파인튜닝, checkpoint 재개
  python main.py eval    --set retrieval.dense_weight=0.7  # RAGAS (진행 표를 콘솔에 실시간 출력)
  python main.py ask "사업 예산은 얼마인가요?" --set retrieval.top_k=8
  python main.py evalset                                   # eval/eval_questions.yaml 답변 + 막판 RAGAS
  python main.py report                                    # 데이터 증가 리포트
  python main.py app                                       # Gradio UI
"""
from __future__ import annotations

import argparse
import os
import sys


from rag.notebook_checkpoint import StopRequested
from rag.config import load_config
from rag.pipeline import Pipeline


def main():
    from rag.credentials import load_tokens

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "qa", "split", "finetune", "eval", "evalset", "ask", "report", "app", "all"])
    ap.add_argument("question", nargs="?")
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", default=None, help="ZIP 또는 폴더 경로")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="설정 덮어쓰기")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    need_key = a.cmd not in ("build", "split", "finetune", "report")  # 인덱스 구축·분할·학습은 OpenAI 키 불필요
    load_tokens(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), ask=sys.stdin.isatty(),
                required=("OPENAI_API_KEY",) if need_key else ())

    cfg = load_config(a.config, a.set)
    if a.input:
        cfg["paths"]["input"] = a.input
    pipe = Pipeline(cfg)

    try:
        if a.cmd in ("build", "all"):
            pipe.build_index()
        if a.cmd in ("qa", "all"):
            pipe.stage_qa()
        if a.cmd in ("split", "all"):
            pipe.stage_split()
        if a.cmd == "finetune" or (a.cmd == "all" and cfg["finetune"].get("enabled")):
            pipe.stage_finetune()
            if a.cmd == "all":
                pipe.build_index()  # 파인튜닝 모델로 재임베딩
        if a.cmd in ("eval", "all"):
            with pipe.stop.install_signal_handlers():  # Ctrl+C → 현재 샘플 저장 후 중지
                run_eval_cli(pipe, cfg)
        if a.cmd == "evalset":
            with pipe.stop.install_signal_handlers():
                run_evalset_cli(pipe, cfg)
        if a.cmd == "ask":
            ask_cli(pipe, cfg, a.question or input("질문: "))
        if a.cmd in ("report", "build", "all"):
            import pandas as pd

            with pd.option_context("display.max_columns", 20, "display.width", 220, "display.max_colwidth", 60):
                print("\n📈 데이터 증가 리포트\n", pipe.growth_report().to_string(index=False))
        if a.cmd == "app":
            from app import build_ui

            build_ui(cfg).queue().launch(server_name="0.0.0.0", server_port=a.port, share=a.share)
    except StopRequested as e:
        print(f"\n⏸  {e or '중지됨'} — 같은 명령을 다시 실행하면 이어서 진행합니다.")
        sys.exit(130)


def _gen_params(cfg):
    lc = cfg["llm"]
    return dict(temperature=lc["temperature"], top_p=lc["top_p"], max_tokens=lc["max_tokens"],
                reasoning_effort=lc.get("reasoning_effort"))


def _ret_params(cfg):
    r = cfg["retrieval"]
    return dict(top_k=r["top_k"], dense_weight=r["dense_weight"], fusion=r["fusion"],
                candidate_k=r["candidate_k"], rrf_k=r["rrf_k"])


def run_evalset_cli(pipe, cfg):
    """eval/eval_questions.yaml 질문별 답변 → 마지막에 RAGAS (중단 후 재실행 시 이어서)"""
    from rag.evaluate import load_eval_set, run_eval_set, score_rows_iter
    from rag.notebook_checkpoint import JsonlStore, fingerprint

    ec = cfg["evaluation"]
    items = load_eval_set(ec["eval_set"])
    chat = pipe.make_chat()
    run_dir = pipe.eval_set_dir(items)
    rows = []
    for row, _new in run_eval_set(chat, items, JsonlStore(os.path.join(run_dir, "answers.jsonl"), key="id"), pipe.stop):
        rows.append(row)
        print(f"\n[{row['id']}] {row['question']}\n  → 정규화: {row['normalized']} | 라우팅: {(row.get('route') or {}).get('mode')}"
              f"\n  {row['answer'][:300]}")
    if len(rows) < len(items):
        print("\n⏸ 중지됨 — 다시 실행하면 이어서 진행합니다.")
        return
    store = JsonlStore(os.path.join(run_dir, f"ragas_{fingerprint(ec['metrics'], ec)}.jsonl"), key="id")
    summary = None
    for _df, summary, status in score_rows_iter(rows, ec["metrics"], ec, chat.ret.embedder, store, pipe.stop):
        print(f"\r{status:<60}", end="", flush=True)
    print("\n\n📊 RAGAS (세트별)\n" + (summary.to_string(index=False) if summary is not None else "-"))


def ask_cli(pipe, cfg, q):
    from rag.llm import Generator

    chat = pipe.make_chat(generator=Generator(cfg["llm"]["model"]))
    out = chat.ask(q)
    print(f"[규칙 {out['rule_ms']}ms] {out['normalized']}  | 라우팅: {out['route'].get('mode')} {out['route'].get('orgs')}")
    print(out["answer"])
    for w in out["warnings"]:
        print("⚠️", w)
    hits = out["hits"]
    print("\n[출처]")
    for i, h in enumerate(hits, 1):
        print(f" [{i}] {h['source']} p.{h['page']} ({h['kind']}) hybrid={h['score']} dense={h['dense']} bm25={h['bm25']}")
    print("\n[증강]", out.get("augmentation"))


def run_eval_cli(pipe, cfg):
    from rag.evaluate import RagasRunner
    from rag.llm import Generator

    ec = cfg["evaluation"]
    samples = pipe.load_split(ec["split"])[: int(ec["max_samples"])]
    ret = pipe.load_retriever()
    rp, gp = _ret_params(cfg), _gen_params(cfg)
    runner = RagasRunner(ret, Generator(cfg["llm"]["model"]), ret.embedder, ec,
                         pipe.eval_run_dir(rp, gp, ec["split"]), pipe.log)
    summary = None
    for _df, summary, status in runner.iter_run(samples, rp, gp, pipe.stop):
        print(f"\r{status:<70}", end="", flush=True)
        if "샘플 완료" in status or status.startswith("재개"):  # 샘플마다 누적 평균 표 갱신
            print("\n" + summary.to_string(index=False))
    print("\n\n📊 RAGAS 최종\n" + (summary.to_string(index=False) if summary is not None else "-"))


if __name__ == "__main__":
    main()
