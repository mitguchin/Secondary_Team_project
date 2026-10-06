"""Gradio UI — 인덱스 구축 / QA·분할 / 파인튜닝 / 챗봇 / RAGAS 실시간 평가 / 데이터 증가 리포트.

실행: python main.py app   (또는 python app.py)
"""
from __future__ import annotations

import copy
import json
import os
import queue
import shutil
import threading
import traceback

import gradio as gr
import pandas as pd

from rag.notebook_checkpoint import StopFlag, StopRequested
from rag.config import load_config
from rag.credentials import load_tokens
from rag.pipeline import Pipeline

try:  # .env 의 OPENAI_API_KEY / HF_TOKEN 을 시스템 환경변수보다 우선 적용
    load_tokens(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), ask=False)
except RuntimeError as _e:
    print(f"⚠️ {_e}  (UI 는 뜨지만 답변 생성·QA·RAGAS 는 키가 있어야 동작)")

STATE = {"cfg": None, "stop": StopFlag(), "ret": None, "ret_key": None, "gen": None}
EMB_MODELS = ["nlpai-lab/KURE-v1", "BAAI/bge-m3", "dragonkue/BGE-m3-ko", "intfloat/multilingual-e5-large"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "factual_correctness"]


def _pipe(log=print) -> Pipeline:
    return Pipeline(STATE["cfg"], log=log, stop=STATE["stop"])


def _bg(fn):
    """fn(log) 을 백그라운드 스레드로 실행하며 로그를 실시간 yield. 중지 버튼 → 체크포인트 저장 후 종료."""
    STATE["stop"] = StopFlag()
    q: queue.Queue = queue.Queue()
    logs: list[str] = []
    res: dict = {}

    def target():
        try:
            res["value"] = fn(lambda m: q.put(str(m)))
        except StopRequested:
            q.put("⏸ 중지됨 — 체크포인트 저장 완료. 같은 버튼을 다시 누르면 이어서 진행합니다.")
        except Exception as e:  # noqa: BLE001
            q.put(f"❌ {type(e).__name__}: {e}")
            q.put(traceback.format_exc()[-2000:])

    th = threading.Thread(target=target, daemon=True)
    th.start()
    while th.is_alive() or not q.empty():
        try:
            logs.append(q.get(timeout=0.4))
        except queue.Empty:
            pass
        yield "\n".join(logs[-400:]), res
    yield "\n".join(logs[-400:]), res


def _stop():
    STATE["stop"].set()
    return "⏹ 중지 요청됨 — 현재 배치/샘플을 저장한 뒤 멈춥니다."


def _growth():
    df = _pipe().growth_report()
    plot = df[["단계", "문자 수"]].dropna().copy()
    plot["문자 수"] = plot["문자 수"].astype(float)
    return df, plot


def _retriever():
    p = _pipe()
    key = json.dumps(p.active, sort_keys=True)
    if STATE["ret_key"] != key:
        STATE["ret"] = p.load_retriever()
        STATE["ret_key"] = key
    return STATE["ret"]


def _generator():
    from rag.llm import Generator

    model = STATE["cfg"]["llm"]["model"]
    if STATE["gen"] is None or STATE["gen"].model != model:
        STATE["gen"] = Generator(model, STATE["cfg"]["llm"].get("timeout", 120))
    return STATE["gen"]


# ============================================================== handlers
def run_build(upload, input_path, skip, hanja, tables, caption, vlm, csize, cover, emb_model, bs, tok):
    c = STATE["cfg"]
    if upload is not None:
        dst = os.path.join(c["paths"]["work_dir"], "upload", os.path.basename(upload))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(upload, dst)
        c["paths"]["input"] = dst
    elif input_path:
        c["paths"]["input"] = input_path
    c["parsing"].update(skip_first_pages=int(skip), hanja_mode=hanja, extract_tables=bool(tables),
                        caption_images=bool(caption), vlm_model=vlm)
    c["chunking"].update(chunk_size=int(csize), overlap_ratio=float(cover), chunk_overlap=None)
    c["embedding"].update(model=emb_model, batch_size=int(bs))
    c["bm25"]["tokenizer"] = tok
    for log, _ in _bg(lambda lg: _pipe(lg).build_index()):
        try:
            df, plot = _growth()
        except Exception:  # noqa: BLE001
            df, plot = pd.DataFrame(), pd.DataFrame({"단계": [], "문자 수": []})
        yield log, df, plot


def run_qa(n_per_doc, ext_file):
    c = STATE["cfg"]["qa_generation"]
    c["n_per_doc"] = int(n_per_doc)
    c["external_path"] = ext_file or ""
    for log, _ in _bg(lambda lg: _pipe(lg).stage_qa()):
        yield log


def run_split(ratio, seed, stratify, ek_on, min_kw, max_kw, in_ctx, must, match, excl):
    s = STATE["cfg"]["split"]
    s.update(ratio=[float(x) for x in str(ratio).replace("/", ",").split(",") if x.strip()], seed=int(seed),
             stratify=stratify)
    s["evidence_keywords"].update(
        enabled=bool(ek_on), min_keywords=int(min_kw), max_keywords=int(max_kw), require_in_context=bool(in_ctx),
        must_include=[x.strip() for x in must.split(",") if x.strip()], match=match,
        exclude=[x.strip() for x in excl.split(",") if x.strip()])
    try:
        _, rep = _pipe().stage_split()
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), f"❌ {e}"
    by = rep.get("by_stratum", {})
    strata = sorted({k for v in by.values() for k in v})
    df = pd.DataFrame([{"층(stratum)": k, **{sp: by.get(sp, {}).get(k, 0) for sp in ("train", "valid", "test")}}
                       for k in strata])
    if len(df):
        df.loc[len(df)] = {"층(stratum)": "합계", **rep["counts"]}
    info = (f"**ratio** {rep['ratio']} · **seed** {rep['seed']} · **stratify** {rep['stratify']}  \n"
            f"**Evidence_keywords 필터**: 입력 {rep['input_samples']} → 유지 {rep['evidence_keywords'].get('kept')}  "
            f"`{rep['evidence_keywords']}`" + (f"  \n⚠️ {rep['warnings']}" if rep.get("warnings") else ""))
    return df, info


def run_finetune(epochs, bs, mbs, lr, save_steps):
    f = STATE["cfg"]["finetune"]
    f.update(epochs=float(epochs), batch_size=int(bs), mini_batch_size=int(mbs), learning_rate=float(lr),
             save_steps=int(save_steps), eval_steps=int(save_steps))
    lq: queue.Queue = queue.Queue()
    hist = []
    for log, _ in _bg(lambda lg: _pipe(lg).stage_finetune(lq)):
        while not lq.empty():
            r = lq.get()
            if "loss" in r:
                hist.append({"step": r["step"], "loss": float(r["loss"])})
        yield log, pd.DataFrame(hist or [{"step": 0, "loss": None}])


def chat(msg, history, temperature, top_p, max_tokens, top_k, dense_w, fusion, cand_k, effort):
    history = history or []
    if not msg.strip():
        return history, "", None, None, ""
    try:
        ret = _retriever()
        hits = ret.search(msg, top_k=int(top_k), dense_weight=float(dense_w), fusion=fusion,
                          candidate_k=int(cand_k), rrf_k=STATE["cfg"]["retrieval"]["rrf_k"])
        out = _generator().answer(msg, hits, history=history, temperature=temperature, top_p=top_p,
                                  max_tokens=int(max_tokens), reasoning_effort=effort)
    except Exception as e:  # noqa: BLE001
        return history + [{"role": "user", "content": msg}, {"role": "assistant", "content": f"❌ {e}"}], "", None, None, ""
    src = pd.DataFrame([{"#": i, "문서": h["source"][:40], "p": h["page"], "유형": h["kind"], "Hybrid": h["score"],
                         "Dense": h["dense"], "BM25": h["bm25"], "D순위": h["dense_rank"], "B순위": h["bm25_rank"],
                         "내용": h["body"][:120]} for i, h in enumerate(hits, 1)])
    aug = pd.DataFrame([{"항목": k, "값": v} for k, v in out["augmentation"].items()])
    warn = "\n".join(f"⚠️ {w}" for w in out["warnings"])
    history = history + [{"role": "user", "content": msg}, {"role": "assistant", "content": out["text"] or "(빈 응답)"}]
    return history, "", src, aug, warn


def run_eval(split, n, metrics, judge, top_k, dense_w, fusion, cand_k, temperature, top_p, max_tokens, effort):
    from rag.evaluate import RagasRunner

    ec = STATE["cfg"]["evaluation"]
    ec.update(split=split, max_samples=int(n), metrics=list(metrics), judge_model=judge)
    STATE["stop"] = StopFlag()
    try:
        p = _pipe()
        samples = p.load_split(split)[: int(n)]
        ret = _retriever()
        rp = dict(top_k=int(top_k), dense_weight=float(dense_w), fusion=fusion, candidate_k=int(cand_k),
                  rrf_k=STATE["cfg"]["retrieval"]["rrf_k"])
        gp = dict(temperature=temperature, top_p=top_p, max_tokens=int(max_tokens), reasoning_effort=effort)
        runner = RagasRunner(ret, _generator(), ret.embedder, ec, p.eval_run_dir(rp, gp, split))
        for df, summ, status in runner.iter_run(samples, rp, gp, STATE["stop"]):
            yield df, summ, status
    except Exception as e:  # noqa: BLE001
        yield None, None, f"❌ {type(e).__name__}: {e}"


# ============================================================== UI
def build_ui(cfg: dict | None = None) -> gr.Blocks:
    STATE["cfg"] = copy.deepcopy(cfg or load_config())
    c = STATE["cfg"]
    with gr.Blocks(title="RFP Naive RAG") as demo:
        gr.Markdown("## 📑 RFP Naive RAG — PyPDFLoader · HWP · FAISS + BM25 하이브리드 · gpt-5-mini · RAGAS\n"
                    "모든 단계는 체크포인트가 저장됩니다. 꺼지거나 멈춘 뒤 **같은 버튼을 다시 누르면 이어서** 진행합니다.")

        # ---------------------------------------------------------- 1
        with gr.Tab("① 인덱스 구축"):
            with gr.Row():
                with gr.Column(scale=1):
                    up = gr.File(label="RFP ZIP 업로드", file_types=[".zip"], type="filepath")
                    in_path = gr.Textbox(label="또는 ZIP/폴더 경로", value=c["paths"]["input"])
                    skip = gr.Slider(0, 5, c["parsing"]["skip_first_pages"], step=1, label="앞 N페이지 제외(표지·목차)")
                    hanja = gr.Dropdown(["append", "replace", "none"], value=c["parsing"]["hanja_mode"],
                                        label="한자 처리 (append: 必(필))")
                    tables = gr.Checkbox(c["parsing"]["extract_tables"], label="표 → Markdown 추출")
                    caption = gr.Checkbox(c["parsing"]["caption_images"], label="그래프·도표 이미지 VLM 캡션 (GPU)")
                    vlm = gr.Textbox(c["parsing"]["vlm_model"], label="VLM 모델(HF)")
                with gr.Column(scale=1):
                    csize = gr.Slider(200, 2000, c["chunking"]["chunk_size"], step=50, label="chunk_size")
                    cover = gr.Slider(0, 0.5, c["chunking"].get("overlap_ratio", 0.2), step=0.05, label="overlap_ratio")
                    emb = gr.Dropdown(EMB_MODELS, value=c["embedding"]["model"], allow_custom_value=True,
                                      label="임베딩 모델(HF)")
                    bs = gr.Slider(1, 256, c["embedding"]["batch_size"], step=1,
                                   label="임베딩 batch_size (OOM 시 자동 절반)")
                    tok = gr.Dropdown(["kiwi", "hf", "whitespace"], value=c["bm25"]["tokenizer"],
                                      label="BM25 토크나이저")
                    with gr.Row():
                        b_build = gr.Button("▶ 구축 / 이어서 재개", variant="primary")
                        b_stop1 = gr.Button("⏹ 중지(체크포인트 저장)")
            log1 = gr.Textbox(label="로그", lines=14, max_lines=14, autoscroll=True)
            growth1 = gr.Dataframe(label="📈 단계별 데이터 증가", wrap=True)
            plot1 = gr.BarPlot(x="단계", y="문자 수", title="단계별 문자 수", height=280)
            b_build.click(run_build, [up, in_path, skip, hanja, tables, caption, vlm, csize, cover, emb, bs, tok],
                          [log1, growth1, plot1])
            b_stop1.click(_stop, None, log1)

        # ---------------------------------------------------------- 2
        with gr.Tab("② QA 생성 · 데이터 분할"):
            with gr.Row():
                n_qa = gr.Slider(1, 100, c["qa_generation"]["n_per_doc"], step=1, label="문서당 QA 수 (gpt-5-mini 생성)")
                ext = gr.Textbox(c["qa_generation"]["external_path"], label="직접 만든 평가셋 경로(csv/jsonl, 선택)")
                b_qa = gr.Button("▶ QA 생성 / 재개", variant="primary")
                b_stop2 = gr.Button("⏹ 중지")
            log2 = gr.Textbox(label="로그", lines=6)
            b_qa.click(run_qa, [n_qa, ext], log2)
            b_stop2.click(_stop, None, log2)
            gr.Markdown("### 분할 설정")
            sc = c["split"]
            ek = sc["evidence_keywords"]
            with gr.Row():
                ratio = gr.Textbox(",".join(str(x) for x in sc["ratio"]), label="Split Ratio (train,valid,test)")
                seed = gr.Number(sc["seed"], precision=0, label="Seed")
                strat = gr.Dropdown(["none", "doc_id", "doc_type", "question_type", "chunk_kind"],
                                    value=sc["stratify"], label="Stratify")
            with gr.Row():
                ek_on = gr.Checkbox(ek["enabled"], label="Evidence_keywords 필터 사용")
                min_kw = gr.Slider(0, 10, ek["min_keywords"], step=1, label="최소 키워드 수")
                max_kw = gr.Slider(1, 20, ek["max_keywords"], step=1, label="최대 키워드 수")
                in_ctx = gr.Checkbox(ek["require_in_context"], label="근거 청크에 실제 존재하는 키워드만")
            with gr.Row():
                must = gr.Textbox(",".join(ek["must_include"]), label="must_include (쉼표 구분)")
                match = gr.Radio(["any", "all"], value=ek["match"], label="must_include 매칭")
                excl = gr.Textbox(",".join(ek["exclude"]), label="exclude (쉼표 구분)")
            b_split = gr.Button("▶ 분할 실행", variant="primary")
            split_info = gr.Markdown()
            split_df = gr.Dataframe(label="분할 결과 (층별 개수)")
            b_split.click(run_split, [ratio, seed, strat, ek_on, min_kw, max_kw, in_ctx, must, match, excl],
                          [split_df, split_info])

        # ---------------------------------------------------------- 3
        with gr.Tab("③ 임베딩 파인튜닝(선택)"):
            fc = c["finetune"]
            gr.Markdown("train split 의 (질문, 근거청크)로 임베딩 모델을 학습합니다. `save_steps` 마다 checkpoint 가 저장되고, "
                        "**컴퓨터가 꺼져도 다시 누르면 마지막 checkpoint 부터 재개**합니다. "
                        "완료 후 `config.yaml` 의 `finetune.enabled: true` 로 두고 ①을 다시 실행하면 재임베딩됩니다.")
            with gr.Row():
                ep = gr.Number(fc["epochs"], label="epochs")
                fbs = gr.Number(fc["batch_size"], precision=0, label="batch_size")
                mbs = gr.Number(fc["mini_batch_size"], precision=0, label="mini_batch_size (GPU 메모리)")
                lr = gr.Number(fc["learning_rate"], label="learning_rate")
                ss = gr.Number(fc["save_steps"], precision=0, label="save_steps (checkpoint 간격)")
            with gr.Row():
                b_ft = gr.Button("▶ 학습 / checkpoint 에서 재개", variant="primary")
                b_stop3 = gr.Button("⏹ 중지(checkpoint 저장)")
            log3 = gr.Textbox(label="로그", lines=12, autoscroll=True)
            loss_plot = gr.LinePlot(x="step", y="loss", title="train loss", height=260)
            b_ft.click(run_finetune, [ep, fbs, mbs, lr, ss], [log3, loss_plot])
            b_stop3.click(_stop, None, log3)

        # ---------------------------------------------------------- 4
        lc, rc = c["llm"], c["retrieval"]
        with gr.Tab("④ 챗봇"):
            with gr.Row():
                with gr.Column(scale=3):
                    try:
                        bot = gr.Chatbot(type="messages", height=480, label=lc["model"])
                    except TypeError:  # Gradio 6: messages 형식이 기본
                        bot = gr.Chatbot(height=480, label=lc["model"])
                    q = gr.Textbox(placeholder="예) 고려대 차세대 포털 사업의 사업기간과 예산은?", label="질문")
                    warn = gr.Markdown()
                with gr.Column(scale=1):
                    temperature = gr.Slider(0, 2, lc["temperature"], step=0.05, label="temperature")
                    top_p = gr.Slider(0, 1, lc["top_p"], step=0.05, label="top_p")
                    max_tokens = gr.Slider(256, 16000, lc["max_tokens"], step=128, label="max tokens (reasoning 포함)")
                    effort = gr.Dropdown(["minimal", "low", "medium", "high"], value=lc["reasoning_effort"],
                                         label="reasoning effort (gpt-5)")
                    top_k = gr.Slider(1, 20, rc["top_k"], step=1, label="top-k (검색 청크 수)")
                    dense_w = gr.Slider(0, 1, rc["dense_weight"], step=0.05, label="Dense 비율 (BM25 = 1 − 값)")
                    ratio_md = gr.Markdown(f"**Dense {rc['dense_weight']:.0%} : BM25 {1 - rc['dense_weight']:.0%}**")
                    fusion = gr.Radio(["score", "rrf"], value=rc["fusion"], label="융합 방식")
                    cand_k = gr.Slider(10, 200, rc["candidate_k"], step=10, label="후보 수(candidate_k)")
                    gr.Markdown("<small>gpt-5-mini 는 API 가 temperature/top_p 를 기본값(1)만 허용합니다. "
                                "값을 바꾸면 자동으로 제외하고 경고를 표시합니다.</small>")
            dense_w.change(lambda w: f"**Dense {w:.0%} : BM25 {1 - w:.0%}**", dense_w, ratio_md)
            src = gr.Dataframe(label="🔎 검색 결과 (Hybrid / Dense / BM25 점수)", wrap=True)
            aug = gr.Dataframe(label="🧩 이번 질의의 데이터 증강량 (질문 → 프롬프트)")
            q.submit(chat, [q, bot, temperature, top_p, max_tokens, top_k, dense_w, fusion, cand_k, effort],
                     [bot, q, src, aug, warn])

        # ---------------------------------------------------------- 5
        with gr.Tab("⑤ RAGAS 평가(실시간)"):
            ev = c["evaluation"]
            with gr.Row():
                e_split = gr.Dropdown(["test", "valid", "train"], value=ev["split"], label="평가 split")
                e_n = gr.Slider(1, 300, ev["max_samples"], step=1, label="최대 샘플 수")
                e_metrics = gr.CheckboxGroup(METRICS, value=ev["metrics"], label="RAGAS 지표")
                judge = gr.Textbox(ev["judge_model"], label="심판 LLM")
            gr.Markdown("검색·생성 파라미터는 **④ 챗봇 탭의 슬라이더 값**을 그대로 사용합니다. "
                        "샘플/지표 하나가 끝날 때마다 표가 갱신되고, 결과는 샘플 단위로 저장되어 중단 후 재개됩니다.")
            with gr.Row():
                b_ev = gr.Button("▶ 평가 시작 / 재개", variant="primary")
                b_stop5 = gr.Button("⏹ 중지")
            ev_status = gr.Markdown()
            ev_summary = gr.Dataframe(label="📊 누적 평균 (실시간)")
            ev_rows = gr.Dataframe(label="샘플별 점수 (실시간)", wrap=True)
            b_ev.click(run_eval, [e_split, e_n, e_metrics, judge, top_k, dense_w, fusion, cand_k, temperature, top_p,
                                  max_tokens, effort], [ev_rows, ev_summary, ev_status])
            b_stop5.click(_stop, None, ev_status)

        # ---------------------------------------------------------- 6
        with gr.Tab("⑥ 데이터 증가 리포트"):
            b_rep = gr.Button("🔄 새로고침")
            rep_df = gr.Dataframe(wrap=True)
            rep_plot = gr.BarPlot(x="단계", y="문자 수", height=280)
            with gr.Accordion("manifest.json (단계별 fingerprint · 체크포인트 상태)", open=False):
                man = gr.Code(language="json")
            b_rep.click(lambda: (*_growth(), _pipe().growth_json()), None, [rep_df, rep_plot, man])
            demo.load(lambda: (*_growth(), _pipe().growth_json()), None, [rep_df, rep_plot, man])
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="0.0.0.0")
