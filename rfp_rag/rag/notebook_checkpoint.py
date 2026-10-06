"""체크포인트 + Jupyter 노트북 헬퍼 (checkpoint + notebook 통합)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .hw import gpu_mem


# ====================================================================================================
# 체크포인트   (원래 checkpoint.py)
# ====================================================================================================
# 체크포인트 코어.
#
# 전원이 꺼지거나 GPU OOM 으로 프로세스가 죽어도 이어서 돌릴 수 있도록:
# - 모든 파일 쓰기는 '임시파일 → fsync → os.replace' 원자적 교체 (반쯤 써진 파일이 남지 않음)
# - 항목 단위 진행은 JSONL append + fsync (마지막 줄이 잘려 있으면 무시하고 재개)
# - 단계(stage)별 fingerprint(설정 해시)를 manifest 에 기록 → 설정이 같으면 건너뛰고, 바뀌면 다시 계산
# - SIGINT/SIGTERM 또는 UI '중지' 버튼 → 현재 배치까지 저장 후 안전 종료
def _fsync_dir(path: str):
    try:
        fd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass  # Windows 등


def atomic_write_bytes(path: str, data: bytes):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path)


def atomic_write_json(path: str, obj: Any):
    atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str).encode("utf-8"))


def atomic_save_npy(path: str, arr):
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.npy"
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path)


def read_json(path: str, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def fingerprint(*objs: Any, n: int = 12) -> str:
    h = hashlib.sha1()
    for o in objs:
        h.update(json.dumps(o, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
    return h.hexdigest()[:n]


def file_fingerprint(path: str) -> str:
    st = os.stat(path)
    return fingerprint(os.path.abspath(path), st.st_size, int(st.st_mtime))


class JsonlStore:
    """append-only 진행 기록. 크래시로 마지막 줄이 깨져도 읽을 때 건너뜀."""

    def __init__(self, path: str, key: str = "id"):
        self.path, self.key = path, key
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def load(self) -> list[dict]:
        rows = []
        if not os.path.exists(self.path):
            return rows
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 끊긴 마지막 줄
        return rows

    def load_map(self) -> dict:
        return {r[self.key]: r for r in self.load() if self.key in r}

    def append(self, row: dict):
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def extend(self, rows: Iterable[dict]):
        rows = list(rows)
        if not rows:
            return
        data = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def rewrite(self, rows: Iterable[dict]):
        data = "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
        atomic_write_bytes(self.path, data.encode("utf-8"))


class Manifest:
    """work_dir/manifest.json : stage → {fingerprint, done, stats, updated_at}"""

    def __init__(self, work_dir: str):
        self.path = os.path.join(work_dir, "manifest.json")
        self._lock = threading.Lock()

    def _load(self) -> dict:
        return read_json(self.path, {}) or {}

    def get(self, stage: str) -> dict:
        return self._load().get(stage, {})

    def is_done(self, stage: str, fp: str) -> bool:
        s = self.get(stage)
        return bool(s.get("done")) and s.get("fingerprint") == fp

    def update(self, stage: str, fp: str, done: bool, stats: dict | None = None, **extra):
        with self._lock:
            m = self._load()
            cur = m.get(stage, {}) if m.get(stage, {}).get("fingerprint") == fp else {}
            cur.update(fingerprint=fp, done=done, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
            if stats is not None:
                cur["stats"] = stats
            m[stage] = cur
            atomic_write_json(self.path, m)

    def all(self) -> dict:
        return self._load()


class StopRequested(Exception):
    """중지 요청으로 체크포인트 저장 후 종료"""


class StopFlag:
    """UI 버튼(threading.Event) + OS 시그널 공용 중지 플래그"""

    def __init__(self, event: threading.Event | None = None):
        self.event = event or threading.Event()

    def is_set(self) -> bool:
        return self.event.is_set()

    def set(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise StopRequested("중지 요청 — 체크포인트 저장 완료")

    @contextmanager
    def install_signal_handlers(self):
        """메인 스레드에서만 동작. 첫 Ctrl+C(Jupyter ■ 인터럽트) → 안전 중지, 두 번째 → 즉시 종료.
        진입 시 이전 실행의 중지 플래그를 초기화한다(Jupyter 에서 셀을 다시 실행할 때 바로 멈추지 않도록)."""
        self.event.clear()
        if threading.current_thread() is not threading.main_thread():
            yield self
            return
        prev = {}

        def handler(signum, frame):
            if self.event.is_set():
                raise KeyboardInterrupt
            print(f"\n[checkpoint] 신호 {signum} 수신 → 현재 배치 저장 후 중지합니다 (한 번 더 누르면 즉시 종료)")
            self.event.set()

        for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
            if sig is not None:
                prev[sig] = signal.signal(sig, handler)
        try:
            yield self
        finally:
            for sig, h in prev.items():
                signal.signal(sig, h)


# ====================================================================================================
# 노트북 헬퍼   (원래 notebook.py)
# ====================================================================================================
# Jupyter(.ipynb) 전용 헬퍼 — 01_rag_build(파싱·인덱스·학습) / 02_generate_eval(답변 생성·막판 RAGAS) 공용.
# 
# 노트북에서 ■(Interrupt) 버튼 = 안전 중지: 현재 배치/샘플/스텝까지 저장한 뒤 멈춤.
# 같은 셀을 다시 실행하면 체크포인트부터 이어서 진행.
try:
    from IPython.display import HTML, Markdown, clear_output, display
except ImportError:  # 노트북 밖에서 import 해도 깨지지 않도록
    display = print  # type: ignore
    Markdown = HTML = str  # type: ignore

    def clear_output(wait=False):  # type: ignore
        pass


def _korean_font():
    import importlib

    try:
        importlib.import_module("koreanize_matplotlib")  # import 시 NanumGothic 자동 적용
        return True
    except ImportError:
        return False


# ------------------------------------------------------------------ 실행 래퍼
def run(pipe, fn, *args, **kw):
    """pipe 의 중지 플래그에 Jupyter 인터럽트를 연결해 fn 실행. 중지되면 예외 대신 안내 메시지."""
    t0 = time.time()
    try:
        with pipe.stop.install_signal_handlers():
            out = fn(*args, **kw)
            stopped = pipe.stop.is_set()
        print(f"⏱ {time.time() - t0:.1f}s · {gpu_mem()}")
        if stopped:  # 중지 요청이 왔지만 이 단계는 끝까지 완료된 경우
            print("ℹ️ 중지 요청이 있었지만 이 단계는 이미 완료되어 저장했습니다.")
        return out
    except (StopRequested, KeyboardInterrupt):
        print("\n⏸ 중지됨 — 체크포인트까지 저장했습니다. 이 셀을 다시 실행하면 이어서 진행합니다.")
        return None


# ------------------------------------------------------------------ 표시
def show_docs(docs: list[dict]):
    rows = [{"문서": d["source"][:45], "형식": d["doc_type"], "전체p": d.get("total_pages"), "제외p": d.get("skipped_pages"),
             "블록": len(d["records"]), "표": d["stats"].get("tables", 0),
             "이미지": d["stats"].get("images_found", 0), "캡션": d["stats"].get("images_captioned", 0),
             "원문자": d["stats"].get("raw_chars", 0), "정제자": d["stats"].get("clean_chars", 0)} for d in docs]
    display(pd.DataFrame(rows))


def show_chunks(chunks: list[dict], kind: str | None = None, n: int = 3, contains: str | None = None):
    sel = [c for c in chunks if (kind is None or c["kind"] == kind) and (contains is None or contains in c["text"])]
    for c in sel[:n]:
        display(Markdown(f"**{c['chunk_id']}** · `{c['kind']}` · {len(c['text'])}자\n\n```\n{c['text'][:700]}\n```"))
    print(f"(조건에 맞는 청크 {len(sel)}개 중 {min(n, len(sel))}개 표시)")


def show_growth(pipe, plot: bool = True):
    df = pipe.growth_report()
    display(df.style.format({"문자 수": "{:,.0f}", "이전 대비(×)": "{:.3f}", "원문 대비(×)": "{:.3f}",
                             "크기(MB)": "{:.3f}"}, na_rep="—"))
    if plot:
        import matplotlib.pyplot as plt

        ko = _korean_font()
        d = df.dropna(subset=["문자 수"])
        fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
        labels = d["단계"] if ko else [f"S{i + 1}" for i in range(len(d))]
        axes[0].barh(list(labels)[::-1], list(d["문자 수"])[::-1], color="#4C72B0")
        axes[0].set_title("단계별 문자 수" if ko else "chars per stage")
        s = df.dropna(subset=["크기(MB)"])
        labels2 = s["단계"] if ko else [f"S{i + 1}" for i in range(len(s))]
        axes[1].barh(list(labels2)[::-1], list(s["크기(MB)"])[::-1], color="#DD8452")
        axes[1].set_title("단계별 크기(MB)" if ko else "size (MB)")
        plt.tight_layout()
        plt.show()
    return df


class LiveLoss:
    """파인튜닝 콜백에서 put() → 손실 곡선/평가 지표를 같은 출력 영역에 갱신"""

    def __init__(self, every: int = 1):
        self.rows, self.evals, self.every, self.n = [], [], every, 0
        self.h = display(Markdown("⏳ 학습 시작 대기…"), display_id=True)

    def put(self, r: dict):
        if "loss" in r:
            self.rows.append({"step": r["step"], "loss": float(r["loss"])})
        ev = {k: v for k, v in r.items() if "ndcg@10" in k or "accuracy@5" in k}
        if ev:
            self.evals.append({"step": r["step"], **ev})
        self.n += 1
        if self.n % self.every == 0:
            self._draw(r)

    def _draw(self, r):
        import matplotlib.pyplot as plt

        _korean_font()
        fig, ax = plt.subplots(figsize=(8, 3))
        if self.rows:
            df = pd.DataFrame(self.rows)
            ax.plot(df["step"], df["loss"], marker="o", ms=3, label="train loss")
        if self.evals:
            e = pd.DataFrame(self.evals)
            ax2 = ax.twinx()
            for c in e.columns[1:]:
                ax2.plot(e["step"], e[c], ls="--", label=c.replace("eval_", ""))
            ax2.set_ylim(0, 1)
            ax2.legend(loc="upper right", fontsize=7)
        ax.set_xlabel("step")
        ax.set_title(f"step {r.get('step')}/{r.get('max_steps')}  ·  {gpu_mem()}")
        ax.legend(loc="upper left", fontsize=7)
        plt.tight_layout()
        self.h.update(fig)
        plt.close(fig)


# ------------------------------------------------------------------ 질의
def ask(retriever, generator, question: str, top_k=5, dense_weight=0.6, fusion="score", candidate_k=50,
        rrf_k=60, temperature=None, top_p=None, max_tokens=2048, reasoning_effort="low", history=None,
        show=True) -> dict:
    hits = retriever.search(question, top_k=top_k, dense_weight=dense_weight, fusion=fusion,
                            candidate_k=candidate_k, rrf_k=rrf_k)
    out = generator.answer(question, hits, history=history, temperature=temperature, top_p=top_p,
                           max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    out["hits"] = hits
    if show:
        display(Markdown(f"### 💬 답변\n{out['text'] or '(빈 응답)'}"))
        for w in out["warnings"]:
            display(Markdown(f"> ⚠️ {w}"))
        display(sources_df(hits))
        display(pd.DataFrame([out["augmentation"]]).T.rename(columns={0: "이번 질의 증강량"}))
    return out


def sources_df(hits: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"#": i, "문서": h["source"][:38], "p": h["page"], "유형": h["kind"],
                          "Hybrid": h["score"], "Dense": h["dense"], "BM25": h["bm25"],
                          "D순위": h["dense_rank"], "B순위": h["bm25_rank"], "내용": h["body"][:90]}
                         for i, h in enumerate(hits, 1)])


def ragas_live(pipe, retriever, generator, samples: list[dict], retrieval: dict, gen: dict, split: str = "test"):
    """샘플/지표 하나 끝날 때마다 누적 평균표 + 샘플별 표를 같은 자리에서 갱신. ■ = 안전 중지, 재실행 = 재개."""
    from .evaluate import RagasRunner

    runner = RagasRunner(retriever, generator, retriever.embedder, pipe.cfg["evaluation"],
                         pipe.eval_run_dir(retrieval, gen, split), print)
    h_status = display(Markdown("⏳ 준비 중…"), display_id=True)
    h_sum = display(pd.DataFrame(), display_id=True)
    h_rows = display(pd.DataFrame(), display_id=True)
    df = summ = None

    def fmt(x):
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"

    try:
        with pipe.stop.install_signal_handlers():
            for df, summ, status in runner.iter_run(samples, retrieval, gen, pipe.stop):
                h_status.update(Markdown(f"**{status}** · {gpu_mem()}"))
                s2 = summ.copy()
                for c in ("평균", "최소", "최대"):
                    s2[c] = pd.to_numeric(s2[c], errors="coerce")
                h_sum.update(s2.style.format({"평균": fmt, "최소": fmt, "최대": fmt})
                             .background_gradient(subset=["평균"], cmap="RdYlGn", vmin=0, vmax=1))
                h_rows.update(df.tail(15))
    except KeyboardInterrupt:
        h_status.update(Markdown("⏸ **중지됨** — 완료된 샘플은 저장됨. 셀을 다시 실행하면 이어서 평가합니다."))
    return df, summ


# ------------------------------------------------------------------ 파싱 품질
def show_parse_quality(docs: list[dict]) -> pd.DataFrame:
    """문서별 파싱 품질 지표 — PARSING 하이퍼파라미터를 바꿔 가며 비교"""
    rows = []
    for d in docs:
        st = d["stats"]
        kept = max(d.get("kept_pages") or 0, 1)
        rows.append({"문서": d["source"][:34], "형식": d["doc_type"], "쪽(유지)": d.get("kept_pages"),
                     "서두 제외": d.get("front_matter", ""),
                     "본문자": sum(len(r["text"]) for r in d["records"] if r["kind"] == "text"),
                     "표": st.get("tables", 0), "표문자": st.get("table_chars", 0),
                     "이음줄": st.get("broken_lines_joined", 0), "표중복제거": st.get("table_lines_dropped_from_body", 0),
                     "머리글제거": st.get("repeated_lines_removed", 0), "쪽번호제거": st.get("page_number_lines_removed", 0),
                     "빈쪽": st.get("empty_pages", 0), "깨진문자": st.get("garbled_chars", 0),
                     "병합셀 보정": st.get("merged_cell_rows_fixed", 0), "그림": st.get("images_captioned", 0),
                     "쪽당 글자": round(st.get("clean_chars", 0) / kept) if d.get("kept_pages") else None,
                     "표지(라우팅용)": len(d.get("cover_text", ""))})
    df = pd.DataFrame(rows)
    display(df)
    return df


# ------------------------------------------------------------------ 규칙 전처리 미리보기
def show_rules(preproc, questions: list[str]) -> pd.DataFrame:
    rows = []
    for q in questions:
        r = preproc(q)
        rows.append({"입력": q, "정규화": "" if r.blocked else r.normalized, "적용 규칙": r.summary(),
                     "차단": "⛔" if r.blocked else "", "ms": round(r.elapsed_ms, 3)})
    df = pd.DataFrame(rows)
    display(df.style.set_properties(subset=["입력", "정규화", "적용 규칙"], **{"text-align": "left"}))
    return df


# ------------------------------------------------------------------ 질문 세트 실행 (답변 표시)
def _answer_md(row: dict) -> str:
    r = row.get("route") or {}
    head = f"#### {row['id']} [{row.get('set')}/{row.get('category')}] {row['question']}\n"
    if row["blocked"]:
        return head + f"> ⛔ **차단**(LLM 미호출, {row['rule_ms']}ms): {row['answer']}\n"
    pre = ""
    if row.get("rule_steps"):  # 규칙 전처리를 켠 경우에만 (기본: 꺼짐 → 질문 그대로)
        rule = " · ".join(f"{a}: `{b}`→`{c}`" for a, b, c in row["rule_steps"])
        pre = f"- 🧹 규칙({row['rule_ms']}ms): {rule}  \n- 🔎 정규화: **{row['normalized']}**  \n"
    route = f"{r.get('mode')} · {', '.join(r.get('orgs') or []) or '-'} · {', '.join(r.get('titles') or [])[:120]}"
    checks = " ".join(f"{v}={row[k]:.2f}" for k, v in (("rule_ok", "Rule"), ("route_ok", "Route"), ("key_fact", "KeyFact"))
                      if isinstance(row.get(k), (int, float)))
    src = "<br>".join(row.get("sources") or [])
    warn = "".join(f"\n> ⚠️ {w}" for w in row.get("warnings") or [])
    return (head + pre + f"- 🧭 라우팅: {route}  \n- ⏱ 검색 {row.get('retrieval_s')}s · LLM {row.get('llm_s')}s   {checks}\n\n"
            f"{row['answer']}{warn}\n\n<details><summary>근거 {len(row.get('sources') or [])}개</summary>{src}</details>\n\n---")


def run_eval_questions(pipe, chat, items: list[dict], eval_key=None) -> list[dict]:
    """질문별 답변을 차례로 표시. 1건마다 저장 → ■ 후 재실행하면 이어서 (대화 문맥도 복원)"""
    from .evaluate import run_eval_set

    run_dir = pipe.eval_set_dir(items if eval_key is None else eval_key)
    store = JsonlStore(os.path.join(run_dir, "answers.jsonl"), key="id")
    h = display(Markdown("⏳ 시작…"), display_id=True)
    rows, n_new = [], 0
    try:
        with pipe.stop.install_signal_handlers():
            for row, new in run_eval_set(chat, items, store, pipe.stop):
                rows.append(row)
                n_new += new
                display(Markdown(_answer_md(row)))
                h.update(Markdown(f"**진행 {len(rows)}/{len(items)}** (새로 계산 {n_new}, 저장본 재사용 {len(rows) - n_new}) · "
                                  f"결과: `{run_dir}`"))
    except KeyboardInterrupt:
        pass
    if len(rows) < len(items):
        h.update(Markdown(f"⏸ **중지됨** {len(rows)}/{len(items)} — 셀을 다시 실행하면 이어서 진행합니다."))
    return rows


def ragas_eval_live(pipe, rows: list[dict], embedder, items, metric_names: list[str] | None = None):
    """막판 RAGAS: 행/지표 하나 끝날 때마다 세트별 요약표 + 질문별 표를 같은 자리에서 갱신"""
    from .evaluate import score_rows_iter

    ec = pipe.cfg["evaluation"]
    names = metric_names or ec["metrics"]
    store = JsonlStore(os.path.join(pipe.eval_set_dir(items), f"ragas_{fingerprint(names, ec)}.jsonl"), key="id")
    h_status = display(Markdown("⏳ 준비 중…"), display_id=True)
    h_sum = display(pd.DataFrame(), display_id=True)
    h_rows = display(pd.DataFrame(), display_id=True)
    df = summ = None
    try:
        with pipe.stop.install_signal_handlers():
            for df, summ, status in score_rows_iter(rows, names, ec, embedder, store, pipe.stop):
                h_status.update(Markdown(f"**{status}**"))
                num = summ.select_dtypes("number").columns.drop("질문 수", errors="ignore")
                h_sum.update(summ.style.format("{:.3f}", subset=num, na_rep="—")
                             .background_gradient(subset=num, cmap="RdYlGn", vmin=0, vmax=1))
                h_rows.update(df)
    except KeyboardInterrupt:
        h_status.update(Markdown("⏸ **중지됨** — 셀을 다시 실행하면 남은 질문만 평가합니다."))
    return df, summ


# ------------------------------------------------------------------ 최종 파라미터
def final_params(cfg: dict, extra: dict | None = None, save_to: str | None = None) -> pd.DataFrame:
    """노트북에서 최종 적용된 하이퍼파라미터를 표로 출력(+JSON 저장)"""
    keys = {
        "로더(라이브러리)": ("parsing", ["pdf_text_loader", "pypdf", "pdf_table_loader", "table_settings", "image_parser",
                                     "vlm_model", "hwp_loader", "hwp", "hwp_front_matter"]),
        "파싱": ("parsing", ["skip_first_pages", "hanja_mode", "extract_tables", "remove_page_numbers",
                           "repeated_line_ratio", "drop_table_text_from_body", "table_dedupe_threshold",
                           "join_broken_lines", "join_min_line_len", "fix_bullet_glyphs"]),
        "청크": ("chunking", ["chunk_size", "overlap_ratio", "chunk_overlap", "separators", "section_headers",
                            "table_size_factor", "min_chunk_chars"]),
        "임베딩": ("embedding", ["model", "device", "batch_size", "fp16", "max_seq_length"]),
        "FAISS": ("faiss", ["index_type", "metric", "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"]),
        "BM25": ("bm25", ["tokenizer", "k1", "b"]),
        "검색(하이브리드)": ("retrieval", ["top_k", "dense_weight", "fusion", "candidate_k", "rrf_k"]),
        "규칙 전처리(기본 꺼짐)": ("rules", ["enabled"]),
        "라우팅": ("routing", ["enabled", "auto_extract_orgs", "followup_query"]),
        "LLM": ("llm", ["model", "temperature", "top_p", "max_tokens", "reasoning_effort", "allow_general",
                        "history_turns"]),
        "RAGAS": ("evaluation", ["metrics", "judge_model", "judge_reasoning_effort", "judge_max_tokens", "eval_set"]),
    }
    rows = []
    for group, (sec, ks) in keys.items():
        for k in ks:
            v = (cfg.get(sec) or {}).get(k)
            rows.append({"구분": group, "파라미터": k, "값": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v})
    if sec := cfg.get("chunking"):
        from .chunk_split import resolve_overlap

        rows.append({"구분": "청크", "파라미터": "→ 실제 overlap(글자)", "값": resolve_overlap(sec)})
    for k, v in (extra or {}).items():
        rows.append({"구분": "추가", "파라미터": k, "값": v})
    df = pd.DataFrame(rows)
    if save_to:
        atomic_write_json(save_to, {r["구분"] + "." + r["파라미터"]: r["값"] for r in rows})
    display(df)
    return df


# ------------------------------------------------------------------ 청크 크기 후보 비교 (GPU·API 없이)
def compare_chunk_sizes(docs: list[dict], ccfg: dict, sizes=(400, 700, 1000), items: list[dict] | None = None,
                        preproc=None, top_k: int = 5, bm25_cfg: dict | None = None, bm25_hit: bool = True) -> pd.DataFrame:
    """후보 chunk_size 별 청크 통계(수 초) + (선택) BM25 만으로 본 근거 적중률.
    근거 적중 = 정답(reference)이 있는 질문에서 top-k 청크 안에 must_include 핵심 사실이 모두 들어있는 비율.
    bm25_hit=True 면 후보 크기마다 전체 청크를 형태소 분석(Kiwi, 멀티스레드)하므로 문서 100개 기준 수 분이 걸립니다."""
    import time

    from rank_bm25 import BM25Okapi

    from .chunk_split import chunk_documents, chunk_stats, resolve_overlap
    from .indexing import tokenize_all
    from .text_utils import make_tokenizer

    bm = bm25_cfg or {}
    targets = [it for it in (items or []) if it.get("reference") and it.get("must_include")] if bm25_hit else []
    tok = make_tokenizer(bm.get("tokenizer", "kiwi")) if targets else None
    try:
        h = display(Markdown("⏳ 청크 크기 비교 준비…"), display_id=True)
    except TypeError:          # 노트북 밖(print 대체)
        h = None
    if h is None:
        class _H:
            def update(self, m): print(m)
        h = _H()
    rows, t0 = [], time.time()
    for si, size in enumerate(sizes, 1):
        c = {**ccfg, "chunk_size": size, "chunk_overlap": None}
        chunks = chunk_documents(docs, c)
        rec = {"chunk_size": size, "overlap": resolve_overlap(c), **chunk_stats(chunks)}
        if targets:
            def prog(i, n, _s=size, _si=si):
                h.update(Markdown(f"⏳ [{_si}/{len(sizes)}] chunk_size={_s} · BM25 형태소 분석 {i:,}/{n:,} 청크 · "
                                  f"경과 {time.time() - t0:,.0f}s"))
            toks = tokenize_all(tok, [x["text"] for x in chunks], block=1000, progress=prog)
            bm25 = BM25Okapi(toks, k1=bm.get("k1", 1.5), b=bm.get("b", 0.75))
            hit = 0
            for it in targets:
                q = preproc(it["question"]).normalized if preproc else it["question"]
                sc = bm25.get_scores(tok(q))
                top = [chunks[i]["body"] for i in np.argsort(-sc)[:top_k]]
                hay = re.sub(r"[\s,]", "", " ".join(top))
                hit += all(any(re.sub(r"[\s,]", "", a) in hay for a in str(f).split("|")) for f in it["must_include"])
            rec[f"BM25 근거적중@{top_k}"] = round(hit / len(targets), 3)
            rec["평가 질문 수"] = len(targets)
        rows.append(rec)
    h.update(Markdown(f"✅ 청크 크기 비교 완료 · {time.time() - t0:,.1f}s"
                      + ("" if targets else " (BM25 적중률 생략 — 청크 통계만)")))
    df = pd.DataFrame(rows)
    display(df)
    return df


# ------------------------------------------------------------------ 질문 세트 (노트북에서 직접 편집)
def questions_table(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([{"id": it["id"], "qid": it.get("qid", ""), "세트": it.get("set", ""), "유형": it.get("category", ""),
                        "대화": it.get("conv", ""), "질문": it["question"],
                        "정답": "✔" if it.get("reference") else "—", "핵심사실": ", ".join(it.get("must_include") or []),
                        "기대 동작": it.get("note", ""),
                        "기대 라우팅": f"{it.get('expect_mode', '')} {','.join(it.get('expect_orgs') or [])}".strip()}
                       for it in items])
    display(df)
    return df


def save_questions(items: list[dict], path: str):
    """노트북에서 고친 질문을 YAML 로 저장 (CLI `python main.py evalset` 에서도 같은 질문 사용)"""
    import yaml

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    atomic_write_bytes(path, yaml.safe_dump(items, allow_unicode=True, sort_keys=False, width=200).encode("utf-8"))


def ratio_sweep_questions(chat, items: list[dict], weights=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0), top_k: int = 5,
                          fusion: str = "score", candidate_k: int = 50) -> pd.DataFrame:
    """LLM 호출 없이: (질문 그대로) 라우팅 + 검색만으로 Dense:BM25 비율별 '핵심 사실이 검색 결과에 있는 비율' 비교"""
    targets = [it for it in items if it.get("must_include") and not it.get("expect_blocked")]
    rows = []
    for w in weights:
        hit, n = 0.0, 0
        for it in targets:
            q = chat.pre(it["question"]).normalized if chat.pre is not None else it["question"]
            route = chat.ret.router.route(q, {}) if chat.ret.router else {"mode": "none"}
            hits = chat.ret.search_routed(q, route, top_k=top_k, dense_weight=w, fusion=fusion,
                                          candidate_k=candidate_k)
            hay = re.sub(r"[\s,]", "", " ".join(h.get("body", h["text"]) for h in hits)).lower()
            facts = [any(re.sub(r"[\s,]", "", a).lower() in hay for a in str(f).split("|")) for f in it["must_include"]]
            hit += sum(facts) / len(facts)
            n += 1
        rows.append({"Dense 비율": w, "BM25 비율": round(1 - w, 2), f"핵심사실 검색적중@{top_k}": round(hit / max(n, 1), 3), "n": n})
    df = pd.DataFrame(rows)
    display(df)
    return df
