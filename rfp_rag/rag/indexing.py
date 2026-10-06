"""② 색인 — HuggingFace 임베딩(배치 체크포인트·OOM 복구) · FAISS · BM25(Kiwi) + (선택) 임베딩 파인튜닝.
담당: 청크 · 색인.  (예전 engine.py 의 임베딩·인덱스·파인튜닝 부분)"""
from __future__ import annotations

import glob
import os
import pickle
import queue
import time

import numpy as np
from langchain_core.embeddings import Embeddings

from .notebook_checkpoint import (StopFlag, StopRequested, atomic_save_npy, atomic_write_bytes, atomic_write_json,
                                  read_json)
from .text_utils import make_tokenizer


# ====================================================================================================
# 임베딩 (HuggingFace sentence-transformers, 배치 체크포인트, OOM 복구)   (원래 embedder.py)
# ====================================================================================================
# HuggingFace 임베딩 (sentence-transformers) + 배치 단위 체크포인트 + OOM 자동 복구.
def _is_oom(e: BaseException) -> bool:
    try:
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return "out of memory" in str(e).lower()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


_MODEL_CACHE: dict = {}  # (model, device, fp16, max_len) → SentenceTransformer  (노트북에서 GPU 중복 적재 방지)


class HFEmbedder(Embeddings):
    """LangChain Embeddings 호환 래퍼 (FAISS / RAGAS 에 그대로 전달 가능)."""

    @staticmethod
    def clear_cache():
        """GPU 에 올라간 임베딩 모델 해제 (파인튜닝 직전 / 노트북 정리용)"""
        _MODEL_CACHE.clear()
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def __init__(self, model_name: str, device: str = "auto", max_seq_length: int = 512, fp16: bool = True,
                 normalize: bool = True, query_prefix: str = "", doc_prefix: str = "", batch_size: int = 32):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_seq_length = max_seq_length
        self.fp16 = fp16
        self.normalize = normalize
        self.query_prefix, self.doc_prefix = query_prefix or "", doc_prefix or ""
        self.batch_size = batch_size
        self._model = None

    @classmethod
    def from_config(cls, ecfg: dict, model_override: str | None = None):
        return cls(model_override or ecfg["model"], ecfg.get("device", "auto"), ecfg.get("max_seq_length", 512),
                   ecfg.get("fp16", True), ecfg.get("normalize", True), ecfg.get("query_prefix", ""),
                   ecfg.get("doc_prefix", ""), ecfg.get("batch_size", 32))

    @property
    def model(self):
        if self._model is None:
            key = (self.model_name, self.device, bool(self.fp16), self.max_seq_length)
            if key not in _MODEL_CACHE:
                from sentence_transformers import SentenceTransformer

                m = SentenceTransformer(self.model_name, device=self.device)
                m.max_seq_length = self.max_seq_length
                if self.fp16 and self.device == "cuda":
                    m.half()  # L4: fp16 텐서코어, 메모리 절반
                m.eval()
                _MODEL_CACHE[key] = m
            self._model = _MODEL_CACHE[key]
        return self._model

    def _encode(self, texts: list[str], bs: int) -> np.ndarray:
        return self.model.encode(texts, batch_size=bs, normalize_embeddings=self.normalize,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def _encode_safe(self, texts: list[str]) -> np.ndarray:
        bs = self.batch_size
        while True:
            try:
                return self._encode(texts, bs)
            except Exception as e:  # noqa: BLE001
                if not _is_oom(e) or bs <= 1:
                    raise
                self._free()
                bs = max(1, bs // 2)
                self.batch_size = bs

    # ---- LangChain Embeddings API ----
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode_safe([self.doc_prefix + t for t in texts]).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode_safe([self.query_prefix + text])[0].tolist()

    def _free(self):
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---- 체크포인트 인코딩 ----
    def encode_with_checkpoint(self, texts: list[str], out_dir: str, shard_every: int = 20,
                               stop: StopFlag | None = None, log=print) -> np.ndarray:
        """out_dir/progress.json + shard_XXXXX.npy 로 진행 저장.
        전원이 꺼져도 마지막으로 저장된 샤드 이후부터 다시 인코딩한다."""
        os.makedirs(out_dir, exist_ok=True)
        final = os.path.join(out_dir, "embeddings.npy")
        if os.path.exists(final):
            arr = np.load(final)
            if arr.shape[0] == len(texts):
                log(f"[embed] 완료된 임베딩 재사용 {arr.shape}")
                return arr
        prog_path = os.path.join(out_dir, "progress.json")
        prog = read_json(prog_path, {}) or {}
        shards = [s for s in prog.get("shards", []) if os.path.exists(os.path.join(out_dir, s["file"]))]
        n_done = shards[-1]["end"] if shards else 0
        if prog.get("batch_size"):
            self.batch_size = min(self.batch_size, int(prog["batch_size"]))  # OOM 으로 줄인 배치 유지
        # 매니페스트에 없는 고아 샤드 정리
        listed = {s["file"] for s in shards}
        for f in glob.glob(os.path.join(out_dir, "shard_*.npy")):
            if os.path.basename(f) not in listed:
                os.remove(f)
        if n_done:
            log(f"[embed] 체크포인트에서 재개: {n_done}/{len(texts)} 완료된 상태 (batch={self.batch_size})")

        buf, buf_start, n_batches = [], n_done, 0
        t0 = time.time()

        def flush():
            nonlocal buf, buf_start
            if not buf:
                return
            arr = np.concatenate(buf, 0)
            name = f"shard_{len(shards):05d}.npy"
            atomic_save_npy(os.path.join(out_dir, name), arr)
            shards.append({"file": name, "start": buf_start, "end": buf_start + len(arr)})
            atomic_write_json(prog_path, {"n_total": len(texts), "shards": shards, "batch_size": self.batch_size,
                                          "model": self.model_name, "dim": int(arr.shape[1])})
            buf_start += len(arr)
            buf = []

        i = n_done
        try:
            while i < len(texts):
                if stop is not None and stop.is_set():
                    raise StopRequested
                bs = self.batch_size
                batch = [self.doc_prefix + t for t in texts[i:i + bs]]
                try:
                    vec = self._encode(batch, bs)
                except Exception as e:  # noqa: BLE001
                    if _is_oom(e) and bs > 1:
                        self._free()
                        self.batch_size = max(1, bs // 2)
                        log(f"[embed] GPU OOM → batch_size {bs} → {self.batch_size} 로 줄여 재시도")
                        continue
                    raise
                buf.append(vec)
                i += len(batch)
                n_batches += 1
                if n_batches % shard_every == 0:
                    flush()
                    rate = (i - n_done) / max(time.time() - t0, 1e-6)
                    log(f"[embed] {i}/{len(texts)} ({rate:.1f} chunks/s) — 샤드 저장")
        finally:
            flush()  # 중지/예외 시에도 지금까지 계산한 것 저장
        arr = np.concatenate([np.load(os.path.join(out_dir, s["file"])) for s in shards], 0)
        atomic_save_npy(final, arr)
        for s in shards:
            os.remove(os.path.join(out_dir, s["file"]))
        atomic_write_json(prog_path, {"n_total": len(texts), "shards": [], "done": True, "dim": int(arr.shape[1])})
        return arr


# ====================================================================================================
# 인덱스: FAISS + BM25 + 하이브리드   (원래 index.py)
# ====================================================================================================
# FAISS(Dense) + BM25(Sparse) 인덱스 구축/로드 및 하이브리드 검색 (Dense:BM25 비율 조절).
# ------------------------------------------------------------------ FAISS
def build_faiss(chunks: list[dict], vectors: np.ndarray, embedder, out_dir: str, fcfg: dict | None = None):
    """LangChain FAISS 벡터스토어 구축.
    fcfg (노트북 FAISS_CFG):
      index_type : "flat"(정확 검색, 수만 청크까지 권장) | "hnsw"(근사, 수십만 청크)
      metric     : "ip"(정규화 벡터 → 코사인) | "l2"
      hnsw_m, hnsw_ef_construction, hnsw_ef_search
    """
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from langchain_core.documents import Document

    fcfg = fcfg or {}
    d = int(vectors.shape[1])
    ip = fcfg.get("metric", "ip") == "ip"
    metric = faiss.METRIC_INNER_PRODUCT if ip else faiss.METRIC_L2
    if fcfg.get("index_type", "flat") == "hnsw":
        index = faiss.IndexHNSWFlat(d, int(fcfg.get("hnsw_m", 32)), metric)
        index.hnsw.efConstruction = int(fcfg.get("hnsw_ef_construction", 200))
        index.hnsw.efSearch = int(fcfg.get("hnsw_ef_search", 128))
    else:
        index = faiss.IndexFlatIP(d) if ip else faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    keys = ("chunk_id", "doc_id", "source", "title", "page", "kind", "doc_type", "section")
    docstore = InMemoryDocstore({c["chunk_id"]: Document(page_content=c["text"], metadata={k: c.get(k) for k in keys})
                                 for c in chunks})
    store = FAISS(embedding_function=embedder, index=index, docstore=docstore,
                  index_to_docstore_id={i: c["chunk_id"] for i, c in enumerate(chunks)},
                  distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT if ip else DistanceStrategy.EUCLIDEAN_DISTANCE)
    tmp = out_dir + ".tmp"
    store.save_local(tmp)
    if os.path.exists(out_dir):
        import shutil

        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return store


def load_faiss(out_dir: str, embedder, fcfg: dict | None = None):
    from langchain_community.vectorstores import FAISS

    import faiss
    from langchain_community.vectorstores.utils import DistanceStrategy

    store = FAISS.load_local(out_dir, embedder, allow_dangerous_deserialization=True)
    # load_local 은 distance_strategy 를 기본값(L2)으로 되돌림 → 인덱스의 실제 metric 으로 복원 (안 하면 IP 점수 순위가 뒤집힘)
    store.distance_strategy = (DistanceStrategy.MAX_INNER_PRODUCT if store.index.metric_type == faiss.METRIC_INNER_PRODUCT
                               else DistanceStrategy.EUCLIDEAN_DISTANCE)
    if fcfg and hasattr(store.index, "hnsw"):
        store.index.hnsw.efSearch = int(fcfg.get("hnsw_ef_search", 128))  # 검색 정확도/속도 조절(재구축 불필요)
    return store


# ------------------------------------------------------------------ BM25
def tokenize_all(tok, texts: list[str], block: int = 2000, progress=None) -> list[list[str]]:
    """청크 전체 토큰화 — 토크나이저에 batch(멀티스레드)가 있으면 사용, block 개마다 progress(i, n) 호출"""
    out: list[list[str]] = []
    for i in range(0, len(texts), block):
        part = texts[i:i + block]
        out.extend(tok.batch(part) if hasattr(tok, "batch") else [tok(t) for t in part])
        if progress:
            progress(len(out), len(texts))
    return out


class BM25Index:
    def __init__(self, chunk_ids: list[str], tokens: list[list[str]], k1: float, b: float, tok_kind: str,
                 hf_model: str | None):
        from rank_bm25 import BM25Okapi

        self.chunk_ids, self.k1, self.b = chunk_ids, k1, b
        self.tok_kind, self.hf_model = tok_kind, hf_model
        self.n_tokens = sum(len(t) for t in tokens)
        self.vocab = len({w for t in tokens for w in t})
        self._tokens = tokens
        self.bm25 = BM25Okapi(tokens, k1=k1, b=b)
        self._tok = None

    @property
    def tokenizer(self):
        if self._tok is None:
            self._tok = make_tokenizer(self.tok_kind, self.hf_model)
        return self._tok

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(self.tokenizer(query))
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k] if k else []
        return sorted(((int(i), float(scores[i])) for i in idx if scores[i] > 0), key=lambda x: -x[1])

    def save(self, path: str):
        state = {"chunk_ids": self.chunk_ids, "tokens": self._tokens, "k1": self.k1, "b": self.b,
                 "tok_kind": self.tok_kind, "hf_model": self.hf_model}
        atomic_write_bytes(path, pickle.dumps(state))

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            s = pickle.load(f)
        return cls(s["chunk_ids"], s["tokens"], s["k1"], s["b"], s["tok_kind"], s["hf_model"])

    @classmethod
    def build(cls, chunks: list[dict], bcfg: dict, hf_model: str | None, log=print) -> "BM25Index":
        tok = make_tokenizer(bcfg.get("tokenizer", "kiwi"), hf_model)
        tokens = tokenize_all(tok, [c["text"] for c in chunks],
                              progress=lambda i, n: log(f"[bm25] tokenized {i}/{n}"))
        idx = cls([c["chunk_id"] for c in chunks], tokens, float(bcfg.get("k1", 1.5)), float(bcfg.get("b", 0.75)),
                  tok.name, hf_model)
        idx._tok = tok
        return idx


# ====================================================================================================
# 임베딩 파인튜닝 (HF Trainer checkpoint 재개)   (원래 finetune.py)
# ====================================================================================================
# (선택) 임베딩 모델 파인튜닝 — HF Trainer 체크포인트로 '중간부터' 재개.
#
# - train split 의 (질문, 근거 청크) 쌍으로 CachedMultipleNegativesRankingLoss 학습
#   (GradCache: 큰 배치의 in-batch negative 효과를 작은 GPU 메모리로)
# - save_steps 마다 output_dir/checkpoint-XXXX 저장 (모델+옵티마이저+스케줄러+RNG 상태)
# - 다시 실행하면 마지막 checkpoint 를 자동으로 찾아 resume_from_checkpoint
# - auto_find_batch_size: OOM 나면 배치를 자동으로 절반으로 줄여 재시작
# - valid split 으로 InformationRetrievalEvaluator(Recall@k, MRR, nDCG) 측정
def last_checkpoint(output_dir: str) -> str | None:
    if not os.path.isdir(output_dir):
        return None
    from transformers.trainer_utils import get_last_checkpoint

    try:
        return get_last_checkpoint(output_dir)
    except Exception:  # noqa: BLE001
        return None


def run_finetune(base_model: str, train: list[dict], valid: list[dict], chunks: list[dict], fcfg: dict,
                 ecfg: dict, output_dir: str, stop: StopFlag | None = None, log=print,
                 log_queue: "queue.Queue | None" = None) -> str:
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    final_dir = os.path.join(output_dir, "final")
    if read_json(os.path.join(final_dir, "done.json"), {}).get("done"):
        log(f"[finetune] 이미 완료된 모델 사용: {final_dir}")
        return final_dir

    by_id = {c["chunk_id"]: c for c in chunks}

    def pairs(rows):
        a, p = [], []
        for r in rows:
            c = by_id.get(r.get("chunk_id"))
            pos = c["text"] if c else r.get("evidence_text")
            if pos:
                a.append(ecfg.get("query_prefix", "") + r["question"])
                p.append(ecfg.get("doc_prefix", "") + pos)
        return a, p

    ta, tp = pairs(train)
    if len(ta) < 8:
        raise ValueError(f"학습 쌍이 너무 적습니다({len(ta)}). QA 생성 수(n_per_doc)나 split.ratio 를 늘리세요.")
    train_ds = Dataset.from_dict({"anchor": ta, "positive": tp})

    model = SentenceTransformer(base_model, device="cuda" if torch.cuda.is_available() else None)
    model.max_seq_length = int(ecfg.get("max_seq_length", 512))
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=int(fcfg.get("mini_batch_size", 8)))

    evaluator = None
    if valid:
        q = {r["id"]: ecfg.get("query_prefix", "") + r["question"] for r in valid if r.get("chunk_id") in by_id}
        rel = {r["id"]: {r["chunk_id"]} for r in valid if r.get("chunk_id") in by_id}
        corpus = {c["chunk_id"]: ecfg.get("doc_prefix", "") + c["text"] for c in chunks}
        if q:
            evaluator = InformationRetrievalEvaluator(q, corpus, rel, name="valid", show_progress_bar=False,
                                                      accuracy_at_k=[1, 5, 10], precision_recall_at_k=[5, 10],
                                                      mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10])

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(fcfg.get("epochs", 2)),
        per_device_train_batch_size=int(fcfg.get("batch_size", 32)),
        learning_rate=float(fcfg.get("learning_rate", 2e-5)),
        warmup_ratio=float(fcfg.get("warmup_ratio", 0.1)),
        bf16=bf16, fp16=torch.cuda.is_available() and not bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps" if evaluator else "no",
        eval_steps=int(fcfg.get("eval_steps", 50)),
        save_strategy="steps",
        save_steps=int(fcfg.get("save_steps", 50)),
        save_total_limit=int(fcfg.get("save_total_limit", 3)),
        logging_steps=10,
        seed=int(fcfg.get("seed", 42)),
        data_seed=int(fcfg.get("seed", 42)),
        auto_find_batch_size=bool(fcfg.get("auto_find_batch_size", True)),
        dataloader_num_workers=int(fcfg.get("dataloader_num_workers", 0)),
        gradient_checkpointing=bool(fcfg.get("gradient_checkpointing", False)),
        report_to="none",
    )

    class _Cb(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            msg = f"[finetune] step {state.global_step}/{state.max_steps} {logs}"
            log(msg)
            if log_queue is not None:
                log_queue.put({"step": state.global_step, "max_steps": state.max_steps, **(logs or {})})

        def on_step_end(self, a, state, control, **kw):
            if stop is not None and stop.is_set():
                log("[finetune] 중지 요청 → 체크포인트 저장 후 종료")
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, a, state, control, **kw):
            log(f"[finetune] 💾 checkpoint-{state.global_step} 저장")

    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds, loss=loss,
                                         evaluator=evaluator, callbacks=[_Cb()])
    ckpt = last_checkpoint(output_dir)
    if ckpt:
        log(f"[finetune] ▶ 체크포인트에서 재개: {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    if stop is not None and stop.is_set():
        raise StopRequested("파인튜닝 중지 — 다음 실행 시 마지막 checkpoint 에서 재개됩니다")
    model.save(final_dir)
    metrics = evaluator(model) if evaluator else {}
    atomic_write_json(os.path.join(final_dir, "done.json"), {"done": True, "base": base_model,
                                                              "n_train": len(ta), "metrics": metrics})
    log(f"[finetune] 완료 → {final_dir}  {metrics}")
    del trainer, model, loss
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_dir
