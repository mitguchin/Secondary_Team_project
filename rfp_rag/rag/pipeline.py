"""단계별 파이프라인 + 체크포인트 오케스트레이션 (CLI / Gradio 공용)."""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from .notebook_checkpoint import (JsonlStore, Manifest, StopFlag, StopRequested, atomic_write_json, fingerprint,
                         read_json)
from .chunk_split import chunk_documents
from .indexing import HFEmbedder
from .parsing import make_doc_id, parse_file


SUPPORTED = (".pdf", ".hwp", ".hwpx")


def _zip_name(info) -> str:
    """ZIP 안 한국어 파일명 보정: UTF-8 플래그가 없으면 cp949 로 해석"""
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    raw = name.encode("cp437", errors="ignore")
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


def _safe_name(name: str, max_bytes: int = 200) -> str:
    """ZIP 속 파일명 정리.
    ① 맥(macOS)에서 만든 ZIP 은 한글이 자모 분리형(NFD)이라 '국' 한 글자가 9바이트 → 리눅스 파일명 한도(255바이트) 초과
       (Errno 36 File name too long). 윈도우는 글자 수 기준이라 문제가 안 됨 → 완성형(NFC)으로 변환
    ② 그래도 길면 확장자는 살리고 앞부분만 max_bytes 까지 자름"""
    import unicodedata

    name = unicodedata.normalize("NFC", name)
    if len(name.encode("utf-8")) <= max_bytes:
        return name
    stem, ext = os.path.splitext(name)
    b = stem.encode("utf-8")[: max_bytes - len(ext.encode("utf-8"))]
    return b.decode("utf-8", "ignore").rstrip() + ext


def collect_inputs(src: str, out_dir: str, log=print) -> list[str]:
    """ZIP(중첩 ZIP 포함) 또는 폴더 → 문서 경로 목록. 이미 추출된 파일은 재사용(재개)."""
    import re as _re
    import shutil
    import zipfile

    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(src):
        files = []
        for root, _, names in os.walk(src):
            for n in sorted(names):
                p = os.path.join(root, n)
                if n.lower().endswith(SUPPORTED):
                    files.append(p)
                elif n.lower().endswith(".zip"):
                    files += collect_inputs(p, os.path.join(out_dir, n[:-4]), log)
        return sorted(files)
    if src.lower().endswith(SUPPORTED):
        return [src]
    files = []
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            name = _zip_name(info)
            base = _safe_name(_re.sub(r'[<>:"|?*\x00-\x1f]', "_", name.replace("\\", "/").split("/")[-1]).strip())
            if info.is_dir() or not base or "__MACOSX" in name or base.startswith("._"):
                continue
            dst = os.path.join(out_dir, base)  # 경로 성분 제거 → zip-slip 방지
            if base.lower().endswith(SUPPORTED + (".zip",)):
                if not (os.path.exists(dst) and os.path.getsize(dst) == info.file_size):
                    with zf.open(info) as r, open(dst + ".part", "wb") as w:
                        shutil.copyfileobj(r, w)
                    os.replace(dst + ".part", dst)
                files += (collect_inputs(dst, os.path.join(out_dir, base[:-4]), log) if base.lower().endswith(".zip")
                          else [dst])
            else:
                log(f"[warn] 미지원 형식 건너뜀: {base}")
    return sorted(files)


def _dir_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)


BUILD_SECTIONS = ("paths", "parsing", "chunking", "embedding", "faiss", "bm25", "finetune", "split")  # 검색·LLM·라우팅은 02 에서 조절
BUILD_CONFIG = "build_config.json"


class Pipeline:
    def __init__(self, cfg: dict, log=print, stop: StopFlag | None = None, generate_only: bool = False):
        self.cfg = cfg
        self.wd = cfg["paths"]["work_dir"]
        os.makedirs(self.wd, exist_ok=True)
        self.m = Manifest(self.wd)
        self.log = log
        self.stop = stop or StopFlag()
        self._timers = {}
        self.generate_only = generate_only  # True: 02 평가 노트북 — 파싱·임베딩·학습 단계 호출 시 오류

    def _guard(self, stage: str):
        if self.generate_only:
            raise RuntimeError(f"generate_only 모드에서는 '{stage}' 단계를 실행하지 않습니다 "
                               "(학습·인덱스 구축은 01_rag_build.ipynb 에서).")

    # ------------------------------------------------------------ 학습 결과 저장 / 불러오기 (노트북 01 → 02)
    def save_build_config(self, extra: dict | None = None) -> str:
        """01 노트북 끝: 인덱스를 만든 설정 + active.json(인덱스 경로)을 저장 → 02 노트북이 그대로 사용"""
        import importlib.metadata as im

        def ver(n):
            try:
                return im.version(n)
            except im.PackageNotFoundError:
                return None

        a = self.active
        missing = [k for k in ("chunks_path", "faiss_dir", "bm25_path") if not a.get(k)]
        if missing:
            raise RuntimeError(f"인덱스가 완성되지 않았습니다(누락: {missing}). 2-1a·2-1b 셀을 먼저 실행하세요.")
        data = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "active": a,
                "cfg": {k: self.cfg.get(k) for k in BUILD_SECTIONS},
                "stats": {k: (self.m.get(k).get("stats") or {}) for k in ("extract", "parse", "chunk", "embed", "faiss", "bm25")},
                "versions": {n: ver(n) for n in ("langchain-core", "langchain-community", "langchain-hwp-hwpx-loader",
                                                 "hwp-hwpx-parser", "pypdf", "pymupdf", "sentence-transformers",
                                                 "faiss-cpu", "torch", "ragas")},
                **(extra or {})}
        path = self.p(BUILD_CONFIG)
        atomic_write_json(path, data)
        return path

    @classmethod
    def from_build(cls, base_cfg: dict, work_dir: str = "artifacts", log=print) -> "Pipeline":
        """02 노트북: 01 에서 저장한 설정·인덱스를 불러와 '생성 전용' 파이프라인 생성 (파싱·임베딩·학습 없음)"""
        path = os.path.join(work_dir, BUILD_CONFIG)
        data = read_json(path)
        if not data:
            raise FileNotFoundError(f"{path} 가 없습니다 → 먼저 01_rag_build.ipynb 를 끝까지 실행하세요 (마지막 셀이 저장).")
        cfg = dict(base_cfg)
        for k, v in (data.get("cfg") or {}).items():
            if isinstance(v, dict):
                cfg[k] = {**(cfg.get(k) or {}), **v}
        cfg["paths"] = {**cfg.get("paths", {}), "work_dir": work_dir}
        pipe = cls(cfg, log=log, generate_only=True)
        a = pipe.active
        bad = [f"{k}={a.get(k)}" for k in ("chunks_path", "faiss_dir", "bm25_path")
               if not a.get(k) or not os.path.exists(a[k])]
        if bad:
            raise FileNotFoundError("인덱스 파일이 없습니다: " + ", ".join(bad) + " → 01_rag_build.ipynb 를 다시 실행하세요.")
        pipe.build_info = data
        return pipe

    # ------------------------------------------------------------ helpers
    def p(self, *a) -> str:
        return os.path.join(self.wd, *a)

    def _t(self, stage):
        self._timers[stage] = time.time()

    def _el(self, stage):
        return round(time.time() - self._timers.get(stage, time.time()), 1)

    @property
    def active(self) -> dict:
        return read_json(self.p("active.json"), {}) or {}

    def _set_active(self, **kw):
        a = self.active
        a.update(kw)
        atomic_write_json(self.p("active.json"), a)

    # ------------------------------------------------------------ 1. extract
    def stage_extract(self) -> list[str]:
        self._guard("stage_extract")
        src = self.cfg["paths"]["input"]
        if not os.path.exists(src):
            raise FileNotFoundError(f"입력 경로가 없습니다: {src}")
        files = collect_inputs(src, self.p("raw"), self.log)
        stats = {"files": len(files), "bytes": sum(os.path.getsize(f) for f in files),
                 "input_bytes": _dir_bytes(src), "pdf": sum(f.lower().endswith(".pdf") for f in files),
                 "hwp": sum(f.lower().endswith((".hwp", ".hwpx")) for f in files)}
        fp = fingerprint(sorted((os.path.basename(f), os.path.getsize(f)) for f in files))
        self.m.update("extract", fp, True, stats)
        self.log(f"[extract] 문서 {len(files)}개 (PDF {stats['pdf']} / HWP {stats['hwp']})")
        return files

    # ------------------------------------------------------------ 2. parse (파일 단위 체크포인트)
    def fp_parse(self) -> str:
        pc = dict(self.cfg["parsing"])
        if not (pc.get("caption_images") or pc.get("image_parser")):  # 캡션을 끈 경우 VLM 설정은 결과에 영향 없음
            for k in ("vlm_model", "vlm_max_new_tokens", "vlm_max_pixels"):
                pc.pop(k, None)
        return fingerprint(pc, "parse-v3-hwploader")

    def stage_parse(self, files: list[str]) -> list[dict]:
        self._guard("stage_parse")
        fp = fingerprint(self.m.get("extract").get("fingerprint"), self.fp_parse())
        out_dir = self.p("parsed", self.fp_parse())
        os.makedirs(out_dir, exist_ok=True)
        from .vision import make_image_parser

        pc = self.cfg["parsing"]
        kind = pc.get("image_parser") or ("hf-vlm" if pc.get("caption_images") else None)
        captioner = make_image_parser(kind, pc, self.p("captions.jsonl"), self.log)  # LangChain images_parser
        docs = []
        for i, f in enumerate(files, 1):
            cache = os.path.join(out_dir, make_doc_id(f) + ".json")
            d = read_json(cache)
            if d is None:
                self.stop.check()
                t0 = time.time()
                try:
                    d = parse_file(f, self.cfg["parsing"], captioner, self.log,
                                   images_dir=self.p("hwp_images", make_doc_id(f)) if captioner is not None else None)
                except Exception as e:  # noqa: BLE001
                    self.log(f"[parse] ❌ {os.path.basename(f)}: {e}")
                    continue
                atomic_write_json(cache, d)  # 파일 1개 끝날 때마다 저장
                pages = f"{d['total_pages']}p" if d.get("total_pages") else f"표지·목차: {d.get('front_matter')}"
                self.log(f"[parse] ({i}/{len(files)}) {d['source']}: {pages} → "
                         f"{len(d['records'])}블록 ({time.time() - t0:.1f}s)")
            else:
                self.log(f"[parse] ({i}/{len(files)}) 캐시 사용: {d['source']}")
            docs.append(d)
        if captioner is not None and hasattr(captioner, "close"):
            captioner.close()
        agg = {}
        for d in docs:
            for k in ("total_pages", "skipped_pages", "kept_pages"):
                agg[k] = agg.get(k, 0) + (d.get(k) or 0)  # HWP 는 쪽 정보 없음(None)
            for k, v in d["stats"].items():
                agg[k] = agg.get(k, 0) + v
        agg["records"] = sum(len(d["records"]) for d in docs)
        agg["docs"] = len(docs)
        agg["hwp_docs"] = sum(d["doc_type"] in ("hwp", "hwpx") for d in docs)
        self.m.update("parse", fp, True, agg)
        # 기관 라우팅용 문서 메타(표지 텍스트 포함) 저장
        meta = [{k: d.get(k, "") for k in ("doc_id", "title", "source", "doc_type", "cover_text", "head_text")}
                for d in docs]
        atomic_write_json(self.p("docs_meta.json"), meta)
        self._set_active(docs_meta_path=self.p("docs_meta.json"))
        return docs

    # ------------------------------------------------------------ 3. chunk
    def fp_chunk(self) -> str:
        return fingerprint(self.m.get("parse").get("fingerprint"), self.cfg["chunking"])

    def stage_chunk(self, docs: list[dict]) -> list[dict]:
        self._guard("stage_chunk")
        fp = self.fp_chunk()
        path = self.p("chunks", f"chunks_{fp}.jsonl")
        if self.m.is_done("chunk", fp) and os.path.exists(path):
            chunks = JsonlStore(path).load()
            self.log(f"[chunk] 캐시 사용: {len(chunks)}개")
            return chunks
        chunks = chunk_documents(docs, self.cfg["chunking"])
        JsonlStore(path).rewrite(chunks)
        by_kind = {}
        for c in chunks:
            by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
        stats = {"chunks": len(chunks), "chars": sum(len(c["text"]) for c in chunks),
                 "body_chars": sum(len(c["body"]) for c in chunks), "by_kind": by_kind,
                 "avg_chars": round(sum(len(c["text"]) for c in chunks) / max(len(chunks), 1), 1),
                 "path": path}
        self.m.update("chunk", fp, True, stats)
        self.log(f"[chunk] {len(chunks)}개 청크 {by_kind}")
        return chunks

    def load_chunks(self) -> list[dict]:
        path = self.active.get("chunks_path") or self.m.get("chunk").get("stats", {}).get("path")
        if not path or not os.path.exists(path):
            raise RuntimeError("청크가 없습니다. 먼저 '인덱스 구축'을 실행하세요.")
        return JsonlStore(path).load()

    # ------------------------------------------------------------ 4. embedding model 선택
    def embed_model_path(self) -> str:
        base = self.cfg["embedding"]["model"]
        if self.cfg["finetune"].get("enabled"):
            ft = self.m.get("finetune")
            final = ft.get("stats", {}).get("final_dir")
            if ft.get("done") and final and os.path.exists(final):
                return final
            self.log("[embed] finetune.enabled 이지만 완료된 파인튜닝 모델이 없어 기본 모델을 사용합니다.")
        return base

    def make_embedder(self, model: str | None = None) -> HFEmbedder:
        return HFEmbedder.from_config(self.cfg["embedding"], model or self.embed_model_path())

    # ------------------------------------------------------------ 5. embed + FAISS (배치 체크포인트)
    def stage_embed_index(self, chunks: list[dict]):
        self._guard("stage_embed_index")
        import numpy as np

        from .indexing import build_faiss

        ec = self.cfg["embedding"]
        model = self.embed_model_path()
        fp = fingerprint(self.fp_chunk(), model, {k: ec.get(k) for k in
                                                  ("max_seq_length", "normalize", "doc_prefix", "fp16")})
        fc = {k: v for k, v in (self.cfg.get("faiss") or {}).items() if k != "hnsw_ef_search"}  # 검색 시 파라미터 제외
        ffp = fingerprint(fp, fc)  # FAISS 설정만 바뀌면 임베딩은 재사용하고 인덱스만 다시 만듦
        emb_dir, faiss_dir = self.p("emb", fp), self.p("faiss", ffp)
        embedder = self.make_embedder(model)
        if self.m.is_done("faiss", ffp) and os.path.exists(faiss_dir):
            self.log("[faiss] 캐시 사용")
            self._set_active(faiss_dir=faiss_dir, embed_model=model)
            return embedder
        self._t("embed")
        self.log(f"[embed] {model} on {embedder.device} — {len(chunks)} chunks")
        if not self.m.is_done("embed", fp):
            self.m.update("embed", fp, False)
        vec = embedder.encode_with_checkpoint([c["text"] for c in chunks], emb_dir,
                                              int(ec.get("shard_every_batches", 20)), self.stop, self.log)
        if not self.m.is_done("embed", fp):
            self.m.update("embed", fp, True, {"vectors": int(vec.shape[0]), "dim": int(vec.shape[1]),
                                              "bytes": int(vec.nbytes), "model": model, "seconds": self._el("embed"),
                                              "final_batch_size": embedder.batch_size})
        build_faiss(chunks, vec.astype(np.float32), embedder, faiss_dir, self.cfg.get("faiss"))
        self.m.update("faiss", ffp, True, {"bytes": _dir_bytes(faiss_dir), "ntotal": int(vec.shape[0]),
                                           "index_type": (self.cfg.get("faiss") or {}).get("index_type", "flat")})
        self._set_active(faiss_dir=faiss_dir, embed_model=model)
        self.log(f"[faiss] 인덱스 저장 {faiss_dir} ({(self.cfg.get('faiss') or {}).get('index_type', 'flat')})")
        return embedder

    # ------------------------------------------------------------ 6. BM25
    def stage_bm25(self, chunks: list[dict]):
        self._guard("stage_bm25")
        from .indexing import BM25Index

        bc = self.cfg["bm25"]
        fp = fingerprint(self.fp_chunk(), bc, self.cfg["embedding"]["model"] if bc.get("tokenizer") == "hf" else "")
        path = self.p("bm25", f"bm25_{fp}.pkl")
        if self.m.is_done("bm25", fp) and os.path.exists(path):
            self.log("[bm25] 캐시 사용")
            self._set_active(bm25_path=path)
            return
        idx = BM25Index.build(chunks, bc, self.cfg["embedding"]["model"], self.log)
        idx.save(path)
        self.m.update("bm25", fp, True, {"tokens": idx.n_tokens, "vocab": idx.vocab, "bytes": os.path.getsize(path),
                                         "tokenizer": idx.tok_kind})
        self._set_active(bm25_path=path)
        self.log(f"[bm25] 토큰 {idx.n_tokens:,} / 어휘 {idx.vocab:,}")

    # ------------------------------------------------------------ build all
    def prepare_chunks(self) -> tuple[list[dict], list[dict]]:
        """ZIP 추출 → 파싱 → 청크 (노트북에서는 이 단계만 따로 실행해 결과를 살펴볼 수 있음)"""
        files = self.stage_extract()
        docs = self.stage_parse(files)
        chunks = self.stage_chunk(docs)
        self._set_active(chunks_path=self.m.get("chunk")["stats"]["path"])
        return docs, chunks

    def build_index(self):
        with self.stop.install_signal_handlers():
            _, chunks = self.prepare_chunks()
            self.stage_bm25(chunks)  # CPU, 빠름
            self.stage_embed_index(chunks)  # GPU, 체크포인트
        self.log("✅ 인덱스 구축 완료")

    # ------------------------------------------------------------ QA / split
    def stage_qa(self) -> list[dict]:
        self._guard("stage_qa")
        from .qa_gen import generate_qa, load_external_qa, qa_path

        chunks = self.load_chunks()
        qc = self.cfg["qa_generation"]
        if qc.get("external_path"):
            rows = load_external_qa(qc["external_path"], chunks)
            fp = fingerprint("ext", qc["external_path"], os.path.getsize(qc["external_path"]))
        else:
            fp = fingerprint(self.active.get("chunks_path"), {k: qc[k] for k in ("n_per_doc", "min_chunk_chars", "seed")},
                             self.cfg["llm"]["model"])
            with self.stop.install_signal_handlers():
                rows = generate_qa(chunks, qc, self.cfg["llm"]["model"], qa_path(self.wd, fp), self.stop, self.log)
        path = self.p("qa", f"qa_all_{fp}.jsonl")
        JsonlStore(path).rewrite(rows)
        self.m.update("qa", fp, True, {"samples": len(rows), "path": path})
        self._set_active(qa_path=path)
        return rows

    def stage_split(self) -> tuple[dict, dict]:
        self._guard("stage_split")
        from .chunk_split import filter_by_evidence_keywords, split_samples

        qa_file = self.active.get("qa_path")
        if not qa_file or not os.path.exists(qa_file):
            raise RuntimeError("QA 데이터가 없습니다. 먼저 'QA 생성'을 실행하세요.")
        rows = JsonlStore(qa_file).load()
        sc = self.cfg["split"]
        kept, kw_rep = filter_by_evidence_keywords(rows, sc["evidence_keywords"])
        splits, rep = split_samples(kept, sc["ratio"], int(sc["seed"]), sc.get("stratify"))
        rep["evidence_keywords"] = kw_rep
        rep["input_samples"] = len(rows)
        fp = fingerprint(qa_file, sc)
        d = self.p("splits", fp)
        for k, v in splits.items():
            JsonlStore(os.path.join(d, f"{k}.jsonl")).rewrite(v)
        atomic_write_json(os.path.join(d, "report.json"), rep)
        self.m.update("split", fp, True, {**rep, "dir": d})
        self._set_active(split_dir=d)
        self.log(f"[split] {rep['counts']}  (evidence 필터: {kw_rep})")
        return splits, rep

    def load_split(self, name: str) -> list[dict]:
        d = self.active.get("split_dir")
        if not d:
            raise RuntimeError("분할된 데이터가 없습니다. 먼저 '데이터 분할'을 실행하세요.")
        return JsonlStore(os.path.join(d, f"{name}.jsonl")).load()

    # ------------------------------------------------------------ finetune
    def stage_finetune(self, log_queue=None) -> str:
        self._guard("stage_finetune")
        from .indexing import run_finetune

        train, valid = self.load_split("train"), self.load_split("valid")
        chunks = self.load_chunks()
        fc, ec = self.cfg["finetune"], self.cfg["embedding"]
        fp = fingerprint(self.active.get("split_dir"), self.active.get("chunks_path"), ec["model"],
                         {k: v for k, v in fc.items() if k not in ("enabled", "save_steps", "eval_steps",
                                                                  "save_total_limit")})
        out = self.p("finetune", fp)
        self.m.update("finetune", fp, False, {"output_dir": out})
        HFEmbedder.clear_cache()  # 검색용으로 올라간 임베딩 모델을 내려 학습에 VRAM 확보
        with self.stop.install_signal_handlers():
            final = run_finetune(ec["model"], train, valid, chunks, fc, ec, out, self.stop, self.log, log_queue)
        self.m.update("finetune", fp, True, {"output_dir": out, "final_dir": final, "n_train": len(train)})
        self.log("다음 단계: finetune.enabled=true 로 두고 '인덱스 구축'을 다시 실행하면 파인튜닝 모델로 재임베딩합니다.")
        return final

    # ------------------------------------------------------------ retriever
    def load_retriever(self):
        from .indexing import BM25Index, load_faiss
        from .retrieval import HybridRetriever

        a = self.active
        if not (a.get("faiss_dir") and a.get("bm25_path")):
            raise RuntimeError("인덱스가 없습니다. 먼저 '인덱스 구축'을 실행하세요.")
        embedder = self.make_embedder(a.get("embed_model"))
        store = load_faiss(a["faiss_dir"], embedder, self.cfg.get("faiss"))
        bm25 = BM25Index.load(a["bm25_path"])
        return HybridRetriever(store, bm25, self.load_chunks(), embedder, self.load_router())

    # ------------------------------------------------------------ (선택) 규칙 전처리 · 라우팅 · 대화
    def rules_enabled(self) -> bool:
        """규칙·사전 전처리 사용 여부 (기본 False = 질문을 그대로 검색·LLM 에 전달)"""
        return bool((self.cfg.get("rules") or {}).get("enabled", False))

    def domain_dict(self) -> dict:
        import yaml

        if not self.rules_enabled():
            return {}
        if self.cfg.get("rules", {}).get("dict_data"):  # 노트북 셀에서 직접 정의한 사전
            return self.cfg["rules"]["dict_data"]
        path = self.cfg.get("rules", {}).get("dict_path")
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load_router(self):
        from .retrieval import DocRouter

        rc = self.cfg.get("routing", {})
        path = self.active.get("docs_meta_path")
        if not rc.get("enabled", True) or not path or not os.path.exists(path):
            return None
        orgs = rc.get("orgs") or self.domain_dict().get("orgs")  # 별칭은 선택 (기본: 파일명·표지에서 자동 추출만)
        return DocRouter(read_json(path, []), orgs, rc.get("auto_extract_orgs", True))

    def make_preprocessor(self):
        """rules.enabled=False(기본) → None: 질문을 손대지 않고 그대로 사용"""
        if not self.rules_enabled():
            return None
        from .query_rules import QueryPreprocessor

        rc = self.cfg.get("rules", {})
        return QueryPreprocessor(rc, rc.get("dict_path"))

    def make_chat(self, retriever=None, generator=None):
        from .llm import Generator, RAGChat

        r, lc, rc = self.cfg["retrieval"], self.cfg["llm"], self.cfg.get("routing", {})
        retriever = retriever or self.load_retriever()
        generator = generator or Generator(lc["model"], lc.get("timeout", 120))
        return RAGChat(retriever, generator, self.make_preprocessor(),
                       retrieval=dict(top_k=r["top_k"], dense_weight=r["dense_weight"], fusion=r["fusion"],
                                      candidate_k=r["candidate_k"], rrf_k=r["rrf_k"]),
                       gen=dict(temperature=lc["temperature"], top_p=lc["top_p"], max_tokens=lc["max_tokens"],
                                reasoning_effort=lc.get("reasoning_effort")),
                       domain=lc.get("domain", "공공기관 제안요청서(RFP)"), allow_general=lc.get("allow_general", False),
                       use_routing=rc.get("enabled", True), history_turns=lc.get("history_turns", 3),
                       followup_query=rc.get("followup_query", "none"))

    def eval_set_dir(self, items) -> str:
        """설정(검색·LLM·규칙·사전·질문 목록)이 같으면 같은 폴더 → 답변/RAGAS 결과 재사용(재개).
        items: 질문 리스트(노트북 QUESTIONS) 또는 YAML 경로"""
        if isinstance(items, str):
            try:
                items = open(items, "rb").read().decode("utf-8", "ignore")
            except OSError:
                items = ""
        rules = dict(self.cfg.get("rules") or {}) if self.rules_enabled() else {"enabled": False}
        if self.rules_enabled():
            rules["dict_data"] = self.domain_dict()
            rules.pop("dict_path", None)
        fp = fingerprint(self.active, self.cfg["retrieval"], self.cfg["llm"], rules,
                         self.cfg.get("routing"), self.cfg.get("faiss"), items)
        return self.p("eval_set", fp)

    def eval_run_dir(self, retrieval: dict, gen: dict, split: str) -> str:
        fp = fingerprint(self.active, retrieval, gen, split, self.cfg["evaluation"], self.cfg["llm"]["model"])
        return self.p("eval", fp)

    # ------------------------------------------------------------ 데이터 증가 리포트
    def growth_report(self) -> pd.DataFrame:
        m = self.m.all()
        g = lambda st, k, d=0: (m.get(st, {}).get("stats") or {}).get(k, d)  # noqa: E731
        rows = []

        def add(stage, unit, count, chars=None, size=None, note=""):
            rows.append({"단계": stage, "단위": unit, "개수": count, "문자 수": chars,
                         "크기(MB)": round(size / 2**20, 3) if size else None, "비고": note})

        skip = self.cfg["parsing"]["skip_first_pages"]
        add("① 입력 파일(ZIP 추출)", "파일", g("extract", "files"), None, g("extract", "bytes"),
            f"PDF {g('extract', 'pdf')} · HWP {g('extract', 'hwp')}")
        add("② 원문 텍스트", "PDF 쪽", g("parse", "total_pages"), g("parse", "raw_chars"),
            note=f"HWP {g('parse', 'hwp_docs')}건은 라이브러리가 쪽 정보를 주지 않아 쪽수 제외")
        add(f"③ 서두 제외 (PDF 앞 {skip}쪽 · HWP 표지·목차 자동감지)", "PDF 쪽", g("parse", "kept_pages"),
            g("parse", "raw_chars") - g("parse", "skipped_chars"),
            note=f"-{g('parse', 'skipped_pages')}p / -{g('parse', 'skipped_chars'):,}자")
        add("④ 정제 + 표(Markdown) + 그림 캡션", "블록", g("parse", "records"), g("parse", "clean_chars"),
            note=f"표 {g('parse', 'tables')}개({g('parse', 'table_chars'):,}자) · 반복 머리글 "
                 f"{g('parse', 'repeated_lines_removed')}줄 제거 · 캡션 {g('parse', 'images_captioned')}개")
        add("⑤ 청크(overlap + 문서 머리말)", "청크", g("chunk", "chunks"), g("chunk", "chars"),
            note=f"평균 {g('chunk', 'avg_chars')}자 · {g('chunk', 'by_kind', {})}")
        add("⑥ Dense 임베딩(float32)", "벡터", g("embed", "vectors"), None, g("embed", "bytes"),
            f"dim={g('embed', 'dim')} · {os.path.basename(str(g('embed', 'model', '')))}")
        add("⑦ FAISS 인덱스(+docstore)", "벡터", g("faiss", "ntotal"), None, g("faiss", "bytes"))
        add("⑧ BM25 색인", "토큰", g("bm25", "tokens"), None, g("bm25", "bytes"),
            f"어휘 {g('bm25', 'vocab'):,} · {g('bm25', 'tokenizer', '')}")
        if m.get("qa"):
            add("⑨ QA 샘플", "샘플", g("qa", "samples"))
        if m.get("split"):
            c = g("split", "counts", {})
            add("⑩ Evidence 필터 → 분할", "샘플", sum(c.values()) if c else None,
                note=f"train {c.get('train')} / valid {c.get('valid')} / test {c.get('test')}")
        df = pd.DataFrame(rows)
        base = g("parse", "raw_chars") or None
        prev = None
        mult_prev, mult_raw = [], []
        for ch in df["문자 수"]:
            mult_prev.append(round(ch / prev, 3) if ch and prev else None)
            mult_raw.append(round(ch / base, 3) if ch and base else None)
            if ch:
                prev = ch
        df.insert(4, "이전 대비(×)", mult_prev)
        df.insert(5, "원문 대비(×)", mult_raw)
        return df

    def growth_json(self) -> str:
        return json.dumps(self.m.all(), ensure_ascii=False, indent=2, default=str)


__all__ = ["Pipeline", "StopRequested"]
