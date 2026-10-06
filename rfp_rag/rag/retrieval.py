"""③ 검색 — 하이브리드(BM25 × Dense) · 기관 라우팅 + 멀티 쿼리 SmartRetriever · 리랭커(CrossEncoder).
담당: 검색 · 리랭킹.  (예전 engine.py 의 검색 부분 + 02·03 노트북 1-1b 셀의 검색 코드)"""
from __future__ import annotations

import math
import re
import time

import numpy as np

from .indexing import BM25Index


# ====================================================================================================
# 인덱스: 하이브리드 검색 + 기관 라우팅   (예전 engine.py)
# ====================================================================================================
# ------------------------------------------------------------------ Hybrid
def _minmax(d: dict[str, float]) -> dict[str, float]:
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


class HybridRetriever:
    """final = w·Dense + (1-w)·BM25   (w = dense_weight, 0=BM25만, 1=Dense만)
    fusion='score' : 각 후보군 min-max 정규화 후 가중합
    fusion='rrf'   : 가중 Reciprocal Rank Fusion  w/(k+rank_d) + (1-w)/(k+rank_b)
    doc_ids        : 지정 시 해당 문서 안에서만 검색 (기관 라우팅)
    """

    def __init__(self, faiss_store, bm25: BM25Index, chunks: list[dict], embedder, router: "DocRouter | None" = None):
        self.store, self.bm25, self.embedder, self.router = faiss_store, bm25, embedder, router
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self._bm25_doc = np.array([self.by_id[cid]["doc_id"] if cid in self.by_id else "" for cid in bm25.chunk_ids])
        self._qcache: dict[str, list[float]] = {}  # 같은 질문 반복 검색(비율 스윕 등) 시 재임베딩 방지
        from langchain_community.vectorstores.utils import DistanceStrategy

        self._l2 = faiss_store.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE

    def _qvec(self, query: str):
        if query not in self._qcache:
            if len(self._qcache) > 2048:
                self._qcache.clear()
            self._qcache[query] = self.embedder.embed_query(query)
        return self._qcache[query]

    def _dense(self, query: str, k: int, doc_ids: set | None) -> dict[str, float]:
        qv = self._qvec(query)
        kw = {}
        if doc_ids:
            kw = {"filter": {"doc_id": list(doc_ids)}, "fetch_k": int(self.store.index.ntotal)}
        out = {}
        for doc, score in self.store.similarity_search_with_score_by_vector(qv, k=k, **kw):
            out[doc.metadata["chunk_id"]] = -float(score) if self._l2 else float(score)  # L2 거리 → 유사도
        return out

    def _sparse(self, query: str, k: int, doc_ids: set | None) -> dict[str, float]:
        scores = self.bm25.bm25.get_scores(self.bm25.tokenizer(query))
        if doc_ids:
            scores = np.where(np.isin(self._bm25_doc, list(doc_ids)), scores, 0.0)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k] if k else []
        return {self.bm25.chunk_ids[i]: float(scores[i]) for i in idx if scores[i] > 0}

    def search(self, query: str, top_k: int = 5, dense_weight: float = 0.5, fusion: str = "score",
               candidate_k: int = 50, rrf_k: int = 60, doc_ids: set | list | None = None) -> list[dict]:
        w = float(min(max(dense_weight, 0.0), 1.0))
        doc_ids = set(doc_ids) if doc_ids else None
        dense = self._dense(query, candidate_k, doc_ids) if w > 0 else {}
        sparse = self._sparse(query, candidate_k, doc_ids) if w < 1 else {}
        d_rank = {cid: r for r, cid in enumerate(sorted(dense, key=dense.get, reverse=True), 1)}
        s_rank = {cid: r for r, cid in enumerate(sorted(sparse, key=sparse.get, reverse=True), 1)}
        if fusion == "rrf":
            fused = {cid: (w / (rrf_k + d_rank[cid]) if cid in d_rank else 0.0)
                     + ((1 - w) / (rrf_k + s_rank[cid]) if cid in s_rank else 0.0)
                     for cid in set(dense) | set(sparse)}
        else:
            dn, sn = _minmax(dense), _minmax(sparse)
            fused = {cid: w * dn.get(cid, 0.0) + (1 - w) * sn.get(cid, 0.0) for cid in set(dense) | set(sparse)}
        out = []
        for cid in sorted(fused, key=fused.get, reverse=True)[:top_k]:
            c = self.by_id[cid]
            out.append({**c, "score": round(fused[cid], 4), "dense": round(dense.get(cid, float("nan")), 4),
                        "bm25": round(sparse.get(cid, float("nan")), 3), "dense_rank": d_rank.get(cid),
                        "bm25_rank": s_rank.get(cid)})
        return out

    def search_routed(self, query: str, route: dict, top_k: int = 5, **kw) -> list[dict]:
        """라우팅 결과에 따라: 단일 기관 → 해당 문서만 / 비교(2개 이상) → 기관별 균등 검색 후 교차 병합
        / 제외('다른 기관') → 이전 기관 문서 제외 / 없음 → 전체"""
        mode = route.get("mode", "none")
        if mode == "compare":
            groups = [set(g) for g in route["groups"].values() if g]
            k_each = max(2, math.ceil(top_k / max(len(groups), 1)))
            lists = [self.search(query, top_k=k_each, doc_ids=g, **kw) for g in groups]
            merged, seen = [], set()
            for i in range(k_each):
                for lst in lists:
                    if i < len(lst) and lst[i]["chunk_id"] not in seen:
                        merged.append(lst[i])
                        seen.add(lst[i]["chunk_id"])
            return merged
        if mode in ("single", "inherit", "exclude"):
            return self.search(query, top_k=top_k, doc_ids=route.get("doc_ids"), **kw)
        return self.search(query, top_k=top_k, **kw)


# ------------------------------------------------------------------ 기관 라우팅
_ORG_SUFFIX = (r"(?:공단|공사|연구원|연구소|대학교|대학|과학기술원|재단|진흥원|협회|센터|위원회|협력단|사회서비스원"
               r"|교육청|시청|도청|구청|은행|박물관|도서관|병원|진흥공사|관리원|정보원|평가원|개발원|기술원|산학협력단)")
_ORG_RE = re.compile(r"(?:한국|국립|재단법인|\(재\)|\(사\))?\s?[가-힣A-Za-z]{2,15}" + _ORG_SUFFIX)
_GENERIC_ORG = re.compile(r"(?:자문|평가|심의|운영|선정|추진|협의|기술|제안)?위원회$|^(?:해당|발주|주관|수요|관련|전문|계약|용역|사업|기타|본|각)"
                          r"|^[가-힣]{1,2}(?:센터|연구소)$|^산학협력단$")
_OTHER_RE = re.compile(r"다른\s*(?:기관|곳|발주처|발주기관|사업|데)|타\s*기관|그\s*외\s*기관|제외하고")


class DocRouter:
    """질문에 등장한 발주기관 → 해당 문서로 검색 범위를 좁힘 (RFP 가 많을수록 정확도 ↑).
    기관명은 ① domain_dict.yaml 의 orgs(별칭) ② 파일명·표지(앞 N페이지, 임베딩 제외분)·본문 앞부분에서 자동 추출."""

    def __init__(self, docs_meta: list[dict], orgs: dict | None = None, auto_extract: bool = True):
        self.docs = {d["doc_id"]: d for d in docs_meta}
        self.orgs: dict[str, list[str]] = {k: list(dict.fromkeys([k] + list(v or []))) for k, v in (orgs or {}).items()}
        # 발주기관은 파일명·표지(앞 N페이지)에 나옴 → 본문(head_text)은 표지가 없을 때만 사용
        hay = {d["doc_id"]: f"{d.get('title', '')}\n" + (d.get("cover_text") or d.get("head_text", "")[:1500])
               for d in docs_meta}
        self._hay = hay
        if auto_extract:
            for did, text in hay.items():
                for m in _ORG_RE.finditer(text[:2500]):
                    name = re.sub(r"^\(?(재|사)\)?\s*|재단법인\s*", "", m.group().strip())
                    if (len(name) >= 4 and not _GENERIC_ORG.search(name)
                            and not any(name in al or any(a in name for a in al if len(a) >= 4)
                                        for al in self.orgs.values())):
                        self.orgs.setdefault(name, [name])
        # 기관 → 문서
        self.org_docs: dict[str, set] = {}
        for org, aliases in self.orgs.items():
            ds = {did for did, text in hay.items() if any(self._has(text, a) for a in aliases)}
            if ds:
                self.org_docs[org] = ds

    @staticmethod
    def _has(text: str, alias: str) -> bool:
        if alias.isascii():
            return re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", text, re.IGNORECASE) is not None
        return alias in text

    def orgs_in(self, query: str) -> list[str]:
        found = [org for org, al in self.orgs.items() if org in self.org_docs and any(self._has(query, a) for a in al)]
        # '한국원자력연구원' 과 자동추출 '원자력연구원' 처럼 겹치면 긴 이름만
        return [o for o in found if not any(o != p and o in p for p in found)]

    def _narrow(self, doc_ids: set, query: str) -> set:
        """같은 기관 문서가 여러 개면 질문 단어와 제목이 가장 많이 겹치는 문서로 좁힘"""
        if len(doc_ids) <= 1:
            return doc_ids
        words = {w for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", query)}
        score = {d: sum(w in self.docs[d].get("title", "") for w in words) for d in doc_ids}
        best = max(score.values())
        return {d for d, s in score.items() if s == best} if best > 0 else doc_ids

    def route(self, query: str, state: dict | None = None) -> dict:
        state = state if state is not None else {}
        found = self.orgs_in(query)
        prev = set(state.get("active_doc_ids") or [])
        if _OTHER_RE.search(query) and prev:
            keep = set(self.docs) - prev
            r = {"mode": "exclude", "orgs": [], "doc_ids": sorted(keep), "excluded": sorted(prev)}
        elif len(found) >= 2:
            groups = {o: sorted(self._narrow(self.org_docs[o], query)) for o in found}
            r = {"mode": "compare", "orgs": found, "groups": groups,
                 "doc_ids": sorted({d for g in groups.values() for d in g})}
        elif len(found) == 1:
            r = {"mode": "single", "orgs": found, "doc_ids": sorted(self._narrow(self.org_docs[found[0]], query))}
        elif prev:
            r = {"mode": "inherit", "orgs": state.get("active_orgs", []), "doc_ids": sorted(prev)}
        else:
            r = {"mode": "none", "orgs": [], "doc_ids": []}
        if r["mode"] in ("single", "compare"):
            state["active_doc_ids"], state["active_orgs"] = r["doc_ids"], r["orgs"]
        r["titles"] = [self.docs[d].get("title", d)[:40] for d in r.get("doc_ids", [])][:6]
        return r


# ====================================================================================================
# 리랭커 (CrossEncoder) — 후보를 질문과 1:1 로 다시 채점   (예전 01 2-4 · 02·03 1-1b 셀)
# ====================================================================================================
def rerank_room(model: str, need_disk_gb: float = 4.0, need_ram_gb: float = 4.0) -> tuple[bool, str]:
    """리랭커를 올릴 여유가 있나: (가능 여부, 안내). 없으면 건너뛰어 커널이 메모리·디스크 부족으로 죽지 않게"""
    import os
    import shutil

    import psutil
    from huggingface_hub import constants

    cached = os.path.isdir(os.path.join(constants.HF_HUB_CACHE, "models--" + model.replace("/", "--"))) or os.path.isdir(model)
    where = constants.HF_HUB_CACHE if os.path.isdir(constants.HF_HUB_CACHE) else os.path.expanduser("~")
    disk = shutil.disk_usage(where).free / 1024**3
    ram = psutil.virtual_memory().available / 1024**3
    if not cached and disk < need_disk_gb:
        return False, f"디스크 여유 {disk:.1f}GB — 리랭커 첫 다운로드(약 2.3GB)에 {need_disk_gb:g}GB 이상 필요"
    if ram < need_ram_gb:
        return False, f"RAM 여유 {ram:.1f}GB — 리랭커를 올리려면 {need_ram_gb:g}GB 이상 필요 (다른 노트북 커널을 끄세요)"
    return True, f"디스크 여유 {disk:.1f}GB · RAM 여유 {ram:.1f}GB" + (" · 캐시에 있음" if cached else " · 첫 다운로드")


def load_cross_encoder(model: str, max_length: int = 512):
    """CrossEncoder 로드 — GPU 면 fp16 (transformers 4.56+ 는 dtype, 이전은 torch_dtype)"""
    import torch
    import transformers
    from packaging.version import Version
    from sentence_transformers import CrossEncoder

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dkey = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    return CrossEncoder(model, device=dev, max_length=max_length, model_kwargs={dkey: torch.float16} if dev == "cuda" else {})


# ====================================================================================================
# SmartRetriever — ① 질문 해석 → ② 멀티 쿼리 하이브리드(RRF) → ③ 리랭커   (예전 02·03 노트북 1-1b 셀)
# ====================================================================================================
class SmartRetriever:
    """chat.ret 자리에 끼우는 래퍼 — 원래 검색기(HybridRetriever)는 그대로 쓰고 search_routed 만 확장.
    rewriter: rag.rewrite.QueryRewriter (None 이면 질문 해석 없이 원래 질문만)"""

    def __init__(self, base, chat=None, rewriter=None, rerank: bool = True, rerank_model: str = "BAAI/bge-reranker-v2-m3",
                 candidates: int = 30, rrf_k: int = 60, rerank_batch: int = 32):
        self.base, self.chat, self.rewriter = base, chat, rewriter
        self.rerank, self.rerank_model, self.candidates = bool(rerank), rerank_model, int(candidates)
        self.rrf_k, self.rerank_batch = int(rrf_k), int(rerank_batch)
        self.last, self._ce = None, None

    def __getattr__(self, name):              # router · by_id · embedder · store 등은 원래 검색기 것을 그대로
        if name in ("base", "__setstate__"):
            raise AttributeError(name)
        return getattr(self.base, name)

    # ── 질문 해석 기록 (02 노트북 표·CSV·비용 셀이 쓰는 이름 그대로)
    def rewrite(self, question):
        history = self.chat.history if self.chat is not None else None
        return self.rewriter.rewrite(question, history) if self.rewriter is not None else None

    def attach(self, path):
        if self.rewriter is not None:
            self.rewriter.attach(path)

    def rw_for(self, question):
        return self.rewriter.rw_for(question) if self.rewriter is not None else None

    def usage(self):
        from .rewrite import ZERO_USAGE

        return self.rewriter.usage() if self.rewriter is not None else dict(ZERO_USAGE)

    def usage_since(self, before):
        now = self.usage()
        return {k: now[k] - before[k] for k in now}

    # ── ③ 리랭커
    def reranker(self):
        if self._ce is None:
            self._ce = load_cross_encoder(self.rerank_model)
        return self._ce

    def prepare_reranker(self) -> tuple[bool, str]:
        """리랭커를 미리 올림. 여유가 없거나 실패하면 리랭커를 끄고 (False, 이유) — 검색은 리랭커 없이 계속"""
        if not self.rerank:
            return False, "RERANK=False"
        ok, why = rerank_room(self.rerank_model)
        if ok:
            try:
                t0 = time.time()
                self.reranker()
                why = f"{time.time() - t0:.1f}s · {why}"
            except Exception as e:  # noqa: BLE001  — 다운로드 실패·메모리 부족 등
                ok, why = False, f"로드 실패 {type(e).__name__}: {str(e)[:150]}"
        if not ok:
            self.rerank, self._ce = False, None
        return ok, why

    # ── ② 멀티 쿼리 하이브리드 + ③ 리랭커  (RAGChat.ask 가 부르는 자리)
    def search_routed(self, query, route, top_k=5, **kw):
        t0 = time.time()
        rw = self.rewrite(query)
        queries = [query] + (rw["queries"] if rw else [])
        # 원래 질문에서 기관을 못 찾았으면(오타·영어·줄임말) 다시 쓴 질문으로 기관 라우팅 재시도
        if route.get("mode") == "none" and rw and rw["queries"] and self.base.router is not None:
            r2 = self.base.router.route(rw["queries"][0], self.chat.state if self.chat is not None else {})
            if r2.get("mode") in ("single", "compare"):
                route = r2
        pool = max(top_k, self.candidates if self.rerank else top_k)
        fused, first = {}, {}
        for q in queries:                                  # 질문마다 BM25 + Dense 하이브리드
            for rank, h in enumerate(self.base.search_routed(q, route, top_k=pool, **kw), 1):
                fused[h["chunk_id"]] = fused.get(h["chunk_id"], 0.0) + 1.0 / (self.rrf_k + rank)   # RRF 병합
                first.setdefault(h["chunk_id"], h)
        hits = [{**first[c], "mq_score": round(fused[c], 5)} for c in sorted(fused, key=fused.get, reverse=True)[:pool]]
        t1 = time.time()
        if self.rerank and hits:
            rq = rw["queries"][0] if rw and rw["queries"] else query
            scores = self.reranker().predict([(rq, h["text"]) for h in hits], batch_size=self.rerank_batch, show_progress_bar=False)
            for h, sc in zip(hits, scores):
                h["rerank"] = round(float(sc), 4)
            hits.sort(key=lambda h: -h["rerank"])
        self.last = dict(question=query, rewrite=rw, queries=queries, route=route.get("mode"),
                         search_s=round(time.time() - t0, 3), rerank_s=round(time.time() - t1, 3) if self.rerank else None)
        return hits[:top_k]
