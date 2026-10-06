"""(호환용) 예전 engine.py — 이제 세 파일로 나뉘었습니다. 새 코드는 아래 모듈을 직접 import 하세요.

    rag/parsing.py    ① 문서 파싱               (담당: 문서 파싱)
    rag/indexing.py   ② 임베딩 · FAISS · BM25 · 파인튜닝 (담당: 청크 · 색인)
    rag/retrieval.py  ③ 하이브리드 검색 · 라우팅 · SmartRetriever · 리랭커 (담당: 검색 · 리랭킹)

`from rag.engine import HFEmbedder` 같은 예전 코드가 그대로 동작하도록 이름만 다시 내보냅니다.
"""
from .indexing import *  # noqa: F401,F403
from .indexing import _MODEL_CACHE, _is_oom  # noqa: F401
from .parsing import *  # noqa: F401,F403
from .retrieval import *  # noqa: F401,F403
from .retrieval import _minmax  # noqa: F401
