# RFP Naive RAG (PDF + HWP · FAISS/BM25 하이브리드 · gpt-5-mini · RAGAS)

RFP ZIP(PDF + HWP)을 넣으면 **파싱 → 청크 → BM25 + HF 임베딩/FAISS → gpt-5-mini 답변 → RAGAS 평가**까지 진행합니다.
**모든 단계에 체크포인트가 있어** GPU 메모리 문제로 멈추거나 컴퓨터가 꺼져도 **같은 셀(명령)을 다시 실행하면 중간부터 이어서** 진행합니다.

## ▶ 노트북 3개로 실행 (권장)
| 순서 | 노트북 | 하는 일 | GPU · 비용 |
|---|---|---|---|
| ① | **`01_rag_build.ipynb`** — RAG 학습 | ZIP → PDF(PyPDFLoader·PyMuPDFLoader) / HWP·HWPX(**HwpHwpxLoader**) → 정제 → 청크 → BM25 → HF 임베딩(체크포인트) → FAISS → `artifacts/build_config.json` 저장 | GPU 사용, OpenAI 불필요 |
| ② | **`02_generate_eval.ipynb`** — 모델 실행 및 실험 평가 | 01의 인덱스를 **불러오기만** → GPT 질문 해석 → 멀티 쿼리 검색 → 리랭커 → 질문별 답변(gpt-5-mini) → 막판 RAGAS → 최종 파라미터 | 학습·임베딩·파싱 없음 (`generate_only=True`라 호출되면 오류로 막음) |
| ③ | **`03_chatbot.ipynb`** — 챗봇 | 01 결과만으로 바로 실행(02 불필요). 2-2 셀에 질문을 적어 실행(멀티턴) + 답변마다 **실시간 RAGAS**(화면 표시만, 파일 저장 없음) | 학습·임베딩 없음 |

- 02는 01이 끝난 뒤 몇 번이든 따로 돌릴 수 있습니다. temperature·top_p·max tokens·top-k·Dense:BM25 비율·질문을 바꿔도 **재학습 없이 Generate만** 합니다.
- 01을 실행하지 않았거나 인덱스 파일이 없으면 02의 `0-2` 셀이 "01_rag_build.ipynb 를 먼저 실행하세요"라는 오류로 멈춥니다.
- **질문은 규칙·사전 전처리 없이 원문 그대로** 검색과 LLM에 보냅니다. 오타(`세액공재`), 한영 자판(`dnjftp rhdwp`), 한자(`月貰`), 영어 혼용 질문이 검색에서 얼마나 버티는지가 그대로 드러납니다. (`rag/query_rules.py`는 `rules.enabled: true`로 켤 수 있는 CLI 실험용으로만 남겨 두었고 기본은 꺼져 있습니다.)

### 설치 — Python 3.10, 로컬 RTX 5070 Ti (Windows 또는 Linux/WSL2)
```bash
nvidia-smi                                             # 드라이버 570 이상
python3.10 -m venv .venv && .venv\Scripts\activate     # Linux/WSL: source .venv/bin/activate
# PyTorch — RTX 50(Blackwell sm_120)은 CUDA 12.8 이상 빌드 필수 (cu124 이하 = 'no kernel image' 오류)
python3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python3.10 -m pip install -r requirements.txt jupyterlab
python3.10 -m pip install langchain-hwp-hwpx-loader    # HWP/HWPX 로더 (requirements 에도 포함)
copy .env.example .env      # Linux: cp  → OPENAI_API_KEY=sk-... / HF_TOKEN=hf_... 입력
mkdir data  &&  (rfp.zip 을 data\rfp.zip 으로 복사)
jupyter lab                 # 커널: 위 .venv (Python 3.10)
```
- 01의 `0-0` 셀이 위 설치를 자동으로 확인합니다. 커널의 python으로 `-m pip install langchain-hwp-hwpx-loader`를 실행하고, torch의 sm_120 지원 여부와 LangChain 조합도 맞춥니다. 설치했다면 커널을 재시작하세요.
- **LangChain 조합 (중요):** `langchain-hwp-hwpx-loader`가 `langchain-core>=1.0`을 요구하므로 core·openai·text-splitters는 1.x를 씁니다. `langchain-community`는 **0.3.31로 고정**합니다. ragas 0.4.3이 community 0.4에서 사라진 모듈을 import하기 때문입니다. 이 조합(core 1.6 + community 0.3.31 + openai 1.6 + ragas 0.4.3)은 Python 3.10에서 PyPDFLoader·PyMuPDFLoader·FAISS·HwpHwpxLoader·RAGAS 6개 지표까지 확인했습니다.
- `0-1` 셀은 `OPENAI_API_KEY`와 `HF_TOKEN`을 **따로** 읽습니다. 01에서는 둘 다 선택이고, 02에서는 `OPENAI_API_KEY`가 필수입니다.
  ```dotenv
  OPENAI_API_KEY = sk-proj-...     # 개인 OpenAI 프로젝트 키 (= 양옆 공백·따옴표 허용)
  HF_TOKEN = hf_...                # 선택
  ```
  - **`.env`가 시스템 환경변수보다 우선**합니다. UTF-8, 메모장 BOM, ANSI(cp949)를 모두 읽습니다.
  - `.env`에 값이 없을 때만 입력창이 뜹니다(VS Code는 창 맨 위). 입력값은 메모리에만 두고, `save_to_env=True`일 때만 `.env`에 저장합니다. 화면에는 `sk-pr…1234`처럼 가려서 표시합니다.
- 01은 `REQUIRE_GPU = True`(기본)라서 5070 Ti를 못 쓰면 원인을 알려주며 멈춥니다. 02는 생성이 API라 GPU가 없어도 CPU로 질문을 임베딩합니다.
- 모델을 처음 받을 때 KURE-v1 약 2.3GB와 Qwen2.5-VL-3B 약 7.5GB를 내려받습니다. C 드라이브가 부족하면 `0-1` 셀의 `HF_CACHE_DIR`를 지정하세요.

### 🪙 토큰 절약 (02 · 03 의 `0-2` 셀)
| 설정 | 효과 |
|---|---|
| `LLM_MODEL = "gpt-6-luna"` + `REASONING_EFFORT = "none"` | 추론 토큰 0, 1M 토큰당 입력 $0.10 · 출력 $0.50 (gpt-5-mini 는 $0.25 · $2.00) |
| `REASONING_EFFORT` = none · minimal · low · medium · high · xhigh · max | 낮을수록 추론 토큰 감소. 모델이 지원하지 않는 값은 가장 가까운 낮은 값으로 자동 조정 (gpt-5-mini: none→minimal, xhigh/max→high) |
| `JUDGE_MODEL` · `JUDGE_REASONING_EFFORT` | RAGAS 심판(평가 토큰의 대부분)의 모델·추론 강도 |
| `EVAL_MODE = "free"` | 골든 데이터셋(`eval/golden_set.jsonl`, 02 의 `0-5`)으로 **토큰 0** 평가 — Key Fact(답변/근거) · Semantic Sim |
| 03 `LIVE_RAGAS = False` | 챗봇 답변마다 RAGAS 심판 호출 안 함 |
- 02 `1-3` 과 `2`, 03 답변 아래에 실제 사용 토큰(입력·출력·추론)과 예상 비용이 표시됩니다.

### 01_rag_build 흐름
`0-0 설치(langchain-hwp-hwpx-loader 포함) → 0-1 토큰·GPU → 0-2 ★학습 하이퍼파라미터 → 1-1 PyPDFLoader·PyMuPDFLoader·HwpHwpxLoader 직접 호출 → 1-2 파싱·품질표 → 1-3 청크 크기 400/700/1000 비교 → 1-4 미리보기 → 2-1 BM25 · 임베딩(■ 재개) → 2-2 FAISS 확인 → 2-3 데이터 증가 → 3 build_config.json 저장`
- `0-2`: `PDF_TEXT_LOADER`, `PDF_TABLE_LOADER`, `PYPDF`, `TABLE_SETTINGS`, `HWP`(table_style·include_notes 등 HwpHwpxLoader 옵션), `HWP_FRONT_MATTER`, `IMAGE_PARSER`, `PARSING`, `CHUNK_SIZE`·`OVERLAP_RATIO`·`SEPARATORS`, `EMBED_MODEL`, `FAISS_CFG`, `BM25_CFG`.

### 02_generate_eval 흐름
`0-1 토큰 → 0-2 ★생성·검색·RAGAS 파라미터 + 01 결과 불러오기 → 0-3 ★질문 세트(직접 편집) → 1-1 인덱스 로드 → 1-2 질문별 답변(■ 재개) → 1-3 요약 → 2 막판 RAGAS(실시간 표) → 3 최종 파라미터 → 부록 Dense:BM25 비율 탐색(LLM 없음)`
- `0-2`: `TOP_K`, `DENSE_WEIGHT`(BM25 = 1 − 값), `FUSION`, `CANDIDATE_K`, `HNSW_EF_SEARCH`, `ROUTING`, `LLM_MODEL`, `TEMPERATURE`, `TOP_P`, `MAX_TOKENS`, `REASONING_EFFORT`, `ALLOW_GENERAL`, `HISTORY_TURNS`, `RAGAS_CFG`.
- `0-3`: `QUESTIONS`(연말정산 15문항, 입력 그대로)와 `RFP_QUESTIONS`(멀티턴, `USE_RFP_QUESTIONS=True`로 포함). 리스트를 직접 고치세요. LLM으로 질문을 만들지 않습니다. `SAVE_QUESTIONS=True`면 `eval/eval_questions.yaml`에도 저장합니다.
- 같은 설정·질문이면 저장된 답변과 RAGAS 점수를 재사용합니다(API 호출 0회). 하나라도 바꾸면 새 결과 폴더에서 다시 계산합니다.

### 기관 라우팅 · 멀티턴 (선택, 사전 없음)
질문에 나온 발주기관을 **파일명·표지에서 자동 추출한 기관명**과 맞춰 검색 범위를 정합니다. 별칭 사전은 쓰지 않습니다(`ROUTING["orgs"]`에 직접 넣을 수는 있음).
- **single:** 해당 기관 문서 안에서만 검색
- **compare:** 두 기관 이상이면 기관별로 똑같이 나눠 검색한 뒤 합침
- **inherit:** 기관 언급이 없는 후속 질문은 직전 기관을 이어받음
- **exclude:** "다른 기관"이 들어간 질문은 직전 기관 문서를 제외

### 평가 세트
- **연말정산 15문항:** 오타, 띄어쓰기, 줄임말, 한영 자판, 영문 혼용, 영어, 한자, 잘못된 전제(`30%` → 15%/17%로 바로잡기, `5월 31일` → 3월 10일).
- **RFP 멀티턴 8문항 (대화 4개, 선택).**

`reference`가 있는 질문은 RAGAS 5개 지표를 모두 계산하고, 없는 질문은 참조가 필요 없는 지표만 계산합니다. 결정적 지표 Route OK(라우팅)와 Key Fact(핵심 사실 포함)도 함께 봅니다.
> ⚠️ 연말정산 질문의 정답 근거(국세청 연말정산 안내 등)가 ZIP에 없으면 "문서에서 확인할 수 없습니다"가 정상 동작입니다. 참고용으로 `ALLOW_GENERAL=True`를 켜면 답변에 `[문서 외 일반지식]`이 표시됩니다.

### 하드웨어 자동 튜닝 (`rag/hw.py`)
| 항목 | 로컬 RTX 5070 Ti 16GB | 원격 L4 24GB |
|---|---|---|
| 임베딩 | fp16 · batch 32 · 10배치마다 샤드 | fp16 · batch 64 · 10배치마다 샤드 |
| OOM 대응 | 배치를 절반씩 자동 축소 | 동일 |
| 그래프 캡션(Qwen2.5-VL-3B, bf16 ≈ 7.5GB) | 켬 · 해상도 상한 768·28·28 · 파싱 후 해제 | 켬 · 1024·28·28 |
| 파인튜닝 | bf16 · 논리배치 64 / mini 8 (≈11~12GB) | bf16 · 64 / 16 |
| DataLoader worker | Windows 0 / Linux 최대 4 | 2 |
| 기타 | TF32 켬 · 모델 1회 적재 공유 | + `expandable_segments` (Linux) |

데스크톱은 화면과 브라우저가 VRAM을 1~2GB 쓰므로 5070 Ti는 L4보다 보수적으로 잡았습니다. 파인튜닝할 때는 게임이나 다른 GPU 앱을 닫아두세요.

## CLI / Gradio (선택)
```bash
python main.py build        # ZIP → 파싱 → 청크 → BM25 → 임베딩 → FAISS
python main.py qa           # gpt-5-mini 로 평가/학습용 QA 생성
python main.py split --set split.ratio=[0.8,0.1,0.1] --set split.seed=7 --set split.stratify=doc_id
python main.py finetune     # (선택) 임베딩 파인튜닝 → finetune.enabled=true 로 build 재실행
python main.py eval         # RAGAS (샘플마다 누적 평균표 출력)
python main.py ask "월세 세액공제 한도는?" --set retrieval.dense_weight=0.7
python main.py evalset --set rules.enabled=true   # (실험) 규칙·사전 전처리를 켜고 비교
python main.py evalset      # 평가 질문 세트 답변 + 막판 RAGAS (중단 후 재실행 시 이어서)
python main.py app --share  # Gradio UI
```

## 체크포인트 / 재개 동작
| 단계 | 저장 단위 | 강제 종료 후 재실행 시 |
|---|---|---|
| ZIP 추출 | 파일 | 이미 추출된 파일은 크기 비교 후 건너뜀 |
| 파싱(PDF/HWP) | **문서 1개마다** JSON | 끝난 문서는 캐시 사용 |
| 그래프 캡션(VLM) | **이미지 1개마다** JSONL | 캡션된 이미지는 다시 돌리지 않음 |
| 임베딩(GPU) | **N배치마다 샤드 `.npy`** + progress.json | 마지막 샤드 다음 청크부터 재개. OOM이면 배치를 절반씩 줄여 재시도하고, 줄인 값을 기억 |
| QA 생성 | **QA 1건마다** JSONL (fsync) | 실패 건만 재시도 |
| 임베딩 파인튜닝 | HF Trainer `checkpoint-XXXX` (가중치+옵티마이저+스케줄러+RNG) | 마지막 checkpoint 에서 `resume_from_checkpoint`. `auto_find_batch_size`로 OOM 대응 |
| RAGAS 평가 | **샘플 1개마다** JSONL | 끝난 샘플은 재사용하고 나머지만 평가 |

- 모든 쓰기는 `임시파일 → fsync → os.replace`라서 쓰다가 꺼져도 파일이 반쯤 깨지지 않습니다.
- `artifacts/manifest.json`에 단계별 **fingerprint(설정 해시)**가 기록됩니다. 설정이 같으면 건너뛰고, 바꾸면(예: chunk_size) 그 단계부터 다시 계산합니다.
- Ctrl+C를 한 번 누르거나 UI의 **⏹ 중지**를 누르면 현재 배치까지 저장하고 멈춥니다. Ctrl+C를 두 번 누르면 즉시 종료합니다.

## 문서 처리 — 라이브러리 로더만 사용 (직접 만든 로더 없음)
| 대상 | 라이브러리 | 비고 |
|---|---|---|
| PDF 본문 | LangChain **PyPDFLoader** (`PDF_TEXT_LOADER`) | `extraction_mode` plain/layout. PyMuPDFLoader로 바꿀 수 있음 |
| PDF 표 | LangChain **PyMuPDFLoader** `extract_tables="markdown"` (`PDF_TABLE_LOADER`) | 여러 줄 셀은 `<br>`로 보존, 본문에 섞인 표 평문은 제거 |
| PDF 이미지 | PyMuPDFLoader `extract_images=True, images_parser=…` | langchain-community 0.3.x의 이미지 건너뛰기 버그를 0.4.2 코드로 패치 |
| HWP 5.x / HWPX | **langchain-hwp-hwpx-loader** `HwpHwpxLoader` (본문·표 Markdown·각주·이미지) | `mode="single"`, 표는 본문 제자리에 Markdown. 탭 자리 깨진 글자 정리, 세로 병합 셀 칸 보정. 쪽 정보가 없어 표지·목차를 자동 감지해 제외 |
| 이미지 → 텍스트 | LangChain `images_parser` 형식: `hf-vlm`(Qwen2.5-VL), `rapidocr`, `tesseract`, `llm` | `IMAGE_PARSER`. 같은 이미지는 캐시하고, 여러 쪽에 반복되는 로고는 제외 |
| 한자 | `必(필)` 병기 + BM25(kiwi) 독음 색인 | `hanja_mode` |
| 임베딩 | HF **`nlpai-lab/KURE-v1`** | 대안: `BAAI/bge-m3`, `dragonkue/BGE-m3-ko` |

> `HwpHwpxLoader`는 HWP·HWPX 전용입니다. PDF는 LangChain의 PyPDFLoader·PyMuPDFLoader가 맡습니다.
> HWP는 라이브러리가 쪽 번호를 주지 않으므로, HWP 청크의 출처는 쪽 대신 **섹션 경로**로 표시됩니다.

## 조절 가능한 파라미터
- **하이브리드 비율**: `retrieval.dense_weight` (0 = BM25만, 1 = Dense만, BM25 = 1 − 값). 융합 방식은 `score`(min-max 가중합)와 `rrf`(가중 RRF) 중 선택
- **생성**: `temperature`, `top_p`, `max_tokens`, `reasoning_effort`. `top-k`는 검색 청크 수
- **분할**: `split.ratio`, `split.seed`, `split.stratify` (none/doc_id/doc_type/question_type/chunk_kind), `split.evidence_keywords` (min/max 개수, 근거 청크 존재 여부 검사, must_include + any/all, exclude)

## 꼭 알아둘 점
- **gpt-5-mini는 API가 temperature·top_p를 기본값(1) 외에는 거부합니다.** 슬라이더 값을 먼저 그대로 보내 보고, 거부되면 해당 파라미터를 빼고 재시도한 뒤 경고를 띄웁니다. 이후 호출은 재시도 없이 바로 보냅니다. temperature를 지원하는 모델(예: `gpt-4.1-mini`)로 바꾸면 값이 그대로 적용됩니다.
- `max_tokens`는 `max_completion_tokens`로 전달되고 **reasoning 토큰도 여기서 소모**됩니다. 답이 비면 값을 늘리거나 reasoning effort를 낮추세요.
- RAGAS는 내부에서 `temperature=0.01`을 강제로 설정하는데, 요청 payload 단계에서 이를 제거하도록 처리해 두었습니다. ragas 사용 통계 전송은 끕니다(`RAGAS_DO_NOT_TRACK`).
- 배포용(암호화) HWP는 본문을 읽을 수 없어 경고 후 건너뜁니다. 한글에서 일반 `.hwp`·`.hwpx`나 PDF로 저장해 넣어주세요.
- 평가셋을 직접 만들었다면 `qa_generation.external_path`에 csv/jsonl을 지정하세요 (`question, ground_truth, evidence_keywords[, doc_id]`).

## 구조
```
01_rag_build.ipynb      ★ RAG 학습: 파싱·청크·BM25·임베딩·FAISS → build_config.json (파인튜닝은 CLI main.py finetune 에만 남김)
02_generate_eval.ipynb  ★ 모델 실행·평가: 저장된 인덱스로 Generate + 응답 시간 + 종합·질문별 RAGAS (학습 없음)
03_chatbot.ipynb        ★ 챗봇: 01 결과만으로 대화 + 실시간 RAGAS (02 불필요, 파일 저장 없음)
main.py / app.py        CLI / Gradio
config.yaml             기본 설정 (노트북이 덮어씀)
eval/
  eval_questions.yaml   최종 평가 질문 세트 (reference·핵심사실·기대 라우팅)
  domain_dict.yaml      (선택·기본 미사용) 규칙 전처리 사전 — rules.enabled=true 일 때만
rag/
  parsing.py              ① 문서 파싱: PyPDFLoader·PyMuPDFLoader·HwpHwpxLoader 호출 + 정제(머리글·쪽번호·끊긴 줄·표 중복) + HWP 파서 패치
  indexing.py             ② 색인: HF 임베딩(샤드 체크포인트·OOM 복구) · FAISS(flat/hnsw) · BM25(Kiwi) · (선택) 임베딩 파인튜닝
  retrieval.py            ③ 검색: 하이브리드(BM25 × Dense) · 기관 라우팅 · SmartRetriever(멀티 쿼리 RRF) · 리랭커(CrossEncoder · rerank_room)
  rewrite.py              ① 질문 해석(QueryRewriter · REWRITE_PROMPT) · ④ 답변 GPT 에 검색용 질문 전달(install_answer_with_rewrite)
  engine.py               (호환용) 예전 import 경로 유지 — 위 세 파일의 이름을 다시 내보내기만 함
  chunk_split.py          청크 분할(정규식 구분자·overlap_ratio·섹션 경로) + QA 분할(ratio·seed·stratify·evidence_keywords)
  notebook_checkpoint.py  원자적 저장·JSONL·manifest·중지 플래그 + 노트북 표시 헬퍼
  pipeline.py             단계 오케스트레이션 · fingerprint 캐시 · build_config 저장/불러오기(generate_only) · 데이터 증가 리포트
  llm.py                  생성기(추론 강도·토큰 사용량) + RAGChat(질문 그대로 → 라우팅 → 검색 → 생성, 멀티턴)
  evaluate.py             RAGAS (reference 유무에 따라 지표 선택) + 토큰 0 지표(Key Fact·Semantic Sim)
  query_rules.py          (선택·기본 꺼짐) 규칙 전처리 — 노트북은 사용하지 않음
  hw.py / credentials.py  L4·5070 Ti 프로필, CUDA 점검 / OPENAI_API_KEY·HF_TOKEN (.env 우선)
  text_utils.py           정규화·한자·토크나이저(Kiwi 멀티스레드) + HwpHwpxLoader 출력 후처리(깨진 글자·병합셀·표지/목차 감지)
  vision.py               LangChain images_parser (hf-vlm 등) + PyMuPDF 이미지 버그 패치
  qa_gen.py / config.py / async_utils.py
```

## 👥 팀 역할과 파일 담당 (6명 · 노트북은 01·02·03 세 개만)
노트북은 셀 단원 단위로 나눠 맡고, 노트북마다 **책임자** 한 명이 셀을 합칩니다. 각 노트북의 단원 제목에 `👥 담당` 이 적혀 있습니다.

| 담당 | 노트북 셀 | 파이썬 파일 | 난이도 |
|---|---|---|---|
| PM · 통합 | 01·02·03 §0 환경 셀 · 02 §3 · §4 | `main.py` `rag/config.py` `rag/credentials.py` `rag/hw.py` · `config.yaml` `requirements.txt` | ★★ |
| 문서 파싱 | 01 §1 (1-1 · 1-2 · 1-4) | `rag/parsing.py` `rag/text_utils.py` `rag/vision.py` | ★★★★ |
| 청크 · 색인 (**01 책임자**) | 01 1-3 · §2 · §3 | `rag/indexing.py` `rag/chunk_split.py` `rag/pipeline.py` `rag/notebook_checkpoint.py` | ★★★ |
| 검색 · 리랭킹 | 02·03 1-1b ②③ · 0-2 검색 설정 · 01 2-4 | `rag/retrieval.py` | ★★★★ |
| 질문 해석 · 답변 · 챗봇 (**03 책임자**) | 03 §2 · 02·03 1-1b ①④ · 0-2 LLM 설정 | `rag/rewrite.py` `rag/llm.py` `rag/query_rules.py` `app.py` | ★★★ |
| 평가 · 데이터셋 (**02 책임자**) | 02 0-3 ~ 0-5 · 1-2 · 1-3 · §2 | `rag/evaluate.py` `rag/qa_gen.py` `rag/async_utils.py` · `eval/` | ★★★ |

- 실험은 새 노트북을 만들지 않고 세 노트북 안에서 합니다 — 청크 크기는 01 의 1-3, 검색·모델·추론 강도는 02·03 의 0-2 를 바꿔 02 를 다시 실행 (설정마다 `eval_set/<해시>/` 결과 폴더가 따로 생김).
- `rag/*.py` 를 고친 뒤에는 **커널 재시작** 후 다시 실행하세요. 단 `rewrite.py`·`retrieval.py` 는 02·03 의 1-1b 셀이 다시 불러오므로 그 셀만 다시 실행해도 반영됩니다.
- Git: `.env` · `artifacts/` 는 커밋 금지, 노트북은 출력을 지우고 커밋, 파일·셀은 담당자만 직접 수정하고 나머지는 리뷰.
