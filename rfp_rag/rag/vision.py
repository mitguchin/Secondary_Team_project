"""(선택) 그래프·도표 이미지 → 텍스트.  LangChain 의 `images_parser`(BaseImageBlobParser) 형식으로 제공.

PDF : PyMuPDFLoader(extract_images=True, images_parser=<여기서 만든 파서>) 가 페이지 이미지를 넘겨줌
HWP : HwpHwpxLoader(include_images=True, image_document_mode="save_and_reference") 가 저장한 이미지를 같은 파서의 _analyze_image() 로 처리
파서 종류(노트북 IMAGE_PARSER):
  "hf-vlm"   : HuggingFace VLM (기본 Qwen/Qwen2.5-VL-3B-Instruct, 로컬 GPU, 무료)
  "rapidocr" : LangChain RapidOCRBlobParser (CPU OCR, 이미지 속 글자만)
  "tesseract": LangChain TesseractBlobParser (kor+eng)
  "llm"      : LangChain LLMImageBlobParser (gpt-5-mini 등 멀티모달 LLM, API 비용)
  None       : 이미지 무시

한국어 차트/표/다이어그램을 텍스트로 풀어 청크로 넣어주면, 이미지 속 수치도 검색·답변에 쓰입니다.
캡션은 이미지 해시 단위로 JSONL 에 저장 → 중간에 꺼져도 이미 만든 캡션은 다시 만들지 않습니다.
"""
from __future__ import annotations

import hashlib
import io

from .notebook_checkpoint import JsonlStore

PROMPT = (
    "이 이미지는 공공기관 제안요청서(RFP)의 일부입니다.\n"
    "- 그래프/차트이면: 제목, 축, 범례, 핵심 수치와 추세를 한국어로 정리\n"
    "- 표이면: Markdown 표로 옮겨 적기\n"
    "- 조직도/개념도/흐름도이면: 구성요소와 관계를 목록으로 설명\n"
    "- 로고, 사진, 장식 이미지이면 'SKIP' 한 단어만 출력"
)


def image_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


class ImageCaptioner:
    def __init__(self, model_name: str, cache_path: str, max_new_tokens: int = 384, log=print,
                 max_pixels: int = 1024 * 28 * 28):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self.cache = JsonlStore(cache_path, key="hash")
        self.done = self.cache.load_map()
        self.log = log
        self._model = None
        self._proc = None
        self.disabled = False  # 모델 로드 실패 시 이후 이미지는 건너뜀

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:  # 구버전 transformers
            from transformers import Qwen2_5_VLForConditionalGeneration as AutoVLM
        self.log(f"[vlm] loading {self.model_name} ... (최초 1회 다운로드)")
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # L4 = bf16 지원
        else:
            dtype = torch.float32
        import transformers
        from packaging.version import Version

        dkey = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        self._model = AutoVLM.from_pretrained(self.model_name, device_map="auto", **{dkey: dtype})
        self._model.eval()
        # max_pixels: 이미지 1장당 비전 토큰 상한 (L4 24GB 에서 1024*28*28 ≈ 1024 토큰이면 여유)
        #   transformers 4.49+ 는 size={'shortest_edge': 최소 픽셀, 'longest_edge': 최대 픽셀} 로 전달
        #   (min_pixels/max_pixels 인자는 v5.23 에서 삭제 예정이라 FutureWarning 이 남)
        self._proc = AutoProcessor.from_pretrained(
            self.model_name, size={"shortest_edge": 256 * 28 * 28, "longest_edge": int(self.max_pixels)})

    def caption(self, pil_image, key: str) -> str:
        if key in self.done:
            return self.done[key]["caption"]
        if self.disabled:
            return ""
        import torch

        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            self.disabled = True
            self.log(f"[vlm] ⚠️ 모델 로드 실패 → 이번 실행에서는 캡션을 건너뜁니다: {type(e).__name__}: {e}")
            return ""
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = pil_image
        for attempt in range(2):
            try:
                inputs = self._proc(text=[text], images=[img], return_tensors="pt").to(self._model.device)
                with torch.inference_mode():
                    out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                cap = self._proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == 1:
                    self.log(f"[vlm] OOM — 이미지 건너뜀 ({key})")
                    cap = "SKIP"
                    break
                img = img.resize((max(1, img.width // 2), max(1, img.height // 2)))  # 해상도 낮춰 재시도
        cap = cap.strip()
        if cap.upper().startswith("SKIP"):
            cap = ""
        row = {"hash": key, "caption": cap}
        self.cache.append(row)
        self.done[key] = row
        return cap

    def close(self):
        """파싱이 끝나면 VLM 을 GPU 에서 내려 임베딩 단계에 메모리를 돌려줌"""
        if self._model is not None:
            import gc

            import torch

            self._model = None
            self._proc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log("[vlm] GPU 메모리 해제")


def pil_from_bytes(data: bytes, ext: str):
    from PIL import Image

    if ext.lower() in ("wmf", "emf"):
        return None  # PIL 미지원 벡터 포맷
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None



# ------------------------------------------------------------------ LangChain images_parser 어댑터
from langchain_community.document_loaders.parsers.images import BaseImageBlobParser  # noqa: E402


class HFVisionImageParser(BaseImageBlobParser):
    """HF VLM 캡셔너를 LangChain images_parser 로 감싼 것. 같은 이미지는 해시 캐시로 1번만 처리(로고 반복 대비)."""

    def __init__(self, captioner: ImageCaptioner, min_px: int = 180):
        super().__init__()
        self.captioner, self.min_px = captioner, min_px

    def _analyze_image(self, img) -> str:
        if max(img.size) < self.min_px:
            return ""
        img = img.convert("RGB")
        return self.captioner.caption(img, image_hash(img.tobytes()))

    def close(self):
        self.captioner.close()


def make_image_parser(kind: str | None, pcfg: dict, cache_path: str, log=print):
    if not kind or kind == "none":
        return None
    if kind == "hf-vlm":
        cap = ImageCaptioner(pcfg.get("vlm_model", "Qwen/Qwen2.5-VL-3B-Instruct"), cache_path,
                             int(pcfg.get("vlm_max_new_tokens", 384)), log, int(pcfg.get("vlm_max_pixels", 1024 * 28 * 28)))
        return HFVisionImageParser(cap, int(pcfg.get("min_image_px", 180)))
    from langchain_community.document_loaders.parsers.images import (LLMImageBlobParser, RapidOCRBlobParser,
                                                                      TesseractBlobParser)
    if kind == "rapidocr":
        return RapidOCRBlobParser()
    if kind == "tesseract":
        return TesseractBlobParser(langs=("kor", "eng"))
    if kind == "llm":
        from .llm import make_judge_llm

        return LLMImageBlobParser(model=make_judge_llm(pcfg.get("image_llm_model", "gpt-5-mini"), 2048, "low"))
    raise ValueError(f"알 수 없는 image_parser: {kind}")


def patch_pymupdf_image_extraction():
    """langchain-community 0.3.x PyMuPDFParser 버그 수정(0.4.2 코드와 동일):
    이미지를 버퍼에 저장하기 '전에' 빈 버퍼인지 검사해서 모든 이미지를 건너뛰던 문제 → 저장 후 검사."""
    import io as _io

    import numpy as _np
    from langchain_community.document_loaders.parsers import pdf as P

    if getattr(P.PyMuPDFParser, "_rag_patched", False):
        return

    def _extract_images_from_page(self, doc, page) -> str:
        if not self.images_parser:
            return ""
        import pymupdf

        images = []
        for img in page.get_images():
            try:
                pix = pymupdf.Pixmap(doc, img[0])
                if pix.n - pix.alpha >= 4:  # CMYK → RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                arr = _np.frombuffer(pix.samples, dtype=_np.uint8).reshape(pix.height, pix.width, -1)
            except Exception:  # noqa: BLE001  손상·특수 인코딩 이미지는 건너뜀
                continue
            buf = _io.BytesIO()
            _np.save(buf, arr)
            if buf.getbuffer().nbytes == 0:
                continue
            blob = P.Blob.from_data(buf.getvalue(), mime_type="application/x-npy")
            text = next(self.images_parser.lazy_parse(blob)).page_content
            images.append(P._format_inner_image(blob, text, self.images_inner_format))
        return P._FORMAT_IMAGE_STR.format(image_text=P._JOIN_IMAGES.join(filter(None, images)))

    P.PyMuPDFParser._extract_images_from_page = _extract_images_from_page
    P.PyMuPDFParser._rag_patched = True
