"""API 토큰 관리 — OPENAI_API_KEY 와 HF_TOKEN 을 각각 따로 다룸.

우선순위: ① 이미 설정된 환경변수 → ② 프로젝트 .env → ③ 노트북 입력창(getpass, 화면에 안 보임)
입력창으로 받은 값은 기본적으로 '메모리에만' 두고 디스크에 쓰지 않음 (save_to_env=True 일 때만 .env 저장).
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path

KEYS = {
    "OPENAI_API_KEY": {"required": True, "prefix": "sk-", "desc": "gpt-5-mini 생성·QA·RAGAS 심판"},
    "HF_TOKEN": {"required": False, "prefix": "hf_", "desc": "HuggingFace 모델 다운로드(KURE-v1, Qwen2.5-VL) 속도제한 완화"},
}


PLACEHOLDERS = {"sk-...", "hf_...", "sk-", "hf_", "your-key", "YOUR_KEY"}


def mask(v: str | None) -> str:
    if not v:
        return "(없음)"
    return f"{v[:5]}…{v[-4:]}" if len(v) > 12 else "****"


def read_env_file(env_path: Path) -> dict:
    """.env 를 인코딩 무관하게 읽기 (UTF-8 / UTF-8 BOM(메모장) / CP949(ANSI) 모두 지원).
    'KEY = value', 따옴표, 주석 모두 허용."""
    import io

    from dotenv import dotenv_values

    raw = env_path.read_bytes()
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    vals = dotenv_values(stream=io.StringIO(text))
    return {k.strip().lstrip("\ufeff"): (v or "").strip().strip('"').strip("'") for k, v in vals.items() if k}


def load_tokens(env_path: str | Path | None = None, ask: bool = True, save_to_env: bool = False,
                env_first: bool = True, required: tuple | list | None = None) -> dict:
    """env_first=True: .env 값이 시스템 환경변수보다 우선 (예전 키가 시스템에 남아 있어도 .env 사용)
    required: 꼭 있어야 하는 키 목록 (None = 기본값: OPENAI_API_KEY). 01 학습 노트북은 () — 둘 다 선택"""
    from dotenv import set_key

    env_path = Path(env_path) if env_path else None
    file_vals: dict = {}
    if env_path and env_path.exists():
        file_vals = read_env_file(env_path)
        print(f"✔ .env 읽음: {env_path}  (키: {', '.join(k for k in file_vals) or '없음'})", flush=True)
    elif env_path:
        # Windows 메모장은 '.env' 를 '.env.txt' 로 저장하는 경우가 많음
        alts = [p.name for p in env_path.parent.glob(".env*") if p.name != ".env.example"] + \
               [p.name for p in env_path.parent.glob("env*.txt")]
        print(f"⚠️ {env_path} 없음" + (f" — 비슷한 파일 발견: {alts} (이름을 정확히 '.env' 로 바꾸세요)" if alts else ""),
              flush=True)
    # OpenAI 부가 설정(선택): 여러 조직/프로젝트에 속한 개인 계정이면 .env 에 지정
    for extra in ("OPENAI_ORG_ID", "OPENAI_PROJECT_ID", "OPENAI_BASE_URL"):
        if file_vals.get(extra):
            os.environ[extra] = file_vals[extra]
    base = os.environ.get("OPENAI_BASE_URL", "")
    if base and "api.openai.com" not in base:
        print(f"⚠️ OPENAI_BASE_URL={base} 가 설정돼 있어 OpenAI 가 아닌 곳으로 요청됩니다. 개인 OpenAI 키를 쓰려면 이 변수를 지우세요.",
              flush=True)

    status = {}
    req = {k for k, m in KEYS.items() if m["required"]} if required is None else set(required)
    for k, meta in KEYS.items():
        meta = {**meta, "required": k in req}
        env_v = os.environ.get(k, "").strip().strip('"').strip("'")
        file_v = file_vals.get(k, "")
        if file_v in PLACEHOLDERS:
            print(f"⚠️ .env 의 {k} 가 예시값('{file_v}') 그대로입니다 → 실제 토큰으로 바꾸세요", flush=True)
            file_v = ""
        if env_v in PLACEHOLDERS:
            env_v = ""
        if file_v and (env_first or not env_v):
            v, src = file_v, ".env"
            if env_v and env_v != file_v:
                print(f"ℹ️ 시스템 환경변수 {k}({mask(env_v)}) 대신 .env 값({mask(file_v)})을 사용합니다", flush=True)
        elif env_v:
            v, src = env_v, "시스템 환경변수"
        else:
            v, src = "", None
        if not v and ask:
            hint = "" if meta["required"] else " (선택, 없으면 빈칸으로 Enter)"
            print(f"🔑 [{k}] 입력 대기 중{hint} — VS Code: 창 맨 위 입력칸 / Jupyter Lab: 셀 아래 입력칸에 붙여넣고 Enter",
                  flush=True)
            try:
                v = getpass.getpass(f"{k}{hint}: ").strip()
            except Exception as e:  # noqa: BLE001  입력창을 못 띄우는 환경(papermill·헤드리스 등)
                print(f"   ↳ 입력창을 열 수 없는 환경({type(e).__name__}) — .env 에 {k} 를 넣어주세요", flush=True)
                v = ""
            src = "입력" if v else None
            if v and save_to_env and env_path:
                env_path.touch(exist_ok=True)
                set_key(str(env_path), k, v)
                src = "입력→.env 저장"
        if v:
            os.environ[k] = v
            if k == "HF_TOKEN":
                os.environ["HUGGING_FACE_HUB_TOKEN"] = v  # 구버전 huggingface_hub 호환
        if not v and meta["required"]:
            raise RuntimeError(f"{k} 가 필요합니다 ({meta['desc']}). .env 에 넣거나 입력하세요.")
        kind = ""
        if k == "OPENAI_API_KEY" and v:
            kind = "프로젝트 키(sk-proj-)" if v.startswith("sk-proj-") else (
                "서비스 계정 키" if v.startswith("sk-svcacct-") else "개인/레거시 키")
        warn = v and not v.startswith(meta["prefix"])
        status[k] = {"값": mask(v), "출처": src or "-", "종류": kind, "용도": meta["desc"],
                     "형식": "⚠️ 접두사 확인" if warn else ("OK" if v else "-")}
    return status


class _Skip(Exception):
    pass


def _with_timeout(fn, timeout: float):
    """응답 없는 네트워크 호출이 셀을 붙잡지 않도록 제한 시간 적용"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout

    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except FTimeout:
        raise TimeoutError(f"{timeout:.0f}초 동안 응답 없음 — 네트워크/방화벽/프록시 확인") from None
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def verify_tokens(llm_model: str = "gpt-5-mini", timeout: float = 15) -> dict:
    """가벼운 호출로 실제 유효성 확인 (과금 없음: 모델 조회 / whoami). 각 최대 timeout 초."""
    import time

    out = {}
    t0 = time.time()
    if not os.environ.get("OPENAI_API_KEY"):
        out["OPENAI_API_KEY"] = "— 미설정 (02 평가 노트북에서 필요)"
    print("… OpenAI 키 확인 중" if "OPENAI_API_KEY" not in out else "… OpenAI 키 없음 — 건너뜀", flush=True)
    try:
        if "OPENAI_API_KEY" in out:
            raise _Skip
        import openai

        _with_timeout(lambda: openai.OpenAI(timeout=timeout, max_retries=0).models.retrieve(llm_model), timeout + 5)
        out["OPENAI_API_KEY"] = f"✅ 유효 · {llm_model} 접근 가능"
    except _Skip:
        pass
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        hint = {"AuthenticationError": "키가 잘못됨(복사 누락·만료·폐기 확인)",
                "NotFoundError": f"{llm_model} 접근 권한 없음 → platform.openai.com 프로젝트 설정의 Allowed models 확인",
                "PermissionDeniedError": "프로젝트 권한(Restricted 키) 확인 — Model capabilities: Write 필요",
                "RateLimitError": "크레딧/결제 수단 확인 (Billing)", "APIConnectionError": "네트워크 확인",
                "TimeoutError": str(e)}.get(name, "")
        out["OPENAI_API_KEY"] = f"❌ {name} {hint}".strip()
    print(f"  ({time.time() - t0:.1f}s)", flush=True)
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        out["HF_TOKEN"] = "— 미설정 (공개 모델은 토큰 없이도 받을 수 있음)"
        return out
    t0 = time.time()
    print("… HF 토큰 확인 중", flush=True)
    try:
        from huggingface_hub import whoami

        name = _with_timeout(lambda: whoami(token=tok).get("name"), timeout)
        out["HF_TOKEN"] = f"✅ 유효 · 사용자 {name}"
    except Exception as e:  # noqa: BLE001
        out["HF_TOKEN"] = f"❌ {type(e).__name__}: {str(e)[:80]}"
    print(f"  ({time.time() - t0:.1f}s)", flush=True)
    return out
