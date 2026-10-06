"""하드웨어 감지 + 자동 튜닝 프로필 (로컬 RTX 5070 Ti / 원격 L4 / CPU).

■ 로컬 RTX 5070 Ti (Blackwell sm_120, 16GB GDDR7, bf16/fp16/TF32)
  - PyTorch 는 반드시 CUDA 12.8+ 빌드(cu128 이상). cu124 이하에는 sm_120 커널이 없어 'no kernel image' 오류
  - 데스크톱은 화면·브라우저가 VRAM 을 1~2GB 쓰므로 여유를 둠
  - 임베딩(KURE-v1 fp16 ≈ 1.1GB) batch 32 / 파인튜닝 bf16 논리배치 64 · mini 8 (AdamW 상태 포함 ≈ 11~12GB)
  - VLM(Qwen2.5-VL-3B bf16 ≈ 7.5GB) 은 파싱 때만 올리고 해제, 이미지 해상도 상한 768·28·28
■ 원격 L4 24GB : 임베딩 batch 64, mini 16, 해상도 1024·28·28
■ Windows      : DataLoader worker 0 (노트북+spawn 이슈 회피), expandable_segments 미지원이라 설정 안 함
"""
from __future__ import annotations

import os
import platform


def _ram_gb() -> float:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                if ln.startswith("MemTotal"):
                    return round(int(ln.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    return 0.0


def cpu_model() -> str:
    """CPU 모델명 (Linux /proc/cpuinfo · Windows 레지스트리 · macOS sysctl)"""
    import subprocess

    try:
        sysname = platform.system()
        if sysname == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if ln.lower().startswith("model name"):
                        return ln.split(":", 1)[1].strip()
        elif sysname == "Windows":
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        elif sysname == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "알 수 없음"


def _physical_cores():
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        return None


def nvidia_smi() -> dict:
    """nvidia-smi 로 드라이버 버전·GPU 이름·메모리 (없으면 {})"""
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        name, drv, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {"name": name, "driver": drv, "mem_mb": int(float(mem))}
    except Exception:  # noqa: BLE001
        return {}


def cuda_smoke_test() -> tuple[bool, str]:
    """is_available() 만으로는 부족 — 실제 커널을 돌려 'no kernel image' 여부 확인"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "CUDA 사용 불가 (CPU 로 실행)"
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        (a @ a).sum().item()
        if torch.cuda.is_bf16_supported():
            b = a.to(torch.bfloat16)
            (b @ b).float().sum().item()
        torch.nn.functional.softmax(a.float(), dim=-1).sum().item()
        return True, "CUDA 커널 실행 OK (fp16/bf16 matmul, softmax)"
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0][:160]
        if "no kernel image" in msg or "sm_120" in msg:
            msg += "  → Blackwell(RTX 50) 은 cu128 이상 PyTorch 필요: 0-0 셀 참고"
        return False, msg


def detect() -> dict:
    info = {"os": platform.system(), "os_release": platform.release(), "python": platform.python_version(),
            "cpu_model": cpu_model(), "cpu_cores": _physical_cores(), "cpus": os.cpu_count() or 1,
            "ram_gb": _ram_gb(), "gpu": None, "vram_gb": 0.0, "bf16": False, "capability": None, "arch_ok": None}
    smi = nvidia_smi()
    info["gpu_driver"] = smi.get("driver")
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        try:
            info["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            info["cudnn"] = None
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cap = f"sm_{p.major}{p.minor}"
            archs = torch.cuda.get_arch_list()
            info.update(gpu=p.name, vram_gb=round(p.total_memory / 1024**3, 1),
                        bf16=torch.cuda.is_bf16_supported(), capability=cap,
                        arch_ok=cap in archs or any(a.startswith(f"sm_{p.major}") for a in archs),
                        blackwell=p.major >= 10)
            import time

            print("… GPU 커널 점검 중 (첫 실행은 CUDA 초기화로 수 초~수십 초)", flush=True)
            t0 = time.time()
            info["kernel_ok"], info["kernel_msg"] = cuda_smoke_test()
            info["kernel_msg"] += f" ({time.time() - t0:.1f}s)"
    except ImportError:
        info["torch"] = None
    return info


def profile_name(info: dict) -> str:
    if not info.get("gpu") or info.get("kernel_ok") is False:
        return "cpu"
    v = info["vram_gb"]
    if v >= 20:
        return "gpu-24gb (L4/A10/4090)"
    if v >= 14:
        return "gpu-16gb (RTX 5070 Ti/4080/T4)" if not info.get("blackwell") else "gpu-16gb Blackwell (RTX 5070 Ti)"
    return "gpu-small (<14GB)"


def profile_overrides(info: dict) -> dict:
    """감지 결과 → config 덮어쓰기 값 (dotted key → value)"""
    vram, ram, cpus = info["vram_gb"], info["ram_gb"], info["cpus"]
    windows = info.get("os") == "Windows"
    usable = bool(info.get("gpu")) and info.get("kernel_ok") is not False
    o: dict = {}
    if not usable:
        o.update({"embedding.device": "cpu", "embedding.batch_size": 8, "embedding.fp16": False,
                  "parsing.image_parser": None, "finetune.enabled": False})
    elif vram >= 20:  # 원격 L4 24GB 등
        o.update({"embedding.device": "cuda", "embedding.batch_size": 64, "embedding.fp16": True,
                  "embedding.shard_every_batches": 10,
                  "finetune.batch_size": 64, "finetune.mini_batch_size": 16,
                  "parsing.image_parser": "hf-vlm", "parsing.vlm_max_pixels": 1024 * 28 * 28})
    elif vram >= 14:  # 로컬 RTX 5070 Ti 16GB (데스크톱 화면 사용분 고려)
        o.update({"embedding.device": "cuda", "embedding.batch_size": 32, "embedding.fp16": True,
                  "embedding.shard_every_batches": 10,
                  "finetune.batch_size": 64, "finetune.mini_batch_size": 8,
                  "finetune.gradient_checkpointing": False,
                  "parsing.image_parser": "hf-vlm", "parsing.vlm_max_pixels": 768 * 28 * 28})
    else:
        o.update({"embedding.device": "cuda", "embedding.batch_size": 16, "embedding.fp16": True,
                  "finetune.batch_size": 32, "finetune.mini_batch_size": 4,
                  "finetune.gradient_checkpointing": True, "parsing.image_parser": None})
    if usable:
        o["parsing.vlm_model"] = "Qwen/Qwen2.5-VL-3B-Instruct"
    o["finetune.dataloader_num_workers"] = 0 if windows else max(0, min(4, cpus - 2))
    o["qa_generation.max_workers"] = max(2, min(4, cpus))  # API 동시 요청 (rate limit 고려)
    if ram and ram <= 16:
        o["parsing.vlm_max_pixels"] = min(o.get("parsing.vlm_max_pixels", 1024 * 28 * 28), 1024 * 28 * 28)
    return o


def apply_runtime_env(cache_dir: str | None = None):
    """토크나이저 fork 경고 방지, CUDA 메모리 단편화 완화(Linux), TF32 활성화, HF 캐시 경로"""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if platform.system() == "Linux":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cache_dir:
        os.environ["HF_HOME"] = str(cache_dir)
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass


def gpu_mem() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "GPU 없음"
        a = torch.cuda.memory_allocated() / 1024**3
        r = torch.cuda.memory_reserved() / 1024**3
        free, total = torch.cuda.mem_get_info()
        return (f"GPU 메모리: 이 프로세스 {a:.2f}GB(예약 {r:.2f}GB) · 장치 여유 {free / 1024**3:.1f}"
                f" / {total / 1024**3:.1f}GB")
    except Exception:  # noqa: BLE001
        return "GPU 정보 없음"


def _cudnn_str(v):
    if not v:
        return "-"
    v = int(v)
    return f"{v // 10000}.{v // 100 % 100}.{v % 100}" if v >= 10000 else f"{v // 1000}.{v // 100 % 10}.{v % 100}"


def hw_rows(info: dict) -> dict:
    """노트북 표용: 지금 쓰는 CPU·GPU·소프트웨어 버전"""
    cores = f"물리 {info['cpu_cores']}코어 / " if info.get("cpu_cores") else ""
    return {
        "CPU": info.get("cpu_model") or "-",
        "CPU 코어·스레드": f"{cores}논리 {info['cpus']}스레드",
        "RAM": f"{info['ram_gb']} GB",
        "GPU": info.get("gpu") or "없음 (CPU 실행)",
        "GPU 메모리": f"{info['vram_gb']} GB" if info.get("gpu") else "-",
        "GPU 아키텍처": f"{info.get('capability')} · bf16 {'지원' if info.get('bf16') else '미지원'}" if info.get("gpu") else "-",
        "NVIDIA 드라이버": info.get("gpu_driver") or "-",
        "PyTorch": str(info.get("torch") or "-"),
        "CUDA (torch 빌드)": str(info.get("cuda") or "-"),
        "cuDNN": _cudnn_str(info.get("cudnn")),
        "Python": info.get("python") or platform.python_version(),
        "OS": f"{info['os']} {info.get('os_release', '')}".strip(),
        "튜닝 프로필": profile_name(info),
    }


def hw_table(info: dict):
    import pandas as pd

    return pd.DataFrame({"값": hw_rows(info)})


def summary(info: dict) -> str:
    r = hw_rows(info)
    s = (f"CPU : {r['CPU']} · {r['CPU 코어·스레드']} · RAM {r['RAM']}\n"
         f"GPU : {r['GPU']}" + (f" · {r['GPU 메모리']} · {r['GPU 아키텍처']} · 드라이버 {r['NVIDIA 드라이버']}" if info.get("gpu") else "")
         + f"\nSW  : Python {r['Python']} · torch {r['PyTorch']} (CUDA {r['CUDA (torch 빌드)']}, cuDNN {r['cuDNN']}) · {r['OS']}"
         f"\n프로필: {r['튜닝 프로필']}")
    if info.get("gpu"):
        s += f"\n커널 점검: {'✅' if info.get('kernel_ok') else '❌'} {info.get('kernel_msg')}"
        if info.get("arch_ok") is False:
            s += f"\n⚠️ 이 torch 빌드에 {info['capability']} 가 없습니다 → 0-0 셀의 cu128 설치를 실행하세요"
    return s


def require_gpu(info: dict):
    """로컬 GPU(5070 Ti) 사용을 강제 — 못 쓰면 CPU 로 조용히 넘어가지 않고 원인과 해결법을 알려주며 중단"""
    if info.get("gpu") and info.get("kernel_ok"):
        return
    tv = str(info.get("torch") or "")
    if not tv:
        why = "torch 가 설치되어 있지 않습니다 → 0-0 셀 실행 후 Kernel ▸ Restart"
    elif tv.endswith("+cpu") or info.get("cuda") is None:
        why = (f"CPU 전용 torch({tv})가 설치되어 있습니다. Windows 에서 'pip install torch' 만 하면 CPU 빌드가 깔립니다.\n"
               "   → 0-0 셀을 실행해 cu128 빌드로 교체한 뒤 Kernel ▸ Restart")
    elif not info.get("gpu"):
        why = (f"torch({tv}, CUDA {info.get('cuda')})는 GPU 빌드인데 GPU 를 찾지 못했습니다.\n"
               "   → 'nvidia-smi' 로 드라이버(570 이상) 확인, 노트북 커널이 올바른 가상환경(.venv)인지 확인")
    else:
        why = (f"GPU({info['gpu']})는 보이지만 커널 실행 실패: {info.get('kernel_msg')}\n"
               "   → RTX 50 시리즈는 cu128 이상 torch 필요: 0-0 셀 실행 후 Kernel ▸ Restart")
    raise RuntimeError("로컬 GPU 를 사용할 수 없습니다.\n   " + why)
