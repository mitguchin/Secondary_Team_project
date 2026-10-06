from __future__ import annotations

import copy
import os

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        set_by_path(cfg, *ov.split("=", 1))
    return cfg


def set_by_path(cfg: dict, dotted: str, raw_value):
    """'split.seed', '7' → cfg['split']['seed'] = 7 (YAML 문법으로 해석)"""
    value = yaml.safe_load(raw_value) if isinstance(raw_value, str) else raw_value
    node = cfg
    keys = dotted.strip().split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def merged(cfg: dict, section: str, **kw) -> dict:
    """UI 에서 넘어온 값으로 섹션을 덮어쓴 사본"""
    c = copy.deepcopy(cfg)
    c[section].update({k: v for k, v in kw.items() if v is not None})
    return c
