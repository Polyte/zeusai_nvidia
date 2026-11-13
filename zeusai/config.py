import copy
import os
import json
import pathlib

# Internal values - never displayed as provider in UI (presentation name is zeusai)
BASE_URL = "https://integrate.api.nvidia.com/v1"
DISPLAY_MODEL = "zeusai-1"

# Built-in profiles: alias -> internal config (api_key filled from disk or env)
DEFAULT_PROFILES = {
    "laguna": {
        "model_id": "poolside/laguna-xs-2.1",
        "api_key": "",
        "base_url": BASE_URL,
        "thinking": False,
        "vision": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "context": 256000,
        "description": "Fast, good for coding tasks",
    },
    "ultra": {
        "model_id": "nvidia/nemotron-3-ultra-550b-a55b",
        "api_key": "",
        "base_url": BASE_URL,
        "thinking": True,
        "vision": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384,
        "context": 1000000,
        "description": "Deep reasoning, thinking enabled",
    },
    "super": {
        "model_id": "nvidia/nemotron-3.5-super-vl-120b-a12b",
        "api_key": "",
        "base_url": BASE_URL,
        "thinking": True,
        "vision": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 16384,
        "context": 128000,
        "description": "Vision + thinking, image analysis",
    },
    "glimmer": {
        "model_id": "meta/muse-glimmer-30b",
        "api_key": "",
        "base_url": BASE_URL,
        "thinking": False,
        "vision": False,
        "reasoning_effort": "high",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 8192,
        "context": 128000,
        "description": "High reasoning effort, math + logic",
    },
    "paligemma": {
        "model_id": "google/paligemma",
        "api_key": "",
        "base_url": "https://ai.api.nvidia.com/v1/vlm/google/paligemma",
        "type": "vlm",
        "thinking": False,
        "vision": True,
        "temperature": 1.0,
        "top_p": 0.7,
        "max_tokens": 512,
        "context": 4096,
        "description": "VLM image description (needs base64 image)",
    },
}

# Ensemble speed knobs (override under "ensemble" in ~/.zeusai/config.json)
ENSEMBLE_DEFAULTS = {
    "draft_thinking": False,    # drafts skip chain-of-thought: biggest latency cut
    "draft_max_tokens": 2048,
    "grace_seconds": 8,         # after the first good draft, wait for the rest (adaptive, capped here)
    "judge_thinking": False,
    "skip_judge_if_single": True,
    "models": None,             # comma list or array of aliases to fan out to; None = all with keys
    "judge": "ultra",           # used when --judge is not given
}

def get_ensemble_settings():
    return {**ENSEMBLE_DEFAULTS, **(_load_raw().get("ensemble") or {})}

def ensemble_judge(judge=None):
    return judge or get_ensemble_settings().get("judge") or "ultra"

def get_config_dir():
    return pathlib.Path.home() / ".zeusai"

def get_config_path():
    return get_config_dir() / "config.json"

_raw_cache = {"mtime": None, "data": {}}

def _load_raw():
    # cached by mtime: this is called on every model call, so avoid re-parsing
    p = get_config_path()
    try:
        mtime = p.stat().st_mtime_ns
    except OSError:
        return {}
    if _raw_cache["mtime"] != mtime:
        try:
            _raw_cache["data"] = json.loads(p.read_text())
        except:
            _raw_cache["data"] = {}
        _raw_cache["mtime"] = mtime
    return copy.deepcopy(_raw_cache["data"])

def _write_raw(data):
    d = get_config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = get_config_path()
    p.write_text(json.dumps(data, indent=2))
    _raw_cache["mtime"] = None
    try:
        p.chmod(0o600)
    except:
        pass

def get_profiles():
    """Merge defaults with disk config + env"""
    raw = _load_raw()
    # start from defaults (deep copy)
    profiles = copy.deepcopy(DEFAULT_PROFILES)

    # overlay disk profiles
    disk_profiles = raw.get("profiles") or {}
    for name, prof in disk_profiles.items():
        base = profiles.get(name, {})
        merged = {**base, **prof}
        profiles[name] = merged

    # legacy single-key support: if raw has api_key/model, treat as laguna
    if raw.get("api_key"):
        if "laguna" not in profiles:
            profiles["laguna"] = copy.deepcopy(DEFAULT_PROFILES["laguna"])
        # only fill if empty
        if not profiles["laguna"].get("api_key"):
            profiles["laguna"]["api_key"] = raw["api_key"]
        if raw.get("model"):
            profiles["laguna"]["model_id"] = raw["model"]

    # env overrides
    if os.getenv("ZEUSAI_API_KEY"):
        # apply to all? apply to default profile
        default_name = raw.get("default") or "laguna"
        if default_name in profiles:
            profiles[default_name]["api_key"] = os.getenv("ZEUSAI_API_KEY")
        else:
            profiles["laguna"]["api_key"] = os.getenv("ZEUSAI_API_KEY")
    if os.getenv("NVIDIA_API_KEY"):
        default_name = raw.get("default") or "laguna"
        if default_name in profiles:
            profiles[default_name]["api_key"] = os.getenv("NVIDIA_API_KEY")

    # per-profile env e.g. ZEUSAI_LAGUNA_API_KEY etc not needed

    return profiles

def get_default_profile_name():
    raw = _load_raw()
    return raw.get("default") or (raw.get("model_profile") or "auto")

# "auto": one fast model per prompt instead of fan-out + judge (one round trip).
# vision prompts go to the vision model, everything else to the strongest coder.
AUTO_ROUTES = {"vision": "super", "default": "ultra"}  # ultra: 10/10 on the coding eval vs laguna 6/10

def resolve_auto(name=None, has_image=False):
    """Map None/"auto"/"ensemble" defaults to a concrete single profile."""
    if name is None:
        name = get_default_profile_name()
    if name in ("auto", "ensemble"):
        return AUTO_ROUTES["vision" if has_image else "default"]
    return name

def get_active_profile(name=None):
    profiles = get_profiles()
    if name is None:
        name = get_default_profile_name()
    if name in ("auto", "ensemble"):
        name = resolve_auto(name)
    # allow passing full model_id as name
    if name not in profiles:
        # try to find by model_id
        for alias, prof in profiles.items():
            if prof.get("model_id") == name:
                return alias, {**prof, "alias": alias}
        # unknown -> create ad-hoc profile using default key?
        raise SystemExit(f"Unknown model/profile: {name}. Available: {', '.join(profiles.keys())}")
    prof = profiles[name]
    if not prof.get("api_key") and not prof.get("no_key"):
        raise SystemExit(f"No API key for profile '{name}'. Run: zeusai configure --profile {name} --api-key <key>")
    return name, {**prof, "alias": name}

def list_profiles():
    profiles = get_profiles()
    default = get_default_profile_name()
    out = []
    for alias, prof in profiles.items():
        key = prof.get("api_key") or ""
        masked = (key[:8] + "..." + key[-4:]) if len(key) > 12 else ("***" if key else "(not set)")
        out.append({
            "alias": alias,
            "model_id": prof.get("model_id"),
            "masked_key": masked,
            "thinking": prof.get("thinking", False),
            "vision": prof.get("vision", False),
            "is_default": alias == default,
            "description": prof.get("description", ""),
        })
    return out

def save_profile(alias, api_key=None, model_id=None, base_url=None, thinking=None, vision=None, no_key=None):
    raw = _load_raw()
    profiles = raw.get("profiles") or {}
    cur = profiles.get(alias) or {}
    if api_key:
        cur["api_key"] = api_key
    if model_id:
        cur["model_id"] = model_id
    if base_url:
        cur["base_url"] = base_url
    if thinking is not None:
        cur["thinking"] = thinking
    if vision is not None:
        cur["vision"] = vision
    if no_key is not None:
        cur["no_key"] = no_key
    profiles[alias] = cur
    raw["profiles"] = profiles
    if not raw.get("default"):
        raw["default"] = alias if alias in DEFAULT_PROFILES else "laguna"
    _write_raw(raw)
    return raw

def set_default(alias):
    raw = _load_raw()
    raw["default"] = alias
    _write_raw(raw)

def configure_legacy(api_key=None, model=None, base_url=None):
    """Backward compat: configure default profile"""
    default_name = get_default_profile_name()
    save_profile(default_name, api_key=api_key, model_id=model, base_url=base_url)

# Backwards compat for old code
def load_config(name=None):
    alias, prof = get_active_profile(name)
    # return flat dict for old callers
    return {
        "api_key": prof["api_key"],
        "base_url": prof["base_url"],
        "model": prof["model_id"],
        "model_id": prof["model_id"],
        "alias": alias,
        "thinking": prof.get("thinking", False),
        "vision": prof.get("vision", False),
        "temperature": prof.get("temperature", 0.6),
        "top_p": prof.get("top_p", 0.95),
        "max_tokens": prof.get("max_tokens", 8192),
    }

def save_config(key=None, model=None, base_url=None, profile=None):
    # if profile given, save there, else default
    alias = profile or get_default_profile_name()
    save_profile(alias, api_key=key, model_id=model, base_url=base_url)
