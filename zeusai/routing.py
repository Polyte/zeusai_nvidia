"""Intelligent model routing: capability + cost aware selection with provider fallback.

Every request gets an ordered list of candidate profiles (primary first, then
capability- and cost-sorted fallbacks). When a model rate-limits (HTTP 429),
the agent loop tries the next candidate instead of failing the turn, and the
rate-limited profile is put on a cooldown (circuit breaker) so later prompts
skip it until it recovers.

Capability matching uses the profile schema: vision models for images,
thinking models for deep tasks, chat models for tool use (VLM profiles like
paligemma can't take tools), and the configured context windows. Cost
optimisation is a relative cost table (cheap/fast models first for trivial
work). All numbers are tunable here, not on the hot path.
"""
import threading
import time

from .config import get_profiles

# Relative cost units (cheap first). Used to order fallbacks and to pick the
# cheapest model that still fits the task. Includes every built-in profile;
# custom profiles default to the middle tier.
COST = {"laguna": 10, "paligemma": 10, "glimmer": 20, "super": 40, "ultra": 100, "ensemble": 999}

COOLDOWN_BASE = 60.0   # seconds before retrying a rate-limited profile
COOLDOWN_CAP = 900.0   # exponential backoff never exceeds this

# Circuit-breaker state per profile alias: consecutive limits and when to retry.
_state: dict[str, dict] = {}
_lock = threading.Lock()

_HARD_WORDS = (
    "debug", "fix", "refactor", "review", "implement", "complete", "optim",
    "merge conflict", "stack trace", "exception", "traceback", "regex", "sql",
    "migration", "algorithm", "performance", "slow", "test fail", "segfault",
    "deadlock", "memory leak", "race condition", "compile", "build fail",
    "broken", "wrong", "explain", "architect", "design", "plan", "spec",
    "security", "threat", "vulnerab", "deploy", "kubernetes", "docker",
    "ci/cd", "production", "database", "transaction",
)


def cost(alias: str) -> int:
    return COST.get(alias, 30)


def classify(messages, mode=None) -> dict:
    """Task hints from the conversation: image?, trivial?, complex?

    Cheap heuristics on the last user prompt (length + hard keywords). Fast
    mode is always trivial (handled by its laguna preset anyway).
    """
    text = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            text = c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c or [] if isinstance(p, dict))
            break
    text = (text or "").lower()
    hard = sum(1 for w in _HARD_WORDS if w in text)
    return {
        "has_image": any(m.get("images") for m in messages or [] if isinstance(m, dict)),
        "trivial": len(text) < 160 and hard == 0,
        "complex": hard >= 2 or len(text) > 900,
    }


def usable_profiles() -> dict:
    """Configured profiles that could actually answer (key present or keyless)."""
    profs = get_profiles()
    return {a: c for a, c in profs.items() if c.get("api_key") or c.get("no_key")}


def _order(has_image=False, complex=False, trivial=False) -> list[str]:
    """Best-first order for each task shape (cost + capability aware)."""
    if has_image:
        # vision models first; chat models can still consume described images
        return ["super", "paligemma", "ultra", "glimmer", "laguna"]
    if complex:
        return ["ultra", "glimmer", "super", "laguna", "paligemma"]
    if trivial:
        return ["laguna", "glimmer", "ultra", "super", "paligemma"]
    return ["ultra", "glimmer", "super", "laguna", "paligemma"]


def candidates(primary=None, has_image=False, trivial=False, complex=False, tools=True) -> tuple[list[str], list[str]]:
    """Ordered fallback chain for one model call: (candidates, skipped_as_limited).

    The requested primary always leads (an explicit --model or a mode preset),
    followed by the capability/cost order. VLM profiles are dropped when tools
    are in play. Limited profiles are skipped; if every candidate is limited,
    the full chain comes back so the caller can still wait + retry.
    """
    usable = usable_profiles()
    chain = [a for a in _order(has_image=has_image, complex=complex, trivial=trivial)
             if a in usable]
    if tools:
        chain = [a for a in chain if usable[a].get("type") != "vlm"]
    if primary:
        if primary in chain:
            chain.remove(primary)
            chain.insert(0, primary)
        elif primary in usable:
            chain.insert(0, primary)
    if not chain and primary:
        chain = [primary]
    skipped = [a for a in chain if is_cooling(a)]
    live = [a for a in chain if a not in skipped]
    if live:
        return live, skipped
    # every candidate is cooling (or only the primary exists): return the chain
    return chain, skipped


def best_for(primary=None, has_image=False, trivial=False, complex=False, tools=True) -> str | None:
    """The single best profile right now (first non-limited candidate)."""
    chain, _ = candidates(primary=primary, has_image=has_image, trivial=trivial,
                          complex=complex, tools=tools)
    return chain[0] if chain else primary


# ---------------------------------------------------------------- circuit breaker

def is_cooling(alias: str) -> bool:
    with _lock:
        s = _state.get(alias)
        return bool(s and s["until"] > time.monotonic())


def mark_limited(alias: str):
    """Note a 429 from this profile; exponential cooldown (60s → 900s)."""
    with _lock:
        s = _state.setdefault(alias, {"fails": 0, "until": 0.0})
        s["fails"] += 1
        s["until"] = time.monotonic() + min(COOLDOWN_CAP, COOLDOWN_BASE * (2 ** (s["fails"] - 1)))


def record_ok(alias: str):
    """A successful call clears the limited streak."""
    with _lock:
        s = _state.get(alias)
        if s:
            s["fails"] = 0
            s["until"] = 0.0


def cooldown_left(alias: str) -> float:
    with _lock:
        s = _state.get(alias)
        if not s:
            return 0.0
        return max(0.0, s["until"] - time.monotonic())


def state(alias: str) -> dict:
    """Snapshot of a profile's routing state (for status displays)."""
    with _lock:
        s = dict(_state.get(alias) or {})
    s["cooling"] = is_cooling(alias)
    s["cooldown_left"] = cooldown_left(alias)
    return s


def status_text() -> str:
    """One-line summary of degraded profiles, e.g. 'routing: ultra limited (45s)'."""
    with _lock:
        bad = [(a, s) for a, s in _state.items() if s.get("fails")]
    if not bad:
        return "routing: all healthy"
    now = time.monotonic()
    parts = [f"{a} limited ({max(0, int(s['until'] - now))}s)" for a, s in bad]
    return "routing: " + " · ".join(parts)
