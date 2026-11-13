"""Ensemble: fan-out to all models, judge merges best answer."""
import base64
import json
import pathlib
import queue
import re
import threading
import time
import urllib.request
import urllib.error

from rich.console import Console

from . import llm
from .config import get_active_profile, get_profiles, get_ensemble_settings, ensemble_judge

console = Console()

# Weights per task type - used in judge prompt and scoring
WEIGHTS = {
    "code": {"laguna": 5, "ultra": 4, "glimmer": 3, "super": 3, "paligemma": 1},
    "reasoning": {"glimmer": 5, "ultra": 4, "laguna": 3, "super": 3, "paligemma": 1},
    "vision": {"super": 5, "paligemma": 5, "ultra": 2, "laguna": 1, "glimmer": 1},
    "creative": {"ultra": 5, "laguna": 4, "super": 3, "glimmer": 2, "paligemma": 1},
    "general": {"laguna": 4, "ultra": 4, "glimmer": 4, "super": 3, "paligemma": 2},
}

def detect_task(prompt: str, has_image: bool):
    p = prompt.lower()
    if has_image:
        return "vision"
    if any(w in p for w in ["code", "bug", "function", "class", "python", "javascript", "typescript", "sql", "write a program", "debug", "refactor", "implement", "api"]):
        return "code"
    if any(w in p for w in ["which", "larger", "calculate", "math", "proof", "logic", "reason", "compare", "9.11", "9.8", "solve"]):
        return "reasoning"
    if any(w in p for w in ["poem", "story", "limerick", "write a", "creative", "brainstorm", "essay"]):
        return "creative"
    return "general"

def _load_image_b64(image_src: str):
    """image_src can be URL (http...), local path, or data URI. Returns data URI string."""
    if not image_src:
        return None
    if image_src.startswith("data:image"):
        return image_src
    if image_src.startswith("http://") or image_src.startswith("https://"):
        # for integrate models, URL is fine. For paligemma, need base64.
        # Download here for paligemma use, but return URL for others.
        # Caller decides: we return URL, paligemma will download again.
        return image_src
    # local file
    p = pathlib.Path(image_src)
    if p.exists():
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{b64}"
    raise FileNotFoundError(f"Image not found: {image_src} (not a URL, path, or data URI)")

def _ensure_base64_for_paligemma(image_value: str):
    """Paligemma needs base64 data URI. If URL, download and convert."""
    if image_value.startswith("data:image"):
        return image_value
    if image_value.startswith("http://") or image_value.startswith("https://"):
        # download
        req = urllib.request.Request(image_value, headers={"User-Agent": "zeusai"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
        # guess mime from content-type or ext
        ct = r.headers.get_content_type() or "image/jpeg"
        if ct not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            ct = "image/jpeg"
        b64 = base64.b64encode(data).decode()
        return f"data:{ct};base64,{b64}"
    return image_value

def call_openai_model(alias, prompt, image_value=None, system=None, max_tokens=None,
                      thinking=None, on_text=None, cancel=None):
    _, prof = get_active_profile(alias)
    if prof.get("type") == "vlm":
        return call_paligemma(alias, prompt, image_value, system=system, max_tokens=max_tokens)
    # build message
    if image_value:
        if prof.get("vision"):
            # OpenAI style dict
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_value}},
            ]
        else:
            content = f"{prompt}\n\n[Image URL: {image_value}]"
    else:
        content = prompt
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    r = llm.chat(prof, messages, max_tokens=max_tokens, thinking=thinking, on_text=on_text, cancel=cancel)
    content = r["content"]
    if r["reasoning"] and not content:
        content = r["reasoning"]
    return {"alias": alias, "model_id": prof["model_id"], "content": content, "reasoning": r["reasoning"], "error": None}

def call_paligemma(alias, prompt, image_value, system=None, max_tokens=None, stream=False):
    _, prof = get_active_profile(alias)
    endpoint = prof["base_url"]  # full URL https://ai.api.nvidia.com/v1/vlm/google/paligemma
    if not image_value:
        return {"alias": alias, "model_id": prof["model_id"], "content": "", "reasoning": None, "error": "paligemma needs an image (--image)"}
    try:
        data_uri = _ensure_base64_for_paligemma(image_value)
    except Exception as e:
        return {"alias": alias, "model_id": prof["model_id"], "content": "", "reasoning": None, "error": f"paligemma image fetch failed: {e}"}

    # Paligemma format: image_url as STRING (not dict)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": data_uri},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens or prof.get("max_tokens", 512),
        "temperature": prof.get("temperature", 1.0),
        "top_p": prof.get("top_p", 0.7),
        "stream": stream,
    }
    headers = {
        "Authorization": f"Bearer {prof['api_key']}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            if stream:
                # SSE: collect content (simple)
                full = ""
                for line in r:
                    try:
                        txt = line.decode() if isinstance(line, bytes) else line
                    except:
                        continue
                    if txt.startswith("data:"):
                        try:
                            j = json.loads(txt[5:].strip())
                            delta = j.get("choices", [{}])[0].get("delta", {})
                            if delta.get("content"):
                                full += delta["content"]
                                print(delta["content"], end="", flush=True)
                        except:
                            pass
                return {"alias": alias, "model_id": prof["model_id"], "content": full, "reasoning": None, "error": None}
            else:
                data = r.read()
                j = json.loads(data.decode())
                content = j["choices"][0]["message"]["content"]
                return {"alias": alias, "model_id": prof["model_id"], "content": content, "reasoning": None, "error": None}
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode()[:1000]
        except:
            err_body = str(e)
        return {"alias": alias, "model_id": prof["model_id"], "content": "", "reasoning": None, "error": f"HTTP {e.code}: {err_body}"}
    except Exception as e:
        return {"alias": alias, "model_id": prof["model_id"], "content": "", "reasoning": None, "error": str(e)[:500]}

def _ask_one(alias, prompt, image_value, settings, cancel):
    try:
        return call_openai_model(alias, prompt, image_value,
                                 max_tokens=settings["draft_max_tokens"],
                                 thinking=settings["draft_thinking"], cancel=cancel)
    except Exception as e:
        return {"alias": alias, "model_id": alias, "content": "", "reasoning": None, "error": str(e)[:500]}

def fan_out(prompt, image_src=None, models=None, weights_task=None, verbose=True, on_status=None):
    """Query all models in parallel; stop waiting `grace_seconds` after the first good answer."""
    settings = get_ensemble_settings()
    image_value = None
    if image_src:
        image_value = _load_image_b64(image_src)

    profiles = get_profiles()
    from_settings = not models and bool(settings.get("models"))
    models = models or settings.get("models")
    if isinstance(models, (list, tuple)):
        models = ",".join(models)
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        if from_settings and image_value and profiles.get("paligemma", {}).get("api_key") and "paligemma" not in model_list:
            model_list.append("paligemma")  # vision prompts still get the VLM
    else:
        model_list = []
        for alias in ["laguna", "ultra", "super", "glimmer", "paligemma"]:
            if alias not in profiles:
                continue
            if not profiles[alias].get("api_key"):
                continue
            if alias == "paligemma" and not image_value:
                continue
            model_list.append(alias)

    if not model_list:
        raise SystemExit("No models available. Check keys: zeusai models")

    task = weights_task or detect_task(prompt, bool(image_value))
    weights = WEIGHTS.get(task, WEIGHTS["general"])

    def status(line, style):
        if on_status:
            on_status(line)
        elif verbose:
            console.print(f"[{style}]{line}[/{style}]")

    status(f"Ensemble ({task}): {', '.join(model_list)}", "dim")

    # daemon threads, not a ThreadPoolExecutor: stragglers must not block exit
    done_q = queue.Queue()
    cancel = threading.Event()
    for alias in model_list:
        threading.Thread(target=lambda a=alias: done_q.put(_ask_one(a, prompt, image_value, settings, cancel)),
                         daemon=True).start()

    results = []
    deadline = None
    start = time.monotonic()
    while len(results) < len(model_list):
        timeout = None if deadline is None else max(0, deadline - time.monotonic())
        try:
            res = done_q.get(timeout=timeout)
        except queue.Empty:
            break
        results.append(res)
        if res["error"] or not res["content"]:
            status(f"✗ {res['alias']} {(res['error'] or 'empty')[:120]}", "red")
        else:
            status(f"✓ {res['alias']} {time.monotonic() - start:.1f}s {res['content'][:60].replace(chr(10), ' ')}…", "green")
            if deadline is None:
                # adaptive: give others as long again as the first answer took (min 2s, max grace_seconds)
                first = time.monotonic() - start
                deadline = time.monotonic() + min(settings["grace_seconds"], max(2.0, first))

    cancel.set()
    answered = {r["alias"] for r in results}
    for alias in model_list:
        if alias not in answered:
            status(f"… {alias} skipped (slower than grace window)", "yellow")
            results.append({"alias": alias, "model_id": alias, "content": "", "reasoning": None, "error": "skipped: too slow"})

    order = {a: i for i, a in enumerate(model_list)}
    results.sort(key=lambda r: order.get(r["alias"], 99))
    return results, task, weights

JUDGE_SYSTEM = """You are zeusai ensemble judge. You receive an original prompt and N candidate answers from different models, each with a weight. Evaluate each on accuracy, completeness, clarity, code correctness, and factual consistency. Cross-check claims (e.g. math, code). Pick the strongest and merge into ONE best answer, preserving code blocks, steps, and nuance. Prefer weighted consensus when answers agree; when they disagree, prefer logically correct answers (show brief check). Do NOT invent facts. Keep it concise but complete. Output format exactly (answer FIRST, scores last):

## Final Answer
<merged best answer>

## Scores
- laguna: X/10 (one-line reason)
- ultra: X/10 ...
(only for models that answered)
"""

class _AnswerOnly:
    """Forwards streamed judge text, hiding the '## Final Answer' header and the trailing '## Scores' block."""

    def __init__(self, sink):
        self.sink = sink
        self.buf = ""
        self.state = "head"  # head -> body -> done

    def __call__(self, chunk):
        if self.state == "done":
            return
        self.buf += chunk
        if self.state == "head":
            m = re.search(r"##\s*Final Answer[^\n]*\n", self.buf, re.IGNORECASE)
            if m:
                self.buf = self.buf[m.end():]
                self.state = "body"
            elif len(self.buf) > 200 or ("\n" in self.buf and not self.buf.lstrip().startswith("#")):
                self.state = "body"  # judge ignored the format; stream as-is
            else:
                return
        i = self.buf.find("## Scores")
        if i >= 0:
            self.sink(self.buf[:i].rstrip())
            self.buf, self.state = "", "done"
            return
        # hold back a tail that could be the start of "## Scores"
        safe = len(self.buf) - 10
        if safe > 0:
            self.sink(self.buf[:safe])
            self.buf = self.buf[safe:]

    def flush(self):
        if self.state != "done" and self.buf:
            self.sink(self.buf)
        self.buf = ""

def judge_merge(prompt, results, judge_alias=None, weights=None, image_value=None, on_text=None):
    """Ask judge to score and merge. Streams the merged answer through on_text if given."""
    settings = get_ensemble_settings()
    judge_alias = ensemble_judge(judge_alias)
    ok = [r for r in results if not r["error"] and r["content"]]
    if not ok:
        errs = "\n".join(f"- {r['alias']}: {r['error']}" for r in results if r["error"])
        msg = f"All models failed:\n{errs}"
        if on_text:
            on_text(msg)
        return msg, {}

    weights = weights or {}
    if len(ok) == 1 and settings["skip_judge_if_single"]:
        if on_text:
            on_text(ok[0]["content"])
        return ok[0]["content"], {"single": ok[0]["alias"]}

    blocks = []
    for r in ok:
        w = weights.get(r["alias"], 3)
        blocks.append(f"--- {r['alias']} (weight {w}) ---\n{r['content'][:4000]}")

    judge_prompt = f"Original user prompt:\n{prompt}\n\nCandidate answers (weights inform trust, not truth):\n\n{chr(10).join(blocks)}\n\nNow merge and score as instructed."

    filt = _AnswerOnly(on_text) if on_text else None
    try:
        jres = call_openai_model(judge_alias, judge_prompt, image_value=None, system=JUDGE_SYSTEM,
                                 max_tokens=8192, thinking=settings["judge_thinking"], on_text=filt)
        if filt:
            filt.flush()
        final = jres.get("content") or ""
        if jres.get("error"):
            raise Exception(jres["error"])
    except Exception as e:
        # fallback: weighted pick longest successful
        final = max(ok, key=lambda r: weights.get(r["alias"], 1) * len(r["content"]))["content"]
        if on_text:
            on_text(f"\n\n(judge failed: {str(e)[:200]}; using best draft)\n\n{final}")
        return final, {"fallback": str(e)[:200], "raw_judge_failed": True}

    m = re.search(r"## Final Answer\s*\n(.*?)(?=\n## Scores|\Z)", final, re.DOTALL | re.IGNORECASE)
    merged = m.group(1).strip() if m else final
    scores = {}
    for alias in weights:
        mm = re.search(rf"-?\s*{alias}:\s*(\d+(?:\.\d+)?)/10", final, re.IGNORECASE)
        if mm:
            scores[alias] = float(mm.group(1))
    return merged, {"scores": scores, "judge_raw": final}

def ensemble_ask(prompt, image_src=None, models=None, judge_alias=None, verbose=True, stream=False,
                 on_text=None, on_status=None, header=None):
    """Full ensemble: fan-out + judge merge. Returns merged answer.

    stream=True prints the merged answer to stdout as the judge writes it;
    on_text overrides where streamed text goes (e.g. the TUI).
    """
    judge_alias = ensemble_judge(judge_alias)
    results, task, weights = fan_out(prompt, image_src=image_src, models=models, verbose=verbose, on_status=on_status)
    n_ok = sum(1 for r in results if not r["error"] and r["content"])
    if n_ok > 1:
        line = f"Judging with {judge_alias} (task={task})..."
        if on_status:
            on_status(line)
        elif verbose:
            console.print(f"[dim]{line}[/dim]")

    out = None
    if on_text is None and stream:
        from .agent import StdoutStream
        out = StdoutStream()
        def on_text(chunk, _first=[True]):
            if _first[0] and header:
                console.print(f"\n[bold]{header}[/bold]")
            _first[0] = False
            out.text(chunk)
    try:
        merged, meta = judge_merge(prompt, results, judge_alias=judge_alias, weights=weights, on_text=on_text)
    finally:
        if out:
            out.end()
    if meta.get("scores"):
        s = ", ".join(f"{k}:{v:.0f}/10" for k, v in meta["scores"].items())
        if on_status:
            on_status(f"Scores: {s}")
        elif verbose:
            console.print(f"[dim]Scores: {s}[/dim]")
    return merged, results, task, meta
