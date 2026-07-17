#!/usr/bin/env python3
"""
Effects Library Batch Generator — GitHub Actions optimized version

Generates a library of small, reusable CSS (+ occasional JS) visual
effects — fade-in, zoom, slide-in, hover-lift, etc. — using the same
7-model free-tier Gemini fallback chain as your template generator, then
pushes the result to your Hugging Face dataset.

Effects are NOT full templates. Each one is a small, self-contained
CSS snippet (occasionally paired with the shared scroll-reveal.js) meant
to be applied to elements inside an already-patched website.

Three trigger types: onload, hover, onscroll.

BEFORE YOU RUN (GitHub Actions):
  - Set GEMINI_API_KEY, HF_TOKEN, HF_REPO_ID as GitHub Secrets.
  - The script will run automatically in the workflow.
"""

import json
import os
import re
import sys
import time
import random
import datetime
import requests
from pathlib import Path

# ============================================================================
# CONFIG — read from environment variables (GitHub Secrets)
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"
AUTO_UPLOAD = True   # set to False if you want to skip upload

BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
OUTPUT_DIR = os.path.join(BASE_DIR, "effects")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "_progress.json")
FAILURES_FILE = os.path.join(OUTPUT_DIR, "_failures.json")

MAX_ATTEMPTS_PER_EFFECT = 4
REQUEST_TIMEOUT_SECONDS = 90

# ============================================================================
# MODEL FALLBACK CHAIN — same 7 free-tier models as the template generator
# ============================================================================

MODEL_CHAIN = [
    {"name": "gemini-3.5-flash",                     "rpm": 10, "rpd": 250},
    {"name": "gemini-3-flash-preview",                "rpm": 10, "rpd": 250},
    {"name": "gemini-2.5-pro",                        "rpm": 5,  "rpd": 50},
    {"name": "gemini-3.1-flash-lite",                 "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash",                      "rpm": 10, "rpd": 250},
    {"name": "gemini-2.5-flash-lite",                 "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash-lite-preview-09-2025", "rpm": 15, "rpd": 1000},
]

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ============================================================================
# RATE LIMITER (identical to template generator)
# ============================================================================

class ModelRateLimiter:
    def __init__(self, name, rpm, rpd):
        self.name = name
        self.rpm = rpm
        self.rpd = rpd
        self.request_timestamps = []
        self.daily_count = 0
        self.daily_reset_date = datetime.datetime.utcnow().date()
        self.exhausted_for_today = False

    def _reset_daily_if_needed(self):
        today = datetime.datetime.utcnow().date()
        if today != self.daily_reset_date:
            self.daily_reset_date = today
            self.daily_count = 0
            self.exhausted_for_today = False

    def can_use(self):
        self._reset_daily_if_needed()
        if self.exhausted_for_today:
            return False
        if self.daily_count >= self.rpd:
            self.exhausted_for_today = True
            return False
        return True

    def wait_for_slot(self):
        while True:
            now = time.time()
            self.request_timestamps = [t for t in self.request_timestamps if now - t < 60]
            if len(self.request_timestamps) < self.rpm:
                return
            sleep_for = 60 - (now - self.request_timestamps[0]) + 0.5
            time.sleep(max(sleep_for, 0.5))

    def record_request(self):
        self.request_timestamps.append(time.time())
        self.daily_count += 1

    def mark_exhausted(self):
        self.exhausted_for_today = True


LIMITERS = {m["name"]: ModelRateLimiter(m["name"], m["rpm"], m["rpd"]) for m in MODEL_CHAIN}


# ============================================================================
# GEMINI API CALL (identical to template generator)
# ============================================================================

class AllModelsExhaustedError(Exception):
    pass


def call_gemini(model_name, prompt_text, max_retries=3):
    limiter = LIMITERS[model_name]
    url = f"{GEMINI_API_BASE}/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }

    for attempt in range(max_retries):
        if not limiter.can_use():
            raise AllModelsExhaustedError(f"{model_name} exhausted for today")

        limiter.wait_for_slot()
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT_SECONDS)
            limiter.record_request()

            if resp.status_code == 429:
                print(f"  [{model_name}] 429 rate-limited (attempt {attempt+1}/{max_retries})")
                if "RESOURCE_EXHAUSTED" in resp.text and "quota" in resp.text.lower():
                    limiter.mark_exhausted()
                    raise AllModelsExhaustedError(f"{model_name} quota exhausted")
                time.sleep((2 ** attempt) + random.uniform(0, 1))
                continue

            if resp.status_code >= 500:
                print(f"  [{model_name}] server error {resp.status_code} (attempt {attempt+1}/{max_retries})")
                time.sleep((2 ** attempt) + random.uniform(0, 1))
                continue

            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError(f"No candidates in response: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip():
                raise ValueError("Empty text in response")
            return text

        except requests.exceptions.RequestException as e:
            print(f"  [{model_name}] request exception: {e} (attempt {attempt+1}/{max_retries})")
            time.sleep((2 ** attempt) + random.uniform(0, 1))
            continue

    raise RuntimeError(f"{model_name} failed after {max_retries} retries")


def call_gemini_with_fallback(prompt_text):
    last_error = None
    for model_cfg in MODEL_CHAIN:
        name = model_cfg["name"]
        if not LIMITERS[name].can_use():
            continue
        try:
            return call_gemini(name, prompt_text), name
        except AllModelsExhaustedError as e:
            last_error = e
            continue
        except Exception as e:
            print(f"  [{name}] failed, trying next model in chain: {e}")
            last_error = e
            continue

    raise AllModelsExhaustedError(
        f"All {len(MODEL_CHAIN)} models exhausted or failing. Last error: {last_error}"
    )


# ============================================================================
# GOLDEN EXAMPLES (the three proven patterns: onload, hover, onscroll)
# ============================================================================

GOLDEN_ONLOAD_META = r"""{
  "id": "fade-in-v1",
  "type": "effect",
  "trigger": "onload",
  "name": "Fade In",
  "category": "entrance",
  "description": "Element fades from transparent to fully visible as soon as the page loads. Good for hero text or a page's main heading.",
  "source": "model-generated",
  "reviewed": false,
  "files": ["effect.css"],
  "requires_js": false,
  "requires_shared_js": null,
  "css_class": "fx-fade-in",
  "usage_notes": "Add class=\"fx-fade-in\" to any element. Plays once automatically on page load. No JS needed."
}"""

GOLDEN_ONLOAD_CSS = r"""/* fx-fade-in — plays once automatically when the page loads.
   Usage: <div class="fx-fade-in">...</div> */

.fx-fade-in {
  animation: fx-fade-in-keyframes 0.8s ease-out both;
}

@keyframes fx-fade-in-keyframes {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .fx-fade-in {
    animation: none;
    opacity: 1;
  }
}"""

GOLDEN_HOVER_META = r"""{
  "id": "zoom-hover-v1",
  "type": "effect",
  "trigger": "hover",
  "name": "Zoom on Hover",
  "category": "hover",
  "description": "Element scales up slightly when the user hovers over it. Commonly used on gallery images, product cards, and thumbnails.",
  "source": "model-generated",
  "reviewed": false,
  "files": ["effect.css"],
  "requires_js": false,
  "requires_shared_js": null,
  "css_class": "fx-zoom-hover",
  "usage_notes": "Add class=\"fx-zoom-hover\" to an image or card element. The parent container should have overflow: hidden to avoid the zoomed element spilling outside its box — wrap the element in a container with class=\"fx-zoom-hover-wrap\" if needed."
}"""

GOLDEN_HOVER_CSS = r""".fx-zoom-hover-wrap {
  overflow: hidden;
  display: block;
}

.fx-zoom-hover {
  transition: transform 0.35s ease;
  transform: scale(1);
  display: block;
}

.fx-zoom-hover:hover,
.fx-zoom-hover:focus-visible {
  transform: scale(1.08);
}

@media (prefers-reduced-motion: reduce) {
  .fx-zoom-hover {
    transition: none;
  }
  .fx-zoom-hover:hover,
  .fx-zoom-hover:focus-visible {
    transform: none;
  }
}"""

GOLDEN_ONSCROLL_META = r"""{
  "id": "slide-in-left-v1",
  "type": "effect",
  "trigger": "onscroll",
  "name": "Slide In From Left",
  "category": "entrance",
  "description": "Element slides in from the left and fades in as the user scrolls it into view. Good for section headings or feature blocks appearing lower on the page.",
  "source": "model-generated",
  "reviewed": false,
  "files": ["effect.css"],
  "requires_js": true,
  "requires_shared_js": "scroll-reveal-observer-v1",
  "css_class": "fx-slide-in-left",
  "usage_notes": "Add classes=\"fx-scroll-target fx-slide-in-left\" to the element. Requires the shared scroll-reveal.js to be included once on the page (see scroll-reveal-observer-v1)."
}"""

GOLDEN_ONSCROLL_CSS = r""".fx-slide-in-left.fx-scroll-target {
  opacity: 0;
  transform: translateX(-40px);
  transition: opacity 0.6s ease, transform 0.6s ease;
}

.fx-slide-in-left.fx-scroll-target.fx-in-view {
  opacity: 1;
  transform: translateX(0);
}

@media (prefers-reduced-motion: reduce) {
  .fx-slide-in-left.fx-scroll-target {
    opacity: 1;
    transform: none;
    transition: none;
  }
}"""


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

GENERATION_PROMPT_TEMPLATE = """You are generating ONE reusable visual effect for a small effects library
used by a website-building AI agent. Effects are applied to elements inside
already-built websites (e.g. adding a class to a gallery image or heading).
Follow the matching pattern below EXACTLY — consistency matters more than
creativity.

## Output format (strict)

Return ONLY a single JSON object, no markdown fences, no commentary:

{{
  "meta": {{ ...meta.json content as a JSON object... }},
  "css": "...full effect.css content as a string..."
}}

## The three trigger types — use the ONE matching what you're asked for

**onload**: Pure CSS animation that plays automatically once, when the
element renders. No JS. Use a @keyframes animation.

**hover**: Pure CSS :hover and :focus-visible pseudo-class effect. No JS,
no @keyframes needed — just a transition between normal and hover state.

**onscroll**: CSS defines TWO states using class selectors:
  - `.{{css_class}}.fx-scroll-target` (the "before" / hidden state)
  - `.{{css_class}}.fx-scroll-target.fx-in-view` (the "after" / revealed state)
  A shared external script (not generated by you) adds the fx-in-view class
  when the element scrolls into view. Do NOT write any JS for onscroll
  effects — only the CSS for both states, using a `transition`.

## Rules you must follow

1. Class name MUST be prefixed `fx-` and match the "css_class" field in meta.
2. ALWAYS include a `@media (prefers-reduced-motion: reduce)` block that
   disables the animation/transition and shows the end-state directly —
   this is not optional, every effect must be accessible.
3. meta.json "trigger" must be exactly one of: "onload", "hover", "onscroll".
4. For "onscroll" effects, "requires_js" must be true and
   "requires_shared_js" must be "scroll-reveal-observer-v1".
   For "onload" and "hover" effects, "requires_js" must be false and
   "requires_shared_js" must be null.
5. Keep effects SUBTLE and professional — this is for real business
   websites, not flashy demos. Avoid effects that could cause motion
   sickness (no spinning, no large bounces, no rapid flashing).
6. "reviewed" must always be false, "source" must be "model-generated".
7. Only output "css" — never output HTML, never output JS (JS only exists
   in the one shared observer script, which you are not generating).

## meta.json schema (fill exactly this shape)

{{
  "id": "[effect-slug]-v1",
  "type": "effect",
  "trigger": "{trigger}",
  "name": "[Human-readable name]",
  "category": "{category}",
  "description": "[1-2 sentences: what it does and when to use it]",
  "source": "model-generated",
  "reviewed": false,
  "files": ["effect.css"],
  "requires_js": [true/false per trigger type rules above],
  "requires_shared_js": [null, or "scroll-reveal-observer-v1" for onscroll],
  "css_class": "fx-[slug]",
  "usage_notes": "[exact HTML class attribute usage instructions]"
}}

## Golden example for trigger type "onload"

GOLDEN meta.json:
{golden_onload_meta}

GOLDEN effect.css:
{golden_onload_css}

## Golden example for trigger type "hover"

GOLDEN meta.json:
{golden_hover_meta}

GOLDEN effect.css:
{golden_hover_css}

## Golden example for trigger type "onscroll"

GOLDEN meta.json:
{golden_onscroll_meta}

GOLDEN effect.css:
{golden_onscroll_css}

## Now generate

Trigger type: {trigger}
Effect name: {name}
Category: {category}
Guidance: {guidance}

Follow the exact pattern for this trigger type shown above. Return ONLY the
JSON object.
"""


def build_prompt(trigger, name, category, guidance):
    return GENERATION_PROMPT_TEMPLATE.format(
        trigger=trigger,
        name=name,
        category=category,
        guidance=guidance,
        golden_onload_meta=GOLDEN_ONLOAD_META,
        golden_onload_css=GOLDEN_ONLOAD_CSS,
        golden_hover_meta=GOLDEN_HOVER_META,
        golden_hover_css=GOLDEN_HOVER_CSS,
        golden_onscroll_meta=GOLDEN_ONSCROLL_META,
        golden_onscroll_css=GOLDEN_ONSCROLL_CSS,
    )


# ============================================================================
# EFFECT SPECS — kept subtle, professional, covering the common real needs
# ============================================================================

EFFECT_SPECS_RAW = [
    # (trigger, name, category, guidance)
    ("onload", "Fade In Up", "entrance", "Fades in while moving up slightly (translateY), a bit more dynamic than plain fade-in."),
    ("onload", "Scale In", "entrance", "Element scales from slightly smaller to full size while fading in."),
    ("onload", "Fade In Down", "entrance", "Fades in while moving down slightly, good for dropdown-style reveals."),

    ("hover", "Lift on Hover", "hover", "Element lifts slightly (translateY up) and gains a soft shadow on hover — good for cards."),
    ("hover", "Glow on Hover", "hover", "Subtle box-shadow glow appears around the element on hover — good for buttons."),
    ("hover", "Underline Grow", "hover", "An underline grows from 0 to full width beneath text on hover — good for nav links."),
    ("hover", "Tilt on Hover", "hover", "Element tilts very slightly in 3D (small rotateX/rotateY) on hover — good for cards or images."),
    ("hover", "Border Pulse on Hover", "hover", "Border color transitions smoothly to an accent color on hover — good for buttons/inputs."),
    ("hover", "Brightness Dim on Hover", "hover", "Image or card slightly dims/brightens on hover — good for gallery thumbnails."),

    ("onscroll", "Slide In From Right", "entrance", "Mirror of slide-in-left — element slides in from the right and fades in."),
    ("onscroll", "Slide In From Bottom", "entrance", "Element slides up from below and fades in as it scrolls into view."),
    ("onscroll", "Scale In On Scroll", "entrance", "Element scales up from slightly smaller to full size while fading in, triggered on scroll."),
    ("onscroll", "Fade In On Scroll", "entrance", "Simple opacity-only fade in when scrolled into view (no movement) — subtler than slide variants."),
]

EFFECT_SPECS = []
for trigger, name, category, guidance in EFFECT_SPECS_RAW:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    spec_id = f"{slug}-v1"
    EFFECT_SPECS.append({
        "id": spec_id,
        "trigger": trigger,
        "name": name,
        "category": category,
        "guidance": guidance,
    })

print(f"Total effects to generate: {len(EFFECT_SPECS)}")


# ============================================================================
# VALIDATION
# ============================================================================

class ValidationError(Exception):
    pass


def strip_code_fences(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def validate_effect(data, expected_trigger):
    if "meta" not in data or "css" not in data:
        raise ValidationError("Missing required top-level keys (meta/css)")

    meta = data["meta"]
    css = data["css"]

    for field in ["id", "type", "trigger", "name", "category", "css_class", "requires_js", "requires_shared_js"]:
        if field not in meta:
            raise ValidationError(f"meta.json missing required field: {field}")

    if meta["trigger"] != expected_trigger:
        raise ValidationError(f"trigger mismatch: expected '{expected_trigger}', got '{meta['trigger']}'")

    css_class = meta["css_class"]
    if not css_class.startswith("fx-"):
        raise ValidationError(f"css_class must start with 'fx-', got: {css_class}")

    if css_class not in css:
        raise ValidationError(f"css_class '{css_class}' declared in meta but not found in css")

    if "prefers-reduced-motion" not in css:
        raise ValidationError("Missing required @media (prefers-reduced-motion: reduce) block")

    if expected_trigger == "onscroll":
        if meta["requires_js"] is not True:
            raise ValidationError("onscroll effect must have requires_js: true")
        if meta["requires_shared_js"] != "scroll-reveal-observer-v1":
            raise ValidationError("onscroll effect must set requires_shared_js to 'scroll-reveal-observer-v1'")
        if "fx-in-view" not in css:
            raise ValidationError("onscroll effect css must reference fx-in-view state")
        if "fx-scroll-target" not in css:
            raise ValidationError("onscroll effect css must reference fx-scroll-target class")
    else:
        if meta["requires_js"] is not False:
            raise ValidationError(f"{expected_trigger} effect must have requires_js: false")
        if meta["requires_shared_js"] is not None:
            raise ValidationError(f"{expected_trigger} effect must have requires_shared_js: null")

    if expected_trigger == "onload" and "@keyframes" not in css:
        raise ValidationError("onload effect should use a @keyframes animation")

    if expected_trigger == "hover" and (":hover" not in css):
        raise ValidationError("hover effect css must include a :hover selector")

    return True


def parse_and_validate(raw_text, expected_trigger):
    cleaned = strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}")
    validate_effect(data, expected_trigger)
    return data


# ============================================================================
# PROGRESS / FAILURES TRACKING
# ============================================================================

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": []}


def save_progress(progress):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def load_failures():
    if os.path.exists(FAILURES_FILE):
        with open(FAILURES_FILE, "r") as f:
            return json.load(f)
    return []


def save_failure(spec, error_msg):
    failures = load_failures()
    failures.append({
        "id": spec["id"],
        "error": str(error_msg),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FAILURES_FILE, "w") as f:
        json.dump(failures, f, indent=2)


# ============================================================================
# SAVE
# ============================================================================

def save_effect_files(spec, data):
    trigger = spec["trigger"]
    spec_id = spec["id"]
    folder = Path(OUTPUT_DIR) / trigger / spec_id
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "meta.json", "w") as f:
        json.dump(data["meta"], f, indent=2)
    with open(folder / "effect.css", "w") as f:
        f.write(data["css"])

    return str(folder)


# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

def generate_one(spec):
    prompt = build_prompt(spec["trigger"], spec["name"], spec["category"], spec["guidance"])

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS_PER_EFFECT + 1):
        try:
            raw_text, model_used = call_gemini_with_fallback(prompt)
            data = parse_and_validate(raw_text, spec["trigger"])
            print(f"  ✓ {spec['id']} generated by {model_used} (attempt {attempt})")
            return data
        except AllModelsExhaustedError:
            print(f"  ✗ {spec['id']}: all models exhausted — stopping batch for now")
            raise
        except ValidationError as e:
            print(f"  ~ {spec['id']} validation failed (attempt {attempt}/{MAX_ATTEMPTS_PER_EFFECT}): {e}")
            last_error = e
            continue
        except Exception as e:
            print(f"  ~ {spec['id']} unexpected error (attempt {attempt}/{MAX_ATTEMPTS_PER_EFFECT}): {e}")
            last_error = e
            continue

    raise ValidationError(f"Failed after {MAX_ATTEMPTS_PER_EFFECT} attempts. Last error: {last_error}")


def run_batch():
    progress = load_progress()
    completed_ids = set(progress["completed"])

    remaining = [s for s in EFFECT_SPECS if s["id"] not in completed_ids]
    print(f"\n{len(completed_ids)} already completed, {len(remaining)} remaining.\n")

    for i, spec in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] Generating {spec['id']} ({spec['trigger']}) ...")
        try:
            data = generate_one(spec)
            path = save_effect_files(spec, data)
            completed_ids.add(spec["id"])
            progress["completed"] = list(completed_ids)
            save_progress(progress)
            print(f"  Saved to {path}")
        except AllModelsExhaustedError:
            print("\nAll free-tier models exhausted for today. Progress saved —")
            print("Re-run this script later to continue where you left off.")
            return False   # signal to stop
        except Exception as e:
            print(f"  FAILED: {spec['id']} — {e}")
            save_failure(spec, e)
            continue

    print("\nAll effects generated (or logged as failures). Check _failures.json for anything to re-run.")
    return True   # all done


# ============================================================================
# UPLOAD TO HUGGING FACE
# ============================================================================

def upload_to_huggingface():
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)
    try:
        create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type=HF_REPO_TYPE, exist_ok=True)
        print(f"✅ Repository {HF_REPO_ID} is ready.")
    except Exception as e:
        print(f"Note: create_repo said: {e} (fine if repo already exists)")

    print(f"📤 Uploading {OUTPUT_DIR} to {HF_REPO_ID} under 'effects/' ...")
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        path_in_repo="effects",
        commit_message=f"Batch upload: generated effects library - {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
    )
    print("✅ Upload complete.")


# ============================================================================
# ENTRYPOINT
# ============================================================================

def main():
    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    if not HF_TOKEN:
        print("❌ HF_TOKEN environment variable not set.")
        sys.exit(1)

    print(f"🚀 Starting effects generation...")
    print(f"📂 Output directory: {OUTPUT_DIR}")

    success = run_batch()

    failures = load_failures()
    if failures:
        print(f"\n⚠️  {len(failures)} effects failed validation after all retries — see {FAILURES_FILE}")
        print("These were NOT uploaded. Review and re-run just those specs manually if needed.")

    if AUTO_UPLOAD and success:
        print("\n📤 Auto-upload enabled – pushing to HuggingFace...")
        upload_to_huggingface()
    elif AUTO_UPLOAD and not success:
        print("\n⚠️  Batch was incomplete (quota exhausted). Skipping upload until all effects are generated.")
    else:
        print(f"\n✅ Generation finished. Files are in {OUTPUT_DIR} — upload manually when ready.")

    print("\n✅ All done.")


if __name__ == "__main__":
    main()