#!/usr/bin/env python3
"""
Template Library Batch Generator – GitHub Actions Optimized
Reads API keys from environment variables (GitHub Secrets)
"""

import json
import os
import re
import time
import random
import datetime
import requests
import sys
from pathlib import Path

# ============================================================================
# CONFIG — read from environment variables (GitHub Secrets)
# ============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/github/workspace/templates")
PROGRESS_FILE = f"{OUTPUT_DIR}/_progress.json"
FAILURES_FILE = f"{OUTPUT_DIR}/_failures.json"

MAX_ATTEMPTS_PER_TEMPLATE = 4
REQUEST_TIMEOUT_SECONDS = 90

# ============================================================================
# MODEL FALLBACK CHAIN
# ============================================================================

MODEL_CHAIN = [
    {"name": "gemini-3.5-flash", "rpm": 10, "rpd": 250},
    {"name": "gemini-3-flash-preview", "rpm": 10, "rpd": 250},
    {"name": "gemini-2.5-pro", "rpm": 5, "rpd": 50},
    {"name": "gemini-3.1-flash-lite", "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash", "rpm": 10, "rpd": 250},
    {"name": "gemini-2.5-flash-lite", "rpm": 15, "rpd": 1000},
    {"name": "gemini-2.5-flash-lite-preview-09-2025", "rpm": 15, "rpd": 1000},
]

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


# ============================================================================
# RATE LIMITER
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
# GEMINI API CALL
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
            "maxOutputTokens": 8192,
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
                backoff = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff)
                continue

            if resp.status_code >= 500:
                print(f"  [{model_name}] server error {resp.status_code} (attempt {attempt+1}/{max_retries})")
                backoff = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(backoff)
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
            backoff = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(backoff)
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
# GOLDEN EXAMPLE
# ============================================================================

GOLDEN_META = r"""{
  "id": "restaurant-basic-v1",
  "level": "whole-site",
  "category": "local-service-business",
  "subcategory": "restaurant",
  "description": "Single-page restaurant site with hero, menu, about blurb, hours, and contact/location. Works for cafes, bistros, and small eateries.",
  "source": "model-generated",
  "license": null,
  "source_url": null,
  "reviewed": false,
  "files": ["index.html", "style.css"],
  "placeholders": {
    "scalar": [
      "business_name",
      "tagline",
      "about_text",
      "phone_number",
      "address",
      "hours_text",
      "primary_color",
      "accent_color",
      "map_embed_url"
    ],
    "repeating": {
      "menu_items": {
        "container_marker": "menu_items",
        "fields": ["item_name", "item_description", "item_price"]
      }
    }
  },
  "notes": "Placeholders use {{scalar_name}} syntax. Repeating blocks are wrapped in <!-- REPEAT:name --> ... <!-- END:name --> comments; the orchestrator expands these from a JSON array, the model never edits the HTML inside directly."
}"""

GOLDEN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{business_name}}</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>

  <header class="site-header">
    <div class="container header-inner">
      <span class="brand">{{business_name}}</span>
      <nav class="nav-links">
        <a href="#menu">Menu</a>
        <a href="#about">About</a>
        <a href="#visit">Visit</a>
      </nav>
    </div>
  </header>

  <section class="hero">
    <div class="container hero-inner">
      <h1>{{business_name}}</h1>
      <p class="tagline">{{tagline}}</p>
      <a href="#visit" class="cta-button">Find Us</a>
    </div>
  </section>

  <section id="menu" class="menu-section">
    <div class="container">
      <h2>Menu</h2>
      <div class="menu-grid">
        <!-- REPEAT:menu_items -->
        <div class="menu-item">
          <div class="menu-item-header">
            <span class="menu-item-name">{{item_name}}</span>
            <span class="menu-item-price">{{item_price}}</span>
          </div>
          <p class="menu-item-description">{{item_description}}</p>
        </div>
        <!-- END:menu_items -->
      </div>
    </div>
  </section>

  <section id="about" class="about-section">
    <div class="container">
      <h2>About Us</h2>
      <p>{{about_text}}</p>
    </div>
  </section>

  <section id="visit" class="visit-section">
    <div class="container visit-grid">
      <div class="visit-details">
        <h2>Visit Us</h2>
        <p class="detail-line"><strong>Address:</strong> {{address}}</p>
        <p class="detail-line"><strong>Phone:</strong> {{phone_number}}</p>
        <p class="detail-line"><strong>Hours:</strong> {{hours_text}}</p>
      </div>
      <div class="map-embed">
        <iframe
          src="{{map_embed_url}}"
          width="100%"
          height="300"
          style="border:0;"
          allowfullscreen=""
          loading="lazy">
        </iframe>
      </div>
    </div>
  </section>

  <footer class="site-footer">
    <div class="container">
      <p>&copy; <span id="year"></span> {{business_name}}. All rights reserved.</p>
    </div>
  </footer>

  <script>
    document.getElementById('year').textContent = new Date().getFullYear();
  </script>

</body>
</html>"""

GOLDEN_CSS = r""":root {
  --primary-color: {{primary_color}};
  --accent-color: {{accent_color}};
  --text-color: #2b2420;
  --bg-color: #fdfaf6;
  --border-color: #e6ddd2;
}

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: Georgia, 'Times New Roman', serif;
  color: var(--text-color);
  background: var(--bg-color);
  line-height: 1.6;
}

.container {
  max-width: 960px;
  margin: 0 auto;
  padding: 0 24px;
}

.site-header {
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-color);
  position: sticky;
  top: 0;
  z-index: 10;
}

.header-inner {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
}

.brand {
  font-size: 1.25rem;
  font-weight: bold;
  color: var(--primary-color);
}

.nav-links a {
  margin-left: 24px;
  text-decoration: none;
  color: var(--text-color);
  font-size: 0.95rem;
}

.nav-links a:hover,
.nav-links a:focus-visible {
  color: var(--primary-color);
}

.hero {
  background: var(--primary-color);
  color: #fff;
  text-align: center;
  padding: 96px 24px;
}

.hero h1 {
  font-size: 2.75rem;
  margin-bottom: 12px;
}

.hero .tagline {
  font-size: 1.15rem;
  opacity: 0.9;
  margin-bottom: 32px;
}

.cta-button {
  display: inline-block;
  background: var(--accent-color);
  color: #fff;
  padding: 12px 28px;
  border-radius: 4px;
  text-decoration: none;
  font-weight: bold;
  transition: opacity 0.2s ease;
}

.cta-button:hover,
.cta-button:focus-visible {
  opacity: 0.85;
}

.menu-section {
  padding: 72px 24px;
}

.menu-section h2,
.about-section h2,
.visit-section h2 {
  font-size: 1.75rem;
  margin-bottom: 32px;
  color: var(--primary-color);
}

.menu-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 24px;
}

.menu-item {
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 16px;
}

.menu-item-header {
  display: flex;
  justify-content: space-between;
  font-weight: bold;
  margin-bottom: 6px;
}

.menu-item-price {
  color: var(--accent-color);
}

.menu-item-description {
  font-size: 0.95rem;
  opacity: 0.85;
}

.about-section {
  background: #fff;
  padding: 72px 24px;
  border-top: 1px solid var(--border-color);
  border-bottom: 1px solid var(--border-color);
}

.visit-section {
  padding: 72px 24px;
}

.visit-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 32px;
  align-items: start;
}

.detail-line {
  margin-bottom: 8px;
}

.map-embed iframe {
  border-radius: 4px;
}

.site-footer {
  background: var(--text-color);
  color: #fff;
  text-align: center;
  padding: 24px;
  font-size: 0.85rem;
}

@media (max-width: 640px) {
  .visit-grid {
    grid-template-columns: 1fr;
  }

  .hero {
    padding: 64px 20px;
  }

  .hero h1 {
    font-size: 2rem;
  }
}"""


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

GENERATION_PROMPT_TEMPLATE = """You are generating ONE website template for a template library used by a small
local AI coding agent. The agent will later patch this template by replacing
placeholders with real business content. Follow the pattern below EXACTLY —
consistency across the whole library matters more than creativity.

## Output format (strict)

Return ONLY a single JSON object, no markdown fences, no commentary, in this
exact shape:

{{
  "meta": {{ ...meta.json content as a JSON object... }},
  "html": "...full index.html content as a string...",
  "css": "...full style.css content as a string..."
}}

## Rules you must follow

1. Use ONLY these two placeholder types, never invent a third kind:
   - Scalar: {{{{snake_case_name}}}} — for single values (name, phone, tagline, colors, etc)
   - Repeating block: wrapped exactly like this, for lists:
     <!-- REPEAT:block_name -->
     ...one item's markup using {{{{field_name}}}} placeholders...
     <!-- END:block_name -->

2. Every placeholder used in the HTML/CSS MUST be listed in meta.json's
   "placeholders" object (scalar array, or repeating object with fields array).
   Every placeholder listed in meta.json MUST actually appear in the HTML/CSS.
   No mismatches — this is the single most important rule.

3. Color placeholders (e.g. {{{{primary_color}}}}, {{{{accent_color}}}}) must ONLY be
   used inside style.css as CSS custom property values (:root block), never
   hardcoded as literal hex/color words elsewhere in the CSS.

4. Structure: single index.html + single style.css. No external JS frameworks,
   no build step, no CDN dependencies except optionally Google Fonts. Vanilla
   HTML/CSS/minimal inline JS only (e.g. footer year, simple mobile nav toggle
   if included).

5. Must be responsive (mobile-friendly) using a media query at max-width: 640px
   minimum, following the same pattern as the golden example.

6. Copy/text content: write real, specific-feeling placeholder-adjacent copy
   for anything that ISN'T user data (e.g. section headers, microcopy, button
   labels) — but anything that's clearly business-specific (name, address,
   description, items/services) MUST be a placeholder, never hardcoded.

7. meta.json fields required (see schema below) — "reviewed" must always be
   false (a human sets this to true later), "source" must be "model-generated".

## meta.json schema (fill exactly this shape)

{{
  "id": "[category-slug]-[variant-slug]-v1",
  "level": "whole-site",
  "category": "{category}",
  "subcategory": "{variant}",
  "description": "[1-2 sentence description of this variant and when it fits]",
  "source": "model-generated",
  "license": null,
  "source_url": null,
  "reviewed": false,
  "files": ["index.html", "style.css"],
  "placeholders": {{
    "scalar": ["...", "..."],
    "repeating": {{
      "block_name": {{
        "container_marker": "block_name",
        "fields": ["...", "..."]
      }}
    }}
  }}
}}

## Golden example (the exact pattern to replicate)

Category: local-service-business
Variant: casual restaurant/cafe

GOLDEN meta.json:
{golden_meta}

GOLDEN index.html:
{golden_html}

GOLDEN style.css:
{golden_css}

## Now generate

Category: {category}
Variant: {variant}
Variant guidance: {guidance}

Follow the exact same file structure, placeholder conventions, and JSON output
shape as the golden example. Return ONLY the JSON object.
"""


def build_prompt(category, variant, guidance):
    return GENERATION_PROMPT_TEMPLATE.format(
        category=category,
        variant=variant,
        guidance=guidance,
        golden_meta=GOLDEN_META,
        golden_html=GOLDEN_HTML,
        golden_css=GOLDEN_CSS,
    )


# ============================================================================
# TEMPLATE SPECS — 20 categories x 4 variants = 80 whole-site templates
# ============================================================================

CATEGORIES = [
    ("local-service-business", [
        ("casual restaurant/cafe", "Friendly, warm tone. Menu-forward. Simple hero + menu + about + visit sections."),
        ("upscale fine-dining", "Photo/ambiance-forward. Story/chef section. Reservation CTA instead of plain contact."),
        ("fast-casual counter-service", "Ordering-forward. Hours/location prominent near top. Minimal about section."),
        ("bar-and-grill/pub", "Casual tone. Include an events/specials section and social links in footer."),
    ]),
    ("portfolio", [
        ("minimal personal portfolio", "Clean, lots of whitespace. Project grid + about + contact."),
        ("creative/designer portfolio", "Bold typography, image-forward project showcase."),
        ("developer portfolio", "Skills/tech-stack section, project cards with links, resume download CTA."),
        ("freelancer services portfolio", "Services list + testimonials + contact form emphasis."),
    ]),
    ("landing-page", [
        ("SaaS product landing page", "Hero + feature grid + pricing teaser + CTA. Single scrolling page."),
        ("mobile app landing page", "App screenshots section, download buttons (App Store/Play Store placeholders), feature highlights."),
        ("event/webinar landing page", "Countdown/date-forward, speaker section, signup form CTA."),
        ("newsletter/community landing page", "Simple, single strong CTA, benefits list, social proof section."),
    ]),
    ("blog", [
        ("personal blog", "Simple post list layout, author bio section, minimal sidebar."),
        ("magazine-style blog", "Featured post hero, grid of post cards with categories."),
        ("niche topic blog (e.g. food/travel)", "Image-forward post cards, newsletter signup CTA."),
        ("company/product blog", "Clean, category filter tags, CTA back to main product."),
    ]),
    ("ecommerce-showcase", [
        ("small product showcase (no cart)", "Product grid, each with image/name/price/contact-to-buy CTA."),
        ("handmade/artisan shop showcase", "Warm, story-driven, product grid with maker's story section."),
        ("single-product showcase", "One hero product, detailed features, FAQ, contact-to-buy CTA."),
        ("boutique/clothing showcase", "Lookbook-style image grid, categories, contact-to-buy CTA."),
    ]),
    ("event-wedding-rsvp", [
        ("wedding site", "Elegant, couple names, date/countdown, RSVP form, registry links."),
        ("birthday/celebration event", "Fun tone, event details, RSVP form."),
        ("conference/summit event", "Agenda/schedule section, speakers, registration CTA."),
        ("baby shower/gender reveal event", "Soft tone, event details, RSVP form, registry links."),
    ]),
    ("nonprofit", [
        ("community nonprofit", "Mission statement hero, programs section, donate CTA, volunteer signup."),
        ("environmental/advocacy nonprofit", "Cause-forward, impact stats section, donate CTA."),
        ("animal rescue nonprofit", "Adoptable animals grid placeholder, donate CTA, volunteer section."),
        ("arts/culture nonprofit", "Events/programs section, donate CTA, gallery section."),
    ]),
    ("real-estate", [
        ("single listing page", "Large hero image, property details grid, gallery, contact agent CTA."),
        ("agent/realtor personal site", "About the agent, listings grid, testimonials, contact form."),
        ("property management company", "Services section, property grid, contact/inquiry form."),
        ("vacation rental listing", "Amenities list, gallery, booking inquiry CTA, reviews section."),
    ]),
    ("resume-cv", [
        ("classic resume site", "Simple sections: summary, experience, education, skills, contact."),
        ("creative/design resume site", "Visual timeline style, portfolio snippets, contact."),
        ("executive/professional resume site", "Formal tone, achievements-forward, downloadable PDF CTA."),
        ("student/entry-level resume site", "Education-forward, projects section, skills, contact."),
    ]),
    ("documentation-knowledge-base", [
        ("simple docs site", "Sidebar nav placeholder, article content area, search box placeholder."),
        ("FAQ/help-center style", "Accordion-style FAQ sections grouped by category."),
        ("API/developer docs style", "Code-block friendly layout, sidebar nav, getting-started section."),
        ("internal wiki style", "Simple nav, article list, minimal styling."),
    ]),
    ("coming-soon-waitlist", [
        ("minimal coming-soon page", "Single centered message, countdown, email signup form."),
        ("product launch waitlist", "Feature teaser, email signup, social links."),
        ("app pre-launch waitlist", "App preview/screenshot placeholder, signup form, social proof."),
        ("under-construction/relaunch page", "Simple message, contact email, social links."),
    ]),
    ("contact-booking-form", [
        ("simple contact page", "Contact form + address/phone/hours, map embed."),
        ("appointment booking page", "Service selection, date/time placeholder fields, confirmation CTA."),
        ("consultation request page", "Form with fields for project details, submit CTA."),
        ("support/help request page", "Category dropdown, message field, submit CTA."),
    ]),
    ("fitness-gym-studio", [
        ("gym/fitness center", "Class schedule section, trainer bios, membership CTA."),
        ("yoga/pilates studio", "Calm tone, class schedule, instructor bios, trial class CTA."),
        ("personal training service", "Trainer bio-forward, before/after or testimonials section, booking CTA."),
        ("martial arts/boxing studio", "Bold tone, class schedule, instructor bios, trial class CTA."),
    ]),
    ("medical-dental-clinic", [
        ("general medical clinic", "Services list, doctor bios, appointment booking CTA, insurance note section."),
        ("dental practice", "Services list, dentist bios, appointment booking CTA, before/after gallery placeholder."),
        ("specialist clinic (e.g. dermatology)", "Services list, specialist bio, appointment booking CTA."),
        ("wellness/therapy clinic", "Calm tone, services list, practitioner bios, booking CTA."),
    ]),
    ("legal-professional-services", [
        ("law firm/attorney", "Trust-focused tone, practice areas list, attorney bios, consultation CTA."),
        ("accounting/tax services", "Services list, team bios, consultation CTA, testimonials."),
        ("business consulting firm", "Services list, case studies section, consultation CTA."),
        ("financial advisory services", "Trust-focused tone, services list, advisor bios, consultation CTA."),
    ]),
    ("photography-creative-services", [
        ("wedding photographer", "Portfolio gallery-forward, packages section, contact/booking CTA."),
        ("portrait/family photographer", "Portfolio gallery, packages section, booking CTA."),
        ("commercial/product photographer", "Portfolio gallery grouped by category, contact CTA."),
        ("videographer/videography studio", "Video-forward portfolio placeholder, packages, contact CTA."),
    ]),
    ("church-religious-organization", [
        ("church home page", "Service times section, welcome message, ministries list, contact/visit CTA."),
        ("synagogue/temple home page", "Service times section, community programs, contact/visit CTA."),
        ("faith-based nonprofit/ministry", "Mission section, programs list, donate CTA."),
        ("religious event/retreat page", "Event details, schedule, registration CTA."),
    ]),
    ("school-tutoring-education", [
        ("private school/academy", "Programs section, admissions CTA, faculty highlights, testimonials."),
        ("tutoring service", "Subjects offered list, tutor bios, booking/trial-session CTA."),
        ("online course/education business", "Course list, instructor bio, enrollment CTA."),
        ("daycare/early education center", "Programs section, enrollment CTA, testimonials."),
    ]),
    ("auto-repair-shop", [
        ("general auto repair shop", "Services list, appointment booking CTA, testimonials."),
        ("specialty shop (e.g. tires/brakes)", "Services list, appointment booking CTA, pricing teaser."),
        ("auto detailing service", "Services/packages list, before/after gallery placeholder, booking CTA."),
        ("mobile mechanic service", "Service area section, services list, booking CTA."),
    ]),
    ("nonprofit-fundraiser-donation", [
        ("single fundraising campaign page", "Goal/progress bar placeholder, story section, donate CTA."),
        ("recurring giving/membership page", "Giving tiers section, impact stats, donate CTA."),
        ("disaster relief/urgent appeal page", "Urgent tone, story section, donate CTA."),
        ("charity gala/fundraising event page", "Event details, ticket/sponsor CTA, past-event highlights."),
    ]),
]

TEMPLATE_SPECS = []
for category, variants in CATEGORIES:
    for i, (variant, guidance) in enumerate(variants, start=1):
        variant_slug = re.sub(r"[^a-z0-9]+", "-", variant.lower()).strip("-")
        spec_id = f"{category}-{variant_slug}-v1"
        TEMPLATE_SPECS.append({
            "id": spec_id,
            "category": category,
            "variant": variant,
            "guidance": guidance,
        })

print(f"Total whole-site templates to generate: {len(TEMPLATE_SPECS)}")


# ============================================================================
# COMPONENT LIBRARY SPECS
# ============================================================================

COMPONENT_SPECS_RAW = [
    ("navbar", ["simple horizontal navbar", "navbar with dropdown menu", "sticky navbar with mobile hamburger toggle"]),
    ("hero", ["centered text hero", "split image+text hero", "full-bleed background image hero"]),
    ("footer", ["simple footer with links", "footer with newsletter signup", "footer with social icons and sitemap"]),
    ("contact-form", ["simple contact form", "contact form with map embed", "multi-field inquiry form"]),
    ("pricing-table", ["3-tier pricing table", "single-tier pricing highlight", "comparison-style pricing table"]),
    ("testimonial", ["single testimonial spotlight", "testimonial carousel-style grid", "testimonial with photo and rating"]),
    ("gallery", ["simple image grid gallery", "masonry-style gallery", "gallery with lightbox-style captions"]),
    ("faq-accordion", ["simple accordion FAQ", "two-column FAQ list", "categorized FAQ accordion"]),
    ("team-about", ["team grid with photos and roles", "single founder about section", "team section with bios and social links"]),
]

COMPONENT_SPECS = []
for comp_type, variants in COMPONENT_SPECS_RAW:
    for i, variant in enumerate(variants, start=1):
        variant_slug = re.sub(r"[^a-z0-9]+", "-", variant.lower()).strip("-")
        spec_id = f"component-{comp_type}-{variant_slug}-v1"
        COMPONENT_SPECS.append({
            "id": spec_id,
            "category": "components",
            "component_type": comp_type,
            "variant": variant,
            "guidance": f"This is a reusable {comp_type} component, not a whole page. Provide it as a self-contained HTML fragment (no <html>/<head>/<body> tags) plus matching CSS, followable by the same placeholder rules.",
        })

print(f"Total component variants to generate: {len(COMPONENT_SPECS)}")

ALL_SPECS = TEMPLATE_SPECS + COMPONENT_SPECS
print(f"Grand total generation jobs: {len(ALL_SPECS)}")


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


def extract_placeholders(text):
    return set(re.findall(r"\{\{(\w+)\}\}", text))


def extract_repeat_blocks(text):
    return set(re.findall(r"<!--\s*REPEAT:(\w+)\s*-->", text))


def validate_template(data):
    if "meta" not in data or "html" not in data or "css" not in data:
        raise ValidationError("Missing required top-level keys (meta/html/css)")

    meta = data["meta"]
    html = data["html"]
    css = data["css"]

    for field in ["id", "level", "category", "subcategory", "placeholders"]:
        if field not in meta:
            raise ValidationError(f"meta.json missing required field: {field}")

    placeholders = meta["placeholders"]
    declared_scalar = set(placeholders.get("scalar", []))
    declared_repeating = placeholders.get("repeating", {})

    used_scalar_html = extract_placeholders(html)
    used_scalar_css = extract_placeholders(css)
    used_repeat_blocks_html = extract_repeat_blocks(html)

    declared_repeat_fields = set()
    for block_name, block_info in declared_repeating.items():
        declared_repeat_fields.update(block_info.get("fields", []))

    all_declared = declared_scalar | declared_repeat_fields
    all_used = used_scalar_html | used_scalar_css

    undeclared = all_used - all_declared
    if undeclared:
        raise ValidationError(f"Placeholders used but not declared in meta: {undeclared}")

    unused_scalar = declared_scalar - all_used
    if unused_scalar:
        raise ValidationError(f"Scalar placeholders declared but never used: {unused_scalar}")

    for block_name in declared_repeating:
        if block_name not in used_repeat_blocks_html:
            raise ValidationError(f"Declared repeating block '{block_name}' not found in html")
        if f"<!-- END:{block_name} -->" not in html and f"<!--END:{block_name}-->" not in html:
            raise ValidationError(f"Repeating block '{block_name}' missing matching END marker")

    return True


def parse_and_validate(raw_text):
    cleaned = strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}")
    validate_template(data)
    return data


# ============================================================================
# PROGRESS TRACKING
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
# SAVE TEMPLATE
# ============================================================================

def save_template_files(spec, data):
    category = spec["category"]
    spec_id = spec["id"]
    folder = Path(OUTPUT_DIR) / category / spec_id
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "meta.json", "w") as f:
        json.dump(data["meta"], f, indent=2)
    with open(folder / "index.html", "w") as f:
        f.write(data["html"])
    with open(folder / "style.css", "w") as f:
        f.write(data["css"])

    return str(folder)


# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

def generate_one(spec):
    category = spec["category"]
    variant = spec.get("variant", spec.get("component_type", ""))
    guidance = spec["guidance"]

    prompt = build_prompt(category, variant, guidance)

    last_error = None
    for attempt in range(1, MAX_ATTEMPTS_PER_TEMPLATE + 1):
        try:
            raw_text, model_used = call_gemini_with_fallback(prompt)
            data = parse_and_validate(raw_text)
            print(f"  ✓ {spec['id']} generated by {model_used} (attempt {attempt})")
            return data
        except AllModelsExhaustedError as e:
            print(f"  ✗ {spec['id']}: all models exhausted — stopping batch for now")
            raise
        except ValidationError as e:
            print(f"  ~ {spec['id']} validation failed (attempt {attempt}/{MAX_ATTEMPTS_PER_TEMPLATE}): {e}")
            last_error = e
            continue
        except Exception as e:
            print(f"  ~ {spec['id']} unexpected error (attempt {attempt}/{MAX_ATTEMPTS_PER_TEMPLATE}): {e}")
            last_error = e
            continue

    raise ValidationError(f"Failed after {MAX_ATTEMPTS_PER_TEMPLATE} attempts. Last error: {last_error}")


def run_batch():
    progress = load_progress()
    completed_ids = set(progress["completed"])

    remaining = [s for s in ALL_SPECS if s["id"] not in completed_ids]
    print(f"\n{len(completed_ids)} already completed, {len(remaining)} remaining.\n")

    for i, spec in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] Generating {spec['id']} ...")
        try:
            data = generate_one(spec)
            path = save_template_files(spec, data)
            completed_ids.add(spec["id"])
            progress["completed"] = list(completed_ids)
            save_progress(progress)
            print(f"  Saved to {path}")
        except AllModelsExhaustedError:
            print("\nAll free-tier models are exhausted for today (or failing).")
            print("Progress has been saved. Re-run this workflow tomorrow (or after")
            print("your quota resets) to continue exactly where you left off.")
            sys.exit(0)
        except Exception as e:
            print(f"  FAILED: {spec['id']} — {e}")
            save_failure(spec, e)
            continue

    print("\nAll templates generated (or logged as failures). Check _failures.json for anything to re-run manually.")


# ============================================================================
# UPLOAD TO HUGGING FACE
# ============================================================================

def upload_to_huggingface():
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("⚠️ huggingface_hub not installed. Installing...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    api = HfApi(token=HF_TOKEN)

    try:
        create_repo(repo_id=HF_REPO_ID, token=HF_TOKEN, repo_type=HF_REPO_TYPE, exist_ok=True)
        print(f"✅ Repository {HF_REPO_ID} is ready.")
    except Exception as e:
        print(f"Note: create_repo said: {e} (fine if repo already exists)")

    print(f"📤 Uploading {OUTPUT_DIR} to {HF_REPO_ID} ...")
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message=f"Batch upload: generated template library - {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
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

    print("🚀 Starting template generation...")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    run_batch()

    failures = load_failures()
    if failures:
        print(f"\n⚠️  {len(failures)} templates failed validation after all retries — see {FAILURES_FILE}")
        print("These were NOT uploaded. Review and re-run just those specs manually if needed.")

    print("\n📤 Auto-upload enabled – pushing to HuggingFace...")
    upload_to_huggingface()

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
