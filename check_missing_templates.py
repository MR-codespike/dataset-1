#!/usr/bin/env python3
"""
Check which templates are missing from your HF dataset.
Downloads the current dataset and compares it against the expected spec.
"""

import json
import os
import re
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# ============================================================================
# CONFIG – read from environment (or hardcode for local testing)
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"

BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
LOCAL_DIR = os.path.join(BASE_DIR, "check_temp")

# ============================================================================
# EXPECTED TEMPLATE SPECS – copy exactly from generation script
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

def build_expected_specs():
    specs = []
    # Whole-site templates
    for category, variants in CATEGORIES:
        for variant, guidance in variants:
            variant_slug = re.sub(r"[^a-z0-9]+", "-", variant.lower()).strip("-")
            spec_id = f"{category}-{variant_slug}-v1"
            specs.append({
                "id": spec_id,
                "category": category,
                "variant": variant,
            })
    # Components
    for comp_type, variants in COMPONENT_SPECS_RAW:
        for variant in variants:
            variant_slug = re.sub(r"[^a-z0-9]+", "-", variant.lower()).strip("-")
            spec_id = f"component-{comp_type}-{variant_slug}-v1"
            specs.append({
                "id": spec_id,
                "category": "components",
                "component_type": comp_type,
                "variant": variant,
            })
    return specs

EXPECTED_SPECS = build_expected_specs()
EXPECTED_IDS = {spec["id"] for spec in EXPECTED_SPECS}

# ============================================================================
# Download dataset
# ============================================================================

print(f"📥 Downloading dataset from {HF_REPO_ID} ...")
os.makedirs(LOCAL_DIR, exist_ok=True)
try:
    local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )
    print(f"✅ Downloaded to {local_path}")
except Exception as e:
    print(f"❌ Failed to download: {e}")
    sys.exit(1)

# ============================================================================
# Scan existing templates
# ============================================================================

existing_meta = list(Path(local_path).rglob("meta.json"))
existing_ids = set()
missing_file_issues = []   # templates that have meta.json but missing index.html or style.css

for meta_path in existing_meta:
    folder = meta_path.parent
    with open(meta_path, "r") as f:
        meta = json.load(f)
    tid = meta.get("id", str(folder.relative_to(local_path)))
    existing_ids.add(tid)
    # Check if files are present
    if not (folder / "index.html").exists() or not (folder / "style.css").exists():
        missing_file_issues.append(tid)

# ============================================================================
# Compare with expected
# ============================================================================

missing_ids = EXPECTED_IDS - existing_ids
extra_ids = existing_ids - EXPECTED_IDS

print("\n" + "="*60)
print("AUDIT REPORT")
print("="*60)
print(f"Expected templates: {len(EXPECTED_IDS)}")
print(f"Found templates (with meta.json): {len(existing_ids)}")
print(f"Missing templates: {len(missing_ids)}")
print(f"Extra templates (not in spec): {len(extra_ids)}")
print(f"Templates with missing files (index.html/css): {len(missing_file_issues)}")
print()

if missing_ids:
    print("🔴 MISSING TEMPLATE IDs:")
    for tid in sorted(missing_ids):
        print(f"  - {tid}")
    print()

if extra_ids:
    print("🟡 EXTRA TEMPLATE IDs (not in expected spec):")
    for tid in sorted(extra_ids):
        print(f"  - {tid}")
    print()

if missing_file_issues:
    print("⚠️  TEMPLATES WITH MISSING FILES (meta.json exists but missing index.html or style.css):")
    for tid in sorted(missing_file_issues):
        print(f"  - {tid}")
    print()

# ============================================================================
# Save a detailed report as artifact
# ============================================================================

report = {
    "expected_count": len(EXPECTED_IDS),
    "found_count": len(existing_ids),
    "missing_ids": sorted(missing_ids),
    "extra_ids": sorted(extra_ids),
    "missing_file_issues": sorted(missing_file_issues),
}

report_path = os.path.join(BASE_DIR, "audit_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"📝 Detailed report saved to {report_path}")

if missing_ids:
    print("❌ Some expected templates are missing. You may need to re-run the generation script.")
else:
    print("✅ All expected templates are present.")

print("\nDone.")