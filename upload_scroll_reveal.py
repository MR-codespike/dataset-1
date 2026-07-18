#!/usr/bin/env python3
"""
Upload the shared scroll-reveal.js observer script to HF dataset.
Run once — this file doesn't change per-effect.
Reads HF_TOKEN and HF_REPO_ID from environment.
"""

import os
import sys
from huggingface_hub import HfApi

# Read from environment (GitHub Secrets)
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

SCROLL_REVEAL_JS = '''// Shared scroll-reveal observer.
// Include ONCE per page. Works with any onscroll effect that uses the
// fx-scroll-target / fx-in-view class pair.
(function () {
  if (!("IntersectionObserver" in window)) {
    document.querySelectorAll(".fx-scroll-target").forEach(function (el) {
      el.classList.add("fx-in-view");
    });
    return;
  }

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("fx-in-view");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );

  document.querySelectorAll(".fx-scroll-target").forEach(function (el) {
    observer.observe(el);
  });
})();
'''

SCROLL_REVEAL_META = '''{
  "id": "scroll-reveal-observer-v1",
  "type": "shared-utility",
  "name": "Scroll Reveal Observer",
  "description": "Shared IntersectionObserver script used by all onscroll-triggered effects. Adds the fx-in-view class to any element with class fx-scroll-target once it enters the viewport. Include this ONCE per page, regardless of how many onscroll effects are used.",
  "source": "model-generated",
  "reviewed": false,
  "files": ["scroll-reveal.js"],
  "usage_notes": "Include this script once, near the end of <body>. Any onscroll effect's CSS defines the before/after state using .fx-scroll-target and .fx-scroll-target.fx-in-view — this script only handles adding/removing that class as elements enter view."
}'''

api = HfApi(token=HF_TOKEN)

print(f"📤 Uploading scroll-reveal.js to {HF_REPO_ID}/effects/shared/ ...")

api.upload_file(
    path_or_fileobj=SCROLL_REVEAL_JS.encode(),
    path_in_repo="effects/shared/scroll-reveal.js",
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
    commit_message="Add shared scroll-reveal observer script",
)

api.upload_file(
    path_or_fileobj=SCROLL_REVEAL_META.encode(),
    path_in_repo="effects/shared/meta.json",
    repo_id=HF_REPO_ID,
    repo_type=HF_REPO_TYPE,
    commit_message="Add shared scroll-reveal observer metadata",
)

print("✅ Upload complete.")