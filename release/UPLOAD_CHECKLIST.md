**日本語版 -> [UPLOAD_CHECKLIST.ja.md](UPLOAD_CHECKLIST.ja.md)**

# Publication Checklist

The initial publication completed the review and approval process. Keep this order for later releases.

1. Confirm that the GitHub/Hugging Face namespace is `PrismPhi`.
2. Pass `python3 scripts/audit_release.py --assets-dir /path/to/release-assets`.
3. Match SHA between `runner/config/model-assets.json` and the model-distribution `asset-manifest.json`.
4. Confirm model staging contains only 11 assets and no EPContext/`*.bin`/GGUF/QNN library.
5. Create the GitHub repository and push code/records/docs only.
6. Create the Hugging Face model repository.
7. Upload the 11 model-staging assets to Hugging Face.
8. Re-download the 11 assets from the published `asset-manifest.json` and re-verify size and SHA-256.
9. Upload `release/MODEL_CARD.md` as the Hugging Face `README.md`. Also retain `MODEL_CARD.md` and `MODEL_CARD.ja.md` under their original names.
10. Verify the reciprocal link from GitHub README to the Hugging Face model card and from the Hugging Face model card to the GitHub repository.
11. Pin the published revision in `runner/config/model-assets.json` and resolve all 11 asset URLs.
12. Re-run `runner/install.sh` into a new Q6A state from the public URL.
13. Confirm the runtime fingerprint, strict finite generation/reload, and post-profile QNN-only fields in install evidence.
14. Require normal `Tokyo`, JSON `{"answer":"Tokyo"}`, and first token ID `40550`; do not accept syntax-only JSON.
15. Check every public Markdown URL and all 11 pinned asset URLs for a non-error HTTP result.
16. Copy fresh time, normal response, JSON, and QNN-only profile into the release note.
17. Register V2a/V2b as the next project.

Use [PUBLIC_SCOPE.md](../PUBLIC_SCOPE.md) as the publication-review baseline.
