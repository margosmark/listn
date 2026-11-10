# F1 evidence — torch 2.8.0 → ≥2.13.0 (card t_ac75a1de)

## Versions (old → new)

| package | old (uv.lock) | new (uv.lock) |
|---|---|---|
| torch | 2.8.0 | **2.14.1** (python_full_version < 3.11) / **2.15.0** (≥ 3.11) |
| torchaudio | 2.8.0 | **2.11.1** |
| torchcodec | 0.7.0 | **0.17.0** |
| pyannote.audio | 4.0.1 → (unchanged range) | 4.0.7 (unchanged, already locked) |

Specifier change in `pyproject.toml` `[project.optional-dependencies].diarize`:
- `torch>=2.8` → `torch>=2.13`
- `torchaudio>=2.8` → `torchaudio>=2.11`
- `torchcodec>=0.7,<0.8` → `torchcodec>=0.12,<0.18`

Why these numbers:
- `torch>=2.13` is the highest fix version among the 9 advisories on 2.8.0 (PYSEC-2025-194), so ≥2.13 clears all of them.
- torchaudio's newest PyPI release is 2.11.1 (declares `torch>=2.11`); there is no 2.13 torchaudio.
- torchcodec's compatibility table (github.com/pytorch/torchcodec README): 0.12–0.17 all target `torch>=2.11`; 0.7 was the torch-2.8 line.
- uv resolves per-python: torch 2.15.0 requires Python ≥3.11, so Python 3.10 gets 2.14.1 (both ≥2.13, both clear the advisories).

## Acceptance 2 — pip-audit

Baseline (before, reproduced on this branch at origin/main content):
```
uvx --python 3.12 --from pip-audit pip-audit -r requirements-all.txt --no-deps
→ rc=1, "Found 9 known vulnerabilities in 1 package"
  torch 2.8.0: PYSEC-2025-203/204/206 (fix 2.9.0), PYSEC-2025-193 (2.9.1),
  PYSEC-2025-194 (2.13.0), PYSEC-2025-195 (2.10.0),
  PYSEC-2026-2286 + PYSEC-2026-1856 (2.10.0, checkpoint RCE CVE-2026-24747),
  PYSEC-2026-139 (no fix)
```
After (`uv lock` + `uv export --no-hashes --no-dev --no-emit-project --all-extras -o requirements-all.txt`):
```
uvx --python 3.12 --from pip-audit pip-audit -r requirements-all.txt --no-deps
→ rc=0, "No known vulnerabilities found"
```
Full logs: `pip-audit-baseline.txt`, `pip-audit-after.txt`.

### CVE-2026-4538 (PYSEC-2026-139) — known/accepted, but NOT present after the bump
OSV record (https://osv.dev/vulnerability/PYSEC-2026-139, aliases CVE-2026-4538 /
BIT-pytorch-2026-4538) has range `introduced: 0, last_affected: 2.10.0` — i.e. it
only affects torch ≤2.10.0, so torch 2.14.1/2.15.0 are out of range and pip-audit
does not report it. No torch advisory remains open for listn. (Upstream fix is
still unpublished; if a future torch regression re-opens it, pip-audit will flag it.)

## Acceptance 3 — diarize smoke (run in this workspace, `.venv` python 3.12)
```
uv sync --group dev --extra diarize            → rc=0
uv run python -c "import torch, torchaudio, torchcodec, pyannote.audio"
→ torch 2.15.0+cu132
  torchaudio 2.11.1+cu132
  torchcodec 0.17.0+cu130
  pyannote.audio 4.0.7  (Model import OK)
→ SMOKE_RC=0   (log: smoke.log)
```

## Tests
- `uv run pytest tests/test_build_speakers.py -q` → **8 passed** (updated
  `test_the_diarize_extra_is_pyannote_4_with_a_matching_torch`, which hard-asserted
  the old `>=2.8` pins, to assert `>=2.13` / `torchaudio>=2.11` / `torchcodec>=0.12`).
- Full suite on this branch: `651 passed, 12 failed, 4 skipped` (pytest2.log).
- Full suite at base origin/main with the same venv: `657 passed, 6 failed` (pytest-base-full.log).
- The failing families are pipewire/tray_sni/capture_contract audio-session tests that
  differ run-to-run and also fail at base; e.g. `test_sni_idle_properties_and_menu`
  fails at base in isolation too. None touch torch/diarize. The one real failure
  caused by this change (the pin assertion) is fixed and green.

## Files changed
- `pyproject.toml` — diarize extra specifiers + comment
- `uv.lock` — re-locked (torch/torchaudio/torchcodec + nvidia cu12→cu13 deps, triton)
- `requirements-all.txt` — re-exported; **`requirements.txt` (base) unchanged**
- `tests/test_build_speakers.py` — pin assertions updated to the new floor
