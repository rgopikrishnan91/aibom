# Handover: Wire the store-aware plugin path into SPDX/CycloneDX export ("Task 13")

Status: Ready to pick up in a fresh session
Author: Claude (handover from the avid-security PR #55 work)
Date: 2026-05-21
Cross-cutting: affects **both** `license_compat` (PR #54, merged) and `avid_security` (PR #55, open)

---

## 1. The problem in one paragraph

When a user runs `aikaboom generate ... --spdx out.json` (or `--cyclonedx`), the exporter asks every registered plugin for its SPDX contributions — but hands each plugin an **empty findings object** and never runs the plugin's `analyze()` step. So plugins emit nothing into generated documents. License-compatibility annotations and AVID Vulnerability/VEX elements are both silently absent from `generate` output, even though both plugins are installed, enabled, and fully functional via their own CLI/web flows. This is a deliberate placeholder; the code comments call it "Task 13". This handover is the spec to actually wire it.

## 2. Why it's cross-cutting (do it once, both plugins benefit)

The empty-findings shortcut lives in the **core exporters**, not in either plugin:
- `src/aikaboom/utils/spdx_validator.py` → `_emit_plugin_annotations(claim_iri)` (~line 246)
- `src/aikaboom/utils/cyclonedx_exporter.py` → `_collect_plugin_properties(claim_iri)` (~line 219)

Both build a `_EmptyFindings()` and loop plugins calling only the *emit* hooks. Fixing these two helpers (and threading a store + findings into them) makes **every** plugin's findings flow into generated documents — license_compat's `spdx_annotations`/CDX properties AND avid_security's `spdx_elements` (Vulnerability/VEX). Build the plumbing once at the core layer; do not add a plugin-specific shortcut.

## 3. The two-step model (recap)

A plugin contributes to a document in two steps:
1. **Analyze** — `findings = plugin.analyze(store, scope)` walks the BOM graph in the store and produces findings.
2. **Emit** — `plugin.spdx_annotations(claim_iri, findings)` + `plugin.spdx_elements(claim_iri, findings)` (SPDX), or the CDX property hook, turn findings into document nodes.

Today the core does step 2 with empty findings and skips step 1. The fix: run step 1 and feed its output into step 2.

## 4. What already exists (the enablers)

- **`generate` already populates a store.** In `src/aikaboom/cli.py` `cmd_generate`: `_store = BomStore.open()` (~line 210) and the generated BOM is persisted via `_saved_claim_iri = _store.save_claim(result, ...)` (~line 309). So at SPDX-export time there IS a populated `BomStore` and a claim IRI for the just-generated BOM.
- **A proven store-aware analyze pattern exists** in `src/aikaboom/plugins/license_compat/web.py` (~lines 35-40):
  ```python
  from aikaboom.store import BomStore
  store = BomStore.open()
  findings = plugin.analyze(store, Scope.single(iri))
  ```
  Mirror this. `Scope.single(artifact_iri, depth=5)` scopes to one artifact + its lineage; `Scope.graph_wide()` scans the whole store.
- **The `spdx_elements` hook exists** (added in PR #55 to `plugins/base.py`, wired into `_emit_plugin_annotations`). The CDX path has the analogous property hook. No new hooks needed.

## 5. The gap precisely

`_emit_plugin_annotations` is called deep inside the SPDX conversion, with no access to the store:
- `spdx_validator.py:1079` — inside `SPDXValidator._convert_ai_bom()`, `claim_iri=ai_subject_id` where `ai_subject_id = "urn:spdx:AIPackage-{uuid}"`
- `spdx_validator.py:1563` — inside `_convert_dataset_bom()`, `claim_iri=dataset_subject_id`
- CDX: `cyclonedx_exporter.py:436` (`f"ai-model:{model_id}"`) and `:511` (`f"dataset:{dataset_id}"`)

Two things are missing at these call sites: (a) the `BomStore` instance + a `Scope`, and (b) the real store **artifact IRI** for the generated BOM.

## 6. Recommended approach — compute findings once at the orchestration layer

Run `analyze()` where the store naturally lives (`cmd_generate`), then thread the *already-computed* per-plugin findings down into the exporters. This avoids passing the store through many SPDX-internal call frames.

### 6a. In `cmd_generate` (cli.py), after `save_claim`

```python
from aikaboom.plugins import all_plugins, Scope

plugin_findings: dict[str, object] = {}
if _store is not None and _saved_claim_iri is not None and not args.no_security:  # honor opt-outs
    artifact_iri = _artifact_iri_for_claim(_store, _saved_claim_iri)   # see 6c
    scope = Scope.single(artifact_iri) if artifact_iri else Scope.graph_wide()
    for p in all_plugins():
        if not p.enabled():
            continue
        try:
            plugin_findings[p.name] = p.analyze(_store, scope)
        except Exception as e:
            log.warning("plugin %s analyze failed: %s", p.name, e)
```

Pass `plugin_findings` into `validate_bom_to_spdx(...)` and the CDX exporter.

### 6b. Thread `plugin_findings` into the exporters

- `validate_bom_to_spdx(bom_data, ..., plugin_findings=None)` → `SPDXValidator(... )` → `_convert_ai_bom` / `_convert_dataset_bom` → `_emit_plugin_annotations(claim_iri, plugin_findings)`.
- In `_emit_plugin_annotations`, replace `_EmptyFindings()` with the per-plugin findings:
  ```python
  for plugin in all_plugins():
      if not plugin.enabled():
          continue
      findings = (plugin_findings or {}).get(plugin.name) or _EmptyFindings()
      out.extend(plugin.spdx_annotations(claim_iri=claim_iri, findings=findings))
      elem_hook = getattr(plugin, "spdx_elements", None)
      if elem_hook:
          out.extend(elem_hook(claim_iri=claim_iri, findings=findings) or [])
  ```
- Mirror the same `plugin_findings` threading in `cyclonedx_exporter._collect_plugin_properties`.
- Keep the `_EmptyFindings()` fallback so direct `validate_bom_to_spdx` callers (tests, library use) that don't pass findings still work unchanged.

### 6c. Deriving the artifact IRI from the saved claim

`save_claim` returns a *claim* IRI; `analyze`/`Scope.single` want the *artifact* IRI. The store models claim→artifact. Add a small helper (likely already expressible via `BomStore`): given `_saved_claim_iri`, return the artifact IRI it describes (look at how `reconstruct_bom`/`resolve` map claims to artifacts in `store/store.py`, and how license_compat's web view obtains its `iri`). If deriving a single artifact IRI is awkward for v1, fall back to `Scope.graph_wide()` — but note the store is **persistent**, so graph_wide may include artifacts from *prior* generate runs. Prefer `Scope.single` to keep generated SPDX scoped to the current BOM.

## 7. The IRI-alignment wrinkle (MOST IMPORTANT — don't skip)

The emitted security/annotation nodes reference the component they're about, and **those references must match the element IDs the generated SPDX document actually uses**, or you get dangling references.

- The generated SPDX doc gives the model package an SPDX-internal ID: `urn:spdx:AIPackage-{uuid}`.
- But `avid_security`'s emitter (`plugins/avid_security/spdx.py`) builds component references from the **store artifact IRI** the walker produced (e.g. `urn:aibom:pkg:{slug}` style), via `Component.spdx_id`. License_compat's annotations similarly key off store artifact IRIs.
- So a VEX relationship's `to` / `security_assessedElement` (and license annotations' `subject`) will point at IRIs that **do not exist** as elements in the generated doc.

This must be reconciled. Options (decide during implementation):
1. **Pass the SPDX element ID down to analyze/emit** and have emitters use it for the affected-element reference (simplest correctness, but the emitter must learn the mapping store-artifact-IRI → SPDX-element-ID for the artifact under export).
2. **Build a `{store_artifact_iri: spdx_element_id}` map** during SPDX conversion (the converter knows both — it creates the package element from the BOM/claim) and pass it into `_emit_plugin_annotations`; emitters remap their references through it.
3. **Make the generated package element's `spdxId` equal the store artifact IRI** so everything keys off one identifier (largest blast radius; check SPDX validity + existing tests).

Option 2 is likely the cleanest. Whatever is chosen, add a test asserting **every** `from`/`to`/`assessedElement`/`subject` in plugin-emitted nodes resolves to an element that exists in the same `@graph` (no dangling IRIs), and that SHACL still passes.

## 8. Acceptance criteria

- `aikaboom generate --type ai --repo bert-base-uncased --spdx out.json` (with the AVID snapshot present) produces an `out.json` whose `@graph` contains `security_Vulnerability` + a VEX relationship for the matched component, and whose VEX `to`/`assessedElement` resolve to the model's Package element (no dangling IRIs).
- The same run, with license_compat applicable, includes its Annotation(s) attached to the right element.
- `--no-security` (avid) and `AIKABOOM_*_DISABLED` env flags still suppress the respective plugin.
- No-store / library callers of `validate_bom_to_spdx` without `plugin_findings` behave exactly as today (empty-findings fallback).
- SHACL validation passes on the enriched document.
- CycloneDX export carries the analogous plugin properties.
- Full suite green; add an integration test per plugin proving findings now reach generated SPDX.

## 9. Files to touch

- `src/aikaboom/cli.py` — `cmd_generate`: run analyze loop after `save_claim`, derive artifact IRI/scope, pass `plugin_findings` to exporters.
- `src/aikaboom/utils/spdx_validator.py` — thread `plugin_findings` through `validate_bom_to_spdx` → `SPDXValidator._convert_ai_bom`/`_convert_dataset_bom` → `_emit_plugin_annotations`; replace `_EmptyFindings()` with real findings (keep fallback). Resolve the IRI-alignment (§7).
- `src/aikaboom/utils/cyclonedx_exporter.py` — same threading into `_collect_plugin_properties`.
- `src/aikaboom/store/store.py` (maybe) — small claim→artifact-IRI helper if one doesn't already exist.
- Tests: a per-plugin "findings reach generated SPDX" integration test + a "no dangling IRIs" assertion + SHACL pass.

## 10. Reference points (verify line numbers — they drift)

- Empty-findings helpers: `spdx_validator.py:_emit_plugin_annotations` (~246, `_EmptyFindings` ~223); `cyclonedx_exporter.py:_collect_plugin_properties` (~219, `_EmptyFindings` ~197).
- SPDX call sites: `spdx_validator.py:1079` (AI), `:1563` (dataset). CDX: `cyclonedx_exporter.py:436`, `:511`.
- Generate store usage: `cli.py` `_store = BomStore.open()` (~210), `save_claim` (~309), `reconstruct_bom` (~245).
- Working analyze pattern: `plugins/license_compat/web.py:35-40`.
- Plugin protocol: `plugins/base.py` (`analyze`, `spdx_annotations`, `spdx_elements`, `Scope`, `Findings`).
- avid emitter (component IRI source): `plugins/avid_security/spdx.py` (`_slug`, `urn:aibom:*` IDs); walker `Component.spdx_id` = store artifact IRI.

## 11. Out of scope

- New plugin hooks (the contract is already sufficient).
- Recursive/linked-bundle SPDX variants — wire the primary `--spdx` path first; extend after.
- Changing matcher/emitter logic in either plugin — this is purely the orchestration wiring + IRI alignment.
