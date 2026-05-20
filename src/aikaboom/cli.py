"""
AIkaBoOM CLI - generate AI and Dataset BOMs from the command line.

Usage:
    aikaboom generate --type ai --repo microsoft/DialoGPT-medium --spdx out.spdx.json
    aikaboom serve --port 5000

"""

# Note: the langchain_core PendingDeprecationWarning suppression now
# lives in ``aikaboom/__init__.py``. The package init runs before this
# module's top-level statements, and the langgraph import chain that
# fires the warning is triggered from there — putting the filter in
# this CLI module was too late. Phase 10 / Finding #13.

import argparse
import json
import os
import sys

from dotenv import load_dotenv

# ``load_dotenv()`` walks up from this file's directory to find the nearest
# ``.env`` — convenient for users running ``python -m aikaboom.cli`` from any
# subdirectory of the repo, but it makes hermetic subprocess tests difficult
# because the developer's real ``.env`` always wins. The ``BOM_SKIP_DOTENV=1``
# env var lets test runners disable that load explicitly so unit tests can
# assert behaviour with a clean provider-credentials environment.
if os.environ.get("BOM_SKIP_DOTENV", "").lower() not in ("1", "true", "yes"):
    load_dotenv()


# Provider -> (env var, default model) mapping for auto-detection.
# Phase 10 retired the OpenRouter free-tier path (rate-limit usability
# trap; see Phase 10 plan / Findings #6, #12). The OpenRouter default
# is now a paid model from the same family (Llama 3.3 70B) so users
# who set ``OPENROUTER_API_KEY`` without specifying ``--model`` still
# get a sane choice.
PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY", "gpt-4o"),
    "openrouter": ("OPENROUTER_API_KEY", "meta-llama/llama-3.3-70b-instruct"),
    "ollama": ("OLLAMA_BASE_URL", "llama3:8b"),
}


def _detect_available_providers():
    """Return list of providers that have credentials/config available."""
    return [name for name, (env, _) in PROVIDER_ENV.items() if os.getenv(env)]


def _confirm(prompt, default=True):
    """Yes/no prompt. Falls back to default in non-interactive shells."""
    if not sys.stdin.isatty():
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        ans = input(prompt + suffix).strip().lower()
    except EOFError:
        return default
    if not ans:
        return default
    return ans.startswith("y")


def _resolve_provider_and_model(args):
    """Pick a provider and model, prompting the user if multiple keys are set.

    Honors --provider/--model when explicitly given. Otherwise auto-detects
    from environment, falling back to OpenAI only as a last resort.
    Returns (provider, model). May call sys.exit on unresolvable input.
    """
    explicit_provider = args.provider
    explicit_model = args.model

    if explicit_provider:
        # User said exactly which one to use. Verify the key is set.
        env_var, default_model = PROVIDER_ENV[explicit_provider]
        if not os.getenv(env_var):
            print(
                f"Error: --provider {explicit_provider} requires {env_var} to be set "
                f"in the environment or .env file.",
                file=sys.stderr,
            )
            sys.exit(1)
        return explicit_provider, explicit_model or default_model

    # Auto-detect.
    available = _detect_available_providers()
    if not available:
        print(
            "Error: no LLM provider credentials detected.\n"
            "Set one of:\n"
            "  - OPENAI_API_KEY (for OpenAI)\n"
            "  - OPENROUTER_API_KEY (for OpenRouter)\n"
            "  - OLLAMA_BASE_URL (for local Ollama, e.g. http://localhost:11434/v1/)\n"
            "Pass --provider explicitly or copy .env.example to .env and edit.",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(available) == 1:
        chosen = available[0]
        env_var, default_model = PROVIDER_ENV[chosen]
        print(f"Using {chosen} (only provider with {env_var} set).")
        return chosen, explicit_model or default_model

    # Multiple keys. Default to first non-OpenAI in the order user-priority,
    # then ask the user to confirm. This avoids silently using OpenAI when
    # the user has set OPENROUTER_API_KEY too.
    preferred = next((p for p in ["openrouter", "ollama", "openai"] if p in available), available[0])
    if args.yes:
        env_var, default_model = PROVIDER_ENV[preferred]
        print(f"Using {preferred} (auto-selected; --yes skipped prompt).")
        return preferred, explicit_model or default_model

    print(f"Multiple LLM providers configured: {', '.join(available)}")
    if _confirm(f"Use {preferred}?", default=True):
        env_var, default_model = PROVIDER_ENV[preferred]
        return preferred, explicit_model or default_model

    # User said no - ask which one.
    print("Which provider would you like to use?")
    for i, p in enumerate(available, 1):
        print(f"  {i}) {p}")
    try:
        choice = input("Number: ").strip()
        idx = int(choice) - 1
        if not (0 <= idx < len(available)):
            raise ValueError
    except (ValueError, EOFError):
        print("Invalid selection.", file=sys.stderr)
        sys.exit(1)
    chosen = available[idx]
    env_var, default_model = PROVIDER_ENV[chosen]
    return chosen, explicit_model or default_model


_MISSING_VALUE_MARKERS = {"", "noassertion", "not found", "not found.", "n/a", "none"}


def _summarise_rag_fields(result):
    """Return (populated_keys, missing_keys) for the RAG section of a BOM.

    A field is "populated" iff its value is non-empty and not one of the
    no-information sentinels (``noAssertion``, ``Not found.`` etc.). Used to
    print a per-run summary so a partially-failed run isn't silent — the
    BOM file looks fine but X of N fields had no extractable answer.
    """
    rag = (result or {}).get("rag_fields") or {}
    populated, missing = [], []
    for key, triplet in rag.items():
        value = triplet.get("value") if isinstance(triplet, dict) else triplet
        if value is None:
            missing.append(key)
            continue
        text = str(value).strip().lower()
        if text in _MISSING_VALUE_MARKERS:
            missing.append(key)
        else:
            populated.append(key)
    return populated, missing


def cmd_generate(args):
    """Generate a BOM for an AI model or dataset."""
    if getattr(args, "linked_bom_output", None) and not args.recursive_bom:
        print(
            "Error: --linked-bom-output requires --recursive-bom (the linked "
            "bundle is built from the recursive walk).",
            file=sys.stderr,
        )
        sys.exit(2)

    provider, model = _resolve_provider_and_model(args)
    print(f"Provider: {provider} | Model: {model} | Mode: {args.mode}")

    from aikaboom.core.agentic_rag import get_fixed_questions
    from aikaboom.core.processors import AIBOMProcessor, DATABOMProcessor
    from aikaboom.utils.use_case import (
        filter_questions_by_use_case,
        normalize_use_case,
    )

    normalized_use_case = normalize_use_case(args.use_case, args.type)
    questions_config = filter_questions_by_use_case(
        normalized_use_case, args.type, get_fixed_questions(args.type),
    )
    if normalized_use_case != "complete":
        print(
            f"Use case: {normalized_use_case} ({len(questions_config)} of "
            f"{len(get_fixed_questions(args.type))} fields)"
        )

    # --- worldofBOMs cache resolution ---
    # ``Identifier`` is needed both inside and outside the store-open branch
    # (for ``_idents`` construction and post-generation persistence), so keep
    # it imported at the outer level. The heavier store imports stay inside
    # the ``else:`` branch so ``AIKABOOM_GRAPH_DISABLE=1`` remains a
    # lightweight bypass.
    from aikaboom.store.naming import Identifier

    _graph_disabled = os.environ.get("AIKABOOM_GRAPH_DISABLE") == "1"
    if _graph_disabled:
        _store = None
    else:
        from aikaboom.store.cache_resolver import CachePolicy, decide, is_interactive
        from aikaboom.store.store import BomStore
        from aikaboom.store.trust import VoteKind
        try:
            _store = BomStore.open()
        except Exception as e:
            print(
                f"warning: graph store unavailable, continuing without cache: {e}",
                file=sys.stderr,
            )
            _store = None
    _claim_to_use = None
    _idents: list[Identifier] = []

    if _store is not None:
        if args.type == "ai" and args.repo:
            _idents.append(Identifier("huggingface", args.repo))
        if getattr(args, "hf_url", None):
            _idents.append(Identifier("huggingface", args.hf_url))
        if args.arxiv:
            _idents.append(Identifier("arxiv", args.arxiv))
        if args.github:
            _idents.append(Identifier("github", args.github))

        if _idents:
            _resolve = _store.resolve(
                identifiers=_idents,
                use_case=normalized_use_case,
                mode=args.mode,
            )
            _policy = CachePolicy(args.cache)
            _decision = decide(
                _resolve, _policy, interactive=is_interactive(), planned_llm=model,
            )
            if _decision == "use" and _resolve.matching_claims:
                _claim_to_use = _resolve.matching_claims[0]["iri"]

    if _claim_to_use is not None:
        # Reconstruct the cached BOM from the graph.
        result = _store.reconstruct_bom(_claim_to_use)
        # Record implicit-use vote (silent positive signal).
        _store.record_trust_vote(_claim_to_use, VoteKind.IMPLICIT_USE)
        # Tag the result so downstream code knows it's a cache hit.
        result["_cached"] = True
        result["_claim_iri"] = _claim_to_use
        _skip_generation = True
    else:
        _skip_generation = False

    if not _skip_generation:
        if args.type == "ai":
            if not any([args.repo, args.arxiv, args.github]):
                print("Error: provide at least --repo, --arxiv, or --github", file=sys.stderr)
                sys.exit(1)

            # Be lenient: users routinely pass the full HF URL to --repo.
            # Normalise so downstream code only ever sees the canonical
            # ``namespace/repo`` form. Mirrors the dataset path's existing
            # ``hf_url`` normalisation in core/processors.py.
            repo_id = args.repo
            if repo_id and "huggingface.co" in repo_id:
                from aikaboom.utils.metadata_fetcher import MetadataFetcher
                normalised = MetadataFetcher.extract_repo_id_from_hf_url(repo_id)
                if normalised:
                    repo_id = normalised

            processor = AIBOMProcessor(
                model=model,
                mode=args.mode,
                llm_provider=provider,
                use_case=normalized_use_case,
                questions_config=questions_config,
            )
            result = processor.process_ai_model(
                repo_id=repo_id,
                arxiv_url=args.arxiv,
                github_url=args.github,
            )
        else:
            if not any([args.hf_url, args.arxiv, args.github]):
                print("Error: provide at least --hf-url, --arxiv, or --github", file=sys.stderr)
                sys.exit(1)

            processor = DATABOMProcessor(
                model=model,
                mode=args.mode,
                llm_provider=provider,
                use_case=normalized_use_case,
                questions_config=questions_config,
            )
            result = processor.process_dataset(
                arxiv_url=args.arxiv,
                github_url=args.github,
                hf_url=args.hf_url,
            )

    # Persist the freshly generated BOM (skipped on cache hit).
    # Persistence is best-effort: if the store backend errors after the LLM
    # cost has already been paid, surface a warning but never destroy the
    # generated BOM by raising before the JSON output is written.
    _saved_claim_iri = None
    if _store is not None and not _skip_generation:
        try:
            _saved_claim_iri = _store.save_claim(
                result,
                run_meta={
                    "provider": provider, "llm_model": model,
                    "prompt_version": "v1", "code_version": "head",
                    "mode": args.mode, "use_case": normalized_use_case,
                },
                identifiers=_idents,
            )
        except Exception as e:
            print(
                f"warning: failed to persist claim to graph store: {e}",
                file=sys.stderr,
            )

    # Write JSON output
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"BOM saved to {args.output}")
    else:
        print(output_json)

    # Per-field RAG summary so a partially-failed run isn't silent.
    populated, missing = _summarise_rag_fields(result)
    total = len(populated) + len(missing)
    if total:
        msg = f"RAG fields: {len(populated)} of {total} populated"
        if missing:
            preview = ", ".join(missing[:5])
            suffix = f"; missing: {preview}"
            if len(missing) > 5:
                suffix += f" (+{len(missing) - 5} more)"
            msg += suffix
        print(msg)

    # Phase 12C — conflict summary so the CLI surfaces both the count
    # and the high/low confidence breakdown alongside RAG coverage.
    from aikaboom.core.conflict_routing import summarise_conflicts
    cs = result.get("conflict_summary") if isinstance(result, dict) else None
    if not cs:
        cs = summarise_conflicts(result)
    if cs and (cs.get("total") or cs.get("suppressed")):
        # The summariser already separates real conflicts (counted in
        # ``total``) from noise (``suppressed`` — grounding < 0.1, kept
        # in the BOM trace but excluded from the headline). Surface both.
        deterministic = cs.get("deterministic", 0)
        if not deterministic and cs.get("total") is not None:
            # Back-compat fallback if an older summariser ran.
            deterministic = cs["total"] - cs.get("high_confidence", 0) - cs.get("low_confidence", 0)
        parts = []
        if cs.get("high_confidence"):
            parts.append(f"{cs['high_confidence']} high-confidence")
        if cs.get("low_confidence"):
            parts.append(f"{cs['low_confidence']} low-confidence")
        if deterministic:
            parts.append(f"{deterministic} deterministic")
        breakdown = f" ({', '.join(parts)})" if parts else ""
        suppressed = cs.get("suppressed", 0)
        suppressed_note = f"; {suppressed} suppressed (low grounding)" if suppressed else ""
        print(f"Conflicts: {cs.get('total', 0)} total{breakdown}{suppressed_note}")

    bom_type = "ai" if args.type == "ai" else "data"

    # Optionally convert to SPDX
    if args.spdx:
        from aikaboom.utils.spdx_validator import validate_bom_to_spdx, validate_spdx_export
        spdx_data = validate_bom_to_spdx(
            result,
            bom_type=bom_type,
            output_path=args.spdx,
            validate=False,
            relationship_cap=args.spdx_relationship_cap,
        )
        print(f"SPDX 3.0.1 BOM saved to {args.spdx}")
        if args.validate_spdx:
            validation = validate_spdx_export(
                spdx_data,
                strict=args.strict_spdx_validation,
                bom_type=bom_type,
            )
            if validation["valid"]:
                beta = " beta" if validation.get("beta") else ""
                print(f"SPDX validation passed ({validation['validator']}{beta})")
                # Implicit-validate: silent positive signal when a freshly
                # generated claim's SPDX export validates. Best-effort: a
                # trust-write failure must never block generation output.
                if (
                    _store is not None
                    and not _skip_generation
                    and _saved_claim_iri is not None
                ):
                    try:
                        _store.record_trust_vote(
                            _saved_claim_iri, VoteKind.IMPLICIT_VALIDATE,
                        )
                    except Exception:
                        pass
            else:
                beta = " beta" if validation.get("beta") else ""
                print(
                    f"SPDX validation failed ({validation['validator']}{beta}) "
                    f"with {len(validation['errors'])} error(s)"
                )
                for error in validation["errors"][:10]:
                    print(f"  - {error}")

    # Optionally convert to CycloneDX
    if args.cyclonedx:
        from aikaboom.utils.cyclonedx_exporter import bom_to_cyclonedx
        from aikaboom.utils.cyclonedx_validator import validate_cyclonedx
        bom_to_cyclonedx(result, bom_type=bom_type, output_path=args.cyclonedx)
        print(f"CycloneDX 1.6 BOM saved to {args.cyclonedx} (beta)")
        result.setdefault("beta_fields", []).append("cyclonedx")

        # Authoritative validation via sbom-utility if available
        cdx_validation = validate_cyclonedx(args.cyclonedx)
        if cdx_validation["valid"] is True:
            print(f"CycloneDX validation passed ({cdx_validation['validator']})")
        elif cdx_validation["valid"] is False:
            print(
                f"CycloneDX validation failed ({cdx_validation['validator']}) "
                f"with {len(cdx_validation['errors'])} error(s)"
            )
            for err in cdx_validation["errors"][:10]:
                print(f"  - {err}")
        else:
            print(f"CycloneDX validation skipped: {cdx_validation['reason']}")

    if args.recursive_bom:
        from aikaboom.utils.recursive_bom import (
            EXHAUST_DEPTH,
            build_linked_spdx_bundle,
            generate_recursive_boms,
            linked_bundle_summary,
        )
        from aikaboom.utils.recursive_enrich import build_enrich_fn
        from aikaboom.utils.spdx_validator import validate_spdx_export

        depth_arg = (args.recursive_depth or "1").strip().lower()
        if depth_arg in ("all", "exhaust"):
            recursive_max_depth = EXHAUST_DEPTH
            depth_label = f"all (capped at {args.recursive_safety_cap} nodes)"
        else:
            try:
                recursive_max_depth = max(0, int(depth_arg))
            except ValueError:
                print(
                    f"Error: --recursive-depth must be an integer or 'all', "
                    f"got {args.recursive_depth!r}",
                    file=sys.stderr,
                )
                sys.exit(1)
            depth_label = str(recursive_max_depth)

        print(
            "[recursive-bom beta] Walking the dependency tree. Each discovered "
            "trainedOn / testedOn / dependsOn target produces a fully-enriched "
            "child BOM via the same RAG pipeline. Targets are deduplicated and "
            "the walk stops at --recursive-depth or when the unique-target set "
            "is exhausted."
        )
        enrich_fn = build_enrich_fn(
            use_case=args.use_case,
            mode=args.mode,
            llm_provider=provider,
            model=model,
        )
        recursive_result = generate_recursive_boms(
            result,
            bom_type=bom_type,
            max_depth=recursive_max_depth,
            breadth=args.recursive_breadth,
            safety_cap=args.recursive_safety_cap,
            validate_spdx=args.validate_spdx,
            strict_spdx=args.strict_spdx_validation,
            enrich_fn=enrich_fn,
        )
        result.setdefault("beta_fields", []).append("recursive_bom")
        if args.recursive_output:
            with open(args.recursive_output, "w", encoding="utf-8") as f:
                json.dump(recursive_result, f, indent=2, ensure_ascii=False)
            print(f"Recursive BOM bundle saved to {args.recursive_output} (beta)")
        enriched_count = sum(
            1 for n in recursive_result["generated"] if n.get("enriched")
        )
        conflict_count = len(recursive_result.get("conflict_walked")
                              or recursive_result.get("skipped_due_to_conflict") or [])
        print(
            f"Recursive BOM beta (depth={depth_label}) walked "
            f"{recursive_result['deepest_level_reached']} level(s) and emitted "
            f"{recursive_result['generated_count']} child BOM(s) "
            f"({enriched_count} enriched); "
            f"{conflict_count} field(s) flagged for conflict (walked with ⚠)."
        )

        if args.linked_bom_output:
            linked = build_linked_spdx_bundle(result, recursive_result, bom_type=bom_type)
            with open(args.linked_bom_output, "w", encoding="utf-8") as f:
                json.dump(linked, f, indent=2, ensure_ascii=False)
            result.setdefault("beta_fields", []).append("linked_spdx_bundle")
            summary = linked_bundle_summary(linked, recursive_result)
            print(
                f"Linked SPDX BOM bundle saved to {args.linked_bom_output} (beta) — "
                f"{summary['recursive_edge_count']} dependency edge(s) across "
                f"{summary['node_count']} elements."
            )
            if args.validate_spdx:
                v = validate_spdx_export(
                    linked, strict=args.strict_spdx_validation, bom_type=bom_type,
                )
                if v["valid"]:
                    beta = " beta" if v.get("beta") else ""
                    print(f"Linked SPDX bundle validation passed ({v['validator']}{beta})")
                else:
                    beta = " beta" if v.get("beta") else ""
                    print(
                        f"Linked SPDX bundle validation failed ({v['validator']}{beta}) "
                        f"with {len(v['errors'])} error(s)"
                    )
                    for error in v["errors"][:10]:
                        print(f"  - {error}")


def cmd_serve(args):
    """Start the web UI."""
    from aikaboom.web.app import app

    host = args.host or os.getenv("BOM_HOST", "127.0.0.1")
    port = args.port or int(os.getenv("BOM_PORT", "5000"))

    print(f"\nBOM Tools web UI starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


def cmd_list_models(args):
    """Print available models for the requested provider."""
    if args.provider != "openrouter":
        print(f"Error: list-models only supports openrouter (got {args.provider})", file=sys.stderr)
        sys.exit(1)

    # Phase 10 retired the free-tier filter (Findings #6, #12) — the
    # picker mislabeled non-chat models and the rate-limited free path
    # was a usability trap. ``list-models`` now returns the full
    # OpenRouter catalogue; users pick a paid model id explicitly.
    from aikaboom.utils.openrouter_models import list_openrouter_models
    models = list_openrouter_models()
    if args.limit:
        models = models[: args.limit]

    if args.json:
        print(json.dumps(models, indent=2))
        return

    if not models:
        print("No models returned.")
        return

    # Plain-text table
    print(f"{'ID':<55} {'CTX':<8} {'PRICING':<25} NAME")
    print("-" * 110)
    for m in models:
        ctx = m.get("context_length")
        ctx_s = f"{ctx//1000}K" if isinstance(ctx, int) and ctx >= 1000 else (str(ctx) if ctx else "-")
        p = m.get("pricing") or {}
        pricing_s = f"in:{p.get('prompt', '?')} out:{p.get('completion', '?')}"
        print(f"{m.get('id', '')[:54]:<55} {ctx_s:<8} {pricing_s:<25} {m.get('name', '')}")


def main():
    parser = argparse.ArgumentParser(
        prog="aikaboom",
        description="AIkaBoOM - aggregate, align, and resolve conflicting metadata "
                    "across the AI supply chain. Generates AI BOMs and Dataset BOMs "
                    "with SPDX 3.0.1 export.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- generate ---
    gen = subparsers.add_parser("generate", help="Generate a BOM")
    gen.add_argument("-t", "--type", required=True, choices=["ai", "data"], help="BOM type")
    gen.add_argument("--repo", help="HuggingFace repo ID (e.g. microsoft/DialoGPT-medium)")
    gen.add_argument("--hf-url", help="HuggingFace dataset URL")
    gen.add_argument("--arxiv", help="arXiv paper URL")
    gen.add_argument("--github", help="GitHub repo URL")
    gen.add_argument(
        "--provider",
        default=None,
        choices=["openai", "ollama", "openrouter"],
        help="LLM provider. If omitted, auto-detected from environment "
             "(OPENAI_API_KEY / OPENROUTER_API_KEY / OLLAMA_BASE_URL).",
    )
    gen.add_argument(
        "--model",
        default=None,
        help="LLM model name. If omitted, a sensible default for the chosen provider is used.",
    )
    gen.add_argument("--mode", default="rag", choices=["rag", "direct"], help="Extraction mode (default: rag)")
    gen.add_argument(
        "--use-case",
        default="complete",
        choices=["complete", "safety", "security", "lineage", "license"],
        help="Use-case preset (default: complete)",
    )
    gen.add_argument("-o", "--output", help="Output JSON file path (default: stdout)")
    gen.add_argument("--spdx", help="Also generate SPDX 3.0.1 output at this path")
    gen.add_argument(
        "--no-validate-spdx",
        dest="validate_spdx",
        action="store_false",
        help="Skip validation of generated SPDX output.",
    )
    gen.add_argument(
        "--strict-spdx-validation",
        action="store_true",
        help="Run slower SHACL validation after the default SPDX JSON Schema validation (beta).",
    )
    gen.set_defaults(validate_spdx=True)
    gen.add_argument(
        "--spdx-relationship-cap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the number of trainedOn/testedOn/dependsOn child packages "
            "emitted per relationship in the standalone SPDX BOM (default: "
            "unbounded). The recursive walker and linked SPDX bundle are "
            "always unbounded; this flag only affects the parent SPDX."
        ),
    )
    gen.add_argument("--cyclonedx", help="Also generate CycloneDX 1.6 output at this path (beta)")
    gen.add_argument(
        "--recursive-bom",
        action="store_true",
        help="Generate child BOM seed exports from trainedOn/testedOn/dependsOn targets (beta).",
    )
    gen.add_argument(
        "--recursive-depth",
        type=str,
        default="1",
        help=(
            "Maximum dependency-tree depth for --recursive-bom (default: 1, beta). "
            "Pass an integer or 'all' / 'exhaust' to walk until the unique-target "
            "set empties or --recursive-safety-cap is reached. Walking deeper "
            "extracts BOMs for the dependencies of each dependency, which can be "
            "expensive."
        ),
    )
    gen.add_argument(
        "--recursive-breadth",
        type=int,
        default=10,
        help=(
            "Maximum children expanded per node (per-node fan-out) under "
            "--recursive-bom. Default: 10. Targets are discovered "
            "lineage-first, so the dependsOn edge always survives the "
            "budget; surplus trainedOn/testedOn dataset edges are recorded "
            "as breadth-capped."
        ),
    )
    gen.add_argument(
        "--recursive-safety-cap",
        type=int,
        default=200,
        help=(
            "Absolute ceiling on the total number of child BOMs generated "
            "across the whole tree — a last-resort runaway guard. Default: "
            "200. Per-node fan-out is bounded by --recursive-breadth and "
            "tree levels by --recursive-depth."
        ),
    )
    gen.add_argument(
        "--recursive-output",
        help="Write the recursive BOM beta bundle (per-child JSON) to this file.",
    )
    gen.add_argument(
        "--linked-bom-output",
        help=(
            "Write a single linked SPDX BOM (parent + every recursive child + "
            "Relationship edges) to this JSON-LD file (beta)."
        ),
    )
    gen.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip provider confirmation prompt when multiple keys are set.",
    )
    gen.add_argument(
        "--cache",
        choices=["use", "regen", "prompt", "auto"],
        default=os.environ.get("AIKABOOM_CACHE_POLICY_DEFAULT") or
                ("prompt" if sys.stdin.isatty() else "use"),
        help="Cache resolution policy when an existing BOM is found.",
    )
    gen.add_argument(
        "--min-trust",
        type=float,
        default=0.0,
        help="Recursive walks: skip child claims with trustScore below this.",
    )
    gen.add_argument(
        "--regen-on-low-trust",
        action="store_true",
        help="Recursive walks: regenerate child BOMs below --min-trust instead of skipping.",
    )
    gen.add_argument(
        "--primary-platform",
        choices=["huggingface", "github", "arxiv", "doi", "url"],
        default=None,
        help="Override which input is treated as the primary identifier.",
    )
    # --- serve ---
    srv = subparsers.add_parser("serve", help="Start the web UI")
    srv.add_argument("--host", help="Bind address (default: 127.0.0.1)")
    srv.add_argument("--port", type=int, help="Port (default: 5000)")

    # --- list-models ---
    lm = subparsers.add_parser("list-models", help="List available models for a provider")
    lm.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter"],
        help="Provider to list models for (default: openrouter).",
    )
    lm.add_argument("--limit", type=int, default=None, help="Max number of models to show.")
    lm.add_argument("--json", action="store_true", help="Output JSON instead of a table.")

    # --- graph / bom subcommands ---
    from aikaboom.store.cli_graph import register_subparsers as _register_graph_subparsers
    _register_graph_subparsers(subparsers)

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    elif args.command in ("graph", "bom"):
        return args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
