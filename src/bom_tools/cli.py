"""
BOM Tools CLI - generate AI and Dataset BOMs from the command line.

Usage:
    bom-tools generate --type ai --repo microsoft/DialoGPT-medium --spdx out.spdx.json
    bom-tools serve --port 5000
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# Provider -> (env var, default model) mapping for auto-detection.
# OpenRouter default uses the :free variant so users without credits can run
# the tool out of the box.
PROVIDER_ENV = {
    "openai": ("OPENAI_API_KEY", "gpt-4o"),
    "openrouter": ("OPENROUTER_API_KEY", "qwen/qwen-2.5-72b-instruct:free"),
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
            "  - OPENROUTER_API_KEY (for OpenRouter; free models available)\n"
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


def cmd_generate(args):
    """Generate a BOM for an AI model or dataset."""
    if getattr(args, "pick_free_model", False):
        if args.model:
            print("Error: --pick-free-model is mutually exclusive with --model", file=sys.stderr)
            sys.exit(1)
        # If provider isn't openrouter, force-select openrouter so the
        # picker has somewhere to dispatch to.
        if args.provider and args.provider != "openrouter":
            print(f"Error: --pick-free-model requires --provider openrouter (got {args.provider})", file=sys.stderr)
            sys.exit(1)
        args.provider = "openrouter"
        from bom_tools.utils.openrouter_models import pick_free_openrouter_model
        args.model = pick_free_openrouter_model()
        print(f"Picked free OpenRouter model: {args.model}")

    provider, model = _resolve_provider_and_model(args)
    print(f"Provider: {provider} | Model: {model} | Mode: {args.mode}")

    from bom_tools.core.processors import AIBOMProcessor, DATABOMProcessor

    if args.type == "ai":
        if not any([args.repo, args.arxiv, args.github]):
            print("Error: provide at least --repo, --arxiv, or --github", file=sys.stderr)
            sys.exit(1)

        processor = AIBOMProcessor(
            model=model,
            mode=args.mode,
            llm_provider=provider,
            use_case=args.use_case,
        )
        result = processor.process_ai_model(
            repo_id=args.repo,
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
            use_case=args.use_case,
        )
        result = processor.process_dataset(
            arxiv_url=args.arxiv,
            github_url=args.github,
            hf_url=args.hf_url,
        )

    # Write JSON output
    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"BOM saved to {args.output}")
    else:
        print(output_json)

    # Optionally convert to SPDX
    if args.spdx:
        from bom_tools.utils.spdx_validator import validate_bom_to_spdx

        bom_type = "ai" if args.type == "ai" else "data"
        validate_bom_to_spdx(result, bom_type=bom_type, output_path=args.spdx)
        print(f"SPDX 3.0.1 BOM saved to {args.spdx}")


def cmd_serve(args):
    """Start the web UI."""
    from bom_tools.web.app import app

    host = args.host or os.getenv("BOM_HOST", "127.0.0.1")
    port = args.port or int(os.getenv("BOM_PORT", "5000"))

    print(f"\nBOM Tools web UI starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


def cmd_list_models(args):
    """Print available models for the requested provider."""
    if args.provider != "openrouter":
        print(f"Error: list-models only supports openrouter (got {args.provider})", file=sys.stderr)
        sys.exit(1)

    from bom_tools.utils.openrouter_models import (
        list_free_openrouter_models,
        list_openrouter_models,
    )
    fn = list_free_openrouter_models if args.free else list_openrouter_models
    models = fn()
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
        prog="bom-tools",
        description="Generate Software Bills of Materials for AI models and datasets.",
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
        "-y", "--yes",
        action="store_true",
        help="Skip provider confirmation prompt when multiple keys are set.",
    )
    gen.add_argument(
        "--pick-free-model",
        action="store_true",
        help="Auto-select the highest-context free model from OpenRouter "
             "(forces --provider openrouter; mutually exclusive with --model).",
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
    lm.add_argument("--free", action="store_true", help="Only show free models.")
    lm.add_argument("--limit", type=int, default=None, help="Max number of models to show.")
    lm.add_argument("--json", action="store_true", help="Output JSON instead of a table.")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
