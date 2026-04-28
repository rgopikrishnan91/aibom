"""
BOM Tools CLI — generate AI and Dataset BOMs from the command line.

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


def cmd_generate(args):
    """Generate a BOM for an AI model or dataset."""
    from bom_tools.core.processors import AIBOMProcessor, DATABOMProcessor

    if args.type == "ai":
        if not any([args.repo, args.arxiv, args.github]):
            print("Error: provide at least --repo, --arxiv, or --github", file=sys.stderr)
            sys.exit(1)

        processor = AIBOMProcessor(
            model=args.model,
            mode=args.mode,
            llm_provider=args.provider,
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
            model=args.model,
            mode=args.mode,
            llm_provider=args.provider,
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
        spdx_data = validate_bom_to_spdx(result, bom_type=bom_type, output_path=args.spdx)
        print(f"SPDX 3.0.1 BOM saved to {args.spdx}")


def cmd_serve(args):
    """Start the web UI."""
    from bom_tools.web.app import app

    host = args.host or os.getenv("BOM_HOST", "127.0.0.1")
    port = args.port or int(os.getenv("BOM_PORT", "5000"))

    print(f"\nBOM Tools web UI starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


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
    gen.add_argument("--provider", default="openai", choices=["openai", "ollama", "openrouter"], help="LLM provider (default: openai)")
    gen.add_argument("--model", default="gpt-4o", help="LLM model name (default: gpt-4o)")
    gen.add_argument("--mode", default="rag", choices=["rag", "direct"], help="Extraction mode (default: rag)")
    gen.add_argument("--use-case", default="complete", choices=["complete", "safety", "security", "lineage", "license"], help="Use-case preset (default: complete)")
    gen.add_argument("-o", "--output", help="Output JSON file path (default: stdout)")
    gen.add_argument("--spdx", help="Also generate SPDX 3.0.1 output at this path")

    # --- serve ---
    srv = subparsers.add_parser("serve", help="Start the web UI")
    srv.add_argument("--host", help="Bind address (default: 127.0.0.1)")
    srv.add_argument("--port", type=int, help="Port (default: 5000)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
