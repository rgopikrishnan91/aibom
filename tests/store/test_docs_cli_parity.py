"""Every command shown in CLI.md exists in the argparse tree."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_MD = REPO_ROOT / "docs" / "worldofboms" / "CLI.md"


def test_cli_md_subcommands_match_argparse():
    text = CLI_MD.read_text()
    # Each h3 starting with `### ` is a documented subcommand.
    documented = set(re.findall(r"^### (\w+)", text, re.MULTILINE))
    # Build the actual subcommand set.
    from aikaboom.store.cli_graph import register_subparsers
    import argparse

    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command")
    register_subparsers(subs)
    actual = set()
    # `graph` and `bom` are top-level groups; collect their sub-subcommands.
    for action in subs._name_parser_map.values():
        for sub_action in action._actions:
            if isinstance(sub_action, argparse._SubParsersAction):
                for name in sub_action._name_parser_map.keys():
                    actual.add(name)
    missing_in_doc = actual - documented
    assert not missing_in_doc, f"Undocumented subcommands: {missing_in_doc}"
