"""Regenerate only marked project-tree sections, never the owner-only rules."""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
START = "<!-- PROJECT_STRUCTURE_START -->"
END = "<!-- PROJECT_STRUCTURE_END -->"


def project_tree(root, *, staged=False):
    command = ["git", "-C", str(root), "ls-files", "-z", "--cached"]
    if not staged:
        command += ["--others", "--exclude-standard"]
    paths = subprocess.check_output(command).decode().split("\0")
    tree = {}
    for path in paths:
        if not path or ".DS_Store" in path.split("/"):
            continue
        if not staged and not (root / path).exists():
            continue
        parts = Path(path).parts
        # Show two levels; data and run artifacts stay summarized.
        node = tree
        for part in parts[:2]:
            node = node.setdefault(part, {})
        if len(parts) > 2:
            node.setdefault("…", {})

    def render(node, prefix=""):
        lines = []
        ordered = sorted(node)
        for index, name in enumerate(ordered):
            last = index == len(ordered) - 1
            children = node[name]
            suffix = "/" if children else ""
            lines.append(prefix + ("└── " if last else "├── ") + name + suffix)
            lines.extend(render(children, prefix + ("    " if last else "│   ")))
        return lines

    # A stable root label keeps local clones and CI paths identical.
    return "```text\n./\n" + "\n".join(render(tree)) + "\n```"


def synchronize(root, *, staged=False, check=False):
    generated = project_tree(root, staged=staged)
    changes = []
    for name in ["README.md", "AGENTS.md"]:
        path = root / name
        text = path.read_text(encoding="utf-8")
        if text.count(START) != 1 or text.count(END) != 1:
            raise ValueError(f"{name}: expected exactly one structure marker pair")
        before, rest = text.split(START)
        _, after = rest.split(END)
        updated = before + START + "\n" + generated + "\n" + END + after
        if name == "AGENTS.md":
            def protected(value):
                return value.split("<!-- OWNER_ONLY_RULES_START -->")[1].split("<!-- OWNER_ONLY_RULES_END -->")[0]
            if protected(text) != protected(updated):
                raise ValueError("Refusing to modify owner-only rules")
        if updated != text:
            changes.append(name)
            if not check:
                path.write_text(updated, encoding="utf-8")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--staged", action="store_true", help="Use the Git index as the structure source")
    args = parser.parse_args()
    changes = synchronize(ROOT, staged=args.staged, check=args.check)
    if changes:
        print(("Outdated: " if args.check else "Updated: ") + ", ".join(changes))
    else:
        print("README.md and AGENTS.md structure sections are synchronized")
    return 1 if changes and args.check else 0


if __name__ == "__main__":
    raise SystemExit(main())

