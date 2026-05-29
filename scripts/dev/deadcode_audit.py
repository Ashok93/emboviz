"""Static dead-code audit: which emboviz modules are unreachable from the
LIVE entry points (the registered `emboviz` CLI commands + the analyze
engine), following EVERY import statement (eager and lazy/in-function),
parent packages, and `from pkg import submodule` edges.

Prints the dead set with each module's importers, so a human can validate
before deleting. Run: uv run --no-sync python scripts/dev/deadcode_audit.py
"""
from __future__ import annotations
import ast, pathlib

ROOT = pathlib.Path("emboviz")


def mod_of(p: pathlib.Path) -> str:
    m = str(p.with_suffix("")).replace("/", ".")
    return m[:-9].rstrip(".") if m.endswith(".__init__") else m


# module -> set of emboviz.* modules it imports (resolved to real modules)
mods: dict[str, set[str]] = {}
modset: set[str] = set()
texts: dict[str, str] = {}
for p in ROOT.rglob("*.py"):
    if "__pycache__" in str(p):
        continue
    modset.add(mod_of(p))
    texts[mod_of(p)] = p.read_text()


def resolve(mention: str) -> set[str]:
    """A dotted mention → the actual modules pulled in: the exact module if
    it exists; otherwise the package __init__ at `mention`; plus, for
    `from a.b import c`, the submodule `a.b.c` if it exists. Always include
    every ancestor package (importing a.b.c runs a/__init__, a.b/__init__)."""
    out: set[str] = set()
    if mention in modset:
        out.add(mention)
    parts = mention.split(".")
    # ancestor packages
    for i in range(1, len(parts) + 1):
        anc = ".".join(parts[:i])
        if anc in modset:
            out.add(anc)
    return out


# parse imports via AST (catches eager + lazy/in-function imports)
def emboviz_mentions(text: str) -> set[str]:
    out: set[str] = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("emboviz"):
                    out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("emboviz"):
                out.add(node.module)
                for a in node.names:               # from a.b import c → a.b.c
                    out.add(f"{node.module}.{a.name}")
    return out


for m in modset:
    refs: set[str] = set()
    for mention in emboviz_mentions(texts[m]):
        refs |= resolve(mention)
    refs.discard(m)
    mods[m] = refs

# importers graph
importers: dict[str, set[str]] = {m: set() for m in modset}
for m, refs in mods.items():
    for r in refs:
        importers[r].add(m)

LIVE_ROOTS = {
    "emboviz", "emboviz.cli", "emboviz.cli.analyze", "emboviz.cli.info",
    "emboviz.cli.convert_pi0", "emboviz.cli.install_adapter",
    "emboviz._internal.runner", "emboviz.datasets.manifest",
    "emboviz.models.registry", "emboviz.models.mock", "emboviz.config",
}
live, stack = set(), [r for r in LIVE_ROOTS if r in modset]
while stack:
    cur = stack.pop()
    if cur in live:
        continue
    live.add(cur)
    stack.extend(mods.get(cur, ()))

dead = sorted(m for m in modset if m not in live)
print(f"=== {len(live)} live / {len(modset)} total → {len(dead)} dead ===\n")
for m in dead:
    imps = sorted(importers[m])
    live_imps = [i for i in imps if i in live]
    tag = "  <-- LIVE IMPORTER!" if live_imps else ""
    print(f"DEAD {m}")
    print(f"     imported by: {imps or '(nobody)'}{tag}")
