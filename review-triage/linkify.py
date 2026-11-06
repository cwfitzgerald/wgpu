"""Add GitHub links to PR, issue, cross-repo, commit and user references in Markdown files."""
import re, sys

REPO = "https://github.com/gfx-rs/wgpu"
# Bare cross-repo names that appear without an owner.
OWNERS = {"wasm-bindgen": "wasm-bindgen/wasm-bindgen"}

CROSS = re.compile(r"(?<![\w/\[])([A-Za-z0-9-]+/[A-Za-z0-9_.-]+)#(\d+)\b")
BARE_CROSS = re.compile(r"(?<![\w/\[])(" + "|".join(map(re.escape, OWNERS)) + r")#(\d+)\b")
COMMIT = re.compile(r"(?<![\w/\[])([A-Za-z0-9-]+/[A-Za-z0-9_.-]+)@([0-9a-f]{7,40})\b")
NUM = re.compile(r"(?<![\w/\[&])#(\d{3,6})\b")
USER = re.compile(r"(?<![\w/\[.@])@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\b(?![.\w]*\.(?:com|org|io)\b)")
# WGSL attributes that look like user mentions.
NOT_USERS = {"align", "binding", "builtin", "compute", "const", "diagnostic", "fragment", "group", "id",
             "interpolate", "invariant", "location", "mesh", "must_use", "payload", "size", "task",
             "vertex", "workgroup_size"}
LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")

def sub_plain(text):
    # Protect existing links, then apply replacements to the gaps only.
    out, last = [], 0
    for m in LINK.finditer(text):
        out.append(apply(text[last:m.start()])); out.append(m.group(0)); last = m.end()
    out.append(apply(text[last:]))
    return "".join(out)

def apply(s):
    s = COMMIT.sub(lambda m: f"[{m.group(0)}](https://github.com/{m.group(1)}/commit/{m.group(2)})", s)
    s = CROSS.sub(lambda m: f"[{m.group(0)}](https://github.com/{m.group(1)}/issues/{m.group(2)})", s)
    s = BARE_CROSS.sub(lambda m: f"[{m.group(0)}](https://github.com/{OWNERS[m.group(1)]}/issues/{m.group(2)})", s)
    s = re.sub(r"(?<![\w/\[])@([A-Za-z0-9-]+)\[bot\]",
               lambda m: f"[@{m.group(1)}&#91;bot&#93;](https://github.com/apps/{m.group(1)})", s)
    # Re-protect links made above before the plain number and user passes.
    parts = re.split(r"(\[[^\]]*\]\([^)]*\))", s)
    for i in range(0, len(parts), 2):
        p = NUM.sub(lambda m: f"[#{m.group(1)}]({REPO}/issues/{m.group(1)})", parts[i])
        parts[i] = re.sub(r"(\[[^\]]*\]\([^)]*\))|" + USER.pattern,
                          lambda m: m.group(1) or (m.group(0) if m.group(2) in NOT_USERS else f"[@{m.group(2)}](https://github.com/{m.group(2)})"), p)
    return "".join(parts)

def linkify(md):
    out, in_fence = [], False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence; out.append(line); continue
        if in_fence:
            out.append(line); continue
        # Skip inline code spans.
        segs = re.split(r"(`+[^`]*`+)", line)
        out.append("".join(seg if i % 2 else sub_plain(seg) for i, seg in enumerate(segs)))
    return "\n".join(out)

for path in sys.argv[1:]:
    src = open(path).read()
    open(path, "w").write(linkify(src))
