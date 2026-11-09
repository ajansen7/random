"""Tiny S-expression reader/writer for KiCad files."""
import re

_TOK = re.compile(r'\s*(?:(\()|(\))|("(?:[^"\\]|\\.)*")|([^\s()"]+))', re.S)


class Sym(str):
    """An unquoted atom (keyword or number)."""


def parse(text):
    stack, cur = [], []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOK.match(text, pos)
        if not m or m.end() == pos:
            break
        pos = m.end()
        lp, rp, s, a = m.groups()
        if lp:
            stack.append(cur); cur = []
        elif rp:
            done = cur; cur = stack.pop(); cur.append(done)
        elif s is not None:
            cur.append(s[1:-1].replace('\\"', '"').replace("\\\\", "\\"))
        elif a is not None:
            cur.append(Sym(a))
    return cur[0] if len(cur) == 1 else cur


def dump(x, indent=0):
    if isinstance(x, list):
        if not x:
            return "()"
        simple = all(not isinstance(e, list) for e in x)
        if simple:
            return "(" + " ".join(dump(e) for e in x) + ")"
        pad = "  " * (indent + 1)
        out = "(" + dump(x[0])
        for e in x[1:]:
            if isinstance(e, list):
                out += "\n" + pad + dump(e, indent + 1)
            else:
                out += " " + dump(e)
        return out + ")"
    if isinstance(x, Sym):
        return str(x)
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:g}" if isinstance(x, float) else str(x)
    s = str(x).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{s}"'


def find(node, key):
    return [e for e in node if isinstance(e, list) and e and e[0] == key]


def find1(node, key):
    r = find(node, key)
    return r[0] if r else None
