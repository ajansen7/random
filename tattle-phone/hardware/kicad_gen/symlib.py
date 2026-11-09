"""Load symbols from KiCad .kicad_sym libraries (resolving `extends`) and list their pins."""
import copy
import os

from sexp import parse, find, find1, Sym

SYM_DIR = os.environ.get("KICAD_SYMBOL_DIR", "/usr/share/kicad/symbols")
_cache = {}


def _lib(name, path=None):
    key = path or name
    if key not in _cache:
        p = path or os.path.join(SYM_DIR, name + ".kicad_sym")
        _cache[key] = parse(open(p).read())
    return _cache[key]


def load_symbol(lib_id, path=None):
    """Return a self-contained symbol node named 'Lib:Name' (flattened if it extends another)."""
    lib, name = lib_id.split(":")
    root = _lib(lib, path)
    syms = {s[1]: s for s in find(root, "symbol")}
    s = copy.deepcopy(syms[name])
    ext = find1(s, "extends")
    if ext:
        base = copy.deepcopy(syms[ext[1]])
        # child keeps its own properties; graphics/pins come from the base
        props = {p[1]: p for p in find(s, "property")}
        out = [Sym("symbol"), name]
        for e in base[2:]:
            if isinstance(e, list) and e[0] == "property" and e[1] in props:
                out.append(props.pop(e[1]))
            elif isinstance(e, list) and e[0] == "symbol":
                sub = copy.deepcopy(e)
                sub[1] = sub[1].replace(ext[1], name, 1)
                out.append(sub)
            else:
                out.append(e)
        out.extend(props.values())
        s = out
    s[1] = lib_id
    return s


def pins(sym):
    """[(number, name, x, y, angle, etype)] for unit 1 / common pins, in symbol coords (y up)."""
    out = []
    for sub in find(sym, "symbol"):
        unit = sub[1].rsplit("_", 2)
        if len(unit) == 3 and unit[1] not in ("0", "1"):
            continue
        for p in find(sub, "pin"):
            at = find1(p, "at")
            name = find1(p, "name")[1]
            num = find1(p, "number")[1]
            length = float(find1(p, "length")[1])
            out.append(dict(num=num, name=name, x=float(at[1]), y=float(at[2]),
                            angle=float(at[3]) if len(at) > 3 else 0.0, etype=str(p[1]),
                            length=length))
    return out
