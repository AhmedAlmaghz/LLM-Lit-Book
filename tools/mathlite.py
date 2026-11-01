# -*- coding: utf-8 -*-
"""
mathlite.py — تحويل الرياضيات اللاجراسية المضمّنة @( ... )@ إلى نص يونيكود مقروء.
مسح بموازنة الأقواس: يفكّ كسورًا وجذورًا وقبعات ومؤشرات كاملة بدقة.
القاعدة: نحول ما يمكن، ونبقي القليل المتبقي مقروءًا دائمًا (بدون بط إخفائي).
"""
import re
from pathlib import Path

_SUPER_MAP = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷",
    "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "n": "ⁿ", "i": "ⁱ", "(": "⁽", ")": "⁾",
    "=": "⁼",
    "a": "ᵃ", "b": "ᵇ", "c": "ᶜ", "d": "ᵈ", "e": "ᵉ", "f": "ᶠ", "g": "ᵍ", "h": "ʰ",
    "j": "ʲ", "k": "ᵏ", "l": "ˡ", "m": "ᵐ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ",
    "t": "ᵗ", "u": "ᵘ", "v": "ᵛ", "w": "ʷ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
    "T": "ᵀ", "A": "ᴬ", "B": "ᴮ", "C": "ᶜ2"[:1], "E": "ᴱ", "H": "ᴴ", "I": "ᴵ", "M": "ᴹ",
    "N": "ᴺ", "O": "ᴼ", "P": "ᴾ", "R": "ᴿ", "S": "ˢ" if False else "ˢ",
    "α": "ᵅ", "β": "ᵝ", "γ": "ᵞ", "δ": "ᵟ", "ε": "ᵋ", "θ": "ᶿ", "ι": "ᶥ",
    "φ": "ᵠ", "χ": "ᵡ", "⋅": "⋅",
}
_SUB_MAP = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆", "7": "₇",
    "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "l": "ₗ", "m": "ₘ",
    "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ", "v": "ᵥ",
    "x": "ₓ", "·": "᛫",
}
_MATHBB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "C": "ℂ", "Q": "ℚ", "E": "𝔼", "S": "𝕊"}
_SCRIPT = {"L": "𝓛", "F": "𝓕", "N": "𝒩", "P": "𝒫", "O": "𝒪", "R": "ℛ", "D": "𝒟", "A": "𝒜"}

_SYMBOLS = {
    "times": "×", "cdot": "·", "div": "÷", "pm": "±", "mp": "∓",
    "approx": "≈", "neq": "≠", "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥",
    "ll": "≪", "gg": "≫", "equiv": "≡", "propto": "∝", "sim": "~", "simeq": "≃",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒",
    "Leftarrow": "⇐", "Leftrightarrow": "⇔", "leftrightarrow": "↔",
    "longrightarrow": "⟶", "mapsto": "↦", "mid": "|", "vert": "|", "given": "|",
    "infty": "∞", "nabla": "∇", "partial": "∂", "hbar": "ℏ", "ell": "ℓ",
    "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂", "subseteq": "⊆",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "varnothing": "∅",
    "forall": "∀", "exists": "∃", "top": "⊤", "perp": "⊥", "angle": "∠",
    "cdots": "⋯", "ldots": "…", "dots": "…", "vdots": "⋮", "ddots": "⋱",
    "langle": "⟨", "rangle": "⟩", "prime": "′", "circ": "∘",
    "star": "*", "bullet": "•", "oplus": "⊕", "ominus": "⊖", "otimes": "⊗",
    "odot": "⊙", "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "checkmark": "✓", "AA": "Å", "degree": "°", "ast": "∗", "dagger": "†",
    "sq": "□", "blacksquare": "■", "Re": "ℜ", "Im": "ℑ", "wp": "℘",
}
_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "varepsilon": "ε", "epsilon": "ϵ", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "ϕ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}
_ACCENTS = {  # تمييز فوق الحرف: \hat{y} → ŷ
    "hat": "\u0302", "widehat": "\u0302", "bar": "\u0304", "overline": "\u0305",
    "tilde": "\u0303", "widetilde": "\u0303", "vec": "\u20D7", "dot": "\u0307",
    "ddot": "\u0308", "underline": "\u0332",
}
_SIZE_CMDS = ("Bigg", "bigg", "Big", "big", "Biggl", "Biggr", "biggl", "biggr",
              "Bigl", "Bigr", "bigl", "bigr", "left", "right", "displaystyle",
              "quad", "qquad", "thinspace")
_FUNCS = ("argmax", "argmin", "softmax", "sigmoid", "clip", "sgn", "log", "ln",
          "exp", "min", "max", "sin", "cos", "tan", "det", "rank", "const",
          "operatorname", "text", "mathrm", "textrm", "mathbf", "bm", "mathbb",
          "mathcal", "mathpzc", "pmb")

_br_open = "{"


def _read_group(s, i):
    """يقرأ قوسًا متوازنًا عند s[i]=='{' ويعيد (المحتوى، موضع ما بعد الاختتام)."""
    if i >= len(s) or s[i] != "{":
        return None, i
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _read_token(s, i):
    """قراءة وحدة واحدة بعد ^ أو _ أو عنوان: قوس متوازن أو حرف/رقم واحد."""
    if i >= len(s):
        return None, i
    if s[i] == "{":
        g, j = _read_group(s, i)
        return g, j
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|[{}%&#$_\\])", s[i:])
        if m:
            return "\\" + m.group(1), i + 1 + len(m.group(0)) - 1
        return "\\", i + 2
    return s[i], i + 1


def _uni(text, table):
    """نص كامل في خريطة يونيكود نفسها — وإلا نتخلص بـ None."""
    if text and all(c in table for c in text):
        return "".join(table[c] for c in text)
    return None


def _pass(s):
    out = []
    i = 0
    changed = False
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            # \frac .. \sqrt.. accents.. commands..
            if s.startswith("\\frac", i):
                g1, j = _read_group(s, i + 5)
                if g1 is not None:
                    j2 = j
                    while j2 < n and s[j2] == " ":
                        j2 += 1
                    g2, j3 = _read_group(s, j2)
                    if g2 is not None:
                        a, b = _convert(g1, False), _convert(g2, False)
                        if len(a.replace(" ", "")) <= 3 and len(b.replace(" ", "")) <= 3:
                            out.append(f"{a}/{b}")
                        else:
                            out.append(f"({a})/({b})")
                        i = j3
                        changed = True
                        continue
                out.append("frac")
                i += 5
                changed = True
                continue
            if s.startswith("\\sqrt", i):
                j = i + 5
                g, j2 = _read_group(s, j)
                if g is not None:
                    out.append("√(" + _convert(g, False) + ")")
                    i = j2
                    changed = True
                    continue
                out.append("√")
                i += 5
                changed = True
                continue
            matched = False
            for acc, comb in _ACCENTS.items():
                if s.startswith("\\" + acc, i):
                    g, j = _read_group(s, i + 1 + len(acc))
                    if g is None:
                        g2, j = _read_token(s, i + 1 + len(acc))
                        g = g2 or ""
                    base = _convert(g, False)
                    base = base[0] + comb + base[1:] if base else ""
                    out.append(base)
                    i = j
                    changed = True
                    matched = True
                    break
            if matched:
                continue
            if s.startswith("\\mathbb", i):
                g, j = _read_group(s, i + 7)
                ch = _MATHBB.get(g.strip(), g.strip()) if g else ""
                out.append(ch)
                i = j
                changed = True
                continue
            if s.startswith("\\mathcal", i):
                g, j = _read_group(s, i + 8)
                ch = _SCRIPT.get(g.strip(), g.strip()) if g else ""
                out.append(ch)
                i = j
                changed = True
                continue
            for name in ("operatorname*", "operatorname", "mathrm", "textrm", "text",
                         "mathbf", "bm", "mathpzc", "pmb"):
                if s.startswith("\\" + name, i):
                    j = i + 1 + len(name)
                    g, j2 = _read_group(s, j)
                    if g is None:
                        t, j2 = _read_token(s, j)
                        g = t or ""
                    inner = _convert(g, False)
                    if name in ("mathbf", "bm", "pmb", "mathpzc"):
                        out.append("*" + inner + "*" if len(inner) <= 3 else inner)
                    else:
                        out.append(inner)
                    i = j2
                    changed = True
                    matched = True
                    break
            if matched:
                continue
            if s.startswith("\\log_", i):
                tok, j = _read_token(s, i + 5)
                u = _uni(tok or "", _SUB_MAP)
                out.append("log" + (u if u else f"_({tok})"))
                i = j
                changed = True
                continue
            if s.startswith("\\log", i):
                out.append("log")
                i += 4
                changed = True
                continue
            # أوامر الحجم والفواصل الرقيقة والفواصل المزدوجة
            for sc in _SIZE_CMDS:
                if s.startswith("\\" + sc, i):
                    i += 1 + len(sc)
                    changed = True
                    matched = True
                    break
            if matched:
                continue
            if s[i + 1:i + 2] in (",", ";", ":", "!", " "):
                out.append(" ")
                i += 2
                changed = True
                continue
            specials = {"%": "%", "&": "&", "#": "#", "$": "$",
                        "_": "_", "{": "{", "}": "}"}
            if s[i + 1:i + 2] in specials:
                out.append(specials[s[i + 1]])
                i += 2
                changed = True
                continue
            # الأوامر المسماة الأخرى
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if m:
                word = m.group(1)
                if word in _GREEK:
                    out.append(_GREEK[word])
                elif word in _SYMBOLS:
                    out.append(_SYMBOLS[word])
                elif word in ("sum",):
                    out.append("Σ")
                elif word in ("prod",):
                    out.append("Π")
                elif word in ("int", "oint"):
                    out.append("∫")
                elif word in ("argmax", "argmin", "softmax", "sigmoid", "clip", "sgn",
                              "exp", "min", "max", "sin", "cos", "tan", "det", "rank",
                              "const", "ln", "top", "ReLU", "GELU", "SiLU", "RoPE"):
                    out.append(word)
                else:
                    out.append(word)  # catchall: يُقيَّم في نهاية القياس
                i += 1 + len(word)
                changed = True
                continue
            out.append("\\")
            i += 1
            changed = True
            continue
        if c in "^_":
            if s[i + 1:i + 2] == "⟦":
                j = s.find("⟧", i)
                if j != -1:
                    out.append(s[i:j + 1])
                    i = j + 1
                    continue
            tok, j = _read_token(s, i + 1)
            if tok is None:
                out.append(c)
                i += 1
                continue
            if i + 1 < len(s) and s[i + 1] == "{":
                tok = _convert(tok, False)
            elif isinstance(tok, str) and "\\" in tok:
                tok = _convert(tok, False)
            table = _SUPER_MAP if c == "^" else _SUB_MAP
            u = _uni(tok, table)
            if u is None:
                if len(tok) <= 10 and re.fullmatch(r"[\w\+\-\=\· ]+", tok):
                    out.append(("^⟦%s⟧" if c == "^" else "_⟦%s⟧") % tok)
                else:
                    out.append(("^⟦%s⟧" if c == "^" else "_⟦%s⟧") % tok)
            else:
                out.append(u)
            i = j
            changed = c != " "
            continue
        out.append(c)
        i += 1
    return "".join(out), changed


def _convert(body, cleanup=True):
    s = body
    s = re.sub(r"\\\((.*?)\\\)", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\\begin\{[a-zA-Z\*]+\}|\\end\{[a-zA-Z\*]+\}", "", s)
    s = re.sub(r"\\\\", " ; ", s)
    for _ in range(8):
        s2, changed = _pass(s)
        s = s2
        if not changed:
            break
    if cleanup:
        s = re.sub(r"_⟦([^⟧]*)⟧", r"_(\1)", s)
        s = re.sub(r"\^⟦([^⟧]*)⟧", r"^( \1)", s)
        s = s.replace("^( ", "^(")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()


def render(text):
    """يحوّل كل صيغة مضمّنة في النص العربي إلى صيغة مقروءة ويعيد النص."""
    last_failures.clear()

    def repl(m):
        out = _convert(m.group(1))
        leftovers = re.findall(r"\\[A-Za-z]{3,}", out)
        tricky = [t for t in ("frac", "begin", "end", "array", "matrix") if t in out]
        if leftovers or tricky:
            last_failures.append((out[:80], leftovers[:3] if leftovers else tricky[:3]))
        return "" + out + ""

    out = re.sub(r"\\\((.*?)\\\)", repl, text, flags=re.DOTALL)
    return out


def convert(body):
    """خروج عام: تحويل جسم صيغة لاتكس واحد (سطرية \\( \\) أو عرض \\[ \\]) إلى نص مقروء."""
    out = _convert(body, cleanup=True)
    leftovers = re.findall(r"\\[A-Za-z]{3,}", out)
    tricky = [t for t in ("frac", "begin", "end", "array", "matrix") if t in out]
    if leftovers or tricky:
        last_failures.append((out[:80], leftovers[:3] if leftovers else tricky[:3]))
    return out


last_failures = []


# قياس شامل
def surveyonly():
    d = json.loads((Path(__file__).resolve().parent.parent / "build" / "book.json").read_text(encoding="utf-8"))
    containers = list(d["chapters"]) + list(d["appendices"])
    for fv in d["front"].values():
        containers.append({"blocks": fv})
    n_ok = n_bad = 0
    fails = []
    for c in containers:
        for b in c["blocks"]:
            if b.get("code"):
                continue
            texts = []
            if b.get("text"):
                texts.append(b["text"])
            if b.get("items"):
                texts.extend(b["items"])
            if b.get("header"):
                texts.extend(b["header"])
            if b.get("rows"):
                for r in b["rows"]:
                    texts.extend(r)
            for t in texts:
                if "\\(" in t:
                    render(t)
                    if last_failures:
                        n_bad += 1
                        fails.extend(last_failures)
                        last_failures.clear()
                    else:
                        n_ok += 1
    return n_ok, n_bad, fails


if __name__ == "__main__":
    import json
    ok, bad, fails = surveyonly()
    (Path(__file__).resolve().parent.parent / "build" / "math_fail.txt").write_text(
        "\n".join(str(f) for f in fails) + "\n", encoding="utf-8")
    print("math ok:", ok, "tricky:", bad)
