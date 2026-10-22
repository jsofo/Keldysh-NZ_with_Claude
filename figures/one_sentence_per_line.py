#!/usr/bin/env python3
"""
Reformat LaTeX sources so that every sentence starts on a new line
("semantic line breaks"): wrapped prose lines of a paragraph are joined and
then split at sentence boundaries.  Display math, TikZ pictures, tables,
verbatim and the preamble are left untouched; blank lines (paragraph breaks)
are preserved exactly; no blank line is ever introduced.

Usage:  python3 one_sentence_per_line.py file1.tex [file2.tex ...]
        (files are rewritten in place; use --check to only report)
"""
import re, sys

PROTECTED_ENVS = {
    "equation", "equation*", "align", "align*", "aligned", "gather", "gather*",
    "multline", "multline*", "eqnarray", "tikzpicture", "tabular", "tabularx",
    "verbatim", "lstlisting", "pmatrix", "bmatrix", "matrix", "cases",
}
# lines that are structural: left alone, and they terminate a prose paragraph
STRUCTURAL_RE = re.compile(
    r"^\s*(%|\\(begin|end|section|subsection|subsubsection|paragraph\*?\{[^}]*\}\s*$|label|centering|"
    r"includegraphics|input|renewcommand|newcommand|usepackage|addcontentsline|clearpage|"
    r"vspace|toprule|midrule|bottomrule|setlength|tcbuselibrary|theoremstyle|newtheorem|"
    r"documentclass|usetikzlibrary|title|author|date|maketitle|tableofcontents|"
    r"bibliographystyle|bibliography|small\s*$|noindent\s*$))"
)
ABBREV = {
    "e.g", "i.e", "cf", "vs", "eq", "eqs", "fig", "figs", "sec", "secs", "ch", "chs",
    "ref", "refs", "al", "dr", "prof", "no", "p", "pp", "vol", "approx", "resp", "etc",
    "st", "nd", "rd", "th", "ca", "viz",
}


def env_name(line, kind):
    m = re.match(r"\s*\\%s\{([^}]*)\}" % kind, line)
    return m.group(1) if m else None


def in_inline_math(text, pos):
    """True if position pos is inside $...$ (unescaped dollars counted from the start)."""
    n = 0
    i = 0
    while i < pos:
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "$":
            n += 1
        i += 1
    return n % 2 == 1


def is_abbreviation(text, dot_pos):
    """Is the period at dot_pos the end of an abbreviation or an initial?"""
    j = dot_pos
    while j > 0 and (text[j - 1].isalpha() or text[j - 1] in ".\\"):
        j -= 1
    token = text[j:dot_pos]
    if token.startswith("\\"):
        return True  # e.g. \dots. -> leave alone
    if len(token) == 1 and token.isupper():
        return True  # initial: "J. O. Sofo"
    return token.lower() in ABBREV


BOUNDARY_RE = re.compile(r"([.?!])([)\]}'\"]*)\s+(?=[A-Z\\$`(\d])")


def split_sentences(text):
    """Split a joined paragraph string into sentences (list of strings)."""
    out, start = [], 0
    for m in BOUNDARY_RE.finditer(text):
        dot = m.start(1)
        if in_inline_math(text, dot):
            continue
        if m.group(1) == "." and is_abbreviation(text, dot):
            continue
        # don't split if the "sentence" that follows is just a closing brace etc.
        out.append(text[start:m.end(2)].rstrip())
        start = m.end()
    out.append(text[start:].rstrip())
    return [s for s in out if s != ""]


def reflow_paragraph(lines):
    """lines: list of raw prose lines (no blank lines).  Returns new list of lines."""
    # keep forced line breaks (\\ or \\[..]) as hard boundaries
    segments, cur = [], []
    for ln in lines:
        s = ln.strip()
        if s == "":
            continue
        cur.append(s)
        if re.search(r"\\\\(\[[^\]]*\])?\s*$", s):
            segments.append(cur); cur = []
    if cur:
        segments.append(cur)
    indent = ""
    first = lines[0]
    m = re.match(r"^(\s*)\\item\b", first)
    if m:
        indent = "  "
    result = []
    for seg in segments:
        joined = " ".join(seg)
        sentences = split_sentences(joined)
        for k, sent in enumerate(sentences):
            if not result and k == 0:
                result.append(sent)           # first line of the paragraph: no indent
            else:
                result.append(indent + sent)
    return result


def process(text, is_main=False):
    lines = text.split("\n")
    out, buf = [], []
    depth = 0           # nesting depth inside protected environments
    in_preamble = is_main

    def flush():
        nonlocal buf
        if buf:
            out.extend(reflow_paragraph(buf))
            buf = []

    for ln in lines:
        if in_preamble:
            out.append(ln)
            if ln.strip().startswith("\\begin{document}"):
                in_preamble = False
            continue
        b = env_name(ln, "begin"); e = env_name(ln, "end")
        if depth > 0:
            out.append(ln)
            if b in PROTECTED_ENVS: depth += 1
            if e in PROTECTED_ENVS: depth -= 1
            continue
        if b in PROTECTED_ENVS:
            flush(); out.append(ln); depth += 1
            continue
        if ln.strip() == "":
            flush(); out.append(ln)
            continue
        if STRUCTURAL_RE.match(ln):
            flush(); out.append(ln)
            continue
        if re.match(r"\s*\\item\b", ln):
            flush()                      # every \item starts its own paragraph
        buf.append(ln)
    flush()
    return "\n".join(out)


def main():
    check = "--check" in sys.argv
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    for f in files:
        src = open(f, encoding="utf-8").read()
        new = process(src, is_main=("\\documentclass" in src[:200]))
        if new != src:
            print(f"{'would change' if check else 'reformatted'}: {f}")
            if not check:
                open(f, "w", encoding="utf-8").write(new)


if __name__ == "__main__":
    main()
