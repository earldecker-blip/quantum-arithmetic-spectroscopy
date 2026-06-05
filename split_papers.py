"""
Split v12 into Paper 1 (core, Secs 1-11) and Paper 2 (quantum extensions, Secs 1-6).
Also applies consistent formatting: heading bold + spacing, body paragraph spacing.
"""

import xml.etree.ElementTree as ET, copy, re, os, shutil, zipfile

# ── Namespace helpers ─────────────────────────────────────────────────────────
NS  = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XS  = 'http://www.w3.org/XML/1998/namespace'
W   = f'{{{NS}}}'
XML = f'{{{XS}}}'

for prefix, uri in [
    ('w',   NS),
    ('r',   'http://schemas.openxmlformats.org/officeDocument/2006/relationships'),
    ('m',   'http://schemas.openxmlformats.org/officeDocument/2006/math'),
    ('wp',  'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'),
    ('a',   'http://schemas.openxmlformats.org/drawingml/2006/main'),
    ('pic', 'http://schemas.openxmlformats.org/drawingml/2006/picture'),
    ('mc',  'http://schemas.openxmlformats.org/markup-compatibility/2006'),
]:
    ET.register_namespace(prefix, uri)

BASE   = '/sessions/peaceful-exciting-bohr/mnt/Thought Experiments'
SRC    = f'{BASE}/unpacked_v12'
OUT1   = f'{BASE}/unpacked_paper1'
OUT2   = f'{BASE}/unpacked_paper2'
PACK   = f'/sessions/peaceful-exciting-bohr/mnt/.claude/skills/docx/scripts/office/pack.py'
ORIG   = f'{BASE}/Arithmetic_Resonances_v12.docx'

def para_text(p):
    return ''.join(t.text or '' for t in p.iter(f'{W}t')).strip()

def is_section_heading(text):
    """Match '1. Title', '2.1 Title', etc."""
    return bool(re.match(r'^\d+(\.\d+)*[. ]', text))

def heading_level(text):
    m = re.match(r'^(\d+(\.\d+)*)', text)
    if not m: return 0
    dots = m.group(1).count('.')
    return dots + 1   # 1 for top-level, 2 for x.x, 3 for x.x.x

def apply_heading_style(p, level):
    """Make paragraph bold, set font size and spacing by heading level."""
    pPr = p.find(f'{W}pPr')
    if pPr is None:
        pPr = ET.Element(f'{W}pPr')
        p.insert(0, pPr)

    # Spacing
    sp = pPr.find(f'{W}spacing')
    if sp is None:
        sp = ET.SubElement(pPr, f'{W}spacing')
    if level == 1:
        sp.set(f'{W}before', '360')
        sp.set(f'{W}after',  '120')
    elif level == 2:
        sp.set(f'{W}before', '240')
        sp.set(f'{W}after',  '80')
    else:
        sp.set(f'{W}before', '160')
        sp.set(f'{W}after',  '60')

    # Bold all runs; set font size
    size = {1: '28', 2: '26', 3: '24'}.get(level, '24')
    for r in p.findall(f'{W}r'):
        rPr = r.find(f'{W}rPr')
        if rPr is None:
            rPr = ET.Element(f'{W}rPr')
            r.insert(0, rPr)
        if rPr.find(f'{W}b') is None:
            ET.SubElement(rPr, f'{W}b')
        sz = rPr.find(f'{W}sz')
        if sz is None:
            sz = ET.SubElement(rPr, f'{W}sz')
        sz.set(f'{W}val', size)

def apply_body_style(p):
    """Consistent body text: 12pt, 1.15 line spacing, 8pt after."""
    pPr = p.find(f'{W}pPr')
    if pPr is None:
        pPr = ET.Element(f'{W}pPr')
        p.insert(0, pPr)
    sp = pPr.find(f'{W}spacing')
    if sp is None:
        sp = ET.SubElement(pPr, f'{W}spacing')
    # Don't touch after/before if it looks like a heading
    sp.set(f'{W}after', '80')
    sp.set(f'{W}line', '276')        # 1.15 × 240 = 276
    sp.set(f'{W}lineRule', 'auto')

def make_title_para(text, size='36', bold=True, centered=True):
    p = ET.Element(f'{W}p')
    pPr = ET.SubElement(p, f'{W}pPr')
    if centered:
        jc = ET.SubElement(pPr, f'{W}jc')
        jc.set(f'{W}val', 'center')
    sp = ET.SubElement(pPr, f'{W}spacing')
    sp.set(f'{W}before', '0')
    sp.set(f'{W}after', '120')
    r = ET.SubElement(p, f'{W}r')
    rPr = ET.SubElement(r, f'{W}rPr')
    if bold:
        ET.SubElement(rPr, f'{W}b')
    sz = ET.SubElement(rPr, f'{W}sz')
    sz.set(f'{W}val', size)
    t = ET.SubElement(r, f'{W}t')
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    return p

def make_simple_para(text, bold=False, centered=False, before='0', after='80'):
    p = ET.Element(f'{W}p')
    pPr = ET.SubElement(p, f'{W}pPr')
    if centered:
        jc = ET.SubElement(pPr, f'{W}jc')
        jc.set(f'{W}val', 'center')
    sp = ET.SubElement(pPr, f'{W}spacing')
    sp.set(f'{W}before', before)
    sp.set(f'{W}after',  after)
    r = ET.SubElement(p, f'{W}r')
    rPr = ET.SubElement(r, f'{W}rPr')
    if bold:
        ET.SubElement(rPr, f'{W}b')
    t = ET.SubElement(r, f'{W}t')
    t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    t.text = text
    return p

# ── Load source document ──────────────────────────────────────────────────────
tree = ET.parse(f'{SRC}/word/document.xml')
body = tree.find(f'{W}body')
children = list(body)

# ── Paragraph index map (from earlier analysis) ───────────────────────────────
# Paper 1: idx 0-96 (title through Discussion) + 133-148 (references)
# Paper 2: idx 97-132 (Outlook through Outlook close) + references subset
PAPER1_MAIN  = list(range(0, 97))
PAPER1_REFS  = list(range(133, 149))
PAPER2_MAIN  = list(range(97, 133))
PAPER2_REFS  = list(range(133, 149))

# ── Renumbering map for Paper 1 ───────────────────────────────────────────────
# Fix: "14. Discussion" → "11. Discussion"
P1_RENUMBER = {
    '14.': '11.',
    '14. ': '11. ',
}

# ── Renumbering map for Paper 2 ───────────────────────────────────────────────
# 15. Outlook → 1. Introduction (replaced by new title paragraphs)
# 15.1 → 2.1, 15.2 → 2.2, 15.3 → 2.3, 15.4 → 2.4
# 16. → 3., 17. → 4.
# 17.1 → 4.1, 17.2 → 4.2, 17.3 → 4.3, 17.4 → 4.4
P2_RENUMBER = {
    '15. Outlook: Exploratory Extensions': '2. Quantum Spectroscopy and Exploratory Extensions',
    '15.1 Preliminary Extension — Phase 5:': '2.1 Phase 5:',
    '15.2 Preliminary Extension — Phase 6:': '2.2 Phase 6:',
    '15.3 Preliminary Extension — Phase 7:': '2.3 Phase 7:',
    '15.4 Preliminary Extension — C5:':      '2.4 C5:',
    '16. Thermal Analogy:': '3. Thermal Analogy:',
    '17. Future Directions': '4. Future Directions',
    '17.1 ': '4.1 ',
    '17.2 ': '4.2 ',
    '17.3 ': '4.3 ',
    '17.4 ': '4.4 ',
}

def renumber_para(p, remap):
    """Replace text in all runs of p according to remap dict."""
    for r in p.iter(f'{W}r'):
        t = r.find(f'{W}t')
        if t is not None and t.text:
            for old, new in remap.items():
                if old in t.text:
                    t.text = t.text.replace(old, new, 1)

def polish_para(p):
    """Apply heading or body formatting based on paragraph text."""
    text = para_text(p)
    if not text:
        return
    if is_section_heading(text):
        level = heading_level(text)
        apply_heading_style(p, min(level, 3))
    else:
        apply_body_style(p)

def copy_paragraphs(indices, remap=None):
    """Deep-copy paragraphs at given indices, optionally renumber, then polish."""
    result = []
    for i in indices:
        if i >= len(children):
            continue
        p = copy.deepcopy(children[i])
        if remap:
            renumber_para(p, remap)
        polish_para(p)
        result.append(p)
    return result

# ─────────────────────────────────────────────────────────────────────────────
# PAPER 1
# ─────────────────────────────────────────────────────────────────────────────
def build_paper1():
    shutil.copytree(SRC, OUT1, dirs_exist_ok=True)
    tree1 = ET.parse(f'{OUT1}/word/document.xml')
    body1 = tree1.find(f'{W}body')
    # Clear body (keep sectPr if present)
    sect_pr = body1.find(f'{W}sectPr')
    for child in list(body1):
        body1.remove(child)

    # New title block
    body1.append(make_title_para(
        'Arithmetic Resonances in the Logarithmic Spectral Weight: '
        'Near-Degeneracies, Prime Structure, and Quantum Detection',
        size='32'
    ))
    body1.append(make_simple_para('Earl Decker', bold=False, centered=True, after='40'))
    body1.append(make_simple_para('Independent Researcher · earldecker@gmail.com',
                                  centered=True, after='200'))

    # Main content
    for p in copy_paragraphs(PAPER1_MAIN[2:], remap=P1_RENUMBER):  # skip old title+author
        body1.append(p)

    # References heading
    body1.append(make_simple_para('References', bold=True, before='360', after='120'))
    for p in copy_paragraphs(PAPER1_REFS[1:]):   # skip old "References" heading
        body1.append(p)

    if sect_pr is not None:
        body1.append(sect_pr)

    tree1.write(f'{OUT1}/word/document.xml', xml_declaration=True, encoding='unicode')
    print('Paper 1 document.xml written')

# ─────────────────────────────────────────────────────────────────────────────
# PAPER 2
# ─────────────────────────────────────────────────────────────────────────────
def build_paper2():
    shutil.copytree(SRC, OUT2, dirs_exist_ok=True)
    tree2 = ET.parse(f'{OUT2}/word/document.xml')
    body2 = tree2.find(f'{W}body')
    sect_pr = body2.find(f'{W}sectPr')
    for child in list(body2):
        body2.remove(child)

    # New title block
    body2.append(make_title_para(
        'Quantum Spectroscopy of Riemann Zeros via Mertens-Ramsey Circuits: '
        'Phases 5–7 and IBM Circuit Design',
        size='30'
    ))
    body2.append(make_simple_para('Earl Decker', centered=True, after='40'))
    body2.append(make_simple_para('Independent Researcher · earldecker@gmail.com',
                                  centered=True, after='200'))

    # New abstract
    body2.append(make_simple_para(
        'Abstract — We present three classical computational experiments and one quantum '
        'circuit design that connect the Mertens weight function f(n) = ln(n)/n to the '
        'Riemann Hypothesis via spectral methods. Phase 5 demonstrates that all 30 tabulated '
        'Riemann zeros are detectable in the Mertens sum S(N) = Σ μ(k)f(k) with '
        'Z-scores up to 476σ using a power-spectrum matched filter. Phase 6 confirms '
        'GUE-consistent nearest-neighbor spacings (KS p = 0.74) and level repulsion in the '
        'first 100 zeros, consistent with the Berry–Keating H=xp conjecture. Phase 7 '
        'reconstructs ψ(x) from the first 100 zeros with sub-percent accuracy and '
        'identifies the f(2)=f(4) near-degeneracy as a reconstruction anomaly. '
        'Experiment C5 designs a 5-qubit GHZ Ramsey circuit that achieves a 5.00× '
        'Heisenberg advantage in zero-frequency precision, validated by Aer simulation. '
        'An IBM-ready batch of 1,000 circuits (200,000 shots, ≈5 min) is provided. '
        'All results are framed as preliminary and exploratory, pending higher-statistics '
        'confirmation.',
        before='0', after='160'
    ))
    body2.append(make_simple_para(
        'Keywords: Riemann zeros, Mertens function, GUE statistics, Heisenberg advantage, '
        'quantum Ramsey spectroscopy, Berry–Keating conjecture, IBM Quantum',
        after='240'
    ))

    # Section 1: Introduction (brief)
    body2.append(make_title_para('1. Introduction', size='28', centered=False))
    body2.append(make_simple_para(
        'This companion paper to "Arithmetic Resonances in the Logarithmic Spectral Weight" '
        '(Decker, 2026) reports three classical spectral experiments (Phases 5–7) and '
        'one quantum circuit design (C5) that extend the core results toward direct detection '
        'of Riemann zeros. Section 2 covers the three computational phases. Section 3 develops '
        'the thermal primon-gas analogy for f(n). Section 4 outlines future directions '
        'including the planned IBM hardware experiment.',
        before='80', after='160'
    ))

    # Main content: paragraphs 97-132, renumbered
    # Skip paragraph 97 (old "15. Outlook" intro text which we replaced)
    for p in copy_paragraphs(PAPER2_MAIN, remap=P2_RENUMBER):
        body2.append(p)

    # References
    body2.append(make_simple_para('References', bold=True, before='360', after='120'))
    for p in copy_paragraphs(PAPER2_REFS[1:]):
        body2.append(p)

    if sect_pr is not None:
        body2.append(sect_pr)

    tree2.write(f'{OUT2}/word/document.xml', xml_declaration=True, encoding='unicode')
    print('Paper 2 document.xml written')

build_paper1()
build_paper2()
print("Done.")
