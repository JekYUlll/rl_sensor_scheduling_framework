#!/usr/bin/env python3
from __future__ import annotations
import json, re, subprocess, unicodedata
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path('/home/horeb/_code/microclimate_demo/rl_sensor_scheduling_framework')
PAPER = ROOT / 'paper'
PLAN = ROOT / '.planning/2026-07-02-terminology-figure-audit'
PLAN.mkdir(parents=True, exist_ok=True)

# ---------- source resolving ----------
def read(p: Path) -> str:
    return p.read_text(errors='ignore')

seen: list[Path] = []
def resolve(path: Path):
    path = path.resolve()
    if path in seen or not path.exists():
        return
    seen.append(path)
    txt = read(path)
    for m in re.finditer(r'\\(?:input|include)\{([^}]+)\}', txt):
        rel = m.group(1)
        p = path.parent / rel
        if p.suffix == '': p = p.with_suffix('.tex')
        if not p.exists():
            p = PAPER / rel
            if p.suffix == '': p = p.with_suffix('.tex')
        resolve(p)

resolve(PAPER / 'main.tex')

def strip_tex(s: str) -> str:
    s = re.sub(r'%.*', '', s)
    s = re.sub(r'\\(?:citep|citet|cite|Cref|cref|ref|label)\{[^}]*\}', '', s)
    s = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', '', s)
    s = s.replace('\\&', '&').replace('\\%', '%').replace('---','-').replace('--','-')
    s = s.replace('{','').replace('}','')
    s = re.sub(r'\$[^$]*\$', ' MATH ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def capture_balanced(txt: str, start: int) -> tuple[str,int]:
    depth=1; j=start; out=[]
    while j < len(txt) and depth:
        ch=txt[j]
        if ch == '{' and (j==0 or txt[j-1] != '\\'):
            depth += 1; out.append(ch)
        elif ch == '}' and (j==0 or txt[j-1] != '\\'):
            depth -= 1
            if depth: out.append(ch)
        else:
            out.append(ch)
        j += 1
    return ''.join(out), j

records=[]
includegraphics=[]
for p in seen:
    rel = str(p.relative_to(PAPER))
    txt = read(p)
    for m in re.finditer(r'\\(section|subsection|subsubsection)\*?\{', txt):
        body,_ = capture_balanced(txt, m.end())
        records.append({'source':'tex','role':'heading:'+m.group(1),'file':rel,'line':txt[:m.start()].count('\n')+1,'text':strip_tex(body),'raw':body})
    for m in re.finditer(r'\\caption(?:\[[^\]]*\])?\{', txt):
        body,_ = capture_balanced(txt, m.end())
        records.append({'source':'tex','role':'caption','file':rel,'line':txt[:m.start()].count('\n')+1,'text':strip_tex(body),'raw':body})
    for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', txt):
        g=m.group(1)
        gp=(p.parent/g)
        if gp.suffix=='':
            for ext in ['.pdf','.png','.jpg','.jpeg']:
                if gp.with_suffix(ext).exists(): gp=gp.with_suffix(ext); break
        if not gp.exists():
            gp=(PAPER/g)
            if gp.suffix=='':
                for ext in ['.pdf','.png','.jpg','.jpeg']:
                    if gp.with_suffix(ext).exists(): gp=gp.with_suffix(ext); break
        if not gp.exists():
            gp=(PAPER/'figures'/g)
            if gp.suffix=='':
                for ext in ['.pdf','.png','.jpg','.jpeg']:
                    if gp.with_suffix(ext).exists(): gp=gp.with_suffix(ext); break
        includegraphics.append({'file':rel,'line':txt[:m.start()].count('\n')+1,'path':str(gp.relative_to(PAPER)) if gp.exists() else g})
    for n,line in enumerate(txt.splitlines(),1):
        if line.strip().startswith('%'): continue
        # table rows / high-visibility row labels
        if '&' in line and ('\\' in line or rel.startswith('tables/')):
            st=strip_tex(line)
            if len(st)>3:
                records.append({'source':'tex','role':'table-row/source','file':rel,'line':n,'text':st[:500],'raw':line.strip()})

# ---------- PDF native extraction ----------
pdf_records=[]
figure_locations=[]
try:
    import fitz
    doc=fitz.open(PAPER/'main.pdf')
    for page_i,page in enumerate(doc, start=1):
        text=page.get_text('text')
        for line in text.splitlines():
            s=' '.join(line.split())
            if not s: continue
            if re.match(r'^(\d+\.|Appendix\s+[A-Z]\.|[A-Z]\.\s)', s):
                pdf_records.append({'source':'pdf-native','role':'heading-or-numbered-line','page':page_i,'text':s})
            if re.match(r'^(Figure|Table)\s+(?:[A-Z]\.)?\d+', s):
                pdf_records.append({'source':'pdf-native','role':'caption-or-table-title','page':page_i,'text':s})
                if s.startswith('Figure'):
                    figure_locations.append({'page':page_i,'text':s})
        # word-level extraction for common chart labels / axis text around figures
        words=page.get_text('words')
        # store title-case/acronym phrases from page words later via full text
    doc.close()
except Exception as e:
    pdf_records.append({'source':'pdf-native','role':'error','text':repr(e)})

# Figure asset text extraction (PDF figures) and OCR fallback for raster/empty.
figure_text=[]
for item in includegraphics:
    gp=PAPER/item['path']
    text=''
    extractor='none'
    if gp.exists() and gp.suffix.lower()=='.pdf':
        try:
            r=subprocess.run(['pdftotext', str(gp), '-'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            text=r.stdout.strip()
            extractor='pdftotext-figure-pdf'
        except Exception as e:
            text=''; extractor='pdftotext-error:'+repr(e)
    if gp.exists() and (not text.strip()) and gp.suffix.lower() in ['.png','.jpg','.jpeg']:
        # OCR only for included raster figures when native text is absent.
        try:
            r=subprocess.run(['tesseract', str(gp), 'stdout', '--oem', '1', '--psm', '11', '-l', 'eng'], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40)
            text=r.stdout.strip(); extractor='tesseract-psm11'
        except Exception as e:
            text=''; extractor='tesseract-error:'+repr(e)
    if gp.exists() and text.strip():
        for line in text.splitlines():
            s=' '.join(line.split())
            if s:
                figure_text.append({'source':'figure-asset','role':'figure-visible-text','figure_path':item['path'],'included_from':item['file'],'line':item['line'],'extractor':extractor,'text':s[:500]})

# ---------- term candidate and evidence clustering ----------
all_records = records + pdf_records + figure_text

# Manual clusters guided by report + current manuscript terminology. Regexes are deliberately broad.
clusters = {
    'C01_method_name_pdppo_prediction_driven': {
        'category':'method name / abbreviation / possible coined title term',
        'patterns':[r'\bPD[- ]PPO\b', r'Prediction[- ]Driven', r'forecast(?:ing)?[- ]reward', r'forecast[- ]oriented', r'forecast(?:ing)? objective'],
        'canonical':'PD-PPO for the method; plain "forecasting objective" for the problem after first definition',
        'priority':'High'
    },
    'C02_macro_staticnorm_margin': {
        'category':'internal metric label / inconsistent variants',
        'patterns':[r'static[- ]normalised|static[- ]normalized|staticnorm|Mstaticnorm|macro score|macro margin|step margin|ordinary step|normalised loss|normalized loss'],
        'canonical':'static-normalised macro score only where formula/metric is discussed; plain "macro forecast score" in captions',
        'priority':'High'
    },
    'C03_fixed_static_reference': {
        'category':'comparator naming / inconsistent variants',
        'patterns':[r'fixed[- ]mask replay|fixed mask replay|fixed mask|static mask|fixed specialist|static specialist|validation[- ]selected fixed|validation[- ]selected static|selected static|true[- ]static|constant mask'],
        'canonical':'validation-selected fixed schedule; fixed-schedule replay when the replay diagnostic is meant',
        'priority':'High'
    },
    'C04_event_label_privileged_diagnostic': {
        'category':'internal diagnostic comparator / inconsistent variants',
        'patterns':[r'event[- ]label replay|event[- ]aware diagnostic replay|privileged(?:[- ]information)? diagnostic|oracle dynamic policy|oracle policy|event[- ]type oracle|oracle reference'],
        'canonical':'event-label diagnostic schedule (define as using event labels, not deployable)',
        'priority':'High'
    },
    'C05_rule_dynamic_rotation': {
        'category':'comparator naming / opaque shorthand',
        'patterns':[r'rule[- ]based dynamic|rule[- ]based scheduling|rotation|rotating specialist|cyclic|simple cycle|static[- ]priority'],
        'canonical':'rule-based schedule; rotation schedule only for the specific cyclic baseline',
        'priority':'Medium'
    },
    'C06_specialist_backbone_slot': {
        'category':'concept naming / inconsistent variants',
        'patterns':[r'mandatory backbone|weather backbone|meteorological backbone|background channel|background weather|specialist slot|specialist channel|specialist instrument|selectable specialist|event[- ]sensitive channel|expert sensor|expert instrument'],
        'canonical':'weather backbone and selectable specialist sensor',
        'priority':'High'
    },
    'C07_operational_feasibility_rules': {
        'category':'constraint wording / inconsistent variants',
        'patterns':[r'operating rules|operational constraints|hard feasibility rules|hard operating rules|feasible[- ]action|feasibility mask|online feasibility|minimum[- ]duration|minimum activation|minimum on[- ]time|duty cycle|duty[- ]cycle'],
        'canonical':'operational constraints; feasibility mask only for the algorithmic mechanism',
        'priority':'Medium'
    },
    'C08_instrument_dataset_names': {
        'category':'proper names / abbreviations / shortened labels',
        'patterns':[r'\bAntAWS\b|\bAWS\b|\bFC4\b|FlowCapt|SPC|\bSPC\b|thermo[- ]hygro|thermohygro|laser disdrometer|particle counter'],
        'canonical':'official instrument/dataset name at first use; short form only afterward',
        'priority':'High'
    },
    'C09_internal_experiment_versions': {
        'category':'internal experiment/version labels',
        'patterns':[r'SCENEBAL|V3\.1|metpair|Scenario\s*Bal|H75|h75|seed\s*sweep|24[- ]seed|18[- ]seed|12[- ]seed|6[- ]seed|Lower[- ]flux mixture|lower[- ]flux mixture'],
        'canonical':'remove internal version labels from reader-facing text; use "24 independent seeds" only where needed',
        'priority':'High'
    },
    'C10_ai_style_chart_titles': {
        'category':'suspected AI-style caption/title wording',
        'patterns':[r'behavio[u]?r(?:al)? audit|mechanism ablation|robustness checks|regime[- ]balanced|event[- ]type decomposition|diagnostic|evidence|boundary|main protocol|reference taxonomy|supporting figure'],
        'canonical':'plain result/action terms, e.g. "specialist use by event type", "ablation study", "sensitivity analysis"',
        'priority':'Medium'
    },
    'C11_training_internal_regularizers': {
        'category':'training internal abbreviations',
        'patterns':[r'AWBC|BC|auxiliary loss|event context auxiliary|candidate[- ]policy|behaviour cloning|behavior cloning|entropy coefficient|clip range'],
        'canonical':'expand once in method table; avoid in high-level captions and abstract',
        'priority':'Medium'
    },
}

cluster_hits=defaultdict(list)
for rec in all_records:
    text=rec.get('text','')
    for cid,cfg in clusters.items():
        for pat in cfg['patterns']:
            if re.search(pat, text, flags=re.I):
                entry={k:v for k,v in rec.items() if k in ['source','role','file','line','page','figure_path','included_from','extractor','text']}
                entry['matched_pattern']=pat
                cluster_hits[cid].append(entry)
                break

# Additional candidate surfaces: acronyms/hyphenated compounds in high visibility records.
compound_counter=Counter()
acronym_counter=Counter()
for rec in all_records:
    if rec.get('role','').startswith(('heading','caption','caption-or-table-title','figure-visible-text','table-row')):
        text=rec.get('text','')
        for m in re.finditer(r'\b[A-Za-z]+(?:[-–][A-Za-z0-9]+){1,4}\b', text):
            token=m.group(0)
            if not re.match(r'^[A-Z][a-z]+-[A-Z][a-z]+$', token):
                compound_counter[token]+=1
        for m in re.finditer(r'\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b', text):
            acronym_counter[m.group(0)]+=1

# Write stores.
(PLAN/'included_sources.json').write_text(json.dumps([str(p.relative_to(PAPER)) for p in seen], indent=2))
(PLAN/'source_records.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
(PLAN/'pdf_records.json').write_text(json.dumps(pdf_records, indent=2, ensure_ascii=False))
(PLAN/'figure_text_records.json').write_text(json.dumps(figure_text, indent=2, ensure_ascii=False))
(PLAN/'cluster_hits.json').write_text(json.dumps(cluster_hits, indent=2, ensure_ascii=False))
(PLAN/'candidate_terms.json').write_text(json.dumps({'hyphenated':compound_counter.most_common(200),'acronyms':acronym_counter.most_common(100),'figure_locations':figure_locations}, indent=2, ensure_ascii=False))

print('included_sources', len(seen))
print('source_records', len(records))
print('pdf_records', len(pdf_records))
print('figure_text_records', len(figure_text))
print('cluster_hits')
for cid,cfg in clusters.items():
    print(cid, cfg['priority'], len(cluster_hits[cid]), cfg['canonical'])
print('first_figures')
for fl in figure_locations[:8]: print(fl)
print('top_hyphenated', compound_counter.most_common(30))
print('top_acronyms', acronym_counter.most_common(30))
