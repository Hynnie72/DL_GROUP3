"""
Fix script for lab2_report.ipynb:
1. Add CSS fix cell after Cell 1 (title markdown)
2. Rewrite Cell 3 (Executive Summary) to avoid complex f-string nesting
3. Ensure TBLSTYLE is defined in import cell (already is, just verify)
"""
import json
import copy

NB_PATH = r'd:\LAB2-DL-NHOM3\notebooks\lab2_report.ipynb'

with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Total cells before fix: {len(cells)}")

# ── 1. Build new CSS fix cell ────────────────────────────────────────────────
CSS_FIX_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from IPython.display import HTML, display\n",
        "display(HTML(\"\"\"\n",
        "<style>\n",
        ".dataframe { color: #000000 !important; background: white !important; }\n",
        ".dataframe td, .dataframe th { color: #000000 !important; }\n",
        "table { color: #000000 !important; }\n",
        "</style>\n",
        "\"\"\"))\n"
    ]
}

# ── 2. Rewrite Executive Summary source (avoid complex nested f-strings) ──────
EXEC_SUMMARY_SOURCE = [
    "# Executive Summary\n",
    "from IPython.display import HTML, display\n",
    "import numpy as np\n",
    "\n",
    "best_row  = df.iloc[0]\n",
    "worst_row = df.iloc[-1]\n",
    "\n",
    "# Best per model\n",
    "best_per_model = {}\n",
    "for _, row in df.iterrows():\n",
    "    m = row['Model']\n",
    "    if m not in best_per_model or row['Test Acc'] > best_per_model[m]['Test Acc']:\n",
    "        best_per_model[m] = row\n",
    "\n",
    "# Per-class mean accuracy\n",
    "cls_accs = {}\n",
    "for _, row in df.iterrows():\n",
    "    for cls, v in row['_per_class'].items():\n",
    "        cls_accs.setdefault(cls, []).append(v.get('recall', 0) * 100)\n",
    "cls_mean = {c: np.mean(vals) for c, vals in cls_accs.items()}\n",
    "hardest  = min(cls_mean, key=cls_mean.get)\n",
    "easiest  = max(cls_mean, key=cls_mean.get)\n",
    "\n",
    "MODEL_ORDER = ['RESNET18', 'VGG16', 'DENSENET121']\n",
    "MODEL_COLOR = {'RESNET18': '#dbeafe', 'VGG16': '#fce7f3', 'DENSENET121': '#d1fae5'}\n",
    "\n",
    "# Build rows list of tuples (bg, label, val, note) using str() not f-string nesting\n",
    "rows_html = []\n",
    "rows_html.append(('#d1fae5', 'Best Test Accuracy (overall)',\n",
    "                  str(round(best_row['Test Acc']*100, 2)) + '%',\n",
    "                  str(best_row['exp'])))\n",
    "rows_html.append(('#ffffff', 'Best F1 Macro (overall)',\n",
    "                  str(round(best_row['F1 Macro'], 4)),\n",
    "                  str(best_row['exp'])))\n",
    "\n",
    "for m in MODEL_ORDER:\n",
    "    if m in best_per_model:\n",
    "        r  = best_per_model[m]\n",
    "        bg = MODEL_COLOR.get(m, '#f9fafb')\n",
    "        rows_html.append((\n",
    "            bg,\n",
    "            'Best ' + m,\n",
    "            str(round(r['Test Acc']*100, 2)) + '%  |  F1=' + str(round(r['F1 Macro'], 4)),\n",
    "            str(r['exp']),\n",
    "        ))\n",
    "\n",
    "rows_html.append(('#fff3cd', 'Worst Test Accuracy (overall)',\n",
    "                  str(round(worst_row['Test Acc']*100, 2)) + '%',\n",
    "                  str(worst_row['exp'])))\n",
    "rows_html.append(('#eff6ff', 'So thi nghiem',\n",
    "                  str(len(df)) + ' TN',\n",
    "                  str(len(best_per_model)) + ' models: ' + ', '.join(best_per_model.keys())))\n",
    "\n",
    "slowest_exp = df.loc[df['Train (h)'].idxmax(), 'exp']\n",
    "rows_html.append(('#ffffff', 'Train time (mean / max)',\n",
    "                  str(round(df['Train (h)'].mean()*60, 1)) + ' / ' + str(round(df['Train (h)'].max()*60, 1)) + ' min',\n",
    "                  'slowest: ' + str(slowest_exp)))\n",
    "\n",
    "fastest_exp = df.loc[df['ms/img'].idxmin(), 'exp']\n",
    "rows_html.append(('#eff6ff', 'Inference (mean ms/img)',\n",
    "                  str(round(df['ms/img'].mean(), 4)) + ' ms',\n",
    "                  'fastest: ' + str(fastest_exp)))\n",
    "rows_html.append(('#fef9c3', 'Hardest class',\n",
    "                  str(hardest) + ' - ' + str(round(cls_mean[hardest], 1)) + '%',\n",
    "                  'mean accuracy qua 9 TN'))\n",
    "rows_html.append(('#ffffff', 'Easiest class',\n",
    "                  str(easiest) + ' - ' + str(round(cls_mean[easiest], 1)) + '%',\n",
    "                  'mean accuracy qua 9 TN'))\n",
    "\n",
    "# Build HTML using string concatenation (no nested f-strings)\n",
    "tr_parts = []\n",
    "for bg, label, val, note in rows_html:\n",
    "    tr_parts.append(\n",
    "        '<tr style=\"background:' + bg + ';color:#000000;\">'\n",
    "        '<td style=\"padding:7px 14px;font-weight:bold;border:1px solid #e5e7eb;\">' + label + '</td>'\n",
    "        '<td style=\"padding:7px 14px;font-weight:bold;border:1px solid #e5e7eb;\">' + val + '</td>'\n",
    "        '<td style=\"padding:7px 14px;border:1px solid #e5e7eb;font-size:12px;\">' + note + '</td></tr>'\n",
    "    )\n",
    "tr_html = ''.join(tr_parts)\n",
    "\n",
    "legend_parts = []\n",
    "for m, c in MODEL_COLOR.items():\n",
    "    legend_parts.append(\n",
    "        '<span style=\"display:inline-block;background:' + c + ';border:1px solid #ccc;'\n",
    "        'padding:2px 10px;border-radius:4px;margin-right:8px;color:#000;font-size:12px;\">' + m + '</span>'\n",
    "    )\n",
    "legend = ''.join(legend_parts)\n",
    "\n",
    "html = (\n",
    "    '<div style=\"border:2px solid #1e3a5f;border-radius:10px;padding:20px;margin:10px 0;'\n",
    "    'background:#f0f7ff;font-family:Arial,sans-serif;\">'\n",
    "    '<h2 style=\"color:#1e3a5f;margin-top:0;\">Executive Summary - Lab 2: Transfer Learning</h2>'\n",
    "    '<p style=\"margin:0 0 10px 0;font-size:12px;\">Legend: ' + legend + '</p>'\n",
    "    '<table style=\"width:100%;border-collapse:collapse;color:#000000;\">'\n",
    "    '<tr style=\"background:#1e3a5f;color:#ffffff;\">'\n",
    "    '<th style=\"padding:8px 14px;text-align:left;\">Chi so</th>'\n",
    "    '<th style=\"padding:8px 14px;text-align:left;\">Gia tri</th>'\n",
    "    '<th style=\"padding:8px 14px;text-align:left;\">Ghi chu</th></tr>'\n",
    "    + tr_html +\n",
    "    '</table>'\n",
    "    '<p style=\"margin-top:12px;font-size:12px;color:#374151;\">'\n",
    "    '* Accuracy va F1 tinh tren <b>test set</b> (3,000 anh). '\n",
    "    'Dataset: Intel Image Classification 6 lop. Device: Kaggle T4 GPU.</p>'\n",
    "    '</div>'\n",
    ")\n",
    "display(HTML(html))\n"
]

# ─────────────────────────────────────────────────────────────────────────────
# Identify cell indices
# Cell 0: Title markdown
# Cell 1: "0. Executive Summary" markdown  
# Cell 2: Executive Summary code  <-- fix source
# Cell 3: Import cell (defines TBLSTYLE)   <-- already OK per earlier read
#
# After inserting CSS cell after Cell 0 (title), indices shift by 1:
# new Cell 1 = CSS fix cell
# new Cell 2 = "0. Executive Summary" markdown
# new Cell 3 = Executive Summary code (was index 2)
# new Cell 4 = Import cell (was index 3)
# ─────────────────────────────────────────────────────────────────────────────

# Check cell 0 is title markdown
assert cells[0]['cell_type'] == 'markdown', "Cell 0 should be title markdown"

# Insert CSS cell at index 1 (after title)
cells.insert(1, CSS_FIX_CELL)
print("Inserted CSS fix cell at index 1")

# Now Executive Summary code cell is at index 3 (was 2)
exec_cell = cells[3]
assert exec_cell['cell_type'] == 'code', f"Cell 3 should be code, got {exec_cell['cell_type']}"
old_src = ''.join(exec_cell['source'])
print(f"Executive Summary cell source starts with: {old_src[:60]!r}")

# Replace source
exec_cell['source'] = EXEC_SUMMARY_SOURCE
print("Replaced Executive Summary source")

# Verify import cell (now index 4) defines TBLSTYLE
import_cell = cells[4]
assert import_cell['cell_type'] == 'code', f"Cell 4 should be code, got {import_cell['cell_type']}"
import_src = ''.join(import_cell['source'])
assert 'TBLSTYLE' in import_src, "TBLSTYLE not found in import cell!"
print("TBLSTYLE confirmed in import cell (index 4)")

# Clear outputs for modified cells
cells[1]['outputs'] = []   # CSS cell (new)
cells[3]['outputs'] = []   # Exec summary
# Reset execution counts for changed cells
cells[1]['execution_count'] = None
cells[3]['execution_count'] = None

print(f"Total cells after fix: {len(cells)}")

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Saved notebook successfully.")
