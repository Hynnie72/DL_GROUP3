import json
nb = json.load(open('notebooks/lab2_report.ipynb', encoding='utf-8'))
errors = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        try:
            compile(src, f'cell_{i}', 'exec')
        except SyntaxError as e:
            errors.append(f"Cell {i}: {e}")
if errors:
    print("ERRORS:")
    for e in errors:
        print(e)
else:
    print(f"OK - all {sum(1 for c in nb['cells'] if c['cell_type']=='code')} code cells passed syntax check")
    print(f"Total cells: {len(nb['cells'])}")
