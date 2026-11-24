import sys
sys.path.insert(0, r'd:\coursework\src')

issues = []

# 1. Check all imports resolve
try:
    import dataset
    import model
    import utils
    import train
    print("  IMPORTS OK: dataset, model, utils, train")
except Exception as e:
    issues.append("  IMPORT ERR: " + str(e))

# 2. Check no duplicate json import inside train() source
with open(r'd:\coursework\src\train.py', encoding='utf-8') as f:
    lines = f.readlines()

json_inside_fn = [(i+1, l.strip()) for i, l in enumerate(lines)
                  if 'import json' in l and i > 50]
if json_inside_fn:
    issues.append("  DUPLICATE import json at lines: " + str(json_inside_fn))
else:
    print("  No duplicate import json inside functions: OK")

# 3. Check old hardcoded D:\\CXR8 path outside env block
old_paths = [(i+1, l.strip()) for i, l in enumerate(lines)
             if 'D:\\\\CXR8' in l or ("D:\\CXR8" in l and 'os.environ' not in l and i > 100)]
if old_paths:
    issues.append("  OLD HARDCODED PATH found: " + str(old_paths))
else:
    print("  No stale hardcoded D:\\CXR8 paths: OK")

# 4. Check KeyboardInterrupt handler
kb = any('KeyboardInterrupt' in l for l in lines)
print("  KeyboardInterrupt handler: " + ("OK" if kb else "MISSING"))
if not kb:
    issues.append("  KeyboardInterrupt handler missing")

# 5. Check atomic write in utils
with open(r'd:\coursework\src\utils.py', encoding='utf-8') as f:
    utxt = f.read()
atomic = 'shutil.move' in utxt and '.tmp' in utxt
print("  Atomic checkpoint write: " + ("OK" if atomic else "MISSING"))
if not atomic:
    issues.append("  Atomic write missing")

# 6. Check optimizer device move
dev_move = 'state[k] = v.to(device)' in utxt
print("  Optimizer tensor device move: " + ("OK" if dev_move else "MISSING"))
if not dev_move:
    issues.append("  Optimizer device move missing")

# 7. Check per-epoch history flush
with open(r'd:\coursework\src\train.py', encoding='utf-8') as f:
    ttxt = f.read()
hist_flush = 'history_json_path' in ttxt
print("  Per-epoch history flush: " + ("OK" if hist_flush else "MISSING"))
if not hist_flush:
    issues.append("  Per-epoch history flush missing")

# 8. Check cudnn.benchmark
bench = 'cudnn.benchmark = True' in ttxt
print("  cudnn.benchmark=True on GPU: " + ("OK" if bench else "MISSING"))
if not bench:
    issues.append("  cudnn.benchmark=True missing")

# 9. Check env-var path override
env_override = 'os.environ.get' in ttxt and 'CXR_DATA_CSV' in ttxt
print("  Env-var path override: " + ("OK" if env_override else "MISSING"))
if not env_override:
    issues.append("  Env-var path override missing")

# 10. Check emergency checkpoint
emergency = 'emergency_checkpoint' in ttxt
print("  Emergency checkpoint on interrupt: " + ("OK" if emergency else "MISSING"))
if not emergency:
    issues.append("  Emergency checkpoint missing")

# 11. Check L1 excludes BN and bias
with open(r'd:\coursework\src\utils.py', encoding='utf-8') as f:
    utxt2 = f.read()
l1_bn = 'exclude_bn' in utxt2 and 'exclude_bias' in utxt2
print("  L1 excludes BN/bias: " + ("OK" if l1_bn else "MISSING"))
if not l1_bn:
    issues.append("  L1 BN/bias exclusion missing")

# 12. Check map_location in load_checkpoint
map_loc = 'map_location=device' in utxt2
print("  map_location in load_checkpoint: " + ("OK" if map_loc else "MISSING"))
if not map_loc:
    issues.append("  map_location missing")

# 13. Check try/except around optimizer.load_state_dict
try_opt = 'Could not restore optimizer' in utxt2
print("  try/except for optimizer restore: " + ("OK" if try_opt else "MISSING"))
if not try_opt:
    issues.append("  try/except for optimizer restore missing")

print()
if issues:
    print("ISSUES FOUND:")
    for i in issues:
        print(i)
    sys.exit(1)
else:
    print("=" * 40)
    print("ALL 13 CHECKS PASSED - Code is clean!")
    print("=" * 40)
    sys.exit(0)
