# Module `runner` — Helper Classes and Utilities

**File:** `src/runner.py`  
**Depends on:** `os`, `json`, `math`, `re`, `threading`, `psutil`, `numpy`, `pandas`, `torch`, `datetime`, `enum`, `dataclasses`

This page documents the supporting classes and standalone helper functions defined at the top of `src/runner.py`. They are used internally by `ExperimentRunner` and are not part of its public API.

---

## Module-level constants

| Name | Value | Description |
|---|---|---|
| `NOT_AVAILABLE` | `"Not Available"` | Placeholder for missing data in reports. |
| `NOT_PERFORMED` | `"Not Performed"` | Placeholder when a step (e.g. verification) was skipped. |
| `DATASET_NAME` | `"FashionMNIST"` | Injected into JSON exports. |
| `JSON_SCHEMA_VERSION` | `1` | Version tag in `export_results_json` output. |
| `MIB` | `1024 ** 2` | Bytes per mebibyte, used for memory calculations. |

### Figure sub-directory constants

| Name | Value | Purpose |
|---|---|---|
| `FIG_GROUP_CURVES` | `"group_curves"` | Group learning-curve plots. |
| `FIG_COMPARISONS` | `"comparisons"` | Final comparison bar charts. |
| `FIG_SWEEPS` | `"sweeps"` | Sweep trend and accuracy-curve plots. |
| `FIG_PER_CLASS` | `"per_class"` | Per-class accuracy heatmap. |
| `FIG_DETAILED` | `"detailed"` | Per-experiment detail figures. |
| `FIG_BENCHMARK` | `"benchmark"` | Inference benchmark plots. |

---

## `ReportContext` (dataclass)

Mutable state shared across the steps of a single `create_report()` call.

| Field | Type | Description |
|---|---|---|
| `output_dir` | `str` | Root directory for all outputs. |
| `report_path` | `str` | Path of the main report file. |
| `include_figures` | `bool` | Whether figures are included. |
| `include_sweeps` | `bool` | Whether sweep sections are included. |
| `include_benchmark` | `bool` | Whether the benchmark section is included. |
| `include_detailed_reports` | `bool` | Whether per-experiment detail reports are included. |
| `include_classification_reports` | `bool` | Whether classification report text is included. |
| `generated_files` | `Dict[str, List[str]]` | Registry of files written, keyed by category. |
| `notes` | `List[str]` | Freeform notes accumulated during generation. |

---

## `PeakRssSampler`

A context manager that estimates peak **process RSS (Resident Set Size)** memory usage while a block of code runs.

A background thread samples the process RSS every `interval` seconds. `delta_mb` returns the largest observed increase over the RSS at entry.

```python
class PeakRssSampler:
    def __init__(self, interval: float = 0.002) -> None
```

| Attribute | Description |
|---|---|
| `start_rss` | RSS at context entry (bytes). |
| `peak_rss` | Maximum RSS observed during the block (bytes). |
| `delta_mb` | `max(0, peak_rss - start_rss) / MIB` — peak increase in MiB. |

**Usage inside `ExperimentRunner`:**

```python
with PeakRssSampler() as sampler:
    model = self._load_model_quiet(name, checkpoint_type)
    model(sample_x)

print(sampler.delta_mb)  # CPU RAM increase in MiB
```

> [!NOTE]
> The measurement is approximate. The OS allocator may reuse already-reserved pages (giving `delta_mb ≈ 0`) and very short blocks can be missed between samples.

---

## Standalone helper functions

### `to_jsonable(obj) -> Any`

Recursively converts nearly any Python object into JSON-compatible types.

| Input type | Output |
|---|---|
| `None`, `str`, `bool` | Unchanged. |
| `int`, `np.integer` | `int` |
| `float`, `np.floating` | `float`; `NaN`/`Inf` → `None` |
| `np.ndarray` | `list` (via `.tolist()`) |
| `torch.Tensor` | `list` (via `.detach().cpu().tolist()`) |
| `pd.DataFrame` | `list[dict]` (via `to_dict("records")`) |
| `pd.Series` | `dict` (via `to_dict()`) |
| dataclass instance | `dict` of field names → values |
| `dict` | `dict` with string keys |
| `list`, `tuple`, `set` | `list` |
| `Enum` | `to_jsonable(obj.value)` |
| `datetime` | ISO 8601 string |
| `os.PathLike` | `os.fspath(obj)` |
| Other | `str(obj)` |

### `_config_to_dict(config) -> Dict[str, Any]`

Serializes an `ExperimentConfig` (or any dataclass/object with `__dict__`) into a JSON-compatible dictionary. Additionally extracts the computed properties `name`, `group`, `architecture`, `optimizer_text`, `batch_size`, `epochs`, and `seed`.

### `_flatten(prefix, value, out) -> None`

Flattens nested dicts into `"a.b"` keys. Lists are joined into a comma-separated string. Used by `build_detail_report()` to format config fields.

### `_hist_at(history, key, index) -> Optional[float]`

Safe lookup into a training history dictionary. Returns `None` if the key or index is out of range.

### `_safe_sheet_name(name, used) -> str`

Produces an Excel-safe, unique sheet name: at most 31 characters, no `[]:*?/\` characters. If the sanitized name collides with an already-used sheet, a numeric suffix is appended.

### `_lines_to_markdown(title, lines) -> str`

Converts a list of console-style lines into simple Markdown. Lines starting with `"  - "` become bullet items; blank lines separate paragraphs.

### `_fmt_delta(value, base) -> str`

Formats the difference from the baseline in percentage points.

```text
_fmt_delta(89.5, 88.0)   # "+1.50 pp vs baseline"
_fmt_delta(87.0, 88.0)   # "-1.00 pp vs baseline"
_fmt_delta(89.5, None)   # ""
```
