"""Self-contained HTML report generator for autogen pipeline runs."""

from datetime import datetime
from pathlib import Path

from jinja2 import Environment

from src import config as cfg
from src.logger import get_logger

logger = get_logger(__name__, cfg.logging["log_file"], cfg.logging["level"])

_TEMPLATE_SRC = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Autogen Report — {{ bank }} {{ instrument }}</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; background: #f8f9fa; }
  h1 { font-size: 1.4rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #dee2e6; padding-bottom: 0.3rem; }
  table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.88rem; }
  th, td { border: 1px solid #dee2e6; padding: 0.4rem 0.6rem; text-align: left; }
  th { background: #e9ecef; }
  .badge { display: inline-block; padding: 0.2rem 0.55rem; border-radius: 0.25rem; font-size: 0.78rem; font-weight: 600; }
  .badge-success { background: #d1e7dd; color: #0a3622; }
  .badge-fail { background: #f8d7da; color: #58151c; }
  .badge-na { background: #e2e3e5; color: #41464b; }
  .winner { background: #fff3cd; font-weight: 600; }
  .warning-banner { background: #fff3cd; border: 1px solid #ffc107; border-radius: 0.3rem; padding: 0.6rem 1rem; margin: 0.75rem 0; font-size: 0.9rem; }
  .info-banner { background: #cff4fc; border: 1px solid #0dcaf0; border-radius: 0.3rem; padding: 0.6rem 1rem; margin: 0.75rem 0; font-size: 0.9rem; }
  pre { background: #212529; color: #f8f9fa; padding: 1rem; border-radius: 0.3rem; overflow-x: auto; font-size: 0.82rem; }
  code { font-family: 'Courier New', monospace; }
  details summary { cursor: pointer; font-weight: 600; }
  .kv { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; font-size: 0.9rem; }
  .kv dt { color: #6c757d; }
</style>
</head>
<body>

<h1>Autogen Pipeline Report</h1>

{# ── 1. Run Summary ── #}
<h2>Run Summary</h2>
{% if below_threshold %}
<div class="warning-banner">&#9888; Winner score is below the 0.70 threshold — review extraction quality before proceeding.</div>
{% endif %}
<dl class="kv">
  <dt>Bank</dt><dd>{{ bank }}</dd>
  <dt>Instrument</dt><dd>{{ instrument }}</dd>
  {% if timestamp %}<dt>Timestamp</dt><dd>{{ timestamp }}</dd>{% endif %}
  <dt>Overall</dt><dd><span class="badge {% if success %}badge-success{% else %}badge-fail{% endif %}">{{ 'PASS' if success else 'FAIL' }}</span></dd>
  {% if winner_extractor is not none %}<dt>Winner Extractor</dt><dd>{{ winner_extractor }}</dd>{% endif %}
  {% if winner_score is not none %}<dt>Winner Score</dt><dd>{{ "%.4f"|format(winner_score) }}</dd>{% endif %}
  {% if codegen_backend is not none %}<dt>Codegen Backend</dt><dd>{{ codegen_backend }}</dd>{% endif %}
  {% if anonymization_applied is not none %}<dt>Anonymization</dt><dd>{{ 'Yes' if anonymization_applied else 'No' }}</dd>{% endif %}
  {% if hitl_approved is not none %}
    <dt>HITL Approved</dt><dd><span class="badge {% if hitl_approved %}badge-success{% else %}badge-fail{% endif %}">{{ 'Yes' if hitl_approved else 'No' }}</span></dd>
  {% elif anonymization_applied %}
    <dt>HITL Approved</dt><dd><span class="badge badge-na">N/A</span></dd>
  {% endif %}
</dl>

{# ── 2. Sample Set ── #}
{% if dev_samples or test_samples %}
<h2>Sample Set</h2>
{% if dev_samples %}
<p><strong>Dev samples</strong></p>
<table>
  <thead><tr><th>ID</th><th>Name</th><th>Date</th><th>File</th></tr></thead>
  <tbody>
  {% for a in dev_samples %}
    <tr>
      <td>{{ a.attachment_id }}</td>
      <td>{{ a.name }}</td>
      <td>{{ a.date.strftime('%Y-%m-%d') if a.date else '—' }}</td>
      <td>{{ a.physical_file or '—' }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}
{% if test_samples %}
<p><strong>Test samples</strong></p>
<table>
  <thead><tr><th>ID</th><th>Name</th><th>Date</th><th>File</th></tr></thead>
  <tbody>
  {% for a in test_samples %}
    <tr>
      <td>{{ a.attachment_id }}</td>
      <td>{{ a.name }}</td>
      <td>{{ a.date.strftime('%Y-%m-%d') if a.date else '—' }}</td>
      <td>{{ a.physical_file or '—' }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}
{% endif %}

{# ── 3. Extraction Scorecard ── #}
{% if scores %}
<h2>Extraction Scorecard</h2>
{% if below_threshold %}
<div class="warning-banner">&#9888; No extractor met the 0.70 quality threshold.</div>
{% endif %}
<table>
  <thead><tr><th>Extractor</th><th>Composite Score</th></tr></thead>
  <tbody>
  {% for extractor, score in scores.items() %}
    <tr class="{{ 'winner' if extractor == winner_extractor else '' }}">
      <td>{{ extractor }}{% if extractor == winner_extractor %} &#10003;{% endif %}</td>
      <td>{{ "%.4f"|format(score) }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{# ── 4. Anonymization Preview ── #}
{% if anonymized_tables %}
<h2>Anonymization Preview</h2>
<details open>
  <summary>Anonymized Tables</summary>
  <div class="info-banner">Cells anonymized. Column headers preserved.</div>
  {% for tbl in anonymized_tables %}
  {% if tbl.rows %}
  <p><em>{{ tbl.name or ('Page ' ~ tbl.page) }}</em></p>
  <table>
    <tbody>
    {% for row in tbl.rows %}
      <tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>
    {% endfor %}
    </tbody>
  </table>
  {% endif %}
  {% endfor %}
</details>
{% endif %}

{# ── 5. Generated Script ── #}
{% if generated_source %}
<h2>Generated Script</h2>
<pre><code>{{ generated_source }}</code></pre>
{% endif %}

{# ── 6. Test Execution Results ── #}
{% if test_results %}
<h2>Test Execution Results</h2>
{% for r in test_results %}
<details>
  <summary>
    {{ r.name }} &mdash; <span class="badge {% if r.success %}badge-success{% else %}badge-fail{% endif %}">{{ 'PASS' if r.success else 'FAIL' }}</span>
  </summary>
  <dl class="kv" style="margin-top:0.4rem">
    <dt>Attachment ID</dt><dd>{{ r.attachment_id }}</dd>
    {% if r.row_count is not none %}<dt>Row Count</dt><dd>{{ r.row_count }}</dd>{% endif %}
    {% if r.schema_conforms is not none %}<dt>Schema Conforms</dt><dd>{{ 'Yes' if r.schema_conforms else 'No' }}</dd>{% endif %}
    {% if r.columns %}<dt>Columns</dt><dd>{{ r.columns | join(', ') }}</dd>{% endif %}
    {% if r.null_rates %}
    <dt>Null Rates</dt>
    <dd>
      {% for col, rate in r.null_rates.items() %}{{ col }}: {{ "%.1f%%"|format(rate * 100) }}{% if not loop.last %}, {% endif %}{% endfor %}
    </dd>
    {% endif %}
    {% if r.error %}<dt>Error</dt><dd>{{ r.error }}</dd>{% endif %}
  </dl>
</details>
{% endfor %}
{% endif %}

{# ── 7. Success Criteria ── #}
{% if criteria %}
<h2>Success Criteria</h2>
<table>
  <thead><tr><th>ID</th><th>Description</th><th>Result</th></tr></thead>
  <tbody>
  {% for cid, desc, passed in criteria %}
    <tr>
      <td>{{ cid }}</td>
      <td>{{ desc }}</td>
      <td><span class="badge {% if passed %}badge-success{% else %}badge-fail{% endif %}">{{ 'PASS' if passed else 'FAIL' }}</span></td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

{# ── 8. Ruff Lint ── #}
{% if lint_findings is not none %}
<h2>Ruff Lint</h2>
{% if lint_findings %}
<pre><code>{% for line in lint_findings %}{{ line }}
{% endfor %}</code></pre>
{% else %}
<div class="info-banner">No ruff findings.</div>
{% endif %}
{% endif %}

</body>
</html>
"""

_ENV = Environment(autoescape=True)
_TEMPLATE = _ENV.from_string(_TEMPLATE_SRC)


def render_report(ctx: dict) -> str:
    """Render the full self-contained HTML report from a context dict."""
    safe = {
        "bank": ctx.get("bank", ""),
        "instrument": ctx.get("instrument", ""),
        "timestamp": ctx.get("timestamp"),
        "success": ctx.get("success", False),
        "winner_extractor": ctx.get("winner_extractor"),
        "winner_score": ctx.get("winner_score"),
        "codegen_backend": ctx.get("codegen_backend"),
        "anonymization_applied": ctx.get("anonymization_applied"),
        "hitl_approved": ctx.get("hitl_approved"),
        "below_threshold": ctx.get("below_threshold", False),
        "scores": ctx.get("scores"),
        "dev_samples": ctx.get("dev_samples"),
        "test_samples": ctx.get("test_samples"),
        "anonymized_tables": ctx.get("anonymized_tables"),
        "generated_source": ctx.get("generated_source"),
        "test_results": ctx.get("test_results"),
        "criteria": ctx.get("criteria"),
        "lint_findings": ctx.get("lint_findings"),
    }
    return _TEMPLATE.render(**safe)


def write_report(ctx: dict, out_dir: str = "data/control") -> str:
    """Render and write to {out_dir}/autogen_{bank}_{instrument}_{YYYYMMDD_HHMMSS}.html; return path."""
    bank = ctx.get("bank", "unknown")
    instrument = ctx.get("instrument", "unknown")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"autogen_{bank}_{instrument}_{ts}.html"
    html = render_report(ctx)
    path.write_text(html, encoding="utf-8")
    logger.info("report written: %s", path)
    return str(path)
