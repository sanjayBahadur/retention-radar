"""Build a self-contained report from the saved retail analysis."""

import csv
import io
import json
from html import escape
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from retention_radar.config import get_project_root, load_project_config

REPOSITORY = "https://github.com/sanjayBahadur/retention-radar"


def category_chart(categories: pd.DataFrame) -> str:
    """Display changes in category share in percentage points."""
    rows = categories.sort_values("net_drop_effect").head(6)
    maximum = max(rows.net_drop_effect.abs().max() * 100, 0.01)
    bars = []
    for row in rows.itertuples():
        change = row.net_drop_effect * 100
        bars.append(
            f"<li><span>{escape(row.commodity_desc.title())}</span>"
            f'<span class="bar-track"><span style="width:{abs(change) / maximum * 100:.1f}%"></span></span>'
            f"<strong>{change:+.2f}</strong></li>"
        )
    return '<ul class="bars">' + "".join(bars) + "</ul>"


def comparison_table(frame: pd.DataFrame) -> str:
    """Show the benchmark and its household-clustered intervals."""
    rows = "".join(
        f'<tr><th scope="row">{escape(row.model)}</th><td>{row.pr_auc:.3f}</td>'
        f"<td>{row.pr_auc_ci_lower:.3f}–{row.pr_auc_ci_upper:.3f}</td>"
        f"<td>{row.top_20_capture:.1%}</td></tr>"
        for row in frame.itertuples()
    )
    return (
        '<div class="table-scroll"><table><caption>Churn models · later test periods</caption>'
        '<thead><tr><th scope="col">Model</th><th scope="col">Average precision</th>'
        '<th scope="col">95% interval</th><th scope="col">Departures captured</th>'
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def roster_download(scores: pd.DataFrame) -> str:
    """Embed the CSV so the report can be used offline."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Household ID",
            "Current state",
            "Estimated departure probability",
            "Estimated 26-week margin ($)",
            "Main score factor",
        ]
    )
    columns = ["household_key", "current_state", "calibrated_p", "clv_26w", "reason_1"]
    ranked = scores.sort_values(["calibrated_p", "raw_p"], ascending=False)
    writer.writerows(ranked[columns].itertuples(index=False, name=None))
    return "data:text/csv;charset=utf-8," + quote(output.getvalue())


def build_interactive_html(out_dir: Path, reports_dir: Path, dest_path: Path) -> Path:
    """Render the findings and an adjustable cost scenario."""
    results = json.loads((out_dir / "results.json").read_text())
    categories = pd.read_parquet(out_dir / "early_warning_categories.parquet")
    comparison = pd.read_parquet(out_dir / "churn_model_comparison.parquet")
    values = pd.read_parquet(out_dir / "clv_comparison.parquet").sort_values("mae_dollars")
    scores = pd.read_parquet(out_dir / "churn_scores.parquet")
    errors = pd.read_parquet(out_dir / "churn_error_analysis.parquet")
    overall = errors.loc[errors.dimension == "Overall"].iloc[0]
    financial = results["financial_impact"]
    capture = results["churn"]["top_20pct_capture_rate"]
    experiment = results["uplift"]["ab_test"]
    diapers = categories.set_index("commodity_desc").loc["DIAPERS & DISPOSABLES"]
    poultry = categories.set_index("commodity_desc").loc["CHICKEN/POULTRY"]
    assumptions = {
        "population": financial["scale_households"],
        "churnRate": financial["base_churn_rate"],
        "capture": capture,
        "contactShare": financial["top_20pct_contacted_households"] / financial["scale_households"],
        "cost": financial["contact_cost_per_household_dollars"],
        "clv": results["clv"]["mean_26w_clv_margin_dollars"],
    }
    value_rows = "".join(
        f'<tr><th scope="row">{escape(row.method)}</th><td>${row.mae_dollars:,.2f}</td></tr>'
        for row in values.itertuples()
    )
    study_plan = (
        f"The proposed study needs {experiment['total_sample_size']:,} households, "
        f"against {experiment['available_active_households']:,} in the current eligible group. "
        "I would start with a pilot to refine the spending variance and size a larger study."
        if not experiment["feasible_in_available_pool"]
        else f"The proposed study needs {experiment['total_sample_size']:,} households "
        f"from an eligible group of {experiment['available_active_households']:,}."
    )
    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Grocery shopping patterns, churn prediction, and retention costs. A data project by Sanjay Bahadur.">
<meta name="theme-color" content="#f5f3ec">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Ccircle cx=%2716%27 cy=%2716%27 r=%2711%27 fill=%27none%27 stroke=%27%23234e3b%27 stroke-width=%274%27/%3E%3C/svg%3E">
<title>Retention Radar | Retail retention analysis</title>
<style>
:root {{ --paper:#f5f3ec; --ink:#252d28; --muted:#59645c; --green:#234e3b; --pale:#e3e9de; --line:#cdd2c8; --amber:#865715; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; scroll-padding-top:24px; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font:16px/1.65 Arial,Helvetica,sans-serif; }}
a {{ color:var(--green); text-underline-offset:4px; }}
a:hover {{ text-decoration-thickness:2px; }}
:focus-visible {{ outline:3px solid var(--amber); outline-offset:4px; }}
.skip {{ position:absolute; top:-80px; left:16px; padding:10px; background:white; z-index:5; }}
.skip:focus {{ top:10px; }}
.wrap {{ width:min(1040px,calc(100% - 80px)); margin:auto; }}
header {{ display:flex; align-items:center; justify-content:space-between; gap:24px; padding:23px 0; border-bottom:1px solid var(--line); }}
.wordmark {{ white-space:nowrap; font-weight:700; text-decoration:none; }}
nav {{ display:flex; gap:22px; font-size:.82rem; }}
nav a {{ text-decoration:none; }}
h1,h2,h3 {{ font-family:Georgia,'Times New Roman',serif; font-weight:400; line-height:1.2; letter-spacing:-.025em; }}
h1 {{ font-size:clamp(2.2rem,4vw,3.25rem); margin:0 0 18px; max-width:18ch; }}
h2 {{ font-size:clamp(1.7rem,2.8vw,2.25rem); margin:0 0 22px; }}
h3 {{ font-size:1.35rem; margin:0 0 12px; }}
p {{ margin:0 0 16px; }}
.hero {{ padding:44px 0 36px; display:grid; grid-template-columns:1.6fr 1fr; gap:64px; align-items:center; }}
.author,.dataset {{ font-size:.82rem; color:var(--muted); }}
.author {{ margin-bottom:14px; }}
.lead {{ max-width:40em; }}
.result {{ border-left:2px solid var(--green); padding-left:24px; }}
.result strong {{ display:block; font:400 3.4rem/1.1 Georgia,serif; color:var(--green); margin-bottom:10px; }}
.result p {{ font-size:.92rem; }}
.result a {{ font-size:.82rem; }}
.dataset {{ border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:14px 0; margin:0; }}
section {{ padding:42px 0; border-bottom:1px solid var(--line); }}
.section-intro {{ max-width:48em; }}
.category-layout {{ display:grid; grid-template-columns:1.5fr 1fr; gap:48px; align-items:center; margin-top:24px; }}
figure {{ margin:0; }}
figcaption,.small {{ font-size:.8rem; color:var(--muted); }}
figcaption {{ margin-top:14px; }}
.bars {{ padding:0; margin:20px 0; list-style:none; }}
.bars li {{ display:grid; grid-template-columns:1.2fr 1fr auto; gap:14px; align-items:center; margin-bottom:20px; font-size:.84rem; }}
.bars strong {{ font:400 .82rem monospace; min-width:52px; text-align:right; }}
.bar-track {{ height:12px; background:#e0e4db; display:block; }}
.bar-track span {{ height:100%; display:block; background:var(--green); }}
.category-summary {{ font-size:.92rem; }}
.risk-layout {{ display:grid; grid-template-columns:1.1fr 1fr; gap:56px; align-items:start; margin:28px 0; }}
.coverage {{ margin:0 0 24px; }}
.coverage p {{ display:flex; justify-content:space-between; gap:12px; margin:0 0 8px; font-size:.88rem; }}
.coverage .bar-track {{ height:22px; }}
.coverage.baseline .bar-track span {{ background:#a8b3a3; }}
.value-layout {{ display:grid; grid-template-columns:1fr 1fr; gap:56px; align-items:start; }}
.table-scroll {{ overflow-x:auto; max-width:100%; }}
table {{ border-collapse:collapse; width:100%; font-size:.84rem; text-align:left; }}
caption {{ text-align:left; font-weight:700; margin-bottom:14px; }}
th,td {{ border-bottom:1px solid var(--line); padding:12px 14px 12px 0; vertical-align:top; }}
th {{ font-weight:500; }}
thead th {{ color:var(--muted); font-weight:400; }}
td {{ font-variant-numeric:tabular-nums; }}
.simulator {{ display:grid; grid-template-columns:1fr 1fr; gap:44px; background:var(--pale); padding:28px; margin:26px 0 18px; }}
.simulator label {{ display:block; font-weight:700; margin-top:18px; }}
input[type=range] {{ display:block; width:100%; accent-color:var(--green); margin:22px 0 5px; cursor:pointer; }}
.range-labels {{ display:flex; justify-content:space-between; font-size:.76rem; }}
.sim-results {{ padding-left:28px; border-left:1px solid #b7c2af; }}
.sim-results p {{ font-size:.87rem; }}
.sim-results .big {{ display:block; font:400 2.8rem/1.15 Georgia,serif; color:var(--green); margin:14px 0; font-variant-numeric:tabular-nums; }}
.sim-results[data-outcome="loss"] .big {{ color:var(--amber); }}
details {{ border-top:1px solid var(--line); padding:18px 0; }}
summary {{ cursor:pointer; font-size:.9rem; font-weight:600; }}
details[open] summary {{ margin-bottom:18px; }}
details p,details li {{ font-size:.86rem; }}
#methods {{ border-bottom:0; }}
.links {{ display:flex; flex-wrap:wrap; gap:10px 24px; margin-top:22px; font-size:.87rem; }}
footer {{ border-top:1px solid var(--line); padding:22px 0 32px; display:flex; justify-content:space-between; gap:20px; font-size:.8rem; color:var(--muted); }}
footer p {{ margin:0; }}
@media(max-width:700px) {{
.wrap {{ width:calc(100% - 36px); }} header {{ align-items:start; }} nav {{ gap:8px 14px; flex-wrap:wrap; justify-content:end; font-size:.74rem; }} nav a:first-child {{ display:none; }}
.hero {{ grid-template-columns:1fr; gap:22px; padding:30px 0 25px; }} .result {{ padding-left:18px; }} .result strong {{ font-size:2.7rem; }}
section {{ padding:32px 0; }} .category-layout,.risk-layout,.value-layout,.simulator {{ grid-template-columns:1fr; gap:20px; }}
.bars li {{ gap:10px; font-size:.75rem; }} .bars strong {{ min-width:44px; font-size:.75rem; }} .dataset {{ font-size:.75rem; }}
.simulator {{ padding:22px; }} .sim-results {{ padding:20px 0 0; border-left:0; border-top:1px solid #b7c2af; }} .sim-results .big {{ font-size:2.4rem; }}
footer {{ display:block; }} footer p {{ margin-bottom:8px; }} }}
@media(prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}
@media print {{ body {{ background:white; }} .wrap {{ width:100%; }} nav,.skip,input {{ display:none; }} section {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<a href="#patterns" class="skip">Skip to the findings</a>
<div class="wrap">
<header>
<a href="#" class="wordmark">Retention Radar</a>
<nav aria-label="Main">
<a href="#patterns">Shopping patterns</a><a href="#risk">Churn model</a><a href="#scenario">Costs</a><a href="#methods">Methods</a><a href="{REPOSITORY}">GitHub</a>
</nav>
</header>
<main>
<div class="hero">
<div>
<p class="author"><a href="https://sanjaymalla.com">Sanjay Bahadur</a> · Retail analytics</p>
<h1>Finding shoppers at risk of leaving</h1>
<p class="lead">I used grocery purchase histories to rank households by the chance of making no shopping trips in the next eight weeks. The aim was to give a retailer a smaller, more useful contact list.</p>
</div>
<div class="result">
<strong>{capture:.0%}</strong>
<p>of later departures fell in the highest-risk fifth of shoppers across four test periods.</p>
<a href="#risk">How the ranking performed</a>
</div>
</div>
<p class="dataset">dunnhumby Complete Journey · 2,500 households · 2.6 million transaction line items · 25 complete 28-day periods</p>
<section id="patterns">
<h2>Where spending changed</h2>
<p class="section-intro">I compared category spending before and around a household’s first dormant period with shoppers who stayed active. Diapers and poultry showed the largest differences.</p>
<div class="category-layout">
<figure>
<p class="small">Change in category share, relative to active households · percentage points</p>
{category_chart(categories)}
<figcaption>Each bar compares the change between two three-period windows around dormancy.</figcaption>
</figure>
<div class="category-summary">
<p>Diapers’ share fell <strong>{abs(diapers.net_drop_effect) * 100:.2f} percentage points</strong> more than in the comparison group. Poultry fell <strong>{abs(poultry.net_drop_effect) * 100:.2f} points</strong> more.</p>
<p class="small">{int(diapers.lapsing_hh_count):,} dormant-group households bought diapers; {int(poultry.lapsing_hh_count):,} bought poultry during the comparison windows.</p>
</div>
</div>
</section>
<section id="risk">
<h2>The highest-risk fifth contained {capture:.0%} of departures</h2>
<p class="section-intro">At each test cutoff, I ranked households by risk and selected the top 20%. That group contained {int(overall.captured_departures):,} of the {int(overall.actual_departures):,} subsequent departures.</p>
<div class="risk-layout">
<figure>
<div class="coverage"><p><span>Highest-risk 20%</span><strong>{capture:.0%} captured</strong></p><span class="bar-track"><span style="width:{capture * 100:.1f}%"></span></span></div>
<div class="coverage baseline"><p><span>Random 20%</span><strong>20% expected</strong></p><span class="bar-track"><span style="width:20%"></span></span></div>
<figcaption>Share of subsequent departures in a contact list of the same size. Random selection is the expected baseline.</figcaption>
</figure>
<div>
<p>Recent visits, spending trends and basket activity feed the ranking. I compared LightGBM with logistic regression and two simple rules based on recency and lifecycle state.</p>
<p><a href="{REPOSITORY}/blob/main/reports/evaluation.md">Error breakdown by period and customer group</a></p>
</div>
</div>
<details><summary>Model comparison</summary>{comparison_table(comparison)}<p class="small">Average precision summarizes ranking quality. Intervals use household-clustered bootstrap samples. Departure means no visits over the next 56 days; the 20% selection budget is applied separately at each cutoff.</p></details>
<details><summary>Household scores</summary>
<p>The scored list includes a household ID, departure probability, estimated 26-week margin and the main factor contributing to the score.</p>
<a download="retention-radar-households.csv" href="{roster_download(scores)}">Download scored households (.csv)</a>
</details>
</section>
<section id="value">
<h2>The simple spending forecast held up best</h2>
<div class="value-layout">
<div><p>For customer value, I compared forecasts of the next 26 weeks of sales. Using the previous 26 weeks of spending gave the lowest average error in this test.</p><p>That baseline supplies the latest customer-value estimates. The historical churn list uses a LightGBM forecast fixed before its scoring date.</p></div>
<div class="table-scroll"><table><caption>Mean absolute error · 26-week sales dollars</caption><thead><tr><th scope="col">Forecast</th><th scope="col">Average error</th></tr></thead><tbody>{value_rows}</tbody></table></div>
</div>
</section>
<section id="scenario">
<h2>What response would cover the contact cost?</h2>
<p class="section-intro">This scenario applies the ranking to {financial["scale_households"]:,} households, with one ${assumptions["cost"]:.2f} contact for each household in the top fifth. Adjust the assumed save rate to compare the estimated return.</p>
<div class="simulator" aria-label="Retention cost scenario">
<div><h3>Offer response</h3><p class="small">Save rate is the share of potential departures prevented by the offer.</p>
<label for="save-rate">Assumed save rate: <output id="save-label" for="save-rate">10%</output></label>
<input id="save-rate" type="range" min="0" max="20" step="1" value="10">
<div class="range-labels"><span>0%</span><span>20%</span></div></div>
<div class="sim-results" aria-live="polite" aria-atomic="true"><p>Estimated net margin · 26 weeks</p>
<output class="big" id="net-margin">${financial["net_margin_per_26w_cycle_dollars"]:,.0f}</output>
<p><output id="saved-households">{financial["saved_households"]:,.0f}</output> households retained at the selected rate<br>${financial["total_campaign_cost_dollars"]:,.0f} contact cost</p></div>
</div>
<noscript><p>The initial estimate uses a 10% save rate. Enable JavaScript to adjust it.</p></noscript>
<details><summary>Scenario inputs and calculation</summary>
<p>Inputs: {assumptions["churnRate"]:.2%} departure rate, {capture:.1%} capture, ${assumptions["clv"]:,.2f} mean estimated 26-week margin and ${assumptions["cost"]:.2f} per contact. Estimated return = departures captured × save rate × mean margin − contact cost.</p>
<p>The save rate is a planning assumption. Customer margin uses a {results["clv"]["validation"]["gross_margin_assumption"]:.0%} gross-margin rate and a {results["clv"]["validation"]["annual_discount_rate_assumption"]:.0%} annual discount rate.</p></details>
<details><summary>Offer experiment</summary>
<p>I would test one offer against a no-offer group over eight weeks, assigning each household once and comparing spending and net margin. {study_plan}</p>
<p><a href="{REPOSITORY}/blob/main/data/outputs/ab_test_design.json">Sample-size calculation</a></p></details>
</section>
<section id="methods">
<h2>Data and methods</h2>
<p>SQL and dbt turn product-line transactions into baskets and household histories. Python handles the forecasts, evaluation and report generation.</p>
<details><summary>Evaluation setup</summary>
<p>Churn models train on cutoffs 6–13, tune and calibrate on 16–17, and evaluate on 20–23. Customer-value training outcomes end on day 518; evaluation outcomes cover days 519–700.</p>
<p>The category comparison is retrospective and the campaign-response analysis uses historical targeting. The technical report contains the study assumptions, subgroup checks and lifecycle diagnostics.</p></details>
<details><summary>Terms used here</summary>
<p><strong>Departure:</strong> no visits in the next 56 days. <strong>Dormant:</strong> two or three missed 28-day periods. <strong>Percentage point:</strong> a share falling from 10% to 7% falls three points. <strong>Customer value:</strong> forecast spending converted to estimated margin.</p></details>
<div class="links"><a href="{REPOSITORY}/blob/main/reports/technical_report.md">Technical report</a><a href="{REPOSITORY}/blob/main/docs/data_dictionary.md">Data dictionary</a><a href="{REPOSITORY}/tree/main/dbt/analyses">SQL queries</a><a href="{REPOSITORY}">Code and setup</a></div>
</section>
</main>
<footer><p>Sanjay Bahadur</p><p>Retention Radar · dunnhumby Complete Journey</p></footer>
</div>
<script>
const assumptions = {json.dumps(assumptions)};
const slider = document.getElementById('save-rate');
function updateScenario() {{
  const rate = Number(slider.value) / 100;
  const saved = assumptions.population * assumptions.churnRate * assumptions.capture * rate;
  const cost = assumptions.population * assumptions.contactShare * assumptions.cost;
  const net = saved * assumptions.clv - cost;
  document.getElementById('save-label').textContent = slider.value + '%';
  document.getElementById('saved-households').textContent = Math.round(saved).toLocaleString('en-US');
  document.querySelector('.sim-results').dataset.outcome = net < 0 ? 'loss' : 'gain';
  document.getElementById('net-margin').textContent = net.toLocaleString('en-US', {{style:'currency',currency:'USD',maximumFractionDigits:0}});
}}
slider.addEventListener('input', updateScenario);
updateScenario();
</script>
</body>
</html>'''
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(html, encoding="utf-8")
    return dest_path


def main() -> None:
    root = get_project_root()
    config = load_project_config()
    out_dir = root / config.paths.data_outputs
    reports_dir = root / config.paths.reports_dir
    path = build_interactive_html(out_dir, reports_dir, root / "docs" / "index.html")
    (reports_dir / "interactive_report.html").write_text(path.read_text(), encoding="utf-8")
    print(f"Built {path}")


if __name__ == "__main__":
    main()
