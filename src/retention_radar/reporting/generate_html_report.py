"""Build the public, self-contained case study from the saved analysis outputs."""

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
    """Render a readable chart with the same information available as text."""
    rows = categories.sort_values("net_drop_effect").head(6)
    maximum = max(rows["net_drop_effect"].abs().max() * 100, 0.01)
    bars = []
    for row in rows.itertuples():
        change = row.net_drop_effect * 100
        width = abs(change) / maximum * 100
        bars.append(
            f'<li><span class="bar-label">{escape(row.commodity_desc.title())}</span>'
            f'<span class="bar-track"><span style="width:{width:.1f}%"></span></span>'
            f"<strong>{change:+.2f} pp</strong></li>"
        )
    return '<ul class="bars">' + "".join(bars) + "</ul>"


def comparison_table(frame: pd.DataFrame) -> str:
    """Keep the model comparison compact and accessible."""
    rows = []
    for row in frame.itertuples():
        rows.append(
            f"<tr><th scope='row'>{escape(row.model)}</th><td>{row.pr_auc:.3f}</td>"
            f"<td>{row.pr_auc_ci_lower:.3f}–{row.pr_auc_ci_upper:.3f}</td>"
            f"<td>{row.top_20_capture:.1%}</td></tr>"
        )
    return (
        '<div class="table-scroll"><table><caption>Churn model performance on future test periods</caption>'
        '<thead><tr><th scope="col">Model</th><th scope="col">PR-AUC ↑</th>'
        '<th scope="col">95% interval</th><th scope="col">Departures captured in top 20%</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def roster_download(scores: pd.DataFrame) -> str:
    """Embed a CSV download so the published page has no asset-path dependency."""
    columns = ["household_key", "current_state", "calibrated_p", "clv_26w", "reason_1"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Household ID",
            "Current state",
            "Estimated departure probability",
            "Estimated 26-week margin ($)",
            "Primary model reason",
        ]
    )
    for row in scores.sort_values("calibrated_p", ascending=False)[columns].itertuples(
        index=False, name=None
    ):
        writer.writerow(row)
    return "data:text/csv;charset=utf-8," + quote(output.getvalue())


def build_interactive_html(out_dir: Path, reports_dir: Path, dest_path: Path) -> Path:
    """Build a report; missing required artifacts fail visibly instead of inventing metrics."""
    with open(out_dir / "results.json", encoding="utf-8") as source:
        results = json.load(source)
    categories = pd.read_parquet(out_dir / "early_warning_categories.parquet")
    comparison = pd.read_parquet(out_dir / "churn_model_comparison.parquet")
    scores = pd.read_parquet(out_dir / "churn_scores.parquet")
    churn = results["churn"]
    financial = results["financial_impact"]
    clv = results["clv"]
    lifecycle = json.loads((out_dir / "markov_metrics.json").read_text())
    experiment = results["uplift"]["ab_test"]
    study_note = (
        "This proposal needs a larger study pool or a revised design before launch."
        if not experiment["feasible_in_available_pool"]
        else "The available pool meets this planning calculation; pilot data should refine it."
    )
    capture = churn["top_20pct_capture_rate"]
    selected = comparison.loc[comparison["model"] == "LightGBM"].iloc[0]
    lr = comparison.loc[comparison["model"].str.contains("Logistic", case=False)].iloc[0]
    assumptions = {
        "population": financial["scale_households"],
        "churnRate": financial["base_churn_rate"],
        "capture": capture,
        "contactShare": financial["top_20pct_contacted_households"] / financial["scale_households"],
        "cost": financial["contact_cost_per_household_dollars"],
        "clv": clv["mean_26w_clv_margin_dollars"],
    }
    default_net = financial["net_margin_per_26w_cycle_dollars"]
    default_saved = financial["saved_households"]
    dots = "".join(
        f'<circle cx="{18 + (i % 10) * 28}" cy="{18 + (i // 10) * 28}" r="8" '
        f'class="{"contact-dot" if i < 20 else "base-dot"}" />'
        for i in range(100)
    )
    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A guided retail data case study: shopping patterns, departure risk, and retention ideas worth testing. By Sanjay Bahadur.">
<meta name="theme-color" content="#f5f3ec">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 32 32%27%3E%3Ccircle cx=%2716%27 cy=%2716%27 r=%2711%27 fill=%27none%27 stroke=%27%23234e3b%27 stroke-width=%274%27/%3E%3C/svg%3E">
<title>Retention Radar — A retail data story</title>
<style>
:root {{ --paper:#f5f3ec; --ink:#252d28; --muted:#59645c; --green:#234e3b; --pale:#e3e9de; --line:#cdd2c8; --amber:#865715; --white:#fffef9; }}
* {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; scroll-padding-top:2rem; }}
body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Arial,Helvetica,sans-serif; line-height:1.65; }}
a {{ color:var(--green); text-underline-offset:4px; }}
a:hover {{ text-decoration-thickness:2px; }}
:focus-visible {{ outline:3px solid var(--amber); outline-offset:5px; }}
.skip {{ position:absolute; top:-80px; left:16px; padding:10px; background:var(--white); z-index:5; }}
.skip:focus {{ top:10px; }}
.wrap {{ width:min(1120px,calc(100% - 96px)); margin:auto; }}
header {{ display:flex; align-items:center; justify-content:space-between; gap:24px; padding:24px 0; border-bottom:1px solid var(--line); }}
.wordmark {{ white-space:nowrap; font-weight:700; letter-spacing:-.5px; text-decoration:none; }}
.mark {{ display:inline-block; width:14px; height:14px; border:2px solid var(--green); border-radius:50%; margin-right:8px; box-shadow:0 0 0 4px var(--pale); }}
nav {{ display:flex; gap:24px; font-size:.82rem; }}
nav a {{ text-decoration:none; }}
.hero {{ padding:80px 0 54px; display:grid; grid-template-columns:1.5fr 1fr; gap:80px; align-items:center; }}
.eyebrow {{ font:700 .72rem/1.5 Arial,sans-serif; letter-spacing:.16em; text-transform:uppercase; color:var(--green); margin:0 0 18px; }}
h1,h2,h3 {{ font-family:Georgia,'Times New Roman',serif; font-weight:400; line-height:1.12; letter-spacing:-.04em; }}
h1 {{ font-size:clamp(2.8rem,5vw,4.8rem); margin:0 0 24px; }}
h2 {{ font-size:clamp(2rem,3.5vw,3rem); margin:0 0 20px; }}
h3 {{ font-size:1.6rem; margin:0 0 16px; }}
p {{ margin:0 0 18px; }}
.intro {{ color:var(--muted); max-width:40em; font-size:1.05rem; }}
.byline {{ font-size:.82rem; color:var(--muted); margin-top:26px; }}
.hero-aside {{ padding:24px 0 24px 30px; border-left:1px solid var(--line); }}
.hero-aside p {{ font-size:.88rem; color:var(--muted); }}
.fact {{ margin:24px 0; }}
.fact strong {{ display:block; font:400 2rem/1.15 Georgia,serif; color:var(--green); }}
.fact span {{ font-size:.8rem; color:var(--muted); }}
.route {{ display:grid; grid-template-columns:repeat(3,1fr); gap:24px; border-top:1px solid var(--line); padding:24px 0 36px; }}
.route a {{ text-decoration:none; display:flex; gap:14px; align-items:baseline; font-size:.88rem; }}
.route span {{ font:italic 1.2rem Georgia,serif; color:var(--muted); }}
section {{ border-top:1px solid var(--line); padding:64px 0; }}
.section-head {{ display:grid; grid-template-columns:1fr 1fr; gap:64px; margin-bottom:36px; }}
.section-head p {{ color:var(--muted); }}
.two-col {{ display:grid; grid-template-columns:1.35fr 1fr; gap:48px; align-items:start; }}
.figure {{ background:var(--white); border:1px solid var(--line); padding:28px; border-radius:3px; margin:0; }}
.figure-label {{ font-weight:700; font-size:.9rem; margin-bottom:4px; }}
.note,figcaption {{ font-size:.8rem; color:var(--muted); }}
figcaption {{ margin-top:20px; }}
.bars {{ padding:0; margin:26px 0 0; list-style:none; }}
.bars li {{ display:grid; grid-template-columns:1.2fr 1fr auto; gap:14px; align-items:center; margin-bottom:20px; font-size:.8rem; }}
.bars strong {{ font:400 .8rem monospace; min-width:66px; text-align:right; }}
.bar-track {{ height:10px; background:#eef0e8; display:block; }}
.bar-track span {{ height:100%; display:block; background:var(--green); }}
.reading {{ padding:8px 0; }}
.reading h3 {{ font-size:1.5rem; }}
.reading p {{ font-size:.92rem; }}
.label {{ color:var(--green); font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; font-weight:700; margin-bottom:12px; }}
.limit {{ border-left:2px solid var(--amber); padding:0 0 0 18px; margin-top:28px; }}
.limit .label {{ color:var(--amber); }}
.risk-figure {{ display:flex; gap:30px; align-items:center; }}
.risk-figure svg {{ width:54%; flex-shrink:0; }}
.contact-dot {{ fill:var(--green); }}
.base-dot {{ fill:#dce0d5; }}
.big {{ display:block; font:400 3.4rem/1.05 Georgia,serif; color:var(--green); margin:8px 0 14px; }}
.callout {{ padding:24px 28px; background:var(--pale); margin-top:28px; }}
.callout p:last-child {{ margin-bottom:0; }}
.simulator {{ display:grid; grid-template-columns:1fr 1fr; gap:40px; background:var(--green); color:var(--white); padding:34px; }}
.simulator h3 {{ font-size:1.5rem; }}
.simulator .note {{ color:#d3ddcf; }}
.simulator label {{ display:block; font-weight:700; margin-top:22px; }}
input[type=range] {{ display:block; width:100%; accent-color:#ded3a7; margin:22px 0 4px; cursor:pointer; }}
.range-labels {{ display:flex; justify-content:space-between; font-size:.76rem; }}
.sim-results {{ border-left:1px solid #718874; padding-left:30px; }}
.sim-results .big {{ color:var(--white); font-size:2.7rem; }}
.sim-results p {{ font-size:.86rem; }}
.sim-results output {{ font-variant-numeric:tabular-nums; }}
.sim-results[data-outcome="loss"] .big {{ color:#f1ce8f; }}
.steps {{ display:grid; grid-template-columns:repeat(3,1fr); gap:28px; margin-top:34px; }}
.steps div {{ border-top:2px solid var(--green); padding-top:18px; }}
.steps h3 {{ font-size:1.35rem; }}
.steps p {{ font-size:.86rem; color:var(--muted); }}
details {{ border-top:1px solid var(--line); padding:20px 0; }}
details:last-child {{ border-bottom:1px solid var(--line); }}
summary {{ cursor:pointer; font-weight:700; font-size:.95rem; }}
details[open] summary {{ margin-bottom:22px; }}
details p,details li {{ font-size:.88rem; }}
.table-scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:.8rem; text-align:left; }}
caption {{ text-align:left; font-weight:700; margin-bottom:14px; }}
th,td {{ border-bottom:1px solid var(--line); padding:12px 14px 12px 0; vertical-align:top; }}
thead th {{ color:var(--muted); }}
.glossary {{ display:grid; grid-template-columns:1fr 1fr; gap:8px 32px; }}
.glossary dt {{ font-weight:700; font-size:.85rem; }}
.glossary dd {{ margin:0 0 16px; font-size:.85rem; color:var(--muted); }}
footer {{ border-top:1px solid var(--line); padding:26px 0 40px; display:flex; gap:20px; justify-content:space-between; font-size:.8rem; color:var(--muted); }}
footer p {{ margin:0; }}
@media(max-width:760px) {{ .wrap {{ width:calc(100% - 40px); }} header {{ align-items:start; }} nav {{ gap:12px; font-size:.75rem; flex-wrap:wrap; justify-content:end; }} nav a:first-child {{ display:none; }} .hero {{ padding:48px 0 30px; gap:28px; grid-template-columns:1fr; }} .hero-aside {{ padding:20px 0 0; border-left:0; border-top:1px solid var(--line); display:flex; gap:24px; flex-wrap:wrap; }} .hero-aside>p {{ width:100%; margin:0; }} .fact {{ margin:0; }} .fact strong {{ font-size:1.6rem; }} .route {{ gap:12px; }} .route a {{ display:block; font-size:.76rem; }} .route span {{ display:block; margin-bottom:6px; }} section {{ padding:42px 0; }} .section-head,.two-col,.simulator,.steps {{ grid-template-columns:1fr; gap:22px; }} .section-head {{ margin-bottom:22px; }} .figure {{ padding:20px 16px; }} .bars li {{ grid-template-columns:1.2fr .8fr auto; gap:10px; font-size:.73rem; }} .bars strong {{ font-size:.72rem; min-width:58px; }} .risk-figure {{ gap:18px; }} .big {{ font-size:2.5rem; }} .simulator {{ padding:24px; }} .sim-results {{ border-left:0; border-top:1px solid #718874; padding:22px 0 0; }} .glossary {{ grid-template-columns:1fr; }} footer {{ display:block; }} footer p {{ margin-bottom:12px; }} }}
@media(prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}
@media print {{ body {{ background:white; }} .wrap {{ width:100%; }} nav,.skip,input {{ display:none; }} details>* {{ display:block; }} section {{ break-inside:avoid; }} }}
</style>
</head>
<body>
<a href="#story" class="skip">Skip to the findings</a>
<div class="wrap">
<header>
<a href="#" class="wordmark">
<span class="mark" aria-hidden="true">
</span>Retention Radar</a>
<nav aria-label="Main">
<a href="#patterns">Shopping patterns</a>
<a href="#risk">Departure risk</a>
<a href="#next">What next?</a>
<a href="{REPOSITORY}">Code ↗</a>
</nav>
</header>
<main id="story">
<div class="hero">
<div>
<p class="eyebrow">A retail data case study · Sanjay Bahadur</p>
<h1>When shoppers<br>start drifting away.</h1>
<p class="intro">Can shopping patterns help a retailer decide who to contact—and what to test before spending money on retention?</p>
<p class="intro">This project follows grocery purchases from changing baskets to departure risk, then explores the limits of a targeted offer.</p>
<p class="byline">Python · SQL · DuckDB · dbt · statistical modeling<br>Historical analysis. Financial outcomes are modeled, not realized.</p>
</div>
<aside class="hero-aside" aria-label="Dataset context">
<p>The Complete Journey, a public grocery dataset from dunnhumby.</p>
<div class="fact">
<strong>2,500</strong>
<span>households in the source dataset</span>
</div>
<div class="fact">
<strong>~2.6 million</strong>
<span>transaction line items</span>
</div>
<div class="fact">
<strong>28 days</strong>
<span>per lifecycle period</span>
</div>
</aside>
</div>
<div class="route" aria-label="Reading guide">
<a href="#patterns">
<span>01</span> Notice the shopping changes ↓</a>
<a href="#risk">
<span>02</span> Understand the risk ↓</a>
<a href="#next">
<span>03</span> Choose a test ↓</a>
</div>
<section id="patterns">
<div class="section-head">
<div>
<p class="eyebrow">01 / Observe</p>
<h2>The basket changes.<br>What does that tell us?</h2>
</div>
<p>Some categories lose a larger share of household spending around the transition to dormancy. Comparing these changes with consistently active shoppers gives us a signal to investigate.</p>
</div>
<div class="two-col">
<figure class="figure">
<p class="figure-label">Where spending share falls most</p>
<p class="note">Difference in share change: dormant group minus active comparison group.</p>{category_chart(categories)}<figcaption>pp = percentage points. A −3 pp difference means the category’s share of spending fell 3 points more in the dormant group. This is different from a 3% reduction in dollars.</figcaption>
</figure>
<div class="reading">
<p class="label">How to read this</p>
<h3>A clue about behavior,<br>not a reason for leaving.</h3>
<p>The bars compare category shares across earlier and later three-period windows around first dormancy. The later window includes dormancy itself, so this analysis does not establish an eight-week advance warning.</p>
<div class="limit">
<p class="label">What we cannot conclude</p>
<p>We see purchases at one chain. We cannot tell whether a shopper changed stores, changed habits, or stopped needing a product. These patterns need prospective testing before becoming alerts.</p>
</div>
</div>
</div>
</section>
<section id="risk">
<div class="section-head">
<div>
<p class="eyebrow">02 / Predict</p>
<h2>Focus limited attention<br>where risk is concentrated.</h2>
</div>
<p>A churn model estimates which households will make no shopping trips in the next two periods (56 days). Purchase history is available before that window; future outcomes are used to evaluate the predictions.</p>
</div>
<div class="two-col">
<figure class="figure">
<p class="figure-label">If we contact the highest-risk 20 in every 100…</p>
<div class="risk-figure">
<svg viewBox="0 0 288 288" role="img" aria-label="100 households: 20 dark green circles represent the highest-risk group; 80 pale circles represent everyone else.">{dots}</svg>
<div>
<span class="big">{capture:.0%}</span>
<p>of observed departures were captured in that group during the test periods.</p>
</div>
</div>
<figcaption>Dark green = highest-risk 20%. This measures whom the model identifies. It does not measure how many people an offer would persuade to stay.</figcaption>
</figure>
<div class="reading">
<p class="label">The practical finding</p>
<h3>Risk is concentrated.<br>Response is still unknown.</h3>
<p>Ranking households gives a retailer a more focused outreach list than contacting people at random. The next question is whether that outreach actually changes behavior.</p>
<div class="limit">
<p class="label">A useful baseline</p>
<p>LightGBM’s PR-AUC is {selected.pr_auc:.3f}, compared with {lr.pr_auc:.3f} for logistic regression. Their confidence intervals overlap. The more complex model has not established a clear advantage on that metric.</p>
</div>
</div>
</div>
</section>
<section id="next">
<div class="section-head">
<div>
<p class="eyebrow">03 / Test</p>
<h2>A prediction is a starting point.<br>An experiment is the next step.</h2>
</div>
<p>Historical campaigns were not randomly assigned. Uplift models explore who might respond differently to an offer, but selection bias can remain. A randomized holdout is needed before claiming incremental profit or suppressing contact.</p>
</div>
<div class="simulator" aria-label="Illustrative retention scenario">
<div>
<p class="eyebrow" style="color:#d3ddcf">Explore an assumption</p>
<h3>What if an offer saved some of the predicted departures?</h3>
<p class="note">Scaled to {financial["scale_households"]:,} households, contacting the highest-risk 20%. Change the assumed save rate to see how the estimate changes.</p>
<label for="save-rate">Assumed save rate: <output id="save-label" for="save-rate">10%</output>
</label>
<input id="save-rate" type="range" min="0" max="20" step="1" value="10">
<div class="range-labels">
<span>0% saved</span>
<span>20% saved</span>
</div>
</div>
<div class="sim-results" aria-live="polite" aria-atomic="true">
<p>Illustrative net margin over 26 weeks</p>
<output class="big" id="net-margin">${default_net:,.0f}</output>
<p>
<output id="saved-households">{default_saved:,.0f}</output> households assumed retained · ${financial["total_campaign_cost_dollars"]:,.0f} contact cost</p>
<p class="note">Uses a {financial["base_churn_rate"]:.2%} baseline departure rate, {capture:.1%} capture, ${clv["mean_26w_clv_margin_dollars"]:,.2f} estimated mean 26-week margin, and ${financial["contact_cost_per_household_dollars"]:.2f} per contact. It assumes retained households earn that average margin. These are scenario inputs, not measured campaign results.</p>
</div>
</div>
<noscript>
<p>The scenario requires JavaScript to update. The initial values above use a 10% assumed save rate.</p>
</noscript>
<div class="steps">
<div>
<p class="label">Design</p>
<h3>Randomize households.</h3>
<p>Choose eligibility in advance, then assign households to an offer or a holdout. Keep each household in one arm and record delivery costs.</p>
</div>
<div>
<p class="label">Measure</p>
<h3>Compare net outcomes.</h3>
<p>Track spend or margin over a fixed follow-up window, together with offer cost. Choose the detectable effect and sample size before running the test.</p>
</div>
<div>
<p class="label">Decide</p>
<h3>Roll out on evidence.</h3>
<p>Use the treatment-versus-holdout difference and its uncertainty. Check discount dependence and subgroup outcomes before expanding the policy.</p>
</div>
</div>
</section>
<section id="methods">
<p class="note">Study planning: {experiment["total_sample_size"]:,} households needed; {experiment["available_active_households"]:,} eligible households available. {study_note}</p>
<p class="eyebrow">For a closer look</p>
<h2>The reasoning behind the story.</h2>
<p class="intro">Definitions, comparisons, and reproducible code are available without interrupting the main findings.</p>
<details>
<summary>Model evaluation and the simple baseline</summary>{comparison_table(comparison)}<p class="note">PR-AUC summarizes how well the model identifies rare departures across contact thresholds; higher is better. Intervals use household-clustered bootstrap resampling. Capture measures recall at a fixed outreach budget. Repeated household-period rows are not independent households.</p>
<p>Calibration, temporal splits, error analysis, and limitations are documented in the <a href="{REPOSITORY}/blob/main/reports/technical_report.md">technical report</a>.</p>
</details>
<details>
<summary>What the lifecycle model adds</summary>
<p>Backtest distribution error (lower is better): Markov {lifecycle["backtest_avg_markov_tvd"]:.4f}; unchanged-distribution baseline {lifecycle["backtest_avg_persistence_tvd"]:.4f}. Assumption checks also find evidence against a time-stable, first-order chain. Use the groups to describe activity; treat long-range forecasts cautiously.</p>
<p>The lifecycle model groups shopping activity into New, Champion, Regular, At-Risk, Dormant, and Lapsed states. A transition matrix describes how frequently households move between states in 28 days.</p>
<p>Long-range projections assume those transition patterns remain stable. Treating Lapsed as absorbing is a mathematical simplification: shoppers can return in real life. State-value estimates and 26-week household margin predictions use different horizons and should not be compared directly.</p>
</details>
<details>
<summary>Data, reproducibility, and limits</summary>
<p>The source contains transaction line items, households, products, campaign exposure, and coupon records. SQL/dbt models create baskets and household-period states; Python produces evaluations and saved outputs. This page is generated from those outputs.</p>
<ul>
<li>Single-chain historical data cannot establish why someone leaves.</li>
<li>Campaign-household observations can repeat the same household. Unique households and campaign observations must be reported separately.</li>
<li>Estimated customer value and offer response introduce uncertainty into financial scenarios.</li>
<li>Small demographic groups limit subgroup comparisons; omitting demographics does not establish fairness.</li>
</ul>
<p>
<a href="{REPOSITORY}">Read the code and setup instructions ↗</a> · <a href="{REPOSITORY}/blob/main/docs/data_dictionary.md">Data dictionary ↗</a>
</p>
</details>
<details>
<summary>Plain-language glossary</summary>
<dl class="glossary">
<div>
<dt>Churn / departure</dt>
<dd>No shopping trips during the specified future window. It does not prove a permanent loss.</dd>
<dt>Customer value (CLV)</dt>
<dd>Estimated future margin from a household, over a stated time horizon.</dd>
</div>
<div>
<dt>Uplift</dt>
<dd>The difference an intervention might make, compared with not intervening.</dd>
<dt>Calibration</dt>
<dd>Whether predicted probabilities agree with observed frequencies.</dd>
</div>
</dl>
</details>
<details>
<summary>Explore the scored household list</summary>
<p>The list contains public-dataset household identifiers, estimated departure probabilities, estimated margin, and model reasons. Reasons describe associations used by the model, not causes.</p>
<p>
<a download="retention-radar-households.csv" href="{roster_download(scores)}">Download scored households (.csv)</a>
</p>
</details>
</section>
</main>
<footer>
<p>Built and analyzed by <a href="https://sanjaymalla.com">Sanjay Bahadur</a>.</p>
<p>
<a href="{REPOSITORY}">Source & methodology ↗</a> · Public data, exploratory analysis.</p>
</footer>
</div>
<script>
const assumptions = {json.dumps(assumptions)};
const slider = document.getElementById('save-rate');
function updateScenario() {{
  const rate = Number(slider.value) / 100;
  const saved = assumptions.population * assumptions.churnRate * assumptions.capture * rate;
  const cost = assumptions.population * assumptions.contactShare * assumptions.cost;
  document.getElementById('save-label').textContent = slider.value + '%';
  document.getElementById('saved-households').textContent = Math.round(saved).toLocaleString('en-US');
  const net = saved * assumptions.clv - cost;
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
    """Write identical reports for GitHub Pages and the local reports directory."""
    root = get_project_root()
    config = load_project_config()
    out_dir = root / config.paths.data_outputs
    reports_dir = root / config.paths.reports_dir
    path = build_interactive_html(out_dir, reports_dir, root / "docs" / "index.html")
    (reports_dir / "interactive_report.html").write_text(
        path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    print(f"Built {path}")


if __name__ == "__main__":
    main()
