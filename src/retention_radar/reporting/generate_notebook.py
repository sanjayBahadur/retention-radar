"""Generate a concise walkthrough from the same artifacts as the dashboard."""

import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

from retention_radar.config import get_project_root


def build_walkthrough_notebook(dest_path: Path) -> Path:
    root = get_project_root()
    out = root / "data" / "outputs"
    results = json.loads((out / "results.json").read_text())
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    cells = [
        nbf.v4.new_markdown_cell(
            "# Retention Radar: from shopping data to a testable decision\n\n"
            "**Question:** who should a retailer contact when its retention budget is limited?\n\n"
            "This walkthrough reads saved analysis outputs. A household can occur at several historical "
            "cutoffs; it appears once in each contact roster. Risk prediction, observed shopping changes, "
            "and modeled intervention effects are different kinds of evidence.\n\n"
            f"Artifact summary generated: {results['last_updated']}. See the README to rebuild."
        ),
        nbf.v4.new_code_cell(
            "import json\nfrom pathlib import Path\n\nimport pandas as pd\n\n"
            'outputs = Path("../data/outputs")\n'
            'results = json.loads((outputs / "results.json").read_text())'
        ),
    ]
    sections = [
        (
            "1. Predict departures",
            "churn_model_comparison.parquet",
            "A departure means no purchases in the next two 28-day periods. Models train on cutoffs "
            "6–13, tune/calibrate on 16–17, and evaluate on 20–23. Household-clustered intervals "
            "account for repeated households. Overlapping intervals alone do not establish a significant difference.",
        ),
        (
            "2. Inspect mistakes",
            "churn_error_analysis.parquet",
            "At each cutoff, select the highest-risk 20%. Review missed departures and false alarms. "
            "Group counts are descriptive; small groups should not drive confident policy changes.",
        ),
        (
            "3. Forecast customer value",
            "clv_comparison.parquet",
            "Train outcomes end on day 518; test outcomes cover days 519–700. Each window lasts 26 weeks. "
            "Errors are sales dollars, not margin. A recent-spending baseline is included. "
            "Final method selection on this test is exploratory and needs another future evaluation.",
        ),
        (
            "4. Investigate shopping changes",
            "early_warning_categories.parquet",
            "Effects are differences in category spend-share changes relative to active controls. "
            "Multiply the fractional effect by 100 for percentage points. These retrospective windows "
            "include dormancy and do not validate a prospective eight-week alert.",
        ),
        (
            "5. Explore offer response",
            "uplift_eval.parquet",
            "All three evaluation folds hold out entire households. Historical offers were not random. "
            "Uplift ranking remains observational, and learner selection uses these exploratory folds. "
            "Negative estimates do not establish that contacting a customer causes harm.",
        ),
    ]
    for title, filename, explanation in sections:
        cells.append(nbf.v4.new_markdown_cell(f"## {title}\n\n{explanation}"))
        table = pd.read_parquet(out / filename).head(15)
        cell = nbf.v4.new_code_cell(f'pd.read_parquet(outputs / "{filename}").head(15)')
        cell.outputs = [
            nbf.v4.new_output(
                "display_data",
                data={
                    "text/html": table.to_html(index=False),
                    "text/plain": table.to_string(index=False),
                },
            )
        ]
        cells.append(cell)
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. Choose the next experiment\n\n"
            "Freeze the eligible household pool, assign one offer or control per household, and preselect "
            "the outcome and analysis. Sample-size planning uses observed spending variance and an assumed "
            "minimum worthwhile effect. If the available pool is too small, revise the study before launch.\n\n"
            "Financial charts are scenarios using assumed response rates, margin and contact costs. "
            "No intervention experiment has been run and no campaign savings have been demonstrated."
        )
    )
    cells.append(nbf.v4.new_code_cell('results["uplift"]["ab_test"]'))
    notebook.cells = cells
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, dest_path)
    return dest_path


def build_eda_notebook(dest_path: Path) -> Path:
    """Show basic retail facts without copied numeric findings in notebook code."""
    summary = json.loads((get_project_root() / "data/outputs/eda_summary.json").read_text())
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Retention Radar: explore the shopping data\n\n"
            "A transaction row is a product line. Shopping trips aggregate those lines into baskets. "
            "The analysis keeps 25 complete 28-day periods (days 1–700). "
            "These descriptive summaries do not establish why customers changed their behavior."
        ),
        nbf.v4.new_code_cell(
            "import json\nfrom pathlib import Path\n\n"
            'summary = json.loads(Path("../data/outputs/eda_summary.json").read_text())'
        ),
    ]
    explanations = {
        "kpis": "Shopping activity and sales. Revenue describes this retailer, not household spending elsewhere.",
        "concentration": "How unevenly sales are distributed across households. Concentration alone does not identify which offer would help.",
        "churn": "No purchases in the next two periods among currently active households at cutoffs 6–13. The predictive model also includes non-lapsed inactive households, so its base rate differs.",
        "demographics": "Coverage of recorded demographic information. Missing information and small subgroup counts limit fairness conclusions.",
    }
    for name, explanation in explanations.items():
        notebook.cells.append(
            nbf.v4.new_markdown_cell(f"## {name.replace('_', ' ').title()}\n\n{explanation}")
        )
        cell = nbf.v4.new_code_cell(f'summary["{name}"]')
        cell.outputs = [
            nbf.v4.new_output(
                "display_data", data={"text/plain": json.dumps(summary[name], indent=2)}
            )
        ]
        notebook.cells.append(cell)
    nbf.write(notebook, dest_path)
    return dest_path


if __name__ == "__main__":
    path = get_project_root() / "notebooks" / "02_executive_walkthrough.ipynb"
    print(build_walkthrough_notebook(path))

    print(build_eda_notebook(get_project_root() / "notebooks" / "01_eda.ipynb"))
