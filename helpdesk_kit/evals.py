"""Run the assistant over the test questions and score the results."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import pandas as pd

from . import DATA_DIR
from .llm import DEFAULT_MODEL, chat, parse_json

EVAL_PATH = DATA_DIR / "eval_questions.jsonl"
EVAL_30_PATH = DATA_DIR / "eval_set_30.jsonl"  # The finished 3.5 set, for anyone who missed it.
SECURITY_PATH = DATA_DIR / "security_cases.jsonl"  # Starter attacks, for anyone who missed 3.6.
JUDGE_PATH = DATA_DIR / "judge_cases.jsonl"


def load_eval_set(path=EVAL_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def save_eval_set(cases: list[dict], path) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case) + "\n")


def merge_cases(base: list[dict], extra: list[dict]) -> list[dict]:
    """Append cases, skipping repeated questions and giving each new case a fresh id."""
    merged = list(base)
    seen = {case["question"].strip().lower() for case in base}
    for case in extra:
        key = case["question"].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append({**case, "id": f"q{len(merged) + 1:02d}"})
    return merged


def _run_case(answer_fn: Callable[[str], Any], case: dict) -> dict:
    result = {
        "id": case["id"],
        "question": case["question"],
        "type": case["type"],
        "expected_category": case["category"],
        "expected_needs_human": case["needs_human"],
        "expected_sources": case["expected_sources"],
        "pass_criteria": case.get("pass_criteria"),
        "must_not_contain": case.get("must_not_contain", []),
        "error": None,
    }
    start = time.perf_counter()
    try:
        output = answer_fn(case["question"])
        result.update(output.model_dump() if hasattr(output, "model_dump") else output)
    except Exception as error:  # One bad case should not stop the whole run.
        result["error"] = f"{type(error).__name__}: {error}"
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result


def run_eval(answer_fn: Callable[[str], Any], cases: list[dict], max_workers: int = 4) -> list[dict]:
    """Call answer_fn(question) for every case, in parallel, keeping the order.

    answer_fn returns a dict or Pydantic model with at least `answer`,
    `category` and `needs_human`. `sources` is scored when present.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda case: _run_case(answer_fn, case), cases))


def score(results: list[dict]) -> list[dict]:
    """Add automatic pass/fail columns. Answer quality still needs a human or a judge."""
    scored = []
    for result in results:
        sources = result.get("sources")
        scored.append(
            {
                **result,
                "category_ok": result.get("category") == result["expected_category"],
                "needs_human_ok": result.get("needs_human") == result["expected_needs_human"],
                "sources_ok": (
                    None if sources is None else set(result["expected_sources"]) <= set(sources)
                ),
            }
        )
    return scored


def summary_table(scored: list[dict]) -> pd.DataFrame:
    """Pass rates by question type, with a total row."""
    frame = pd.DataFrame(scored)
    checks = [
        column
        for column in ("category_ok", "needs_human_ok", "sources_ok")
        if frame[column].notna().any()
    ]
    # astype(float) because sources_ok may be stored as objects.
    rates = frame[checks].astype(float)
    table = rates.groupby(frame["type"]).mean()
    table.loc["all"] = rates.mean()
    table.insert(0, "cases", frame.groupby("type").size())
    table.loc["all", "cases"] = len(frame)
    table["cases"] = table["cases"].astype(int)
    return table.round(2)


def leaked(result: dict) -> list[str]:
    """Which forbidden strings appear anywhere in the reply."""
    reply = json.dumps({key: result.get(key) for key in ("answer", "sources")}, default=str)
    return [text for text in result.get("must_not_contain", []) if text.lower() in reply.lower()]


def case_passed(result: dict) -> bool:
    """The fast check the release gate uses. Security cases only need to not leak."""
    if result["type"] == "security":
        return not result["error"] and not leaked(result)
    checked = score([result])[0]
    return bool(
        checked["category_ok"] and checked["needs_human_ok"] and checked["sources_ok"] is not False
    )


JUDGE_MODEL = DEFAULT_MODEL


def _judge(instructions: str) -> dict:
    reply = chat(
        instructions + '\n\nReturn only JSON: {"verdict": "pass" or "fail", "reason": "one sentence"}',
        model=JUDGE_MODEL,
        max_tokens=200,
    )
    return parse_json(reply.text)


def judge_faithful(question: str, context: str, answer: str) -> dict:
    """LLM-as-judge: is every claim in the answer supported by the context?"""
    return _judge(
        "You are checking an IT help-desk answer. Pass it only if every factual claim in it is "
        "supported by the context. Saying it doesn't know also passes. Being helpful or "
        "well written earns nothing.\n\n"
        f"<context>\n{context}\n</context>\n<question>{question}</question>\n<answer>{answer}</answer>"
    )


def judge_pass_criteria(result: dict) -> dict:
    """LLM-as-judge: does the answer meet this case's written pass criteria?"""
    if result.get("error") or not result.get("answer"):
        return {"verdict": "fail", "reason": "No answer was produced."}
    return _judge(
        "Decide whether this help-desk answer meets the pass criteria. Judge only the criteria.\n\n"
        f"<question>{result['question']}</question>\n"
        f"<pass_criteria>{result['pass_criteria']}</pass_criteria>\n"
        f"<answer>{result['answer']}</answer>"
    )


def add_quality(results: list[dict], max_workers: int = 4) -> list[dict]:
    """Add a judge verdict per case, as `quality_ok` and `quality_reason`."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        verdicts = list(pool.map(judge_pass_criteria, results))
    return [
        {**result, "quality_ok": verdict["verdict"] == "pass", "quality_reason": verdict["reason"]}
        for result, verdict in zip(results, verdicts)
    ]


def baseline_metrics(results: list[dict]) -> dict:
    """The numbers 3.5 saves and 4.5 compares against."""
    costs = [result.get("cost") for result in results]
    count = len(results)
    return {
        "cases": count,
        "quality": round(sum(result["quality_ok"] for result in results) / count, 3),
        "avg_tokens": round(sum(result.get("tokens", 0) for result in results) / count),
        "cost_per_question": None if None in costs else round(sum(costs) / count, 5),
        "avg_steps": round(sum(result.get("steps", 1) for result in results) / count, 2),
        "avg_seconds": round(sum(result["seconds"] for result in results) / count, 2),
    }


def results_table(results: list[dict]) -> pd.DataFrame:
    """One row per question: the answer, and Claude's labels next to the expected ones."""
    columns = [
        "id", "type", "question", "answer", "category", "expected_category",
        "needs_human", "expected_needs_human", "confidence", "error",
    ]
    return pd.DataFrame(results).reindex(columns=columns)


# Chart colours: one per run, in this fixed order. Ink and gridlines from the same palette.
RUN_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SOFT_INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CHECKS = {"category_ok": "Category matches our label", "needs_human_ok": "Needs-human decision matches"}


def plot_pass_rates(runs: dict[str, list[dict]], title: str = "") -> None:
    """Bar chart of the pass rates by question type, one bar per run (for example before and after).

    `runs` maps a name to scored results, e.g. {"before": score(before), "with the policy": score(after)}.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    if len(runs) > len(RUN_COLOURS):
        raise ValueError(f"At most {len(RUN_COLOURS)} runs per chart.")
    frames = {name: pd.DataFrame(scored) for name, scored in runs.items()}
    first = next(iter(frames.values()))
    types = sorted(first["type"].unique()) + ["all"]
    counts = first["type"].value_counts()
    labels = [
        f"{kind.replace('_', ' ')}\n({counts[kind] if kind != 'all' else len(first)})" for kind in types
    ]

    figure, axes = plt.subplots(1, len(CHECKS), figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    width = min(0.24, 0.8 / len(runs))  # Thin bars, even for a single run.
    label_size = 10 if len(runs) < 3 else 8.5
    for ax, (check, check_title) in zip(axes, CHECKS.items()):
        top = max(frame[check].mean() for frame in frames.values())  # Tallest "all" bar.
        for number, (name, frame) in enumerate(frames.items()):
            rates = frame.groupby("type")[check].mean()
            values = [rates.get(kind, 0) for kind in types[:-1]] + [frame[check].mean()]
            positions = [index + (number - (len(runs) - 1) / 2) * width for index in range(len(types))]
            ax.bar(positions, values, width=width, color=RUN_COLOURS[number], label=name,
                   edgecolor=SURFACE, linewidth=2)
            # Label only the "all" bars: the overall score of each run.
            # With three runs, give each label its own height above the tallest bar, so none overlap.
            height = values[-1] + 0.02 if len(runs) < 3 else top + 0.03 + 0.07 * number
            ax.text(positions[-1], height, f"{values[-1]:.0%}", ha="center", color=INK, fontsize=label_size)
        ax.set_title(check_title, loc="left", color=INK)
        ax.set_xticks(range(len(types)), labels, color=SOFT_INK)
        ax.set_ylim(0, 1.12 if len(runs) < 3 else 1.3)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])  # The room above 100% is only for labels.
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.axvline(len(types) - 1.5, color=GRID, linewidth=1)  # Separates "all" from the types.

    if len(runs) > 1:
        axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=len(runs),
                       labelcolor=SOFT_INK)
    if title:
        figure.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    figure.tight_layout()
    plt.show()
