from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ACTION_ORDER = ["continue", "compress", "redirect", "mode_switch", "stop"]
SUCCESS_KEYS = ["acc", "accuracy", "success", "won"]
EM_KEYS = ["em", "exact_match"]
F1_KEYS = ["f1"]


@dataclass
class Trajectory:
    task_id: str
    raw: Mapping[str, Any]
    metrics: Mapping[str, Any]
    acc: Optional[float]
    em: Optional[float]
    f1: Optional[float]
    total_input_tokens: Optional[float]
    total_output_tokens: Optional[float]
    total_steps: Optional[float]
    pred_answer: str
    ground_truth_candidates: List[str]
    intervention_log: List[Mapping[str, Any]]
    steps: List[Mapping[str, Any]]
    controller_forced_answer: str


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        if math.isnan(float(x)):
            return None
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _normalize_action(action: Any) -> str:
    s = str(action or "").strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if s in {"modeswitch", "mode", "mode_switch"}:
        return "mode_switch"
    if s in {"continue", "compress", "redirect", "stop"}:
        return s
    return s or "unknown"


def _truncate(s: str, n: int = 160) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."


def _first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    lower_map = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if k in d:
            return d[k]
        lk = k.lower()
        if lk in lower_map:
            return lower_map[lk]
    return None


def _extract_records(obj: Any) -> List[Mapping[str, Any]]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, Mapping)]
    if isinstance(obj, Mapping):
        # likely wrappers
        for key in ["data", "results", "items", "samples", "predictions", "records"]:
            v = obj.get(key)
            if isinstance(v, list) and v and isinstance(v[0], Mapping):
                return [x for x in v if isinstance(x, Mapping)]
        # fallback: if dict itself looks like one sample
        if any(k in obj for k in ["task_id", "Metrics", "metrics", "intervention_log", "steps"]):
            return [obj]
    raise ValueError("Cannot locate trajectory list in JSON. Expected list or dict[data/results/items/...].")


def _extract_gt_candidates(gt: Any) -> List[str]:
    out: List[str] = []
    if gt is None:
        return out
    if isinstance(gt, str):
        s = gt.strip()
        if s:
            out.append(s)
        return out
    if isinstance(gt, (int, float, bool)):
        out.append(str(gt))
        return out
    if isinstance(gt, list):
        for x in gt:
            out.extend(_extract_gt_candidates(x))
        return list(dict.fromkeys(out))
    if isinstance(gt, Mapping):
        for key in ["target", "targets", "answer", "answers", "ground_truth", "gold", "reference"]:
            if key in gt:
                out.extend(_extract_gt_candidates(gt[key]))
        if not out:
            for v in gt.values():
                out.extend(_extract_gt_candidates(v))
        return list(dict.fromkeys(out))
    return out


def _contains_match(pred: str, gts: Sequence[str]) -> bool:
    p = (pred or "").strip().lower()
    if not p:
        return False
    for gt in gts:
        g = (gt or "").strip().lower()
        if not g:
            continue
        if g in p or p in g:
            return True
    return False


def _extract_metrics(rec: Mapping[str, Any]) -> Mapping[str, Any]:
    m = rec.get("Metrics")
    if isinstance(m, Mapping):
        return m
    m = rec.get("metrics")
    if isinstance(m, Mapping):
        return m
    return {}


def _metric_from_sources(rec: Mapping[str, Any], metrics: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    v = _first_present(metrics, keys)
    fv = _to_float(v)
    if fv is not None:
        return fv
    v = _first_present(rec, keys)
    fv = _to_float(v)
    if fv is not None:
        return fv
    return None


def _extract_task_id(rec: Mapping[str, Any], idx: int) -> str:
    for k in ["task_id", "id", "qid", "question_id", "uid"]:
        if k in rec:
            return str(rec[k])
    return f"idx_{idx}"


def _extract_pred_answer(rec: Mapping[str, Any]) -> str:
    for k in ["controller_forced_answer", "agent_answer", "Pred_Answer", "pred_answer", "prediction", "final_answer", "answer"]:
        if k in rec and rec[k] is not None:
            return str(rec[k])
    return ""


def _extract_number(rec: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for k in keys:
        if k in rec:
            fv = _to_float(rec.get(k))
            if fv is not None:
                return fv
    return None


def _extract_steps(rec: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    v = rec.get("steps")
    if isinstance(v, list):
        return [x for x in v if isinstance(x, Mapping)]
    return []


def _extract_intervention_log(rec: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    v = rec.get("intervention_log")
    if isinstance(v, list):
        return [x for x in v if isinstance(x, Mapping)]
    return []


def load_trajectories(path: str) -> Dict[str, Trajectory]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    recs = _extract_records(obj)
    out: Dict[str, Trajectory] = {}
    for i, rec in enumerate(recs):
        task_id = _extract_task_id(rec, i)
        metrics = _extract_metrics(rec)

        acc = _metric_from_sources(rec, metrics, SUCCESS_KEYS)
        em = _metric_from_sources(rec, metrics, EM_KEYS)
        f1 = _metric_from_sources(rec, metrics, F1_KEYS)

        total_input = _extract_number(rec, ["total_input_tokens", "input_tokens", "prompt_tokens"])
        total_output = _extract_number(rec, ["total_output_tokens", "output_tokens", "completion_tokens"])
        total_steps = _extract_number(rec, ["total_steps", "num_steps", "steps_count", "turn_count"])

        if total_steps is None:
            st = _extract_steps(rec)
            if st:
                total_steps = float(len(st))

        gt = rec.get("ground_truth", rec.get("Ground_Truth", rec.get("answer", rec.get("gold"))))
        gts = _extract_gt_candidates(gt)
        pred = _extract_pred_answer(rec)

        traj = Trajectory(
            task_id=task_id,
            raw=rec,
            metrics=metrics,
            acc=acc,
            em=em,
            f1=f1,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_steps=total_steps,
            pred_answer=pred,
            ground_truth_candidates=gts,
            intervention_log=_extract_intervention_log(rec),
            steps=_extract_steps(rec),
            controller_forced_answer=str(rec.get("controller_forced_answer") or ""),
        )
        out[task_id] = traj
    return out


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x * 100:.2f}%"


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "NA"
    return f"{x:.2f}"


def _delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return new - old


def summarize_basic(trajs: Sequence[Trajectory]) -> Dict[str, Optional[float]]:
    return {
        "acc": _mean([t.acc for t in trajs if t.acc is not None]),
        "em": _mean([t.em for t in trajs if t.em is not None]),
        "f1": _mean([t.f1 for t in trajs if t.f1 is not None]),
        "avg_input_tokens": _mean([t.total_input_tokens for t in trajs if t.total_input_tokens is not None]),
        "avg_output_tokens": _mean([t.total_output_tokens for t in trajs if t.total_output_tokens is not None]),
        "avg_steps": _mean([t.total_steps for t in trajs if t.total_steps is not None]),
    }


def intervention_distribution(controller_trajs: Sequence[Trajectory]) -> Tuple[Dict[str, int], Dict[str, float]]:
    counts = Counter({k: 0 for k in ACTION_ORDER})
    total_events = 0

    for t in controller_trajs:
        non_continue = 0
        for item in t.intervention_log:
            act = _normalize_action(item.get("action"))
            if act in counts:
                counts[act] += 1
            else:
                counts[act] += 1
            total_events += 1
            if act != "continue":
                non_continue += 1

        if t.total_steps is not None:
            cont = max(int(t.total_steps) - non_continue, 0)
            counts["continue"] += cont
            total_events += cont

    pct: Dict[str, float] = {}
    for k, v in counts.items():
        pct[k] = (v / total_events) if total_events > 0 else 0.0
    return dict(counts), pct


def any_non_continue_intervention(t: Trajectory) -> bool:
    for item in t.intervention_log:
        if _normalize_action(item.get("action")) != "continue":
            return True
    return False


def action_post_tokens(controller_trajs: Sequence[Trajectory]) -> Dict[str, Optional[float]]:
    by_action: Dict[str, List[float]] = defaultdict(list)
    for t in controller_trajs:
        if not t.steps:
            continue

        # map step id -> remaining tokens after that step
        step_ids: List[int] = []
        step_total: List[float] = []
        for idx, s in enumerate(t.steps):
            sid = _to_float(s.get("step_id"))
            sid_int = int(sid) if sid is not None else idx
            step_ids.append(sid_int)
            tin = _to_float(s.get("tokens_input")) or 0.0
            tout = _to_float(s.get("tokens_output")) or 0.0
            step_total.append(tin + tout)

        suffix_after: Dict[int, float] = {}
        running = 0.0
        for i in range(len(step_ids) - 1, -1, -1):
            suffix_after[step_ids[i]] = running
            running += step_total[i]

        for it in t.intervention_log:
            act = _normalize_action(it.get("action"))
            s = _to_float(it.get("step"))
            if s is None:
                s = _to_float(it.get("step_id"))
            if s is None:
                continue
            sid = int(s)
            if sid in suffix_after:
                by_action[act].append(suffix_after[sid])

    return {k: _mean(v) for k, v in by_action.items()}


def _success(tr: Trajectory) -> Optional[float]:
    return tr.acc


def build_case_summary(base: Trajectory, ctrl: Trajectory) -> str:
    actions = []
    for it in ctrl.intervention_log:
        step = it.get("step", it.get("step_id", "?"))
        act = _normalize_action(it.get("action"))
        actions.append(f"{step}:{act}")
    actions_s = ", ".join(actions[:8]) if actions else "none"

    b_tok = (base.total_input_tokens or 0.0) + (base.total_output_tokens or 0.0)
    c_tok = (ctrl.total_input_tokens or 0.0) + (ctrl.total_output_tokens or 0.0)
    gt = ctrl.ground_truth_candidates[0] if ctrl.ground_truth_candidates else ""

    return (
        f"actions=[{actions_s}] | "
        f"tokens baseline={b_tok:.0f}, controller={c_tok:.0f} | "
        f"pred={_truncate(ctrl.pred_answer, 90)} | gt={_truncate(gt, 90)}"
    )


def pick_cases(pairs: Sequence[Tuple[Trajectory, Trajectory]]) -> Dict[str, List[Tuple[Trajectory, Trajectory]]]:
    def is_true(x: Optional[float]) -> bool:
        return (x is not None) and (x >= 0.5)

    cat1 = [(b, c) for b, c in pairs if not is_true(_success(b)) and is_true(_success(c))]

    cat2 = []
    for b, c in pairs:
        if not is_true(_success(b)):
            continue
        has_stop = any(_normalize_action(it.get("action")) == "stop" for it in c.intervention_log)
        if not has_stop:
            continue
        forced = (c.controller_forced_answer or "").strip()
        if forced and _contains_match(forced, c.ground_truth_candidates):
            cat2.append((b, c))

    cat3 = []
    for b, c in pairs:
        if is_true(_success(c)):
            continue
        if any_non_continue_intervention(c):
            cat3.append((b, c))

    # rank by token saving descending for cat2, improvement for cat1, intervention count for cat3
    def token_saving(pair: Tuple[Trajectory, Trajectory]) -> float:
        b, c = pair
        bt = (b.total_input_tokens or 0.0) + (b.total_output_tokens or 0.0)
        ct = (c.total_input_tokens or 0.0) + (c.total_output_tokens or 0.0)
        return bt - ct

    cat1 = sorted(cat1, key=token_saving, reverse=True)[:5]
    cat2 = sorted(cat2, key=token_saving, reverse=True)[:5]
    cat3 = sorted(cat3, key=lambda p: len(p[1].intervention_log), reverse=True)[:5]

    return {
        "baseline_fail_controller_success": cat1,
        "baseline_success_controller_stop_correct": cat2,
        "controller_intervened_but_failed": cat3,
    }


def write_csv(
    path: str,
    overall_rows: List[Dict[str, str]],
    dist_rows: List[Dict[str, str]],
    effect_rows: List[Dict[str, str]],
    case_rows: List[Dict[str, str]],
) -> None:
    fields = [
        "section",
        "metric",
        "baseline",
        "controller",
        "delta",
        "action",
        "count",
        "percent",
        "group",
        "value",
        "task_id",
        "case_category",
        "summary",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in overall_rows + dist_rows + effect_rows + case_rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline vs controller-enabled online intervention outputs.")
    parser.add_argument("--baseline_path", required=True, help="Path to baseline output JSON.")
    parser.add_argument("--controller_path", required=True, help="Path to controller-enabled output JSON.")
    parser.add_argument(
        "--output_md",
        default="/data/wanghy/agent_traj/deliberation_controller/eval/online_intervention_compare.md",
        help="Path to markdown report.",
    )
    parser.add_argument(
        "--output_csv",
        default="/data/wanghy/agent_traj/deliberation_controller/eval/online_intervention_compare.csv",
        help="Path to merged CSV report.",
    )
    args = parser.parse_args()

    baseline = load_trajectories(args.baseline_path)
    controller = load_trajectories(args.controller_path)

    common_ids = sorted(set(baseline.keys()) & set(controller.keys()))
    if not common_ids:
        raise ValueError("No overlapping task_id between baseline and controller files.")

    base_trajs = [baseline[i] for i in common_ids]
    ctrl_trajs = [controller[i] for i in common_ids]
    pairs = [(baseline[i], controller[i]) for i in common_ids]

    base_sum = summarize_basic(base_trajs)
    ctrl_sum = summarize_basic(ctrl_trajs)

    overall_rows: List[Dict[str, str]] = []
    overall_metrics = [
        ("success_rate_acc", "acc", True),
        ("success_rate_em", "em", True),
        ("success_rate_f1", "f1", True),
        ("avg_total_input_tokens", "avg_input_tokens", False),
        ("avg_total_output_tokens", "avg_output_tokens", False),
        ("avg_total_steps", "avg_steps", False),
    ]

    md_lines: List[str] = []
    md_lines.append("# Online Intervention Comparison Report")
    md_lines.append("")
    md_lines.append(f"- Baseline: `{args.baseline_path}`")
    md_lines.append(f"- Controller: `{args.controller_path}`")
    md_lines.append(f"- Overlap trajectories: **{len(common_ids)}**")
    md_lines.append("")

    md_lines.append("## 1) Overall Metrics")
    md_lines.append("")
    md_lines.append("| Metric | Baseline | Controller | Delta (Ctrl-Base) |")
    md_lines.append("|---|---:|---:|---:|")
    for label, key, as_pct in overall_metrics:
        b = base_sum.get(key)
        c = ctrl_sum.get(key)
        d = _delta(c, b)
        b_s = _fmt_pct(b) if as_pct else _fmt_num(b)
        c_s = _fmt_pct(c) if as_pct else _fmt_num(c)
        d_s = _fmt_pct(d) if as_pct else _fmt_num(d)
        md_lines.append(f"| {label} | {b_s} | {c_s} | {d_s} |")
        overall_rows.append(
            {
                "section": "overall",
                "metric": label,
                "baseline": b_s,
                "controller": c_s,
                "delta": d_s,
            }
        )

    counts, pct = intervention_distribution(ctrl_trajs)
    dist_rows: List[Dict[str, str]] = []
    md_lines.append("")
    md_lines.append("## 2) Intervention Distribution")
    md_lines.append("")
    md_lines.append("| Action | Count | Percent |")
    md_lines.append("|---|---:|---:|")
    for act in ACTION_ORDER + [k for k in counts.keys() if k not in ACTION_ORDER]:
        if act not in counts:
            continue
        c = counts[act]
        p = pct.get(act, 0.0)
        md_lines.append(f"| {act} | {c} | {p * 100:.2f}% |")
        dist_rows.append(
            {
                "section": "distribution",
                "action": act,
                "count": str(c),
                "percent": f"{p * 100:.2f}%",
            }
        )

    # Effect analysis
    intervened = [t for t in ctrl_trajs if any_non_continue_intervention(t)]
    not_intervened = [t for t in ctrl_trajs if not any_non_continue_intervention(t)]
    inter_sr = _mean([t.acc for t in intervened if t.acc is not None])
    nointer_sr = _mean([t.acc for t in not_intervened if t.acc is not None])

    stop_records = []
    for t in ctrl_trajs:
        has_stop = any(_normalize_action(it.get("action")) == "stop" for it in t.intervention_log)
        if has_stop and (t.controller_forced_answer or "").strip():
            stop_records.append(t)
    stop_acc = _mean([
        1.0 if _contains_match(t.controller_forced_answer, t.ground_truth_candidates) else 0.0
        for t in stop_records
    ])

    post_tokens = action_post_tokens(ctrl_trajs)

    effect_rows: List[Dict[str, str]] = []
    md_lines.append("")
    md_lines.append("## 3) Intervention Effect Analysis")
    md_lines.append("")
    md_lines.append(f"- Success rate (intervened trajectories): **{_fmt_pct(inter_sr)}**")
    md_lines.append(f"- Success rate (non-intervened trajectories): **{_fmt_pct(nointer_sr)}**")
    md_lines.append(f"- Stop answer accuracy (controller_forced_answer vs ground_truth): **{_fmt_pct(stop_acc)}**")

    effect_rows.append({"section": "effect", "group": "intervened_success_rate", "value": _fmt_pct(inter_sr)})
    effect_rows.append({"section": "effect", "group": "non_intervened_success_rate", "value": _fmt_pct(nointer_sr)})
    effect_rows.append({"section": "effect", "group": "stop_answer_accuracy", "value": _fmt_pct(stop_acc)})

    md_lines.append("")
    md_lines.append("### Avg Subsequent Tokens After Each Intervention")
    md_lines.append("")
    md_lines.append("| Action | Avg Subsequent Tokens |")
    md_lines.append("|---|---:|")
    for act in ACTION_ORDER:
        v = post_tokens.get(act)
        if v is None:
            continue
        md_lines.append(f"| {act} | {_fmt_num(v)} |")
        effect_rows.append({"section": "effect", "group": f"post_tokens_{act}", "value": _fmt_num(v)})

    # Case studies
    cases = pick_cases(pairs)
    case_rows: List[Dict[str, str]] = []

    md_lines.append("")
    md_lines.append("## 4) Case Study Candidates")
    md_lines.append("")

    title_map = {
        "baseline_fail_controller_success": "A. Baseline failed + Controller succeeded (top 5)",
        "baseline_success_controller_stop_correct": "B. Baseline success + Controller early-stop still correct (top 5)",
        "controller_intervened_but_failed": "C. Controller intervened but still failed (top 5)",
    }

    for cat_key, title in title_map.items():
        md_lines.append(f"### {title}")
        lst = cases.get(cat_key, [])
        if not lst:
            md_lines.append("- (none)")
            md_lines.append("")
            continue
        for b, c in lst:
            summary = build_case_summary(b, c)
            md_lines.append(f"- task_id={c.task_id}: {summary}")
            case_rows.append(
                {
                    "section": "case",
                    "task_id": c.task_id,
                    "case_category": cat_key,
                    "summary": summary,
                }
            )
        md_lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    write_csv(args.output_csv, overall_rows, dist_rows, effect_rows, case_rows)

    print(f"Saved markdown report: {args.output_md}")
    print(f"Saved csv report: {args.output_csv}")


if __name__ == "__main__":
    main()
