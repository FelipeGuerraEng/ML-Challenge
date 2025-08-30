import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


def _load_jsonl(path: str) -> List[Dict]:
    """
    Load a JSONL file into a list of dicts.
    """
    rows: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    except Exception as e:
        raise RuntimeError(f"Failed to load metrics JSONL '{path}': {e}")


def _load_json(path: str) -> Dict:
    """
    Load a JSON file into a dict.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON '{path}': {e}")


def _parse_ts(s: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO-like timestamp string to datetime if possible.
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _window_by_days(metrics: List[Dict], days: int, ts_key: str) -> List[Dict]:
    """
    Return a sublist of metrics within the last 'days' from the latest timestamp.
    """
    if not metrics:
        return []
    times = [(_parse_ts(m.get(ts_key)), i) for i, m in enumerate(metrics)]
    valid = [t for t in times if t[0] is not None]
    if not valid:
        n = min(len(metrics), 20)
        return metrics[-n:]
    latest = max(t[0] for t in valid)
    start = latest - timedelta(days=days)
    keep_idx = [i for t, i in valid if t >= start]
    if not keep_idx:
        return []
    lo = min(keep_idx)
    return metrics[lo:]


def _median(values: List[float]) -> float:
    """
    Return the median of a list of floats.
    """
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)


def _extract_series(metrics: List[Dict], key: str) -> List[float]:
    """
    Extract a numeric series from a list of dicts, skipping missing values.
    """
    out: List[float] = []
    for m in metrics:
        val = m.get(key, None)
        if val is None:
            continue
        try:
            out.append(float(val))
        except Exception:
            continue
    return out


def _consecutive_latency_breaches(metrics: List[Dict], lat_key: str, threshold: float, k: int = 2) -> bool:
    """
    Return True if the last k points have latency above the threshold.
    """
    series = _extract_series(metrics, lat_key)
    if len(series) < k:
        return False
    tail = series[-k:]
    return all(v > float(threshold) for v in tail)


def _percent_drop(current: float, baseline: float) -> float:
    """
    Return percentage drop from baseline to current. Positive means a drop.
    """
    if baseline <= 0.0:
        return 0.0
    return float(max(0.0, (baseline - current) / baseline * 100.0))


def _decide_status(
    roc_drop_pct: float,
    pr_drop_pct: float,
    latency_breached: bool,
    drift_overall: bool,
    roc_warn: float = 3.0,
    roc_critical: float = 6.0,
    pr_critical: float = 5.0,
) -> str:
    """
    Decide overall status based on rules.
    """
    if roc_drop_pct >= roc_critical:
        return "critical"
    if drift_overall and pr_drop_pct >= pr_critical:
        return "critical"
    if roc_drop_pct >= roc_warn or latency_breached:
        return "warn"
    return "healthy"


def _choose_actions(status: str, roc_drop_pct: float, latency_breached: bool, drift_overall: bool) -> List[str]:
    """
    Map status and signals to an action list.
    """
    actions: List[str] = []
    if status == "critical":
        if roc_drop_pct >= 6.0:
            actions.extend(["open_incident", "roll_back_model", "trigger_retraining"])
        elif drift_overall:
            actions.extend(["trigger_retraining", "open_incident"])
        actions.append("page_oncall=false")
        return actions
    if status == "warn":
        if latency_breached:
            actions.append("raise_thresholds")
        if roc_drop_pct >= 3.0:
            actions.append("trigger_retraining")
        actions.append("page_oncall=false")
        return actions
    return ["do_nothing"]


def _to_yaml(plan: Dict) -> str:
    """
    Serialize a simple plan dict to YAML.
    """
    try:
        lines: List[str] = []
        lines.append(f"status: {plan.get('status','healthy')}")
        lines.append("findings:")
        for item in plan.get("findings", []):
            if isinstance(item, dict):
                for k, v in item.items():
                    lines.append(f"  - {k}: {v}")
            else:
                lines.append(f"  - {item}")
        lines.append("actions:")
        for a in plan.get("actions", []):
            lines.append(f"  - {a}")
        rationale = plan.get("rationale", "")
        lines.append("rationale: >")
        lines.append(f"  {rationale}")
        return "\n".join(lines) + "\n"
    except Exception as e:
        raise RuntimeError(f"Failed to serialize YAML: {e}")


def _save_text(path: str, content: str) -> None:
    """
    Save text content to a file.
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise RuntimeError(f"Failed to save file '{path}': {e}")


def main() -> None:
    """
    CLI that observes metrics and drift, thinks with simple rules, and outputs an action plan in YAML.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--drift", type=str, required=True)
    parser.add_argument("--out", type=str, default="artifacts/agent_plan.yaml")
    parser.add_argument("--ts_key", type=str, default="timestamp")
    parser.add_argument("--roc_key", type=str, default="roc_auc")
    parser.add_argument("--pr_key", type=str, default="pr_auc")
    parser.add_argument("--latency_key", type=str, default="latency_p95_ms")
    parser.add_argument("--latency_threshold", type=float, default=400.0)
    parser.add_argument("--window_days", type=int, default=7)
    parser.add_argument("--consecutive", type=int, default=2)
    args = parser.parse_args()

    metrics_all = _load_jsonl(args.metrics)
    if not metrics_all:
        raise RuntimeError("Metrics history is empty.")
    drift = _load_json(args.drift)
    drift_overall = bool(drift.get("overall_drift", False))

    window = _window_by_days(metrics_all, days=int(args.window_days), ts_key=args.ts_key)
    if not window:
        n = min(len(metrics_all), 20)
        window = metrics_all[-n:]

    roc_series = _extract_series(window, args.roc_key)
    pr_series = _extract_series(window, args.pr_key)
    if len(roc_series) < 2 or len(pr_series) < 2:
        raise RuntimeError("Not enough points in metrics window to compute drops.")

    baseline_roc = _median(roc_series[:-1])
    current_roc = float(roc_series[-1])
    roc_drop_pct = _percent_drop(current_roc, baseline_roc)

    baseline_pr = _median(pr_series[:-1])
    current_pr = float(pr_series[-1])
    pr_drop_pct = _percent_drop(current_pr, baseline_pr)

    latency_breached = _consecutive_latency_breaches(
        metrics_all, args.latency_key, threshold=float(args.latency_threshold), k=int(args.consecutive)
    )

    status = _decide_status(
        roc_drop_pct=roc_drop_pct,
        pr_drop_pct=pr_drop_pct,
        latency_breached=latency_breached,
        drift_overall=drift_overall,
    )

    actions = _choose_actions(status, roc_drop_pct, latency_breached, drift_overall)

    rationale_parts: List[str] = []
    rationale_parts.append(f"ROC-AUC now {current_roc:.4f} vs 7-day median {baseline_roc:.4f} (drop {roc_drop_pct:.2f}%).")
    rationale_parts.append(f"PR-AUC now {current_pr:.4f} vs 7-day median {baseline_pr:.4f} (drop {pr_drop_pct:.2f}%).")
    if latency_breached:
        rationale_parts.append(f"Latency p95 > {int(args.latency_threshold)} ms for {int(args.consecutive)} consecutive points.")
    rationale_parts.append(f"overall_drift={str(drift_overall).lower()}.")

    plan = {
        "status": status,
        "findings": [
            {"roc_auc_drop_pct": round(roc_drop_pct, 3)},
            {"pr_auc_drop_pct": round(pr_drop_pct, 3)},
            {"latency_consecutive_breach": bool(latency_breached)},
            {"drift_overall": bool(drift_overall)},
        ],
        "actions": actions,
        "rationale": " ".join(rationale_parts),
    }

    yaml_text = _to_yaml(plan)
    _save_text(args.out, yaml_text)


if __name__ == "__main__":
    main()
