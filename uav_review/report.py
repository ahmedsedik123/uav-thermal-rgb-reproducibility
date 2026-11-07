"""One source of truth for manuscript tables and independent-run figures."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import digest_object, read_csv, read_json, require_empty, write_csv, write_json
from .metrics import coco_metrics, paired_bootstrap
from .protocol import protocol_scope


def summarize(results_root, output, bootstrap_iterations=0):
    cohorts = defaultdict(list)
    for run_path in sorted(Path(results_root).rglob("run.json")):
        run = read_json(run_path)
        if run["status"] != "TRAINED":
            continue
        protocol = read_json(Path(run["fold"]) / "protocol.json")
        scope = protocol_scope(protocol)
        if bootstrap_iterations and not scope["source_sessions_verified"]:
            raise ValueError("Session-bootstrap intervals need verified source sessions. "
                             "Use --bootstrap 0 for an exploratory report.")
        split_identity = protocol["source_manifest_sha"] if protocol["mode"] == "grouped_cv" else protocol["protocol_sha"]
        for metrics_path in run_path.parent.glob("evaluation_*/test_metrics.json"):
            metrics = read_json(metrics_path)
            settings = read_json(metrics_path.parent / "frozen_evaluation_settings.json")
            if metrics["protocol_sha"] != run["protocol_sha"]:
                raise ValueError("Evaluation/training lineage mismatch")
            settings_without_run = {k: v for k, v in settings.items()
                                    if k not in {"checkpoint_sha256", "protocol_sha", "head_mode"}}
            recipe = {k: v for k, v in run["config"]["training"].items()}
            identity = {"dataset": run["dataset"], "modality": run["modality"], "mode": protocol["mode"],
                        "split_identity": split_identity, "training_recipe": recipe,
                         "evaluation_settings": settings_without_run, **scope}
            cohorts[digest_object(identity)[:12]].append({
                "run": run, "run_path": run_path, "metrics": metrics,
                "evaluation": metrics_path.parent, "identity": identity, "protocol": protocol})
    output = require_empty(output)
    if not cohorts:
        raise ValueError("No completed and evaluated runs found; no tables were fabricated.")
    for cohort_id, items in cohorts.items():
        folder = output / cohort_id
        folder.mkdir()
        write_json(folder / "cohort_definition.json", items[0]["identity"])
        scope = protocol_scope(items[0]["protocol"])
        if not scope["source_sessions_verified"]:
            (folder / "RESULT_SCOPE.txt").write_text(
                "EXPLORATORY IMAGE-FAMILY HOLDOUT\n"
                "Source flight/video sessions are unverified. Known augmentation families and identical pixels "
                "stay together, but temporal/scene dependencies can remain across partitions.\n"
                "These results do not establish independent-flight or unseen-site generalization.\n"
                + ("The paired cohort uses verified aligned RGB/thermal cases and shared partitions.\n"
                   if scope.get("paired_sensors_verified") else
                   "Separate unmatched RGB and thermal collections do not establish a fusion advantage.\n") +
                "Seed standard deviations describe optimization variability, not session generalization.\n"
                "Session-bootstrap confidence intervals are disabled.\n", encoding="utf-8")
        by_model_seed = defaultdict(list)
        for item in items:
            by_model_seed[(item["run"]["model_key"], str(item["run"]["seed"]))].append(item)
        all_rows = None
        seed_metrics = []
        predictions_by_model = defaultdict(dict)
        for (model, seed), seed_items in sorted(by_model_seed.items()):
            rows_by_id, predictions, tp, fp, fn = {}, {}, 0, 0, 0
            for item in seed_items:
                rows = [r for r in read_csv(Path(item["run"]["fold"]) / "assignments.csv") if r["split"] == "test"]
                expected_folds = int(item["protocol"].get("total_folds", 1))
                if item["protocol"]["mode"] == "grouped_cv" and len(seed_items) != expected_folds:
                    raise ValueError(f"Incomplete grouped CV for {model}, seed {seed}; complete every outer fold.")
                for row in rows:
                    if row["image_id"] in rows_by_id:
                        raise ValueError("Repeated test image within one model/seed; do not combine overlapping experiments.")
                    rows_by_id[row["image_id"]] = row
                predictions.update(read_json(item["evaluation"] / "test_predictions.json"))
                tp += item["metrics"]["TP"]
                fp += item["metrics"]["FP"]
                fn += item["metrics"]["FN"]
            rows = [rows_by_id[k] for k in sorted(rows_by_id)]
            if all_rows is None:
                all_rows = rows
            elif {r["image_id"] for r in all_rows} != set(rows_by_id):
                raise ValueError("All compared models and seeds must share exactly the same test images.")
            ap = coco_metrics(rows, predictions)
            p = tp/(tp+fp) if tp+fp else 0.
            r = tp/(tp+fn) if tp+fn else 0.
            seed_metrics.append({"model": model, "seed": seed, **scope, **ap,
                                 "precision": p, "recall": r, "F1": 2*p*r/(p+r) if p+r else 0.,
                                 "TP": tp, "FP": fp, "FN": fn,
                                 "test_images": len(rows),
                                 "operating_point": "validation-selected within each outer fold"})
            predictions_by_model[model][seed] = predictions
        write_csv(folder / "test_results_by_seed.csv", seed_metrics)
        mean_sd = []
        for model in sorted(predictions_by_model):
            selected = [r for r in seed_metrics if r["model"] == model]
            result = {"model": model, "completed_seeds": len(selected), **scope}
            for metric in ["AP50_95", "AP50", "AP75", "precision", "recall", "F1"]:
                values = [r[metric] for r in selected]
                result[f"{metric}_mean"] = float(np.mean(values))
                result[f"{metric}_SD"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            mean_sd.append(result)
        write_csv(folder / "model_comparison_mean_SD.csv", mean_sd)
        if bootstrap_iterations:
            interval, differences = paired_bootstrap(all_rows, predictions_by_model, bootstrap_iterations)
            write_csv(folder / "session_seed_confidence_intervals.csv", interval)
            write_csv(folder / "paired_model_differences.csv", differences)
        plot_comparison(mean_sd, folder)
        for item in items:
            label = f"{item['run']['model_key']}_seed{item['run']['seed']}_fold{item['protocol']['fold']}"
            plot_history(item["run_path"].parent / "history.csv", folder / "learning_curves" / f"{label}.png",
                         evaluation_scope=scope["evaluation_scope"])
            plot_detection_outputs(item, folder / "qualitative" / f"{label}.png")
        timing_rows = []
        for item in items:
            for path in item["run_path"].parent.glob("timing_*/timing_summary.json"):
                data = read_json(path)
                evaluation_precision = item["metrics"].get("numerical_precision", "fp32")
                if isinstance(item["metrics"].get("precision"), str):
                    evaluation_precision = item["metrics"]["precision"]  # archived pre-fix records
                if data["variant"] != item["metrics"]["variant"] or data["precision"] != evaluation_precision:
                    continue
                timing_rows.append({"model": item["run"]["model_key"], "seed": item["run"]["seed"], **scope,
                                    **{k: v for k, v in data.items() if not isinstance(v, (dict, list))}})
        if timing_rows:
            write_csv(folder / "runtime_measurements.csv", timing_rows)
    return output


def plot_comparison(rows, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    if rows and rows[0].get("evaluation_scope", "").startswith("exploratory"):
        fig.suptitle("EXPLORATORY image-family holdout: source sessions unverified")
    for axis, metric in zip(axes, ["AP50", "AP50_95", "F1"]):
        values = [r[f"{metric}_mean"] for r in rows]
        errors = [r[f"{metric}_SD"] or 0. for r in rows]
        bars = axis.bar([r["model"] for r in rows], values, yerr=errors, capsize=3, color="#173b5e")
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        axis.set_ylim(0, 1.12)
        axis.set_title(f"{metric}: mean and seed SD")
        axis.tick_params(axis="x", rotation=40)
    fig.savefig(Path(folder) / "detection_comparison.png", dpi=220)
    plt.close(fig)


def plot_history(path, output, evaluation_scope="verified_source_session"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [{k.strip(): v for k, v in r.items()} for r in read_csv(path)]
    if not rows:
        return
    epochs = [float(r["epoch"]) for r in rows]
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError("Plotting concatenated training runs is forbidden.")
    metrics = ["train/box_loss", "val/box_loss", "metrics/precision(B)", "metrics/recall(B)",
               "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    if evaluation_scope.startswith("exploratory"):
        fig.suptitle("EXPLORATORY training: source sessions unverified")
    for axis, key in zip(axes.flat, metrics):
        if key in rows[0]:
            axis.plot(epochs, [float(r[key]) for r in rows], color="#173b5e")
            axis.set_title(key)
            axis.set_xlabel("Epoch within this run")
        else:
            axis.text(.5, .5, f"Not logged: {key}", ha="center", transform=axis.transAxes)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_detection_outputs(item, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from .metrics import match_boxes, truths
    rows = sorted((r for r in read_csv(Path(item["run"]["fold"]) / "assignments.csv")
                   if r["split"] == "test"), key=lambda r: r["image_id"])[:4]
    predictions = read_json(item["evaluation"] / "test_predictions.json")
    confidence = read_json(item["evaluation"] / "operating_point.json")["confidence"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 5*len(rows)), squeeze=False, constrained_layout=True)
    exploratory = not protocol_scope(item["protocol"])["source_sessions_verified"]
    if exploratory:
        fig.suptitle("EXPLORATORY image-family holdout: source sessions unverified")
    for pair, row in zip(axes, rows):
        with Image.open(row["image_path"]) as image:
            for axis in pair:
                axis.imshow(image)
                axis.axis("off")
        ground_truth = truths(row)
        matched, fp, missed = match_boxes(ground_truth, predictions[row["image_id"]], confidence)
        for box in ground_truth:
            x, y, r, b = box
            pair[0].add_patch(Rectangle((x, y), r-x, b-y, fill=False, ec="#00ff99", lw=1.7))
        pair[0].set_title(f"Ground truth: {Path(row['image_path']).name}")
        for prediction, color in [(p, "#00ff99") for _, p in matched]+[(p, "#ff4b4b") for p in fp]:
            x, y, r, b = prediction["xyxy"]
            pair[1].add_patch(Rectangle((x, y), r-x, b-y, fill=False, ec=color, lw=1.7))
        for index in missed:
            x, y, r, b = ground_truth[index]
            pair[1].add_patch(Rectangle((x, y), r-x, b-y, fill=False, ec="#ffff00", lw=1.7, ls="--"))
        pair[1].set_title(f"TP {len(matched)} / FP {len(fp)} / FN {len(missed)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    metrics = item["metrics"]
    matrix = np.asarray([[metrics["TP"], metrics["FP"]], [metrics["FN"], np.nan]])
    fig, axis = plt.subplots(figsize=(5.8, 5), constrained_layout=True)
    if exploratory:
        fig.suptitle("EXPLORATORY: source sessions unverified")
    axis.imshow(np.ma.masked_invalid(matrix), cmap="Blues")
    axis.set_xticks([0, 1], ["Human", "Background"])
    axis.set_yticks([0, 1], ["Human", "Background"])
    axis.set_xlabel("Ground truth")
    axis.set_ylabel("Prediction")
    for y in range(2):
        for x in range(2):
            value = matrix[y, x]
            text_color = "white" if np.isfinite(value) and value > np.nanmax(matrix)/2 else "black"
            axis.text(x, y, "Not defined" if np.isnan(value) else str(int(value)),
                      ha="center", va="center", color=text_color)
    axis.set_title("Detection counts at validation-selected confidence")
    fig.savefig(output.with_name(output.stem+"_confusion.png"), dpi=200)
    plt.close(fig)
    curve = metrics.get("PR50", [])
    if curve:
        fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
        if exploratory:
            fig.suptitle("EXPLORATORY: source sessions unverified")
        axis.plot([r["recall"] for r in curve], [r["precision"] for r in curve], color="#173b5e")
        axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
                 title=f"Person detection at IoU 0.50; AP50={metrics['AP50']:.3f}")
        fig.savefig(output.with_name(output.stem+"_PR50.png"), dpi=200)
        plt.close(fig)
