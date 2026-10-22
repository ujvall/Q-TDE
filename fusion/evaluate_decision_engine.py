"""
Evaluation script for Hierarchical QR Decision Engine on Validation and Test Splits.

Evaluates:
1. Physical / Structural Tampering Detection:
   - Reed-Solomon ECC alone (Reed-Solomon integrity violation via non-zero syndromes)
   - Classical Peak High-Frequency Anomaly alone (local high-frequency anomalies exceeding tau = 0.525490)
   - Combined Hierarchical Decision Engine
2. Complementarity Breakdown (detected by ECC alone, Peak alone, both, neither)
3. URL Destination Risk Distribution (evaluating 21 structural URL risk indicator rules)
"""

import os
import argparse
from typing import Dict, List, Any
from fusion.decision_engine import HierarchicalQRDecisionEngine, PEAK_ANOMALY_THRESHOLD


def evaluate_split(split_name: str = "val", base_dir: str = "data") -> Dict[str, Any]:
    engine = HierarchicalQRDecisionEngine(peak_anomaly_threshold=PEAK_ANOMALY_THRESHOLD)

    split_dir = os.path.join(base_dir, split_name)
    genuine_dir = os.path.join(split_dir, "genuine")
    tampered_dir = os.path.join(split_dir, "tampered")

    genuine_files = sorted(os.listdir(genuine_dir))
    tampered_files = sorted(os.listdir(tampered_dir))

    records = []

    # Process genuine samples (Label = 0)
    for fname in genuine_files:
        fpath = os.path.join(genuine_dir, fname)
        res = engine.inspect(image_path=fpath)
        records.append({
            "filename": fname,
            "ground_truth": 0,
            "res": res
        })

    # Process tampered samples (Label = 1)
    for fname in tampered_files:
        fpath = os.path.join(tampered_dir, fname)
        res = engine.inspect(image_path=fpath)
        records.append({
            "filename": fname,
            "ground_truth": 1,
            "res": res
        })

    def calc_metrics(tp: int, tn: int, fp: int, fn: int):
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy
        }

    # 1. ECC-only metrics
    ecc_tp = sum(1 for r in records if r["ground_truth"] == 1 and r["res"]["tampering"]["ecc"]["residual"] > 0)
    ecc_fp = sum(1 for r in records if r["ground_truth"] == 0 and r["res"]["tampering"]["ecc"]["residual"] > 0)
    ecc_tn = sum(1 for r in records if r["ground_truth"] == 0 and r["res"]["tampering"]["ecc"]["residual"] == 0)
    ecc_fn = sum(1 for r in records if r["ground_truth"] == 1 and r["res"]["tampering"]["ecc"]["residual"] == 0)
    ecc_metrics = calc_metrics(ecc_tp, ecc_tn, ecc_fp, ecc_fn)

    # 2. Peak-anomaly-only metrics
    peak_tp = sum(1 for r in records if r["ground_truth"] == 1 and r["res"]["tampering"]["peak_anomaly"]["triggered"])
    peak_fp = sum(1 for r in records if r["ground_truth"] == 0 and r["res"]["tampering"]["peak_anomaly"]["triggered"])
    peak_tn = sum(1 for r in records if r["ground_truth"] == 0 and not r["res"]["tampering"]["peak_anomaly"]["triggered"])
    peak_fn = sum(1 for r in records if r["ground_truth"] == 1 and not r["res"]["tampering"]["peak_anomaly"]["triggered"])
    peak_metrics = calc_metrics(peak_tp, peak_tn, peak_fp, peak_fn)

    # 3. Combined Hierarchical metrics
    comb_tp = sum(1 for r in records if r["ground_truth"] == 1 and r["res"]["tampering"]["status"] == "tampering_detected")
    comb_fp = sum(1 for r in records if r["ground_truth"] == 0 and r["res"]["tampering"]["status"] == "tampering_detected")
    comb_tn = sum(1 for r in records if r["ground_truth"] == 0 and r["res"]["tampering"]["status"] == "no_tampering_evidence_detected")
    comb_fn = sum(1 for r in records if r["ground_truth"] == 1 and r["res"]["tampering"]["status"] != "tampering_detected")
    comb_metrics = calc_metrics(comb_tp, comb_tn, comb_fp, comb_fn)

    # 4. Complementarity on tampered samples
    tampered_records = [r for r in records if r["ground_truth"] == 1]
    both_detected = 0
    ecc_only_detected = 0
    peak_only_detected = 0
    neither_detected = 0

    for r in tampered_records:
        ecc_trig = (r["res"]["tampering"]["ecc"]["residual"] > 0)
        peak_trig = r["res"]["tampering"]["peak_anomaly"]["triggered"]
        if ecc_trig and peak_trig:
            both_detected += 1
        elif ecc_trig and not peak_trig:
            ecc_only_detected += 1
        elif not ecc_trig and peak_trig:
            peak_only_detected += 1
        else:
            neither_detected += 1

    # 5. URL Risk Level Distribution
    url_dist = {}
    for r in records:
        lvl = r["res"]["url"]["risk_level"]
        url_dist[lvl] = url_dist.get(lvl, 0) + 1

    return {
        "split": split_name,
        "n_genuine": len(genuine_files),
        "n_tampered": len(tampered_files),
        "records": records,
        "ecc_metrics": ecc_metrics,
        "peak_metrics": peak_metrics,
        "comb_metrics": comb_metrics,
        "complementarity": {
            "total_tampered": len(tampered_records),
            "ecc_alone_count": ecc_tp,
            "peak_alone_count": peak_tp,
            "combined_count": comb_tp,
            "both_count": both_detected,
            "ecc_only_count": ecc_only_detected,
            "peak_only_count": peak_only_detected,
            "neither_count": neither_detected
        },
        "url_distribution": url_dist
    }


def print_evaluation_report(results: Dict[str, Any]):
    split = results["split"].upper()
    print("=" * 70)
    print(f"EVALUATION REPORT: {split} SET ({results['n_genuine']} Genuine + {results['n_tampered']} Tampered = {results['n_genuine'] + results['n_tampered']} Total)")
    print("=" * 70)

    print("\n1. PHYSICAL / STRUCTURAL TAMPERING METRICS:")
    print("-" * 70)
    print(f"{'Branch / Engine':<26} | {'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4} | {'Sens':>8} {'Spec':>8} {'Acc':>8}")
    print("-" * 70)

    for name, m in [("Reed-Solomon ECC", results["ecc_metrics"]),
                    ("Peak Anomaly (tau=0.525)", results["peak_metrics"]),
                    ("Combined Hierarchical", results["comb_metrics"])]:
        print(f"{name:<26} | {m['tp']:>4} {m['tn']:>4} {m['fp']:>4} {m['fn']:>4} | "
              f"{m['sensitivity']*100:>7.2f}% {m['specificity']*100:>7.2f}% {m['accuracy']*100:>7.2f}%")
    print("-" * 70)

    c = results["complementarity"]
    print("\n2. COMPLEMENTARITY ANALYSIS (ON TAMPERED SAMPLES):")
    print(f"   Total tampered samples: {c['total_tampered']}")
    print(f"   - Detected by ECC alone:             {c['ecc_alone_count']} / {c['total_tampered']} ({c['ecc_alone_count']/c['total_tampered']*100:.1f}%)")
    print(f"   - Detected by Peak Anomaly alone:    {c['peak_alone_count']} / {c['total_tampered']} ({c['peak_alone_count']/c['total_tampered']*100:.1f}%)")
    print(f"   - Detected by Combined Engine:       {c['combined_count']} / {c['total_tampered']} ({c['combined_count']/c['total_tampered']*100:.1f}%)")
    print(f"   Breakdown:")
    print(f"     * Triggered by BOTH ECC and Peak:  {c['both_count']}")
    print(f"     * Triggered by ECC ONLY:           {c['ecc_only_count']}")
    print(f"     * Triggered by Peak Anomaly ONLY:  {c['peak_only_count']}")
    print(f"     * Missed by BOTH (FN):             {c['neither_count']}")

    print("\n3. URL RISK LEVEL DISTRIBUTION:")
    for lvl, count in sorted(results["url_distribution"].items()):
        pct = count / (results["n_genuine"] + results["n_tampered"]) * 100
        print(f"   - {lvl:<12}: {count:>3} samples ({pct:.1f}%)")
    print("   Note: Synthetic example.com URLs used in dataset; reflects structural analyzer baseline.")

    print("\n4. PER-SAMPLE DETAILED AUDIT:")
    print("-" * 70)
    print(f"{'Filename':<18} | {'GT':<3} | {'ECC Res':>8} {'ECC Trg':>7} | {'Peak Sc':>8} {'Pk Trg':>6} | {'Status':<28} | {'URL Lvl'}")
    print("-" * 70)
    for r in results["records"]:
        gt_str = "TAM" if r["ground_truth"] == 1 else "GEN"
        ecc = r["res"]["tampering"]["ecc"]
        pk = r["res"]["tampering"]["peak_anomaly"]
        status = r["res"]["tampering"]["status"]
        url_lvl = r["res"]["url"]["risk_level"]
        print(f"{r['filename']:<18} | {gt_str:<3} | {ecc['residual']:>8} {str(ecc['residual']>0):>7} | "
              f"{pk['score']:>8.4f} {str(pk['triggered']):>6} | {status:<28} | {url_lvl}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "all"])
    args = parser.parse_args()

    if args.split in ["val", "all"]:
        val_res = evaluate_split("val")
        print_evaluation_report(val_res)

    if args.split in ["test", "all"]:
        test_res = evaluate_split("test")
        print_evaluation_report(test_res)
