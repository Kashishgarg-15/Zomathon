"""
Data Curation Pipeline — Orchestrator
======================================
Runs all phases sequentially:
  Phase 0: Derived features (free, pandas)
  Phase 1: LLM item enrichment (Groq API, ~6 calls)
  Phase 2: Meal completeness scoring
  Phase 3: Cart sequence simulation
  Phase 4: Training data construction

Usage:
  cd /home/rkp/coding/zomatothon
  python -m data_curation.run_pipeline
"""
import json
import sys
import time
import traceback

from data_curation.config import PIPELINE_LOG


def main():
    start_time = time.time()
    log = {"phases": {}, "success": False}

    try:
        # ── Phase 0 ────────────────────────────────────────────────────
        from data_curation.phase0_derived_features import run_phase0
        t0 = time.time()
        log["phases"]["phase0"] = run_phase0()
        log["phases"]["phase0"]["elapsed_sec"] = round(time.time() - t0, 1)

        # ── Phase 1 ────────────────────────────────────────────────────
        from data_curation.phase1_llm_enrichment import run_phase1
        t1 = time.time()
        log["phases"]["phase1"] = run_phase1()
        log["phases"]["phase1"]["elapsed_sec"] = round(time.time() - t1, 1)

        # ── Phase 2 ────────────────────────────────────────────────────
        from data_curation.phase2_completeness import run_phase2
        t2 = time.time()
        log["phases"]["phase2"] = run_phase2()
        log["phases"]["phase2"]["elapsed_sec"] = round(time.time() - t2, 1)

        # ── Phase 3 ────────────────────────────────────────────────────
        from data_curation.phase3_cart_sequences import run_phase3
        t3 = time.time()
        log["phases"]["phase3"] = run_phase3()
        log["phases"]["phase3"]["elapsed_sec"] = round(time.time() - t3, 1)

        # ── Phase 4 ────────────────────────────────────────────────────
        from data_curation.phase4_training_data import run_phase4
        t4 = time.time()
        log["phases"]["phase4"] = run_phase4()
        log["phases"]["phase4"]["elapsed_sec"] = round(time.time() - t4, 1)

        # ── Phase 4.5 ──────────────────────────────────────────────────
        from data_curation.phase4_5_augmentation import run_phase4_5
        t45 = time.time()
        log["phases"]["phase4_5"] = run_phase4_5()
        log["phases"]["phase4_5"]["elapsed_sec"] = round(time.time() - t45, 1)

        # ── Phase 5 ────────────────────────────────────────────────────
        from data_curation.phase5_city_assignment import run_phase5
        t5 = time.time()
        log["phases"]["phase5"] = run_phase5()
        log["phases"]["phase5"]["elapsed_sec"] = round(time.time() - t5, 1)

        log["success"] = True

    except Exception as e:
        log["error"] = str(e)
        log["traceback"] = traceback.format_exc()
        print(f"\n❌ Pipeline failed: {e}")
        traceback.print_exc()

    log["total_elapsed_sec"] = round(time.time() - start_time, 1)

    # Save log
    with open(PIPELINE_LOG, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Pipeline {'SUCCEEDED' if log['success'] else 'FAILED'} in {log['total_elapsed_sec']}s")
    print(f"Log saved to {PIPELINE_LOG}")
    print(f"{'='*60}")

    if not log["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
