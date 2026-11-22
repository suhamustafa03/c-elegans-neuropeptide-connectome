from behavior_forecast.paired_comparison import (
    paired_win_count,
    neural_beats_prevalence,
    combined_vs_behavior_only,
    run_all_comparisons,
    format_report,
)


def _fold(pr_auc=None, prevalence=None, undefined=False):
    return {"pr_auc": pr_auc, "prevalence": prevalence, "undefined": undefined}


def _recording(name, blocked_combined, blocked_behavior, blocked_neural, blocked_prevalence):
    """Build a minimal fixture recording with one fold per model/task."""
    def results_for(pr):
        return {"reversal": [_fold(pr_auc=pr, prevalence=blocked_prevalence)]}
    return {
        "file": name,
        "blocked": {"results": {
            "combined": results_for(blocked_combined),
            "behavior_only": results_for(blocked_behavior),
            "neural_only": results_for(blocked_neural),
            "persistence": results_for(blocked_prevalence),
        }},
    }


def test_paired_win_count_counts_wins_ties_losses_correctly():
    records = [
        _recording("a", blocked_combined=0.6, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2),
        _recording("b", blocked_combined=0.5, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2),
        _recording("c", blocked_combined=0.4, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2),
    ]
    out = paired_win_count(records, "blocked", "reversal", "pr_auc", "combined", "behavior_only")
    assert out["a_gt_b"] == 1
    assert out["tie"] == 1
    assert out["b_gt_a"] == 1
    assert out["n_compared"] == 3
    assert out["n_excluded"] == 0
    assert out["median_diff_a_minus_b"] == 0.0


def test_paired_win_count_excludes_recordings_with_undefined_metric():
    records = [
        _recording("a", blocked_combined=0.6, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2),
    ]
    # Make behavior_only's fold undefined for this recording.
    records[0]["blocked"]["results"]["behavior_only"]["reversal"][0]["undefined"] = True
    out = paired_win_count(records, "blocked", "reversal", "pr_auc", "combined", "behavior_only")
    assert out["n_compared"] == 0
    assert out["n_excluded"] == 1
    assert out["median_diff_a_minus_b"] is None


def test_neural_beats_prevalence_true_case():
    records = [
        _recording("a", blocked_combined=0.6, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2),
    ]
    out = neural_beats_prevalence(records, "blocked")
    assert out["neural_gt_prevalence"] == 1
    assert out["n_compared"] == 1
    assert out["n_excluded"] == 0


def test_neural_beats_prevalence_false_case_when_neural_below_prevalence():
    records = [
        _recording("a", blocked_combined=0.6, blocked_behavior=0.5, blocked_neural=0.1, blocked_prevalence=0.2),
    ]
    out = neural_beats_prevalence(records, "blocked")
    assert out["neural_gt_prevalence"] == 0
    assert out["n_compared"] == 1


def test_combined_vs_behavior_only_matches_paired_win_count():
    records = [
        _recording("a", blocked_combined=0.6, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2),
    ]
    direct = combined_vs_behavior_only(records, "blocked")
    via_generic = paired_win_count(records, "blocked", "reversal", "pr_auc", "combined", "behavior_only")
    assert direct == via_generic


def test_run_all_comparisons_covers_both_schemes():
    rec = _recording("a", blocked_combined=0.6, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2)
    rec["forward_chaining"] = rec["blocked"]  # reuse same fixture shape for both schemes
    out = run_all_comparisons([rec])
    assert set(out.keys()) == {"blocked", "forward_chaining"}
    assert "neural_vs_prevalence" in out["blocked"]
    assert "combined_vs_behavior_only" in out["blocked"]


def test_format_report_includes_both_groups_and_is_nonempty_string():
    rec = _recording("a", blocked_combined=0.6, blocked_behavior=0.5, blocked_neural=0.4, blocked_prevalence=0.2)
    rec["forward_chaining"] = rec["blocked"]
    results = run_all_comparisons([rec])
    report = format_report(results, results)
    assert "marker-free (primary)" in report
    assert "marker-present (sensitivity)" in report
    assert "neural-only PR-AUC >" in report
    assert "combined PR-AUC >" in report


def test_median_diff_is_none_when_no_recordings_compared():
    out = paired_win_count([], "blocked", "reversal", "pr_auc", "combined", "behavior_only")
    assert out["n_compared"] == 0
    assert out["median_diff_a_minus_b"] is None
