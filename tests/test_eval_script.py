"""
Tests for scripts/eval.py — the routing-accuracy + model-metrics eval
harness (Phase 3). Verifies the harness itself runs correctly and its
labeled routing set is fully accurate against the deterministic regex
router, not that any particular model hits a target number (those numbers
are reported honestly, not asserted against a threshold -- see
scripts/eval.py's module docstring).
"""
import scripts.eval as ev


class TestRoutingEval:
    def test_labeled_set_is_fully_routable_and_correct(self):
        """Every question in the eval set is deliberately unambiguous --
        the whole point of the set is to be a clean, deterministic ground
        truth. If this regresses, either a routing regex broke or the eval
        set itself needs updating alongside it."""
        result = ev.eval_routing()
        assert result["wrong"] == []
        assert result["no_opinion"] == 0
        assert result["accuracy_overall"] == 1.0

    def test_returns_expected_shape(self):
        result = ev.eval_routing()
        assert set(result) == {
            "total", "correct", "no_opinion", "wrong",
            "accuracy_on_attempted", "accuracy_overall",
        }
        assert result["total"] == len(ev.ROUTING_EVAL_SET)


class TestModelMetricsEval:
    def test_reads_real_model_files_not_placeholders(self):
        """Model metrics must be read from the actual trained files, not a
        hardcoded/mocked constant -- run models/train_models.py first if
        this fails with a 'not found' style result."""
        result = ev.eval_models()
        assert result["churn"] is not None, "run models/train_models.py first"
        assert result["forecast"] is not None, "run models/train_models.py first"
        assert 0.0 <= result["churn"]["accuracy"] <= 1.0
        assert result["forecast"]["mape"] >= 0


class TestMainRuns:
    def test_main_runs_without_raising(self, capsys):
        ev.main()
        out = capsys.readouterr().out
        assert "EVAL: Routing accuracy" in out
        assert "EVAL: Trained model metrics" in out
