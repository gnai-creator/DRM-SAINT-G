from types import SimpleNamespace

from saint.adapters.drm_grafting_graftblock_routed_utils import marco_name


def test_marco_name_prioritizes_tt_mps_adapter_as_marco_4o():
    args = SimpleNamespace(
        adapter_type="tt_mps",
        candidate_score_mode="composed_gain_orthogonal",
        candidate_top_k=4,
        ntk_activation_probe_batches=0,
        post_first_stage_size=1,
        candidate_learning_rates=[1e-7],
        candidate_init_scales=[0.001],
        candidate_activations=["silu"],
    )

    assert marco_name(args) == "4o_tt_mps_adapter_baseline"


def test_marco_name_prefers_explicit_marco_label_for_followups():
    args = SimpleNamespace(
        marco_label="4o_b_tt_mps_capacity_sanity",
        adapter_type="tt_mps",
        candidate_score_mode="composed_gain_orthogonal",
        candidate_top_k=4,
        ntk_activation_probe_batches=0,
        post_first_stage_size=1,
        candidate_learning_rates=[1e-7],
        candidate_init_scales=[0.001],
        candidate_activations=["silu"],
    )

    assert marco_name(args) == "4o_b_tt_mps_capacity_sanity"
