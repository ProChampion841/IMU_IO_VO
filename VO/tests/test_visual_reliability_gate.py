"""The reliability gate, and the deterministic loss that goes with it.

Two properties matter most and are easy to break silently:

1. With every gate at its default the frontend must behave EXACTLY as it did
   before the gate existed, so an ungated run stays comparable with every run
   recorded before this change and ``frontend_id`` stays honest.

2. A rejected pair must become an ABSENT one, not a zeroed one. Those are the
   two states ``visual_present`` exists to keep apart, and a zeroed token
   delivered with ``visual_present=1`` would be read as "the image showed no
   motion" - the opposite of what a rejection means.
"""

from __future__ import annotations

import csv

import pytest
import torch
import torch.nn.functional as F

from vio.models.vision_mamba_vo import (
    VisionMambaFlowFrontend,
    VisionMambaVO,
    AIDING_INPUT_DIM,
)

import tools.train_fixedwing_vo as train_fixedwing_vo
from tools.train_fixedwing_vo import (
    build_parser,
    compute_velocity_loss,
    simple_velocity_loss,
    velocity_loss,
)

# The tiny synthetic flight and the argv that trains on it, reused rather than
# rebuilt so the gate is exercised through exactly the geometry every other
# end-to-end test uses.
from test_train_fixedwing_vo_integration import (  # noqa: F401
    _train_argv,
    flight,
)


def read_metrics(run_dir):
    """The one row a single-epoch run writes, as a dict."""

    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def make_frontend(**gates):
    """A small frontend, so the test runs in seconds rather than minutes."""

    return VisionMambaFlowFrontend(
        image_size=(144, 256), patch_size=8, token_grid=6, d_model=32, depth=1,
        **gates,
    )


def make_pair(batch=2):
    """Sharp, high-texture frames - the EASY matching case.

    Independent uniform noise is the most matchable image there is: every cell
    has a unique best candidate, so the correlation is sharply peaked (median
    usable confidence ~0.77, median normalized entropy ~0.18). Useful for
    checking that the gate does not reject good measurements, and a poor probe
    for whether it rejects bad ones - see :func:`make_ambiguous_pair`.
    """

    torch.manual_seed(0)
    image0 = torch.rand(batch, 1, 144, 256)
    image1 = torch.rand(batch, 1, 144, 256)
    pair_dt_s = torch.full((batch, 1), 0.05)
    body_rate = torch.zeros(batch, 3)
    return image0, image1, pair_dt_s, body_rate


def make_ambiguous_pair(batch=2):
    """Smooth, low-texture frames - the regime the gate exists for.

    Heavy low-pass filtering destroys the fine detail correspondence needs, so
    many candidates explain each cell about equally well. Measured on this
    frontend it gives normalized entropy ~0.44 and usable confidence ~0.38,
    against ~0.18 / ~0.73 for the sharp pair above - moved a long way toward
    the real dataset's ~0.60 / ~0.26 (artifacts/eval_field.json) without
    reaching it, which is enough for a threshold to have something to bite on.
    Do not read the exact numbers as a claim to reproduce the real imagery;
    they are a regime, not a match.
    """

    torch.manual_seed(7)
    sharp0 = torch.rand(batch, 1, 144, 256)
    sharp1 = torch.rand(batch, 1, 144, 256)
    # A wide box blur, applied twice, approximates a Gaussian well enough and
    # needs no extra dependency.
    blur = torch.ones(1, 1, 15, 15) / 225.0
    def smooth(image):
        for _ in range(2):
            image = F.conv2d(F.pad(image, (7, 7, 7, 7), mode="reflect"), blur)
        return image
    pair_dt_s = torch.full((batch, 1), 0.05)
    body_rate = torch.zeros(batch, 3)
    return smooth(sharp0), smooth(sharp1), pair_dt_s, body_rate


def test_default_gates_keep_every_pair():
    """Gates off means nothing is rejected - and the grid is not empty.

    ``pair_reliable`` alone would be a vacuous assertion here: with
    ``min_reliable_cell_fraction`` at 0.0 the test is ``fraction >= 0.0``,
    which any non-negative mask satisfies including an all-zero one. Checking
    that cells actually survived is what makes "kept every pair" mean
    something.
    """

    frontend = make_frontend()
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_pair()
    with torch.no_grad():
        output = frontend(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)
    assert torch.all(output["pair_reliable"] == 1.0)
    assert float(output["diagnostics"]["reliable_cell_fraction"].min()) > 0.0


def test_default_gate_matches_the_pre_gate_reference_implementation():
    """The default gate must reproduce what the code did BEFORE it existed.

    An earlier version of this test compared the new frontend against itself
    with the gate arguments spelled out explicitly, which cannot fail: both
    sides ran the same code with the same values. It proved the defaults were
    typed consistently, not that they were INERT.

    So the reference is computed here from the pre-gate formula - ``weight``
    is ``usable_confidence * flow_valid``, with no reliability mask anywhere -
    and the frontend's own pooled output is required to match it. If the
    default ever stops meaning "reject nothing", this fails.
    """

    torch.manual_seed(1)
    frontend = make_frontend()
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_pair()

    # Capture the correlation the frontend actually used, so the reference is
    # built from identical inputs rather than a re-run that could differ.
    captured = {}
    original_correlation = frontend.correlation.forward

    def spy(features0, features1, **kwargs):
        result = original_correlation(features0, features1, **kwargs)
        captured["correlation"] = result
        return result

    frontend.correlation.forward = spy
    with torch.no_grad():
        output = frontend(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)
    frontend.correlation.forward = original_correlation

    correlation = captured["correlation"]
    # The pre-gate weight, written out longhand.
    reference_weight = (
        correlation["usable_confidence"]
        * correlation["flow_valid"].to(correlation["usable_confidence"].dtype)
    )
    gated_weight = reference_weight * frontend._reliable_cells(
        correlation, reference_weight.dtype
    )
    assert torch.equal(gated_weight, reference_weight)

    # And the pooled weight channel the fusion model actually reads.
    pooled_reference = F.adaptive_avg_pool2d(
        reference_weight, (frontend.token_grid, frontend.token_grid)
    )
    assert torch.equal(output["pooled_cells"][:, 2:3], pooled_reference)


def test_an_impossible_gate_rejects_every_pair():
    """Thresholds nothing can satisfy must refuse the pair, not pass it thinly."""

    frontend = make_frontend(
        max_cell_entropy=0.01, min_cell_confidence=0.99,
        reject_boundary_peaks=True, min_reliable_cell_fraction=0.5,
    )
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_pair()
    with torch.no_grad():
        output = frontend(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)
    assert torch.all(output["pair_reliable"] == 0.0)
    assert float(output["diagnostics"]["reliable_cell_fraction"].max()) < 0.5


def test_tighter_gates_never_keep_more_cells():
    """Reliability must be monotone: raising a threshold cannot admit a cell.

    ``sorted(reverse=True)`` alone admits ties, so a gate that did nothing at
    all would pass a pure monotonicity check. The strict inequality at the end
    is what distinguishes "monotone" from "inert".
    """

    image0, image1, pair_dt_s, body_rate = make_pair()
    kept = []
    for entropy_limit in (1.0, 0.6, 0.3, 0.05):
        torch.manual_seed(2)
        frontend = make_frontend(max_cell_entropy=entropy_limit)
        frontend.eval()
        with torch.no_grad():
            output = frontend(
                image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
            )
        kept.append(float(output["diagnostics"]["reliable_cell_fraction"].mean()))
    assert kept == sorted(kept, reverse=True)
    assert kept[0] > kept[-1], kept


def test_entropy_gate_alone_rejects_cells():
    """max_cell_entropy must do something on its own.

    Every other gate test sets several thresholds at once, so any ONE of them
    could be dead code and the file would stay green. This isolates the
    entropy gate: nothing else is tightened.
    """

    image0, image1, pair_dt_s, body_rate = make_pair()

    torch.manual_seed(3)
    open_gate = make_frontend()
    torch.manual_seed(3)
    entropy_only = make_frontend(max_cell_entropy=0.3)
    entropy_only.load_state_dict(open_gate.state_dict())
    open_gate.eval()
    entropy_only.eval()

    with torch.no_grad():
        loose = open_gate(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)
        tight = entropy_only(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)

    loose_kept = float(loose["diagnostics"]["reliable_cell_fraction"].mean())
    tight_kept = float(tight["diagnostics"]["reliable_cell_fraction"].mean())
    assert tight_kept < loose_kept, (tight_kept, loose_kept)


def test_boundary_gate_alone_rejects_cells():
    """reject_boundary_peaks must do something on its own too.

    Same reasoning as the entropy test above: isolated, so the flag cannot be
    dead code while the suite passes. Body rates are large here, which pushes
    the search centre outward and puts peaks on the window edge - without that
    there may be no boundary hits to reject and the test would be vacuous.
    """

    image0, image1, pair_dt_s, _ = make_pair()
    fast_rotation = torch.tensor([[0.6, -0.5, 0.7], [0.6, -0.5, 0.7]])

    torch.manual_seed(4)
    open_gate = make_frontend()
    torch.manual_seed(4)
    boundary_only = make_frontend(reject_boundary_peaks=True)
    boundary_only.load_state_dict(open_gate.state_dict())
    open_gate.eval()
    boundary_only.eval()

    with torch.no_grad():
        loose = open_gate(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=fast_rotation
        )
        tight = boundary_only(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=fast_rotation
        )

    # The setup has to actually produce boundary peaks, or this proves nothing.
    assert float(loose["diagnostics"]["boundary_hit_fraction"].max()) > 0.0
    loose_kept = float(loose["diagnostics"]["reliable_cell_fraction"].mean())
    tight_kept = float(tight["diagnostics"]["reliable_cell_fraction"].mean())
    assert tight_kept < loose_kept, (tight_kept, loose_kept)


def test_confidence_gate_alone_rejects_cells():
    """min_cell_confidence must do something on its own."""

    image0, image1, pair_dt_s, body_rate = make_pair()

    torch.manual_seed(5)
    open_gate = make_frontend()
    torch.manual_seed(5)
    confidence_only = make_frontend(min_cell_confidence=0.5)
    confidence_only.load_state_dict(open_gate.state_dict())
    open_gate.eval()
    confidence_only.eval()

    with torch.no_grad():
        loose = open_gate(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)
        tight = confidence_only(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
        )

    loose_kept = float(loose["diagnostics"]["reliable_cell_fraction"].mean())
    tight_kept = float(tight["diagnostics"]["reliable_cell_fraction"].mean())
    assert tight_kept < loose_kept, (tight_kept, loose_kept)


def test_a_refused_pair_becomes_absent_not_a_zeroed_present_token():
    """The contract the whole gate exists for, pinned directly.

    A refused pair must arrive as visual_present=0 with a zero token. The
    failure this guards against is the opposite and is invisible in every
    aggregate metric: a zeroed token delivered with visual_present=1, which
    the fusion model reads as "the image showed no motion" rather than "no
    image arrived" - the two states visual_present exists to separate.
    """

    from vio.data.fixedwing_vo import scatter_visual_tokens

    batch, events, window, dim = 1, 3, 10, 4
    tokens = torch.arange(1.0, batch * events * dim + 1.0).reshape(batch, events, dim)
    quality = torch.ones(batch, events, 1)
    offsets = torch.tensor([[1, 4, 7]])
    valid = torch.ones(batch, events)
    # Middle pair refused.
    delivered = torch.tensor([[1.0, 0.0, 1.0]])

    field, _, present = scatter_visual_tokens(
        tokens, quality, offsets, valid,
        window_length=window, visual_dim=dim, delivered=delivered,
    )

    assert float(present[0, 1, 0]) == 1.0
    assert float(present[0, 7, 0]) == 1.0
    # The refused pair: absent, and its token zeroed rather than left behind.
    assert float(present[0, 4, 0]) == 0.0
    assert torch.all(field[0, 4] == 0.0)
    # A tick that never had an event is indistinguishable from the refused one,
    # which is the point - both mean "no image arrived".
    assert float(present[0, 5, 0]) == 0.0
    assert torch.all(field[0, 5] == 0.0)


def test_a_refused_pair_keeps_the_frontend_in_the_autograd_graph():
    """Refusing every pair must still leave the frontend differentiable.

    If a refused pair were dropped from the scatter index set instead of
    multiplied by zero, a batch where everything was refused would contribute
    no gradient to the frontend at all - and static_graph DDP records the
    used-parameter set on iteration one and fails when a later one differs.
    """

    from vio.data.fixedwing_vo import scatter_visual_tokens

    batch, events, window, dim = 1, 2, 6, 3
    tokens = torch.ones(batch, events, dim, requires_grad=True)
    quality = torch.ones(batch, events, 1)
    offsets = torch.tensor([[0, 3]])
    valid = torch.ones(batch, events)
    delivered = torch.zeros(batch, events)  # every pair refused

    field, _, present = scatter_visual_tokens(
        tokens, quality, offsets, valid,
        window_length=window, visual_dim=dim, delivered=delivered,
    )
    assert torch.all(present == 0.0)

    field.sum().backward()
    # The edge exists and the gradient is exactly zero - both required.
    assert tokens.grad is not None
    assert torch.all(tokens.grad == 0.0)


def test_simple_loss_leaves_the_uncertainty_heads_out_of_the_graph():
    """No gradient may reach the variance or concentration heads.

    This is what makes ``find_unused_parameters=True`` necessary for a
    multi-GPU run with ``--velocity-loss simple``; without it DDP hangs
    partway through the first epoch waiting for gradients that never arrive.
    """

    torch.manual_seed(0)
    model = VisionMambaVO(visual_dim=16, aiding_dim=16, fusion_dim=16)
    batch, ticks = 2, 4
    aiding = torch.randn(batch, ticks, AIDING_INPUT_DIM)
    token = torch.randn(batch, ticks, 16)
    present = torch.ones(batch, ticks, 1)
    age = torch.zeros(batch, ticks, 1)
    log_altitude = torch.full((batch, ticks), 5.0)
    target = torch.randn(batch, ticks, 3)
    mask = torch.ones(batch, ticks)

    model.zero_grad()
    prediction = model(aiding, token, present, age, log_altitude=log_altitude)
    loss, _ = compute_velocity_loss(
        prediction, target, mask,
        direction_weight=0.5, loss_mode="simple", huber_delta=1.0,
    )
    loss.backward()

    assert model.log_variance_head.weight.grad is None
    assert model.log_concentration_head.weight.grad is None
    # The heads that DO define the prediction must still be trained.
    assert model.direction_head.weight.grad is not None
    assert model.log_rate_head.weight.grad is not None


def test_both_losses_agree_about_direction_at_unit_concentration():
    """At kappa = 1 the vMF term reduces to the fixed cosine term.

    That equality is why switching loss mode does not move the starting point:
    the two objectives differ in whether the model may argue about its own
    confidence, not in what they consider a good heading.
    """

    torch.manual_seed(0)
    batch, ticks = 2, 5
    prediction = {
        "predicted_velocity": torch.randn(batch, ticks, 3),
        "predicted_direction": torch.nn.functional.normalize(
            torch.randn(batch, ticks, 3), dim=-1
        ),
        "velocity_log_variance": torch.zeros(batch, ticks, 3),
        # log kappa = 0, so kappa = 1 - the head's initialisation.
        "direction_log_concentration": torch.zeros(batch, ticks),
    }
    target = torch.randn(batch, ticks, 3)
    mask = torch.ones(batch, ticks)

    _, nll_parts = velocity_loss(prediction, target, mask, direction_weight=0.5)
    _, simple_parts = simple_velocity_loss(
        prediction, target, mask, direction_weight=0.5, huber_delta=1.0
    )
    assert nll_parts["direction"] == simple_parts["direction"]


def test_huber_bounds_an_outlier_label():
    """Beyond the delta the penalty grows linearly, not quadratically."""

    batch, ticks = 1, 1
    direction = torch.tensor([[[1.0, 0.0, 0.0]]])
    mask = torch.ones(batch, ticks)

    def loss_for(error_m_s: float) -> float:
        prediction = {
            "predicted_velocity": torch.tensor([[[error_m_s, 0.0, 0.0]]]),
            "predicted_direction": direction,
            "velocity_log_variance": torch.zeros(batch, ticks, 3),
            "direction_log_concentration": torch.zeros(batch, ticks),
        }
        target = torch.zeros(batch, ticks, 3)
        # Direction weight 0 isolates the velocity term.
        loss, _ = simple_velocity_loss(
            prediction, target, mask, direction_weight=0.0, huber_delta=1.0
        )
        return float(loss)

    # Growth past the delta is LINEAR, so doubling the error roughly doubles
    # the cost. Squared error would quadruple it - that factor of two is the
    # whole point, and it is what stops one bad label owning its batch.
    ratio = loss_for(20.0) / loss_for(10.0)
    assert 2.0 < ratio < 2.2, ratio

    # Inside the delta the penalty is still quadratic, so small errors keep
    # the gradient shape plain MSE would give them.
    assert loss_for(0.5) / loss_for(0.25) > 3.0


def test_gate_flags_reach_the_parser():
    """The gates are opt-in, and default to rejecting nothing."""

    args = build_parser().parse_args(["--dataset", "ignored"])
    assert args.max_cell_entropy == 1.0
    assert args.min_cell_confidence == 0.0
    assert args.reject_boundary_peaks is False
    assert args.min_reliable_cell_fraction == 0.0
    # The loss default depends on the frontend and is resolved at start-up;
    # for the original frontend it is still the NLL.
    from tools.train_fixedwing_vo import resolve_frontend_defaults

    resolve_frontend_defaults(args, frame_interval_s=0.05, tick_interval_s=0.01)
    assert args.velocity_loss == "nll"

    tightened = build_parser().parse_args([
        "--dataset", "ignored",
        "--max-cell-entropy", "0.6",
        "--min-cell-confidence", "0.05",
        "--reject-boundary-peaks",
        "--min-reliable-cell-fraction", "0.25",
        "--velocity-loss", "simple",
    ])
    assert tightened.max_cell_entropy == 0.6
    assert tightened.reject_boundary_peaks is True
    assert tightened.velocity_loss == "simple"


def test_ungated_run_reports_every_pair_delivered(flight, tmp_path):
    """End to end: with the gate off, metrics.csv says nothing was dropped."""

    run_dir = tmp_path / "ungated"
    assert train_fixedwing_vo.main(_train_argv(flight, run_dir)) == 0

    rows = read_metrics(run_dir)
    assert rows, "the run wrote no metrics row"
    assert float(rows[-1]["train_visual_kept_fraction"]) == pytest.approx(1.0)


def test_gated_run_drops_pairs_and_says_so(flight, tmp_path):
    """End to end: an impossible gate must reject pairs AND report the fraction.

    This is the accounting the threshold sweep is read off. A gate that
    silently dropped pairs without moving this column would be untunable.
    """

    run_dir = tmp_path / "gated"
    assert train_fixedwing_vo.main(_train_argv(
        flight, run_dir,
        max_cell_entropy="0.01",
        min_cell_confidence="0.99",
        reject_boundary_peaks=True,
        min_reliable_cell_fraction="0.5",
    )) == 0

    rows = read_metrics(run_dir)
    assert rows, "the run wrote no metrics row"
    assert float(rows[-1]["train_visual_kept_fraction"]) == pytest.approx(0.0)


def test_simple_loss_trains_end_to_end(flight, tmp_path):
    """The deterministic loss must survive a real epoch, not just a unit call.

    ``--velocity-loss simple`` leaves two heads unused, which is exactly the
    condition that breaks a naive DDP setup; running the whole trainer once
    proves the single-process path at least is wired correctly.
    """

    run_dir = tmp_path / "simple_loss"
    assert train_fixedwing_vo.main(_train_argv(
        flight, run_dir, velocity_loss="simple", huber_delta="1.0",
    )) == 0

    rows = read_metrics(run_dir)
    assert rows, "the run wrote no metrics row"
    # A Smooth L1 loss is non-negative, unlike the NLL it replaces - which can
    # and did go below zero by growing confident.
    assert float(rows[-1]["train_loss"]) >= 0.0


def test_the_ambiguous_fixture_really_is_ambiguous():
    """Guard the guard: if the blur stops working, the tests below go vacuous.

    A gate test on sharply-peaked correlations proves nothing, because there
    is nothing ambiguous to reject. This pins the fixture into the regime the
    real dataset sits in - high entropy, low usable confidence - so a later
    change to make_ambiguous_pair cannot quietly turn the tests that depend
    on it into no-ops.
    """

    frontend = make_frontend()
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_ambiguous_pair()
    with torch.no_grad():
        output = frontend(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)

    diagnostics = output["diagnostics"]
    entropy = float(diagnostics["mean_entropy_normalized"].mean())
    confidence = float(diagnostics["mean_usable_confidence"].mean())

    sharp = make_pair()
    with torch.no_grad():
        reference = frontend(
            sharp[0], sharp[1], pair_dt_s=sharp[2], body_rate_rad_s=sharp[3]
        )
    sharp_entropy = float(reference["diagnostics"]["mean_entropy_normalized"].mean())

    assert entropy > sharp_entropy, (entropy, sharp_entropy)
    assert confidence < 0.6, confidence


def test_confidence_gate_bites_hardest_where_the_image_is_ambiguous():
    """The gate must reject MORE on smooth frames than on sharp ones.

    This is the property the whole feature is for: throw away measurements the
    correlator could not actually make, and keep the ones it could. A gate
    that rejected the same fraction either way would be filtering on something
    unrelated to image quality.
    """

    threshold = 0.5
    torch.manual_seed(11)
    gated = make_frontend(min_cell_confidence=threshold)
    gated.eval()

    sharp = make_pair()
    ambiguous = make_ambiguous_pair()
    with torch.no_grad():
        on_sharp = gated(
            sharp[0], sharp[1], pair_dt_s=sharp[2], body_rate_rad_s=sharp[3]
        )
        on_ambiguous = gated(
            ambiguous[0], ambiguous[1],
            pair_dt_s=ambiguous[2], body_rate_rad_s=ambiguous[3],
        )

    kept_sharp = float(on_sharp["diagnostics"]["reliable_cell_fraction"].mean())
    kept_ambiguous = float(on_ambiguous["diagnostics"]["reliable_cell_fraction"].mean())
    assert kept_ambiguous < kept_sharp, (kept_ambiguous, kept_sharp)


def test_pair_gate_refuses_an_ambiguous_pair_a_sharp_one_survives():
    """End of the chain: the pair-level verdict follows image quality."""

    # 0.5 means "half the cells that COULD be measured", because the fraction
    # divides by valid cells rather than by the whole grid. At this setting the
    # sharp pair keeps 0.77 and the ambiguous one 0.26, so the threshold sits
    # well clear of both. (An earlier version used 0.3 against a whole-grid
    # denominator, where the ceiling was 0.42 at this geometry and 0.3 meant a
    # far stricter 71% of measurable cells - the kind of hidden dependence on
    # image size and correlation radius the valid-cell denominator removes.)
    torch.manual_seed(12)
    gated = make_frontend(min_cell_confidence=0.5, min_reliable_cell_fraction=0.5)
    gated.eval()

    sharp = make_pair()
    ambiguous = make_ambiguous_pair()
    with torch.no_grad():
        on_sharp = gated(
            sharp[0], sharp[1], pair_dt_s=sharp[2], body_rate_rad_s=sharp[3]
        )
        on_ambiguous = gated(
            ambiguous[0], ambiguous[1],
            pair_dt_s=ambiguous[2], body_rate_rad_s=ambiguous[3],
        )

    assert torch.all(on_sharp["pair_reliable"] == 1.0)
    assert torch.all(on_ambiguous["pair_reliable"] == 0.0)


def test_the_gate_is_immune_to_the_learned_correlation_temperature():
    """The model must not be able to talk its way past its own quality gate.

    Confidence and entropy are properties of the SOFTMAX as much as of the
    match: shrinking the temperature sharpens the distribution and raises
    confidence on identical images. With a learnable temperature feeding the
    gate, "pass the gate" becomes something training can achieve without
    improving a single measurement - and a fixed threshold would silently mean
    something different at epoch 100 than at epoch 1.
    """

    torch.manual_seed(21)
    frontend = make_frontend(max_cell_entropy=0.5, min_cell_confidence=0.4)
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_ambiguous_pair()

    with torch.no_grad():
        before = frontend(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
        )
    # A twelve-fold sharpening - far more than training would ever produce.
    with torch.no_grad():
        frontend.correlation.log_temperature.fill_(
            frontend.correlation.log_temperature.item() - 2.5
        )
        after = frontend(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
        )

    # The temperature really did move, or this proves nothing.
    assert float(after["diagnostics"]["correlation_temperature"][0]) < 0.5 * float(
        before["diagnostics"]["correlation_temperature"][0]
    )
    # And the gate did not notice.
    assert torch.equal(
        before["diagnostics"]["reliable_cell_fraction"],
        after["diagnostics"]["reliable_cell_fraction"],
    )
    assert torch.equal(before["pair_reliable"], after["pair_reliable"])


def test_gating_statistics_carry_no_gradient():
    """A gate in the autograd graph is a gate the optimiser can push on."""

    torch.manual_seed(22)
    frontend = make_frontend()
    image0, image1, pair_dt_s, body_rate = make_pair()
    output = frontend(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)
    # pair_reliable comes from a comparison, so it is non-differentiable by
    # construction; what matters is that the statistics behind it are detached
    # too, so no path exists even before the threshold is applied.
    assert not output["pair_reliable"].requires_grad


def test_rejection_reasons_are_reported_per_gate():
    """Which threshold did the rejecting - not just how many were rejected."""

    torch.manual_seed(23)
    frontend = make_frontend(min_cell_confidence=0.6)
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_ambiguous_pair()
    with torch.no_grad():
        output = frontend(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
        )

    diagnostics = output["diagnostics"]
    # The gate that is ON did work; the ones that are OFF rejected nothing.
    assert float(diagnostics["rejected_low_confidence"].mean()) > 0.0
    assert float(diagnostics["rejected_high_entropy"].mean()) == 0.0
    assert float(diagnostics["rejected_low_score_margin"].mean()) == 0.0
    assert float(diagnostics["rejected_boundary_peak"].mean()) == 0.0


def test_score_margin_gate_is_temperature_free_and_bites():
    """The alternative gate: raw top-1 minus top-2, no softmax involved."""

    image0, image1, pair_dt_s, body_rate = make_ambiguous_pair()

    torch.manual_seed(24)
    open_gate = make_frontend()
    torch.manual_seed(24)
    margin_only = make_frontend(min_score_margin=0.05)
    margin_only.load_state_dict(open_gate.state_dict())
    open_gate.eval()
    margin_only.eval()

    with torch.no_grad():
        loose = open_gate(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)
        tight = margin_only(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
        )

    assert float(tight["diagnostics"]["reliable_cell_fraction"].mean()) < float(
        loose["diagnostics"]["reliable_cell_fraction"].mean()
    )
    assert float(tight["diagnostics"]["rejected_low_score_margin"].mean()) > 0.0


def test_reliable_fraction_is_measured_against_valid_cells_not_the_whole_grid():
    """1.0 must be reachable, so the pair threshold means what it says.

    Border cells have a clipped search window and are excluded by flow_valid
    before any threshold is consulted. Counting them in the denominator would
    cap the fraction below 1.0 by an amount set by image size, patch size and
    correlation radius - so --min-reliable-cell-fraction 0.5 would silently
    mean a different strictness at every geometry.
    """

    frontend = make_frontend()
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_pair()
    with torch.no_grad():
        output = frontend(image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate)

    # Ungated: every measurable cell is reliable, so the fraction is exactly 1.
    fraction = output["diagnostics"]["reliable_cell_fraction"]
    assert torch.allclose(fraction, torch.ones_like(fraction))

    # And that is NOT simply because every cell is valid - a good number of
    # them are border cells, which is what made the old whole-grid mean cap
    # out well below 1.0.
    assert float(output["diagnostics"]["occupied_fraction"].mean()) < 0.9


def test_the_fraction_is_stable_across_correlation_radius():
    """The same imagery must give the same fraction at a different radius.

    A whole-grid denominator fails this: a larger radius makes a wider border,
    so more cells are invalid and the ceiling drops - moving the fraction
    without anything about the match quality changing.
    """

    image0, image1, pair_dt_s, body_rate = make_pair()
    fractions = []
    for radius in (2, 4, 6):
        torch.manual_seed(31)
        frontend = make_frontend(correlation_radius=radius)
        frontend.eval()
        with torch.no_grad():
            output = frontend(
                image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
            )
        fractions.append(float(output["diagnostics"]["reliable_cell_fraction"].mean()))
    # Ungated, every valid cell is reliable at every radius.
    assert all(abs(value - 1.0) < 1e-6 for value in fractions), fractions


def test_occupancy_is_immune_to_the_learned_temperature():
    """The SECOND filtering stage must not move with the temperature either.

    pair_reliable was protected first; min_pool_weight was not. Because
    occupancy thresholds a confidence-weighted quantity, a shrinking learned
    temperature alone would push blocks past min_pool_weight and change the
    token, with no measurement having improved.
    """

    torch.manual_seed(32)
    frontend = make_frontend(min_pool_weight=0.15)
    frontend.eval()
    image0, image1, pair_dt_s, body_rate = make_ambiguous_pair()

    with torch.no_grad():
        before = frontend(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
        )
        frontend.correlation.log_temperature.fill_(
            frontend.correlation.log_temperature.item() - 2.5
        )
        after = frontend(
            image0, image1, pair_dt_s=pair_dt_s, body_rate_rad_s=body_rate
        )

    assert float(after["diagnostics"]["correlation_temperature"][0]) < 0.5 * float(
        before["diagnostics"]["correlation_temperature"][0]
    )
    assert torch.equal(
        before["diagnostics"]["occupied_fraction"],
        after["diagnostics"]["occupied_fraction"],
    )
