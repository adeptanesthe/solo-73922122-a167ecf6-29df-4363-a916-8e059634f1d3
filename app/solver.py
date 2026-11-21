"""Shared-FIR equalizer optimisation for a multi-seat listening area.

The theatre loudspeaker can only load one set of EQ coefficients, so a
single 32-tap causal FIR ``g`` is optimised for all seats at once.

For seat ``k`` with impulse response ``h_k`` (all seats share the same
sample rate and are aligned on the direct-sound onset) the equalised
response is the *full* convolution ``y_k = h_k * g`` of length
``N = len(h_k) + 31``.  The target is a unit impulse delayed by
``target_delay`` samples, and the per-seat error is the mean squared
error between ``y_k`` and that delayed impulse.

The optimiser minimises the worst per-seat MSE subject to

* ``sum(abs(g)) <= gain_limit`` (the hardware gain budget), and
* ``mse_k(g) <= mse_k(passthrough)`` for every seat, i.e. no seat may
  end up worse than the plain unit-gain passthrough at the same delay.

The problem is convex (a max of convex quadratics under convex
constraints) and is solved in epigraph form with SLSQP.  The non-smooth
L1 budget is modelled exactly with a split variable
``g = g_plus - g_minus`` (``g_plus, g_minus >= 0``) so that
``sum(abs(g)) = sum(g_plus + g_minus)`` becomes a *linear* constraint.
Modelling it directly with ``-sign(g)`` makes the constraint
non-differentiable exactly on the budget sphere, which is where the
optimum lives when ``gain_limit`` is tight (e.g. ``gain_limit = 1``);
SLSQP then stalls in the line search and reports a failure even though a
good feasible point is at hand.

The baseline passthrough FIR ``[1, 0, ...]`` is always feasible because
``gain_limit >= 1``.  Candidate coefficients are projected back onto the
budget if the solver overshoots, and the per-seat errors reported to the
caller are always recomputed from the *returned* coefficients with a
full-length convolution.  Because SLSQP may stop a few ulps outside an
active non-regression constraint, that projection is followed by a
gradient repair (falling back to a bisection towards the passthrough) so
the returned point really satisfies ``mse_k(g) <= mse_k(passthrough)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import convolve

FIR_TAPS = 32

STATUS_IMPROVED = "improved"
STATUS_NO_IMPROVEMENT = "no_improvement"
STATUS_FAILED = "failed"


@dataclass
class SolveResult:
    status: str
    detail: str
    coefficients: list[float] | None
    l1_norm: float | None
    errors_before: list[float]
    errors_after: list[float] | None
    worst_before: float
    worst_after: float | None
    solver_success: bool | None


def _convolution_matrix(h: np.ndarray, taps: int) -> np.ndarray:
    """Dense matrix ``H`` such that ``H @ g == convolve(h, g, mode="full")``."""
    n_out = h.size + taps - 1
    mat = np.zeros((n_out, taps))
    for col in range(taps):
        mat[col:col + h.size, col] = h
    return mat


def full_length_mse(h, g, target_delay: int) -> float:
    """MSE between the full convolution ``h * g`` and the delayed unit impulse."""
    y = convolve(np.asarray(h, dtype=np.float64), np.asarray(g, dtype=np.float64), mode="full")
    target = np.zeros(y.size)
    target[target_delay] = 1.0
    residual = y - target
    return float(residual @ residual) / residual.size


def _budget_project(g: np.ndarray, gain_limit: float) -> np.ndarray:
    """Scale ``g`` onto the L1 budget, guaranteeing ``sum(abs(g)) <= gain_limit``.

    The scaling is only applied when the solver overshoots.  A final
    ``(1 - 1e-12)`` nudge keeps the re-evaluated L1 norm from rounding
    back above the limit, so the returned coefficients are always
    loadable within the requested budget.
    """
    l1 = float(np.abs(g).sum())
    if l1 <= gain_limit:
        return g
    if l1 > 0.0 and np.isfinite(l1):
        g = g * (gain_limit / l1)
        l1 = float(np.abs(g).sum())
        if l1 > gain_limit:  # guard against floating-point rounding
            g = g * (1.0 - 1e-12)
    return g


def _enforce_non_regression(
    hs: list[np.ndarray],
    mats: list[np.ndarray],
    target: np.ndarray,
    scale: float,
    errors_before: np.ndarray,
    gain_limit: float,
    target_delay: int,
    g: np.ndarray,
    max_rounds: int = 12,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Repair ``g`` so every seat is no worse than its passthrough.

    SLSQP can leave a candidate a few ulps *above* the passthrough error
    on a seat whose constraint is active (which is typical when the L1
    budget is tight).  Because the feasible set is convex and the
    passthrough FIR is feasible, small corrective steps always exist.
    We take a damped gradient step on the worst-offending seat, reproject
    onto the L1 budget, and repeat.  If the gradient steps stall at
    round-off level (e.g. several seats tie exactly), we bisect along the
    straight segment towards the passthrough FIR, which is exactly
    feasible, and return the least-blended feasible point on it.

    Returns ``(coefficients, full_convolution_errors, feasible)`` where
    ``feasible`` means the returned errors are at or below the baseline
    for every seat.
    """
    g = np.array(g, dtype=np.float64, copy=True)
    errors = np.array([full_length_mse(h, g, target_delay) for h in hs])
    for _ in range(max_rounds):
        excess = errors - errors_before
        if np.all(excess <= 0.0):
            return g, errors, True
        # Steepest-descent step on the worst-offending seat only.
        k = int(np.argmax(excess))
        grad = 2.0 * scale * (mats[k].T @ (mats[k] @ g - target))
        denom = float(grad @ grad)
        if denom > 0.0:
            # Take 1.25x the first-order step: the excess is a convex
            # quadratic in g, so a slight overshoot lands safely below the
            # target instead of a floating-point hair above it.
            step = 1.25 * float(excess[k]) / denom
            g = _budget_project(g - step * grad, gain_limit)
        else:
            break
        errors = np.array([full_length_mse(h, g, target_delay) for h in hs])
    if np.all(errors <= errors_before):
        return g, errors, True

    # Fallback: bisect the segment towards the always-feasible passthrough.
    baseline = np.zeros_like(g)
    baseline[0] = 1.0

    def segment_feasible(t: float) -> tuple[bool, np.ndarray]:
        blended = (1.0 - t) * g + t * baseline
        errs = np.array([full_length_mse(h, blended, target_delay) for h in hs])
        return bool(np.all(errs <= errors_before)), errs

    # Coarse geometric scan for the first feasible blend, then bisect.
    t_lo, t_hi = 0.0, 1.0
    best = (baseline.copy(), errors_before.copy())
    t = 1e-12
    while t < 1.0:
        feasible, errs = segment_feasible(t)
        if feasible:
            t_hi = t
            best = ((1.0 - t) * g + t * baseline, errs)
            break
        t_lo = t
        t *= 10.0
    if t_hi < 1.0:
        for _ in range(30):
            mid = 0.5 * (t_lo + t_hi)
            feasible, errs = segment_feasible(mid)
            if feasible:
                t_hi = mid
                best = ((1.0 - mid) * g + mid * baseline, errs)
            else:
                t_lo = mid
    return best[0], best[1], bool(np.all(best[1] <= errors_before))


def solve_shared_fir(
    impulse_responses: list[list[float]],
    target_delay: int,
    gain_limit: float,
    taps: int = FIR_TAPS,
) -> SolveResult:
    """Optimise one shared causal FIR for all seats.

    Returns a :class:`SolveResult` whose ``status`` is one of
    ``"improved"`` / ``"no_improvement"`` / ``"failed"``.  Coefficients
    and per-seat before/after errors are only populated when the worst
    seat error was strictly reduced.
    """
    hs = [np.asarray(ir, dtype=np.float64) for ir in impulse_responses]
    n_seats = len(hs)
    n_out = hs[0].size + taps - 1

    mats = [_convolution_matrix(h, taps) for h in hs]
    target = np.zeros(n_out)
    target[target_delay] = 1.0
    scale = 1.0 / n_out

    baseline = np.zeros(taps)
    baseline[0] = 1.0

    # Baselines and final verifications use the same full-length
    # convolution routine, independent of the optimiser's matrix form.
    errors_before = np.array([
        full_length_mse(h, baseline, target_delay) for h in hs
    ])
    worst_before = float(errors_before.max())

    def seat_errors(g: np.ndarray) -> np.ndarray:
        return np.array([scale * float(np.sum((m @ g - target) ** 2)) for m in mats])

    def _result(
        status: str,
        detail: str,
        coefficients: np.ndarray | None = None,
        l1_norm: float | None = None,
        errors_after: np.ndarray | None = None,
        worst_after: float | None = None,
        solver_success: bool | None = None,
    ) -> SolveResult:
        return SolveResult(
            status=status,
            detail=detail,
            coefficients=None if coefficients is None else [float(c) for c in coefficients],
            l1_norm=l1_norm,
            errors_before=[float(e) for e in errors_before],
            errors_after=None if errors_after is None else [float(e) for e in errors_after],
            worst_before=worst_before,
            worst_after=worst_after,
            solver_success=solver_success,
        )

    if worst_before <= 1e-15:
        return _result(
            STATUS_NO_IMPROVEMENT,
            "passthrough already matches the delayed unit impulse (worst MSE ~ 0); nothing to improve",
        )

    # ----- convex epigraph formulation -------------------------------------
    # variables x = [g_plus (taps), g_minus (taps), t]; g = g_plus - g_minus
    # minimise t  subject to
    #   e_k(g) <= t                 for every seat k   (minimax objective)
    #   e_k(g) <= e_k(baseline)     for every seat k   (no seat gets worse)
    #   sum(g_plus + g_minus) <= gain_limit            (exact L1 budget)
    #
    # The box bounds g_plus, g_minus >= 0 make the last row exactly
    # sum(abs(g)), and being linear it is differentiable everywhere.

    n_vars = 2 * taps + 1
    obj_index = n_vars - 1

    def objective(x: np.ndarray) -> float:
        return float(x[obj_index])

    def objective_jac(x: np.ndarray) -> np.ndarray:
        grad = np.zeros(n_vars)
        grad[obj_index] = 1.0
        return grad

    def unpack(x: np.ndarray) -> np.ndarray:
        return x[:taps] - x[taps:2 * taps]

    def constraints(x: np.ndarray) -> np.ndarray:
        g = unpack(x)
        t = x[obj_index]
        errs = seat_errors(g)
        return np.concatenate((
            t - errs,
            errors_before - errs,
            [gain_limit - float(np.sum(x[:2 * taps]))],
        ))

    def constraints_jac(x: np.ndarray) -> np.ndarray:
        g = unpack(x)
        jac = np.zeros((2 * n_seats + 1, n_vars))
        for i, m in enumerate(mats):
            grad_e = 2.0 * scale * (m.T @ (m @ g - target))
            # d/dg_plus = +grad_e, d/dg_minus = -grad_e
            jac[i, :taps] = -grad_e
            jac[i, taps:2 * taps] = grad_e
            jac[i, obj_index] = 1.0
            jac[n_seats + i, :taps] = -grad_e
            jac[n_seats + i, taps:2 * taps] = grad_e
        jac[2 * n_seats, :taps] = -1.0
        jac[2 * n_seats, taps:2 * taps] = -1.0
        return jac

    x0 = np.zeros(n_vars)
    x0[0] = 1.0  # g_plus[0] = 1 -> passthrough, feasible and within budget
    x0[obj_index] = worst_before
    bounds = [(0.0, None)] * (2 * taps) + [(0.0, None)]

    def run_slsqp(maxiter: int, ftol: float):
        # The SLSQP inequality system is dense, so its per-iteration cost
        # grows with the number of constraints; this 65-variable problem
        # needs only a few dozen iterations in practice.
        return minimize(
            objective,
            x0,
            jac=objective_jac,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraints, "jac": constraints_jac}],
            options={"maxiter": maxiter, "ftol": ftol},
        )

    # A tight ftol is preferred; if the line search stalls anyway, a
    # slightly looser tolerance plus more iterations gives SLSQP room to
    # finish.  Both attempts are always re-verified below.
    attempts = ((200, 1e-12), (800, 1e-9))

    improvement_tol = 1e-9 * max(1.0, abs(worst_before))
    # Round-off slack for the "no seat worse than passthrough" check.  It
    # is relative to each seat's baseline (falling back to the worst
    # baseline when that seat's own baseline is zero), so a seat that is
    # already perfect may move only at floating-point noise level while a
    # genuinely worse seat is always rejected.
    nreg_tol = 1e-12 * np.maximum(np.abs(errors_before), abs(worst_before))
    checked: tuple[bool, bool, bool] | None = None  # (has_coeffs, feasible, improved)
    converged_but_infeasible = False
    res = None
    solver_success = False
    coefficients: np.ndarray | None = None
    errors_after: np.ndarray | None = None
    l1: float | None = None
    worst_after: float | None = None

    for maxiter, ftol in attempts:
        try:
            res = run_slsqp(maxiter, ftol)
        except Exception as exc:  # pragma: no cover - exercised via monkeypatched test
            return _result(STATUS_FAILED, f"optimizer raised an exception: {exc!r}")

        solver_success = bool(res.success)
        if res.x is None or not np.all(np.isfinite(res.x)) or res.x.size != n_vars:
            checked = None
            continue

        # The solver may leave a tiny overshoot on the linear budget row;
        # project first, then judge the *actual* coefficients we would return.
        candidate = _budget_project(unpack(res.x).astype(np.float64), gain_limit)
        candidate_l1 = float(np.abs(candidate).sum())
        if candidate_l1 > gain_limit:  # pragma: no cover - defensive
            checked = None
            continue

        # Recompute from the final coefficients with a full-length
        # convolution (not with the optimiser's matrix) before returning.
        # A tiny active-constraint overshoot is repaired first; every
        # coefficient set reported to the caller is checked against the
        # passthrough errors of the *same* full-convolution routine.
        candidate, candidate_errors, _ = _enforce_non_regression(
            hs, mats, target, scale, errors_before, gain_limit, target_delay, candidate
        )
        candidate_l1 = float(np.abs(candidate).sum())
        if candidate_l1 > gain_limit:  # pragma: no cover - defensive
            checked = None
            continue
        candidate_worst = float(candidate_errors.max())
        # A seat whose passthrough error is already ~0 cannot be beaten;
        # the return rule is "no seat worse" (within numerical tolerance)
        # and "worst error strictly lower".
        no_regression = bool(np.all(candidate_errors <= errors_before + nreg_tol))
        improved = no_regression and candidate_worst < worst_before - improvement_tol

        if improved or (solver_success and no_regression):
            # A verified improved point, or a converged feasible point that
            # does not strictly improve: both are final answers.
            coefficients = candidate
            errors_after = candidate_errors
            l1 = candidate_l1
            worst_after = candidate_worst
            checked = (True, no_regression, improved)
            break

        # Non-converged or infeasible iterate: remember it and retry with a
        # looser tolerance before giving up.
        if solver_success:
            converged_but_infeasible = True
        checked = (False, no_regression, improved)

    if checked is not None and checked[0]:
        if checked[2]:
            detail = "worst-seat MSE reduced vs. passthrough"
            if not solver_success:
                detail += (
                    f" (solver stopped early: {res.message}; "
                    "returned point re-verified with the full convolution)"
                )
            return _result(
                STATUS_IMPROVED,
                detail,
                coefficients=coefficients,
                l1_norm=l1,
                errors_after=errors_after,
                worst_after=worst_after,
                solver_success=solver_success,
            )
        if checked[1]:
            return _result(
                STATUS_NO_IMPROVEMENT,
                "no feasible 32-tap FIR within the gain limit lowers the worst-seat MSE",
                solver_success=True,
            )
        return _result(
            STATUS_FAILED,
            "optimizer converged but the returned coefficients violate the "
            "per-seat non-regression constraints (full-convolution check)",
            solver_success=True,
        )

    if res is None:  # pragma: no cover - defensive
        return _result(STATUS_FAILED, "optimizer produced no result", solver_success=False)
    if converged_but_infeasible:
        return _result(
            STATUS_FAILED,
            "optimizer converged but no iterate satisfied the per-seat "
            "non-regression constraints after full-convolution verification",
            solver_success=True,
        )
    if not solver_success:
        return _result(
            STATUS_FAILED,
            f"optimizer did not converge: {res.message}",
            solver_success=False,
        )
    return _result(
        STATUS_FAILED,
        "optimizer returned a non-finite iterate",
        solver_success=True,
    )
