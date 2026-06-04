"""SmartAnalyze: Robust OpenSees analysis with automatic retry strategies."""

from __future__ import annotations

import logging
import os
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from rich import print as rprint
from tqdm import tqdm
from typing_extensions import Literal, TypedDict, Unpack

from ..utils import get_opensees_module, get_random_color, on_notebook

if TYPE_CHECKING:
    from collections.abc import Generator

ops = get_opensees_module()

LOG_FILE = ".SmartAnalyze-OpenSees.log"
ON_NOTEBOOK = on_notebook()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def suppress_ops_print(verbose: bool = False) -> Generator[None, None, None]:
    """Suppress OpenSees console output by redirecting to a log file.

    Parameters
    ----------
    verbose:
        If ``True``, output is *not* suppressed.
    """
    if verbose:
        yield
        return

    redirected = False
    try:
        ops.logFile(LOG_FILE, "-noEcho")
        redirected = True
    except Exception:
        logger.warning("Failed to redirect OpenSees log to %s", LOG_FILE)

    try:
        yield
    finally:
        if redirected:
            try:
                ops.logFile(os.devnull)
            except Exception:
                logger.warning("Failed to reset OpenSees log output")


# ---------------------------------------------------------------------------
# TypedDict for keyword arguments accepted by SmartAnalyze.__init__
# ---------------------------------------------------------------------------


class _KargsTypes(TypedDict, total=False):
    testType: Literal[
        "EnergyIncr",
        "NormDispIncr",
        "NormUnbalance",
        "RelativeNormUnbalance",
        "RelativeNormDispIncr",
        "RelativeTotalNormDispIncr",
        "RelativeEnergyIncr",
        "FixedNumIter",
    ]
    testTol: float
    testIterTimes: int
    testPrintFlag: int
    tryAddTestTimes: bool
    normTol: float
    testIterTimesMore: int | list[int]
    tryLooseTestTol: bool
    looseTestTolTo: float | list[float]
    tryAlterTestTypes: bool
    testTypesMore: list[str] | tuple[str, ...]
    tryAlterAlgoTypes: bool
    algoTypes: list[int]
    tryRelaxStep: bool
    UserAlgoArgs: list | None
    initialStep: float | None
    relaxation: float
    minStep: float
    debugMode: bool
    printPer: int
    recordNormHistory: bool
    recordDiagnostics: bool


# ---------------------------------------------------------------------------
# Lightweight data structures for history / diagnostics
# ---------------------------------------------------------------------------


@dataclass
class NormTrend:
    """Convergence-norm trend for a single analysis attempt."""

    is_empty: bool = True
    has_nan: bool = False
    has_inf: bool = False
    has_nonfinite: bool = False
    first: float = np.nan
    last: float = np.nan
    min: float = np.nan
    max: float = np.nan
    improvement_ratio: float = np.nan
    is_improving: bool = False
    is_stagnating: bool = False
    is_diverging: bool = False


@dataclass
class NormHistoryEntry:
    """One entry in the lightweight norm-history log."""

    step_index: int = 0
    strategy: str = ""
    ok: int = -1
    first_norm: float = np.nan
    last_norm: float = np.nan
    min_norm: float = np.nan
    max_norm: float = np.nan
    num_iter: int = 0
    improvement_ratio: float = np.nan
    is_diverging: bool = False
    is_stagnating: bool = False


@dataclass
class DiagnosticRecord:
    """Detailed diagnostic record for a single failed attempt."""

    timestamp: str = ""
    step_index: int = 0
    analysis: str = ""
    strategy: str = ""
    step: float = 0.0
    ok: int = -1
    test_type: str = ""
    test_tol: float = 0.0
    test_iter_times: int = 0
    test_print_flag: int = 0
    algorithm_type: int = -1
    algorithm_args: list = field(default_factory=list)
    algorithm_text: str = ""
    norms: list = field(default_factory=list)
    trend: NormTrend | None = None
    reason: str = ""
    partial_advance: bool = False
    partial_advance_info: dict | None = None
    suggestion: str = ""


# ---------------------------------------------------------------------------
# Algorithm lookup table (module-level constant, not rebuilt per call)
# ---------------------------------------------------------------------------

_ALGO_MAP: dict[int, list[Any]] = {
    0: ["Linear"],
    1: ["Linear", "-Initial"],
    2: ["Linear", "-Secant"],
    3: ["Linear", "-FactorOnce"],
    4: ["Linear", "-Initial", "-FactorOnce"],
    5: ["Linear", "-Secant", "-FactorOnce"],
    10: ["Newton"],
    11: ["Newton", "-Initial"],
    12: ["Newton", "-initialThenCurrent"],
    13: ["Newton", "-Secant"],
    20: ["NewtonLineSearch"],
    21: ["NewtonLineSearch", "-type", "Bisection"],
    22: ["NewtonLineSearch", "-type", "Secant"],
    23: ["NewtonLineSearch", "-type", "RegulaFalsi"],
    24: ["NewtonLineSearch", "-type", "LinearInterpolated"],
    25: ["NewtonLineSearch", "-type", "InitialInterpolated"],
    30: ["ModifiedNewton"],
    31: ["ModifiedNewton", "-initial"],
    32: ["ModifiedNewton", "-secant"],
    40: ["KrylovNewton"],
    41: ["KrylovNewton", "-iterate", "initial"],
    42: ["KrylovNewton", "-increment", "initial"],
    43: ["KrylovNewton", "-iterate", "initial", "-increment", "initial"],
    44: ["KrylovNewton", "-maxDim", 10],
    45: ["KrylovNewton", "-iterate", "initial", "-increment", "initial", "-maxDim", 10],
    50: ["SecantNewton"],
    51: ["SecantNewton", "-iterate", "initial"],
    52: ["SecantNewton", "-increment", "initial"],
    53: ["SecantNewton", "-iterate", "initial", "-increment", "initial"],
    60: ["BFGS"],
    61: ["BFGS", "-initial"],
    62: ["BFGS", "-secant"],
    70: ["Broyden"],
    71: ["Broyden", "-initial"],
    72: ["Broyden", "-secant"],
    80: ["PeriodicNewton"],
    81: ["PeriodicNewton", "-maxDim", 10],
    90: ["ExpressNewton"],
    91: ["ExpressNewton", "-InitialTangent"],
}

_VALID_SENSITIVITY_ALGORITHMS = frozenset({"-computeAtEachStep", "-computeByCommand"})
_VALID_ANALYSIS_TYPES = frozenset({"Transient", "Static"})


# ---------------------------------------------------------------------------
# SmartAnalyze
# ---------------------------------------------------------------------------


class SmartAnalyze:
    """Robust OpenSees analysis with automatic retry strategies.

    SmartAnalyze stores analysis configuration, progress information, and
    retry strategies for transient and displacement-control static analyses.

    Retry order
    -----------
    Each failed analysis step is retried in this order:

    1. Add convergence-test iteration limits, if *tryAddTestTimes* is ``True``.
    2. Try alternate algorithm types, if *tryAlterAlgoTypes* is ``True``.
    3. Try alternate convergence-test types, if *tryAlterTestTypes* is ``True``.
    4. Loosen the test tolerance, if *tryLooseTestTol* is ``True``.
    5. Split the current step using relaxation until *minStep* is reached,
       if *tryRelaxStep* is ``True``.

    Step relaxation is intentionally tried last because successful substeps
    can partially advance the OpenSees model state. Keeping it last avoids
    applying other whole-step retry strategies after the model has already
    moved through part of the requested step.

    Parameters
    ----------
    analysis_type : {"Transient", "Static"}, default "Transient"
        Analysis type.

    Keyword Arguments
    -----------------
    testType : str, default "EnergyIncr"
        OpenSees convergence-test type.
    testTol : float, default 1e-10
        Convergence-test tolerance.
    testIterTimes : int, default 10
        Default maximum number of convergence-test iterations.
    testPrintFlag : int, default 0
        OpenSees convergence-test print flag.

    tryAddTestTimes : bool, default False
        If ``True``, retry failed steps with larger iteration limits from
        *testIterTimesMore* when the latest norm is less than *normTol*.
    normTol : float, default 1e3
        Maximum latest test norm that allows *tryAddTestTimes* retries.
    testIterTimesMore : int or list[int], default [50]
        Additional convergence-test iteration limits to try.

    tryLooseTestTol : bool, default False
        If ``True``, retry failed steps with *looseTestTolTo* after the other
        retry strategies fail.
    looseTestTolTo : float or list[float], default 100 * testTol
        Looser convergence-test tolerance(s) used by *tryLooseTestTol*.

    tryAlterTestTypes : bool, default False
        If ``True``, retry failed steps with alternate convergence-test
        types from *testTypesMore*.
    testTypesMore : list[str], default ["NormDispIncr", "NormUnbalance", "RelativeEnergyIncr"]
        Alternate convergence-test types to try.

    tryAlterAlgoTypes : bool, default False
        If ``True``, retry failed steps with the remaining entries in
        *algoTypes*.
    algoTypes : list[int], default [40, 10, 20, 30, 50, 60, 70, 90]
        Algorithm type codes. The first entry is applied during construction;
        subsequent entries are used as fallback algorithms.
    tryRelaxStep : bool, default False
        If ``True``, split a failed step using *relaxation* down to *minStep*.
    UserAlgoArgs : list, optional
        User-defined arguments passed to ``ops.algorithm`` when *algoTypes*
        includes 100.

    initialStep : float, optional
        Initial analysis step stored in the configuration.
    relaxation : float, default 0.5
        Factor used to split a failed step during step relaxation.
    minStep : float, default 1e-6
        Minimum absolute substep allowed during step relaxation.

    debugMode : bool, default False
        Whether to print retry and progress messages.
    printPer : int, default 20
        Print progress every *printPer* successful steps when *debugMode* is
        ``True``.

    recordNormHistory : bool, default True
        If ``True``, record a summary of convergence norms for each analysis
        attempt.
    recordDiagnostics : bool, default False
        If ``True``, record detailed failure diagnostics.

    Examples
    --------
    **Transient analysis with full retry strategies**

    >>> import opstool as opst
    >>> import openseespy.opensees as ops
    >>>
    >>> ops.constraints("Transformation")
    >>> ops.numberer("Plain")
    >>> ops.system("BandGeneral")
    >>> ops.integrator("Newmark", 0.5, 0.25)
    >>> analysis = opst.anlys.SmartAnalyze(
    ...     analysis_type="Transient",
    ...     testType="EnergyIncr",
    ...     testTol=1e-10,
    ...     testIterTimes=10,
    ...     tryAddTestTimes=True,
    ...     normTol=1e3,
    ...     testIterTimesMore=[50, 100],
    ...     tryLooseTestTol=True,
    ...     looseTestTolTo=[1e-8, 1e-6],
    ...     tryAlterTestTypes=True,
    ...     testTypesMore=["NormDispIncr", "NormUnbalance"],
    ...     tryAlterAlgoTypes=True,
    ...     algoTypes=[40, 10, 20, 30],
    ...     tryRelaxStep=True,
    ...     relaxation=0.5,
    ...     minStep=1e-6,
    ...     debugMode=True,
    ...     printPer=100,
    ... )
    >>> npts, dt = 1000, 0.01
    >>> analysis.set_total_steps(npts)  # set total steps for progress tracking
    >>> for _ in range(npts):
    ...     ok = analysis.TransientAnalyze(dt)
    ...     if ok < 0:
    ...         raise RuntimeError("Transient analysis failed.")

    **Static pushover analysis (minimal settings)**

    >>> analysis = opst.anlys.SmartAnalyze(
    ...     analysis_type="Static",
    ...     testType="EnergyIncr",
    ...     testTol=1e-8,
    ...     tryAlterAlgoTypes=True,
    ...     algoTypes=[40, 10, 20],
    ...     tryRelaxStep=True,
    ...     relaxation=0.5,
    ...     minStep=1e-6,
    ... )
    >>> targets = [0, 0.5, 1.0]
    >>> segs = analysis.static_split(targets, 0.01)
    >>> for seg in segs:
    ...     ok = analysis.StaticAnalyze(node=1, dof=1, seg=seg)
    ...     if ok < 0:
    ...         raise RuntimeError("Static analysis failed.")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        analysis_type: Literal["Transient", "Static"] = "Transient",
        **kargs: Unpack[_KargsTypes],
    ) -> None:
        if analysis_type not in _VALID_ANALYSIS_TYPES:
            raise ValueError('analysis_type must be "Transient" or "Static".')  # noqa: TRY003

        # ---- default control arguments ---------------------------------
        self.control_args: dict[str, Any] = {
            "analysis": analysis_type,
            "testType": "EnergyIncr",
            "testTol": 1.0e-10,
            "testIterTimes": 10,
            "testPrintFlag": 0,
            "tryAddTestTimes": False,
            "normTol": 1.0e3,
            "testIterTimesMore": [50],
            "tryLooseTestTol": False,
            "looseTestTolTo": 1.0e-8,
            "tryAlterTestTypes": False,
            "testTypesMore": ["NormDispIncr", "NormUnbalance", "RelativeEnergyIncr"],
            "tryAlterAlgoTypes": False,
            "algoTypes": [40, 10],
            "tryRelaxStep": False,
            "UserAlgoArgs": None,
            "initialStep": None,
            "relaxation": 0.5,
            "minStep": 1.0e-6,
            "debugMode": False,
            "printPer": 20,
            "recordNormHistory": True,
            "recordDiagnostics": False,
        }
        # Default looseTestTolTo is derived from testTol; override after
        # the dict is populated so the multiplier uses the default testTol.
        self.control_args["looseTestTolTo"] = 100 * self.control_args["testTol"]

        # Validate and merge user-supplied keyword arguments.
        valid_keys = set(self.control_args.keys())
        unknown = sorted(set(kargs) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown argument(s): {unknown}. Valid arguments are: {sorted(valid_keys)}")  # noqa: TRY003
        self.control_args.update(kargs)

        # Normalise list-like config entries to consistent types.
        self._normalize_list_args()

        self.analysis_type = analysis_type
        # Floating-point epsilon used for near-zero comparisons.
        self.eps = 1.0e-12

        # Console prefix strings (notebook vs terminal).
        if ON_NOTEBOOK:
            self.logo = "OPSTOOL::SmartAnalyze::"
        else:
            self.logo = "[bold magenta]OPSTOOL::SmartAnalyze::[/bold magenta]"
        self.logo_progress = "\033[95mOPSTOOL::SmartAnalyze\033[0m"

        self.debug_mode = bool(self.control_args["debugMode"])

        # Optional sensitivity algorithm tag; set via set_sensitivity_algorithm().
        self.sensitivity_algorithm: str | None = None

        # Runtime progress and state counters.
        self.current_args: dict[str, Any] = {
            "startTime": time.time(),
            "counter": 0,
            "progress": 0,
            "npts": 0,
            "step": 0.0,
            "node": 0,
            "dof": 0,
        }
        self.progress: tqdm | None = None

        # Norm history: last_norms holds the norm list from the most recent
        # _analyze_one_step call; norm_history accumulates summary entries.
        self.last_norms: list[float] = []
        self.norm_history: list[NormHistoryEntry] = []
        self.diagnostics: dict[str, Any] = {
            "records": [],
            "lastFailure": None,
            "maxRecords": 200,
        }

        # Mirror of the OpenSees test/algorithm state for diagnostic records.
        self.current_test_type: str = self.control_args["testType"]
        self.current_test_tol: float = self.control_args["testTol"]
        self.current_test_iter_times: int = self.control_args["testIterTimes"]
        self.current_test_print_flag: int = self.control_args["testPrintFlag"]
        self.current_algorithm_type: int = self.control_args["algoTypes"][0]
        self.current_algorithm_args: list = []

        # Apply the initial convergence test and algorithm to OpenSees.
        self._set_init_test()
        self._set_algorithm(
            self.control_args["algoTypes"][0],
            self.control_args["UserAlgoArgs"],
            verbose=self.debug_mode,
        )

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize_list_args(self) -> None:
        """Ensure list-like config entries are well-formed."""
        # testIterTimesMore -> list of positive ints
        tim = self.control_args["testIterTimesMore"]
        if isinstance(tim, (int, float)):
            tim = [int(tim)]
        tim = [int(x) for x in tim if np.isfinite(x) and x > 0]
        self.control_args["testIterTimesMore"] = tim if tim else [50]

        # looseTestTolTo -> list of positive floats
        tol = self.control_args["looseTestTolTo"]
        tol = [float(tol)] if isinstance(tol, (int, float)) else [float(x) for x in tol]
        tol = [x for x in tol if np.isfinite(x) and x > 0]
        self.control_args["looseTestTolTo"] = tol if tol else [100 * self.control_args["testTol"]]

        # testTypesMore -> list of non-empty strings
        ttm = self.control_args["testTypesMore"]
        if isinstance(ttm, str):
            ttm = [ttm]
        self.control_args["testTypesMore"] = [str(x) for x in ttm if x]

        # algoTypes -> list of ints
        algo = self.control_args["algoTypes"]
        algo = [int(x) for x in algo if np.isfinite(x)]
        self.control_args["algoTypes"] = algo if algo else [40, 10, 20, 30, 50, 60, 70, 90]

    # ------------------------------------------------------------------
    # Progress bar helpers
    # ------------------------------------------------------------------

    def _set_progress_bar(self, npts: int) -> None:
        self.progress = tqdm(
            total=npts,
            desc=f"🚀 {self.logo_progress}",
            colour="#5170d7",
            unit=" step",
        )

    def _stop_progress_bar(self) -> None:
        if self.progress is not None:
            self.progress.total = self.progress.n
            self.progress.refresh()
            self.progress.close()
            print(f"Note: OpenSees LogFile has been generated in {LOG_FILE}.")
        self.progress = None

    # ------------------------------------------------------------------
    # Step splitting
    # ------------------------------------------------------------------

    def transient_split(self, npts: int) -> list[int]:
        """Split a transient analysis into step indices for progress tracking.

        Parameters
        ----------
        npts : int
            Number of transient analysis steps.

        Returns
        -------
        list[int]
            List ``range(1, npts + 1)`` suitable for direct looping.
        """
        self.current_args["npts"] = npts
        if not self.debug_mode and self.progress is None:
            self._set_progress_bar(npts)
        return list(range(1, npts + 1))

    def set_total_steps(self, npts: int) -> None:
        """Set the total number of analysis steps for progress tracking.

        Parameters
        ----------
        npts : int
            Total number of expected analysis steps.
        """
        if npts < 0:
            raise ValueError("npts must be non-negative.")  # noqa: TRY003
        self.current_args["npts"] = int(npts)
        self.current_args["progress"] = 0
        self.current_args["counter"] = 0
        self.current_args["startTime"] = time.time()
        if not self.debug_mode and self.progress is None:
            self._set_progress_bar(self.current_args["npts"])

    def static_split(
        self,
        targets: list | tuple | np.ndarray | float,
        maxStep: float | None = None,
    ) -> list[float]:
        """Split displacement-control target values into bounded static steps.

        Parameters
        ----------
        targets : array-like
            Target displacement values. If a scalar is given, it is treated as
            ``[0, target]``.
        maxStep : float, optional
            Maximum absolute displacement increment for a generated segment.
            If omitted, the absolute difference between the first two targets
            is used.

        Returns
        -------
        list[float]
            Displacement increments. Each generated segment has
            ``abs(seg) <= maxStep``.
        """
        targets = np.atleast_1d(targets).astype(float)
        if targets.ndim != 1:
            raise ValueError("targets must be 1-D.")  # noqa: TRY003
        if targets.size == 0:
            raise ValueError("targets must not be empty.")  # noqa: TRY003
        if targets.size == 1:
            targets = np.array([0.0, targets[0]])

        maxStep = abs(targets[1] - targets[0]) if maxStep is None else abs(float(maxStep))

        if maxStep <= self.eps:
            raise ValueError("maxStep must be positive.")  # noqa: TRY003

        segs: list[float] = []
        for start, end in zip(targets[:-1], targets[1:]):
            delta = end - start
            if abs(delta) < self.eps:
                continue
            direction = float(np.sign(delta))
            num_full = int(abs(delta) // maxStep)
            remainder = abs(delta) - num_full * maxStep
            segs.extend([direction * maxStep] * num_full)
            if remainder > self.eps:
                segs.append(direction * remainder)

        self.current_args["npts"] = len(segs)
        if not self.debug_mode and self.progress is None:
            self._set_progress_bar(len(segs))
        return segs

    # ------------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------------

    def set_sensitivity_algorithm(self, algorithm: str = "-computeAtEachStep") -> None:
        """Set the OpenSees sensitivity algorithm.

        Parameters
        ----------
        algorithm : {"-computeAtEachStep", "-computeByCommand"}
            Sensitivity algorithm option.
        """
        if algorithm not in _VALID_SENSITIVITY_ALGORITHMS:
            raise ValueError("algorithm must be '-computeAtEachStep' or '-computeByCommand'.")  # noqa: TRY003
        self.sensitivity_algorithm = algorithm

    def _run_sensitivity_algorithm(self) -> None:
        if self.sensitivity_algorithm is not None:
            ops.sensitivityAlgorithm(self.sensitivity_algorithm)

    # ------------------------------------------------------------------
    # Public analysis entry points
    # ------------------------------------------------------------------

    def TransientAnalyze(self, dt: float) -> int:
        """Run one transient analysis step with automatic retry strategies.

        Parameters
        ----------
        dt : float
            Time-step size passed to ``ops.analyze(1, dt)``.

        Returns
        -------
        int
            ``0`` if the step succeeds. A negative value means all enabled
            retry strategies failed.
        """
        if self.control_args["analysis"] != "Transient":
            raise ValueError('Current analysis type is not "Transient". Please check the analysis_type parameter.')  # noqa: TRY003
        self.control_args["initialStep"] = dt
        ops.analysis(self.control_args["analysis"])
        return self._analyze()

    def StaticAnalyze(self, node: int, dof: int, seg: float) -> int:
        """Run one displacement-control static analysis step with retries.

        Parameters
        ----------
        node : int
            Node tag for the OpenSees ``DisplacementControl`` integrator.
        dof : int
            Degree of freedom for the ``DisplacementControl`` integrator.
        seg : float
            Displacement increment for this segment.

        Returns
        -------
        int
            ``0`` if the segment succeeds. A negative value means all enabled
            retry strategies failed.
        """
        if self.control_args["analysis"] != "Static":
            raise ValueError('Current analysis type is not "Static". Please check the analysis_type parameter.')  # noqa: TRY003
        self.control_args["initialStep"] = seg
        self.current_args["node"] = node
        self.current_args["dof"] = dof
        self.current_args["step"] = seg

        ops.integrator("DisplacementControl", node, dof, seg)
        ops.analysis(self.control_args["analysis"])
        self._run_sensitivity_algorithm()
        return self._analyze()

    # ------------------------------------------------------------------
    # Reset / close
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset progress counters and clear history.

        Progress counters (``progress``, ``counter``, ``npts``) are reset to
        zero.  The OpenSees command interface and sensitivity algorithm are
        preserved.  History and diagnostics are cleared.
        """
        self.current_args["progress"] = 0
        self.current_args["counter"] = 0
        self.current_args["npts"] = 0
        self.current_args["startTime"] = time.time()
        self.last_norms = []
        self.norm_history = []
        self.diagnostics = {
            "records": [],
            "lastFailure": None,
            "maxRecords": 200,
        }
        if self.progress is not None:
            self._stop_progress_bar()

    def close(self) -> None:
        """Close the progress bar and clean up."""
        self._stop_progress_bar()

    # ------------------------------------------------------------------
    # History / diagnostics getters
    # ------------------------------------------------------------------

    def get_norm_history(self) -> list[NormHistoryEntry]:
        """Get the convergence norm history collected during analysis.

        Returns
        -------
        list[NormHistoryEntry]
            One entry per analysis attempt.  Empty if *recordNormHistory* was
            disabled.
        """
        if not self.control_args["recordNormHistory"]:
            warnings.warn("Norm history is not recorded.", UserWarning, stacklevel=2)
        return list(self.norm_history)

    def get_diagnostics(self) -> dict[str, Any]:
        """Get SmartAnalyze retry and convergence diagnostics.

        Returns
        -------
        dict
            Diagnostic records with keys ``records``, ``lastFailure``, and
            ``maxRecords``.  Empty if *recordDiagnostics* was disabled.
        """
        if not self.control_args["recordDiagnostics"]:
            warnings.warn("Diagnostics are not recorded.", UserWarning, stacklevel=2)
        return dict(self.diagnostics)

    def print_last_failure(self) -> None:
        """Print a concise diagnostic report for the most recent failed attempt."""
        lf = self.diagnostics.get("lastFailure")
        if lf is None:
            print(f">>> {self.logo} No failed SmartAnalyze attempt has been recorded.")
            return

        print(f">>>❌ {self.logo} Last failure diagnostic")
        print(f"    Time         : {lf.timestamp}")
        print(f"    Step index   : {lf.step_index}")
        print(f"    Analysis     : {lf.analysis}")
        print(f"    Strategy     : {lf.strategy}")
        print(f"    Step size    : {lf.step:.6e}")
        print(f"    Algorithm    : {lf.algorithm_text}")
        print(
            f"    Test         : {lf.test_type}, tol={lf.test_tol:.3e}, "
            f"iter={lf.test_iter_times}, printFlag={lf.test_print_flag}"
        )
        print(f"    Return code  : {lf.ok}")
        print(f"    Reason       : {lf.reason}")
        if lf.partial_advance and lf.partial_advance_info:
            info = lf.partial_advance_info
            print(
                f"    Partial step : model state may have advanced by "
                f"{info.get('completedStep', 0):.6e} before relaxation failed"
            )
            print(
                f"                   remaining={info.get('remainingStep', 0):.6e}, "
                f"failedSubstep={info.get('failedSubstep', 0):.6e}, "
                f"reason={info.get('reason', '')}"
            )
        if lf.suggestion:
            print(f"    Suggestion   : {lf.suggestion}")

        trend = lf.trend
        if trend is not None and not trend.is_empty:
            print(
                f"    Norm history : first={trend.first:.3e}, last={trend.last:.3e}, "
                f"min={trend.min:.3e}, max={trend.max:.3e}"
            )
            print(
                f"    Norm trend   : improving={trend.is_improving}, "
                f"stagnating={trend.is_stagnating}, diverging={trend.is_diverging}, "
                f"nonfinite={trend.has_nonfinite}"
            )
        else:
            print("    Norm history : unavailable")

    # ------------------------------------------------------------------
    # Internal analysis orchestration
    # ------------------------------------------------------------------

    def _get_time(self) -> float:
        return time.time() - self.current_args["startTime"]

    def _analyze(self) -> int:
        initial_step = self.control_args["initialStep"]
        verbose = bool(self.debug_mode)

        # Initial attempt with the current algorithm and test settings.
        ok = self._analyze_one_step(initial_step, verbose=verbose, strategy="initial")

        # Retry strategies applied in the prescribed order.
        if ok < 0:
            ok = self._try_add_test_times(initial_step, verbose)
        if ok < 0:
            ok = self._try_alter_algo_types(initial_step, verbose)
        if ok < 0:
            ok = self._try_alter_test_types(initial_step, verbose)
        if ok < 0:
            ok = self._try_loose_test_tol(initial_step, verbose)
        if ok < 0 and self.control_args["tryRelaxStep"]:
            ok = self._try_relax_step(initial_step, verbose)

        if ok < 0:
            self._stop_progress_bar()
            self._print_status(success=False)
            # if verbose:
            #     self.print_last_failure()
            return ok

        # Bookkeeping for a successful step.
        self.current_args["progress"] += 1
        self.current_args["counter"] += 1

        if verbose and self.current_args["counter"] >= self.control_args["printPer"]:
            self._print_progress()
            self.current_args["counter"] = 0

        if self.progress is not None:
            self.progress.update(1)

        npts = self.current_args.get("npts", 0)
        if npts > 0 and self.current_args["progress"] >= npts:
            self._stop_progress_bar()
            self._print_status(success=True)

        return 0

    # ------------------------------------------------------------------
    # Single-step execution
    # ------------------------------------------------------------------

    def _analyze_one_step(
        self,
        step: float,
        verbose: bool,
        strategy: str = "analyzeOne",
    ) -> int:
        """Execute one analysis increment and record convergence norm data.

        ``ops.testNorm()`` returns a list whose length equals ``testIterTimes``.
        The first ``testIter`` entries are the per-iteration convergence norms
        produced during this step; the remaining entries are zero-padded
        placeholders.  Sampling is done **after** ``ops.analyze`` so the full
        iteration history for the completed (or failed) step is available.
        """
        if self.analysis_type == "Static":
            ops.integrator(
                "DisplacementControl",
                self.current_args["node"],
                self.current_args["dof"],
                step,
            )
            self._run_sensitivity_algorithm()
            with suppress_ops_print(verbose=verbose):
                ok = ops.analyze(1)
        else:
            with suppress_ops_print(verbose=verbose):
                ok = ops.analyze(1, step)

        self.current_args["step"] = step

        # Retrieve the full per-iteration norm list for this step, stripping
        # the trailing zero-padding that OpenSees appends up to testIterTimes.
        norms = self._test_norms()

        trend = self._norm_trend(norms)
        self._record_attempt(strategy, step, ok, norms, trend)
        return ok

    # ------------------------------------------------------------------
    # Norm sampling helper
    # ------------------------------------------------------------------

    def _test_norms(self) -> list[float]:
        """Return the non-zero per-iteration norms from the last analysis step.

        ``ops.testNorm()`` returns a list of length ``testIterTimes``.
        The first ``testIter`` entries hold the actual convergence norms
        produced during each Newton iteration; the remainder are zero-padded.
        This method strips the trailing zeros so callers see only the real
        iteration history.

        Returns an empty list if the call fails or produces no data.
        """
        try:
            val = ops.testNorm()
            if val is None:
                return []
            arr = np.asarray(val, dtype=float).ravel()
            if arr.size == 0:
                return []
            # Keep only the leading non-zero (and finite) entries.
            # Trailing zeros are padding up to testIterTimes; they carry no
            # information and would bias trend statistics (min, improvement
            # ratio, stagnation detection) toward zero.
            nonzero_mask = arr != 0.0
            last_nonzero = int(np.flatnonzero(nonzero_mask)[-1]) + 1 if nonzero_mask.any() else 0
            meaningful = arr[:last_nonzero]
            # Further filter out any non-finite values (NaN / Inf).
            return [float(v) for v in meaningful if np.isfinite(v)]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Retry strategies
    # ------------------------------------------------------------------

    def _try_add_test_times(self, step: float, verbose: bool) -> int:
        if not self.control_args["tryAddTestTimes"]:
            return -1

        # Only attempt if the last recorded norm is below normTol,
        # indicating that convergence may be achievable with more iterations.
        nrm = self._last_norm()
        if not (np.isfinite(nrm) and nrm < self.control_args["normTol"]):
            if verbose:
                self._print_msg(f"Not adding test times for norm {nrm:.3e}.", icon="✳️")
            return -1

        ok = -1
        for num in self.control_args["testIterTimesMore"]:
            if verbose:
                self._print_msg(f"Adding test times to {num}.", icon="✳️")
            ops.test(
                self.control_args["testType"],
                self.control_args["testTol"],
                num,
                self.control_args["testPrintFlag"],
            )
            self._set_current_test(
                self.control_args["testType"],
                self.control_args["testTol"],
                num,
                self.control_args["testPrintFlag"],
            )
            ok = self._analyze_one_step(step, verbose, strategy=f"tryAddTestTimes:{num}")
            if ok == 0:
                self._set_init_test()
                return ok

        self._set_init_test()
        return ok

    def _try_alter_algo_types(self, step: float, verbose: bool) -> int:
        if not self.control_args["tryAlterAlgoTypes"]:
            return -1
        if len(self.control_args["algoTypes"]) <= 1:
            return -1

        ok = -1
        for algo_flag in self.control_args["algoTypes"][1:]:
            if verbose:
                self._print_msg(f"Setting algorithm to {algo_flag}.", icon="✳️")
            self._set_algorithm(
                algo_flag,
                self.control_args["UserAlgoArgs"],
                verbose=self.debug_mode,
            )
            ok = self._analyze_one_step(step, verbose, strategy=f"tryAlterAlgo:{algo_flag}")
            if ok == 0:
                return ok

        # Restore the primary algorithm after all fallbacks fail.
        self._set_algorithm(
            self.control_args["algoTypes"][0],
            self.control_args["UserAlgoArgs"],
            verbose=self.debug_mode,
        )
        return ok

    def _try_alter_test_types(self, step: float, verbose: bool) -> int:
        if not self.control_args["tryAlterTestTypes"]:
            return -1
        if not self.control_args["testTypesMore"]:
            return -1

        ok = -1
        for tt in self.control_args["testTypesMore"]:
            tt_str = str(tt)
            if verbose:
                self._print_msg(f"Switching test type to {tt_str}.", icon="✳️")
            ops.test(
                tt_str,
                self.control_args["testTol"],
                self.control_args["testIterTimes"],
                self.control_args["testPrintFlag"],
            )
            self._set_current_test(
                tt_str,
                self.control_args["testTol"],
                self.control_args["testIterTimes"],
                self.control_args["testPrintFlag"],
            )
            ok = self._analyze_one_step(step, verbose, strategy=f"tryAlterTestTypes:{tt_str}")
            if ok == 0:
                self._set_init_test()
                return ok

        self._set_init_test()
        return ok

    def _try_loose_test_tol(self, step: float, verbose: bool) -> int:
        if not self.control_args["tryLooseTestTol"]:
            return -1

        ok = -1
        for tol in self.control_args["looseTestTolTo"]:
            if verbose:
                self._print_msg(f"Loosing test tolerance to {tol:.3e}.", icon="✳️")
            ops.test(
                self.control_args["testType"],
                tol,
                self.control_args["testIterTimes"],
                self.control_args["testPrintFlag"],
            )
            self._set_current_test(
                self.control_args["testType"],
                tol,
                self.control_args["testIterTimes"],
                self.control_args["testPrintFlag"],
            )
            ok = self._analyze_one_step(step, verbose, strategy=f"tryLooseTol:{tol:.3e}")
            if ok == 0:
                self._set_init_test()
                return ok

        self._set_init_test()
        return ok

    def _try_relax_step(self, step: float, verbose: bool) -> int:
        alpha = abs(self.control_args["relaxation"])
        min_step = abs(self.control_args["minStep"])

        remain = float(step)
        step_try = step * alpha
        completed = 0.0

        if verbose:
            self._print_msg(
                f"Dividing current step {step:.3e} into {step_try:.3e} and {step - step_try:.3e}.",
                icon="✳️",
            )

        ok = -1
        while abs(remain) > self.eps:
            if abs(step_try) < min_step:
                # Sub-step has shrunk below the minimum allowed step.
                if abs(completed) > self.eps:
                    self._flag_partial_advance(step, completed, remain, step_try, "minStepReached")
                if verbose:
                    self._print_msg(
                        f"Current step {step_try:.3e} is below minStep {min_step:.3e}.",
                        icon="❌",
                    )
                    if abs(completed) > self.eps:
                        self._print_msg(
                            f"Relaxed substeps partially advanced the model by {completed:.3e} "
                            f"before failure. Remaining step: {remain:.3e}.",
                            icon="⚠️",
                        )
                return -1

            if abs(step_try) > abs(remain):
                step_try = remain

            ok = self._analyze_one_step(step_try, verbose, strategy="tryRelaxStep")

            if ok == 0:
                remain -= step_try
                completed = step - remain
                step_try = remain
                if verbose:
                    self._print_msg(
                        f"Total step {step:.3e}, completed {completed:.3e}, remaining {remain:.3e}.",
                        icon="✳️",
                    )
            else:
                # Shrink the sub-step and try again.
                step_try *= alpha
                if verbose:
                    self._print_msg(
                        f"Dividing failed substep into smaller step {step_try:.3e}.",
                        icon="✳️",
                    )
        return ok

    # ------------------------------------------------------------------
    # Test / algorithm state helpers
    # ------------------------------------------------------------------

    def _set_init_test(self) -> None:
        """Apply the default convergence test to OpenSees and update the mirror."""
        ops.test(
            self.control_args["testType"],
            self.control_args["testTol"],
            self.control_args["testIterTimes"],
            self.control_args["testPrintFlag"],
        )
        self._set_current_test(
            self.control_args["testType"],
            self.control_args["testTol"],
            self.control_args["testIterTimes"],
            self.control_args["testPrintFlag"],
        )

    def _set_current_test(
        self,
        test_type: str,
        test_tol: float,
        test_iter: int,
        test_print: int,
    ) -> None:
        """Update the local mirror of the active OpenSees convergence test."""
        self.current_test_type = test_type
        self.current_test_tol = test_tol
        self.current_test_iter_times = test_iter
        self.current_test_print_flag = test_print

    def _set_algorithm(
        self,
        algotype: int,
        user_algo_args: list | None = None,
        verbose: bool = True,
    ) -> None:
        """Apply an algorithm to OpenSees and update the local mirror."""
        args = self._algorithm_args(algotype, user_algo_args)
        if verbose:
            arg_txt = " ".join(str(a) for a in args)
            self._print_msg(f"Setting algorithm to {arg_txt}", icon="✳️")
        ops.algorithm(*args)
        self.current_algorithm_type = algotype
        self.current_algorithm_args = list(args)

    # ------------------------------------------------------------------
    # Norm handling
    # ------------------------------------------------------------------

    def _last_norm(self) -> float:
        """Return the last per-iteration norm from the most recent analysis step.

        ``last_norms`` holds the stripped iteration-norm list produced by
        ``_test_norms()`` after the previous ``_analyze_one_step`` call.
        The final element is the norm at the last Newton iteration — the value
        most relevant for deciding whether ``_try_add_test_times`` should fire.

        Returns ``np.inf`` when no finite norms are available (e.g. the very
        first call before any step has been attempted).
        """
        arr = np.array(self.last_norms, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            return float(finite[-1])
        return np.inf

    @staticmethod
    def _norm_trend(norms: list[float]) -> NormTrend:
        """Compute a NormTrend summary from a list of norm values."""
        norms_arr = np.array(norms, dtype=float)
        trend = NormTrend()
        trend.has_nan = bool(np.any(np.isnan(norms_arr)))
        trend.has_inf = bool(np.any(np.isinf(norms_arr)))
        trend.has_nonfinite = bool(np.any(~np.isfinite(norms_arr)))

        finite = norms_arr[np.isfinite(norms_arr)]
        if finite.size == 0:
            return trend

        trend.is_empty = False
        trend.first = float(finite[0])
        trend.last = float(finite[-1])
        trend.min = float(finite.min())
        trend.max = float(finite.max())

        denom = max(abs(trend.first), np.finfo(float).eps)
        trend.improvement_ratio = abs(trend.last) / denom
        trend.is_improving = trend.improvement_ratio < 0.5
        trend.is_diverging = finite.size >= 2 and abs(trend.last) > abs(trend.first)

        if finite.size >= 3:
            recent = finite[-3:]
            rel_change = np.abs(np.diff(recent)) / np.maximum(np.abs(recent[:-1]), np.finfo(float).eps)
            trend.is_stagnating = bool(np.all(rel_change < 1e-3))

        return trend

    # ------------------------------------------------------------------
    # Recording (history + diagnostics)
    # ------------------------------------------------------------------

    def _record_attempt(
        self,
        strategy: str,
        step: float,
        ok: int,
        norms: list[float],
        trend: NormTrend,
    ) -> None:
        """Store norm data and (on failure) a diagnostic record."""
        # Always update last_norms so _last_norm() reflects the current step.
        self.last_norms = norms
        self.current_args["step"] = step

        # Lightweight norm-history entry (recorded for every attempt).
        if self.control_args["recordNormHistory"]:
            entry = NormHistoryEntry(
                step_index=self.current_args["progress"] + 1,
                strategy=str(strategy),
                ok=ok,
                first_norm=trend.first,
                last_norm=trend.last,
                min_norm=trend.min,
                max_norm=trend.max,
                num_iter=len(norms),
                improvement_ratio=trend.improvement_ratio,
                is_diverging=trend.is_diverging,
                is_stagnating=trend.is_stagnating,
            )
            self.norm_history.append(entry)

        # Detailed diagnostic record (only on failure, only if enabled).
        if ok != 0 and self.control_args["recordDiagnostics"]:
            rec = DiagnosticRecord(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                step_index=self.current_args["progress"] + 1,
                analysis=self.analysis_type,
                strategy=str(strategy),
                step=step,
                ok=ok,
                test_type=self.current_test_type,
                test_tol=self.current_test_tol,
                test_iter_times=self.current_test_iter_times,
                test_print_flag=self.current_test_print_flag,
                algorithm_type=self.current_algorithm_type,
                algorithm_args=list(self.current_algorithm_args),
                algorithm_text=self._algorithm_text(self.current_algorithm_args),
                norms=list(norms),
                trend=trend,
                reason=self._failure_reason(ok, trend),
                suggestion=self._convergence_suggestion(ok, trend, str(strategy)),
            )
            self.diagnostics["records"].append(rec)
            max_rec = self.diagnostics.get("maxRecords", 200)
            if len(self.diagnostics["records"]) > max_rec:
                self.diagnostics["records"] = self.diagnostics["records"][-max_rec:]
            self.diagnostics["lastFailure"] = rec

    def _flag_partial_advance(
        self,
        total_step: float,
        completed_step: float,
        remaining_step: float,
        failed_substep: float,
        reason: str,
    ) -> None:
        """Annotate the last diagnostic record with partial-advance information.

        Called when step relaxation succeeds for some sub-steps but eventually
        fails, leaving the OpenSees model state partially advanced.
        """
        info = {
            "totalStep": total_step,
            "completedStep": completed_step,
            "remainingStep": remaining_step,
            "failedSubstep": failed_substep,
            "reason": str(reason),
        }
        records = self.diagnostics["records"]
        if records:
            rec = records[-1]
            rec.partial_advance = True
            rec.partial_advance_info = info
            rec.reason = f"{rec.reason}; partial model advancement detected"
            rec.suggestion = (
                f"{rec.suggestion} The OpenSees model state may already be partially "
                "advanced; avoid applying another whole-step retry unless the model can be restored."
            )
            records[-1] = rec
            self.diagnostics["lastFailure"] = rec

    @staticmethod
    def _failure_reason(ok: int, trend: NormTrend) -> str:
        """Classify the failure reason from the return code and norm trend."""
        if ok == 0:
            return "converged"
        if trend.has_nonfinite:
            return "nonfinite convergence norm detected"
        if trend.is_empty:
            return "convergence norm unavailable"
        if trend.is_diverging:
            return "convergence norm is diverging"
        if trend.is_stagnating:
            return "convergence norm is stagnating"
        if trend.is_improving:
            return "convergence norm is improving but did not reach tolerance"
        return "analysis command returned a failure code"

    @staticmethod
    def _convergence_suggestion(ok: int, trend: NormTrend, strategy: str) -> str:
        """Return a human-readable suggestion based on the failure mode."""
        if ok == 0:
            return "No action needed."
        if trend.has_nonfinite:
            return (
                "Detected NaN/Inf convergence norms. Try reducing the step size, "
                "checking material state limits, and switching to a robust line-search algorithm."
            )
        if trend.is_empty:
            return (
                "OpenSees did not provide convergence norms. Enable test print output, "
                "inspect the OpenSees warning log, and check whether the system/integrator "
                "failed before the convergence test ran."
            )
        if trend.is_diverging:
            return (
                "The convergence norm is growing. Try a smaller step, NewtonLineSearch, "
                "KrylovNewton, or a more stable constraint/system configuration."
            )
        if trend.is_stagnating:
            return (
                "The convergence norm is stagnating. Try changing the convergence test type, "
                "switching algorithms, or using a smaller step size."
            )
        if trend.is_improving:
            return (
                "The convergence norm is decreasing but not enough. Try increasing testIterTimes, "
                "using tryAddTestTimes, or loosening testTol within an acceptable engineering tolerance."
            )
        if "tryRelaxStep" in strategy:
            return (
                "Step relaxation failed. Consider reducing the target step size, increasing relaxation, "
                "decreasing minStep, or checking for severe local nonlinearities."
            )
        if "tryAlterAlgo" in strategy:
            return (
                "Algorithm switching failed. Add more algorithm candidates, especially "
                "NewtonLineSearch variants, ModifiedNewton, BFGS, or Broyden."
            )
        if "tryLooseTol" in strategy:
            return (
                "Loose tolerance retry failed. The issue is likely not only tolerance-related; "
                "try smaller steps or a different algorithm/test type."
            )
        return (
            "Try enabling tryAddTestTimes, tryAlterAlgoTypes, and tryLooseTestTol, "
            "then inspect the recorded norm trend and OpenSees warning messages."
        )

    # ------------------------------------------------------------------
    # Algorithm lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _algorithm_args(algotype: int, user_args: list | None = None) -> list:
        """Resolve an algorithm type code to its OpenSees argument list."""
        if algotype == 100:
            if user_args is None:
                raise ValueError("UserAlgoArgs must be provided for algorithm type 100.")  # noqa: TRY003
            return list(user_args)
        if algotype in _ALGO_MAP:
            return list(_ALGO_MAP[algotype])
        raise ValueError(f"Wrong algorithm type: {algotype}")  # noqa: TRY003

    @staticmethod
    def _algorithm_text(args: list) -> str:
        """Format an algorithm argument list as a human-readable string."""
        return " ".join(str(a) for a in args) if args else "<unset>"

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _print_msg(self, text: str, icon: str = "✳️") -> None:
        """Print a coloured or plain message depending on the environment."""
        if ON_NOTEBOOK:
            print(f">>> {icon} {self.logo} {text}")
        else:
            color = get_random_color()
            rprint(f">>> {icon} {self.logo} [bold {color}]{text}[/bold {color}]")

    def _print_status(self, success: bool) -> None:
        t = self._get_time()
        npts = self.current_args.get("npts", 0)
        if npts > 0:
            pct = min(100.0, 100.0 * self.current_args["progress"] / npts)
            progress_text = f" Progress: {pct:.3f} % ({self.current_args['progress']}/{npts})."
        else:
            progress_text = f" Progress: {self.current_args['progress']} steps."

        if ON_NOTEBOOK:
            time_val = f"{t:.3f}"
            if success:
                print(f">>> 🎉 {self.logo} Successfully finished!{progress_text} Time consumption: {time_val} s. 🎉")
            else:
                print(f">>> ❌ {self.logo} Analyze failed.{progress_text} Time consumption: {time_val} s.")
        else:
            color = get_random_color()
            time_val = f"[bold {color}]{t:.3f}[/bold {color}]"
            if success:
                rprint(
                    f">>> 🎉 {self.logo} [{color}]Successfully finished[/{color}]!"
                    f"{progress_text} Time consumption: {time_val} s. 🎉"
                )
            else:
                rprint(f">>> ❌ {self.logo} Analyze failed.{progress_text} Time consumption: {time_val} s.")

    def _print_progress(self) -> None:
        prog = self.current_args["progress"]
        total = self.current_args.get("npts", 0)
        t = self._get_time()
        if ON_NOTEBOOK:
            time_val = f"{t:.3f}"
            if total > 0:
                pct = 100 * prog / total
                print(f">>> ✅ {self.logo} progress {pct:.3f} %. Time consumption: {time_val} s.")
            else:
                print(f">>> ✅ {self.logo} progress {prog} steps. Time consumption: {time_val} s.")
        else:
            color = get_random_color()
            time_val = f"[bold {color}]{t:.3f}[/bold {color}]"
            if total > 0:
                pct = f"[bold {color}]{100 * prog / total:.3f}[/bold {color}]"
                rprint(f">>> ✅ {self.logo} progress {pct} %. Time consumption: {time_val} s.")
            else:
                rprint(f">>> ✅ {self.logo} progress {prog} steps. Time consumption: {time_val} s.")
