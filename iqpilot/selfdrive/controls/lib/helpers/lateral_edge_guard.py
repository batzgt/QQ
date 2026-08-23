"""
Lateral Edge Guard uses the model's lateral road-edge geometry to withhold lane
changes that lack room for a target lane. The model standard deviation remains
in metres: measurements above the validity limit are rejected, while valid
measurements use a two-sigma lower confidence bound for conservative clearance.
Unavailable geometry briefly holds the last output, then fails open because a
model dropout is not geometric evidence of a nearby edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from iqpilot.cereal import custom, log
from iqpilot.common.constants import CV
from iqpilot.common.swaglog import cloudlog


MIN_ACTIVE_SPEED_MPS = 20.0 * CV.MPH_TO_MS  # Matches the lane-change speed gate and excludes parking manoeuvres.
MAX_VALID_ROAD_EDGE_STD_M = 1.0  # A 2-sigma bound beyond 2 m cannot distinguish an adjacent 3.5 m lane reliably.
EDGE_CONFIDENCE_SIGMA = 2.0  # 97.7% one-sided confidence under the model's Gaussian uncertainty assumption.
ROAD_EDGE_LOOKAHEAD_MIN_M = 5.0  # Ignore near-field edge points dominated by vehicle-body perspective.
ROAD_EDGE_LOOKAHEAD_MAX_M = 40.0  # Covers about 2 s at the 20 m/s model-training reference speed.
LANE_CENTER_OFFSET_M = 3.5  # Typical freeway lane width and the target-centre lateral displacement.
# CarParams exposes neither width nor track; 0.95 m is half of an assumed conservative 1.90 m body width.
VEHICLE_LATERAL_HALF_WIDTH_M = 1.90 / 2.0
EDGE_CLEARANCE_MARGIN_M = 0.25  # Additional lateral separation between the vehicle body and detected road edge.
REQUIRED_ROAD_EDGE_DISTANCE_M = LANE_CENTER_OFFSET_M + VEHICLE_LATERAL_HALF_WIDTH_M + EDGE_CLEARANCE_MARGIN_M
BLOCK_DEBOUNCE_S = 0.30  # Six model frames reject a transient close-edge prediction before blocking.
CLEAR_DEBOUNCE_S = 0.50  # Ten model frames make release slower than assertion for conservative hysteresis.
UNAVAILABLE_HOLD_S = 0.50  # Ten model frames bridge a short model-data dropout before failing open.
TIMER_EPSILON_S = 1e-9  # Floating-point comparison tolerance, far below one model tick.

LaneChangeDirection = log.LaneChangeDirection
LateralEdgeBlock = custom.IQLateralEdgeBlock


class RoadEdgeDataState(IntEnum):
  VALID = 0
  UNAVAILABLE = 1
  INVALID = 2


@dataclass(frozen=True, slots=True)
class RoadEdgeMeasurement:
  state: RoadEdgeDataState
  lateral_distance_m: float | None = None
  conservative_distance_m: float | None = None
  should_block: bool | None = None


@dataclass(frozen=True, slots=True)
class _SideState:
  blocked: bool = False
  block_timer_s: float = 0.0
  clear_timer_s: float = 0.0
  unavailable_timer_s: float = 0.0
  fallback_reported: bool = False


def evaluate_road_edge(edge: Any, std_m: Any, direction: int) -> RoadEdgeMeasurement:
  if edge is None or std_m is None:
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  try:
    xs = edge.x
    ys = edge.y
    count = len(xs)
    y_count = len(ys)
  except (AttributeError, TypeError):
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  if count == 0 or y_count != count:
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  try:
    std = float(std_m)
  except (TypeError, ValueError):
    return RoadEdgeMeasurement(RoadEdgeDataState.INVALID)
  if not math.isfinite(std) or std < 0.0 or std > MAX_VALID_ROAD_EDGE_STD_M:
    return RoadEdgeMeasurement(RoadEdgeDataState.INVALID)

  lateral_distance_m: float | None = None
  for idx in range(count):
    try:
      x_m = float(xs[idx])
      y_m = float(ys[idx])
    except (IndexError, TypeError, ValueError):
      return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)
    if not math.isfinite(x_m) or not math.isfinite(y_m):
      return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)
    if not ROAD_EDGE_LOOKAHEAD_MIN_M <= x_m <= ROAD_EDGE_LOOKAHEAD_MAX_M:
      continue
    if ((direction == LaneChangeDirection.left and y_m >= 0.0) or
        (direction == LaneChangeDirection.right and y_m <= 0.0)):
      return RoadEdgeMeasurement(RoadEdgeDataState.INVALID)
    distance_m = abs(y_m)
    lateral_distance_m = distance_m if lateral_distance_m is None else min(lateral_distance_m, distance_m)

  if lateral_distance_m is None:
    return RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  conservative_distance_m = lateral_distance_m - EDGE_CONFIDENCE_SIGMA * std
  return RoadEdgeMeasurement(
    RoadEdgeDataState.VALID,
    lateral_distance_m,
    conservative_distance_m,
    conservative_distance_m < REQUIRED_ROAD_EDGE_DISTANCE_M,
  )


def step_side_guard(state: _SideState, measurement: RoadEdgeMeasurement, speed_active: bool,
                    dt_s: float) -> tuple[_SideState, bool]:
  if not speed_active:
    return _SideState(), False

  if measurement.state == RoadEdgeDataState.UNAVAILABLE:
    unavailable_timer_s = state.unavailable_timer_s + dt_s
    if unavailable_timer_s < UNAVAILABLE_HOLD_S - TIMER_EPSILON_S:
      return _SideState(state.blocked, unavailable_timer_s=unavailable_timer_s,
                        fallback_reported=state.fallback_reported), False
    fallback_started = not state.fallback_reported
    return _SideState(unavailable_timer_s=unavailable_timer_s, fallback_reported=True), fallback_started

  should_block = bool(measurement.should_block) if measurement.state == RoadEdgeDataState.VALID else False
  if should_block == state.blocked:
    return _SideState(blocked=state.blocked), False

  if should_block:
    block_timer_s = state.block_timer_s + dt_s
    if block_timer_s >= BLOCK_DEBOUNCE_S - TIMER_EPSILON_S:
      return _SideState(blocked=True), False
    return _SideState(block_timer_s=block_timer_s), False

  clear_timer_s = state.clear_timer_s + dt_s
  if clear_timer_s >= CLEAR_DEBOUNCE_S - TIMER_EPSILON_S:
    return _SideState(), False
  return _SideState(blocked=True, clear_timer_s=clear_timer_s), False


class LateralEdgeGuard:
  def __init__(self) -> None:
    self._left = _SideState()
    self._right = _SideState()
    self.left_measurement = RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)
    self.right_measurement = RoadEdgeMeasurement(RoadEdgeDataState.UNAVAILABLE)

  @staticmethod
  def _model_side(modeldata: Any, side_index: int) -> tuple[Any | None, Any | None]:
    if modeldata is None:
      return None, None
    try:
      edges = modeldata.roadEdges
      stds = modeldata.roadEdgeStds
      if len(edges) <= side_index or len(stds) <= side_index:
        return None, None
      return edges[side_index], stds[side_index]
    except (AttributeError, TypeError):
      return None, None

  def update(self, modeldata: Any, v_ego_mps: float, dt_s: float) -> None:
    dt = max(float(dt_s), 0.0)
    left_edge, left_std = self._model_side(modeldata, 0)
    right_edge, right_std = self._model_side(modeldata, 1)
    self.left_measurement = evaluate_road_edge(left_edge, left_std, LaneChangeDirection.left)
    self.right_measurement = evaluate_road_edge(right_edge, right_std, LaneChangeDirection.right)
    speed_active = math.isfinite(v_ego_mps) and v_ego_mps >= MIN_ACTIVE_SPEED_MPS
    self._left, left_fallback = step_side_guard(self._left, self.left_measurement, speed_active, dt)
    self._right, right_fallback = step_side_guard(self._right, self.right_measurement, speed_active, dt)
    if left_fallback:
      cloudlog.warning(f"lateral edge guard: left road edge unavailable for {UNAVAILABLE_HOLD_S:.2f} s; falling back to not blocking")
    if right_fallback:
      cloudlog.warning(f"lateral edge guard: right road edge unavailable for {UNAVAILABLE_HOLD_S:.2f} s; falling back to not blocking")

  def block_for_direction(self, direction: int) -> custom.IQLateralEdgeBlock:
    if direction == LaneChangeDirection.left and self._left.blocked:
      return LateralEdgeBlock.left
    if direction == LaneChangeDirection.right and self._right.blocked:
      return LateralEdgeBlock.right
    return LateralEdgeBlock.none
