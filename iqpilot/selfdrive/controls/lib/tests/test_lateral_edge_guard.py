from __future__ import annotations

from dataclasses import dataclass
import math

from iqpilot.cereal import custom, log
import iqpilot.cereal.messaging as messaging
from iqpilot.common.realtime import DT_MDL
from iqpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from iqpilot.selfdrive.controls.lib.helpers.lane_change import AutoLaneChangeMode
from iqpilot.selfdrive.controls.lib.helpers.lateral_edge_guard import (
  BLOCK_DEBOUNCE_S,
  CLEAR_DEBOUNCE_S,
  MAX_VALID_ROAD_EDGE_STD_M,
  MIN_ACTIVE_SPEED_MPS,
  REQUIRED_ROAD_EDGE_DISTANCE_M,
  UNAVAILABLE_HOLD_S,
  LateralEdgeGuard,
  RoadEdgeDataState,
  evaluate_road_edge,
)
from iqpilot.selfdrive.selfdrived.iq_events import EVENTS_IQ, ET
from iqpilot.selfdrive.selfdrived.selfdrived import SelfdriveD


@dataclass
class Edge:
  x: list[float]
  y: list[float]


@dataclass
class ModelData:
  roadEdges: list[Edge]
  roadEdgeStds: list[float]


class CarState:
  def __init__(self, left_blindspot: bool = False) -> None:
    self.vEgo = MIN_ACTIVE_SPEED_MPS + 1.0
    self.leftBlinker = True
    self.rightBlinker = False
    self.leftBlindspot = left_blindspot
    self.rightBlindspot = False
    self.steeringPressed = True
    self.steeringTorque = 1.0
    self.brakePressed = False
    self.standstill = False


def edge_model(left_distance_m: float = 6.0, right_distance_m: float = 6.0,
               left_std_m: float = 0.0, right_std_m: float = 0.0) -> ModelData:
  xs = [5.0, 20.0, 40.0]
  return ModelData(
    [Edge(xs, [-left_distance_m] * len(xs)), Edge(xs, [right_distance_m] * len(xs))],
    [left_std_m, right_std_m],
  )


def cycles(duration_s: float) -> int:
  return math.ceil(duration_s / DT_MDL)


def update_for(guard: LateralEdgeGuard, modeldata: ModelData | None, duration_s: float,
               speed_mps: float = MIN_ACTIVE_SPEED_MPS) -> None:
  for _ in range(cycles(duration_s)):
    guard.update(modeldata, speed_mps, DT_MDL)


def test_valid_geometry_blocks_and_clear_geometry_does_not_block() -> None:
  blocked = evaluate_road_edge(edge_model(4.0).roadEdges[0], 0.2, log.LaneChangeDirection.left)
  clear = evaluate_road_edge(edge_model(6.0).roadEdges[0], 0.2, log.LaneChangeDirection.left)
  assert blocked.state == RoadEdgeDataState.VALID
  assert blocked.should_block is True
  assert clear.state == RoadEdgeDataState.VALID
  assert clear.should_block is False


def test_unavailable_and_invalid_are_distinct() -> None:
  unavailable = evaluate_road_edge(Edge([5.0], []), 0.2, log.LaneChangeDirection.left)
  invalid = evaluate_road_edge(edge_model().roadEdges[0], MAX_VALID_ROAD_EDGE_STD_M + 0.01,
                               log.LaneChangeDirection.left)
  assert unavailable.state == RoadEdgeDataState.UNAVAILABLE
  assert unavailable.lateral_distance_m is None
  assert invalid.state == RoadEdgeDataState.INVALID
  assert invalid.should_block is None


def test_two_sigma_bound_uses_std_in_metres() -> None:
  measurement = evaluate_road_edge(edge_model(5.0).roadEdges[0], 0.2, log.LaneChangeDirection.left)
  assert measurement.lateral_distance_m == 5.0
  assert measurement.conservative_distance_m == 4.6
  assert measurement.should_block is True


def test_distance_threshold_on_either_side() -> None:
  epsilon_m = 0.001
  for direction, edge_index in ((log.LaneChangeDirection.left, 0), (log.LaneChangeDirection.right, 1)):
    below = edge_model(REQUIRED_ROAD_EDGE_DISTANCE_M - epsilon_m, REQUIRED_ROAD_EDGE_DISTANCE_M - epsilon_m)
    above = edge_model(REQUIRED_ROAD_EDGE_DISTANCE_M + epsilon_m, REQUIRED_ROAD_EDGE_DISTANCE_M + epsilon_m)
    assert evaluate_road_edge(below.roadEdges[edge_index], 0.0, direction).should_block is True
    assert evaluate_road_edge(above.roadEdges[edge_index], 0.0, direction).should_block is False


def test_block_debounce_rejects_a_single_clear_frame() -> None:
  guard = LateralEdgeGuard()
  blocking = edge_model(4.0)
  clear = edge_model(6.0)
  update_for(guard, blocking, BLOCK_DEBOUNCE_S - DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none
  guard.update(clear, MIN_ACTIVE_SPEED_MPS, DT_MDL)
  update_for(guard, blocking, BLOCK_DEBOUNCE_S)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left


def test_clear_debounce_rejects_a_single_blocking_frame() -> None:
  guard = LateralEdgeGuard()
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  update_for(guard, edge_model(6.0), CLEAR_DEBOUNCE_S - DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left
  guard.update(edge_model(4.0), MIN_ACTIVE_SPEED_MPS, DT_MDL)
  update_for(guard, edge_model(6.0), CLEAR_DEBOUNCE_S)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_unavailable_holds_then_falls_back_to_not_blocking() -> None:
  guard = LateralEdgeGuard()
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  update_for(guard, None, UNAVAILABLE_HOLD_S - DT_MDL)
  assert guard.left_measurement.state == RoadEdgeDataState.UNAVAILABLE
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left
  guard.update(None, MIN_ACTIVE_SPEED_MPS, DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_invalid_measurement_clears_through_release_debounce() -> None:
  guard = LateralEdgeGuard()
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  invalid = edge_model(4.0, left_std_m=MAX_VALID_ROAD_EDGE_STD_M + 0.01)
  update_for(guard, invalid, CLEAR_DEBOUNCE_S - DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left
  guard.update(invalid, MIN_ACTIVE_SPEED_MPS, DT_MDL)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none


def test_speed_gate_is_inactive_below_threshold() -> None:
  guard = LateralEdgeGuard()
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S, MIN_ACTIVE_SPEED_MPS - 0.01)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.none
  update_for(guard, edge_model(4.0), BLOCK_DEBOUNCE_S, MIN_ACTIVE_SPEED_MPS)
  assert guard.block_for_direction(log.LaneChangeDirection.left) == custom.IQLateralEdgeBlock.left


def test_desire_helper_keeps_edge_block_out_of_blindspot_path() -> None:
  helper = DesireHelper()
  helper.alc.lane_change_set_timer = AutoLaneChangeMode.NUDGE
  helper.lane_change_state = log.LaneChangeState.preLaneChange
  helper.lane_change_direction = log.LaneChangeDirection.left
  update_for(helper.lateral_edge_guard, edge_model(4.0), BLOCK_DEBOUNCE_S)
  blindspot_arguments: list[bool] = []

  def record_blindspot(blindspot_detected: bool, brake_pressed: bool) -> None:
    blindspot_arguments.append(blindspot_detected)

  helper.alc.update_lane_change = record_blindspot
  helper.update(CarState(left_blindspot=False), True, 1.0, modeldata=edge_model(4.0))
  assert blindspot_arguments == [False]
  assert helper.lateral_edge_block == custom.IQLateralEdgeBlock.left
  assert helper.lane_change_state == log.LaneChangeState.preLaneChange

  helper.update(CarState(left_blindspot=True), True, 1.0, modeldata=edge_model(4.0))
  assert blindspot_arguments[-1] is True


def test_published_edge_block_maps_to_distinct_event_and_alert() -> None:
  message = messaging.new_message("iqDriveModelData")
  message.iqDriveModelData.lateralEdgeBlock = custom.IQLateralEdgeBlock.right

  class SubMaster:
    updated = {"iqDriveModelData": True}

    def __getitem__(self, service: str):
      assert service == "iqDriveModelData"
      return message.iqDriveModelData

  selfdrived = SelfdriveD.__new__(SelfdriveD)
  selfdrived.sm = SubMaster()
  selfdrived._cached_model_event_names = ()
  selfdrived._refresh_cached_model_events()

  event_name = custom.IQOnroadEvent.EventName.lateralEdgeBlocked
  assert selfdrived._cached_model_event_names == (event_name,)
  alert = EVENTS_IQ[event_name][ET.WARNING]
  assert alert.alert_text_1 == "Lane Change Blocked"
  assert alert.alert_text_2 == "Road edge detected"
