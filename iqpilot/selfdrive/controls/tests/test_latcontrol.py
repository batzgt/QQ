from parameterized import parameterized

from iqpilot.cereal import car, log
from iqdbc.car.car_helpers import interfaces
from iqdbc.car.honda.values import CAR as HONDA
from iqdbc.car.toyota.values import CAR as TOYOTA
from iqdbc.car.nissan.values import CAR as NISSAN
from iqdbc.car.gm.values import CAR as GM
from iqdbc.car.vehicle_model import VehicleModel
from iqpilot.common.realtime import DT_CTRL
from iqpilot.selfdrive.car.helpers import convert_to_capnp
from iqpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from iqpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from iqpilot.selfdrive.controls.lib.latcontrol_torque_pq import LatControlTorquePQ
import iqpilot.selfdrive.controls.lib.latcontrol_torque_pq as latcontrol_torque_pq
from iqpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle
from iqpilot.selfdrive.locationd.helpers import Pose
from iqpilot.common.mock.generators import generate_deviceMotion
from iqpilot.selfdrive.car import interfaces as iqpilot_interfaces


class TestLatControl:

  @staticmethod
  def build_pq_controller():
    car_name = TOYOTA.TOYOTA_RAV4
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_IQ = CarInterface.get_non_essential_params_iq(CP, car_name)
    CI = CarInterface(CP, CP_IQ)
    iqpilot_interfaces.apply_iq_car_config(CI)
    return CP, LatControlTorquePQ(CP.as_reader(), convert_to_capnp(CP_IQ).as_reader(), CI, DT_CTRL)

  @parameterized.expand([(HONDA.HONDA_CIVIC, LatControlPID), (TOYOTA.TOYOTA_RAV4, LatControlTorque),
                         (NISSAN.NISSAN_LEAF, LatControlAngle), (GM.CHEVROLET_BOLT_EUV, LatControlTorque)])
  def test_saturation(self, car_name, controller):
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_IQ = CarInterface.get_non_essential_params_iq(CP, car_name)
    CI = CarInterface(CP, CP_IQ)
    iqpilot_interfaces.apply_iq_car_config(CI)
    CP_IQ = convert_to_capnp(CP_IQ)
    VM = VehicleModel(CP)

    controller = controller(CP.as_reader(), CP_IQ.as_reader(), CI, DT_CTRL)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    CS.steeringPressed = False

    params = log.VehicleParameters.new_message()

    lp = generate_deviceMotion()
    pose = Pose.from_live_pose(lp.deviceMotion)

    # Saturate for curvature limited and controller limited
    for _ in range(1000):
      _, _, lac_log = controller.update(True, CS, VM, params, False, 0, pose, True, 0.2)
    assert lac_log.saturated

    for _ in range(1000):
      _, _, lac_log = controller.update(True, CS, VM, params, False, 0, pose, False, 0.2)
    assert not lac_log.saturated

    for _ in range(1000):
      _, _, lac_log = controller.update(True, CS, VM, params, False, 1, pose, False, 0.2)
    assert lac_log.saturated

  def test_pq_controller_update(self):
    CP, controller = self.build_pq_controller()
    VM = VehicleModel(CP)

    CS = car.CarState.new_message()
    CS.vEgo = 30
    params = log.VehicleParameters.new_message()
    pose = Pose.from_live_pose(generate_deviceMotion().deviceMotion)

    _, _, lac_log = controller.update(True, CS, VM, params, False, 0.001, pose, False, 0.2)
    assert lac_log.active

  def test_pq_controller_inactive_lookahead_and_slew_reset(self):
    CP, controller = self.build_pq_controller()
    controller.curvature_lookahead_enabled = True
    controller.lateral_acceleration_slew_limiter.enabled = True
    VM = VehicleModel(CP)
    CS = car.CarState.new_message(vEgo=30)
    params = log.VehicleParameters.new_message()
    pose = Pose.from_live_pose(generate_deviceMotion().deviceMotion)

    torque, angle, lac_log = controller.update(False, CS, VM, params, False, 0.001, pose, False, 0.2, lookahead_curvature=0.002)
    assert torque == 0.0
    assert angle == 0.0
    assert not lac_log.active
    assert controller.lateral_acceleration_slew_limiter.a_lim == 1.8

  def test_pq_live_torque_update_freeze_and_unfreeze(self, monkeypatch):
    _, controller = self.build_pq_controller()
    initial = (controller.torque_params.latAccelFactor, controller.torque_params.latAccelOffset, controller.torque_params.friction)
    controller.update_live_torque_params(3.0, 0.2, 0.4)
    assert (controller.torque_params.latAccelFactor, controller.torque_params.latAccelOffset, controller.torque_params.friction) == initial

    monkeypatch.setattr(latcontrol_torque_pq, "FREEZE_LIVE_TORQUE_PARAMS", False)
    controller.update_live_torque_params(3.0, 0.2, 0.4)
    assert controller.torque_params.latAccelFactor == 3.0
    assert abs(controller.torque_params.latAccelOffset - 0.2) < 1e-6
    assert abs(controller.torque_params.friction - 0.4) < 1e-6
