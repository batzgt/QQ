"""Copyright (c) IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved."""

from iqdbc.can import CANPacker, CANParser
from iqdbc.car.volkswagen import mebcan


EA_HUD_VALUES = {
  "EA_Texte": 0,
  "ACF_Lampe_Hands_Off": 0,
  "EA_Infotainment_Anf": 0,
  "EA_Tueren_Anf": 0,
  "EA_Innenraumlicht_Anf": 0,
  "zFAS_Warnblinken": 0,
  "STP_Primaeranz": 0,
  "EA_Bremslichtblinken": 0,
  "EA_Blinken": 0,
  "EA_Unknown": 0,
}
EA_CONTROL_VALUES = {"EA_Funktionsstatus": 2}


def blinker_values(dbc, requests):
  packer = CANPacker(dbc)
  parser = CANParser(dbc, [("EA_02", 50)], 0)
  values = []
  for left, right in requests:
    msg = mebcan.create_blinker_control(
      packer, 0, EA_HUD_VALUES, EA_CONTROL_VALUES, left, right, False,
    )
    parser.update([0, [msg]])
    values.append(int(parser.vl["EA_02"]["EA_Blinken"]))
  return values


def test_nav_blinker_request_remains_asserted_until_released():
  requests = [(True, False)] * 100 + [(False, False)]
  assert blinker_values("vw_meb", requests) == [1] * 100 + [0]


def test_nav_blinker_directions_on_meb_variants():
  requests = [(True, False), (False, True), (False, False)]
  for dbc in ("vw_meb", "vw_meb_2024", "vw_mqbevo"):
    assert blinker_values(dbc, requests) == [1, 2, 0]
