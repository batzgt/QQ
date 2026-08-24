import asyncio
import os

import pytest

from iqpilot.system.webrtc.rtc import WebRTCOfferBuilder
from iqpilot.system.webrtc.rtc.stream import RTCSessionDescription
from iqpilot.system.webrtc.session import StreamSession


@pytest.mark.asyncio
async def test_native_video_audio_and_data_channel():
  if not os.environ.get("CI"):
    return

  answer_session = None
  video_received = asyncio.Event()
  audio_received = asyncio.Event()

  async def connect(offer):
    nonlocal answer_session
    answer_session = StreamSession(offer.sdp, offer.video, [], [], [], debug_mode=True)
    answer = await answer_session.get_answer()
    answer_session.start()
    return RTCSessionDescription(answer.sdp, answer.type)

  builder = WebRTCOfferBuilder(connect, ice_servers=[])
  builder.offer_to_receive_video_stream("road")
  builder.offer_to_receive_audio_stream()
  builder.add_messaging()
  stream = builder.stream()

  try:
    await asyncio.wait_for(stream.start(), 10)
    stream.get_incoming_video_track("road").on_frame(lambda *_: video_received.set())
    stream.get_incoming_audio_track().on_frame(lambda *_: audio_received.set())
    await asyncio.wait_for(stream.wait_for_connection(), 10)
    await asyncio.wait_for(video_received.wait(), 10)
    await asyncio.wait_for(audio_received.wait(), 10)
    stream.get_messaging_channel().send('{"type":"timingSei","enabled":true}')
    await asyncio.sleep(0.1)
    assert answer_session is not None
    assert answer_session.video_tracks[0].timing_sei_enabled
  finally:
    await stream.stop()
    if answer_session is not None:
      await answer_session.stop_async()


@pytest.mark.asyncio
async def test_native_duplex_audio_negotiation():
  if not os.environ.get("CI"):
    return

  answer_session = None

  async def connect(offer):
    nonlocal answer_session
    sendrecv_offer = offer.sdp.replace("a=recvonly", "a=sendrecv", 1)
    answer_session = StreamSession(sendrecv_offer, offer.video, [], [], [], debug_mode=True)
    answer = await answer_session.get_answer()
    answer_session.start()
    return RTCSessionDescription(answer.sdp, answer.type)

  builder = WebRTCOfferBuilder(connect, ice_servers=[])
  builder.offer_to_receive_audio_stream()
  stream = builder.stream()

  try:
    await asyncio.wait_for(stream.start(), 10)
    await asyncio.wait_for(stream.wait_for_connection(), 10)
    assert stream.has_incoming_audio_track()
    assert answer_session is not None
    assert answer_session.audio_recv_requested
    assert answer_session.audio_output is not None
  finally:
    await stream.stop()
    if answer_session is not None:
      await answer_session.stop_async()
