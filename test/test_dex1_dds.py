"""Dex1 gripper DDS state probe.

Subscribes to rt/dex1/left/state and rt/dex1/right/state on DDS domain 0
for ~5 seconds and prints what arrives. Use to verify that the gripper
feedback service on the robot is actually publishing before launching
eval_g1.py -- if this script reports "NO MESSAGES RECEIVED", eval_g1.py
will hang in Dex1_1_Gripper_Controller's subscribe-wait loop.

Usage (from the repo root or any cwd):
    bash -ic 'use_conda unitree-lerobot-groot && python test/test_dex1_dds.py'

Override the listen window or DDS domain via env vars if needed:
    DEX1_PROBE_SECONDS=10 DEX1_PROBE_DOMAIN=0 python test/test_dex1_dds.py
"""

import os
import time

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


# Domain 0 = real robot, 1 = sim. eval_g1.py defaults to real (motion=False -> 0).
DOMAIN = int(os.environ.get("DEX1_PROBE_DOMAIN", 0))
WINDOW_SEC = float(os.environ.get("DEX1_PROBE_SECONDS", 5.0))
TOPICS = {
    "left":  "rt/dex1/left/state",
    "right": "rt/dex1/right/state",
}


def main() -> None:
    ChannelFactoryInitialize(DOMAIN)

    last_msg: dict[str, object] = {side: None for side in TOPICS}
    msg_count: dict[str, int] = {side: 0 for side in TOPICS}

    def make_handler(side: str):
        def _cb(msg: MotorStates_) -> None:
            last_msg[side] = msg
            msg_count[side] += 1
        return _cb

    subscribers = {}
    for side, topic in TOPICS.items():
        sub = ChannelSubscriber(topic, MotorStates_)
        sub.Init(make_handler(side), 10)  # queue depth 10
        subscribers[side] = sub
        print(f"Subscribed to {topic}")

    print(f"\nListening for {WINDOW_SEC:.1f} s on DDS domain {DOMAIN}...\n")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < WINDOW_SEC:
        time.sleep(0.5)
        elapsed = time.perf_counter() - t0
        print(
            f"  t={elapsed:4.1f}s | left: {msg_count['left']:>3} msgs | "
            f"right: {msg_count['right']:>3} msgs"
        )

    print("\n--- Final ---")
    for side in TOPICS:
        if msg_count[side] == 0:
            print(
                f"  {side:>5}: NO MESSAGES RECEIVED -- gripper service for this side is not publishing."
            )
            continue
        msg = last_msg[side]
        # MotorStates_ exposes a list of MotorState_ at msg.states; each has .q .dq .tau_est etc.
        states = getattr(msg, "states", None)
        if states is None:
            print(f"  {side:>5}: {msg_count[side]} msgs received but couldn't decode .states field.")
            continue
        qs = [round(getattr(s, "q", float("nan")), 4) for s in states]
        print(f"  {side:>5}: {msg_count[side]} msgs | q (rad): {qs}")


if __name__ == "__main__":
    main()
