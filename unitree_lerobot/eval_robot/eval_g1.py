"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import select
import sys
import termios
import time
import torch
import tty
import logging
import cv2

import numpy as np
import rerun as rr
from datetime import datetime
from pathlib import Path
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
    TimingLog,
    _REALTIME_BUDGET_MS,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

# Per-frame arm-joint delta cap for the policy loop.
# 0.5 rad ≈ 28.6° per frame; at 30 Hz that's ~15 rad/s peak joint velocity --
# well above fast-but-normal teleop (typically peaks at 2-3 rad/s = ~0.07 rad/frame).
# A misfiring policy that spikes a joint by >0.5 rad in one frame trips this and the loop aborts.
_MAX_ARM_DELTA_PER_FRAME = 0.5


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    # Rerun is "active" if either flag is set: --visualization spawns the live viewer,
    # --save_rrd records the session to disk. Image/scalar logging happens whenever
    # either is on; rr.spawn (viewer) is gated on --visualization specifically.
    rerun_active = cfg.visualization or cfg.save_rrd
    rerun_logger = None
    if rerun_active:
        rerun_logger = RerunLogger(spawn=cfg.visualization)

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # Initialized at outer scope so the `except`/`finally` branches can mutate / read them
    # even if the policy loop never starts (cam_check_only, soft_start-only, early returns).
    abort_reason = "completed"
    timing: TimingLog | None = None

    try:
        # --- Setup Phase ---
        logger_mp.info("Setting up image client...")
        image_client, image_config = setup_image_client(cfg)
        logger_mp.info("Image client ready.")
        logger_mp.info("Setting up robot interface...")
        robot_interface = setup_robot_interface(cfg)
        logger_mp.info("Robot interface ready.")

        # --- Unpack interfaces for convenience ---
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )

        # --- Cam check mode: pull one observation, save the four images, log shapes, exit.
        # Enable with --cam_check_only=true. Does NOT command any robot motion.
        # Verifies the head-camera binocular split + resize and confirms wrist feeds.
        if cfg.cam_check_only:
            out_dir = Path("cam_dryrun")
            out_dir.mkdir(parents=True, exist_ok=True)
            logger_mp.info(f"--cam_check_only: pulling one observation, saving to {out_dir.resolve()}")

            obs, arm_q = process_images_and_observations(image_client, image_config, arm_ctrl)
            for key, tensor in obs.items():
                if not key.startswith("observation.images."):
                    continue
                cam_name = key.removeprefix("observation.images.")
                if tensor is None:
                    logger_mp.warning(f"  {cam_name}: None (no frame received)")
                    continue
                # `to_tensor_rgb` produced an HWC RGB uint8 tensor; cv2.imwrite needs BGR.
                arr = tensor.numpy() if hasattr(tensor, "numpy") else tensor
                logger_mp.info(f"  {cam_name}: shape={tuple(arr.shape)} dtype={arr.dtype}")
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_dir / f"{cam_name}.png"), bgr)
                # Also stream into Rerun when either viz flag is on so cam_check is useful in the GUI.
                # arr is RGB HWC uint8 -- exactly what rr.Image expects, no further conversion.
                if rerun_active:
                    rr.log(f"images/{cam_name}", rr.Image(arr))
            logger_mp.info(f"  current_arm_q: {None if arm_q is None else tuple(arm_q.shape)}")
            logger_mp.info("Camera check complete. Exiting before robot motion.")
            return

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][1]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        logger_mp.info(f"Stages: soft_start={cfg.soft_start}, run_policy={cfg.run_policy}")
        if not cfg.soft_start and not cfg.run_policy:
            logger_mp.info("No stage flags set; nothing to do. Pass --soft_start=true and/or --run_policy=true.")
            return

        user_input = input("Enter 's' to begin (Ctrl+C to abort safely): ")
        if user_input.lower() != "s":
            logger_mp.info("Aborted by user before any motion.")
            return

        # --- Stage 1: soft-start (linear interpolation from current arm pose to init_arm_pose) ---
        # Gets the robot from arms-down (OOD for the policy) into the dataset's start pose so the
        # policy loop receives in-distribution observations. Per-step delta is bounded by
        # |target - current| / n_steps -- typically a fraction of a degree per step over 3 seconds.
        if cfg.soft_start:
            current_q = arm_ctrl.get_current_dual_arm_q()
            target_q = init_arm_pose
            soft_start_duration_s = 3.0
            n_steps = max(1, int(soft_start_duration_s * cfg.frequency))
            logger_mp.info(
                f"Soft-start: interpolating to init_arm_pose over {soft_start_duration_s:.1f}s "
                f"({n_steps} steps at {cfg.frequency} Hz)."
            )
            logger_mp.info(f"  current: {np.array2string(np.asarray(current_q), precision=3, suppress_small=True)}")
            logger_mp.info(f"  target:  {np.array2string(np.asarray(target_q),  precision=3, suppress_small=True)}")
            # Log per-joint delta so a human can eyeball the move before any motor command. The
            # per-step delta is delta/n_steps -- tiny by construction unless one of these vectors is bad.
            delta_q = np.asarray(target_q) - np.asarray(current_q)
            logger_mp.info(f"  delta:   {np.array2string(delta_q, precision=3, suppress_small=True)}")
            if not np.all(np.isfinite(current_q)) or not np.all(np.isfinite(target_q)):
                logger_mp.error("Non-finite values in current_q or target_q; aborting soft-start before motion.")
                return
            for i in range(1, n_steps + 1):
                alpha = i / n_steps
                q_step = (1.0 - alpha) * current_q + alpha * target_q
                tau = arm_ik.solve_tau(q_step)
                arm_ctrl.ctrl_dual_arm(q_step, tau)
                time.sleep(1.0 / cfg.frequency)

            time.sleep(0.5)  # brief settle before reading state in the policy loop
            logger_mp.info("Soft-start complete; robot at init_arm_pose.")

        # Initialize the gripper to the dataset's first-frame value (e.g., open for tasks that
        # start with an open gripper). This runs independently of soft_start because the gripper
        # uses scalar position commands -- it doesn't go through the radians-based arm interpolation.
        # The 14-arm + 2-gripper state layout matches the converter / ROBOT_CONFIGS; state[arm_dof:]
        # gives [left_ee, right_ee]. A single shared-memory write is enough -- the EE controller's
        # background thread picks it up and the gripper actuates within the settle window below.
        if cfg.ee:
            init_gripper_state = step["observation.state"][arm_dof:].cpu().numpy()
            left_init = init_gripper_state[:ee_dof]
            right_init = init_gripper_state[ee_dof : 2 * ee_dof]
            logger_mp.info(f"Init grippers from dataset frame 0: left={left_init} right={right_init}")
            if isinstance(ee_shared_mem["left"], SynchronizedArray):
                ee_shared_mem["left"][:] = to_list(left_init)
                ee_shared_mem["right"][:] = to_list(right_init)
            elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                ee_shared_mem["left"].value = to_scalar(left_init)
                ee_shared_mem["right"].value = to_scalar(right_init)
            time.sleep(0.3)  # let gripper actuate before the policy loop starts commanding it

        # --- Stage 2: policy loop ---
        if not cfg.run_policy:
            logger_mp.info("run_policy=false: skipping inference loop.")
            return

        # Second confirmation: if soft-start just ran, give the human a chance to physically
        # verify the robot is in the right pose before the policy starts driving it.
        if cfg.soft_start:
            confirm = input(
                "Soft-start complete. Verify the robot is at the expected init pose, then enter 's' "
                "to start the policy loop (anything else aborts): "
            )
            if confirm.lower() != "s":
                logger_mp.info("Aborted between soft-start and policy loop.")
                return



        logger_mp.info(
            f"Starting policy loop at {cfg.frequency} Hz "
            f"(max_steps={'unlimited' if cfg.max_steps == 0 else cfg.max_steps})."
        )

        # --- Realtime eval logging setup ---
        # Save dir sits sibling to the loaded checkpoint:
        #   <run>/realtime_eval/<YYYYMMDD_HHMMSS>/  (timing.csv, timing.npz, summary.txt, config.json)
        # cfg.policy.pretrained_path points at .../checkpoints/<step>/pretrained_model so .parents[2] is <run>.
        run_dir = Path(cfg.policy.pretrained_path).resolve().parents[2]
        timing_dir = run_dir / "realtime_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
        timing = TimingLog(out_dir=timing_dir)
        logger_mp.info(f"Realtime eval logs -> {timing_dir}")

        # Opt-in Rerun session recording. rr.save adds a file sink to the active recording;
        # everything we rr.log from this point on goes to the .rrd. Replay later with `rerun session.rrd`.
        if cfg.save_rrd:
            if not rerun_active:
                # rr.init wasn't called (neither --visualization nor --save_rrd activated RerunLogger).
                # This branch only triggers if save_rrd=true got somehow set without rerun_active true,
                # which the logic above prevents. Belt-and-suspenders.
                rr.init(f"eval_g1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            rrd_path = timing_dir / "session.rrd"
            rr.save(str(rrd_path))
            logger_mp.info(f"Rerun session recording -> {rrd_path}")

        idx = 0
        full_state = None

        # Seed the per-frame delta guard with the robot's *actual* current pose so the first frame's
        # check catches "the policy's first command is far from where the robot is right now."
        last_arm_action = np.asarray(arm_ctrl.get_current_dual_arm_q()).copy()
        logger_mp.info(f"Per-frame arm-delta cap: {_MAX_ARM_DELTA_PER_FRAME} rad (~{np.degrees(_MAX_ARM_DELTA_PER_FRAME):.1f}° per frame at {cfg.frequency} Hz).")

        # Emergency-stop key setup: put stdin into cbreak so 'q' is captured as a single keypress
        # (no Enter required). Only applies if stdin is a TTY -- if piped/redirected, fall back
        # gracefully to no stop key (Ctrl+C still works).
        stdin_is_tty = sys.stdin.isatty()
        old_term_settings = None
        if stdin_is_tty:
            try:
                old_term_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
                logger_mp.info("Emergency stop: press 'q' (no Enter) at any time during the loop to abort safely.")
            except Exception as e:
                logger_mp.warning(f"Could not set terminal to cbreak mode; 'q' emergency stop disabled. ({e})")
                old_term_settings = None

        while cfg.max_steps == 0 or idx < cfg.max_steps:
            loop_start_time = time.perf_counter()

            # === Stage A: observations ===
            t_obs_start = time.perf_counter()
            observation, current_arm_q = process_images_and_observations(
                image_client, image_config, arm_ctrl
            )
            left_ee_state = right_ee_state = np.array([])

            if cfg.ee:
                with ee_shared_mem["lock"]:
                    full_state = np.array(ee_shared_mem["state"][:])
                    left_ee_state = full_state[:ee_dof]
                    right_ee_state = full_state[ee_dof:]
            state_tensor = torch.from_numpy(
                np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
            ).float()
            observation["observation.state"] = state_tensor
            t_obs_ms = (time.perf_counter() - t_obs_start) * 1000.0

            # === Stage B: policy inference ===
            # Peek at the action queue BEFORE select_action to detect chunk boundaries (queue empty
            # → this call triggers a fresh forward pass; non-empty → it just pops a cached action).
            # This is the single most diagnostic signal for the half-second-beat jitter hypothesis.
            queue_len_before = len(getattr(policy, "_action_queue", []))
            chunk_boundary = (queue_len_before == 0)
            # Explicit CUDA sync so the measured time reflects real GPU compute, not async-enqueue.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer_start = time.perf_counter()
            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=cfg.use_dataset,
                robot_type=None,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            action_np = action.cpu().numpy()

            # NaN / inf guard -- fail fast rather than forward garbage to the motors.
            if not np.all(np.isfinite(action_np)):
                logger_mp.error(f"Non-finite values in policy action: {action_np}. Aborting loop.")
                abort_reason = "nan_guard"
                break

            # 3. Execute Action
            arm_action = action_np[:arm_dof]

            # Per-frame arm-joint delta guard. Compare to the previous commanded arm pose; if any
            # single arm joint would jump by more than _MAX_ARM_DELTA_PER_FRAME radians in one
            # frame, that's outside the bounds of plausible normal motion -- abort and inspect.
            # Gripper dims (action_np[arm_dof:]) are excluded by construction.
            arm_delta = arm_action - last_arm_action
            max_abs_delta = float(np.max(np.abs(arm_delta)))
            if max_abs_delta > _MAX_ARM_DELTA_PER_FRAME:
                worst_joint = int(np.argmax(np.abs(arm_delta)))
                logger_mp.error(
                    f"Large arm-joint delta at step {idx}: "
                    f"joint[{worst_joint}] changed by {arm_delta[worst_joint]:+.4f} rad "
                    f"(|max|={max_abs_delta:.4f} > cap {_MAX_ARM_DELTA_PER_FRAME}). Aborting loop."
                )
                abort_reason = "delta_cap"
                break

            # === Stage C: IK (gravity-comp torques) ===
            t_tau_start = time.perf_counter()
            tau = arm_ik.solve_tau(arm_action)
            t_tau_ms = (time.perf_counter() - t_tau_start) * 1000.0

            # === Stage D: motor command + EE shared-mem write ===
            t_ctrl_start = time.perf_counter()
            arm_ctrl.ctrl_dual_arm(arm_action, tau)
            last_arm_action = arm_action.copy()

            if cfg.ee:
                ee_action_start_idx = arm_dof
                left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]

                if isinstance(ee_shared_mem["left"], SynchronizedArray):
                    ee_shared_mem["left"][:] = to_list(left_ee_action)
                    ee_shared_mem["right"][:] = to_list(right_ee_action)
                elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                    ee_shared_mem["left"].value = to_scalar(left_ee_action)
                    ee_shared_mem["right"].value = to_scalar(right_ee_action)
            t_ctrl_ms = (time.perf_counter() - t_ctrl_start) * 1000.0

            # === Loop close: deadline check + frequency maintenance ===
            t_before_sleep = time.perf_counter()
            sleep_budget_s = (1.0 / cfg.frequency) - (t_before_sleep - loop_start_time)
            t_sleep_ms = max(0.0, sleep_budget_s) * 1000.0
            missed_deadline = sleep_budget_s <= 0
            if missed_deadline:
                logger_mp.warning(
                    f"[step {idx}] missed {cfg.frequency:.0f}Hz deadline: "
                    f"loop={(t_before_sleep - loop_start_time) * 1000.0:.1f}ms "
                    f"(obs={t_obs_ms:.1f} infer={t_infer_ms:.1f} tau={t_tau_ms:.1f} ctrl={t_ctrl_ms:.1f}) "
                    f"chunk_boundary={chunk_boundary}"
                )
            time.sleep(max(0.0, sleep_budget_s))
            t_loop_ms = (time.perf_counter() - loop_start_time) * 1000.0

            # === Record per-step timing ===
            timing.append(
                step=idx,
                t_obs_ms=t_obs_ms,
                t_infer_ms=t_infer_ms,
                t_tau_ms=t_tau_ms,
                t_ctrl_ms=t_ctrl_ms,
                t_loop_ms=t_loop_ms,
                t_sleep_ms=t_sleep_ms,
                chunk_boundary=int(chunk_boundary),
                queue_len_before=queue_len_before,
                missed_deadline=int(missed_deadline),
                arm_delta_max=max_abs_delta,
            )
            timing.add_step_data(action=action_np, state=state_tensor.numpy())

            if rerun_active:
                rr.set_time("frame", sequence=idx)
                rr.log("timings/process_obs_ms",    rr.Scalars(t_obs_ms))
                rr.log("timings/predict_action_ms", rr.Scalars(t_infer_ms))
                rr.log("timings/solve_tau_ms",      rr.Scalars(t_tau_ms))
                rr.log("timings/ctrl_arm_ms",       rr.Scalars(t_ctrl_ms))
                rr.log("timings/loop_total_ms",     rr.Scalars(t_loop_ms))
                rr.log("timings/budget_ms",         rr.Scalars(_REALTIME_BUDGET_MS))
                rr.log("events/chunk_boundary",    rr.Scalars(1 if chunk_boundary else 0))
                rr.log("events/missed_deadline",   rr.Scalars(1 if missed_deadline else 0))
                rr.log("events/queue_len_before",  rr.Scalars(queue_len_before))
                visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)

            # Heartbeat every 30 steps so the console gives running feedback even without --visualization.
            if idx > 0 and idx % 30 == 0:
                recent = timing.rows[-30:]
                rec_infer = [float(r["t_infer_ms"]) for r in recent]
                n_boundary = sum(int(r["chunk_boundary"]) for r in recent)
                n_missed = sum(int(r["missed_deadline"]) for r in recent)
                logger_mp.info(
                    f"[step {idx}] last 30: infer mean={np.mean(rec_infer):.1f}ms "
                    f"max={np.max(rec_infer):.1f}ms | chunk_boundaries={n_boundary} | "
                    f"missed_deadlines={n_missed}"
                )

            idx += 1

            # Emergency stop: non-blocking read of stdin. If 'q' was pressed, abort cleanly.
            # Worst-case latency between keypress and abort is one loop period (~33ms at 30Hz).
            if old_term_settings is not None and select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    logger_mp.warning("Emergency stop requested ('q' pressed). Aborting policy loop.")
                    abort_reason = "q_pressed"
                    break

        logger_mp.info(f"Policy loop completed after {idx} steps.")
    except KeyboardInterrupt:
        abort_reason = "ctrl_c"
        logger_mp.warning("Ctrl+C received; flushing timing data before exit.")
    except Exception as e:
        abort_reason = f"exception:{type(e).__name__}"
        logger_mp.info(f"An error occurred: {e}")
    finally:
        # Restore terminal mode if we put it into cbreak for the emergency stop key.
        # Critical: leaving the terminal in cbreak after the script exits makes the shell unusable.
        if locals().get("old_term_settings") is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, locals()["old_term_settings"])
            except Exception as term_err:
                logger_mp.warning(f"Failed to restore terminal mode: {term_err}")
        # Guard with locals() in case setup failed before image_client was assigned.
        if "image_client" in locals():
            try:
                image_client.close()
            except Exception as close_err:
                logger_mp.warning(f"Failed to close image_client cleanly: {close_err}")
        # Flush realtime-eval timing data. Idempotent + safe even if the loop never started
        # (timing stays None for cam_check_only / soft_start-only / early-return paths).
        if timing is not None:
            try:
                timing.finalize(abort_reason=abort_reason, cfg_snapshot=asdict(cfg))
                logger_mp.info(f"Timing logs flushed to {timing.out_dir} (abort_reason={abort_reason})")
            except Exception as e:
                logger_mp.error(f"Failed to finalize timing log: {e}")


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
