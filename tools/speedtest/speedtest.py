#!/usr/bin/env python3
import argparse
import concurrent
import copy
from pathlib import Path
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures.process import ProcessPoolExecutor
import subprocess
from time import perf_counter

import mujoco
import numpy as np

N_TIME = 10000


def rollout_one_trajectory(model, data, controls):
    qs = []
    for t in range(N_TIME):
        control_t = controls[t]
        np.copyto(data.ctrl, control_t)
        mujoco.mj_step(model, data)
        qs.append(data.qpos.copy())
    return qs


def call_rollout(model, ctrl):
    data = mujoco.MjData(model)
    rollout_one_trajectory(model, data, ctrl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xml_path", type=str)
    parser.add_argument("mode", type=str, choices=["thread", "process", "subprocess"])
    parser.add_argument("-n", "--nthreads", type=int, default=8)

    args = parser.parse_args()
    args_lists = copy.deepcopy(args)

    if not Path(args_lists.xml_path).exists():
        args_lists.xml_path = Path(__file__).parent / "models" / args_lists.xml_path
        assert (
            args_lists.xml_path.exists()
        ), f"File {args_lists.xml_path} does not exist"
    model = mujoco.MjModel.from_xml_path(str(args_lists.xml_path))

    controls = np.random.randn(N_TIME, model.nu)

    args_lists = [controls] * args.nthreads

    timestep_ms = model.opt.timestep * 1000
    print(f"Running {N_TIME} steps per thread at dt = {timestep_ms} ms ...")

    t0 = perf_counter()
    if args.mode == "thread":
        with ThreadPoolExecutor(max_workers=args.nthreads) as pool:
            futures = []
            for args_list in args_lists:
                futures.append(pool.submit(call_rollout, model, args_list))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    elif args.mode == "process":
        with ProcessPoolExecutor(max_workers=args.nthreads) as pool:
            futures = []
            for args_list in args_lists:
                futures.append(pool.submit(call_rollout, model, args_list))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    elif args.mode == "subprocess":
        processes = []
        cmd = f"python {__file__} {args.xml_path} thread -n 1"
        for args_list in args_lists:
            processes.append(
                subprocess.Popen(cmd.split(" "), stdout=subprocess.DEVNULL)
            )

        for process in processes:
            process.wait()
    total_dt = perf_counter() - t0
    steps_per_second = N_TIME * args.nthreads / total_dt
    rtf = steps_per_second * model.opt.timestep
    time_per_step = total_dt / (N_TIME * args.nthreads) * 1000

    print(f"Total simulation time  : {total_dt:.2f} s")
    print(f"Total steps per second : {steps_per_second:.0f}")
    print(f"Total realtime factor  : {rtf:.3f} x")
    print(f"Total time per step    : {time_per_step:.3f} ms")


if __name__ == "__main__":
    main()
