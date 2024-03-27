import time

from cambrian.utils.config import MjCambrianConfig, run_hydra

def main(config: MjCambrianConfig):
    t0 = time.time()
    for _ in range(1000):
        for _ in config.env.animals["point"].eyes_lat_range:
            pass
    t1 = time.time()

    print(f"Time: {(t1 - t0) * 1000:.2f} ms")

if __name__ == "__main__":
    run_hydra(main)