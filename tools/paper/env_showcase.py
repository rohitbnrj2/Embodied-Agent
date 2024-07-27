from cambrian.ml.trainer import MjCambrianTrainer
from cambrian.utils.config import MjCambrianConfig, run_hydra


def main(config: MjCambrianConfig):
    trainer = MjCambrianTrainer(config)
    trainer.eval()


if __name__ == "__main__":
    run_hydra(main)
