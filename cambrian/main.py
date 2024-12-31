import argparse

from hydra_config import run_hydra

from cambrian import MjCambrianConfig, MjCambrianTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the model")
    action.add_argument("--eval", action="store_true", help="Evaluate the model")
    action.add_argument(
        "--eval-fast", action="store_true", help="Evaluate the model without saving"
    )
    action.add_argument("--test", action="store_true", help="Test the evo loop")

    def main(
        config: MjCambrianConfig,
        *,
        train: bool,
        eval: bool,
        eval_fast: bool,
        test: bool,
    ) -> float:
        """This method will return a float if training. The float represents the
        "fitness" of the agent that was trained. This can be used by hydra to
        determine the best hyperparameters during sweeps."""
        runner = MjCambrianTrainer(config)

        if train:
            return runner.train()
        elif eval:
            return runner.eval()
        elif eval_fast:
            return runner.eval_fast()
        elif test:
            return runner.test()

    run_hydra(main, parser=parser)
