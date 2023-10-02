import os
import math
import argparse
from typing import Union, List, Tuple, Any
from pathlib import Path
from prodict import Prodict
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

from cambrian.reinforce.env_v2 import make_env
from cambrian.reinforce.evo import Agent, AgentPool, AgentPoolFactory, AgentPoolCallback
from cambrian.reinforce.models import MultiInputFeatureExtractor
from cambrian.reinforce.evo.utils import write_yaml, load_config


def _update_config_with_overrides(config: Prodict, overrides: List[Tuple[str, Any]]):
    """Helper method to update the config based on some cli override arguments. The override string is the dot separated yaml config key, and the value is the value to set it to."""
    for k, v in overrides:
        keys = k.split(".")
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = type(config[keys[-1]])(v)


class EvoRunner:
    def __init__(
        self,
        config: Union[Path, str, Prodict],
        overrides: List[Tuple[str, Any]],
        rank: int,
    ):
        self.config = load_config(config)
        self.rank = rank
        self.config.animal_config.rank = rank

        _update_config_with_overrides(self.config, overrides)

        self.env_config = self.config.env_config
        self.evo_config = self.config.evo_config
        self.verbose = self.config.env_config.verbose

        self.epoch = 0

        self.logdir = Path(self.env_config.logdir) / self.env_config.exp_name
        self.logdir.mkdir(parents=True, exist_ok=True)

        self.agent: Agent = None
        self.agent_pool: AgentPool = AgentPoolFactory(
            self.config, self.rank, verbose=self.verbose
        )
        self.agent_pool.write_to_pool(-math.inf, self.config.animal_config)

    def start(self):
        """Begin the evolution.

        This does 4 things:
        1. Select from the best performing agents to train (or use the initial config if it's the first generation)
        2. Mutate it
        3. Train the agent.
        4. Repeat until max number of generations is reached.
        """
        while self.epoch < self.evo_config.num_generations:
            print(f"Starting epoch {self.epoch}...")
            self.select_agent()
            self.mutate_agent()
            self.train_agent()

            self.epoch += 1

        self.agent_pool.close()

    def select_agent(self):
        new_agent_config = self.agent_pool.get_new_agent_config()
        self.agent = Agent(new_agent_config, verbose=self.verbose)

    def mutate_agent(self):
        self.agent.modify(self.evo_config.modification_type)
        self.config.animal_config = self.agent.config

    def train_agent(self):
        verbose = self.verbose

        # Agent metadata
        current_logdir = self.logdir / f"generation_{self.epoch}" / f"rank_{self.rank}"
        ppodir = current_logdir / "ppo"
        ppodir.mkdir(parents=True, exist_ok=True)
        write_yaml(self.config, current_logdir / "config.yml")

        env = self._make_env(ppodir)

        check_frequency = self.env_config.check_freq
        agent_pool_cb = AgentPoolCallback(self.agent, self.agent_pool, verbose=verbose)
        stop_training_on_reward_cb = StopTrainingOnRewardThreshold(
            reward_threshold=self.env_config.reward_threshold, verbose=verbose
        )
        eval_cb = EvalCallback(
            env,
            callback_on_new_best=agent_pool_cb,
            callback_after_eval=stop_training_on_reward_cb,
            eval_freq=check_frequency,
            best_model_save_path=current_logdir,
            log_path=current_logdir,
            verbose=verbose,
        )

        policy_kwargs = dict(
            features_extractor_class=MultiInputFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=self.env_config.n_steps,
            batch_size=self.env_config.batch_size,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
        )
        
        #th.save()
        model.learn(total_timesteps=self.env_config.total_timesteps, callback=eval_cb)
        
        #save model weights
        th.save(model.policy.state_dict(), "ppodir/model_weights.pth")
        
        env.close()
    
    def transfer_weights(self, child_model, parent_weights_file):
        #copies all weights into the child model except the ones in first layer
        
        input_layer_keys = ["mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"]
        
        parent_model_dict = th.load(parent_weights_file)
        
        child_model_dict = child_model.policy.state_dict()
        
        for k in parent_model_dict.keys():
            
            if k not in input_layer_keys:
                
                child_model_dict[k] = parent_model_dict[k]
        
        del parent_model_dict
        
                
        

    def _make_env(self, ppodir: Path) -> VecEnv:
        env = SubprocVecEnv(
            [
                make_env(
                    self.rank,
                    self.env_config.seed,
                    self.config,
                    i,
                    animal_override=self.agent,
                )
                for i in range(self.env_config.num_cpu)
            ]
        )
        env = VecMonitor(env, ppodir.as_posix())
        return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed evolutionary training with synchornization after each generation."
    )
    parser.add_argument("config", type=str, help="The config file to use.")
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        type=lambda v: v.split("="),
        help="Override config values. Do <dot separated yaml config>=<value>",
        default=[],
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        help="The rank of this process. Used for distributed training. Defaults to the SLURM_ARRAY_TASK_ID environment variable.",
        default=os.environ.get("SLURM_ARRAY_TASK_ID", None),
    )
    args = parser.parse_args()

    for o in args.overrides:
        if len(o) != 2:
            raise ValueError(
                "Override must be in the form <dot separated yaml config>=<value>"
            )

    if args.rank is None:
        raise ValueError(
            "Rank cannot be determined. Pass rank using the `-r` argument."
        )

    runner = EvoRunner(args.config, args.overrides, args.rank)
    runner.start()
