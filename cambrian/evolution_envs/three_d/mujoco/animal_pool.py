from typing import Tuple, List
from abc import ABC, abstractmethod
from pathlib import Path
from collections import deque
import bisect
import random
import fcntl
import math
import pickle
import time
from enum import Enum

from config import MjCambrianConfig

Performance = int


class MjCambrianAnimalPoolType(Enum):
    SHARED_FILE = "shared_file"


class MjCambrianAnimalPool(ABC):
    """This is the base class for an animal pool. It should be seen as a database of
    animal configs and their performance scores. As animals are trained over generations,
    runners will query the animal pool to select the best performing animal(s) and
    mutate from them."""

    def __init__(self, config: MjCambrianConfig, rank: int):
        self.config = config
        self.evo_config = config.evo_config
        self.training_config = config.training_config

        self.population_size = self.evo_config.population_size
        self.num_top_performers = self.evo_config.num_top_performers

        self.rank = rank
        self.verbose = config.training_config.verbose

        self.pool = deque(maxlen=self.population_size + 1)
        self.write_to_pool(-math.inf, self.config)

    def get_new_config(self) -> MjCambrianConfig:
        pool = self.get_pool()
        top_performer_pool = pool[: min(len(pool), self.num_top_performers)]
        return random.choice(top_performer_pool)[1].copy()

    def get_pool(self) -> List[Tuple[Performance, MjCambrianConfig]]:
        """Get the animal pool. This may be overridden based on the different storage
        mechanism.

        Returns:
            List[Tuple[Performance, Prodict]]: A **sorted** list from worst to best of
                animal configs in terms of performance
        """
        return list(self.pool)

    @abstractmethod
    def write_to_pool(self, performance: Performance, config: MjCambrianConfig):
        pass

    def close(self):
        pass

    def __del__(self):
        self.close()

    @staticmethod
    def create(config: MjCambrianConfig, rank: int) -> "MjCambrianAnimalPool":
        animal_pool_type = MjCambrianAnimalPoolType(config.evo_config.animal_pool_type)
        if animal_pool_type == MjCambrianAnimalPoolType.SHARED_FILE:
            return MjCambrianSharedFileAnimalPool(config, rank=rank)
        else:
            raise ValueError(f"Unknown animal pool type `{animal_pool_type}`")

    def _insert_into_pool(self, performance: Performance, config: MjCambrianConfig):
        """Insert a new performance score and animal config into the pool. This may be
        overridden based on the different storage mechanism. This method guarantees
        that the pool is always sorted from worst to best."""
        position = bisect.bisect_left([perf for perf, _ in self.pool], performance)
        if self.verbose > 1:
            print(
                f"{self.rank}: Performance {performance:0.2f} received. "
                f"Current pool: {[p for p,_ in self.pool]}. Inserting into pool at "
                f"position {position}."
            )
        self.pool.insert(position, (performance, config))
        if len(self.pool) == self.population_size + 1:
            self.pool.popleft()


class MjCambrianSharedFileAnimalPool(MjCambrianAnimalPool):
    """This AnimalPool implements the storage mechanism as a shared file.

    If `use_flock` is set to True, then the file will be locked using `fcntl` to ensure
    that only one process can write to the file at a time. If `use_flock` is set to
    False, then a lock file will be created and the process will busy wait until the
    lock file is removed. This is only recommended for systems that don't support file
    locking.
    """

    def __init__(self, config: MjCambrianConfig, rank: int):
        self.shared_file_config = config.evo_config.shared_file_config

        self.filename: Path = (
            Path(config.training_config.logdir)
            / config.training_config.exp_name
            / self.shared_file_config.filename
        )

        self.lock_file = self.filename.with_suffix(".lock")
        if self.shared_file_config.use_flock:
            self.lock_file.touch()
            try:
                fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_EX)
                fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_UN)
            except OSError:
                raise OSError(
                    "ERROR: This OS doesn't support fcntl. Set `use_flock` to False or "
                    "pick another animal pool type."
                )

        with self:
            self.filename.touch()

        super().__init__(config, rank)

    def get_pool(self) -> List[Tuple[Performance, MjCambrianConfig]]:
        with self:
            if self.filename.stat().st_size > 0:
                with open(self.filename, "rb") as f:
                    self.pool = pickle.load(f)
        return super().get_pool()

    def write_to_pool(self, performance: Performance, config: MjCambrianConfig):
        with self:
            if self.filename.stat().st_size > 0:
                with open(self.filename, "rb") as f:
                    self.pool = pickle.load(f)

            self._insert_into_pool(performance, config)

            with open(self.filename, "wb") as f:
                pickle.dump(self.pool, f)

    def _acquire_lock(self):
        if self.shared_file_config.use_flock:
            fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_EX)
        else:
            timeout = self.shared_file_config.lock_timeout
            while timeout > 0:
                try:
                    self.lock_file.touch(exist_ok=False)
                    break
                except FileExistsError:
                    timeout -= 0.1
                    time.sleep(0.1)

    def _release_lock(self):
        if self.shared_file_config.use_flock:
            fcntl.flock(open(self.lock_file, "w"), fcntl.LOCK_UN)
        else:
            self.lock_file.unlink()

    def __enter__(self) -> "MjCambrianSharedFileAnimalPool":
        self._acquire_lock()
        return self

    def __exit__(self, *args):
        self._release_lock()
