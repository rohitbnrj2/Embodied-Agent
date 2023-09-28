from typing import TypeVar, Union, List
from enum import Flag, auto
from prodict import Prodict
from functools import reduce
import numpy as np

from cambrian.evolution_envs.animal import OculozoicAnimal

AgentType = TypeVar("AgentType", bound="Agent")


class Agent(OculozoicAnimal):
    """Agent class that wraps an BeeSim. It handles mutation and/or crossover and fitness evaluation.

    One agent will be trained at a time with a vectorized environment in a single job,i.e. one job per agent.

    Furthermore, the agent will be considered done with training after two possible conditions: (i) The agent has reached the maximum number of training epochs, and (ii) The agent has reached a certain fitness threshold.
    """

    class ModificationType(Flag):
        """Use as bitmask to specify which type of modification to perform on the agent.

        Example:
        >>> # Only mutation
        >>> type = ModificationType.MUTATION
        >>> # Both mutation and crossover
        >>> type = ModificationType.MUTATION | ModificationType.CROSSOVER
        """

        MUTATION = auto()
        CROSSOVER = auto()

    class MutationType(Flag):
        """Use as bitmask to specify which type of mutation to perform on the agent.

        Example:
        >>> # Only adding a photoreceptor
        >>> type = MutationType.ADD_PHOTORECEPTOR
        >>> # Both adding a photoreceptor and changing a simple eye to lens
        >>> type = MutationType.ADD_PHOTORECEPTOR | MutationType.SIMPLE_TO_LENS
        """

        ADD_PHOTORECEPTOR = auto()
        SIMPLE_TO_LENS = auto()
        ADD_PIXEL = auto()
        UPDATE_PIXEL = auto()

    ModificationTypeMap = {
        "MUTATION": ModificationType.MUTATION,
        "CROSSOVER": ModificationType.CROSSOVER,
    }

    MutationOptions = [
        MutationType.ADD_PHOTORECEPTOR,
        MutationType.SIMPLE_TO_LENS,
        MutationType.ADD_PIXEL,
        MutationType.UPDATE_PIXEL,
    ]

    def __init__(self, config: Prodict, *, verbose: int = 0):
        super().__init__(config)
        self.verbose = verbose

    def _parse_modification_type(
        self, modification_type: Union[ModificationType, str, List[str]]
    ):
        if isinstance(modification_type, str):
            modification_type = Agent.ModificationTypeMap[modification_type]
        elif isinstance(modification_type, list):
            modification_type = [
                self._parse_modification_type(m) for m in modification_type
            ]
            modification_type = reduce(lambda x, y: x | y, modification_type)
        return modification_type

    def modify(self, modification_type: Union[ModificationType, str]):
        """Modify the agent according to the specified modification type."""
        modification_type = self._parse_modification_type(modification_type)

        if self.verbose > 2:
            print(f"Modifying agent with modification type {modification_type}")

        if modification_type & Agent.ModificationType.CROSSOVER:
            self.crossover()

        if modification_type & Agent.ModificationType.MUTATION:
            self.mutate()

    def mutate(self, mutations: MutationType = None):
        """Mutate the agent. Mutation differs from crossover in that it is asexual reproduction (or mutation of properties post-reproduction), i.e. no combination/crossover with another agent.

        NOTE: If crossover AND mutation is specified, then crossover will occur first, followed by mutation. This basically implies that mutation in this scenario is random variation in offspring not attributed to the parents.
        """
        if self.verbose > 2:
            print("Mutating agent...")
        if mutations is None:
            number_of_mutations = np.random.randint(1, len(Agent.MutationOptions) + 1)
            mutations = np.random.choice(
                Agent.MutationOptions, size=number_of_mutations, replace=False
            )
            mutations = reduce(lambda x, y: x | y, mutations)

        if mutations & Agent.MutationType.ADD_PHOTORECEPTOR:
            if self.verbose > 1:
                print("Adding photoreceptor...")
            self.config.init_photoreceptors += self.config.increment_photoreceptor
            self.config.init_photoreceptors = min(
                self.config.init_photoreceptors, self.config.max_photoreceptors
            )

        if mutations & Agent.MutationType.SIMPLE_TO_LENS:
            if self.verbose > 1:
                print("Changing simple eye to lens...")
            for i, eye in enumerate(self.config.init_configuration.imaging_model):
                if eye == "simple":
                    self.config.init_configuration.imaging_model[i] = "lens"

        if mutations & Agent.MutationType.ADD_PIXEL:
            if self.verbose > 1:
                print("Adding pixel...")

            # TODO: Does it need to be symmetric?
            imaging_model = np.random.choice(["simple", "lens"])
            fov = np.random.uniform(0, 180)
            angle = 0  # np.random.uniform(0, 180)
            sensor_size = np.random.uniform(0, 20)

            init_config = self.config.init_configuration
            init_config.num_pixels += 2
            init_config.direction.extend(["left", "right"])
            init_config.imaging_model.extend([imaging_model, imaging_model])
            init_config.sensor_size.extend([sensor_size, sensor_size])
            init_config.angle.extend([angle, angle])
            init_config.fov.extend([fov, fov])

        if mutations & Agent.MutationType.UPDATE_PIXEL:
            if self.verbose > 1:
                print("Updating pixel...")

            init_config = self.config.init_configuration
            # TODO: Does it need to be symmetric? This one doesn't maintain symmetry
            pixel_idx = np.random.randint(0, self.config.init_configuration.num_pixels)

            # add random noise at roughly 10% of the original magnitude
            def add_noise(x):
                return x + np.random.normal(0, 0.1 * x)

            imaging_model = np.random.choice(["simple", "lens"])
            sensor_size = add_noise(init_config.sensor_size[pixel_idx])
            angle = init_config.angle[
                pixel_idx
            ]  # add_noise(init_config.angle[pixel_idx])
            fov = add_noise(init_config.fov[pixel_idx])

            # init_config.imaging_model[pixel_idx] = imaging_model # NOTE: keep imaging model the same
            init_config.sensor_size[pixel_idx] = sensor_size
            init_config.angle[pixel_idx] = angle
            init_config.fov[pixel_idx] = fov

    def crossover(self):
        """Crossover the agent with another agent. Crossover differs from mutation in that it is sexual reproduction, i.e. there is a combination/crossover of genes with another agent."""
        raise NotImplementedError
