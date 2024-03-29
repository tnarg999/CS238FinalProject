"""Malfunction generators for rail systems"""

from typing import Callable, NamedTuple, Optional, Tuple

import msgpack
import numpy as np
from numpy.random.mtrand import RandomState

from flatland.envs.agent_utils import EnvAgent, RailAgentStatus

Malfunction = NamedTuple('Malfunction', [('num_broken_steps', int)])
MalfunctionParameters = NamedTuple('MalfunctionParameters',
                                   [('malfunction_rate', float), ('min_duration', int), ('max_duration', int)])
MalfunctionGenerator = Callable[[EnvAgent, RandomState, bool], Optional[Malfunction]]
MalfunctionProcessData = NamedTuple('MalfunctionProcessData',
                                    [('malfunction_rate', float), ('min_duration', int), ('max_duration', int)])


def _malfunction_prob(rate: float) -> float:
    """
    Probability of a single agent to break. According to Poisson process with given rate
    :param rate:
    :return:
    """
    if rate <= 0:
        return 0.
    else:
        return 1 - np.exp(- (1 / rate))


def malfunction_from_file(filename: str) -> Tuple[MalfunctionGenerator, MalfunctionProcessData]:
    """
    Utility to load pickle file

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    with open(filename, "rb") as file_in:
        load_data = file_in.read()
    data = msgpack.unpackb(load_data, use_list=False, encoding='utf-8')
    # TODO: make this better by using namedtuple in the pickle file. See issue 282
    data['malfunction'] = MalfunctionProcessData._make(data['malfunction'])
    if "malfunction" in data:
        # Mean malfunction in number of time steps
        mean_malfunction_rate = data["malfunction"].malfunction_rate

        # Uniform distribution parameters for malfunction duration
        min_number_of_steps_broken = data["malfunction"].min_duration
        max_number_of_steps_broken = data["malfunction"].max_duration
    else:
        # Mean malfunction in number of time steps
        mean_malfunction_rate = 0.
        # Uniform distribution parameters for malfunction duration
        min_number_of_steps_broken = 0
        max_number_of_steps_broken = 0

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        """
        Generate malfunctions for agents
        Parameters
        ----------
        agent
        np_random

        Returns
        -------
        int: Number of time steps an agent is broken
        """

        # Dummy reset function as we don't implement specific seeding here
        if reset:
            return Malfunction(0)

        if agent.malfunction_data['malfunction'] < 1:
            if np_random.rand() < _malfunction_prob(mean_malfunction_rate):
                num_broken_steps = np_random.randint(min_number_of_steps_broken,
                                                     max_number_of_steps_broken + 1) + 1
                return Malfunction(num_broken_steps)
        return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)


def malfunction_from_params(parameters: MalfunctionParameters) -> Tuple[MalfunctionGenerator, MalfunctionProcessData]:
    """
    Utility to load malfunction from parameters

    Parameters
    ----------

    parameters : contains all the parameters of the malfunction
        malfunction_rate : float how many time steps it takes for a sinlge agent befor it breaks
        min_duration : int minimal duration of a failure
        max_number_of_steps_broken : int maximal duration of a failure

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    mean_malfunction_rate = parameters.malfunction_rate
    min_number_of_steps_broken = parameters.min_duration
    max_number_of_steps_broken = parameters.max_duration

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        """
        Generate malfunctions for agents
        Parameters
        ----------
        agent
        np_random

        Returns
        -------
        int: Number of time steps an agent is broken
        """

        # Dummy reset function as we don't implement specific seeding here
        if reset:
            return Malfunction(0)

        if agent.malfunction_data['malfunction'] < 1:
            if np_random.rand() < _malfunction_prob(mean_malfunction_rate):
                num_broken_steps = np_random.randint(min_number_of_steps_broken,
                                                     max_number_of_steps_broken + 1) + 1
                return Malfunction(num_broken_steps)
        return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)


def no_malfunction_generator() -> Tuple[MalfunctionGenerator, MalfunctionProcessData]:
    """
    Malfunction generator which generates no malfunctions

    Parameters
    ----------
    Nothing

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    # Mean malfunction in number of time steps
    mean_malfunction_rate = 0.

    # Uniform distribution parameters for malfunction duration
    min_number_of_steps_broken = 0
    max_number_of_steps_broken = 0

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)


def single_malfunction_generator(earlierst_malfunction: int, malfunction_duration: int) -> Tuple[
    MalfunctionGenerator, MalfunctionProcessData]:
    """
    Malfunction generator which guarantees exactly one malfunction during an episode of an ACTIVE agent.

    Parameters
    ----------
    malfunction_duration: The duration of the single malfunction

    Returns
    -------
    generator, Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """
    # Mean malfunction in number of time steps
    mean_malfunction_rate = 0.

    # Uniform distribution parameters for malfunction duration
    min_number_of_steps_broken = 0
    max_number_of_steps_broken = 0

    # Keep track of the total number of malfunctions in the env
    global_nr_malfunctions = 0

    # Malfunction calls per agent
    malfunction_calls = dict()

    def generator(agent: EnvAgent = None, np_random: RandomState = None, reset=False) -> Optional[Malfunction]:
        # We use the global variable to assure only a single malfunction in the env
        nonlocal global_nr_malfunctions
        nonlocal malfunction_calls

        # Reset malfunciton generator
        if reset:
            nonlocal global_nr_malfunctions
            nonlocal malfunction_calls
            global_nr_malfunctions = 0
            malfunction_calls = dict()
            return Malfunction(0)

        # No more malfunctions if we already had one, ignore all updates
        if global_nr_malfunctions > 0:
            return Malfunction(0)

        # Update number of calls per agent
        if agent.handle in malfunction_calls:
            malfunction_calls[agent.handle] += 1
        else:
            malfunction_calls[agent.handle] = 1

        # Break an agent that is active at the time of the malfunction
        if agent.status == RailAgentStatus.ACTIVE and malfunction_calls[agent.handle] >= earlierst_malfunction:
            global_nr_malfunctions += 1
            return Malfunction(malfunction_duration)
        else:
            return Malfunction(0)

    return generator, MalfunctionProcessData(mean_malfunction_rate, min_number_of_steps_broken,
                                             max_number_of_steps_broken)
