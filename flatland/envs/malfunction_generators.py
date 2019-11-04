"""Malfunction generators for rail systems"""

from typing import Tuple, Callable

import msgpack

MalfunctionGenerator = Callable[[], Tuple[float, int, int]]


def malfunction_from_file(filename) -> MalfunctionGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """

    def generator():
        with open(filename, "rb") as file_in:
            load_data = file_in.read()
        data = msgpack.unpackb(load_data, use_list=False, encoding='utf-8')

        if "malfunction" in data:
            # Mean malfunction in number of time steps
            mean_malfunction_rate = data["malfunction"]["malfunction_rate"]
            # Uniform distribution parameters for malfunction duration
            min_number_of_steps_broken = data["malfunction"]["min_duration"]
            max_number_of_steps_broken = data["malfunction"]["max_duration"]
            agents_speed = None
        return mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken

    return generator


def malfunction_from_params(parameters) -> MalfunctionGenerator:
    """
    Utility to load malfunction from parameters

    Parameters
    ----------
    parameters containing
    malfunction_rate : float how many time steps it takes for a sinlge agent befor it breaks
    min_duration : int minimal duration of a failure
    max_number_of_steps_broken : int maximal duration of a failure

    Returns
    -------
    Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """

    def generator():
        mean_malfunction_rate = parameters['malfunction_rate']
        min_number_of_steps_broken = parameters['min_duration']
        max_number_of_steps_broken = parameters['max_duration']
        return mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken

    return generator


def no_malfunction_generator() -> MalfunctionGenerator:
    """
    Utility to load malfunction from parameters

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[float, int, int] with mean_malfunction_rate, min_number_of_steps_broken, max_number_of_steps_broken
    """

    def generator():
        return 0, 0, 0

    return generator
