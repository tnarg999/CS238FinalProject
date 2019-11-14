"""Schedule generators (railway undertaking, "EVU")."""
import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any

import msgpack
import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.schedule_utils import Schedule

AgentPosition = Tuple[int, int]
ScheduleGenerator = Callable[[GridTransitionMap, int, Optional[Any], Optional[int]], Schedule]


def speed_initialization_helper(nb_agents: int, speed_ratio_map: Mapping[float, float] = None,
                                seed: int = None) -> List[float]:
    """
    Parameters
    ----------
    nb_agents : int
        The number of agents to generate a speed for
    speed_ratio_map : Mapping[float,float]
        A map of speeds mappint to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
    List[float]
        A list of size nb_agents of speeds with the corresponding probabilistic ratios.
    """
    if seed:
        np.random.seed(seed)

    if speed_ratio_map is None:
        return [1.0] * nb_agents

    nb_classes = len(speed_ratio_map.keys())
    speed_ratio_map_as_list: List[Tuple[float, float]] = list(speed_ratio_map.items())
    speed_ratios = list(map(lambda t: t[1], speed_ratio_map_as_list))
    speeds = list(map(lambda t: t[0], speed_ratio_map_as_list))
    return list(map(lambda index: speeds[index], np.random.choice(nb_classes, nb_agents, p=speed_ratios)))


def complex_schedule_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:
    """

    Generator used to generate the levels of Round 1 in the Flatland Challenge. It can only be used together
    with complex_rail_generator. It places agents at end and start points provided by the rail generator.
    It assigns speeds to the different agents according to the speed_ratio_map
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    :return:
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0) -> Schedule:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the schedule
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """
        _runtime_seed = seed + num_resets
        np.random.seed(_runtime_seed)

        start_goal = hints['start_goal']
        start_dir = hints['start_dir']
        agents_position = [sg[0] for sg in start_goal[:num_agents]]
        agents_target = [sg[1] for sg in start_goal[:num_agents]]
        agents_direction = start_dir[:num_agents]

        if speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, speed_ratio_map, seed=_runtime_seed)
        else:
            speeds = [1.0] * len(agents_position)

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None)

    return generator


def sparse_schedule_generator(speed_ratio_map: Mapping[float, float] = None, seed: int = 1) -> ScheduleGenerator:
    """

    This is the schedule generator which is used for Round 2 of the Flatland challenge. It produces schedules
    to railway networks provided by sparse_rail_generator.
    :param speed_ratio_map: Speed ratios of all agents. They are probabilities of all different speeds and have to
            add up to 1.
    :param seed: Initiate random seed generator
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None, num_resets: int = 0) -> Schedule:
        """

        The generator that assigns tasks to all the agents
        :param rail: Rail infrastructure given by the rail_generator
        :param num_agents: Number of agents to include in the schedule
        :param hints: Hints provided by the rail_generator These include positions of start/target positions
        :param num_resets: How often the generator has been reset.
        :return: Returns the generator to the rail constructor
        """

        _runtime_seed = seed + num_resets
        np.random.seed(_runtime_seed)

        train_stations = hints['train_stations']
        city_positions = hints['city_positions']
        city_orientation = hints['city_orientations']
        max_num_agents = hints['num_agents']
        city_orientations = hints['city_orientations']
        if num_agents > max_num_agents:
            num_agents = max_num_agents
            warnings.warn("Too many agents! Changes number of agents.")
        # Place agents and targets within available train stations
        agents_position = []
        agents_target = []
        agents_direction = []

        for agent_idx in range(num_agents):
            infeasible_agent = True
            tries = 0
            while infeasible_agent:
                tries += 1
                infeasible_agent = False
                # Set target for agent
                city_idx = np.random.choice(len(city_positions), 2, replace=False)
                start_city = city_idx[0]
                target_city = city_idx[1]

                start_idx = np.random.choice(np.arange(len(train_stations[start_city])))
                target_idx = np.random.choice(np.arange(len(train_stations[target_city])))
                start = train_stations[start_city][start_idx]
                target = train_stations[target_city][target_idx]

                while start[1] % 2 != 0:
                    start_idx = np.random.choice(np.arange(len(train_stations[start_city])))
                    start = train_stations[start_city][start_idx]
                while target[1] % 2 != 1:
                    target_idx = np.random.choice(np.arange(len(train_stations[target_city])))
                    target = train_stations[target_city][target_idx]
                possible_orientations = [city_orientation[start_city],
                                         (city_orientation[start_city] + 2) % 4]
                agent_orientation = np.random.choice(possible_orientations)
                if not rail.check_path_exists(start[0], agent_orientation, target[0]):
                    agent_orientation = (agent_orientation + 2) % 4
                if not (rail.check_path_exists(start[0], agent_orientation, target[0])):
                    infeasible_agent = True
                if tries >= 100:
                    warnings.warn("Did not find any possible path, check your parameters!!!")
                    break
            agents_position.append((start[0][0], start[0][1]))
            agents_target.append((target[0][0], target[0][1]))

            agents_direction.append(agent_orientation)
            # Orient the agent correctly

        if speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, speed_ratio_map, seed=_runtime_seed)
        else:
            speeds = [1.0] * len(agents_position)

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=speeds, agent_malfunction_rates=None)

    return generator


def random_schedule_generator(speed_ratio_map: Optional[Mapping[float, float]] = None,
                              seed: int = 1) -> ScheduleGenerator:
    """
    Given a `rail` GridTransitionMap, return a random placement of agents (initial position, direction and target).

    Parameters
    ----------
        speed_ratio_map : Optional[Mapping[float, float]]
            A map of speeds mapping to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
        Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
            initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None,
                  num_resets: int = 0) -> Schedule:
        _runtime_seed = seed + num_resets

        np.random.seed(_runtime_seed)

        valid_positions = []
        for r in range(rail.height):
            for c in range(rail.width):
                if rail.get_full_transitions(r, c) > 0:
                    valid_positions.append((r, c))
        if len(valid_positions) == 0:
            return Schedule(agent_positions=[], agent_directions=[],
                            agent_targets=[], agent_speeds=[], agent_malfunction_rates=None)

        if len(valid_positions) < num_agents:
            warnings.warn("schedule_generators: len(valid_positions) < num_agents")
            return Schedule(agent_positions=[], agent_directions=[],
                            agent_targets=[], agent_speeds=[], agent_malfunction_rates=None)

        agents_position_idx = [i for i in np.random.choice(len(valid_positions), num_agents, replace=False)]
        agents_position = [valid_positions[agents_position_idx[i]] for i in range(num_agents)]
        agents_target_idx = [i for i in np.random.choice(len(valid_positions), num_agents, replace=False)]
        agents_target = [valid_positions[agents_target_idx[i]] for i in range(num_agents)]
        update_agents = np.zeros(num_agents)

        re_generate = True
        cnt = 0
        while re_generate:
            cnt += 1
            if cnt > 1:
                print("re_generate cnt={}".format(cnt))
            if cnt > 1000:
                raise Exception("After 1000 re_generates still not success, giving up.")
            # update position
            for i in range(num_agents):
                if update_agents[i] == 1:
                    x = np.setdiff1d(np.arange(len(valid_positions)), agents_position_idx)
                    agents_position_idx[i] = np.random.choice(x)
                    agents_position[i] = valid_positions[agents_position_idx[i]]
                    x = np.setdiff1d(np.arange(len(valid_positions)), agents_target_idx)
                    agents_target_idx[i] = np.random.choice(x)
                    agents_target[i] = valid_positions[agents_target_idx[i]]
            update_agents = np.zeros(num_agents)

            # agents_direction must be a direction for which a solution is
            # guaranteed.
            agents_direction = [0] * num_agents
            re_generate = False
            for i in range(num_agents):
                valid_movements = []
                for direction in range(4):
                    position = agents_position[i]
                    moves = rail.get_transitions(position[0], position[1], direction)
                    for move_index in range(4):
                        if moves[move_index]:
                            valid_movements.append((direction, move_index))

                valid_starting_directions = []
                for m in valid_movements:
                    new_position = get_new_position(agents_position[i], m[1])
                    if m[0] not in valid_starting_directions and rail.check_path_exists(new_position, m[1],
                                                                                        agents_target[i]):
                        valid_starting_directions.append(m[0])

                if len(valid_starting_directions) == 0:
                    update_agents[i] = 1
                    warnings.warn(
                        "reset position for agent[{}]: {} -> {}".format(i, agents_position[i], agents_target[i]))
                    re_generate = True
                    break
                else:
                    agents_direction[i] = valid_starting_directions[
                        np.random.choice(len(valid_starting_directions), 1)[0]]

        agents_speed = speed_initialization_helper(num_agents, speed_ratio_map, seed=_runtime_seed)
        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=agents_speed, agent_malfunction_rates=None)

    return generator


def schedule_from_file(filename, load_from_package=None) -> ScheduleGenerator:
    """
    Utility to load pickle file

    Parameters
    ----------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None,
                  num_resets: int = 0) -> Schedule:
        if load_from_package is not None:
            from importlib_resources import read_binary
            load_data = read_binary(load_from_package, filename)
        else:
            with open(filename, "rb") as file_in:
                load_data = file_in.read()
        data = msgpack.unpackb(load_data, use_list=False, encoding='utf-8')
        if "agents_static" in data:
            agents = EnvAgent.load_legacy_static_agent(data["agents_static"])
        else:
            agents = [EnvAgent(*d[0:12]) for d in data["agents"]]

        # setup with loaded data
        agents_position = [a.initial_position for a in agents]
        agents_direction = [a.direction for a in agents]
        agents_target = [a.target for a in agents]
        agents_speed = [a.speed_data['speed'] for a in agents]
        agents_malfunction = [a.malfunction_data['malfunction_rate'] for a in agents]

        return Schedule(agent_positions=agents_position, agent_directions=agents_direction,
                        agent_targets=agents_target, agent_speeds=agents_speed, agent_malfunction_rates=None)

    return generator
