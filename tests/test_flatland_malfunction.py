import random
from typing import Dict, List

import numpy as np
from test_utils import Replay, ReplayConfig, run_replay_config, set_penalties_for_replay

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail2


class SingleAgentNavigationObs(ObservationBuilder):
    """
    We build a representation vector with 3 binary components, indicating which of the 3 available directions
    for each agent (Left, Forward, Right) lead to the shortest path to its target.
    E.g., if taking the Left branch (if available) is the shortest route to the agent's target, the observation vector
    will be [1, 0, 0].
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def get(self, handle: int = 0) -> List[int]:
        agent = self.env.agents[handle]

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent_virtual_position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1

        return observation


def test_malfunction_process():
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 1000,
                       'min_duration': 3,
                       'max_duration': 3}

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    # reset to initialize agents_static
    obs, info = env.reset(False, False, True, random_seed=0)
    print(env.agents[0].malfunction_data)
    # Check that a initial duration for malfunction was assigned
    assert env.agents[0].malfunction_data['next_malfunction'] > 0
    for agent in env.agents:
        agent.status = RailAgentStatus.ACTIVE

    agent_halts = 0
    total_down_time = 0
    agent_old_position = env.agents[0].position

    # Move target to unreachable position in order to not interfere with test
    env.agents[0].target = (0, 0)
    for step in range(100):
        actions = {}

        for i in range(len(obs)):
            actions[i] = np.argmax(obs[i]) + 1

        if step % 5 == 0:
            # Stop the agent and set it to be malfunctioning
            env.agents[0].malfunction_data['malfunction'] = -1
            env.agents[0].malfunction_data['next_malfunction'] = 0
            agent_halts += 1

        obs, all_rewards, done, _ = env.step(actions)

        if env.agents[0].malfunction_data['malfunction'] > 0:
            agent_malfunctioning = True
        else:
            agent_malfunctioning = False

        if agent_malfunctioning:
            # Check that agent is not moving while malfunctioning
            assert agent_old_position == env.agents[0].position

        agent_old_position = env.agents[0].position
        total_down_time += env.agents[0].malfunction_data['malfunction']

    # Check that the appropriate number of malfunctions is achieved
    assert env.agents[0].malfunction_data['nr_malfunctions'] == 21, "Actual {}".format(
        env.agents[0].malfunction_data['nr_malfunctions'])

    # Check that 20 stops where performed
    assert agent_halts == 20

    # Check that malfunctioning data was standing around
    assert total_down_time > 0


def test_malfunction_process_statistically():
    """Tests hat malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 2,
                       'min_duration': 3,
                       'max_duration': 3}

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    # reset to initialize agents_static
    env.reset(False, False, False, random_seed=0)

    env.agents[0].target = (0, 0)
    nb_malfunction = 0
    for step in range(20):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            # We randomly select an action
            action_dict[agent.handle] = RailEnvActions(np.random.randint(4))

        env.step(action_dict)
    # check that generation of malfunctions works as expected
    assert env.agents[0].malfunction_data["nr_malfunctions"] == 4


def test_malfunction_before_entry():
    """Tests hat malfunctions are produced by stochastic_data!"""
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 2,
                       'min_duration': 10,
                       'max_duration': 10}

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    # reset to initialize agents_static
    env.reset(False, False, False, random_seed=0)
    env.agents[0].target = (0, 0)
    nb_malfunction = 0
    for step in range(20):
        action_dict: Dict[int, RailEnvActions] = {}
        for agent in env.agents:
            # We randomly select an action
            if step < 10:
                action_dict[agent.handle] = RailEnvActions(0)
                assert env.agents[0].malfunction_data['malfunction'] == 0
            else:
                action_dict[agent.handle] = RailEnvActions(2)

        print(env.agents[0].malfunction_data)
        env.step(action_dict)
    assert env.agents[0].malfunction_data['malfunction'] > 0


def test_initial_malfunction():

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 70,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )

    # reset to initialize agents_static
    env.reset(False, False, True, random_seed=0)

    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=2,
                reward=env.step_penalty  # full step penalty when malfunctioning
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action MOVE_FORWARD, agent should restart and move to the next cell
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=1,
                reward=env.start_penalty + env.step_penalty * 1.0
                # malfunctioning ends: starting and running at speed 1.0
            ),
            Replay(
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0  # running at speed 1.0
            ),
            Replay(
                position=(3, 4),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0  # running at speed 1.0
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target,
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [replay_config])


def test_initial_malfunction_stop_moving():
    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 70,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    # reset to initialize agents_static

    print(env.agents[0].initial_position, env.agents[0].direction, env.agents[0].position, env.agents[0].status)

    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=None,
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty,  # full step penalty when stopped
                status=RailAgentStatus.READY_TO_DEPART
            ),
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=2,
                reward=env.step_penalty,  # full step penalty when stopped
                status=RailAgentStatus.ACTIVE
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action STOP_MOVING, agent should restart without moving
            #
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.STOP_MOVING,
                malfunction=1,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            # we have stopped and do nothing --> should stand still
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=0,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            # we start to move forward --> should go to next cell now
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            Replay(
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target,
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )

    run_replay_config(env, [replay_config], activate_agents=False)


def test_initial_malfunction_do_nothing():
    random.seed(0)
    np.random.seed(0)

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 70,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    rail, rail_map = make_simple_rail2()


    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )
    # reset to initialize agents_static
    env.reset()
    set_penalties_for_replay(env)
    replay_config = ReplayConfig(
        replay=[
            Replay(
                position=None,
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                set_malfunction=3,
                malfunction=3,
                reward=env.step_penalty,  # full step penalty while malfunctioning
                status=RailAgentStatus.READY_TO_DEPART
            ),
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=2,
                reward=env.step_penalty,  # full step penalty while malfunctioning
                status=RailAgentStatus.ACTIVE
            ),
            # malfunction stops in the next step and we're still at the beginning of the cell
            # --> if we take action DO_NOTHING, agent should restart without moving
            #
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=1,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),
            # we haven't started moving yet --> stay here
            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.DO_NOTHING,
                malfunction=0,
                reward=env.step_penalty,  # full step penalty while stopped
                status=RailAgentStatus.ACTIVE
            ),

            Replay(
                position=(3, 2),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.start_penalty + env.step_penalty * 1.0,  # start penalty + step penalty for speed 1.0
                status=RailAgentStatus.ACTIVE
            ),  # we start to move forward --> should go to next cell now
            Replay(
                position=(3, 3),
                direction=Grid4TransitionsEnum.EAST,
                action=RailEnvActions.MOVE_FORWARD,
                malfunction=0,
                reward=env.step_penalty * 1.0,  # step penalty for speed 1.0
                status=RailAgentStatus.ACTIVE
            )
        ],
        speed=env.agents[0].speed_data['speed'],
        target=env.agents[0].target,
        initial_position=(3, 2),
        initial_direction=Grid4TransitionsEnum.EAST,
    )
    run_replay_config(env, [replay_config], activate_agents=False)


def test_initial_nextmalfunction_not_below_zero():
    random.seed(0)
    np.random.seed(0)

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 70,  # Rate of malfunction occurence
                       'min_duration': 2,  # Minimal duration of malfunction
                       'max_duration': 5  # Max duration of malfunction
                       }

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=SingleAgentNavigationObs()
                  )
    # reset to initialize agents_static
    env.reset()
    agent = env.agents[0]
    env.step({})
    # was next_malfunction was -1 befor the bugfix https://gitlab.aicrowd.com/flatland/flatland/issues/186
    assert agent.malfunction_data['next_malfunction'] >= 0, \
        "next_malfunction should be >=0, found {}".format(agent.malfunction_data['next_malfunction'])
