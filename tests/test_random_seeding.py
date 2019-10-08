import numpy as np

from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail2


def test_random_seeding():
    # Set fixed malfunction duration for this test
    rail, rail_map = make_simple_rail2()

    # Move target to unreachable position in order to not interfere with test
    for idx in range(100):
        env = RailEnv(width=25,
                      height=30,
                      rail_generator=rail_from_grid_transition_map(rail),
                      schedule_generator=random_schedule_generator(seed=12),
                      number_of_agents=10
                      )
        env.reset(True, True, False, random_seed=1)
        # Test generation print

        env.agents[0].target = (0, 0)
        for step in range(10):
            actions = {}
            actions[0] = 2
            env.step(actions)
        agent_positions = []
        for a in range(env.get_num_agents()):
            agent_positions += env.agents[a].initial_position
        # print(agent_positions)
        assert agent_positions == [3, 2, 3, 5, 3, 6, 5, 6, 3, 4, 3, 1, 3, 9, 4, 6, 0, 3, 3, 7]
        # Test generation print
        assert env.agents[0].position == (3, 6)
        # print("env.agents[0].initial_position == {}".format(env.agents[0].initial_position))
        #print("assert env.agents[0].position ==  {}".format(env.agents[0].position))


def test_seeding_and_observations():
    # Test if two different instances diverge with different observations
    rail, rail_map = make_simple_rail2()

    # Make two seperate envs with different observation builders
    # Global Observation
    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=12),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv()
                  )
    # Tree Observation
    env2 = RailEnv(width=25,
                   height=30,
                   rail_generator=rail_from_grid_transition_map(rail),
                   schedule_generator=random_schedule_generator(seed=12),
                   number_of_agents=10,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())
                   )

    env.reset(False, False, False, random_seed=12)
    env2.reset(False, False, False, random_seed=12)

    # Check that both environments produce the same initial start positions
    assert env.agents[0].initial_position == env2.agents[0].initial_position
    assert env.agents[1].initial_position == env2.agents[1].initial_position
    assert env.agents[2].initial_position == env2.agents[2].initial_position
    assert env.agents[3].initial_position == env2.agents[3].initial_position
    assert env.agents[4].initial_position == env2.agents[4].initial_position
    assert env.agents[5].initial_position == env2.agents[5].initial_position
    assert env.agents[6].initial_position == env2.agents[6].initial_position
    assert env.agents[7].initial_position == env2.agents[7].initial_position
    assert env.agents[8].initial_position == env2.agents[8].initial_position
    assert env.agents[9].initial_position == env2.agents[9].initial_position

    action_dict = {}
    for step in range(10):
        for a in range(env.get_num_agents()):
            action = np.random.randint(4)
            action_dict[a] = action
        env.step(action_dict)
        env2.step(action_dict)

    # Check that both environments end up in the same position

    assert env.agents[0].position == env2.agents[0].position
    assert env.agents[1].position == env2.agents[1].position
    assert env.agents[2].position == env2.agents[2].position
    assert env.agents[3].position == env2.agents[3].position
    assert env.agents[4].position == env2.agents[4].position
    assert env.agents[5].position == env2.agents[5].position
    assert env.agents[6].position == env2.agents[6].position
    assert env.agents[7].position == env2.agents[7].position
    assert env.agents[8].position == env2.agents[8].position
    assert env.agents[9].position == env2.agents[9].position
    for a in range(env.get_num_agents()):
        print("assert env.agents[{}].position == env2.agents[{}].position".format(a, a))


def test_seeding_and_malfunction():
    # Test if two different instances diverge with different observations
    rail, rail_map = make_simple_rail2()

    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 2,
                       'min_duration': 10,
                       'max_duration': 10}
    # Make two seperate envs with different and see if the exhibit the same malfunctions
    # Global Observation
    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(seed=12),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv(),
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )

    # Tree Observation
    env2 = RailEnv(width=25,
                   height=30,
                   rail_generator=rail_from_grid_transition_map(rail),
                   schedule_generator=random_schedule_generator(seed=12),
                   number_of_agents=10,
                   obs_builder_object=GlobalObsForRailEnv(),
                   stochastic_data=stochastic_data,  # Malfunction data generator
                   )

    env.reset(False, False, False, random_seed=12)
    env2.reset(False, False, False, random_seed=12)

    # Check that both environments produce the same initial start positions
    assert env.agents[0].initial_position == env2.agents[0].initial_position
    assert env.agents[1].initial_position == env2.agents[1].initial_position
    assert env.agents[2].initial_position == env2.agents[2].initial_position
    assert env.agents[3].initial_position == env2.agents[3].initial_position
    assert env.agents[4].initial_position == env2.agents[4].initial_position
    assert env.agents[5].initial_position == env2.agents[5].initial_position
    assert env.agents[6].initial_position == env2.agents[6].initial_position
    assert env.agents[7].initial_position == env2.agents[7].initial_position
    assert env.agents[8].initial_position == env2.agents[8].initial_position
    assert env.agents[9].initial_position == env2.agents[9].initial_position

    action_dict = {}
    for step in range(10):
        for a in range(env.get_num_agents()):
            action = np.random.randint(4)
            action_dict[a] = action
            print(env.agents[a].malfunction_data['malfunction'], env2.agents[a].malfunction_data['malfunction'])
        env.step(action_dict)
        env2.step(action_dict)


    # Check that both environments end up in the same position

    assert env.agents[0].position == env2.agents[0].position
    assert env.agents[1].position == env2.agents[1].position
    assert env.agents[2].position == env2.agents[2].position
    assert env.agents[3].position == env2.agents[3].position
    assert env.agents[4].position == env2.agents[4].position
    assert env.agents[5].position == env2.agents[5].position
    assert env.agents[6].position == env2.agents[6].position
    assert env.agents[7].position == env2.agents[7].position
    assert env.agents[8].position == env2.agents[8].position
    assert env.agents[9].position == env2.agents[9].position
    for a in range(env.get_num_agents()):
        print("assert env.agents[{}].position == env2.agents[{}].position".format(a, a))
