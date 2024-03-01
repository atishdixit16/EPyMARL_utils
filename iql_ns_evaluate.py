import gym, divgoals, time
import torch as th
import torch.nn.functional as F
from iql_ns_model import RNNNSAgent

env = gym.make("DivGoals-10x10-5p-v1")
weights_path = '/home/ad181/epymarl_exp/epymarl/results/models/iql_ns_5p_1/5000050/agent.th'

# create model and load weights
model = RNNNSAgent(env.observation_space.spaces[0].shape[0], 64, env.action_space.spaces[0].n, len(env.observation_space))
model.load_state_dict(th.load(weights_path))

# define function to get actions and new hidden states given the obs array and hidden array
def get_actions_and_new_hidden(obs_array, hidden_array):
    # pass through model
    qs, hidden_array = model(th.tensor(obs_array), hidden_array)
    actions = th.argmax(qs, dim=-1).tolist()
    return actions, hidden_array

#reset and render environment
obs_array = env.reset()
env.render()
# get initial array of hidden states for each agent
hidden_array = model.init_hidden()

done = False
r = [0.0]*len(env.observation_space)
while not done:
    # get actions and new hidden states
    actions, hidden_array = get_actions_and_new_hidden(obs_array, hidden_array)
    # take step in environment
    obs_array, reward, d, _ = env.step(actions)
    done = all(d)
    # agents rewards to r
    for i in range(len(env.observation_space)):
        r[i] = r[i] + reward[i]
    # render environment
    env.render()
    time.sleep(0.25)

# print agents rewards
for i in range(len(env.observation_space)):
    print('agent ', i, ' reward: ', r[i])
# print total reward
print('total reward: ' ,sum(r))

env.close()