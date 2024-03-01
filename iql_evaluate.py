import gym, divgoals, time
import torch as th
import torch.nn.functional as F
from iql_model import RNNAgent

env = gym.make("DivGoals-10x10-5p-v1")
weights_path = '/home/ad181/epymarl_exp/epymarl/results/models/iql_5p_1/5000050/agent.th'

# create model and load weights
model = RNNAgent(env.observation_space.spaces[0].shape[0]+len(env.observation_space), 64, env.action_space.spaces[0].n)
model.load_state_dict(th.load(weights_path))

# define function to get actions and new hidden states given the obs array and hidden array
def get_actions_and_new_hidden(obs_array, hidden_array):
    # append one-hot encoding of agent id to observation
    obs_cats = []
    for i,ob in enumerate(obs_array):
        obs_cat = F.one_hot(th.tensor([i]), len(env.observation_space)).float()
        obs_cats.append(th.cat(( th.tensor(ob), obs_cat[0])))

    # pass through model
    new_hidden_array = []
    actions = []
    for obs, hidden in zip(obs_cats, hidden_array):
        q, hidden = model(obs, hidden)
        new_hidden_array.append(hidden)
        actions.append(th.argmax(q).item())
    return actions, new_hidden_array

#reset and render environment
obs_array = env.reset()
env.render()
# get initial array of hidden states for each agent
hidden_array = [model.init_hidden() for _ in range(len(env.observation_space))]

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