import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super(RNNAgent, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_actions = n_actions

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h


