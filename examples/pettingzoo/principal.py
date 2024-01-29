import collections
from typing import Callable

import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from torch.optim import Adam
import torchvision

from examples.pettingzoo.ConvLSTM.ConvLSTM import ConvLSTM
from examples.pettingzoo.ConvLSTM.Seq2Seq import Seq2Seq

"""
class PrincipalAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_model = nn.Sequential(
            self.conv_layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            self.conv_layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.conv_layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.conv_layer_init(nn.Linear(64 * 14 * 20, 512)),
            nn.ReLU(),
        )
        self.network = nn.Sequential(
            self.layer_init(nn.Linear(519, 300)),
            nn.ReLU(),
            self.layer_init(nn.Linear(300, 300)),
            nn.ReLU(),
            self.layer_init(nn.Linear(300, 300)),
            nn.ReLU(),
            nn.BatchNorm1d(300)
        )


        self.lstm = torch.nn.LSTM(300, 600)

        self.actor = self.layer_init(nn.Linear(600, 3))
        self.critic = self.layer_init(nn.Linear(600, 1))

    def get_action_and_value(self, world_obs, endowments, action=None):

        world_obs = torch.Tensor(world_obs).clone().unsqueeze(0)
        print(world_obs.shape)
        world_obs[:, :, :3] /= 255.0
        world_obs = world_obs.permute((0, 3, 1, 2))


        new_inputs = torch.cat((self.conv_model(world_obs), torch.Tensor(endowments).unsqueeze(0)),dim=1)
        # new_imputs is shape 1x(512+7)

        hidden = self.network(new_inputs)
        lstm_outputs, (h_n, c_n) = self.lstm(hidden)

        logits = self.actor(lstm_outputs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(lstm_outputs)

    def conv_layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def layer_init(self, layer, bias_const=0.0):
        torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
"""

class PrincipalAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_network = nn.Sequential(
            self.layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 14 * 20, 512)),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(519, 600)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = self.layer_init(nn.Linear(600, 3), std=0.01)
        self.critic = self.layer_init(nn.Linear(600, 1), std=1)

    def get_states(self, world_obs, endowment, lstm_state, done):

        world_obs = torch.Tensor(world_obs).clone().unsqueeze(0)
        world_obs /= 255.0
        print(world_obs.shape)
        conv_out = self.conv_network(world_obs.permute((0, 3, 1, 2)))

        hidden = torch.cat((conv_out, torch.Tensor(endowment).unsqueeze(0)),dim=1)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        # done flag resets states to zero
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),  # input
                (  # h0, c0
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, world_obs, endowment, lstm_state, done):
        hidden, _ = self.get_states(world_obs, endowment, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, world_obs, endowment, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(world_obs, endowment, lstm_state, done)
        logits = self.actor(hidden)
        print(logits)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            lstm_state,
        )

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer



class Principal:
    """Principal class for universal mechanism design
    In this setting, the principal computes a tax on the reward based on the number of apples collected by the agent.
    The tax should be between 0 and 1.5. 0 means no tax, 1.5 means 150% tax.
    """
    def __init__(self, objective: Callable, num_players, episode_length) -> None:
        self.set_objective(objective)
        self.episode_length = episode_length
        self.num_players = num_players
        self.apple_counts = [0] * num_players
        self.collected_tax = 0
        self.episode_frames = [0] * episode_length
        self.objective_vals = [0] * episode_length
        self.principal_rewards = [0] * episode_length
        self.episode_endowments = [0] * episode_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.agent = PrincipalAgent().to(self.device)


    """
    let's do this for one env first then think how to generalise

    CURRENTLY ONLY SET TO WORK FOR ONE PARALLEL GAME
    """

    # collect world obs frame and reward of a step
    def collect_step(self, step, frame, reward):
        self.episode_frames[step] = frame
        running_wealth = self.episode_endowments[step-1] if step>0 else 0
        self.episode_endowments[step] = running_wealth + reward
        self.objective_vals[step] = self.objective(reward)
        prev_objective_val = self.objective_vals[step-1] if step>0 else 0
        self.principal_rewards[step] = self.objective(reward) - prev_objective_val

    def end_of_episode(self):
        #video = torch.Tensor(np.stack(self.episode_frames))
        #torchvision.io.write_video("./evidtempt.mp4", video, fps=12) # wants time x H x W x channels

        #video = video.unsqueeze(0).permute(0,4,1,2,3) # convLSTM wants batch_size x channels x time x H x W

        #output = self.agent(video)

        self.train()


    def train(self):

        gamma = 0.99
        gae_lambda = 0.95
        update_epochs = 4
        minibatch_size = 1
        clip_coef = 0.2
        norm_adv =  True
        clip_vloss = True
        ent_coef = 0.01
        vf_coef = 0.5
        learning_rate = 2.5e-4
        adam_eps = 1e-5
        max_grad_norm = 0.5
        target_kl = None

        optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate, eps=adam_eps)

        obs = torch.Tensor(np.stack(self.episode_frames))
        endows = torch.Tensor(np.stack(self.episode_endowments))

        dones = torch.zeros(self.episode_length).to(self.device)
        dones[-1] = 1 # manually setting only last done to True

        actions = torch.zeros(self.episode_length).to(self.device)
        logprobs = torch.zeros(self.episode_length).to(self.device)

        rewards = torch.Tensor(np.stack(self.principal_rewards))
        values = torch.zeros(self.episode_length).to(self.device)

        next_lstm_state = (
            torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size).to(self.device),
            torch.zeros(self.agent.lstm.num_layers, 1, self.agent.lstm.hidden_size).to(self.device)
        )


        for step in range(0, self.episode_length):

            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = self.agent.get_action_and_value(
                    self.episode_frames[step], self.episode_endowments[step], next_lstm_state, dones[step])
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob



        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.agent.get_value(self.episode_frames[-1], self.episode_endowments[-1], next_lstm_state, dones[-1]).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.episode_length)):
                if t == self.episode_length - 1:
                    nextnonterminal = 1.0
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs # doesn't need flattening as normally we have rollout_length x num_envs x H x W x channels
                    # and we flattend to (rollout_length*num_envs) x H x W x channels but here num_envs is 1 implicit and ignored
        b_endows = endows
        b_logprobs = logprobs
        b_actions = actions

        b_dones = dones
        b_advantages = advantages
        b_returns = returns
        b_values = values

        # Optimizing the policy and value network
        b_inds = np.arange(len(b_obs))
        clipfracs = []


        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for mb_inds in range(0, self.episode_length):
                # removed minibatching for now
                print()
                print(mb_inds)
                print(b_dones[mb_inds], dones[mb_inds])
                print("here" , b_actions.long()[mb_inds])

                _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_endows[mb_inds], initial_lstm_state, b_dones[mb_inds], b_actions.long()[mb_inds])



                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y




    def set_objective(self, objective: Callable) -> None:
        print("********\nSetting objective to", objective.__name__, "\n********")
        self.objective = objective

    def calculate_tax(self, num_apples) -> float:
        """very simple baseline principal: no tax on utilitarian, 100% tax on egalitarian if num_apples > 10"""
        if self.objective.__name__ == "utilitarian":
            return 0
        if self.objective.__name__ == "egalitarian":
            if num_apples > 10:
                return 1.5  # punish the agent for being too greedy
            else:
                return 0

    def collect_tax(self, tax: float) -> None:
        """store collected tax"""
        self.collected_tax += tax
