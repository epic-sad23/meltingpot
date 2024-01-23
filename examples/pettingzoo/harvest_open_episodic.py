import argparse
from distutils.util import strtobool
import os
import random
import time

from meltingpot import substrate
import numpy as np
import supersuit as ss
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from examples.pettingzoo import video_recording

from . import new_utils


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture_video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--agent_indicators", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to add agent indicator channels")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="harvest_open",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000001,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--minibatch_size", type=int, default=32,
        help="size of minibatches when training policy network")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    # fmt: on
    return args



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, num_actions, num_channels):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(num_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        x = x.clone()
        x = x.float()
        num_rgb_channels = 12
        """
        we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
        """
        x[:, :, :, :num_rgb_channels] /= 255.0
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        """
        x is an observation - in our case with shape 7x88x88x19
        """
        x = x.clone()
        x = x.float()
        num_rgb_channels = 12
        """
        we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
        """
        x[:, :, :, :num_rgb_channels] /= 255.0
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.unwrapped.possible_agents)}

    return x


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    """ TRY NOT TO MODIFY: seeding """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    num_frames = 4
    total_episodes = 400

    """ ENV SETUP """
    env_name = "commons_harvest__open"
    env_config = substrate.get_config(env_name)
    env = new_utils.parallel_env(
        max_cycles=args.num_steps,
        env_config=env_config,
    )
    env.render_mode = "rgb_array"
    env = ss.frame_stack_v1(env, stack_size=num_frames)
    if args.agent_indicators:
        env = ss.agent_indicator_v0(env, type_only=False)
    num_agents = len(env.unwrapped.possible_agents)
    num_actions = env.action_space(env.unwrapped.possible_agents[0]).n
    observation_dims = env.observation_space(env.unwrapped.possible_agents[0]).shape[:2]
    num_channels = env.observation_space(env.unwrapped.possible_agents[0]).shape[2]
    env = video_recording.RecordVideo(env, f"videos/", episode_trigger=(lambda x: x%5==0))

    """ LEARNER SETUP """
    agent = Agent(num_actions, num_channels).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    ep_obs = torch.zeros((args.num_steps, num_agents, num_channels) + observation_dims).to(device)
    ep_actions = torch.zeros((args.num_steps, num_agents)).to(device)
    ep_logprobs = torch.zeros((args.num_steps, num_agents)).to(device)
    ep_rewards = torch.zeros((args.num_steps, num_agents)).to(device)
    ep_terms = torch.zeros((args.num_steps, num_agents)).to(device)
    ep_values = torch.zeros((args.num_steps, num_agents)).to(device)

    """ TRAINING LOGIC """
    global_step = 0
    for episode in range(total_episodes):

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (episode - 1.0) / total_episodes
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # collect observations and convert to batch of torch tensors
        next_obs, info = env.reset(seed=None)
        # reset the episodic return
        total_episodic_return = 0

        # each episode has num_steps
        for step in range(0, args.num_steps):
            global_step += 1
            obs = batchify_obs(next_obs, device)
            ep_obs[step] = obs
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(obs)
                ep_values[step] = values.flatten()
            ep_actions[step] = actions
            ep_logprobs[step] = logprobs

            next_obs, rewards, terms, truncs, infos = env.step(
                unbatchify(actions, env)
            )

            ep_rewards[step] = batchify(rewards, device)
            ep_terms[step] = batchify(terms, device)

            total_episodic_return += ep_rewards[step].cpu().numpy()
            # if we reach termination or truncation, end
            if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                end_step = step
                break


        # bootstrap value if not done
        with torch.no_grad():
            ep_advantages = torch.zeros_like(ep_rewards).to(device)
            for t in reversed(range(end_step)):
                nextvalues = ep_values[t + 1]
                delta = ep_rewards[t] + args.gamma * nextvalues * ep_terms[t + 1] - ep_values[t]

                ep_advantages[t] = delta + args.gamma * args.gamma * ep_advantages[t + 1]
            ep_returns = ep_advantages + ep_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(ep_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(ep_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(ep_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(ep_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(ep_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(ep_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), args.minibatch_size):
                # select the indices we want to train on
                end = start + args.minibatch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(b_obs[batch_index], b_actions.long()[batch_index])
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                if args.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -b_advantages[batch_index] * ratio
                pg_loss2 = -b_advantages[batch_index] * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value = value.flatten()
                if args.clip_vloss:
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((value - b_returns[batch_index]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Mean episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("*******************************")
        writer.add_scalar("charts/episode", episode, global_step)
        writer.add_scalar("charts/episode_length", end_step, global_step)
        writer.add_scalar("charts/mean_episodic_return", np.mean(total_episodic_return), global_step)
        for player_idx in range(num_agents):
            writer.add_scalar(f"charts/episodic_return-player{player_idx}", total_episodic_return[player_idx], global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

    env.close()
    writer.close()
