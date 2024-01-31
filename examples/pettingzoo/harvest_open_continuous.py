# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_pettingzoo_ma_ataripy
import argparse
from distutils.util import strtobool
import importlib
import os
import pickle
import random
import time
import warnings
import shutil

import gymnasium as gym
from meltingpot import substrate
import numpy as np
import stable_baselines3
import supersuit as ss
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision

from examples.pettingzoo import hug_upload
from examples.pettingzoo import video_recording

from . import utils
from .vector_constructors import pettingzoo_env_to_vec_env_v1
from .vector_constructors import sb3_concat_vec_envs_v1


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
    parser.add_argument("--wandb-project-name", type=str, default="continuous-harvest-utilitarian",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances")
    parser.add_argument("--video-freq", type=int, default=100,
        help="capture video every how many episodes?")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model parameters")
    parser.add_argument("--save-model-freq", type=int, default=1,
        help="save model parameters every how many episodes?")

    # Algorithm specific arguments
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--adam-eps", type=float, default=1e-5,
        help="eps value for the optimizer")
    parser.add_argument("--num-parallel-games", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-episodes", type=int, default=100000,
        help="the number of steps in an episode")
    parser.add_argument("--episode-length", type=int, default=100,
        help="the number of steps in an episode")
    parser.add_argument("--sampling-horizon", type=int, default=100,
        help="the number of timesteps between policy update iterations")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--minibatch_size", type=int, default=128,
        help="size of minibatches when training policy network")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            self.layer_init(nn.Conv2d(envs.single_observation_space.shape[2], 32, 8, stride=4)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            self.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            self.layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = self.layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = self.layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        #x = x.clone()
        #num_rgb_channels = 12
        """
        we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
        """
        #x[:, :, :, :num_rgb_channels] /= 255.0
        return self.critic(self.network(x.permute((0, 3, 1, 2))))

    def get_action_and_value(self, x, action=None):
        """
        x is an observation - in our case with shape 88x88x19
        """
        #x = x.clone()
        #if x.shape[3] != 19:
        #    warnings.warn("hardcoded value of 12 RGB channels - check RBG/indicator channel division here")
        #num_rgb_channels = 12
        """
        we only divide the 4 stack frames x 3 RGB channels - NOT the agent indicators
        """
        #x[:, :, :, :num_rgb_channels] /= 255.0

        hidden = self.network(x.permute((0, 3, 1, 2)))
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer


if __name__ == "__main__":
    args = parse_args()
    print("ep length ", args.episode_length)
    print("sampling horizon ", args.sampling_horizon)
    #run_name = f"commons_harvest__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name = f"{args.episode_length}-{args.sampling_horizon}-utilitarian"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("device: ", device)
    env_name = "commons_harvest__open"
    env_config = substrate.get_config(env_name)
    num_cpus = 0  # number of cpus
    num_frames = 4
    model_path = None  # Replace this with a saved model

    env = utils.parallel_env(
        max_cycles=args.sampling_horizon,
        env_config=env_config,
    )
    num_agents = env.max_num_agents
    num_envs = args.num_parallel_games * num_agents
    env.render_mode = "rgb_array"

    env = ss.observation_lambda_v0(env, lambda x, _: x["RGB"], lambda s: s["RGB"])
    env = ss.frame_stack_v1(env, num_frames)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = pettingzoo_env_to_vec_env_v1(env)
    envs = sb3_concat_vec_envs_v1( # need our own as need reset to pass up world obs in info
        env,
        num_vec_envs=args.num_parallel_games)

    #test_env = new single env for recording
    #recorded_env = video_recording.RecordVideo(test_env, f"videos_temp/", episode_trigger=(lambda x: x%args.video_freq == 0))

    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.sampling_horizon, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.sampling_horizon, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.sampling_horizon, num_envs)).to(device)
    rewards = torch.zeros((args.sampling_horizon, num_envs)).to(device)
    dones = torch.zeros((args.sampling_horizon, num_envs)).to(device)
    values = torch.zeros((args.sampling_horizon, num_envs)).to(device)
    world_obs = torch.zeros((args.sampling_horizon, args.num_parallel_games) + (144,192,3)).to(device)


    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    next_world_obs = torch.stack([torch.Tensor(envs.reset_infos[i][1]) for i in range(0,num_envs,num_agents)]).to(device)

    num_policy_updates_per_ep = args.episode_length // args.sampling_horizon
    num_policy_updates_total = args.num_episodes * num_policy_updates_per_ep
    num_updates_for_this_ep = 0
    current_episode = 1
    episode_step = 0
    episode_rewards = torch.zeros(num_envs).to(device)
    start_time = time.time()
    # fill this with sampling horizon chunks for recording if needed
    episode_world_obs = [0] * (args.episode_length//args.sampling_horizon)

    """
    temporarily loading in a pretrained agent model to see what happens
    """
    #warnings.warn("loading pretrained agents")
    #agent.load_state_dict(torch.load("./model9399.pth"))

    for update in range(1, num_policy_updates_total + 1):

        # annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_policy_updates_total
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # collect data for policy update
        start_step = episode_step
        for step in range(0, args.sampling_horizon):
            if next_obs.shape[3] != 19:
              warnings.warn("hardcoded value of 12 RGB channels - check RBG/indicator channel division here")
            num_rgb_channels = 12
            next_obs[:, :, :, :num_rgb_channels] /= 255.0
            obs[step] = next_obs
            dones[step] = next_done
            world_obs[step] = next_world_obs

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            """
            NOTE: info has been changed to return a list of entries for each
                  environment (over num_agents and num_parallel_games), with
                  each entry being a tuple of the old info dict (asfaik always
                  empty until last step when it gets a 'terminal_observation')
                  and the world observation numpy array.
                  IMPORTANT:
                  The environments are listed out equally in a list, but will
                  be in chunks of length num_agents corresponding to which game
                  is being played. If you are playing 2 simultaneous games of
                  seven players, info will be a list of length 7. In this, the
                  first seven entries will have the same info[i][1] world
                  observation, and so will the next seven all share a world obs
                  - but the two will differ between each other.
            """
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            next_world_obs = torch.stack([torch.Tensor(info[i][1]) for i in range(0,num_envs,num_agents)]).to(device)


            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            episode_step += 1

        episode_rewards += torch.sum(rewards,0)
        end_step = episode_step - 1
        episode_world_obs[num_updates_for_this_ep-1] = world_obs[:,0,:,:,:].clone()

        if args.save_model and current_episode%args.save_model_freq == 0:
          try:
              os.mkdir(f"./saved_params_{run_name}")
          except FileExistsError:
              pass
          try:
              os.mkdir(f"./saved_params_{run_name}/ep{current_episode}")
          except FileExistsError:
              pass
          torch.save(obs,f"./saved_params_{run_name}/ep{current_episode}/obs_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt")
          torch.save(actions,f"./saved_params_{run_name}/ep{current_episode}/actions_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt")
          torch.save(logprobs,f"./saved_params_{run_name}/ep{current_episode}/logprobs_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt")
          torch.save(rewards,f"./saved_params_{run_name}/ep{current_episode}/rewards_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt")
          torch.save(dones,f"./saved_params_{run_name}/ep{current_episode}/dones_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt")
          torch.save(values,f"./saved_params_{run_name}/ep{current_episode}/values_samplerun{num_updates_for_this_ep}_ep{current_episode}.pt")


        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.sampling_horizon)):
                if t == args.sampling_horizon - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)


        # Optimizing the policy and value network
        b_inds = np.arange(len(b_obs))
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_obs), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

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

        # one more policy update done
        num_updates_for_this_ep += 1
        print(f"Completed policy update {num_updates_for_this_ep} for episode {current_episode} - used steps {start_step} through {end_step}")

        if num_updates_for_this_ep == num_policy_updates_per_ep:
            # episode finished
            if args.capture_video and current_episode%args.video_freq == 0:
              # currently only records first of any parallel games running but this is easily changed
              # at the point where we add to episode_world_obs
              video = torch.cat(episode_world_obs, dim=0).cpu()
              try:
                  os.mkdir(f"./videos_{run_name}")
              except FileExistsError:
                  pass
              torchvision.io.write_video(f"./videos_{run_name}/episode_{current_episode}.mp4", video, fps=20)
              hug_upload.upload(f"./videos_{run_name}", run_name,username="hegasz")
              if args.track:
                wandb.log({"video": wandb.Video(f"./videos_{run_name}/episode_{current_episode}.mp4")})
              os.remove(f"./videos_{run_name}/episode_{current_episode}.mp4")


            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], current_episode)
            writer.add_scalar("losses/value_loss", v_loss.item(), current_episode)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), current_episode)
            writer.add_scalar("losses/entropy", entropy_loss.item(), current_episode)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), current_episode)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), current_episode)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), current_episode)
            writer.add_scalar("losses/explained_variance", explained_var, current_episode)
            writer.add_scalar("charts/mean_episodic_return", torch.mean(episode_rewards), current_episode)
            writer.add_scalar("charts/episode", current_episode, current_episode)

            mean_rewards_across_envs = {player_idx:0 for player_idx in range(0, num_agents)}
            for idx in range(len(episode_rewards)):
                mean_rewards_across_envs[idx%num_agents] += episode_rewards[idx].item()
            mean_rewards_across_envs = list(map(lambda x: x/args.num_parallel_games, mean_rewards_across_envs.values()))

            for player_idx in range(num_agents):
                writer.add_scalar(f"charts/episodic_return-player{player_idx}", mean_rewards_across_envs[player_idx], current_episode)
            print(f"Finished episode {current_episode}, with {num_policy_updates_per_ep} policy updates")
            print(f"Mean episodic return: {torch.mean(episode_rewards)}")
            print(f"Episode returns: {mean_rewards_across_envs}")
            print("*******************************")
            if args.save_model and current_episode%args.save_model_freq == 0:
                try:
                    os.mkdir(f"./models_{run_name}")
                except FileExistsError:
                    pass
                torch.save(agent.state_dict(),f"./models_{run_name}/agent_{current_episode}.pth")
                hug_upload.upload(f"./models_{run_name}", run_name,username="hegasz")
                os.remove(f"./models_{run_name}/agent_{current_episode}.pth")

                hug_upload.upload(f"./saved_params_{run_name}", run_name,username="hegasz")
                shutil.rmtree(f"./saved_params_{run_name}/ep{current_episode}")
                print("model saved")
            # start a new episode:
            next_obs = torch.Tensor(envs.reset()).to(device)
            next_done = torch.zeros(num_envs).to(device)
            # don't need to reset obs,actions,logprobs etc as they have length
            # args.sampling_horizon so will just be overwritten
            current_episode += 1
            num_updates_for_this_ep = 0
            episode_step = 0
            episode_rewards = torch.zeros(num_envs).to(device)

    envs.close()
    writer.close()
