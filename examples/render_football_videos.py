# render_football_videos.py - Complete script to generate videos from pretrained HAPPO models

import os
import torch
import numpy as np
import argparse
from harl.algorithms.happo.algorithm.happo_trainer import HAPPO as TrainAlgo
from harl.envs.football.football_env import FootballEnv
from harl.utils.configs import get_config

def parse_args():
    parser = argparse.ArgumentParser(description='Render videos from pretrained HAPPO football models')
    parser.add_argument('--model_dir', type=str, 
                      default="./results/football/academy_counterattack_hard/happo/gfootball_test_preloaded/seed-00001-2025-04-05-01-27-56/models/", 
                      help='directory containing the models')
    parser.add_argument('--config_path', type=str, 
                      default="./results/football/academy_counterattack_hard/happo/gfootball_test_preloaded/seed-00001-2025-04-05-01-27-56/config.json", 
                      help='path to original config file')
    parser.add_argument('--num_episodes', type=int, default=10, help='number of episodes to render')
    parser.add_argument('--scenario', type=str, default="academy_counterattack_hard", help='football scenario')
    parser.add_argument('--resolution', type=str, default="96,72", help='video resolution (width,height)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directory for videos if it doesn't exist
    os.makedirs("./rendered_videos", exist_ok=True)
    
    # Load original config if it exists, otherwise use default values
    if os.path.exists(args.config_path):
        config = get_config(args.config_path)
        algo_args = config["algo_args"]
        env_args = config["env_args"]
    else:
        print(f"Config file {args.config_path} not found. Using default values.")
        algo_args = {
            "algo": {
                "action_aggregation": "prod",
                "use_policy_active_masks": True,
                "gae_lambda": 0.95,
                "gamma": 0.99,
                "use_gae": True
            },
            "model": {
                "hidden_sizes": [64, 64],
                "activation_func": "relu"
            }
        }
        
    # Override environment arguments to enable rendering
    width, height = map(int, args.resolution.split(','))
    env_args = {
        "env_name": args.scenario,
        "representation": "simple115v2",
        "number_of_left_players_agent_controls": 4,
        "number_of_right_players_agent_controls": 0,
        "render": True,
        "write_full_episode_dumps": True,
        "write_video": True,
        "channel_dimensions": [width, height],
        "write_goal_dumps": True,
        "dump_frequency": 1,
        "logdir": "./logs"
    }
    
    print(f"Creating environment: {args.scenario}")
    env = FootballEnv(env_args)
    
    # Initialize algorithm
    print("Initializing HAPPO trainer...")
    happo_trainer = TrainAlgo(algo_args)
    
    # Load pretrained models
    print(f"Loading models from {args.model_dir}")
    for i in range(4):  # 4 agents
        model_path = os.path.join(args.model_dir, f"actor_agent{i}.pt")
        if os.path.exists(model_path):
            happo_trainer.actors[i].load_state_dict(
                torch.load(model_path, map_location=torch.device('cpu'))
            )
            print(f"Loaded actor_agent{i}.pt")
        else:
            print(f"Warning: {model_path} not found!")
    
    critic_path = os.path.join(args.model_dir, "critic_agent.pt")
    if os.path.exists(critic_path):
        happo_trainer.critic.load_state_dict(
            torch.load(critic_path, map_location=torch.device('cpu'))
        )
        print("Loaded critic_agent.pt")
    else:
        print(f"Warning: {critic_path} not found!")
    
    # Optional: load value normalizer if using
    normalizer_path = os.path.join(args.model_dir, "value_normalizer.pt")
    if hasattr(happo_trainer, 'value_normalizer') and os.path.exists(normalizer_path):
        happo_trainer.value_normalizer.load_state_dict(
            torch.load(normalizer_path, map_location=torch.device('cpu'))
        )
        print("Loaded value_normalizer.pt")
    
    # Run evaluation with rendering
    print(f"Starting to render {args.num_episodes} episodes...")
    for episode in range(args.num_episodes):
        print(f"Episode {episode+1}/{args.num_episodes}")
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done:
            with torch.no_grad():
                actions, _, _ = happo_trainer.compute_actions(obs)
            
            obs, rewards, done, info = env.step(actions)
            episode_reward += sum(rewards)
            step += 1
            
            # Print periodic updates
            if step % 100 == 0:
                print(f"  Step {step}, Episode reward so far: {episode_reward}")
        
        print(f"Episode {episode+1} finished. Total reward: {episode_reward}, Steps: {step}")
    
    print("Rendering complete. Videos saved to logs directory.")
    env.close()

if __name__ == "__main__":
    main()