import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training SAC discrete')
    # environment
    parser.add_argument('--env_name', default='LunarLander-v2')
    # replay buffer
    parser.add_argument('--buffer_size', default=1000000, type=int)
    # train
    parser.add_argument('--start_step', default=1000, type=int)
    parser.add_argument('--total_env_step', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--gradient_steps', default=1, type=int)
    parser.add_argument('--train_freq', default=1, type=int)
    # eval
    parser.add_argument('--eval_interval', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    # actor
    parser.add_argument('--actor_lr', default=3e-4, type=float)
    
    parser.add_argument('--num_layers', default=3, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=1, type=float)
    parser.add_argument('--alpha_lr', default=3e-4, type=float)
    # misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--model_dir', default='/model', type=str)
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    args, unknown = parser.parse_known_args()
    return args

def make_envs(envs_name):
    pass

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def pprint(dict_data):
    '''Pretty print Hyper-parameters'''
    hyper_param_space, value_space = 30, 40
    format_str = "| {:<"+ f"{hyper_param_space}" + "} | {:<"+f"{value_space}"+"}|"
    hbar = '-'*(hyper_param_space + value_space+6)
    
    print(hbar)
    print(format_str.format('Hyperparams', 'Values'))
    print(hbar)
    
    for k, v in dict_data.items():
        print(format_str.format(str(k), str(v)))
        
    print(hbar)