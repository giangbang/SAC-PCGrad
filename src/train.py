'''This file is not well-maintained and contains ugly code
since it should not be copied to other projects
'''

from sac_discrete import ReplayBuffer, SACDiscrete as SAC
import numpy as np
import gym
from utils import parse_args, pprint, seed_everything


if __name__ == '__main__':
    args = parse_args()
    
    if args.seed > 0: seed_everything(args.seed)
    
    env = gym.make(args.env_name)
    
    action_shape      = env.action_space.n
    observation_shape = env.observation_space.shape
    
    sac_agent = SAC(observation_shape[0], action_shape, **vars(args))
    buffer    = ReplayBuffer(observation_shape, [action_shape], 
                args.buffer_size, args.batch_size)
    
    pprint(vars(args))
    print('Action dim: {} | Observation dim: {}'.format(action_shape, observation_shape))
    
    his = []
    loss = []
    
    state = env.reset()
    for env_step in  range(int(args.total_env_step)):
        if env_step < args.start_step: 
            action = env.action_space.sample()
        else :
            action = sac_agent.select_action(state, False)
        
        next_state, reward, done, info = env.step(action)
        buffer.add(state, action, reward, next_state, done, info)
        
        if (env_step + 1) % args.train_freq == 0:
            loss.append(sac_agent.update(buffer))
        
        state = next_state
        if done: 
            state = env.reset()
        if (env_step + 1) % args.eval_interval == 0:
            eval_return = evaluate(gym.make(args.env_name), sac_agent, args.num_eval_episodes)
            his.append(eval_return)
            print('mean reward after {} env step: {:.2f}'.format(env_step+1, eval_return))
            print('critic loss: {:.2f} | actor loss: {:.2f} | alpha loss: {:.2f}'.format(
                    *np.mean(list(zip(*loss[-10:])), axis=-1)
                    ))
            print('alpha: {:.2f}'.format(sac_agent.log_ent_coef.exp().item()))
            
    import matplotlib.pyplot as plt
    x, y = np.linspace(0, args.total_env_step, len(his)), his
    plt.plot(x, y)
    plt.title(args.env_name)
    plt.savefig('res.png')

    import pandas as pd 
    data_dict = {'rollout/ep_rew_mean': y, 'time/total_timesteps': x} # formated as stable baselines
    df = pd.DataFrame(data_dict)

    df.to_csv('sac_progress.csv', index=False)
    sac_agent.save('model', args.total_env_step)
    