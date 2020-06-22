import numpy as np
import torch
import torch.nn.functional as F

def task_one_hot(task_num, env_tasks_num):
    task_onehot = np.zeros(shape=(env_tasks_num,), dtype=np.float32)
    task_onehot[task_num] = 1.
    return task_onehot

def inverseRL(steps, memory, env, agent, cuda):
    device = torch.device("cuda" if cuda else "cpu")

    rpm_start_pos = memory.position - steps
    
    if rpm_start_pos < 0:
        rpm_start_pos = memory.capacity + rpm_start_pos
        episode_exp = np.concatenate((memory.buffer[rpm_start_pos:], memory.buffer[:memory.position]))
    else:
        episode_exp = memory.buffer[rpm_start_pos:memory.position] 
    state, action, reward, next_state, done, rightFinger, leftFinger = map(np.stack, zip(*episode_exp))  

    obs_dim = env.observation_space.shape[0]

    # R_1 = []
    # Z = []
    Q_all = torch.empty([env.num_tasks, steps], dtype=torch.float32)

    # get R and Z for each phi
    for task_num in range(env.num_tasks):     # 10 tasks for MT10
        current_task = env._task_envs[task_num] 
        one_hot = task_one_hot(task_num, env.num_tasks)
 
        state[:, obs_dim-env.num_tasks:] = one_hot

        state_batch = torch.FloatTensor(state).to(device)
        action_batch = torch.FloatTensor(action).to(device) 

        with torch.no_grad():
            q1, q2 = agent.critic(state_batch, action_batch)
            Q = torch.min(q1, q2)	# TODO:
        Q_all[task_num] = torch.squeeze(Q)
    
    Q_all = (Q_all - Q_all.mean()) / Q_all.std()  # TODO: 
  
    Z = torch.mean(torch.exp(Q_all), dim=-1)
    R_1 = Q_all[:, 0]    
    # Z = torch.tensor(Z)
    # R_1 = torch.tensor(R_1)
    # print(Q_all[-1])
    # print(R_1)
    # print(Z)
    
    logits = torch.pow(R_1-torch.log(Z),10)
    prob = F.softmax(logits, dim=-1)
    # print(prob)
    m = torch.distributions.Categorical(prob)
    reward_index = m.sample()
 
    # relabel experience
    current_task = env._task_envs[reward_index]
    reward_function = current_task.compute_reward
    one_hot = task_one_hot(task_num, env.num_tasks)
    state[:, obs_dim-env.num_tasks:] = one_hot
  
    for step in range(steps):
        obs = {'state_observation': state[step][:6]} 
        relabel_reward, *rest = reward_function(action[step], obs, rightFinger[step], leftFinger[step])
        # print(relabel_reward)
        memory.push(state[step], action[step], relabel_reward, next_state[step], done[step], rightFinger[step], leftFinger[step])
    
