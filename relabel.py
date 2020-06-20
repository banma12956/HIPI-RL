def inverseRL(steps, memory, env, agent):
    rpm_start_pos = memory.position - steps
    episode_exp = memory[rpm_start_pos:memory.position]
    for task_num in range(env.num_tasks):     # 10 tasks for MT10
        current_task = env._task_envs[task_num]
        reward_function = current_task.compute_reward
        one_hot = env.active_task_one_hot
        for step in steps:
            pass