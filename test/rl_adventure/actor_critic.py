from gym_testing.rl_adventure.worker.actor_critic import ActorCriticWorker

# for num_steps in [5, 20]:
for num_steps in [20]:
    # for use_gae in [True, False]:
    for use_gae in [True]:
        for env_name in [
            'Acrobot-v1',
            'CartPole-v1',
            'MountainCar-v0',
            # 'MountainCarContinuous-v0', # Would need to modify network
            # 'Pendulum-v0' # Would need to modify network
        ]:
            for tau in [0.95, 0.9, 0.8, 0.5]:
                run_id = f"{num_steps}steps"
                if use_gae:
                    run_id = f"{run_id}-gae-lambda{tau}"
                worker = ActorCriticWorker(env_name=env_name, run_id=run_id, output_dir='output')
                worker.train(num_steps=num_steps, use_gae=use_gae, tau=tau)
                # worker.infer(num_frames=500, video_save='infer.avi')