from gym_testing.ddpg_her.train import DDPG_HER_Trainer

trainer = DDPG_HER_Trainer(
    weight_path='HandManipulateBlock-v0.pth',
    env_name="HandManipulateBlock-v0"
)
trainer.train(
    max_epochs=5, max_cycles=50,
    max_episodes=2, num_updates=40,
    plot_save_path='plot.png',
    plot_include=['success', 'actor', 'critic']
)