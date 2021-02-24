from gym_testing.tianshou_util.train.robotics import FetchPickAndPlaceTrainer

trainer = FetchPickAndPlaceTrainer(seed=1, gamma=0.95)
trainer.train(prioritized_replay=True)