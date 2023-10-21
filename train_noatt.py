import os
import tensorflow as tf
import keras
import numpy as np
from ER import ER
from DQN import DQN_DENSE
import AGENT_DENSE as AGENT, ENV
import VGG16
import ParticleFilter

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

EPISODES = 1000
MEMORY_SIZE = 25000
REM_STEP = 1
EPSILON = 1.0 # exploration probability at start
EPSILON_MIN = 0.02  # minimum exploration probability
EPSILON_DECAY = 0.00002  # exponential decay rate for exploration prob
BATCH_SIZE = 32
GAMMA = 0.99  # discount rate
NPARTICLES = 250

if __name__ == "__main__":
    save_path = "./Models"
    env_name = "PongNoFrameskip-v4"
    model_path = os.path.join(save_path, env_name + ".h5")
    # VGG16_INPUT_SHAPE = (210, 160, 3) # shape of pong
    VGG16_INPUT_SHAPE = (80, 80, 3) # shape of cropped pong
    INPUT_SHAPE = ( REM_STEP, 2048 ) # shape of dqn

    vgg_model = VGG16.create( input_shape=VGG16_INPUT_SHAPE )

    env = ENV.create( env_name, render=False )
    agent = AGENT.create( REM_STEP,
                          INPUT_SHAPE,
                          env.action_space.n,
                          EPSILON, EPSILON_MIN, EPSILON_DECAY )
    dqn = DQN_DENSE.create( env.action_space.n, INPUT_SHAPE )
    model = dqn["model"]
    target_dqn = DQN_DENSE.create( env.action_space.n, INPUT_SHAPE )
    target_model = target_dqn["model"]
    experiences = ER.create( MEMORY_SIZE )

    DONE = False
    game_steps = 0
    max_average = -21.0
    for e in range( EPISODES ):
        score = 0
        DONE = False
        SAVING = False
        win = 0
        lose = 0

        state = AGENT.reset(
            agent,
            VGG16.predict(
                vgg_model,
                DQN_DENSE.pongFrame2input( ENV.reset( env )[0] )
            ).flatten()
        )
        for _ in range(65):
            ENV.step(env, 0)

        while not DONE:
            game_steps += 1
            action, explore_prob = AGENT.act( agent, model, game_steps )
            frame_rgb, reward, DONE, _ = ENV.step( env, action )
            vgg_output = VGG16.predict(
                vgg_model, DQN_DENSE.pongFrame2input( frame_rgb )
            )
            
            next_state = AGENT.add_frame( agent, vgg_output.flatten() )
            ER.remember( experiences, state, action, reward, next_state, DONE )
            state = next_state
            score += reward
            if reward != 0.0:
                if reward == 1.0:
                    win += 1
                else:
                    lose += 1
                print( f"{lose}/{win}", end=" " )
            if DONE:
                average = AGENT.set_score_average( agent, score )
                if average >= max_average:
                    max_average = average
                    DQN_DENSE.save( model, model_path )
                    SAVING = True

                print(f"\nepisode: {e}/{EPISODES}, score: {score}, average: {average} e: {explore_prob}, {'SAVING' if SAVING else ''}")

            if game_steps % agent["update_model_steps"] == 0:
                DQN_DENSE.copy_weights( model, target_model )

            DQN_DENSE.train( model, target_model, experiences, BATCH_SIZE, GAMMA )
