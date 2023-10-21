import gym
import collections
import numpy as np
import cv2

def create( env_name, difficulty=None, render=False ):
    env = gym.make( env_name, render_mode="human" ) if render else gym.make( env_name )

    if type( difficulty ) == int:
        setDifficulty( env, difficulty )

    return env

def reset( env ):
    return env.reset()

def render( env ):
    env.render()

def skip_step( env, action, skip=4 ):
    total_reward = 0.0
    done = None
    obs_buffer = collections.deque( maxlen=skip )
    for _ in range( skip ):
        obs, reward, done, info = step( env, action )
        obs_buffer.append( obs )
        total_reward += reward
        if done:
            break
    max_frame = np.max( np.stack( obs_buffer ), axis=0 )
    return max_frame, total_reward, done, info

def step( env, action ):
    # return env.step( action )
    next_frame, reward, done, _, info = env.step( action )

    # calc moment and update reward according to distance btw
    # moment and position of agent
    # frame_cropped = next_frame[35:195:2, 20:-20:2,:]
    # gray_image = cv2.cvtColor( frame_cropped, cv2.COLOR_BGR2GRAY )
    # ret,thresh = cv2.threshold( gray_image, 127, 255, 0 )
    # M = cv2.moments(thresh)
    # imshow( gray_image )

    return next_frame, reward, done, info

def freezeStep(env,a):
    old_state = env.unwrapped.clone_full_state()
    state = env.step(a)
    env.unwrapped.restore_full_state(old_state)
#     old_state=env.unwrapped.clone_state()
#     state=env.step(a)
#     env.unwrapped.restore_state(old_state)
    return state

def set_difficulty( env, difficulty ):
    # diff levels: 0 1 2 3
    env.ale.setDifficulty( difficulty )

def crop_frame2ball( frame ):
    return frame[35:195:2, 20:-20:2,:]

def crop_frame2player( frame ):
    return frame[35:195:2, 140:-16:2,:]

def crop_frame2opponent( frame ):
    return frame[35:195:2, 16:20:2,:]

def close( env ):
    env.close()

def print_info( env ):
    print(env.action_space.n)
    print(env.unwrapped.get_action_meanings())
    print(env.observation_space.shape)
