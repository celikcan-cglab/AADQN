import collections
import numpy as np

def create( capacity ):
    return {
        "buffer": collections.deque( maxlen=capacity )
    }

def append( er, experience ):
    er["buffer"].append( experience )

def remember( er, state, action, reward, next_state, done ):
    experience = state, action, reward, next_state, done
    er["buffer"].append( ( experience ) )

def sample( er, batch_size ):
    if len( er["buffer"] ) < batch_size:
        return [], [], [], [], []
    indices = np.random.choice(
        len( er["buffer"] ),
        batch_size,
        replace=False
    )
    states, actions, rewards, next_states, dones = zip(
        *[er["buffer"][idx] for idx in indices]
    )
    return np.array(states, dtype=np.float32), np.array(actions), np.array(rewards, dtype=np.float32), np.array(next_states, dtype=np.float32), np.array(dones, dtype=np.uint8)
