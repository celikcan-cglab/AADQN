import random
import numpy as np

def create( rem_step, shape,
            n_actions,
            epsilon=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.00002,
            epsilon_greedy_improved=True ):
    return {
        "pos": 0,
        "rem_step": rem_step,
        "n_actions": n_actions,
        "state": np.zeros( shape ),
        "scores": [],
        "total_reward": 0,
        "update_model_steps": 1000,
        "epsilon": epsilon,            # exploration probability
        "epsilon_min": epsilon_min,    # minimum exploration probability
        "epsilon_decay": epsilon_decay,# exponential decay rate for exploration prob
        "epsilon_greedy_improved": epsilon_greedy_improved,
        "ball_distance": float( "inf" )
    }

def reset( agent, frame ):
    for _ in range( agent["rem_step"] ):
        add_frame( agent, frame )
    return get_state( agent )

def act_test( agent, model ):
    return np.argmax(
        model.predict(
            np.expand_dims( agent["state"], axis=0 ),
            verbose=0
        )
    )
# a = 0
# b = False
def act( agent, model, decay_step ):
    # global a,b
    # if b:
    #     if a%3 == 0:
    #         acti = 2
    #     else:
    #         acti = 3
    #     a += 1
    #     if agent["pos"] == -17:
    #         b = False
    # else:
    #     if a%3 == 0:
    #         acti = 3
    #     else:
    #         acti = 2
    #     a -= 1
    #     if agent["pos"] == 14:
    #         b = True
    # update_position( agent, acti )
    # return acti, 0.9
    if agent["epsilon_greedy_improved"]:
        explore_prob = agent["epsilon_min"] + (agent["epsilon"] - agent["epsilon_min"]) * np.exp(-agent["epsilon_decay"] * decay_step)
    else: # OLD EPSILON STRATEGY
        if agent["epsilon"] > agent["epsilon_min"]:
            agent["epsilon"] *= ( 1 - agent["epsilon_decay"] )
        explore_prob = agent["epsilon"]

    if explore_prob > np.random.rand():
        # Make a random action (exploration)
        action = random.randrange( agent["n_actions"] )
        # update_position( agent, action )
        return action, explore_prob

    # Get action from Q-network (exploitation)
    # Estimate the Qs values state
    # Take the biggest Q value (= the best action)
    action = np.argmax(
        model.predict( np.expand_dims( agent["state"], axis=0 ), verbose=0 )
    )
    # update_position( agent, action )
    return action, explore_prob

def add_frame( agent, state ):
    # push our data by 1 frame, similar as deq() function work
    # agent["state"] = np.roll( agent["state"], 1, axis = 0 )
    agent["state"] = np.roll( agent["state"], -1, axis = 0 )

    # inserting new frame to free space
    # agent["state"][0,:,:] = state
    agent["state"][-1:] = state
    return agent["state"]

def get_state( agent ):
    # return np.expand_dims( agent["state"], axis=0 )
    return agent["state"]

def set_score_average( agent, score ):
    agent["scores"].append( score )
    return sum( agent["scores"][-50:] ) / len( agent["scores"][-50:] )

def update_position( agent, act ):
    if ( act == 2 or act == 4 ) and ( agent["pos"] < 14 ):
        agent["pos"] += 1
    elif ( act == 3 or act == 5 ) and ( agent["pos"] > -17 ):
        agent["pos"] -= 1

def set_ball_distance( agent, distance ):
    agent["ball_distance"] = distance

def get_ball_distance( agent ):
    return agent["ball_distance"]
