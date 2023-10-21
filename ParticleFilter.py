import numpy as np
import numpy.random as random
from scipy.stats import entropy
from scipy.stats import chisquare 
import ENV

tavBU = 1
tavTD = 10

genParticle = np.vectorize(lambda pi: random.choice([0,1], 1, p=[1-pi, pi])[0])

# featureMaps = (x,y,K) feature maps
def selectFiltersByAttention( featureMaps, nParticles, dqn, env ):
    # f = calc average values for each (x*y) map, totally K averages
    f = np.mean( featureMaps, axis=(0,1) )

    # v = normalize each average to sum 1
    v = f / np.sum( f )
    
    
    #caclulate and normalized the entropy of each feature map
    E = entropy(featureMaps)
    E = E / np.sum(E,axis=(0,1))
    
    #calculate the chi-square value of each feature map
    X2 = chisquare(featureMaps, f_exp=None, ddof=0, axis=0)
    
    # v = normalize each average to sum 1
    v = 1/2 * (v + E) / X2;

    # exponentiate and normalize, so max will be 1
    p = np.exp(v * tavBU)
    p = p / np.max( p )

    # generate particles using p
    particles = [genParticle(p) for _ in range(nParticles)]
    #particles = np.ones((250,512))

    # normalize each particle
    A = particles / np.sum( particles, axis=1 )[:, np.newaxis]

    # find prediction errors between R (reward using all filters)
    filteredFeats = [
        featureMaps *
        np.repeat(
            np.repeat(
                pi[np.newaxis], featureMaps.shape[0], axis=0
            )[np.newaxis], featureMaps.shape[1], axis=0
        ) for pi in A
    ]
    actions = np.argmax( dqn.predict(
        np.concatenate(
            (np.expand_dims(np.expand_dims(featureMaps.flatten(),0),0),
             *[np.expand_dims(np.expand_dims(a.flatten(),0),0) for a in filteredFeats]),
            axis=0
        ),
        verbose=False
    ), axis=1 )
    uniqueRewards = [r[1] for r in [ENV.freezeStep( env, act ) for act in range(env.action_space.n)]]
    rewards = np.array([uniqueRewards[act] for act in actions])

    sigma = np.square( rewards[0] - rewards[1:] )

    # resample new particles
    Px = np.exp( -(sigma - np.min(sigma)) * tavTD )
    Px = Px / np.sum(Px)
    newParticleIndexes = random.choice( np.arange(0,250), nParticles, p=Px )
    newParticles = np.take( particles, newParticleIndexes, axis=0 )

    # average of new particles
    avgParticles = np.mean( newParticles, axis=0 )

    # attention vector
    attVector = avgParticles / np.sum( avgParticles )
    attentionFeats = featureMaps * np.repeat(
        np.repeat(
            attVector[np.newaxis], featureMaps.shape[0], axis=0
        )[np.newaxis], featureMaps.shape[1], axis=0
    )
    return attentionFeats.flatten()
