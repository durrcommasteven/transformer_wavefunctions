import numpy as np
"""
We would like to understand the degree to which the transformer is learning structure within states
as opposed to memorizing. We can quantify this by invertibly scrambling each spin, and trying to learn
the resulting wavefunction
"""

def scramble(
    states: np.array,
    invert=False,
    reps = 4,
    seed: int = 0,
):
    """
    use a feistel cipher to map states invertibly to 
    pseudorandom states
    """
    sequence_length = states.shape[-1]
    assert sequence_length%2==0, (
        "expected even length sequence, recieved length {sequence_length}"
    )
    
    #define a round function
    def round_function(states, key):
        """
        Given states and a key, compute a random 
        array of binary strings with the same shape 
        as states
        
        
        """
        state_length = states.shape[-1]
        key_hash = hash(key)
        
        output = np.zeros_like(states, dtype=np.int32)
        
        for idx in range(states.shape[0]):
            total_hash = hash(key_hash*hash(states[idx, :].tobytes()) % 2**63)
            
            bin_state_hash = bin(abs(total_hash))[2:]
            
            """
            I want to use this for arbitrarily long systems
            """
            while len(bin_state_hash)<state_length:
                bin_state_hash += bin(abs(hash(bin_state_hash)))[2:]
            
            output[idx, :] = [int(n) for n in bin_state_hash[:state_length]]
        
        return output
    
    argument = states
    
    if invert == False:
        keys = np.arange(reps)+seed
    else:
        keys = np.arange(reps)+seed
    
    for i, key in enumerate(keys):
        L = argument[:, :sequence_length//2]
        R = argument[:, sequence_length//2:]
        
        F = round_function(
            R,
            key=key
        )
        
        next_R = np.mod(L+F, 2).astype(np.int32)
        next_L = R.astype(np.int32)
        
        argument[:, :sequence_length//2] = next_L
        argument[:, sequence_length//2:] = next_R
    
    argument[:, :sequence_length//2] = next_R
    argument[:, sequence_length//2:] = next_L
    
    return argument