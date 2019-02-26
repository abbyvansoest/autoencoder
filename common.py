import scipy

log = True
noisy = False

BATCH_SIZE = 256
TEST_SIZE = 4

learning_rate = 0.01
num_steps = 100000
display_step = 1000

# normalize data to the [0,1] range
def normalize_obs(data): 
    # normalize full state.
    for i in range(len(data[0])):
        i_vals = [x[i] for x in data]
        max_i_val = max(i_vals)
        for obs in data:
            obs[i] /= max_i_val
    return data

def log_test(test, pred):
    print("-----")
    if not log:
        return    
    for j in range(len(pred)):
        print("eucl: " + str(scipy.spatial.distance.euclidean(pred[j], test[j])))
        print("mse: " + str(((pred[j] - test[j]) ** 2).mean()))
    print("GLOBAL MSE: " + str(((pred - test) ** 2).mean()))