# representation details, could be anything
DIRECTION_THETAS_SIZE = 36
# The larger thetas size, the more sparse the distance representation will be
DISTANCE_THETAS_SIZE = 100
# max distance that can go int SDDS network
MAX_DISTANCE = 3
# what is considered the standard step. The robot distance sensors will detect null when they are smaller than step*2

STEP_DISTANCE = 0.5

STEP_DISTANCE_LOWER_BOUNDARY = STEP_DISTANCE / 2
STEP_DISTANCE_UPPER_BOUNDARY = STEP_DISTANCE * 2

STEP_DISTANCE_CLOSE_THRESHOLD = STEP_DISTANCE * 2
STEP_DISTANCE_NULL_CONNECTION = STEP_DISTANCE
STEP_DISTANCE_BASIC_STEP = STEP_DISTANCE

# number of sampled rotations for each datapoint ( at roughly equal intervals )
ROTATIONS = 24
# ROTATIONS = 1
# how many rotations are taken into account for each datapoint ( so how many cameras shot at once )
ROTATIONS_PER_FULL = 1
# how many offsets are included in the training of the abstraction block
OFFSETS_PER_DATAPOINT = 24
MANIFOLD_SIZE = 128

# These are optimal parameters when the networks are trained during exploration
# Need to be found empirically
THRESHOLD_RECONSTRUCTION_ABSTRACTION_NETWORK = 0.020
THRESHOLD_IMAGE_DISTANCE_NETWORK = 0.020
THRESHOLD_IMAGE_RAW_DISTANCE_NETWORK = 0.010
THRESHOLD_ADJACENCY_DETECTOR = 0.040

THRESHOLD_MANIFOLD_NON_ADJACENT_LOSS = 0.040
THRESHOLD_MANIFOLD_PERMUTATION_LOSS = 0.040

THRESHOLD_SSDIR_NETWORK = 0.020
THRESHOLD_SDIRDISTSTATE_NETWORK = 0.005

# OTHER PARAMS
# removes connections from the same datapoint which are too aligned

# removes up to 80% of the connections without significant loss in the structure of the topological graph
REDUNDANCY_CONNECTION_ANGLE = 5

# different experimental parameters
EXPERIMENTAL_BINARY_SIZE = 24
