
############# General info ####################
NUMBER_OF_AXES = 3  # x, y, z

#################  Header indices and values ######################

ACCELEROMETER_OFFSET_POSITION = 2
GYROSCOPE_OFFSET_POSITION = 3
MAGNETOMETER_OFFSET_POSITION = 4
ACCELEROMETER_LENGTH_POSITION = 6
GYROSCOPE_LENGTH_POSITION = 7
MAGNETOMETER_LENGTH_POSITION = 8
TIMESTAMP_INDEX = [10, 11]

CONST_ID = [13579, 24680]

############# Accelerometerinfo ###########
G = 9.81
ACCELEROMETER_NUMBER_OF_BITS = 16
ACCELEROMETER_SAMPLING_FREQUENCY = 1000  # Hz

#################### Gyroscope info ####################
GYROSCOPE_NUMBER_OF_BITS = 16
GYROSCOPE_SAMPLING_FREQUENCY = 1000  # Hz

################### Magnetometerinfo ####################
# use 9150 values for Spikelog16 and Ratlog64
MAGNETOMETER_9150_NUMBER_OF_BITS = 13
MAGNETOMETER_9150_RANGE = 1200e-6

# for all other loggers use 9250 values
MAGNETOMETER_9250_NUMBER_OF_BITS = 14
MAGNETOMETER_9250_RANGE = 4800e-6

MAGNETOMETER_SAMPLING_FREQUENCY = 1000  # Hz
MAGNETOMETER_NUMBER_OF_BITS = 13

###################### Timestamp info ###################
TIME_RESOLUTION = 62.5e-3  # ms
