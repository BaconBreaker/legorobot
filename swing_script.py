import tensorflow as tf
import numpy as np
import cma
import rpyc
import time
from sklearn.linear_model import LinearRegression

class RobotController:
    """
    The robot controller class.
    """
    def __init__(self, connection):
        motor_module = connection.modules['ev3dev2.motor']
        self.motors = [motor_module.LargeMotor('outB'), motor_module.LargeMotor('outC')]
        sensors = conn.modules['ev3dev2.sensor.lego'] 
        self.gyro = sensors.GyroSensor('in2')
        
        self.motors[0].stop_action = 'hold'
        self.motors[1].stop_action = 'coast'
        
        self.up_position_degrees = -90 # position of legs fully in front
        self.maxspeed = 80 # maximum speed of leg movement
        self.minspeed = 3 # minimum speed of leg movement
        self.angle_normalizer = 0.01
        self.rate_normalizer = 0.01
    
    def reset(self):
        """ resets the robot, performs calibration and maintainance, return when robot is reset.
        Called at the start of each episode"""
        # move legs to base position
        self._move_motors_to_pos(0, 20)
        self._wait_until_no_standstill(tolerance = 0.05)
        _, angle = self.get_rate_and_angle()
        if abs(angle) >= 0.04:
            # drifted pretty far, recalibrate
            self._wait_until_no_standstill(tolerance = 0.01)
            self._calibrate_gyro()
            return
    
    def get_rate_and_angle(self):
        """ returns the rate and angle of the robot movement"""
        angle, rate = self.gyro.angle_and_rate
        return (rate * self.rate_normalizer, angle * self.angle_normalizer)

    def get_current_leg_pos(self):
        """ returns the curent leg pos, on a [-1, 1] interval"""
        return self.motors[0].position /  self.up_position_degrees
    
    def move_legs(self, pos):
        """ instructs the robot where to move its legs to, on a [-1, 1] interval. Function returns immediatly."""
        pos = max(-1., min(pos, 1.))
        pos_now = self.get_current_leg_pos()
        
        s = abs(pos - pos_now) / 2 *  self.maxspeed
        s = max(self.minspeed, min(s,  self.maxspeed))
        
        self._move_motors_to_pos(pos, s)
        return
    
    def _move_motors_to_pos(self, pos, speed):
        """
        moves the motors to a position [-1, 1] at speed [0,100].
        """
        assert -1. <= pos <= 1.,\
                    "{} is an invalid position, must be between -1 and 1 (inclusive)".format(pos)
        assert 0. <= speed <= 100.,\
                    "{} is an invalid position, must be between -1 and 1 (inclusive)".format(pos)
        
        self.motors[0].on_to_position(speed, position=pos*self.up_position_degrees, brake=True, block=False) # moves motors, non blocking.
        return
            
            
    def _calibrate_gyro(self):
        print('calibrating gyro')
        self.gyro.mode='GYRO-CAL'
        time.sleep(1)
        self.gyro.mode="GYRO-ANG"
        
    def _wait_until_no_standstill(self, tolerance=0.01):
        """
        waits until the gyro acceleration does not change for 3 seconds
        """
        print("Waiting to stand still.")
        change = 9999
        while abs(change) > tolerance: 

            rates = []
            times = []
            # sample for one second
            t0 = t = time.time()
            while t - t0 < 3:
                _, rate = self.gyro.angle_and_rate
                rates.append(rate)
                t = time.time()
                times.append(t)


            # check change in accel with linreg
            rates = abs(np.array(rates)) # take absolute rates so swinging doesnt cancel each other out
            times = np.array(times)
            lr = LinearRegression()
            lr.fit(times[:, np.newaxis], rates)  
            change = lr.coef_

        return

class RobotControllerVerboseWrapper(RobotController):
    """
    Wrapper / decorator for the robot controller to add verbose logging
    for design pattern see https://github.com/faif/python-patterns/blob/master/structural/decorator.py
    """
    def __init__(self, robotController):
        self.wrapped = robotController
        self.last_move = time.time()
        
    def reset(self):
        return self.wrapped.reset()
    
    def get_rate_and_angle(self):
        return self.wrapped.get_rate_and_angle()
    
    def get_current_leg_pos(self):
        return self.wrapped.get_current_leg_pos()
    
    def move_legs(self, pos):
        print("Moving legs to {}, time since last move {}".format(pos, time.time() - self.last_move))
        self.last_move = time.time()
        return self.wrapped.move_legs(pos)

class RobotControllerExceptionWrapper(RobotController):
    """
    Wrapper / decorator for the robot controller to handle time out exceptions
    for design pattern see https://github.com/faif/python-patterns/blob/master/structural/decorator.py
    """
    def __init__(self, robotController):
        self.wrapped = robotController
        
    def reset(self):
        try:
            return self.wrapped.reset()
        except TimeoutError:
            print('TimeoutError occured! retrying...')
            time.sleep(3)
            return self.reset()
    
    def get_rate_and_angle(self):
        try:
            return self.wrapped.get_rate_and_angle()
        except TimeoutError:
            print('TimeoutError occured! retrying...')
            time.sleep(3)
            return self.get_rate_and_angle()
    
    def get_current_leg_pos(self):
        try:
            return self.wrapped.get_current_leg_pos()
        except TimeoutError:
            print('TimeoutError occured! retrying...')
            time.sleep(3)
            return self.get_current_leg_pos()
    
    def move_legs(self, pos):
        try:
            return self.wrapped.move_legs(pos)
        except TimeoutError:
            print('TimeoutError occured! retrying...')
            time.sleep(3)
            return self.move_legs(pos)

try:
    ROBOT_HOSTNAME = "ev3dev.local"
    conn = rpyc.classic.connect(ROBOT_HOSTNAME, ipv6=True)
    conn.execute("print('Hello Slave. I am your , master!')")
except Exception as e:
    print(e)
    raise Exception('No conection to rpyc server on robot possible! Is the robot conneced? Is the rpyc server started?')

weights = []
timestep = []
rewards = []

rc = RobotControllerExceptionWrapper(RobotController(conn))

# Helper functions for setting the parameters of the neural network
def number_of_trainable_parameters(session):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = session.run(variables_names)    
    number_of_parameters = 0
    for v in values:
        number_of_parameters += v.size
    return number_of_parameters

def set_trainable_parameters(session, parameter_vector):
    assert number_of_trainable_parameters(session) == parameter_vector.size, \
      'number of parameters do not match: %r vs. %r' % (number_of_trainable_parameters(session), parameter_vector.size)
    variables_names = [v.name for v in tf.trainable_variables()]
    values = session.run(variables_names) 
    idx = 0
    for k, v in zip(tf.trainable_variables(), values):    
        new_value = parameter_vector[idx:idx + v.size].reshape(v.shape)
        k.load(new_value, session)  # load does not add a new node to the graph
        idx += v.size

# This is the function we attempt to optimize. It recieves data about the current state and feeds this into a neural network (defined later) to get an action, which it performs. During this, it keeps track of the maximal absolute rate observed to be able to pass a final reward back 
# Simple evaluation function, just considering a single (random) start state
def swing_motion(x, sess, duration=30):
    # set params to NN
    set_trainable_parameters(sess, x)
    
    # reset swing
    rc.reset()
    
    # simulate
    reward = 0
    pos = 0
    starttime = time.time()
    rate, angle = rc.get_rate_and_angle()
    while time.time()-starttime < duration :

        state = np.array([rate, angle, pos])
        
        pos = sess.run(action, {inputs: state.reshape(1,*state.shape)})[0][0]
        
        rc.move_legs(pos)
        
        rate, angle = rc.get_rate_and_angle()
        reward = max(reward, abs(rate))
    
    # log rewards of session
    print("total reward for session: ",reward)
    weights.append(x)
    timestep.append(starttime)
    rewards.append(reward)
    return -reward

# Neural network parameters
state_space_dimension = 3
number_of_actions = 1
number_of_hidden_neurons = 2 # currently unused, becomes relevant if the hidden layer is reimplemented.

# ## Neural network and CMA-ES
# Here, we define the policy network, which is very simple for our implementation, but with this setup it can easily be expanded to include hidden layers, or (with slightly more work) be recurrent.  
# Afterwards, a session is started and we run the CMA minimizer, using our guess for initial weights.
tf.reset_default_graph()

# Define policy network mapping state to action
with tf.name_scope('policy_network'):
    inputs = tf.placeholder(shape=[1, state_space_dimension],dtype=tf.float32) 
    #hidden = tf.layers.dense(inputs, number_of_hidden_neurons, activation=tf.tanh, use_bias=True)
    action = tf.layers.dense(inputs, number_of_actions, activation=tf.tanh, use_bias=True)

# Do the learning
init = tf.global_variables_initializer()

# final CMA cluster center
x_final = np.array([27.6228857208, -26.4064626352, -11.7426286304, 6.24397330849])
x_middle = np.array([13.3768323511, -7.8391528124, -2.50238021705, 1.06962506962])
x_early = np.array([10.4974431805, -3.76115577714, -0.857282768428, 0.824748263562])

with tf.Session() as sess:
    sess.run(init)
    sess.graph.finalize()
    initial_weights = x_final #np.array([10, -5, -1, 0.2]) 
    res = cma.fmin(swing_motion, initial_weights, 1, {'timeout': 14 * 60**2, 'ftarget':-4.,}, args=([sess]), restart_from_best=True)




