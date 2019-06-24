from vrep_env import vrep_env
from vrep_env import vrep
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math as Math
import time


vrep_scenes_path = "C:\\Program Files\\V-REP3\\V-REP_PRO_EDU\\scenes"

NUM_OF_MOTORS = 18
g = np.transpose(np.array([0.0, 0.0, -9.8100004196167]))
MOTORS_PREFIX = "ART_"
BIOLOID_IN_VREP = 'BIOLOID'
FLOOR_IN_VREP = '5mx5mWoodenFloor'
IMU_IN_VREP = 'GyroSensor'
R_PROXIMITY = 'right_proximity'
L_PROXIMITY = 'left_proximity'
L_LEG = 'F12_F5_DERECHA'
R_LEG = 'F12_F5_IZQUIERDA'

collisionR = 'collisionR#'
collisionL = 'collisionL#'

gmax = 45
steps_count_limit = 120

A_DIM = 10
S_DIM = 32

MARGIN = 15 * (math.pi / 180)

# ############################################ POSTUREs ###############################################

INITIAL_POS = [-0.9006481, 0.89546058, -1.09509861, 1.08479397, -0.51174708, 0.50655956
    , -0.80342284, 0.77776685, -0.10749469, 0.09207294, -0.60385521, 0.57819922,
               -1.19744099, 1.18201923, 0.53214515, -0.54756691, -0.11261181, 0.09207294]
STAND_POS = [0.0] * 18
STAND_POS[6] = 1.84731506875
STAND_POS[7] = -1.84731506875  # 151 = 45 degrees


# #####################################################################################################


class Humanoid(vrep_env.VrepEnv):
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=vrep_scenes_path + '/Bioloid.ttt'
    ):
        vrep_env.VrepEnv.__init__(self, server_addr, server_port, scene_path)

        joint_names = [
            MOTORS_PREFIX + '1',
            MOTORS_PREFIX + '2',
            MOTORS_PREFIX + '3',
            MOTORS_PREFIX + '4',
            MOTORS_PREFIX + '5',
            MOTORS_PREFIX + '6',
            MOTORS_PREFIX + '7',
            MOTORS_PREFIX + '8',
            MOTORS_PREFIX + '9',
            MOTORS_PREFIX + '10',
            MOTORS_PREFIX + '11',
            MOTORS_PREFIX + '12',
            MOTORS_PREFIX + '13',
            MOTORS_PREFIX + '14',
            MOTORS_PREFIX + '15',
            MOTORS_PREFIX + '16',
            MOTORS_PREFIX + '17',
            MOTORS_PREFIX + '18',
        ]
        shape_names = [IMU_IN_VREP, R_LEG, L_LEG, FLOOR_IN_VREP, BIOLOID_IN_VREP]
        # self.camera = self.get_object_handle('camera')

        # Actuators
        self.oh_joint = list(map(self.get_object_handle, joint_names))
        # Shapes
        self.oh_shape = list(map(self.get_object_handle, shape_names))

        # todo
        # self.collision_handle_r = self.get_collision_handle(collisionR)
        # self.collision_handle_l = self.get_collision_handle(collisionL)

        # number of actuators
        num_act = len(self.oh_joint)

        # size of observations space
        num_obs = S_DIM

        act = np.array([1] * num_act)
        obs = np.array([np.inf] * num_obs)

        self.action_space = spaces.Box(-act, act)
        self.observation_space = spaces.Box(-obs, obs)


        self.t = 1.0

        self.steps_count = 0

        print('HumanoidVrepEnv: initialized')

        pass  # end init

    def _make_observation(self):
        ret = []

        g = self.get_gravity_vector()

        # imu
        lin_vel, ang_vel = self.obj_get_velocity(self.oh_shape[0])

        # joint pos
        q = []

        # joint vel
        qd = []

        for handle in self.oh_joint[8:]:
            q += [self.obj_get_joint_angle(handle)]
            qd += [self.get_obj_floating_parameter(handle)]

        assert len(qd) == len(self.oh_joint[8:])

        # touch sensor todo : print the output
        # tc_r = self.read_collision(self.collision_handle_r)
        # tc_l = self.read_collision(self.collision_handle_l)
        # print(" tc_r = " + str(tc_r))
        tc = [1, 1]  # todo

        # duration of sim
        duration = [self.t * 0.01]  # it's multiply by this factor to make state-vec homogeneous i.e (max = 0.14)

        ret= np.r_[g,np.r_[lin_vel,np.r_[ang_vel,np.r_[q,np.r_[qd,np.r_[tc,duration]]]]]]
        self.observation = np.array(ret).astype('float32')
        assert len(self.observation) == S_DIM

    # this fun must receive an Action where len(Action) = NUM_OF_ACTIVE_MOTORS i.e. (10)
    def _make_action(self, a):

        # here are the fixed joint i.e. first 8 joints
        fixed_joints = INITIAL_POS[0:8]

        for i_oh, i_a in zip(self.oh_joint[0:8], fixed_joints):
            self.obj_set_position_target(i_oh, i_a)

        action_scaled = scale_action(a)
        action_ipm = dynamexil2rad(np.array(self.ipm()))

        # this must be 10-sized
        final_action = action_ipm + action_scaled
        assert len(final_action) == 10

        for i_oh, i_a in zip(self.oh_joint[8:], final_action):
            self.obj_set_position_target(i_oh, i_a)

        pass

    # todo : edit this to be better
    def _get_reward(self):

        gx, gy, gz = self.observation[0:3]
        vx, vy, vz = self.observation[3:6]
        wx, wy, wz = self.observation[6:9]
        tc_r, tc_l = self.observation[29:31]
        duration = self.observation[31]

        a0 = 1.0
        a1 = 0.001
        a2 = 25
        a3 = 0.01
        a4 = 0.001
        a5 = 0.015
        a6 = 0.01  # this weight for duration

        speed_reward = a0 * vx
        oscillations_penalty = - a1 * abs(wy)
        falling_penalty = -a2 * (np.abs(Math.degrees(angle_between([gx, gy, gz], [0, 0, -1]))) >= gmax)
        tilting_penalty = -a3 * angle_between([gx, gy, gz], [0, 0, -1])
        slip_rotation_penalty = - a4 * abs(wz)
        one_foot_support_reawrd = a5 * abs(tc_r - tc_l)
        long_duration_reward = 0.12  # todo : review this..

        reward = speed_reward + \
                 oscillations_penalty + \
                 falling_penalty + \
                 slip_rotation_penalty + \
                 tilting_penalty + \
                 one_foot_support_reawrd + \
                 long_duration_reward

        return reward

    def fall(self):
        gx, gy, gz = self.observation[0:3]
        is_fall = np.abs(Math.degrees(angle_between([gx, gy, gz], [0, 0, -1]))) >= gmax
        return is_fall

    def _done(self):
        done = self.fall() | (self.steps_count >= steps_count_limit)
        return done

    def step(self, action):

        # Actuate
        self._make_action(action)
        # Step
        self.step_simulation()
        self.steps_count += 1
        # Observe
        self._make_observation()

        # Reward
        reward = self._get_reward()

        # is done
        done = self._done()

        return self.observation, reward, done, {}

    # TODO : debug this
    def reset(self):
        """Gym environment 'reset'
        """
        self.t= 1.0
        self.steps_count=0
        if self.sim_running:
            self.stop_simulation()
            time.sleep(2)
        self.start_simulation()
        self.step([0]*10)
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()
        self.step_simulation()

        return self.observation

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # return the active motors action (10) in [dy] TODO : try to replace the equations
    def ipm(self):
        increment = 0.2
        IPM = [0] * 10
        i = self.t

        IPM[0] = -0.000275921606029018 * Math.pow(i, 6) + 0.0224866058355711 * Math.pow(i,
                                                                                        5) - 0.554002659792176 * Math.pow(
            i, 4) + 5.27359561063168 * Math.pow(i, 3) - 16.6250714681205 * Math.pow(i,
                                                                                    2) + 1.07731363098685 * i + 492.577239841013
        IPM[1] = -0.000140124106979082 * Math.pow(i, 6) + 0.00957747098059501 * Math.pow(i,
                                                                                         5) - 0.17601617035749 * Math.pow(
            i, 4) + 0.366976848153587 * Math.pow(i, 3) + 13.7934195723444 * Math.pow(i,
                                                                                     2) - 82.4753256531787 * i + 597.355314009854;
        IPM[2] = 0.00000140489885303541 * Math.pow(i, 12) - 0.000137098037479588 * Math.pow(i,
                                                                                            11) + 0.00593962266650624 * Math.pow(
            i, 10) - 0.150467310047963 * Math.pow(i, 9) + 2.47028882826084 * Math.pow(i,
                                                                                      8) - 27.5207922276621 * Math.pow(
            i, 7) + 211.699439008591 * Math.pow(i, 6) - 1121.79501433143 * Math.pow(i, 5) + 4012.95422656784 * Math.pow(
            i, 4) - 9303.65422686671 * Math.pow(i, 3) + 13044.0770540131 * Math.pow(i,
                                                                                    2) - 9823.0075743571 * i + 3401.38059312374;
        IPM[3] = 0.00000011672084327608 * Math.pow(i, 14) - 0.000012662080312946 * Math.pow(i,
                                                                                            13) + 0.000615851466539234 * Math.pow(
            i, 12) - 0.017737457506048 * Math.pow(i, 11) + 0.336649535916704 * Math.pow(i,
                                                                                        10) - 4.43504476680317 * Math.pow(
            i, 9) + 41.6304852819428 * Math.pow(i, 8) - 281.48687768283 * Math.pow(i, 7) + 1370.97665082275 * Math.pow(
            i, 6) - 4763.85605618472 * Math.pow(i, 5) + 11574.9816970237 * Math.pow(i, 4) - 18997.7828225834 * Math.pow(
            i, 3) + 19842.4241425091 * Math.pow(i, 2) - 11754.8617513084 * i + 3597.71125928176;
        IPM[4] = -0.0000148141287899076 * Math.pow(i, 10) + 0.00119700195674767 * Math.pow(i,
                                                                                           9) - 0.0415853177723568 * Math.pow(
            i, 8) + 0.813243772863493 * Math.pow(i, 7) - 9.8518102730438 * Math.pow(i, 6) + 76.7204688187178 * Math.pow(
            i, 5) - 385.724100593995 * Math.pow(i, 4) + 1221.58291298422 * Math.pow(i, 3) - 2295.15666871898 * Math.pow(
            i, 2) + 2304.35552800843 * i - 643.058578051134;
        IPM[5] = 0.00000083846544852191 * Math.pow(i, 10) - 0.00000812251364211665 * Math.pow(i,
                                                                                              9) - 0.00194976423616432 * Math.pow(
            i, 8) + 0.0861933684303852 * Math.pow(i, 7) - 1.69603346375747 * Math.pow(i,
                                                                                      6) + 18.9297051399361 * Math.pow(
            i, 5) - 127.429816133823 * Math.pow(i, 4) + 517.304913254853 * Math.pow(i, 3) - 1215.69884225685 * Math.pow(
            i, 2) + 1501.00777573452 * i + 34.6214069610031;
        IPM[6] = 0.0000217898674376092 * Math.pow(i, 10) - 0.00168504911464163 * Math.pow(i,
                                                                                          9) + 0.0556171425516435 * Math.pow(
            i, 8) - 1.02212510842425 * Math.pow(i, 7) + 11.4361911769592 * Math.pow(i, 6) - 79.9218400189809 * Math.pow(
            i, 5) + 343.33892677256 * Math.pow(i, 4) - 852.044744171629 * Math.pow(i, 3) + 1066.9698822212 * Math.pow(i,
                                                                                                                      2) - 535.985198422995 * i + 651.394169313619;
        IPM[7] = 0.0000248742798796263 * Math.pow(i, 10) - 0.00191083364372945 * Math.pow(i,
                                                                                          9) + 0.0626362655558014 * Math.pow(
            i, 8) - 1.1448672116933 * Math.pow(i, 7) + 12.8113562500397 * Math.pow(i, 6) - 90.8916292359355 * Math.pow(
            i, 5) + 410.536161792987 * Math.pow(i, 4) - 1157.63710507665 * Math.pow(i, 3) + 1944.09966170448 * Math.pow(
            i, 2) - 1764.8116655254 * i + 1056.68768177018;
        IPM[8] = -0.000602821242505626 * Math.pow(i, 6) + 0.0405315649727053 * Math.pow(i,
                                                                                        5) - 0.916931723020958 * Math.pow(
            i, 4) + 8.3576698861698 * Math.pow(i, 3) - 25.507120415259 * Math.pow(i,
                                                                                  2) - 1.29495830014807 * i + 497.088587623408;
        IPM[9] = -0.00021958932773343 * Math.pow(i, 6) + 0.0135980370756124 * Math.pow(i,
                                                                                       5) - 0.239784506804081 * Math.pow(
            i, 4) + 0.466436883005924 * Math.pow(i, 3) + 18.9979103475509 * Math.pow(i,
                                                                                     2) - 113.421209371814 * i + 620.07665583061;
        TIME_LIMIT = 14.7

        if self.t >= TIME_LIMIT:
            self.t = 2.1
        else:
            self.t += increment
        return IPM

    def get_gravity_vector(self):
        euler_angles = self.obj_get_orientation(self.oh_shape[0], self.oh_shape[3])
        roll, pitch, yaw = euler_angles
        R = np.array([[Math.cos(pitch) * Math.cos(yaw),
                       Math.sin(pitch) * Math.sin(roll) * Math.cos(yaw) - Math.sin(yaw) * Math.cos(roll),
                       Math.sin(pitch) * Math.cos(roll) * Math.cos(yaw) + Math.sin(roll) * Math.sin(yaw)],
                      [Math.sin(yaw) * Math.cos(pitch),
                       Math.sin(pitch) * Math.sin(roll) * Math.sin(yaw) + Math.cos(roll) * Math.cos(yaw),
                       Math.sin(pitch) * Math.sin(yaw) * Math.cos(roll) - Math.sin(roll) * Math.cos(yaw)],
                      [-Math.sin(pitch), Math.sin(roll) * Math.cos(pitch), Math.cos(pitch) * Math.cos(roll)]
                      ])
        gravity = np.matmul(g, R)
        return gravity


def dynamexil2rad(dy):
    degree = -2.62 + (dy * 0.00511711875)
    return degree


# retern val between [-margin,margin]
# action between [-1,1]
def scale_action(action, action_diminstion=10):
    max_array = np.array([MARGIN] * action_diminstion)
    min_array = np.array([-MARGIN] * action_diminstion)
    assert len(max_array) == action_diminstion
    a = (max_array - min_array) / 2
    b = max_array - a
    action_scaled = action * a + b
    return action_scaled


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    ret=np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    # print(math.degrees(ret))
    return ret


if __name__ == "__main__":

    robot = Humanoid()
    robot.reset()
    while True:
        robot.step([0.0] * 10)
