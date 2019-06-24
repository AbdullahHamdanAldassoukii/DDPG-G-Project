# !/usr/bin/python

import smbus
import math
import time
import numpy as np


class IMU:

    def __init__(self):

        ##################### sensor Consts & Power management registers ######################
        self.power_mgmt_1 = 0x6b
        self.power_mgmt_2 = 0x6c
        self.gyro_scale = 131.0
        self.accel_scale = 16384.0
        self.bus = smbus.SMBus(1)  # or bus = smbus.SMBus(1) for Revision 2 boards
        self.address = 0x68  # This is the address value read via the i2cdetect command
        # Now wake the 6050 up as it starts in sleep mode
        self.bus.write_byte_data(self.address, self.power_mgmt_1, 0)
        #######################################################################################

        # calibrating...
        self.calibrate(1000)

        ########################### Initial Values #############################

        # This value must be adjusted after powring the Motors in zero state
        # and they are the deferance from g=[0 0 1] (the gravity vector)
        # todo : test with clibrating the acc
        self.acc_x_bias = -0.095
        self.acc_y_bias = -0.515
        self.acc_z_bias = -0.096

        (gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z) = self.read_all()
        self.last_x = self.get_x_rotation(acc_x, acc_y, acc_z)
        self.last_y = self.get_y_rotation(acc_x, acc_y, acc_z)
        self.last_z=0.0

        self.gyro_x_total = (self.last_x) - self.gyro_x_bias
        self.gyro_y_total = (self.last_y) - self.gyro_y_bias
        self.gyro_z_total = 0.0

        self.g0 = np.array([0, 0, -1])  # this must be computed
        self.v = np.array([0, 0, 0])  # linear velocity

        self.K = 0.9
        self.K1 = 1 - self.K
        self.time_diff = 0.01
        #########################################################################

        # start
        self.loop()

        return

    def calibrate(self, n):
        print("\ncalbring ...")
        x, y, z, ex, ey, ez = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(0, n):
            (gyro_x, gyro_y, gyro_z, acc_x, acc_y,acc_z) = self.read_all()
            # print("\n\nacc_x ="+str(acc_x)+"\nacc_y ="+str(acc_y)+"\nacc_z = "+str(acc_z))

            ex += -acc_x
            ey += -acc_x
            ez += -1.0 - acc_x

            x += gyro_x
            y += gyro_y
            z += gyro_z
            time.sleep(0.003)

        print("end calibrating !!")

        self.acc_x_bias= ex / n
        self.acc_y_bias= ey / n
        self.acc_z_bias= ez / n

        self.gyro_x_bias = x / n
        self.gyro_y_bias = y / n
        self.gyro_z_bias = z / n
        return

    def read_all(self):
        raw_gyro_data = self.bus.read_i2c_block_data(self.address, 0x43, 6)
        raw_accel_data = self.bus.read_i2c_block_data(self.address, 0x3b, 6)

        gyro_scaled_x = self.twos_compliment((raw_gyro_data[0] << 8) + raw_gyro_data[1]) / self.gyro_scale
        gyro_scaled_y = self.twos_compliment((raw_gyro_data[2] << 8) + raw_gyro_data[3]) / self.gyro_scale
        gyro_scaled_z = self.twos_compliment((raw_gyro_data[4] << 8) + raw_gyro_data[5]) / self.gyro_scale

        accel_scaled_x = self.twos_compliment((raw_accel_data[0] << 8) + raw_accel_data[1]) / self.accel_scale
        accel_scaled_y = self.twos_compliment((raw_accel_data[2] << 8) + raw_accel_data[3]) / self.accel_scale
        accel_scaled_z = self.twos_compliment((raw_accel_data[4] << 8) + raw_accel_data[5]) / self.accel_scale

        return (gyro_scaled_x, gyro_scaled_y, gyro_scaled_z, accel_scaled_x, accel_scaled_y, accel_scaled_z)

    def twos_compliment(self, val):
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

    def dist(self, a, b):
        return math.sqrt((a * a) + (b * b))

    def dist3(self, a, b, c):
        return math.sqrt((a * a) + (b * b) + (c * c))

    def get_y_rotation(self, x, y, z):
        radians = math.atan2(x, self.dist(y, z))
        return -math.degrees(radians)

    def get_x_rotation(self, x, y, z):
        radians = math.atan2(y, self.dist(x, z))
        return math.degrees(radians)

    def loop(self):
        roll = 0
        pitch = 0
        yaw = 0
        for i in range(0, 20000):
            time.sleep(self.time_diff - 0.005)

            # read sensor values
            (gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z) = self.read_all()

            ##########add the offset to the readings##########
            gyro_x -= self.gyro_x_bias
            gyro_y -= self.gyro_y_bias
            gyro_z -= self.gyro_z_bias

            acc_x -= self.acc_x_bias
            acc_y -= self.acc_y_bias
            acc_z -= self.acc_z_bias
            ###################################################

            #################### Calculate (Roll , Pitch , Yaw ) #####################
            """
            here we will calculate the (r,p,y) angles from the start position which is the 
            standing posture i.e. z axis aligned with the g (gravity) vector.
            """

            # w * dt { (dth/dt) x dt }
            gyro_x_delta = (gyro_x * self.time_diff)
            gyro_y_delta = (gyro_y * self.time_diff)
            gyro_z_delta = (gyro_z * self.time_diff)

            self.gyro_x_total += gyro_x_delta
            self.gyro_y_total += gyro_y_delta
            self.gyro_z_total += gyro_z_delta

            rotation_x = self.get_x_rotation(acc_x, acc_y, acc_z)
            rotation_y = self.get_y_rotation(acc_x, acc_y, acc_z)

            self.last_x = self.K * (self.last_x + gyro_x_delta) + (self.K1 * rotation_x)  # Roll
            self.last_y = self.K * (self.last_y + gyro_y_delta) + (self.K1 * rotation_y)  # Pitch
            self.last_z = self.K * (self.last_z + gyro_y_delta)  # Yaw

            # here we calculate the deference in degrees between old and current ( r , p , y ) to calculate the def_R
            roll_def = self.last_x - math.degrees(roll)
            pitch_def = self.last_y - math.degrees(pitch)
            yaw_def = self.last_z - math.degrees(yaw)

            roll = math.radians(self.last_x)
            pitch = math.radians(self.last_y)
            yaw = math.radians(self.last_z)

            print("\n**********************\nRoll : " + str(self.last_x) + "\nPitch : " + str(
                self.last_y) + "\nYaw :" + str(self.last_z) + "\n**********************\n")

            ###################################################################################

            ######################### Calculating the Rotation Matrix ############################
            """
            here we calculate the Rotation Matrix to use it for g vector  
            there R will be from the current Orientation of the MPU to the standing posture Orientation of MPU
            """

            self.R = np.array([
                [math.cos(pitch) * math.cos(yaw)
                    ,math.sin(pitch) * math.sin(roll) * math.cos(yaw) - math.sin(yaw) * math.cos(roll)
                    ,math.sin(pitch) * math.cos(roll) * math.cos(yaw) + math.sin(roll) * math.sin(yaw)],
                [math.sin(yaw) * math.cos(pitch),math.sin(pitch) * math.sin(roll) * math.sin(yaw) + math.cos(roll) * math.cos(yaw),math.sin(pitch) * math.sin(yaw) * math.cos(roll) - math.sin(roll) * math.cos(yaw)],
                [-math.sin(pitch), math.sin(roll) * math.cos(pitch), math.cos(pitch) * math.cos(roll)]
            ])

            # here we calculate the temp Rotation matrix from the previous orientation
            self.R_def = np.array([
                [math.cos(pitch_def) * math.cos(yaw_def),math.sin(pitch_def) * math.sin(roll_def) * math.cos(yaw_def) - math.sin(yaw_def) * math.cos(roll_def),math.sin(pitch_def) * math.cos(roll_def) * math.cos(yaw_def) + math.sin(roll_def) * math.sin(yaw_def)],
                [math.sin(yaw_def) * math.cos(pitch_def),math.sin(pitch_def) * math.sin(roll_def) * math.sin(yaw_def) + math.cos(roll_def) * math.cos(yaw_def),math.sin(pitch_def) * math.sin(yaw_def) * math.cos(roll_def) - math.sin(roll_def) * math.cos(yaw_def)],
                [-math.sin(pitch_def), math.sin(roll_def) * math.cos(pitch_def),math.cos(pitch_def) * math.cos(roll_def)]
            ])

            print("\n R :" + str(self.R))
            print("\n R_def :" + str(self.R_def))  # as a test : it must be eye(3) when the Robot is Stationary

            ########################################################################################

            ######### calculate the current gravity Vector #########

            self.g = np.dot(self.R, self.g0)

            # as a test : self.g in the r=0 p=90 y=0 posture must be g=[-1 0 0]
            print("current g : \n" + str(self.g))

            ########################################################

            ######################### Calculate linear velocity ###########################

            # first compute total acc vector
            self.a = np.array([acc_x, acc_y, acc_z])

            # calculate the integral of (a-g)
            v_inc = integral = np.dot(self.a - self.g, self.time_diff).dot(9.81)

            # calculate linear velocity
            self.v = self.v + integral


            ######################### ADD DATA FUSION ########################
            # Bv=0.2
            # v_inc= integral
            # v_old += Bv
            # self.v_net = v_old + v_inc
            #################################################################



            print("velocity (1):\n" + str(self.v))

            """ SECOND method to calc vel"""

            # calculate linear velocity
            v_old = np.dot(self.R_def,self.v) 
            self.v = v_old + integral
            print("velocity (2) :\n"+str(self.v))

            
            ###############################################################################

            # end for
        return
"                 ->->->->->-> NOTE  : I think this is the last version of the IMU script <-<-<-<-<-<-<-<-   "

if __name__=="__main__":
    a=IMU()