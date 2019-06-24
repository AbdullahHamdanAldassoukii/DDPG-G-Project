import modern_robotics as mr
import numpy as np

class Kinematic:
    """
        In this class will find the forward kinematic for the (Modified) humanoid Robot
        using a method called POE (Product Of Exponential) Formula which
        consists of these steps :

        (1) Attach tow frames -> spatial frame (Space frame) {S} and -> body frame {B} ( i.e. {H} )

        (2) Find M = T(0) the T matrix when all theta are zeroes
    """

    def __init__(self,g0_h):
        L1, L2, L3, L4, L5, L6, L7 = [75, 74, 14, 14, 0, 100, 42];
        pos = np.transpose([L7, L5, L1 + L2 + L6])  # 3x1 position of {H} in {S} coords
        self.M = np.r_[np.c_[np.eye(3), pos],
                       [[0, 0, 0, 1]]]  # 4x4

        """
        (3) Find Screw axis S1,S2...S6 for each joint when theta = 0 i.e. 512 in the motor coordinates    
        """
        # here we calc the angular velocity of Joints along {S}
        w1 = np.transpose(np.array([[0, 1, 0]]))
        w2 = np.transpose(np.array([[1, 0, 0]]))
        w3 = np.transpose(np.array([[1, 0, 0]]))
        w4 = np.transpose(np.array([[1, 0, 0]]))
        w5 = np.transpose(np.array([[0, 1, 0]]))
        w6 = np.transpose(np.array([[0, 0, -1]]))

        # here are the q list (note : q is a point along the joint axis in {S} coords )
        q1 = np.transpose(np.array([[0, 0, 0]]))
        q2 = np.transpose(np.array([[0, 0, 0]]))
        q3 = np.transpose(np.array([[0, L5, L1]]))
        q4 = np.transpose(np.array([[0, L4, L1 + L2]]))
        q5 = np.transpose(np.array([[0, 0, L1 + L2]]))
        q6 = np.transpose(np.array([[0, L5, 0]]))

        # linear vel
        v1 = np.cross(-w1.transpose(), q1.transpose()).transpose()
        v2 = np.cross(-w2.transpose(), q2.transpose()).transpose()
        v3 = np.cross(-w3.transpose(), q3.transpose()).transpose()
        v4 = np.cross(-w4.transpose(), q4.transpose()).transpose()
        v5 = np.cross(-w5.transpose(), q5.transpose()).transpose()
        v6 = np.cross(-w6.transpose(), q6.transpose()).transpose()

        # Screw axis
        S1 = np.r_[w1, v1]
        S2 = np.r_[w2, v2]
        S3 = np.r_[w3, v3]
        S4 = np.r_[w4, v4]
        S5 = np.r_[w5, v5]
        S6 = np.r_[w6, v6]
        # Screw axis list
        self.Slist = np.c_[S1, S2, S3, S4, S5, S6]
        self.setInitialHeadGravity(g0_h)
        # CLASS END

    """(5) Calculate Trans. matrix :"""

    # this method will return the forward kinematic (Trans. Matrix) Tsh (i.e. {S} -> {H} )
    def Tsh(self, thetaList):
        thetaList = thetaList.transpose()
        return mr.FKinSpace(self.M, self.Slist, thetaList)

    # this method will return the forward kinematic (Trans. Matrix) Ths (i.e. {H} ->{S} )
    def Ths(self, thetaList):
        return np.linalg.pinv( self.Tsh(thetaList))

    # this method will return the jacobian matrix in {S} space coordinates
    def Js(self, thetaList):
        return mr.JacobianSpace(self.Slist, thetaList)

    # this method return the jacobian in the {H} head frame
    def Jb(self, thetaList):
        x = mr.Adjoint(self.Ths(thetaList))
        y = self.Js(thetaList)
        return x.dot(y)

    # this method return the Twist vector of the end effector (i.e. the head frame) in {H} coordinates
    def Vb(self, theta, theta_dot):
        # the returned vector is Twist vector [ vb wb ] ; 6x1
        Vb = self.Jb(theta).dot(theta_dot)
        # linear velocity of end-effector ( {H} ) in {H} coordinates
        vb = Vb[0:3]
        # angular veocity of end-effector ( {H} )  in {H} coordinates
        wb = Vb[3:6]
        return vb, wb

    """
    this method is a setter for g0 where g0 is the gravity vector (3x1) of the head {H}
    at 'STANDING' posture represented in {H} coordinates at the initializing stage 
    """
    def setInitialHeadGravity(self,
                              g0_h  # 3x1 vector [x;y;z]
                              ):
        g0_h=np.r_[g0_h, np.array([[1]])]
        Ths=self.Ths(np.zeros((6, 1)))
        self.g0 = Ths.dot(g0_h) # 4x1 [ x ; y ; z ; 1]
        return

    """
    this method is a getter for gravity vector of {H} in the {H} coordinates ((the final g vector ))
    output: 3x1
    """
    def g_km(self,theta):
        g_km=self.Tsh(theta).dot(self.g0);
        return g_km[0:3,:]