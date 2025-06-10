""" 
This module provides utilities class for the simulator
"""

import numpy as np

class EulerInt:
    ''' Provides methods relevant for using the
        Euler method to integrate an ODE.

        Usage:

        int=EulerInt()
        while int.time <= int.sim_time:
            dx = f(x)
            int.integrate(x,dx)
            int.next_time
    '''

    def __init__(self):
        self.dt = 0.01
        self.sim_time = 10
        self.time = 0.0
        self.times = []
        self.global_times = []

    def set_dt(self, val):
        ''' Sets the integrator step length
        '''
        self.dt = val

    def set_sim_time(self, val):
        ''' Sets the upper time integration limit
        '''
        self.sim_time = val

    def set_time(self, val):
        ''' Sets the time variable to "val"
        '''
        self.time = val

    def next_time(self, time_shift=0):
        ''' Increment the time variable to the next time instance
            and store in an array
        '''
        self.time = self.time + self.dt
        self.times.append(self.time)
        self.global_times.append(self.time + time_shift)

    def integrate(self, x, dx):
        ''' Performs the Euler integration step
        '''
        return x + dx * self.dt


class ShipDraw:
    ''' This class is used to calculate the coordinates of each
        corner of 80 meter long and 20meter wide ship seen from above,
        and rotate and translate the coordinates according to
        the ship heading and position
    '''

    def __init__(self):
        self.l = 80.0
        self.b = 20.0

    def local_coords(self):
        ''' Here the ship is pointing along the local
            x-axix with its center of origin (midship)
            at the origin
            1 denotes the left back corner
            2 denotes the left starting point of bow curvatiure
            3 denotes the bow
            4 the right starting point of the bow curve
            5 the right back cornier
        '''
        x1, y1 = -self.l / 2, -self.b / 2
        x2, y2 = self.l / 4, -self.b / 2
        x3, y3 = self.l / 2, 0.0
        x4, y4 = self.l / 4, self.b / 2
        x5, y5 = -self.l / 2, self.b / 2

        x = np.array([x1, x2, x3, x4, x5, x1])
        y = np.array([y1, y2, y3, y4, y5, y1])
        return x, y

    def rotate_coords(self, x, y, psi):
        ''' Rotates the ship an angle psi
        '''
        x_t = np.cos(psi) * x - np.sin(psi) * y
        y_t = np.sin(psi) * x + np.cos(psi) * y
        return x_t, y_t

    def translate_coords(self, x_ned, y_ned, north, east):
        ''' Takes in coordinates of the corners of the ship (in the ned-frame)
            and translates them in the north and east direction according to
            "north" and "east"
        '''
        x_t = x_ned + north
        y_t = y_ned + east
        return x_t, y_t