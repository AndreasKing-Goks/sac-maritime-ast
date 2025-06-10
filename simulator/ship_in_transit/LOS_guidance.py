""" 
This module provides classes that can be used for Line of Sight Guidance.
"""


import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, NamedTuple

###################################################################################################################
##################################### CONFIGURATION FOR LOS GUIDANCE ##############################################
###################################################################################################################

class LosParameters(NamedTuple):
    radius_of_acceptance: float
    lookahead_distance: float
    integral_gain: float
    integrator_windup_limit: float


###################################################################################################################
###################################################################################################################


class NavigationSystem:
    ''' This class provides a way of following a predifined route using
        line-og-sight (LOS) guidance law. The path to the textfile where
        the route is specified is given as an argument when calling the
        class. The route text file is formated as follows:
        x1 y1
        x2 y2
        ...
        where (x1,y1) are the coordinates to the first waypoint,
        (x2,y2) to the second, etc.
    '''

    def __init__(
            self, route,
            radius_of_acceptance=600,
            lookahead_distance=450,
            integral_gain=0.01,
            integrator_windup_limit=0.5,
    ):
        ## Initial internal attributes
        self.init_route = route
        self.init_ra = radius_of_acceptance
        self.init_r = lookahead_distance
        self.init_ki = integral_gain
        self.init_e_ct = 0
        self.init_e_ct_int = 0
        self.init_integrator_limit = integrator_windup_limit
        
        ## Internal attributes
        self.route = self.init_route
        self.ra = self.init_ra
        self.r = self.init_r
        self.ki = self.init_ki
        self.e_ct = self.init_e_ct
        self.e_ct_int = self.init_e_ct_int
        self.integrator_limit = self.init_integrator_limit
        
        self.load_waypoints(self.route)

    def load_waypoints(self, route, print_init_msg=False):
        ''' Reads the file containing the route and stores it as an
            array of north positions and an array of east positions
        '''
        # self.data = np.loadtxt(route)
        # self.data = route
        if print_init_msg:
            print(f"Route received in load_waypoints: {route}")
        
        # Load the file if the input is a string (file path)
        if isinstance(route, str):
            if print_init_msg:
                print(f"Loading route file from: {route}")  # Debugging
            self.data = np.loadtxt(route)
        else:
            self.data = route  # Assume it's already an array
        
        self.north = []
        self.east = []
        for i in range(0, (int(np.size(self.data) / 2))):
            self.north.append(self.data[i][0])
            self.east.append(self.data[i][1])

    def next_wpt(self, k, N, E):
        ''' Returns the index of the next and current waypoint. The method, if
            called at each time step, will detect when the ship has arrived
            close enough to a waypoint, to proceed ot the next waypoint. Example
            of usage in the method "rudderang_from_route()" from the ShipDyn-class.
        '''
        # print(f"Length of waypoints: {len(self.north)}")
        # print(k)
        if (self.north[k] - N) ** 2 + (
                self.east[k] - E) ** 2 <= self.ra ** 2:  # Check that we are within circle of acceptance
            if len(self.north) > k + 1:  # If number of waypoints are greater than current waypoint index
                return k + 1, k  # Then move on to next waypoint and let current become previous
            else:
                return k, k  # At the end of the route, let the next wpt also be the previous wpt
        else:
            return k, k - 1

    def los_guidance(self, k, x, y):
        ''' Returns the desired heading (i.e. reference signal to
            a ship heading controller). The parameter "k" is the
            index of the next waypoint.
        '''
        dx = self.north[k] - self.north[k - 1]
        dy = self.east[k] - self.east[k - 1]
        alpha_k = math.atan2(dy, dx)
        e_ct = -(x - self.north[k - 1]) * math.sin(alpha_k) + (y - self.east[k - 1]) * math.cos(alpha_k) # Cross-track error
        self.e_ct = np.abs(e_ct)
        if e_ct ** 2 >= self.r ** 2:
            e_ct = 0.99 * self.r
        delta = math.sqrt(self.r ** 2 - e_ct ** 2)
        if abs(self.e_ct_int + e_ct / delta) <= self.integrator_limit:
            self.e_ct_int += e_ct / delta
        chi_r = math.atan(-e_ct / delta - self.e_ct_int*self.ki)
        return alpha_k + chi_r
    
    def reset(self):
        ''' Reset the internal attributes of the Navigation System 
            to its initial values, while also resetting the route 
            container
        '''
        self.route = self.init_route
        self.ra = self.init_ra
        self.r = self.init_r
        self.ki = self.init_ki
        self.e_ct = self.init_e_ct
        self.e_ct_int = self.init_e_ct_int
        self.integrator_limit = self.init_integrator_limit
        
        self.load_waypoints(self.route)
    
# class StaticObstacle:
#     ''' This class is used to define a static obstacle. It can only make
#         circular obstacles. The class is instantiated with the following
#         input paramters:
#         - n_pos: The north coordinate of the center of the obstacle.
#         - e_pos: The east coordinate of the center of the obstacle.
#         - radius: The radius of the obstacle.
#     '''

#     def __init__(self, n_pos, e_pos, radius):
#         self.n = n_pos
#         self.e = e_pos
#         self.r = radius

#     def distance(self, n_ship, e_ship):
#         ''' Returns the distance from a ship with coordinates (north, east)=
#             (n_ship, e_ship), to the closest point on the perifery of the
#             circular obstacle.
#         '''
#         rad_2 = (n_ship - self.n) ** 2 + (e_ship - self.e) ** 2
#         rad = np.sqrt(abs(rad_2))
#         return rad - self.r

#     def plot_obst(self, ax):
#         ''' This method can be used to plot the obstacle in a
#             map-view.
#         '''
#         # ax = plt.gca()
#         ax.add_patch(plt.Circle((self.e, self.n), radius=self.r, fill=True, color='grey'))