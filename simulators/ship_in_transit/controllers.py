""" 
This module provides classes of controllers used to control the ship inside the simulator.
"""


import numpy as np
from typing import List, NamedTuple, Union

from simulators.ship_in_transit.LOS_guidance import NavigationSystem 

###################################################################################################################
#################################### CONFIGURATION FOR PID CONTROLLER #############################################
###################################################################################################################


class ThrottleControllerGains(NamedTuple):
    kp_ship_speed: float
    ki_ship_speed: float
    kp_shaft_speed: float
    ki_shaft_speed: float
    
    
class HeadingControllerGains(NamedTuple):
    kp: float
    kd: float
    ki: float
    
    
class HeadingControllerGains(NamedTuple):
    kp: float
    kd: float
    ki: float

class LosParameters(NamedTuple):
    radius_of_acceptance: float
    lookahead_distance: float
    integral_gain: float
    integrator_windup_limit: float
    

###################################################################################################################
###################################################################################################################


class PiController:
    def __init__(self, kp: float, ki: float, time_step: float, initial_integral_error=0):
        self.kp = kp
        self.ki = ki
        self.error_i = initial_integral_error
        self.time_step = time_step

    def pi_ctrl(self, setpoint, measurement, *args):
        ''' Uses a proportional-integral control law to calculate a control
            output. The optional argument is an 2x1 array and will specify lower
            and upper limit for error integration [lower, upper]
        '''
        error = setpoint - measurement
        error_i = self.error_i + error * self.time_step
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.error_i = error_i
        return error * self.kp + error_i * self.ki

    @staticmethod
    def sat(val, low, hi):
        ''' Saturate the input val such that it remains
        between "low" and "hi"
        '''
        return max(low, min(val, hi))


class PidController:
    def __init__(self, kp: float, kd: float, ki: float, time_step: float):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.error_i = 0
        self.prev_error = 0
        self.time_step = time_step

    def pid_ctrl(self, setpoint, measurement, *args):
        ''' Uses a proportional-derivative-integral control law to calculate
            a control output. The optional argument is a 2x1 array and will
            specify lower and upper [lower, upper] limit for error integration
        '''
        error = setpoint - measurement
        d_error = (error - self.prev_error) / self.time_step
        error_i = self.error_i + error * self.time_step
        if args:
            error_i = self.sat(error_i, args[0], args[1])
        self.prev_error = error
        self.error_i = error_i
        return error * self.kp + d_error * self.kd + error_i * self.ki

    @staticmethod
    def sat(val, low, hi):
        ''' Saturate the input val such that it remains
        between "low" and "hi"
        '''
        return max(low, min(val, hi))
    
    
###################################################################################################################
################################## DESCENDANT CLASS FROM THE BASE CONTROLLER ######################################
###################################################################################################################


class EngineThrottleFromSpeedSetPoint:
    """
    Calculates throttle setpoint for power generation based on the ship´s speed, the propeller shaft speed
    and the desires ship speed.
    """

    def __init__(
            self,
            gains: ThrottleControllerGains,
            max_shaft_speed: float,
            time_step: float,
            initial_shaft_speed_integral_error: float
    ):
        # Initial internal attribute
        self.init_ship_speed_controller = PiController(
            kp=gains.kp_ship_speed, ki=gains.ki_ship_speed, time_step=time_step
        )
        self.init_shaft_speed_controller = PiController(
            kp=gains.kp_shaft_speed,
            ki=gains.ki_shaft_speed,
            time_step=time_step,
            initial_integral_error=initial_shaft_speed_integral_error
        )
        self.init_max_shaft_speed = max_shaft_speed
        
        # Internal attribute
        self.ship_speed_controller = self.init_ship_speed_controller
        self.shaft_speed_controller = self.init_shaft_speed_controller
        self.max_shaft_speed = self.init_max_shaft_speed

    def throttle(self, speed_set_point, measured_speed, measured_shaft_speed):
        desired_shaft_speed = self.ship_speed_controller.pi_ctrl(setpoint=speed_set_point, measurement=measured_speed)
        # desired_shaft_speed = self.ship_speed_controller.sat(val=desired_shaft_speed, low=0, hi=self.max_shaft_speed)
        throttle = self.shaft_speed_controller.pi_ctrl(setpoint=desired_shaft_speed, measurement=measured_shaft_speed)
        # return self.shaft_speed_controller.sat(val=throttle, low=0, hi=1.1)
        return throttle
    
    def reset(self):
        ''' Reset the internal attributes of the throttle controller
            its initial values
        '''
        self.ship_speed_controller = self.init_ship_speed_controller
        self.shaft_speed_controller = self.init_shaft_speed_controller
        self.max_shaft_speed = self.init_max_shaft_speed


class ThrottleFromSpeedSetPointSimplifiedPropulsion:
    """
    Calculates throttle setpoint for power generation based on the ship´s speed, the propeller shaft speed
    and the desires ship speed.
    """

    def __init__(
            self,
            kp: float,
            ki: float,
            time_step: float,
    ):
        self.ship_speed_controller = PiController(
            kp=kp, ki=ki, time_step=time_step
        )

    def throttle(self, speed_set_point, measured_speed):
        throttle = self.ship_speed_controller.pi_ctrl(setpoint=speed_set_point, measurement=measured_speed)
        return self.ship_speed_controller.sat(val=throttle, low=0, hi=1.1)
    
    
class HeadingByReferenceController:
    def __init__(self, gains: HeadingControllerGains, time_step, max_rudder_angle):
        self.ship_heading_controller = PidController(kp=gains.kp, kd=gains.kd, ki=gains.ki, time_step=time_step)
        self.max_rudder_angle = max_rudder_angle

    def rudder_angle_from_heading_setpoint(self, heading_ref: float, measured_heading: float):
        ''' This method finds a suitable rudder angle for the ship to
            sail with the heading specified by "heading_ref" by using
            PID-controller. The rudder angle is saturated according to
            |self.rudder_ang_max|. The mathod should be called from within
            simulation loop if the user want the ship to follow a specified
            heading reference signal.
        '''
        rudder_angle = -self.ship_heading_controller.pid_ctrl(setpoint=heading_ref, measurement=measured_heading)
        return self.ship_heading_controller.sat(rudder_angle, -self.max_rudder_angle, self.max_rudder_angle)


class HeadingByRouteController:
    def __init__(
            self, route_name,
            heading_controller_gains: HeadingControllerGains,
            los_parameters: LosParameters,
            time_step: float,
            max_rudder_angle: float,
    ):
        self.heading_controller = HeadingByReferenceController(
            gains=heading_controller_gains, time_step=time_step, max_rudder_angle=max_rudder_angle
        )
        self.navigate = NavigationSystem(
            route=route_name,
            radius_of_acceptance=los_parameters.radius_of_acceptance,
            lookahead_distance=los_parameters.lookahead_distance,
            integral_gain=los_parameters.integral_gain,
            integrator_windup_limit=los_parameters.integrator_windup_limit
        )
        ## Initial internal attributes
        self.init_next_wpt = 1
        self.init_prev_wpt = 0
        
        self.init_heading_ref = 0
        self.init_heading_mea = 0
        
        ## Internal attributes
        self.next_wpt = self.init_next_wpt
        self.prev_wpt = self.init_prev_wpt
        
        self.heading_ref = self.init_heading_ref
        self.heading_mea = self.init_heading_mea

    def rudder_angle_from_route(self, north_position, east_position, heading):
        ''' This method finds a suitable rudder angle for the ship to follow
            a predefined route specified in the "navigate"-instantiation of the
            "NavigationSystem"-class.
        '''
        self.next_wpt, self.prev_wpt = self.navigate.next_wpt(self.next_wpt, north_position, east_position)
        self.heading_ref = self.navigate.los_guidance(self.next_wpt, north_position, east_position)
        self.heading_mea = heading
        return self.heading_controller.rudder_angle_from_heading_setpoint(heading_ref=self.heading_ref, measured_heading=heading)
    
    ## ADDITIONAL ##
    def get_heading_error(self):
        return np.abs(self.heading_mea - self.heading_ref)
    
    def get_error_cross_track(self):
        return self.navigate.e_ct
    
    def reset(self):
        # Internal attributes reset
        self.next_wpt = self.init_next_wpt
        self.prev_wpt = self.init_prev_wpt
        
        self.heading_ref = self.init_heading_ref
        self.heading_mea = self.init_heading_mea
        
        # Navigation system attributes
        self.navigate.reset()
    

class HeadingBySampledRouteController:
    def __init__(
            self, route_name,
            heading_controller_gains: HeadingControllerGains,
            los_parameters: LosParameters,
            time_step: float,
            max_rudder_angle: float,
            num_of_samplings: int,
    ):
        
        self.heading_controller = HeadingByReferenceController(
            gains=heading_controller_gains, time_step=time_step, max_rudder_angle=max_rudder_angle
        )
        self.navigate = NavigationSystem(
            route=route_name,
            radius_of_acceptance=los_parameters.radius_of_acceptance,
            lookahead_distance=los_parameters.lookahead_distance,
            integral_gain=los_parameters.integral_gain,
            integrator_windup_limit=los_parameters.integrator_windup_limit,
        )
        
        ## Initial internal attributes
        self.init_next_wpt = 1
        self.init_prev_wpt = 0
        
        self.init_heading_ref = 0
        self.init_heading_mea = 0
        
        self.init_num_of_samplings = num_of_samplings
        self.init_sampling_counters = 0
        self.init_distance_points_north = self.navigate.north[1] - self.navigate.north[0]
        self.init_distance_points_east = self.navigate.east[1] - self.navigate.east[0]
        
        ## Internal attributes
        self.next_wpt = self.init_next_wpt
        self.prev_wpt = self.init_prev_wpt
        
        self.heading_ref = self.init_heading_ref
        self.heading_mea = self.init_heading_mea
        
        self.num_of_samplings = self.init_num_of_samplings
        self.sampling_counters = self.init_sampling_counters
        self.distance_points_north = self.init_distance_points_north
        self.distance_points_east = self.init_distance_points_east
        
    def update_route(self, route_shifts):
        # Get the route shifts and update the new route
        route_shift_n, route_shift_e = route_shifts
        
        self.navigate.north.insert(-1, float(route_shift_n))
        self.navigate.east.insert(-1, float(route_shift_e))
        

    def rudder_angle_from_sampled_route(self, north_position, east_position, heading):
        ''' This method finds a suitable rudder angle for the ship to follow
            a predefined route specified in the "navigate"-instantiation of the
            "NavigationSystem"-class.
        '''
        self.next_wpt, self.prev_wpt = self.navigate.next_wpt(self.next_wpt, north_position, east_position)
        self.heading_ref = self.navigate.los_guidance(self.next_wpt, north_position, east_position)
        self.heading_mea = heading
        return self.heading_controller.rudder_angle_from_heading_setpoint(heading_ref=self.heading_ref, measured_heading=heading)
    
    ## ADDITIONAL ##
    def if_reach_radius_of_acceptance(self, n_pos, e_pos, r_o_a):
        # print(n_pos, e_pos)
        # print(self.next_wpt)
        # print(self.navigate.north, self.navigate.east)
        # print(self.navigate.north[self.next_wpt], self.navigate.east[self.next_wpt])
        dist_to_next_route = np.sqrt((n_pos - self.navigate.north[self.next_wpt])**2 + (e_pos - self.navigate.east[self.next_wpt])**2)
        # print(dist_to_next_route, r_o_a)
        reach_radius_of_acceptance =  dist_to_next_route < r_o_a
        return reach_radius_of_acceptance 
    
    def get_heading_error(self):
        return np.abs(self.heading_mea - self.heading_ref)
    
    def get_cross_track_error(self):
        return self.navigate.e_ct
    
    def reset(self):
        ''' Reset the internal attributes of the heading controller
            and the navigation system to its initial values
        '''
        # Internal attributes reset
        self.next_wpt = self.init_next_wpt
        self.prev_wpt = self.init_prev_wpt
        
        self.heading_ref = self.init_heading_ref
        self.heading_mea = self.init_heading_mea
        
        self.num_of_samplings = self.init_num_of_samplings
        self.sampling_counters = self.init_sampling_counters
        self.distance_points_north = self.init_distance_points_north
        self.distance_points_east = self.init_distance_points_east
        
        # Navigation system attributes
        self.navigate.reset()
        
        