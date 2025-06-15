""" 
This modules provide classes for the reinforcement learning environment based on the Ship-Transit Simulator
"""

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from collections import defaultdict
import numpy as np
import torch

from simulator.ship_model import ShipModel, ShipModelAST
from simulator.controllers import EngineThrottleFromSpeedSetPoint, HeadingByRouteController, HeadingBySampledRouteController
from simulator.obstacle import StaticObstacle, PolygonObstacle

from dataclasses import dataclass, field
from typing import Union, List

import copy
# from ast_sac.reward_function import reward_function

@dataclass
class ShipAssets:
    ship_model: Union[ShipModel, ShipModelAST]
    throttle_controller: EngineThrottleFromSpeedSetPoint
    auto_pilot: Union[HeadingBySampledRouteController, HeadingByRouteController]
    desired_forward_speed: float
    integrator_term: List[float]
    time_list: List[float]
    type_tag: str
    stop_flag: bool
    init_copy: 'ShipAssets' = field(default=None, repr=False, compare=False)

class MultiShipRLEnv(Env):
    """
    This class is the main class for the reinforcement learning environment based on the Ship-Transit Simulator suited for 
    two ship actor
    """
    def __init__(self, 
                 assets:List[ShipAssets],
                 map: PolygonObstacle,
                 ship_draw:bool,
                 time_since_last_ship_drawing:float,
                 args):
        super().__init__()

        #  For test ship
        self.test, self.obs = assets
        
        ## Unpack assets [test, obs]
        self.assets = [self.test, self.obs]
        
        # Store init values
        for i, asset in enumerate(self.assets):
            asset.init_copy=copy.deepcopy(asset)
        
        # Ship drawing configuration
        self.ship_draw = ship_draw
        self.time_since_last_ship_drawing = time_since_last_ship_drawing
        
        # Define observation space 
        # [test_n_pos, test_e_pos, test_headings, 
        #   test_shaft_speed, test_los_e_ct, test_power_load, \
        #   obs_n_pos, obs_e_pos, obs_headings, \
        #   obs_los_e_ct] (10 states)
        self.observation_space = Box(
            low = np.array([0, 0, -np.pi,
                            -3000, 0, 0, 
                            0, 0, -np.pi,
                            0], dtype=np.float32),
            high = np.array([10000, 20000, np.pi, 
                             3000, 1000, 2000,
                             10000, 20000, np.pi,
                             1000], dtype=np.float32),
        )
        
        # Define action space [route_point_shift, desired_forward_speed] # FOR LATER
        # Define action space [route_point_shift] 
        self.action_space = Box(
            low = np.array([-np.pi/6], dtype=np.float32),
            high = np.array([np.pi/6], dtype=np.float32),
        )
        
        # Define initial state
        self.initial_state = np.array([self.test.ship_model.north, self.test.ship_model.east, self.test.ship_model.yaw_angle,
                                       0.0, 0.0, 0.0,
                                       self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle,
                                       0.0], dtype=np.float32)
        self.state = self.initial_state
        
        # Container for the next state
        # [test_n_pos, test_e_pos, test_headings, test_forward speed, 
        #   test_shaft_speed, test_los_e_ct, test_power_load, \
        #   obs_n_pos, obs_e_pos, obs_headings, obs_forward_speed] (11 states)
        self.initial_next_states = np.zeros(self.state.shape[0], dtype=np.float32)
        self.next_states = self.initial_next_states
        
        # Store the map class as attribute
        self.map = map 
        
        # Store args as attribute
        self.args = args
        
        # Simulation time and travel distance counter
        self.eps_simu_time = 0
        self.eps_distance_travelled = 0
        self.sampling_distance_travelled = 0
        
        # Previously sampled route coordinate
        self.prev_route_coordinate = None
        
        # Initialize Reward Function Parameters
        self.reward_function_params()
        

    def reward_function_params(self):
        # Reward function parameters for test and obstacle ship
        self.e_tolerance = 1000
        
        # Reward function parameters for obstacle ship
        self.AB_distance_n = self.obs.auto_pilot.navigate.north[-1] - self.obs.auto_pilot.navigate.north[0]
        self.AB_distance_e = self.obs.auto_pilot.navigate.east[-1] - self.obs.auto_pilot.navigate.east[0]
        self.AB_distance = np.sqrt(self.AB_distance_n ** 2 + self.AB_distance_e ** 2)
        self.AB_segment_length = self.AB_distance/self.args.sampling_frequency
        self.AB_alpha = np.arctan2(self.AB_distance_e, self.AB_distance_n)
        self.AB_beta = np.pi/2 - self.AB_alpha 
        
        # Navigation failure coefficient
        self.theta = self.args.theta
        
        # Reward container
        self.reward_results = {"test_ship":{"reward_e_ct": [],
                                            "reward_near_col": [],
                                            "total_non_terminal": [],},
                                "obs_ship":{"reward_base": [],
                                            "reward_e_ct": [],
                                            "reward_near_col": [],
                                            "total_non_terminal": [],},
                                 "shared": {"total_non_terminal": [],}
                            }

        
    
    def reset(self):
        ''' Reset the ship environment for each model (or both)
            Args:
            'test_ship', 'obs_ship', 'all'
        '''
        # Reset the assets
        for i, ship in enumerate(self.assets):
            # Call upon the copied initial values
            init = ship.init_copy
            
            #  Reset the ship simulator
            ship.ship_model.reset()
            
            # Reset the ship throttle controller
            ship.throttle_controller.reset() 
            
            # Reset the autopilot controlller
            ship.auto_pilot.reset()
            
            # Reset parameters and lists
            ship.desired_forward_speed = init.desired_forward_speed
            ship.integrator_term = copy.deepcopy(init.integrator_term)
            ship.time_list = copy.deepcopy(init.time_list)
            
            # Reset the stop flag
            ship.stop_flag = False
        
        # Reset the simulation time and travel distance counter
        self.simu_time = 0
        self.eps_distance_travelled = 0
        self.sampling_distance_travelled = 0
        
        # Reset the changing states into its initial state
        self.state = self.initial_state
        
        # Reset the previous sampled route coordinate
        self.prev_route_coordinate = None
        
        # Initialize Reward Function Parameters
        self.reward_function_params()
        
        return self.state
    
    def init_step(self):
        ''' The initial step to place the ship and the controller 
            to work inside the digital simulation (WITHOUT STORING)
        '''
        # For all assets
        for ship in self.assets:
            # Measure ship position and speed
            north_position = ship.ship_model.north
            east_position = ship.ship_model.east
            heading = ship.ship_model.yaw_angle
            forward_speed = ship.ship_model.forward_speed
        
            # Find appropriate rudder angle and engine throttle
            rudder_angle = ship.auto_pilot.rudder_angle_from_sampled_route(
                north_position=north_position,
                east_position=east_position,
                heading=heading
            )
        
            throttle = ship.throttle_controller.throttle(
                speed_set_point = ship.desired_forward_speed,
                measured_speed = forward_speed,
                measured_shaft_speed = forward_speed
            )
            
            # Step
            ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
            ship.ship_model.integrate_differentials()
    
    def test_step(self):
        ''' The method is used for stepping up the simulator for all the ship under test
        '''          
        # Measure ship position and speed
        north_position = self.test.ship_model.north
        east_position = self.test.ship_model.east
        heading = self.test.ship_model.yaw_angle
        forward_speed = self.test.ship_model.forward_speed
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.test.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading
        )
        
        throttle = self.test.throttle_controller.throttle(
            speed_set_point = self.test.desired_forward_speed,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )
            
        ## COLLISION AVOIDANCE
        collision_risk = self.is_collision_imminent(self.next_states[0:3], self.next_states[7:10])
                
        if collision_risk:
            # Reduce throttle
            throttle *= 0.5
            throttle = np.clip(throttle, 0.0, 1.1)

            # Add a small rudder bias to steer away (rudder angle in radians)
            rudder_angle += np.deg2rad(3)                    
            rudder_angle = np.clip(rudder_angle, -self.test.auto_pilot.heading_controller.max_rudder_angle, self.test.auto_pilot.heading_controller.max_rudder_angle)

            # Optional print for debugging
            # print("Collision Avoidance Activated")
        
        # Update and integrate differential equations for current time step
        self.test.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.test.auto_pilot.get_cross_track_error(),
                                                   self.test.auto_pilot.get_heading_error())
        self.test.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.test.ship_model.integrate_differentials()
        
        self.test.integrator_term.append(self.test.auto_pilot.navigate.e_ct_int)
        self.test.time_list.append(self.test.ship_model.int.time)
        
        # Compute reward
        pos = [self.test.ship_model.north, self.test.ship_model.east, self.test.ship_model.yaw_angle]
        measured_shaft_rpm = self.test.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
        los_ct_error = self.test.ship_model.simulation_results['cross track error [m]'][-1]
        power_load = self.test.ship_model.simulation_results['power me [kw]'][-1]
        
        # Set the next state, then reset the next_state container to zero
        next_state = [self.ensure_scalar(pos[0]),
                      self.ensure_scalar(pos[1]), 
                      self.ensure_scalar(pos[2]),
                      self.ensure_scalar(forward_speed),
                      self.ensure_scalar(measured_shaft_rpm),
                      self.ensure_scalar(los_ct_error),
                      self.ensure_scalar(power_load)]
        
        # Step up the simulator
        self.test.ship_model.int.next_time()
        
        return next_state
    
    def obs_step(self, 
                 converted_action, 
                 SAC_update,
                 init):
        ''' The method is used for stepping up the simulator for teh obstacle ship
        '''          
        if self.obs.stop_flag:
            # Ship reached endpoint, keep time aligned but don't integrate
            self.obs.ship_model.store_last_simulation_data()
            self.obs.ship_model.int.next_time()  # still progress time
            
            # Compute reward
            pos = [self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle]
            measured_shaft_rpm = self.obs.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
            los_ct_error = self.obs.ship_model.simulation_results['cross track error [m]'][-1]
            power_load = self.obs.ship_model.simulation_results['power me [kw]'][-1]
        
            # Store integrator term and timestamp
            self.obs.integrator_term.append(self.obs.auto_pilot.navigate.e_ct_int)
            self.obs.time_list.append(self.obs.ship_model.int.time)
        
            # Step up the simulator
            self.obs.ship_model.int.next_time()
            
            # Stop the ship
            forward_speed = self.obs.ship_model.forward_speed
            forward_speed = 0
        
            # Set the next state using the last position of the ship
            next_state = [self.ensure_scalar(pos[0]),
                          self.ensure_scalar(pos[1]), 
                          self.ensure_scalar(pos[2]),
                          self.ensure_scalar(forward_speed),
                          self.ensure_scalar(measured_shaft_rpm),
                          self.ensure_scalar(los_ct_error),
                          self.ensure_scalar(power_load)]
        
            return next_state
            
        if SAC_update:
            route_coord_n, route_coord_e = converted_action
            
            # Update route_point based on the action
            route_coordinate = route_coord_n, route_coord_e
            self.obs.auto_pilot.update_route(route_coordinate)
            
            # Update desired_forward_speed based on the action
            # self.desired_forward_speed = desired_forward_speed
            
            # Store the sampled route coordinate to the holder variable
            self.prev_route_coordinate = route_coordinate
                
            # Reset the travel distance counter for sampling
            self.sampling_distance_travelled = 0
        
        # If it is not the time to use action as simulation input, use saved route coordinate
        else:
            route_coordinate = self.prev_route_coordinate
        
        
        # Measure ship position and speed
        north_position = self.obs.ship_model.north
        east_position = self.obs.ship_model.east
        heading = self.obs.ship_model.yaw_angle
        forward_speed = self.obs.ship_model.forward_speed
        
        # Find appropriate rudder angle and engine throttle
        rudder_angle = self.obs.auto_pilot.rudder_angle_from_sampled_route(
            north_position=north_position,
            east_position=east_position,
            heading=heading
        )
        
        throttle = self.obs.throttle_controller.throttle(
            speed_set_point = self.obs.desired_forward_speed,
            measured_speed = forward_speed,
            measured_shaft_speed = forward_speed,
        )
        
        # Update and integrate differential equations for current time step
        self.obs.ship_model.store_simulation_data(throttle, 
                                                   rudder_angle,
                                                   self.obs.auto_pilot.get_cross_track_error(),
                                                   self.obs.auto_pilot.get_heading_error())
        self.obs.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
        self.obs.ship_model.integrate_differentials()
        
        self.obs.integrator_term.append(self.obs.auto_pilot.navigate.e_ct_int)
        self.obs.time_list.append(self.obs.ship_model.int.time)
        
        # Compute reward
        pos = [self.obs.ship_model.north, self.obs.ship_model.east, self.obs.ship_model.yaw_angle]
        measured_shaft_rpm = self.obs.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
        los_ct_error = self.obs.ship_model.simulation_results['cross track error [m]'][-1]
        power_load = self.obs.ship_model.simulation_results['power me [kw]'][-1]
        
        # Get the next state
        next_state = [self.ensure_scalar(pos[0]),
                      self.ensure_scalar(pos[1]), 
                      self.ensure_scalar(pos[2]),
                      self.ensure_scalar(forward_speed),
                      self.ensure_scalar(measured_shaft_rpm),
                      self.ensure_scalar(los_ct_error),
                      self.ensure_scalar(power_load)]

        # Compute travelled distance for the obstacle ship
        if init == False:
            dist_trav_north = self.obs.ship_model.simulation_results['north position [m]'][-1] - self.obs.ship_model.simulation_results['north position [m]'][-2]
            dist_trav_east = self.obs.ship_model.simulation_results['east position [m]'][-1] - self.obs.ship_model.simulation_results['east position [m]'][-2]
            self.eps_distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
            self.sampling_distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
        
        # Step up the simulator
        self.obs.ship_model.int.next_time()
        
        return next_state
    
    def step(self, 
             converted_action, 
             SAC_update,
             init):
        ''' The method is used for stepping up the simulator for all the reinforcement
            learning assets
        '''
        # Do test ship step
        test_next_state = self.test_step()
        
        # Do obstacle ship step
        obs_next_state = self.obs_step(converted_action, SAC_update, init)
        
        # Apply ship drawing (set as optional function) after stepping
        if self.ship_draw:
            if self.time_since_last_ship_drawing > 30:
                self.test.ship_model.ship_snap_shot()
                self.obs.ship_model.ship_snap_shot()
                self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
            self.time_since_last_ship_drawing += self.test.ship_model.int.dt
            
        # Gather the necessary state for the SAC
        next_state = [
                      test_next_state[0], # test_n_pos
                      test_next_state[1], # test_e_pos
                      test_next_state[2], # test_heading
                      test_next_state[4], # test_measured_shaft_rpm
                      test_next_state[5], # test_e_ct
                      test_next_state[6], # test_power_load
                      obs_next_state[0],  # obs_n_pos
                      obs_next_state[1],  # obs_e_pos
                      obs_next_state[2],  # obs_heading
                      obs_next_state[5],  # obs_e_ct
                      ] 
        
        # Compute the 
        reward, done, status = self.reward_function(next_state, converted_action)
        
        return next_state, reward, done, status
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility"""
        self.np_random, seed = seeding.np_random(seed)
        
    # Make sure all values are scalars
    def ensure_scalar(self, x):
        return float(x[0]) if isinstance(x, (np.ndarray, list)) else float(x)
        
###################################################################################################################
############################################### FAILURE MODES #####################################################
###################################################################################################################
    def is_collision_imminent(self, test_pos, obs_pos, safety_distance=500):
        n_test, e_test, _ = test_pos
        n_obs, e_obs, _ = obs_pos
        dist_squared = (n_test - n_obs)**2 + (e_test - e_obs)**2
        return dist_squared < safety_distance**2 
     
        
    def is_pos_outside_horizon(self, pos, ship_length):
        ''' Checks if the ship positions are outside the map horizon. 
            Map horizons are determined by the edge point of the determined route point. 
            Allowed additional margin are default 100 m for North and East boundaries.
            Only works with start to end route points method (Two initial points).
        '''
        # Unpack ship position
        n_pos, e_pos, _ = pos
        
        # Get the map boundaries
        min_north = self.map.min_north
        min_east = self.map.min_east
        max_north = self.map.max_north
        max_east = self.map.max_east
        
        # Get the obstacle margin due to all assets ship length
        margin = ship_length/2
            
        # min_bound and max_bound
        n_route_bound = [min_north + margin , max_north - margin]
        e_route_bound = [min_east + margin, max_east - margin]
        
        # Check if position is outside bound
        outside_n = n_pos < n_route_bound[0] or n_pos > n_route_bound[1]
        outside_e = e_pos < e_route_bound[0] or e_pos > e_route_bound[1]
        
        is_outside = outside_n or outside_e
        
        return is_outside
    
    def is_pos_inside_obstacles(self, pos, ship_length):
        ''' Checks if the tagged position is inside any obstacle
        '''
        # Unpack ship position
        n_pos, e_pos, _ = pos
        
        # Get the obstacle margin due to all assets ship length
        # Margin is defined as a square patch enveloping the ships
        margin = ship_length/2
        
        # Get the max reach and min reach of the ship
        min_north = n_pos - margin
        min_east = e_pos - margin
        max_north = n_pos + margin
        max_east = e_pos + margin
        
        # All patch's hard point
        hard_points = [(min_north, min_east), (min_north, max_east), (max_north, min_east), (max_north, max_east)]
        
        is_inside = False
        
        for hard_point in hard_points:
            if self.map.if_pos_inside_obstacles(hard_point[0], hard_point[1]):
                is_inside =  True
      
        return is_inside
    
    def is_route_outside_horizon(self, route):
        ''' Checks if the ship positions are outside the map horizon. 
            Map horizons are determined by the edge point of the determined route point. 
            Allowed additional margin are default 100 m for North and East boundaries.
            Only works with start to end route points method (Two initial points).
        '''
        # Unpack ship position
        n_route, e_route = route
        
        # Get the map boundaries
        min_north = self.map.min_north
        min_east = self.map.min_east
        max_north = self.map.max_north
        max_east = self.map.max_east
            
        # min_bound and max_bound
        n_route_bound = [min_north , max_north]
        e_route_bound = [min_east, max_east]
        
        # Check if position is outside bound
        outside_n = n_route < n_route_bound[0] or n_route > n_route_bound[1]
        outside_e = e_route < e_route_bound[0] or e_route > e_route_bound[1]
        
        is_outside = outside_n or outside_e
        
        return is_outside
    
    def is_route_inside_obstacles(self, route):
        ''' Checks if the tagged position is inside any obstacle
        '''
        is_inside = False
        
        if self.map.if_pos_inside_obstacles(route[0], route[1]):
            is_inside =  True
      
        return is_inside
    
    def is_mechanical_failure(self, measured_shaft_rpm):
        ## Check this website:
        ## https://spicerparts.com/calculators/critical-speed-rpm-calculator
        shaft_rpm_max = 2000 # rpm
        return np.abs(measured_shaft_rpm) > shaft_rpm_max

    def is_test_ship_navigation_failure(self, e_ct):
        ## Ship deviates off the course beyond tolerance defined by these two conditions
        condition_1 = np.abs(e_ct) > self.e_tolerance
        
        return condition_1
    
    def is_obs_ship_navigation_failure(self, e_ct):
        ## Ship deviates off the course beyond tolerance defined by these two conditions
        condition_1 = np.abs(e_ct) > self.e_tolerance
        condition_2 = self.sampling_distance_travelled > self.AB_segment_length * self.theta
        
        # print('condition 1: ', condition_1)
        # # print('condition 2: ', condition_2)
        
        # print(self.sampling_distance_travelled)
        
        return condition_1 or condition_2

    def is_blackout_failure(self,
                            power_load, 
                            available_power_load):
        ## Diesel engine overloaded
        return power_load > available_power_load
    
    def is_ship_collision(self, test_pos, obs_pos):
        ''' If test ship and obstacle ship distance is below some threshold, categorized it as collision
        '''
        # Unpack ship position
        n_test, e_test, _ = test_pos
        n_obs, e_obs , _ = obs_pos
        
        # Set minimum ship distacne
        minimum_ship_distance = 50 # arbitrary number
        
        # Compute the ship distance
        ship_distance =  (n_test - n_obs)**2 + (e_test - e_obs)**2
        
        # Collision logic
        is_collide = False
        
        if ship_distance < minimum_ship_distance ** 2:
            is_collide = True
        
        return is_collide
    
    #### EXPERIMENTAL #### NOT USED FOR NOW ####
    # def is_too_slow(self, recorded_time):
    #     ''' Expected time = Scaling Factor * Start-to-end point distances / Expected forward speed
            
    #         Expected time is defined as the time needed to travel form start to end
    #         point with the expected forward speed.
            
    #         If the ship couldn't travelled enough distances for the sampling within the
    #         expected time, it is deemed as a termination and give huge negative rewards.
            
    #         The reason is this occurence most likely happened because the sampled speed
    #         is too slow 
    #     '''
    #     scale_factor = 2.5
        
    #     time_expected = scale_factor * self.AB_distance / self.expected_forward_speed
        
    #     return recorded_time > time_expected
    
###################################################################################################################
############################################## REWARD FUNCTION ####################################################
###################################################################################################################

    def test_ship_non_terminal_state_reward(self, 
                                            test_pos,
                                            test_los_ct_error):
        ''' Reward per simulation time step should be in order 10**0
            As it will be accumulated over time. Negative reward proportional
            to positive reward shall be added for each simulation time step as
            well.
            
            For Ship Under Test
        '''
        ## Unpack test_ship
        n_test, e_test, _ = test_pos
        
        # ## Base stepping reward
        # reward_base = 0.0
        
        ## Cross-track error reward        
        # Normalized cross_track error by the tolerance
        # Big cross track error is preferable. But when navigation loss happened give more reward
        reward_e_ct = (np.abs(test_los_ct_error) / self.e_tolerance)
        
        # ## Distance to end point
        # # Get the relative distance between the ship and the end point
        # n_route_end = self.test.auto_pilot.navigate.north[-1]
        # e_route_end = self.test.auto_pilot.navigate.east[-1]
        # distance_to_reward = np.sqrt((n_test - n_route_end)**2 + (e_test - e_route_end)**2)
        # # Closer to end point is more rewarding, normalized by the maximum east position
        # reward_to_distance = (1 - distance_to_reward/self.map.max_east)/100
        
        # Miss distance from collision reward
        # Normalized by the maximum north position
        # Closer to obstacle is better
        reward_near_col = (1 - self.map.obstacles_distance(n_test, e_test) / self.map.max_north)/100
        
        reward_total = reward_e_ct + reward_near_col
        
        return reward_total, reward_e_ct, reward_near_col
    
    def obs_ship_non_terminal_state_reward(self, 
                                           obs_pos,
                                           obs_los_ct_error):
        ''' Reward per simulation time step should be in order 10**0
            As it will be accumulated over time. Negative reward proportional
            to positive reward shall be added for each simulation time step as
            well.
            
            For Obstacle Ship
        '''
        # If the obstacle ship simulation ended, stop giving non terminal reward
        if self.obs.stop_flag is False:
            ## Unpack test_ship
            n_obs, e_obs, _ = obs_pos
        
            ## Base stepping reward, we want the obstacle ship to keep sailing
            reward_base = 0.1
        
            ## Cross-track error reward        
            # Normalized cross_track error by the tolerance
            # Huge cross track error is not preferable. But when navigation loss happened give more reward
            reward_e_ct = -(np.abs(obs_los_ct_error) / self.e_tolerance)/100 
        
            # ## Distance to end point
            # # Get the relative distance between the ship and the end point
            # n_route_end = self.obs.auto_pilot.navigate.north[-1]
            # e_route_end = self.obs.auto_pilot.navigate.east[-1]
            # distance_to_reward = np.sqrt((n_obs - n_route_end)**2 + (e_obs - e_route_end)**2)
            # # Closer to end point is more rewarding, normalized by the maximum east position
            # reward_to_distance = (1 - distance_to_reward/self.map.max_east)/100
        
            # Miss distance from collision reward
            # Normalized by the maximum north position
            # Closer to obstacle is worse
            reward_near_col = -(1 - self.map.obstacles_distance(n_obs, e_obs) / self.map.max_north)/100
        
            reward_total = reward_base  + reward_e_ct + reward_near_col
        
        else:
            reward_total = 0
            reward_base = 0
            reward_e_ct = 0
            reward_near_col = 0
        
        return reward_total, reward_base, reward_e_ct, reward_near_col
    
    def shared_non_terminal_state_reward(self,
                                         test_pos,
                                         obs_pos):
        ''' For computing reward function based on the test and obstacle ship distance
        '''
        # If the obstacle ship simulation ended, stop giving non terminal reward
        if self.obs.stop_flag is False:
        
            # Unpack ship position
            n_test, e_test, _ = test_pos
            n_obs, e_obs , _ = obs_pos
        
            # Compute the ship distance
            ship_distance =  np.sqrt((n_test - n_obs)**2 + (e_test - e_obs)**2)
        
            reward = (1 - ship_distance/self.map.max_north)/1000
        else:
            reward = 0
        
        return reward
#################################################################################################################################
#################################################################################################################################
    def test_ship_terminal_state_reward(self,
                                        test_pos,
                                        test_los_ct_error,
                                        test_power_load,
                                        test_available_power_load,
                                        test_measured_shaft_rpm):
        ## Initial value
        reward_terminal = 0
        status =" "
        done = False
        
        n_test, e_test, _ = test_pos
        
        ####################################### TERMINATION CONDITION #######################################
        ## Shutdown simulator because SHIP UNDER TEST REACHES ENDPOINT
        # Get the relative distance between the ship and the end point
        n_route_end = self.test.auto_pilot.navigate.north[-1]
        e_route_end = self.test.auto_pilot.navigate.east[-1]
        relative_dist = np.sqrt((n_test - n_route_end)**2 + (e_test - e_route_end)**2)
        # Check if the ship arrive at the end point
        arrival_radius = 200 # Arrival radius zone
        if relative_dist <= arrival_radius:
            if not self.test.stop_flag:
                reward_terminal += 0
            self.test.stop_flag = True
            done = True
            status += "|Test ship reaches endpoint|"
            
        ## Shutdown simulator because SHIP UNDER TEST GOES OUTSIDE MAP HORIZON
        if self.is_pos_outside_horizon(test_pos, self.test.ship_model.l_ship):
            if not self.test.stop_flag:
                reward_terminal += 0
            self.test.stop_flag = True
            done = True
            status += "|Test ship hits map horizon|"
        #####################################################################################################
        
        ########################################### FAILURE MODES ###########################################                   
        ## Reward for SHIP UNDER TEST HITS THE TERRAIN
        # We want the test ship to hit the obstacle
        if self.is_pos_inside_obstacles(test_pos, self.test.ship_model.l_ship): # When using polygon obstacle
            if not self.test.stop_flag:
                reward_terminal += 1000
            self.test.stop_flag = True
            done = True
            status += "|Test ship collides with the terrain|"

        ## Reward for SHIP UNDER TEST UNDERGOES MECHANICAL FAILURE
        if self.is_mechanical_failure(test_measured_shaft_rpm):
            if not self.test.stop_flag:
                reward_terminal += 1000
            self.test.stop_flag = True
            done = True
            status += "|Test ship mechanical failure|"
        
        ## Reward for SHIP UNDER TEST EXPERIENCES NAVIGATIONAL FAILURE
        if self.is_test_ship_navigation_failure(test_los_ct_error):
            if not self.test.stop_flag:
                reward_terminal += 1000
            self.test.stop_flag = True
            done = True
            status += "|Test ship navigation failure|"
        
        ## Reward for SHIP UNDER TEST SUFFERS BLACKOUT FAILURE  
        if self.is_blackout_failure(test_power_load, test_available_power_load):
            if not self.test.stop_flag:
                reward_terminal += 1000
            self.test.stop_flag = True
            done = True
            status += "|Test ship blackout failure|"
        #####################################################################################################
        
        if done == False:
            status += "|Test ship not in terminal state|"
        
        return reward_terminal, done, status
    
    def obs_ship_terminal_state_reward(self,
                                        obs_pos,
                                        obs_route_coordinate,
                                        obs_los_ct_error):
        ## Initial value
        reward_terminal = 0
        status = " "
        done = False # Flags to terminate the episodes
        
        n_obs, e_obs, _ = obs_pos
        
        ####################################### TERMINATION CONDITION #######################################
        ## Shutdown simulator because OBSTACLE SHIP REACHS ENDPOINT
        # Get the relative distance between the ship and the end point
        n_route_end = self.obs.auto_pilot.navigate.north[-1]
        e_route_end = self.obs.auto_pilot.navigate.east[-1]
        relative_dist = np.sqrt((n_obs - n_route_end)**2 + (e_obs - e_route_end)**2)
        # Check if the ship arrive at the end point
        arrival_radius = 200 # Arrival radius zone
        if relative_dist <= arrival_radius:
            if not self.obs.stop_flag:
                reward_terminal += 0
            self.obs.stop_flag = True
            status += "|Obstacle ship reaches endpoint|"
            
        ## Shutdown simulator because OBSTACLE SHIP GOES OUTSIDE MAP HORIZON 
        if self.is_pos_outside_horizon(obs_pos, self.obs.ship_model.l_ship):
            if not self.obs.stop_flag:    
                reward_terminal += 0
            self.obs.stop_flag = True
            done = True
            status += "|Obstacle ship hits map horizon|"
                   
        ## Reward for OBSTACLE SHIP HITS TERRAIN
        ## We don't want the obstacle ship to hit the obstacle
        if self.is_pos_inside_obstacles(obs_pos, self.obs.ship_model.l_ship): # When using polygon obstacle
            if not self.obs.stop_flag:
                reward_terminal += -1000
            done = True            
            status += "|Obstacle ship collides with the terrain|"
            
        ## Reward for OBSTACLE SHIP NEW INTERMEDIATE WAYPOINT SAMPLED ON THE TERRAIN
        # Exclusive for obstacle ship
        if self.is_route_outside_horizon(obs_route_coordinate) or\
            self.is_route_inside_obstacles(obs_route_coordinate): # When using polygon obstacle
            if not self.obs.stop_flag:
                reward_terminal += -1000
            self.obs.stop_flag = True
            done = True
            status += "|Obstacle ship IW sampled in terminal state|"
            
        # ## Reward for unnecessary slow ship movement
        # if self.is_too_slow(recorded_time):
        #     reward_terminal += -1000
        #     done = True
        #     status += "|Slow progress failure|"
        
        ## Reward for OBSTACLE SHIP EXPERIENCES NAVIGATIONAL FAILURE
        if self.is_obs_ship_navigation_failure(obs_los_ct_error):
            if not self.obs.stop_flag:
                reward_terminal += -1000
            self.obs.stop_flag = True
            done = True
            status += "|Obstacle ship navigation failure|"
        #####################################################################################################    
        
        
        if done == False:
            status += "|Obstacle ship not in terminal state|"
        
        return reward_terminal, done, status
    
    def shared_terminal_state_reward(self,
                                     test_pos,
                                     obs_pos):
        ''' For computing reward function based on the test and obstacle ship distance
        '''
        ## Initial value
        reward_terminal = 0
        status = " "
        done = False
        
        # Check if the ship collide with each other
        is_collide = self.is_ship_collision(test_pos, obs_pos)
        
        # Compute reward
        if is_collide:
            reward_terminal += 2000
            status += "|Ship collision|"
            self.test.stop_flag = True
            self.obs.stop_flag = True
            done = True
        
        return reward_terminal, done, status
    
    def reward_function(self, states, converted_actions):
         
        # Unpack states for test ship
        test_pos = states[0:3]
        test_measured_shaft_rpm = states[3]
        test_los_ct_error = states[4]
        test_power_load = states[5]
        
        # Unpack states for obstacle ship
        obs_pos = states[6:9]
        obs_los_ct_error = states[9]
        
        # Unpack args
        obs_route_coordinate = converted_actions
        test_available_power_load = self.test.ship_model.simulation_results['available power me [kw]'][-1]
        
        ## FOR TEST SHIP
        # Non terminal reward for test ship
        reward_ntt, r_e_ct_ntt, r_near_col_ntt = self.test_ship_non_terminal_state_reward(test_pos, test_los_ct_error)
        # Get previous cumulative value or start from 0 for plotting
        prev_r_e_ct_ntt = self.reward_results["test_ship"]["reward_e_ct"][-1] if self.reward_results["test_ship"]["reward_e_ct"] else 0
        self.reward_results["test_ship"]["reward_e_ct"].append(prev_r_e_ct_ntt + r_e_ct_ntt)
        prev_r_near_col_ntt = self.reward_results["test_ship"]["reward_near_col"][-1] if self.reward_results["test_ship"]["reward_near_col"] else 0
        self.reward_results["test_ship"]["reward_near_col"].append(prev_r_near_col_ntt + r_near_col_ntt)
        prev_reward_ntt = self.reward_results["test_ship"]["total_non_terminal"][-1] if self.reward_results["test_ship"]["total_non_terminal"] else 0
        self.reward_results["test_ship"]["total_non_terminal"].append(prev_reward_ntt + reward_ntt)
        
        # Terminal reward for test ship
        reward_test_ship_terminal, test_ship_done, test_ship_status = self.test_ship_terminal_state_reward(test_pos, 
                                                                                                           test_los_ct_error, 
                                                                                                           test_power_load, 
                                                                                                           test_available_power_load, 
                                                                                                           test_measured_shaft_rpm)
        
        ## FOR OBSTACLE SHIP
        # Non terminal reward for test ship
        reward_nto, r_base_nto, r_e_ct_nto, r_near_col_nto = self.obs_ship_non_terminal_state_reward(obs_pos, obs_los_ct_error)
        # Get previous cumulative value or start from 0 for plotting
        prev_r_base_nto = self.reward_results["obs_ship"]["reward_base"][-1] if self.reward_results["obs_ship"]["reward_base"] else 0
        self.reward_results["obs_ship"]["reward_base"].append(prev_r_base_nto + r_base_nto)
        prev_r_e_ct_nto = self.reward_results["obs_ship"]["reward_e_ct"][-1] if self.reward_results["obs_ship"]["reward_e_ct"] else 0
        self.reward_results["obs_ship"]["reward_e_ct"].append(prev_r_e_ct_nto + r_e_ct_nto)
        prev_r_near_col_nto = self.reward_results["obs_ship"]["reward_near_col"][-1] if self.reward_results["obs_ship"]["reward_near_col"] else 0
        self.reward_results["obs_ship"]["reward_near_col"].append(prev_r_near_col_nto + r_near_col_nto)
        prev_reward_nto = self.reward_results["obs_ship"]["total_non_terminal"][-1] if self.reward_results["obs_ship"]["total_non_terminal"] else 0
        self.reward_results["obs_ship"]["total_non_terminal"].append(prev_reward_nto + reward_nto)
        
        # Terminal reward for test ship
        reward_obs_ship_terminal, obs_ship_done, obs_ship_status = self.obs_ship_terminal_state_reward(obs_pos,
                                                                                                       obs_route_coordinate,
                                                                                                       obs_los_ct_error)

        ## FOR ALL SHIPS
        # Non terminal reward for obstacle ship
        reward_snt = self.shared_non_terminal_state_reward(test_pos, 
                                                           obs_pos)
        # Get previous cumulative value or start from 0 for plotting        
        prev_reward_snt = self.reward_results["shared"]["total_non_terminal"][-1] if self.reward_results["shared"]["total_non_terminal"] else 0
        self.reward_results["shared"]["total_non_terminal"].append(prev_reward_snt + reward_snt)
        
        # Terminal reward for obstacle ship
        reward_shared_terminal, shared_done, shared_status = self.shared_terminal_state_reward(test_pos, 
                                                                                               obs_pos)

        # Compute output
        reward = reward_ntt + reward_test_ship_terminal + \
                 reward_nto + reward_obs_ship_terminal + \
                 reward_snt + reward_shared_terminal
                 
        status = test_ship_status + obs_ship_status + shared_status
        
        dones = [test_ship_done, obs_ship_done, shared_done]
        done = any(dones)
        
        return reward, done, status
    
    # def old_step(self, 
    #          action, 
    #          SAC_update,
    #          init):
    #     ''' The method is used for stepping up the simulator for all the reinforcement
    #         learning assets
    #     '''
    #     # For all assets
    #     for ship in self.assets:
    #         # Check if the obstacle ship has stop simulating
    #         if hasattr(ship, 'stop_flag') and ship.stop_flag:
    #             # Ship reached endpoint, keep time aligned but don't integrate
    #             ship.ship_model.store_last_simulation_data()
    #             ship.ship_model.int.next_time()  # still progress time
    #             continue
            
    #         # Measure ship position and speed
    #         north_position = ship.ship_model.north
    #         east_position = ship.ship_model.east
    #         heading = ship.ship_model.yaw_angle
    #         forward_speed = ship.ship_model.forward_speed

    #         # If the it is time for the SAC update, alter the obstacle ship autopilot
    #         # Test ship autopilot shall not be altered y the reinforcement learning agent
    #         if SAC_update and ship.type_tag == 'obs_ship':
    #             route_coord_n, route_coord_e = action
            
    #             # Update route_point based on the action
    #             route_coordinate = route_coord_n, route_coord_e
    #             ship.auto_pilot.update_route(route_coordinate)
            
    #             # Update desired_forward_speed based on the action
    #             # self.desired_forward_speed = desired_forward_speed
            
    #             # Store the sampled route coordinate to the holder variable
    #             self.prev_route_coordinate = route_coordinate
                
    #             # Reset the travel distance counter for sampling
    #             self.sampling_distance_travelled = 0
        
    #         # If it is not the time to use action as simulation input, use saved route coordinate
    #         else:
    #             route_coordinate = self.prev_route_coordinate
        
    #         # Find appropriate rudder angle and engine throttle
    #         rudder_angle = ship.auto_pilot.rudder_angle_from_sampled_route(
    #             north_position=north_position,
    #             east_position=east_position,
    #             heading=heading
    #         )
        
    #         throttle = ship.throttle_controller.throttle(
    #             speed_set_point = ship.desired_forward_speed,
    #             measured_speed = forward_speed,
    #             measured_shaft_speed = forward_speed,
    #         )
            
    #         ## COLLISION AVOIDANCE
    #         if ship.type_tag == 'test_ship':
    #             collision_risk = self.is_collision_imminent(self.next_states[0:3], self.next_states[7:10])
                
    #             if collision_risk:
    #                 # Reduce throttle
    #                 throttle *= 0.5
    #                 throttle = np.clip(throttle, 0.0, 1.1)

    #                 # Add a small rudder bias to steer away (rudder angle in radians)
    #                 rudder_angle += np.deg2rad(3)                    
    #                 rudder_angle = np.clip(rudder_angle, -ship.auto_pilot.heading_controller.max_rudder_angle, ship.auto_pilot.heading_controller.max_rudder_angle)

    #                 # Optional print for debugging
    #                 print("Collision Avoidance Activated")
        
    #         # Update and integrate differential equations for current time step
    #         ship.ship_model.store_simulation_data(throttle, 
    #                                               rudder_angle,
    #                                               ship.auto_pilot.get_cross_track_error(),
    #                                               ship.auto_pilot.get_heading_error())
    #         ship.ship_model.update_differentials(engine_throttle=throttle, rudder_angle=rudder_angle)
    #         ship.ship_model.integrate_differentials()
        
    #         ship.integrator_term.append(ship.auto_pilot.navigate.e_ct_int)
    #         ship.time_list.append(ship.ship_model.int.time)
        
    #         # Apply ship drawing (set as optional function)
    #         if self.ship_draw:
    #             if self.time_since_last_ship_drawing > 30:
    #                 ship.ship_model.ship_snap_shot()
    #                 self.time_since_last_ship_drawing = 0 # The ship draw timer is reset here
    #             self.time_since_last_ship_drawing += ship.ship_model.int.dt
        
    #         # Compute reward
    #         pos = [ship.ship_model.north, ship.ship_model.east, ship.ship_model.yaw_angle]
    #         # heading_error = ship.ship_model.simulation_results['heading error [deg]'][-1]
    #         measured_shaft_rpm = ship.ship_model.simulation_results['propeller shaft speed [rpm]'][-1]
    #         los_ct_error = ship.ship_model.simulation_results['cross track error [m]'][-1]
    #         power_load = ship.ship_model.simulation_results['power me [kw]'][-1]
    #         # available_power_load = ship.ship_model.simulation_results['available power me [kw]'][-1]
            
    #         # Store the states required for RL method
    #         self.get_next_state(pos, 
    #                             forward_speed, 
    #                             measured_shaft_rpm, 
    #                             los_ct_error, 
    #                             power_load, 
    #                             ship.type_tag)
        
    #         # Compute travelled distance for the obstacle ship
    #         if init == False and ship.type_tag == "obs_ship":
    #             dist_trav_north = ship.ship_model.simulation_results['north position [m]'][-1] - ship.ship_model.simulation_results['north position [m]'][-2]
    #             dist_trav_east = ship.ship_model.simulation_results['east position [m]'][-1] - ship.ship_model.simulation_results['east position [m]'][-2]
    #             self.eps_distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
    #             self.sampling_distance_travelled += np.sqrt(dist_trav_north**2 + dist_trav_east**2)
        
    #         # Step up the simulator
    #         ship.ship_model.int.next_time()
        
    #     # Set the next state, then reset the next_state container to zero
    #     next_states = self.next_states
    #     self.next_states = self.initial_next_states
        
    #     ## MORE WORK ON THE REWARD FUNCTION NOW
    #     action = self.prev_route_coordinate
    #     reward, done, status = self.reward_function(next_states, action)
        
    #     return next_states, reward, done, status
    
    # def get_next_state(self, pos, forward_speed, measured_shaft_rpm, los_ct_error, power_load, type_tag):
    #     ''' This method is used to get the next RL steps required to compute the reward function and to update the policy.
    #         It is based on the simulator step from each asset. The RL states take various of asset simulator step, hence 
    #         this method is fully depends on the context of the agent's learning purposes.
    #     '''
        
    #     if type_tag == 'test_ship':
    #         self.next_states[0] = self.ensure_scalar(pos[0])
    #         self.next_states[1] = self.ensure_scalar(pos[1]) 
    #         self.next_states[2] = self.ensure_scalar(pos[2])
    #         self.next_states[3] = self.ensure_scalar(forward_speed)
    #         self.next_states[4] = self.ensure_scalar(measured_shaft_rpm)
    #         self.next_states[5] = self.ensure_scalar(los_ct_error)
    #         self.next_states[6] = self.ensure_scalar(power_load)
    #     elif type_tag == 'obs_ship':
    #         self.next_states[7] = self.ensure_scalar(pos[0]) 
    #         self.next_states[8] = self.ensure_scalar(pos[1]) 
    #         self.next_states[9] = self.ensure_scalar(pos[2])
    #         self.next_states[10] = self.ensure_scalar(forward_speed)
    #         self.next_states[11] = self.ensure_scalar(los_ct_error)
        
    #     next_states = self.next_states
        
    #     return next_states
