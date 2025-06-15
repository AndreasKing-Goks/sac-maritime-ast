'''
This modulus provide classes for the reinforcement learning environment 
based on the Ship in Transit Simulator and Tensorflow library.
'''

from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from collections import defaultdict
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from simulators.ship_in_transit.ship_model import ShipModel, ShipModelAST
from simulators.ship_in_transit.controllers import EngineThrottleFromSpeedSetPoint, HeadingByRouteController, HeadingBySampledRouteController
from simulators.ship_in_transit.obstacle import StaticObstacle, PolygonObstacle

from dataclasses import dataclass, field
from typing import Union, List

import copy

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
    '''
    This class is the main class for the reinforcement learning environment based on the
    Ship-Transit Simulator suited for two ship actor
    '''
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