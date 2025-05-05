"""Biometric Data Collector Module

This module handles the collection of biometric data from various sensors,
including EEG, heart rate, temperature, eye tracking, and facial expressions.
"""

import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class BiometricCollector:
    """Collects and processes biometric data from various sensors
    
    This class manages the connection to biometric sensors, collects data in real-time,
    and performs initial processing before sending the data for further analysis.
    It supports ear-insertable sensors, eye tracking, facial recognition, and other
    biometric data sources.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the biometric collector
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings.
                If None, uses default settings. Defaults to None.
        """
        self.is_initialized = False
        self.sensors = {}
        self.data_buffer = {}
        self.sampling_rates = {}
        self.last_samples = {}
        self.config = config or self._get_default_config()
        self.simulation_mode = self.config.get("simulation_mode", True)  # Default to simulation for development
        print("Biometric Collector created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "simulation_mode": True,  # Use simulated data instead of real sensors
            "sensors": {
                "ear_sensor": {
                    "enabled": True,
                    "sampling_rate": 256,  # Hz
                    "buffer_size": 1024,  # samples
                    "channels": ["eeg", "hr", "temperature"],
                    "device_id": "ear001",
                    "connection_type": "bluetooth"
                },
                "eye_tracker": {
                    "enabled": True,
                    "sampling_rate": 90,  # Hz
                    "buffer_size": 270,  # samples
                    "calibration_required": True,
                    "device_id": "eye001",
                    "connection_type": "usb"
                },
                "facial_camera": {
                    "enabled": True,
                    "sampling_rate": 30,  # Hz
                    "buffer_size": 90,  # samples
                    "resolution": "720p",
                    "device_id": "cam001",
                    "connection_type": "usb"
                },
                "environment_sensors": {
                    "enabled": True,
                    "sampling_rate": 1,  # Hz
                    "buffer_size": 60,  # samples
                    "sensors": ["temperature", "humidity", "light", "noise"],
                    "device_id": "env001",
                    "connection_type": "wifi"
                }
            },
            "data_processing": {
                "real_time_filtering": True,
                "noise_reduction": "adaptive",  # "none", "fixed", "adaptive"
                "artifact_rejection": True,
                "feature_extraction": True,
                "compression": "lossless"  # "none", "lossy", "lossless"
            },
            "edge_computing": {
                "enabled": True,
                "preprocessing_level": "medium",  # "minimal", "medium", "extensive"
                "local_storage_time": 3600,  # seconds (1 hour)
                "battery_optimization": True
            },
            "data_transmission": {
                "protocol": "bluetooth_le",  # "bluetooth_le", "wifi", "5g"
                "encryption": "aes256",
                "compression": True,
                "batch_size": 64,  # samples
                "transmission_interval": 1.0  # seconds
            },
            "simulation": {
                "eeg": {
                    "channels": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
                    "baseline": {  # Baseline amplitude values for different frequency bands
                        "delta": 20.0,  # 0.5-4 Hz
                        "theta": 10.0,  # 4-8 Hz
                        "alpha": 8.0,   # 8-13 Hz
                        "beta": 5.0,    # 13-30 Hz
                        "gamma": 2.0    # 30-100 Hz
                    },
                    "noise_level": 0.2
                },
                "heart_rate": {
                    "baseline": 75.0,  # beats per minute
                    "variation": 5.0,  # standard deviation
                    "stress_increase": 15.0,  # bpm increase during stress
                    "relaxation_decrease": 10.0  # bpm decrease during relaxation
                },
                "temperature": {
                    "baseline": 36.5,  # degrees Celsius
                    "variation": 0.2  # standard deviation
                },
                "eye_tracking": {
                    "fixation_duration": {  # milliseconds
                        "reading": 200,
                        "scanning": 150,
                        "searching": 100
                    },
                    "saccade_length": {  # pixels
                        "reading": 100,
                        "scanning": 200,
                        "searching": 300
                    },
                    "blink_rate": {  # blinks per minute
                        "focused": 10,
                        "normal": 15,
                        "tired": 25
                    }
                },
                "facial_expression": {
                    "emotions": ["neutral", "happy", "sad", "angry", "surprised", "confused", "frustrated"],
                    "transition_probability": 0.1  # Probability of changing emotion state each second
                },
                "environment": {
                    "temperature": {  # degrees Celsius
                        "baseline": 22.0,
                        "variation": 1.0
                    },
                    "humidity": {  # percentage
                        "baseline": 45.0,
                        "variation": 5.0
                    },
                    "light": {  # lux
                        "baseline": 500.0,
                        "variation": 100.0
                    },
                    "noise": {  # decibels
                        "baseline": 40.0,
                        "variation": 10.0
                    }
                }
            }
        }
    
    def initialize(self) -> bool:
        """Initialize sensors and data collection
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # If in simulation mode, just initialize the data structures
            if self.simulation_mode:
                print("Initializing in simulation mode (no physical sensors)")
                self._initialize_simulation()
                self.is_initialized = True
                return True
            
            # Real sensor initialization (would connect to physical devices)
            self._initialize_sensors()
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing biometric collector: {e}")
            return False
    
    def _initialize_simulation(self) -> None:
        """Initialize simulation mode"""
        # Set up data structures for each simulated sensor
        sensor_config = self.config["sensors"]
        
        # Ear sensor (EEG, heart rate, temperature)
        if sensor_config["ear_sensor"]["enabled"]:
            self.sensors["ear_sensor"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["eeg"] = sensor_config["ear_sensor"]["sampling_rate"]
            self.sampling_rates["heart_rate"] = 1  # HR is typically calculated once per second
            self.sampling_rates["temperature"] = 1  # Body temperature changes slowly
            
            self.data_buffer["eeg"] = []
            self.data_buffer["heart_rate"] = []
            self.data_buffer["temperature"] = []
            
            self.last_samples["eeg"] = self._generate_simulated_eeg()
            self.last_samples["heart_rate"] = self.config["simulation"]["heart_rate"]["baseline"]
            self.last_samples["temperature"] = self.config["simulation"]["temperature"]["baseline"]
        
        # Eye tracker
        if sensor_config["eye_tracker"]["enabled"]:
            self.sensors["eye_tracker"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["eye_tracking"] = sensor_config["eye_tracker"]["sampling_rate"]
            self.data_buffer["eye_tracking"] = []
            self.last_samples["eye_tracking"] = {
                "gaze_x": 0.5,  # normalized screen coordinates (0-1)
                "gaze_y": 0.5,
                "pupil_diameter": 3.0,  # mm
                "fixation": True,
                "blink": False,
                "fixation_duration": 0.0,  # ms
                "saccade_length": 0.0,  # pixels
                "state": "reading"  # "reading", "scanning", "searching"
            }
        
        # Facial camera
        if sensor_config["facial_camera"]["enabled"]:
            self.sensors["facial_camera"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["facial_expression"] = sensor_config["facial_camera"]["sampling_rate"]
            self.data_buffer["facial_expression"] = []
            self.last_samples["facial_expression"] = {
                "primary_emotion": "neutral",
                "emotion_probabilities": {
                    "neutral": 0.8,
                    "happy": 0.1,
                    "sad": 0.02,
                    "angry": 0.01,
                    "surprised": 0.02,
                    "confused": 0.03,
                    "frustrated": 0.02
                },
                "face_detected": True,
                "face_landmarks": {}
            }
        
        # Environment sensors
        if sensor_config["environment_sensors"]["enabled"]:
            self.sensors["environment_sensors"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["environment"] = sensor_config["environment_sensors"]["sampling_rate"]
            self.data_buffer["environment"] = []
            self.last_samples["environment"] = {
                "temperature": self.config["simulation"]["environment"]["temperature"]["baseline"],
                "humidity": self.config["simulation"]["environment"]["humidity"]["baseline"],
                "light": self.config["simulation"]["environment"]["light"]["baseline"],
                "noise": self.config["simulation"]["environment"]["noise"]["baseline"]
            }
        
        print("Simulation initialized successfully")
    
    def _initialize_sensors(self) -> None:
        """Initialize physical sensors
        
        In a real implementation, this would establish connections to the physical devices.
        For this reference implementation, we'll just print the steps that would be taken.
        """
        sensor_config = self.config["sensors"]
        
        # Connect to ear sensor
        if sensor_config["ear_sensor"]["enabled"]:
            print(f"Connecting to ear sensor (ID: {sensor_config['ear_sensor']['device_id']}) via {sensor_config['ear_sensor']['connection_type']}...")
            print("Ear sensor connected successfully")
            self.sensors["ear_sensor"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["eeg"] = sensor_config["ear_sensor"]["sampling_rate"]
            self.sampling_rates["heart_rate"] = 1  # HR is typically calculated once per second
            self.sampling_rates["temperature"] = 1  # Body temperature changes slowly
            
            self.data_buffer["eeg"] = []
            self.data_buffer["heart_rate"] = []
            self.data_buffer["temperature"] = []
        
        # Connect to eye tracker
        if sensor_config["eye_tracker"]["enabled"]:
            print(f"Connecting to eye tracker (ID: {sensor_config['eye_tracker']['device_id']}) via {sensor_config['eye_tracker']['connection_type']}...")
            print("Eye tracker connected successfully")
            
            if sensor_config["eye_tracker"]["calibration_required"]:
                print("Calibrating eye tracker...")
                print("Calibration complete")
            
            self.sensors["eye_tracker"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["eye_tracking"] = sensor_config["eye_tracker"]["sampling_rate"]
            self.data_buffer["eye_tracking"] = []
        
        # Connect to facial camera
        if sensor_config["facial_camera"]["enabled"]:
            print(f"Connecting to facial camera (ID: {sensor_config['facial_camera']['device_id']}) via {sensor_config['facial_camera']['connection_type']}...")
            print("Facial camera connected successfully")
            
            self.sensors["facial_camera"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["facial_expression"] = sensor_config["facial_camera"]["sampling_rate"]
            self.data_buffer["facial_expression"] = []
        
        # Connect to environment sensors
        if sensor_config["environment_sensors"]["enabled"]:
            print(f"Connecting to environment sensors (ID: {sensor_config['environment_sensors']['device_id']}) via {sensor_config['environment_sensors']['connection_type']}...")
            print("Environment sensors connected successfully")
            
            self.sensors["environment_sensors"] = {
                "status": "connected",
                "last_reading": time.time(),
                "battery": 100.0,  # percentage
                "error_rate": 0.0,  # percentage
                "calibration": 100.0  # percentage
            }
            
            self.sampling_rates["environment"] = sensor_config["environment_sensors"]["sampling_rate"]
            self.data_buffer["environment"] = []
    
    def start_collection(self, duration: Optional[float] = None) -> bool:
        """Start collecting biometric data
        
        Args:
            duration (Optional[float], optional): Duration to collect data for in seconds.
                If None, collects until stop_collection is called. Defaults to None.
                
        Returns:
            bool: True if collection started successfully, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Biometric collector must be initialized first")
        
        try:
            print(f"Starting biometric data collection" + 
                  (f" for {duration} seconds" if duration else ""))
            
            # In a real implementation, this would start the data collection threads/processes
            # For this reference implementation, we'll just simulate a collection session
            
            if self.simulation_mode:
                # Generate some simulated data
                end_time = time.time() + (duration if duration else 60)  # Default to 60 seconds in simulation mode
                while time.time() < end_time:
                    # Generate and collect simulated data
                    self._collect_simulated_data()
                    
                    # Process and transmit data
                    self._process_data()
                    self._transmit_data()
                    
                    # Sleep to simulate collection frequency
                    time.sleep(0.25)  # 4 Hz update rate for simulation
            else:
                # In a real implementation, this would start continuous data collection
                # For this reference implementation, just simulate a fixed amount of collection
                time.sleep(duration if duration else 5)  # Wait for the specified duration or 5 seconds
            
            print("Data collection completed")
            return True
            
        except Exception as e:
            print(f"Error starting biometric data collection: {e}")
            return False
    
    def stop_collection(self) -> bool:
        """Stop collecting biometric data
        
        Returns:
            bool: True if collection stopped successfully, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Biometric collector must be initialized first")
        
        try:
            print("Stopping biometric data collection")
            
            # In a real implementation, this would stop the data collection threads/processes
            # For this reference implementation, we'll just print a message
            
            # Process any remaining data
            self._process_data()
            self._transmit_data()
            
            print("Data collection stopped successfully")
            return True
            
        except Exception as e:
            print(f"Error stopping biometric data collection: {e}")
            return False
    
    def _collect_simulated_data(self) -> None:
        """Collect simulated biometric data"""
        current_time = time.time()
        
        # Simulate EEG data collection
        if "eeg" in self.sampling_rates:
            # Update the simulated EEG data
            eeg_data = self._generate_simulated_eeg()
            self.last_samples["eeg"] = eeg_data
            self.data_buffer["eeg"].append({
                "timestamp": current_time,
                "data": eeg_data
            })
        
        # Simulate heart rate data collection
        if "heart_rate" in self.sampling_rates:
            # Update the simulated heart rate data
            hr_data = self._generate_simulated_heart_rate()
            self.last_samples["heart_rate"] = hr_data
            self.data_buffer["heart_rate"].append({
                "timestamp": current_time,
                "data": hr_data
            })
        
        # Simulate temperature data collection
        if "temperature" in self.sampling_rates:
            # Update the simulated temperature data
            temp_data = self._generate_simulated_temperature()
            self.last_samples["temperature"] = temp_data
            self.data_buffer["temperature"].append({
                "timestamp": current_time,
                "data": temp_data
            })
        
        # Simulate eye tracking data collection
        if "eye_tracking" in self.sampling_rates:
            # Update the simulated eye tracking data
            eye_data = self._generate_simulated_eye_tracking()
            self.last_samples["eye_tracking"] = eye_data
            self.data_buffer["eye_tracking"].append({
                "timestamp": current_time,
                "data": eye_data
            })
        
        # Simulate facial expression data collection
        if "facial_expression" in self.sampling_rates:
            # Update the simulated facial expression data
            face_data = self._generate_simulated_facial_expression()
            self.last_samples["facial_expression"] = face_data
            self.data_buffer["facial_expression"].append({
                "timestamp": current_time,
                "data": face_data
            })
        
        # Simulate environment data collection
        if "environment" in self.sampling_rates:
            # Update the simulated environment data
            env_data = self._generate_simulated_environment()
            self.last_samples["environment"] = env_data
            self.data_buffer["environment"].append({
                "timestamp": current_time,
                "data": env_data
            })
    
    def _generate_simulated_eeg(self) -> Dict[str, Any]:
        """Generate simulated EEG data
        
        Returns:
            Dict[str, Any]: Simulated EEG data
        """
        # Get EEG configuration
        eeg_config = self.config["simulation"]["eeg"]
        channels = eeg_config["channels"]
        bands = eeg_config["baseline"]
        noise_level = eeg_config["noise_level"]
        
        # Generate simulated data for each channel
        channel_data = {}
        attention_level = random.uniform(0.3, 0.9)  # Randomize attention level for simulation
        
        for channel in channels:
            # Generate band powers based on attention level
            # - Alpha power increases with relaxation (inversely correlated with attention)
            # - Beta power increases with attention
            # - Delta power decreases with attention
            band_powers = {
                "delta": bands["delta"] * (1 - attention_level * 0.3) + random.normalvariate(0, noise_level),
                "theta": bands["theta"] + random.normalvariate(0, noise_level),
                "alpha": bands["alpha"] * (1 - attention_level * 0.5) + random.normalvariate(0, noise_level),
                "beta": bands["beta"] * (1 + attention_level * 0.8) + random.normalvariate(0, noise_level),
                "gamma": bands["gamma"] + random.normalvariate(0, noise_level)
            }
            
            # Ensure all values are positive
            for band in band_powers:
                band_powers[band] = max(0, band_powers[band])
            
            channel_data[channel] = band_powers
        
        # Calculate attention and meditation metrics from the band powers
        # This is a simplified model - real implementations would use more sophisticated algorithms
        attention_metric = 0.0
        meditation_metric = 0.0
        
        # Average beta/alpha ratio across channels as attention indicator
        beta_alpha_ratio = 0.0
        for channel in channels:
            if channel_data[channel]["alpha"] > 0:
                beta_alpha_ratio += channel_data[channel]["beta"] / channel_data[channel]["alpha"]
        beta_alpha_ratio /= len(channels)
        
        # Scale to 0-1 range (with some typical range assumptions)
        attention_metric = min(1.0, max(0.0, (beta_alpha_ratio - 0.5) / 2.0))
        
        # Meditation is indicated by higher alpha and lower beta
        alpha_beta_ratio = 0.0
        for channel in channels:
            if channel_data[channel]["beta"] > 0:
                alpha_beta_ratio += channel_data[channel]["alpha"] / channel_data[channel]["beta"]
        alpha_beta_ratio /= len(channels)
        
        # Scale to 0-1 range (with some typical range assumptions)
        meditation_metric = min(1.0, max(0.0, (alpha_beta_ratio - 0.5) / 2.0))
        
        return {
            "channels": channel_data,
            "metrics": {
                "attention": attention_metric,
                "meditation": meditation_metric,
                "cognitive_load": 1.0 - meditation_metric  # Simplified model
            }
        }
    
    def _generate_simulated_heart_rate(self) -> float:
        """Generate simulated heart rate data
        
        Returns:
            float: Simulated heart rate (bpm)
        """
        # Get heart rate configuration
        hr_config = self.config["simulation"]["heart_rate"]
        baseline = hr_config["baseline"]
        variation = hr_config["variation"]
        
        # Use the current heart rate as the baseline if available
        current_hr = self.last_samples.get("heart_rate", baseline)
        
        # Add some random variation to simulate natural fluctuation
        hr = current_hr + random.normalvariate(0, variation * 0.3)
        
        # Apply some constraints to keep the heart rate within realistic bounds
        hr = max(50.0, min(120.0, hr))  # Constrain between 50-120 bpm for normal variation
        
        return hr
    
    def _generate_simulated_temperature(self) -> float:
        """Generate simulated body temperature data
        
        Returns:
            float: Simulated body temperature (Celsius)
        """
        # Get temperature configuration
        temp_config = self.config["simulation"]["temperature"]
        baseline = temp_config["baseline"]
        variation = temp_config["variation"]
        
        # Use the current temperature as the baseline if available
        current_temp = self.last_samples.get("temperature", baseline)
        
        # Add some random variation to simulate natural fluctuation
        temp = current_temp + random.normalvariate(0, variation * 0.3)
        
        # Apply some constraints to keep the temperature within realistic bounds
        temp = max(36.0, min(37.5, temp))  # Constrain between 36-37.5°C for normal variation
        
        return temp
    
    def _generate_simulated_eye_tracking(self) -> Dict[str, Any]:
        """Generate simulated eye tracking data
        
        Returns:
            Dict[str, Any]: Simulated eye tracking data
        """
        # Get eye tracking configuration
        eye_config = self.config["simulation"]["eye_tracking"]
        
        # Get current state if available
        current_state = self.last_samples.get("eye_tracking", {})
        current_state_name = current_state.get("state", "reading")
        
        # Occasionally change state
        if random.random() < 0.05:  # 5% chance to change state each call
            states = ["reading", "scanning", "searching"]
            states.remove(current_state_name)  # Don't pick the same state
            current_state_name = random.choice(states)
        
        # Determine parameters based on the state
        fixation_duration = eye_config["fixation_duration"][current_state_name]
        saccade_length = eye_config["saccade_length"][current_state_name]
        blink_rate = eye_config["blink_rate"]["normal"]  # Use normal by default
        
        # Determine if currently fixating or making saccade
        is_fixating = current_state.get("fixation", True)
        if random.random() < 0.1:  # 10% chance to switch between fixation and saccade
            is_fixating = not is_fixating
        
        # Determine if currently blinking
        is_blinking = current_state.get("blink", False)
        if not is_blinking and random.random() < (blink_rate / 60 / 4):  # Convert from blinks per minute to probability per call (assuming 4 calls per second)
            is_blinking = True
        elif is_blinking:  # Blinks typically last 100-150 ms
            is_blinking = False
        
        # Update gaze position based on fixation/saccade state
        current_x = current_state.get("gaze_x", 0.5)
        current_y = current_state.get("gaze_y", 0.5)
        
        if is_fixating:
            # Small random movements during fixation
            new_x = current_x + random.normalvariate(0, 0.01)
            new_y = current_y + random.normalvariate(0, 0.01)
        else:
            # Larger movements during saccades
            saccade_pixels = saccade_length * random.uniform(0.5, 1.5)
            saccade_angle = random.uniform(0, 2 * np.pi)
            saccade_x = saccade_pixels * np.cos(saccade_angle) / 1920  # Normalize to 0-1 range assuming 1920x1080 screen
            saccade_y = saccade_pixels * np.sin(saccade_angle) / 1080
            
            new_x = current_x + saccade_x
            new_y = current_y + saccade_y
        
        # Ensure gaze is within screen bounds
        new_x = max(0.0, min(1.0, new_x))
        new_y = max(0.0, min(1.0, new_y))
        
        # Update pupil diameter based on cognitive load and light conditions
        current_pupil = current_state.get("pupil_diameter", 3.0)
        cognitive_load = 0.5  # Placeholder - in a real implementation, this would come from EEG data
        light_level = self.last_samples.get("environment", {}).get("light", 500) / 1000  # Normalize to 0-1 range
        
        # Pupil constricts with bright light, dilates with cognitive load
        target_pupil = 2.0 + 3.0 * (1 - light_level) + 1.0 * cognitive_load
        new_pupil = current_pupil * 0.9 + target_pupil * 0.1  # Smooth changes
        
        # Return the simulated eye tracking data
        return {
            "gaze_x": new_x,
            "gaze_y": new_y,
            "pupil_diameter": new_pupil,
            "fixation": is_fixating,
            "blink": is_blinking,
            "fixation_duration": fixation_duration if is_fixating else 0.0,
            "saccade_length": saccade_length if not is_fixating else 0.0,
            "state": current_state_name
        }
    
    def _generate_simulated_facial_expression(self) -> Dict[str, Any]:
        """Generate simulated facial expression data
        
        Returns:
            Dict[str, Any]: Simulated facial expression data
        """
        # Get facial expression configuration
        face_config = self.config["simulation"]["facial_expression"]
        emotions = face_config["emotions"]
        transition_prob = face_config["transition_probability"]
        
        # Get current state if available
        current_state = self.last_samples.get("facial_expression", {})
        current_emotion = current_state.get("primary_emotion", "neutral")
        current_probs = current_state.get("emotion_probabilities", 
                                         {emotion: 0.1 for emotion in emotions})
        
        # Occasionally change primary emotion
        if random.random() < transition_prob:
            # Don't pick the same emotion
            other_emotions = [e for e in emotions if e != current_emotion]
            current_emotion = random.choice(other_emotions)
        
        # Generate new probability distribution
        new_probs = {}
        # Primary emotion gets high probability
        new_probs[current_emotion] = random.uniform(0.7, 0.9)
        
        # Distribute remaining probability among other emotions
        remaining_prob = 1.0 - new_probs[current_emotion]
        other_emotions = [e for e in emotions if e != current_emotion]
        
        for emotion in other_emotions:
            # Last emotion gets whatever probability is left
            if emotion == other_emotions[-1]:
                new_probs[emotion] = remaining_prob
            else:
                # Random fraction of remaining probability
                emotion_prob = remaining_prob * random.random()
                new_probs[emotion] = emotion_prob
                remaining_prob -= emotion_prob
        
        # Smooth transition from previous state
        smoothed_probs = {}
        for emotion in emotions:
            previous = current_probs.get(emotion, 0.0)
            new = new_probs.get(emotion, 0.0)
            smoothed_probs[emotion] = previous * 0.7 + new * 0.3
        
        # Determine primary emotion based on highest probability
        primary_emotion = max(smoothed_probs.items(), key=lambda x: x[1])[0]
        
        # Simulate face landmarks (simplified)
        face_landmarks = {
            "left_eye": [(0.3, 0.4), (0.35, 0.38)],
            "right_eye": [(0.7, 0.4), (0.65, 0.38)],
            "nose": [(0.5, 0.5)],
            "mouth": [(0.4, 0.7), (0.6, 0.7)]
        }
        
        # Return simulated facial expression data
        return {
            "primary_emotion": primary_emotion,
            "emotion_probabilities": smoothed_probs,
            "face_detected": True,
            "face_landmarks": face_landmarks
        }
    
    def _generate_simulated_environment(self) -> Dict[str, float]:
        """Generate simulated environment data
        
        Returns:
            Dict[str, float]: Simulated environment data
        """
        # Get environment configuration
        env_config = self.config["simulation"]["environment"]
        
        # Get current state if available
        current_state = self.last_samples.get("environment", {})
        
        # Generate new values with smooth transitions from current state
        result = {}
        
        # Temperature
        current_temp = current_state.get("temperature", env_config["temperature"]["baseline"])
        temp_variation = env_config["temperature"]["variation"]
        new_temp = current_temp + random.normalvariate(0, temp_variation * 0.1)
        # Ensure temperature stays within realistic bounds
        new_temp = max(18.0, min(28.0, new_temp))
        result["temperature"] = new_temp
        
        # Humidity
        current_humidity = current_state.get("humidity", env_config["humidity"]["baseline"])
        humidity_variation = env_config["humidity"]["variation"]
        new_humidity = current_humidity + random.normalvariate(0, humidity_variation * 0.1)
        # Ensure humidity stays within realistic bounds
        new_humidity = max(20.0, min(80.0, new_humidity))
        result["humidity"] = new_humidity
        
        # Light
        current_light = current_state.get("light", env_config["light"]["baseline"])
        light_variation = env_config["light"]["variation"]
        new_light = current_light + random.normalvariate(0, light_variation * 0.1)
        # Ensure light stays within realistic bounds
        new_light = max(100.0, min(1000.0, new_light))
        result["light"] = new_light
        
        # Noise
        current_noise = current_state.get("noise", env_config["noise"]["baseline"])
        noise_variation = env_config["noise"]["variation"]
        new_noise = current_noise + random.normalvariate(0, noise_variation * 0.1)
        # Ensure noise stays within realistic bounds
        new_noise = max(20.0, min(80.0, new_noise))
        result["noise"] = new_noise
        
        return result
    
    def _process_data(self) -> None:
        """Process collected biometric data
        
        In a real implementation, this would apply filters, extract features, etc.
        For this reference implementation, we'll just print some stats.
        """
        # Skip if no data has been collected
        if not any(self.data_buffer.values()):
            return
        
        # Apply the processing steps based on configuration
        processing_config = self.config["data_processing"]
        
        # Count number of samples for each data type
        stats = {}
        for data_type, buffer in self.data_buffer.items():
            stats[data_type] = len(buffer)
        
        # If real-time filtering is enabled
        if processing_config["real_time_filtering"]:
            # In a real implementation, this would apply appropriate filters to each data type
            pass
        
        # If noise reduction is enabled
        if processing_config["noise_reduction"] != "none":
            # In a real implementation, this would apply noise reduction algorithms
            pass
        
        # If artifact rejection is enabled
        if processing_config["artifact_rejection"]:
            # In a real implementation, this would detect and remove artifacts
            pass
        
        # If feature extraction is enabled
        if processing_config["feature_extraction"]:
            # In a real implementation, this would extract relevant features
            pass
        
        # If edge computing is enabled, process data on device
        if self.config["edge_computing"]["enabled"]:
            # Extract cognitive state metrics
            cognitive_state = self._extract_cognitive_state()
            
            # Print some stats for reference
            if cognitive_state:
                print(f"Cognitive State: Attention={cognitive_state.get('attention', 0):.2f}, " + 
                      f"Focus={cognitive_state.get('focus', 0):.2f}, " + 
                      f"Cognitive Load={cognitive_state.get('cognitive_load', 0):.2f}, " + 
                      f"Fatigue={cognitive_state.get('fatigue', 0):.2f}")
    
    def _extract_cognitive_state(self) -> Dict[str, float]:
        """Extract cognitive state metrics from collected data
        
        Returns:
            Dict[str, float]: Cognitive state metrics
        """
        # Skip if no data is available
        if not self.last_samples:
            return {}
        
        # Extract metrics from different data sources
        cognitive_state = {}
        
        # Extract from EEG data if available
        if "eeg" in self.last_samples:
            eeg_metrics = self.last_samples["eeg"].get("metrics", {})
            cognitive_state["attention"] = eeg_metrics.get("attention", 0.5)
            cognitive_state["meditation"] = eeg_metrics.get("meditation", 0.5)
            cognitive_state["cognitive_load"] = eeg_metrics.get("cognitive_load", 0.5)
        
        # Extract from eye tracking data if available
        if "eye_tracking" in self.last_samples:
            eye_data = self.last_samples["eye_tracking"]
            
            # Focus metric based on fixation/saccade patterns
            if eye_data.get("state") == "reading" and eye_data.get("fixation"):
                focus = 0.8  # High focus during reading fixations
            elif eye_data.get("state") == "scanning" and not eye_data.get("fixation"):
                focus = 0.5  # Medium focus during scanning saccades
            elif eye_data.get("blink"):
                focus = 0.3  # Low focus during blinks
            else:
                focus = 0.6  # Default focus level
            
            cognitive_state["focus"] = focus
            
            # Fatigue metric based on blink rate and pupil diameter
            blink_rate = self._calculate_blink_rate()
            pupil_diameter = eye_data.get("pupil_diameter", 3.0)
            
            # Higher blink rate and smaller pupils indicate fatigue
            if blink_rate > 20 and pupil_diameter < 2.5:
                fatigue = 0.8  # High fatigue
            elif blink_rate > 15 or pupil_diameter < 3.0:
                fatigue = 0.5  # Medium fatigue
            else:
                fatigue = 0.2  # Low fatigue
            
            cognitive_state["fatigue"] = fatigue
        
        # Extract from facial expression data if available
        if "facial_expression" in self.last_samples:
            face_data = self.last_samples["facial_expression"]
            emotion_probs = face_data.get("emotion_probabilities", {})
            
            # Engagement metric based on emotional response
            engagement = 1.0 - (emotion_probs.get("neutral", 0.5) * 0.8)  # Less neutral = more engaged
            cognitive_state["engagement"] = engagement
            
            # Cognitive state might also be affected by specific emotions
            confusion = emotion_probs.get("confused", 0.0)
            frustration = emotion_probs.get("frustrated", 0.0)
            
            if "cognitive_load" in cognitive_state:
                # Increase cognitive load based on confusion and frustration
                cognitive_state["cognitive_load"] = min(1.0, cognitive_state["cognitive_load"] + (confusion + frustration) * 0.3)
        
        # Extract from physiological data if available
        if "heart_rate" in self.last_samples:
            heart_rate = self.last_samples["heart_rate"]
            
            # Stress metric based on heart rate
            # This is a simplified model - real implementations would use HRV and other metrics
            heart_rate_baseline = self.config["simulation"]["heart_rate"]["baseline"]
            stress = max(0, min(1, (heart_rate - heart_rate_baseline) / 25))  # Scale to 0-1
            
            cognitive_state["stress"] = stress
        
        # Combined metrics
        if "attention" in cognitive_state and "focus" in cognitive_state:
            # Combined concentration metric
            cognitive_state["concentration"] = (cognitive_state["attention"] * 0.5 + cognitive_state["focus"] * 0.5)
        
        if "stress" in cognitive_state and "fatigue" in cognitive_state and "cognitive_load" in cognitive_state:
            # Combined cognitive well-being metric (inverse of stress, fatigue, and load)
            well_being = 1.0 - (cognitive_state["stress"] * 0.3 + cognitive_state["fatigue"] * 0.3 + cognitive_state["cognitive_load"] * 0.4)
            cognitive_state["cognitive_well_being"] = well_being
        
        return cognitive_state
    
    def _calculate_blink_rate(self) -> float:
        """Calculate blink rate from recent eye tracking data
        
        Returns:
            float: Blink rate (blinks per minute)
        """
        # Count blinks in the recent buffer
        blink_count = 0
        time_window = 0.0
        
        if "eye_tracking" in self.data_buffer and self.data_buffer["eye_tracking"]:
            # Use up to the last 60 seconds of data
            recent_data = self.data_buffer["eye_tracking"][-240:]  # Assuming 4 Hz sampling rate
            
            for sample in recent_data:
                if sample["data"].get("blink", False):
                    blink_count += 1
            
            # Calculate time window
            if len(recent_data) >= 2:
                time_window = (recent_data[-1]["timestamp"] - recent_data[0]["timestamp"]) / 60.0  # Convert to minutes
            
            # Prevent division by zero
            if time_window > 0:
                return blink_count / time_window
        
        # Default to a typical blink rate if calculation isn't possible
        return 15.0
    
    def _transmit_data(self) -> None:
        """Transmit processed data to the central system
        
        In a real implementation, this would send the data via the configured protocol.
        For this reference implementation, we'll just clear the buffer.
        """
        # Skip if no data has been collected
        if not any(self.data_buffer.values()):
            return
        
        # In a real implementation, this would send the data to the central system
        # For now, just print a message and clear the buffers
        total_samples = sum(len(buffer) for buffer in self.data_buffer.values())
        
        # Extract the latest cognitive state metrics
        cognitive_state = self._extract_cognitive_state()
        
        print(f"Transmitting {total_samples} samples of biometric data" + 
              (f" with cognitive state metrics: {cognitive_state}" if cognitive_state else ""))
        
        # Clear the buffers after transmission
        for data_type in self.data_buffer:
            self.data_buffer[data_type] = []

# Example usage
if __name__ == "__main__":
    # Create and initialize biometric collector in simulation mode
    collector = BiometricCollector()
    collector.initialize()
    
    # Start collection for 5 seconds
    collector.start_collection(5)
    
    # Get the latest cognitive state
    cognitive_state = collector._extract_cognitive_state()
    print("\nFinal Cognitive State:")
    for metric, value in cognitive_state.items():
        print(f"  {metric}: {value:.2f}")
