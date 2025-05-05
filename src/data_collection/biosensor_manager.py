"""Biosensor Manager Module

This module handles the integration with various biometric sensors including EEG, 
eye tracking, heart rate, and other physiological data collection devices.

Based on Ucaretron Inc.'s patent application for AI-based personalized learning systems.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class BiosensorManager:
    """Manager for biosensor data collection and processing
    
    This class handles:
    - Connection to biosensor devices
    - Data collection and stream management
    - Initial edge processing of raw biosensor data
    - Buffering and synchronization of multi-modal data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the biosensor manager
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings
        """
        self.is_initialized = False
        self.config = config or self._get_default_config()
        
        # Data storage
        self.data_buffers = {}
        self.current_state = {}
        self.connected_devices = {}
        
        print("Biosensor Manager created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "enabled": True,
            "devices": {
                "ear_eeg": {
                    "enabled": True,
                    "sampling_rate": 250,  # Hz
                    "buffer_size": 10000,  # samples (40 seconds at 250Hz)
                    "channels": ["Fp1", "Fp2", "T3", "T4"],
                    "edge_processing": True,
                    "connection": "bluetooth"
                },
                "eye_tracker": {
                    "enabled": True,
                    "sampling_rate": 60,  # Hz
                    "buffer_size": 1800,  # samples (30 seconds at 60Hz)
                    "edge_processing": True,
                    "connection": "usb"
                },
                "heart_rate": {
                    "enabled": True,
                    "sampling_rate": 1,  # Hz
                    "buffer_size": 300,  # samples (5 minutes at 1Hz)
                    "edge_processing": False,
                    "connection": "bluetooth"
                },
                "skin_conductance": {
                    "enabled": False,
                    "sampling_rate": 10,  # Hz
                    "buffer_size": 600,  # samples (1 minute at 10Hz)
                    "edge_processing": False,
                    "connection": "bluetooth"
                }
            },
            "edge_computing": {
                "enabled": True,
                "feature_extraction": True,
                "data_compression": True,
                "anomaly_detection": True,
                "privacy_filtering": True
            },
            "data_storage": {
                "enabled": True,
                "storage_format": "hdf5",
                "storage_path": "./data/biosensor",
                "storage_limit": 1000  # MB
            },
            "simulation": {
                "enabled": True,  # Use simulated data if real devices not available
                "noise_level": 0.1,
                "scenarios": ["focused", "distracted", "tired", "engaged"]
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the biosensor manager
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            if not self.config["enabled"]:
                print("Biosensor Manager disabled in config")
                return False
                
            # Create storage directory if enabled
            if self.config["data_storage"]["enabled"]:
                storage_path = Path(self.config["data_storage"]["storage_path"])
                storage_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize data buffers for each enabled device
            for device_id, device_config in self.config["devices"].items():
                if device_config["enabled"]:
                    buffer_size = device_config["buffer_size"]
                    channels = device_config.get("channels", [device_id])
                    
                    # Create buffer for this device
                    self.data_buffers[device_id] = {
                        "timestamps": np.zeros(buffer_size),
                        "data": np.zeros((len(channels), buffer_size)),
                        "buffer_index": 0,
                        "channels": channels
                    }
                    
                    # Try to connect to device
                    connected = self._connect_to_device(device_id, device_config)
                    self.connected_devices[device_id] = connected
                    
                    if not connected and self.config["simulation"]["enabled"]:
                        print(f"Using simulated data for {device_id}")
            
            # Initialize current state variables
            self.current_state = {
                "attention": 0.5,
                "meditation": 0.5,
                "focus": 0.5,
                "stress": 0.3,
                "engagement": 0.5,
                "cognitive_load": 0.4,
                "timestamp": time.time()
            }
            
            self.is_initialized = True
            print("Biosensor Manager initialized successfully")
            print(f"Connected devices: {[dev for dev, status in self.connected_devices.items() if status]}")
            print(f"Simulated devices: {[dev for dev, status in self.connected_devices.items() if not status]}")
            return True
        
        except Exception as e:
            print(f"Error initializing Biosensor Manager: {e}")
            return False
    
    def _connect_to_device(self, device_id: str, device_config: Dict[str, Any]) -> bool:
        """Connect to a biosensor device
        
        Args:
            device_id (str): Device identifier
            device_config (Dict[str, Any]): Device configuration
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        # This is a simplified implementation - in a real system this would
        # actually connect to physical devices using the appropriate protocols
        
        try:
            connection_type = device_config.get("connection", "usb")
            
            # Simulate connection attempt
            print(f"Attempting to connect to {device_id} via {connection_type}...")
            
            # In a real implementation, this would attempt to establish connection
            # Here we'll just simulate success/failure
            
            # For demonstration purposes, consider specific device types
            if device_id == "ear_eeg":
                # Simulate ear-insertable EEG device - special focus of the system
                print(f"Connecting to ear-insertable EEG device...")
                print(f"Channels: {device_config.get('channels', [])}")
                print(f"Sampling rate: {device_config.get('sampling_rate', 0)} Hz")
                
                # Simulate connection based on simulation setting
                # (in a real system this would actually try to connect)
                return False  # Pretend connection failed to demonstrate simulation
                
            elif device_id == "eye_tracker":
                print(f"Connecting to eye tracking device...")
                return False  # Pretend connection failed
                
            elif device_id == "heart_rate":
                print(f"Connecting to heart rate monitor...")
                return False  # Pretend connection failed
            
            else:
                print(f"Connecting to generic device {device_id}...")
                return False  # Pretend connection failed
                
        except Exception as e:
            print(f"Error connecting to {device_id}: {e}")
            return False
    
    def get_latest_data(self) -> Dict[str, Any]:
        """Get the latest processed biosensor data
        
        Returns:
            Dict[str, Any]: Latest processed biosensor state
        """
        if not self.is_initialized:
            raise RuntimeError("Biosensor Manager not initialized")
        
        # Update simulated data to change over time
        if self.config["simulation"]["enabled"]:
            self._update_simulated_data()
        
        # Return the current state
        return self.current_state.copy()
    
    def _update_simulated_data(self) -> None:
        """Update simulated biosensor data"""
        # Get current time
        current_time = time.time()
        time_diff = current_time - self.current_state["timestamp"]
        
        # Only update if enough time has passed (at least 1 second)
        if time_diff < 1.0:
            return
            
        # Update timestamp
        self.current_state["timestamp"] = current_time
        
        # Add subtle random changes to simulate real data
        noise_level = self.config["simulation"]["noise_level"]
        
        # Select one of the predefined scenarios
        scenarios = self.config["simulation"]["scenarios"]
        scenario = scenarios[int(current_time / 30) % len(scenarios)]
        
        # Update state based on selected scenario plus noise
        if scenario == "focused":
            self.current_state["attention"] = min(1.0, max(0.0, 0.8 + np.random.normal(0, noise_level)))
            self.current_state["meditation"] = min(1.0, max(0.0, 0.6 + np.random.normal(0, noise_level)))
            self.current_state["focus"] = min(1.0, max(0.0, 0.85 + np.random.normal(0, noise_level)))
            self.current_state["stress"] = min(1.0, max(0.0, 0.3 + np.random.normal(0, noise_level)))
            self.current_state["engagement"] = min(1.0, max(0.0, 0.7 + np.random.normal(0, noise_level)))
            self.current_state["cognitive_load"] = min(1.0, max(0.0, 0.5 + np.random.normal(0, noise_level)))
            
        elif scenario == "distracted":
            self.current_state["attention"] = min(1.0, max(0.0, 0.3 + np.random.normal(0, noise_level)))
            self.current_state["meditation"] = min(1.0, max(0.0, 0.4 + np.random.normal(0, noise_level)))
            self.current_state["focus"] = min(1.0, max(0.0, 0.25 + np.random.normal(0, noise_level)))
            self.current_state["stress"] = min(1.0, max(0.0, 0.6 + np.random.normal(0, noise_level)))
            self.current_state["engagement"] = min(1.0, max(0.0, 0.4 + np.random.normal(0, noise_level)))
            self.current_state["cognitive_load"] = min(1.0, max(0.0, 0.3 + np.random.normal(0, noise_level)))
            
        elif scenario == "tired":
            self.current_state["attention"] = min(1.0, max(0.0, 0.4 + np.random.normal(0, noise_level)))
            self.current_state["meditation"] = min(1.0, max(0.0, 0.7 + np.random.normal(0, noise_level)))
            self.current_state["focus"] = min(1.0, max(0.0, 0.4 + np.random.normal(0, noise_level)))
            self.current_state["stress"] = min(1.0, max(0.0, 0.5 + np.random.normal(0, noise_level)))
            self.current_state["engagement"] = min(1.0, max(0.0, 0.3 + np.random.normal(0, noise_level)))
            self.current_state["cognitive_load"] = min(1.0, max(0.0, 0.6 + np.random.normal(0, noise_level)))
            
        elif scenario == "engaged":
            self.current_state["attention"] = min(1.0, max(0.0, 0.7 + np.random.normal(0, noise_level)))
            self.current_state["meditation"] = min(1.0, max(0.0, 0.4 + np.random.normal(0, noise_level)))
            self.current_state["focus"] = min(1.0, max(0.0, 0.7 + np.random.normal(0, noise_level)))
            self.current_state["stress"] = min(1.0, max(0.0, 0.4 + np.random.normal(0, noise_level)))
            self.current_state["engagement"] = min(1.0, max(0.0, 0.9 + np.random.normal(0, noise_level)))
            self.current_state["cognitive_load"] = min(1.0, max(0.0, 0.7 + np.random.normal(0, noise_level)))
    
    def get_device_status(self) -> Dict[str, bool]:
        """Get connection status for all devices
        
        Returns:
            Dict[str, bool]: Dictionary of device IDs to connection status
        """
        if not self.is_initialized:
            raise RuntimeError("Biosensor Manager not initialized")
            
        return self.connected_devices.copy()
    
    def start_recording(self) -> bool:
        """Start recording data from all connected devices
        
        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Biosensor Manager not initialized")
            
        try:
            print("Starting biosensor recording...")
            
            # In a real implementation, this would start actual recording
            # Here we'll just simulate it
            
            # Reset all buffers
            for device_id, buffer in self.data_buffers.items():
                buffer["buffer_index"] = 0
                
            print("Biosensor recording started")
            return True
            
        except Exception as e:
            print(f"Error starting biosensor recording: {e}")
            return False
    
    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and return the recorded data
        
        Returns:
            Dict[str, Any]: Recorded data from all devices
        """
        if not self.is_initialized:
            raise RuntimeError("Biosensor Manager not initialized")
            
        try:
            print("Stopping biosensor recording...")
            
            # In a real implementation, this would stop actual recording
            # and return the collected data
            
            # Here we'll just simulate it with the current state
            recording = {
                "timestamp": time.time(),
                "duration": 60,  # Simulate 60 seconds of recording
                "devices": list(self.connected_devices.keys()),
                "data": {
                    device_id: {
                        "timestamps": [time.time() - 60 + i for i in range(60)],
                        "values": [self.current_state.copy() for _ in range(60)]
                    } for device_id in self.connected_devices
                }
            }
            
            print("Biosensor recording stopped")
            return recording
            
        except Exception as e:
            print(f"Error stopping biosensor recording: {e}")
            return {}
    
    def process_raw_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw biosensor data into cognitive state features
        
        Args:
            raw_data (Dict[str, Any]): Raw biosensor data
            
        Returns:
            Dict[str, Any]: Processed cognitive state data
        """
        if not self.is_initialized:
            raise RuntimeError("Biosensor Manager not initialized")
            
        try:
            # In a real implementation, this would apply signal processing and
            # feature extraction to the raw biosensor data
            
            # Here we'll just echo the current state
            if self.config["simulation"]["enabled"]:
                self._update_simulated_data()
                
            # Add a timestamp if not present
            processed_data = self.current_state.copy()
            if "timestamp" not in processed_data:
                processed_data["timestamp"] = time.time()
                
            return processed_data
            
        except Exception as e:
            print(f"Error processing raw biosensor data: {e}")
            return {"error": str(e)}
    
    def get_cognitive_features(self) -> Dict[str, float]:
        """Extract cognitive features from current biosensor state
        
        Returns:
            Dict[str, float]: Cognitive features
        """
        if not self.is_initialized:
            raise RuntimeError("Biosensor Manager not initialized")
            
        # Get latest data
        latest_data = self.get_latest_data()
        
        # Calculate composite features
        composites = {}
        
        # Attention level (0-1) - combination of attention and focus
        composites["attention_level"] = (
            0.7 * latest_data.get("attention", 0.5) +
            0.3 * latest_data.get("focus", 0.5)
        )
        
        # Engagement level (0-1) - combination of engagement and meditation (inverted)
        composites["engagement_level"] = (
            0.8 * latest_data.get("engagement", 0.5) +
            0.2 * (1.0 - latest_data.get("meditation", 0.5))
        )
        
        # Cognitive load level (0-1) - combination of cognitive load and stress
        composites["cognitive_load_level"] = (
            0.7 * latest_data.get("cognitive_load", 0.5) +
            0.3 * latest_data.get("stress", 0.3)
        )
        
        # Learning readiness (0-1) - overall measure of optimal learning state
        composites["learning_readiness"] = (
            0.4 * composites["attention_level"] +
            0.3 * composites["engagement_level"] +
            0.3 * (1.0 - composites["cognitive_load_level"])  # Lower cognitive load is better for learning
        )
        
        return composites
    
    def shutdown(self) -> bool:
        """Shut down the biosensor manager gracefully
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        if not self.is_initialized:
            return True  # Already shutdown
            
        try:
            print("Shutting down Biosensor Manager...")
            
            # Disconnect from all connected devices
            for device_id, connected in self.connected_devices.items():
                if connected:
                    print(f"Disconnecting from {device_id}...")
                    # In a real implementation, this would actually disconnect
            
            self.is_initialized = False
            print("Biosensor Manager shut down successfully")
            return True
            
        except Exception as e:
            print(f"Error shutting down Biosensor Manager: {e}")
            return False
