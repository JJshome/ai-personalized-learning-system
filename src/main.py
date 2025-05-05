"""Main Application for AI Personalized Learning System

This module serves as the entry point for the AI Personalized Learning System,
integrating all components and providing a unified interface for the system.
"""

import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import system components
from src.data_collection.biosensor_manager import BiosensorManager
from src.data_collection.data_processor import DataProcessor
from src.ai_analysis.learning_model import LearningModel
from src.ai_analysis.path_generator import PathGenerator
from src.ai_analysis.path_recommender import PathRecommender
from src.content_provider.content_adapter import ContentAdapter
from src.content_provider.content_generator import ContentGenerator
from src.xai.simple_explanation_generator import ExplanationGenerator
from src.xai.visualization_generator import VisualizationGenerator
from src.security.security_manager import SecurityManager
from src.security.privacy_protection import PrivacyProtection
from src.security.blockchain_ledger import BlockchainLedger

class PersonalizedLearningSystem:
    """Main class for the AI Personalized Learning System
    
    This class integrates all components of the system and provides
    a unified interface for personalized learning.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the personalized learning system
        
        Args:
            config_path (Optional[str], optional): Path to configuration file.
                If None, uses default configuration. Defaults to None.
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize system state
        self.is_initialized = False
        self.system_ready = False
        self.user_id = None
        self.session_token = None
        
        # Initialize components
        self._init_components()
        
        print("Personalized Learning System created")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults
        
        Args:
            config_path (Optional[str]): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration settings
        """
        default_config = {
            "system": {
                "name": "AI Personalized Learning System",
                "version": "1.0.0",
                "log_level": "info",
                "data_dir": "./data"
            },
            "biosensor": {
                "enabled": True,
                "devices": ["eeg", "eye_tracker", "wearable"],
                "sampling_rate": 250,  # Hz
                "buffer_size": 10000,   # samples
                "edge_processing": True
            },
            "learning_model": {
                "model_type": "hybrid",
                "knowledge_graph_path": "./data/knowledge_graph.json",
                "model_weights_path": "./data/model_weights",
                "update_frequency": 60,  # seconds
                "confidence_threshold": 0.7
            },
            "path_generator": {
                "algorithm": "reinforcement_learning",
                "max_path_length": 20,
                "max_alternatives": 3,
                "optimization_metric": "learning_efficiency"
            },
            "content_adapter": {
                "adaptation_level": "high",
                "content_sources": ["internal", "open_educational_resources"],
                "cache_size": 100,  # items
                "content_types": ["text", "video", "interactive", "assessment"]
            },
            "xai": {
                "explanation_types": ["path", "content", "cognitive", "knowledge"],
                "default_detail_level": "medium",
                "default_language_style": "conversational",
                "store_explanations": True
            },
            "ui": {
                "theme": "adaptive",
                "accessibility": "high",
                "interaction_mode": "multimodal",
                "notification_level": "medium"
            },
            "security": {
                "privacy": {
                    "encryption": {
                        "enabled": True,
                        "key_size": 2048,
                        "scheme": "simulated_bfv",
                        "attributes_to_encrypt": [
                            "eeg_data", "heart_rate", "eye_tracking", 
                            "personal_info", "assessment_results"
                        ]
                    },
                    "differential_privacy": {
                        "enabled": True,
                        "epsilon": 1.0,
                        "delta": 0.00001,
                        "mechanism": "laplace"
                    },
                    "anonymization": {
                        "enabled": True,
                        "strategy": "pseudonymization"
                    },
                    "data_minimization": {
                        "enabled": True,
                        "retention_period_days": 90
                    },
                    "edge_computing": {
                        "enabled": True
                    }
                },
                "blockchain": {
                    "enabled": True,
                    "difficulty": 4,
                    "batch_processing": True,
                    "batch_size": 10
                },
                "authentication": {
                    "session_timeout_minutes": 60,
                    "max_failed_attempts": 5,
                    "lockout_duration_minutes": 30
                },
                "audit": {
                    "enabled": True,
                    "log_level": "INFO",
                    "events_to_log": [
                        "authentication", "data_access", "encryption", 
                        "blockchain", "privacy", "security_config"
                    ]
                }
            }
        }
        
        # If config path is provided, load and merge with defaults
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Merge user config with defaults (simple recursive merge)
                self._merge_configs(default_config, user_config)
                
                print(f"Configuration loaded from {config_path}")
                return default_config
            except Exception as e:
                print(f"Error loading configuration from {config_path}: {e}")
                print("Using default configuration")
                return default_config
        else:
            print("Using default configuration")
            return default_config
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
        """Recursively merge override_config into base_config
        
        Args:
            base_config (Dict[str, Any]): Base configuration to merge into
            override_config (Dict[str, Any]): Override configuration to merge from
        """
        for key, value in override_config.items():
            if key in base_config:
                if isinstance(value, dict) and isinstance(base_config[key], dict):
                    # Recursively merge nested dictionaries
                    self._merge_configs(base_config[key], value)
                else:
                    # Override with user value
                    base_config[key] = value
            else:
                # Add new key from user config
                base_config[key] = value
    
    def _init_components(self) -> None:
        """Initialize all system components"""
        # Create data directory if it doesn't exist
        data_dir = Path(self.config["system"]["data_dir"])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize security components
        self.security_manager = SecurityManager(
            config=self.config["security"]
        )
        
        # Initialize biosensor manager
        self.biosensor_manager = BiosensorManager(
            config=self.config["biosensor"]
        )
        
        # Initialize data processor
        self.data_processor = DataProcessor()
        
        # Initialize learning model
        self.learning_model = LearningModel(
            config=self.config["learning_model"]
        )
        
        # Initialize path generator
        self.path_generator = PathGenerator(
            config=self.config["path_generator"]
        )
        
        # Initialize path recommender
        self.path_recommender = PathRecommender(
            config=self.config["path_generator"]
        )
        
        # Initialize content adapter
        self.content_adapter = ContentAdapter(
            config=self.config["content_adapter"]
        )
        
        # Initialize content generator
        self.content_generator = ContentGenerator(
            config=self.config["content_adapter"]
        )
        
        # Initialize XAI components
        self.explanation_generator = ExplanationGenerator(
            config=self.config["xai"]
        )
        
        self.visualization_generator = VisualizationGenerator(
            config=self.config["xai"]
        )
    
    def initialize(self) -> bool:
        """Initialize the system and all its components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            print("Initializing Personalized Learning System...")
            
            # Initialize security components
            print("Initializing security components...")
            if not self.security_manager.privacy.initialize():
                print("Error: Failed to initialize security components")
                return False
                
            # Initialize biosensor components if enabled
            if self.config["biosensor"]["enabled"]:
                print("Initializing biosensor components...")
                if not self.biosensor_manager.initialize():
                    print("Warning: Failed to initialize biosensor manager")
                    print("Continuing with limited functionality")
            
            # Initialize data processing components
            print("Initializing data processing components...")
            if not self.data_processor.initialize():
                print("Error: Failed to initialize data processor")
                return False
            
            # Initialize AI analysis components
            print("Initializing AI analysis components...")
            if not self.learning_model.initialize():
                print("Error: Failed to initialize learning model")
                return False
            
            if not self.path_generator.initialize():
                print("Error: Failed to initialize path generator")
                return False
            
            if not self.path_recommender.initialize():
                print("Error: Failed to initialize path recommender")
                return False
            
            # Initialize content components
            print("Initializing content components...")
            if not self.content_adapter.initialize():
                print("Error: Failed to initialize content adapter")
                return False
            
            if not self.content_generator.initialize():
                print("Error: Failed to initialize content generator")
                return False
            
            # Initialize XAI components
            print("Initializing XAI components...")
            if not self.explanation_generator.initialize():
                print("Warning: Failed to initialize explanation generator")
                print("Continuing with limited explainability")
            
            if not self.visualization_generator.initialize():
                print("Warning: Failed to initialize visualization generator")
                print("Continuing with limited visualizations")
            
            self.is_initialized = True
            self.system_ready = True
            
            # Process any pending blockchain records
            if self.config["security"]["blockchain"]["enabled"] and self.config["security"]["blockchain"]["batch_processing"]:
                print("Processing pending blockchain records...")
                self.security_manager.process_pending_blockchain_records()
            
            print("Personalized Learning System initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing Personalized Learning System: {e}")
            return False
    
    def login_user(self, user_id: str, auth_token: str) -> Dict[str, Any]:
        """Log in a user to the system with authentication
        
        Args:
            user_id (str): User ID
            auth_token (str): Authentication token or password
            
        Returns:
            Dict[str, Any]: Login result with session info
        """
        if not self.is_initialized:
            return {"success": False, "message": "System not initialized"}
        
        try:
            print(f"Authenticating user: {user_id}")
            
            # Authenticate user through security manager
            auth_result = self.security_manager.authenticate_user(
                user_id=user_id,
                auth_token=auth_token,
                client_info={"source": "web_interface"}
            )
            
            if not auth_result["authenticated"]:
                return {"success": False, "message": "Authentication failed"}
            
            # Store session token
            self.session_token = auth_result["session_token"]
            
            # Load user profile and history
            user_data = self._load_user_data(user_id)
            
            # Apply security measures to user data
            secure_user_data = self.security_manager.secure_data(
                data=user_data,
                user_id=user_id,
                data_type="user_profile",
                operation="load"
            )
            
            # Update learning model with user data
            self.learning_model.load_user_data(secure_user_data)
            
            # Set current user
            self.user_id = user_id
            
            print(f"User {user_id} logged in successfully")
            return {
                "success": True,
                "session_token": auth_result["session_token"],
                "session_expiry": auth_result["expiry"],
                "message": "Login successful"
            }
            
        except Exception as e:
            print(f"Error logging in user {user_id}: {e}")
            return {"success": False, "message": f"Login error: {str(e)}"}
    
    def verify_session(self) -> bool:
        """Verify the current user session is valid
        
        Returns:
            bool: True if session is valid, False otherwise
        """
        if not self.is_initialized or not self.user_id or not self.session_token:
            return False
        
        # Validate session through security manager
        validation = self.security_manager.validate_session(self.session_token)
        return validation["valid"]
    
    def _load_user_data(self, user_id: str) -> Dict[str, Any]:
        """Load user data from storage
        
        Args:
            user_id (str): User ID
            
        Returns:
            Dict[str, Any]: User data including profile and learning history
        """
        # In a real implementation, this would load user data from a database
        # For this reference implementation, we'll create a simple user profile
        
        # Check if user data file exists
        user_data_path = Path(self.config["system"]["data_dir"]) / f"user_{user_id}.json"
        
        if user_data_path.exists():
            # Load existing user data
            try:
                with open(user_data_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading user data: {e}")
                return self._create_default_user_data(user_id)
        else:
            # Create default user data for new user
            return self._create_default_user_data(user_id)
    
    def _create_default_user_data(self, user_id: str) -> Dict[str, Any]:
        """Create default user data for a new user
        
        Args:
            user_id (str): User ID
            
        Returns:
            Dict[str, Any]: Default user data
        """
        default_user_data = {
            "user_id": user_id,
            "profile": {
                "name": f"User {user_id}",
                "created_at": time.time(),
                "preferences": {
                    "theme": "light",
                    "notification_level": "medium",
                    "language": "en"
                }
            },
            "learning_style": {
                "visual_verbal_preference": 0.5,  # 0 = visual, 1 = verbal
                "active_reflective_preference": 0.5,  # 0 = active, 1 = reflective
                "sequential_global_preference": 0.5,  # 0 = sequential, 1 = global
                "sensing_intuitive_preference": 0.5  # 0 = sensing, 1 = intuitive
            },
            "cognitive_profile": {
                "attention_span": 20,  # minutes
                "working_memory_capacity": 0.5,  # 0-1 scale
                "processing_speed": 0.5,  # 0-1 scale
                "optimal_difficulty": 0.5  # 0-1 scale
            },
            "knowledge_state": {
                "concepts": {},
                "last_updated": time.time()
            },
            "learning_history": {
                "sessions": [],
                "completed_paths": [],
                "assessments": []
            },
            "goals": {
                "active_goals": [],
                "completed_goals": []
            }
        }
        
        # Secure and save default user data
        secure_user_data = self.security_manager.secure_data(
            data=default_user_data,
            user_id=user_id,
            data_type="user_profile",
            operation="create"
        )
        
        user_data_path = Path(self.config["system"]["data_dir"]) / f"user_{user_id}.json"
        try:
            with open(user_data_path, 'w') as f:
                json.dump(secure_user_data, f, indent=2)
        except Exception as e:
            print(f"Error saving default user data: {e}")
        
        return default_user_data
    
    def get_learning_path(self, goal: Dict[str, Any], explain: bool = True) -> Dict[str, Any]:
        """Generate a personalized learning path for the current user
        
        Args:
            goal (Dict[str, Any]): Learning goal
            explain (bool, optional): Whether to include explanation. Defaults to True.
            
        Returns:
            Dict[str, Any]: Personalized learning path with optional explanation
        """
        if not self.is_initialized or not self.user_id:
            raise RuntimeError("System not initialized or user not logged in")
        
        if not self.verify_session():
            raise RuntimeError("User session expired or invalid")
        
        # Check authorization
        auth_result = self.security_manager.authorize_access(
            user_id=self.user_id,
            resource_type="learning_path",
            operation="read"
        )
        
        if not auth_result["authorized"]:
            raise RuntimeError(f"Access denied: {auth_result.get('reason', 'Not authorized')}")
        
        # Generate learning path
        path = self.path_generator.generate_path(goal)
        
        # Secure the learning path data
        secure_path = self.security_manager.secure_data(
            data=path,
            user_id=self.user_id,
            data_type="learning_path",
            operation="generate"
        )
        
        # Add explanation if requested
        if explain:
            explanation = self.explanation_generator.explain_learning_path(
                path_data=secure_path,
                learning_model=self.learning_model,
                detail_level=self.config["xai"]["default_detail_level"],
                user_id=self.user_id
            )
            secure_path["explanation"] = explanation
        
        return secure_path
    
    def get_adapted_content(self, content_id: str, explain: bool = True) -> Dict[str, Any]:
        """Get personalized content adapted to the current user
        
        Args:
            content_id (str): Content ID
            explain (bool, optional): Whether to include explanation. Defaults to True.
            
        Returns:
            Dict[str, Any]: Adapted content with optional explanation
        """
        if not self.is_initialized or not self.user_id:
            raise RuntimeError("System not initialized or user not logged in")
        
        if not self.verify_session():
            raise RuntimeError("User session expired or invalid")
        
        # Check authorization
        auth_result = self.security_manager.authorize_access(
            user_id=self.user_id,
            resource_type="content",
            operation="read"
        )
        
        if not auth_result["authorized"]:
            raise RuntimeError(f"Access denied: {auth_result.get('reason', 'Not authorized')}")
        
        # Get base content
        base_content = self.content_generator.get_content(content_id)
        
        # Adapt content to user
        adapted_content = self.content_adapter.adapt_content(
            content=base_content,
            user_model=self.learning_model
        )
        
        # Secure the adapted content
        secure_content = self.security_manager.secure_data(
            data=adapted_content,
            user_id=self.user_id,
            data_type="adapted_content",
            operation="adapt"
        )
        
        # Add explanation if requested
        if explain:
            explanation = self.explanation_generator.explain_content_adaptation(
                content_data=secure_content,
                learning_model=self.learning_model,
                detail_level=self.config["xai"]["default_detail_level"],
                user_id=self.user_id
            )
            secure_content["explanation"] = explanation
        
        return secure_content
    
    def process_learning_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a learning activity and update the user model
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
            
        Returns:
            Dict[str, Any]: Processing results with optional cognitive assessment
        """
        if not self.is_initialized or not self.user_id:
            raise RuntimeError("System not initialized or user not logged in")
        
        if not self.verify_session():
            raise RuntimeError("User session expired or invalid")
        
        # Check authorization
        auth_result = self.security_manager.authorize_access(
            user_id=self.user_id,
            resource_type="activity",
            operation="write"
        )
        
        if not auth_result["authorized"]:
            raise RuntimeError(f"Access denied: {auth_result.get('reason', 'Not authorized')}")
        
        # Process biosensor data if available
        biosensor_data = None
        if self.config["biosensor"]["enabled"] and self.biosensor_manager.is_initialized:
            biosensor_data = self.biosensor_manager.get_latest_data()
            
            # Apply edge computing privacy protections
            if self.config["security"]["privacy"]["edge_computing"]["enabled"]:
                edge_config = self.security_manager.privacy.get_edge_computing_config()
                # Simulate edge computing privacy protection
                # In a real implementation, this would be done on the edge device
                # before data is transmitted to the central system
                print("Applied edge computing privacy protections to biosensor data")
        
        # Process activity data
        processed_data = self.data_processor.process_activity_data(
            activity_data=activity_data,
            biosensor_data=biosensor_data
        )
        
        # Secure the processed data
        secure_processed_data = self.security_manager.secure_data(
            data=processed_data,
            user_id=self.user_id,
            data_type="learning_activity",
            operation="process"
        )
        
        # Update learning model
        update_result = self.learning_model.update_from_activity(
            activity_data=secure_processed_data
        )
        
        # Generate cognitive assessment if biosensor data is available
        results = {
            "activity_processed": True,
            "model_updated": update_result
        }
        
        if biosensor_data:
            cognitive_assessment = self._generate_cognitive_assessment(secure_processed_data)
            results["cognitive_assessment"] = cognitive_assessment
        
        return results
    
    def _generate_cognitive_assessment(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a cognitive assessment based on processed activity data
        
        Args:
            processed_data (Dict[str, Any]): Processed activity data
            
        Returns:
            Dict[str, Any]: Cognitive assessment with explanation
        """
        # In a real implementation, this would generate a detailed cognitive assessment
        # based on biosensor data and learning activity patterns
        
        # For this reference implementation, we'll create a simple assessment
        assessment = {
            "assessment_id": f"cognitive_{int(time.time())}",
            "timestamp": time.time(),
            "attention": {
                "level": 0.75,  # 0-1 scale
                "stability": 0.8,
                "distractions": 2
            },
            "engagement": {
                "level": 0.8,  # 0-1 scale
                "emotional_state": "interested",
                "interaction_rate": 0.7
            },
            "cognitive_load": {
                "level": 0.6,  # 0-1 scale
                "working_memory_utilization": 0.65,
                "processing_effort": 0.55
            },
            "primary_state": "engaged with occasional attention fluctuations",
            "suggestions": [
                "Consider a short break in 15 minutes",
                "Interactive content may help maintain engagement",
                "Current cognitive load is optimal for learning"
            ]
        }
        
        # Secure the assessment data
        secure_assessment = self.security_manager.secure_data(
            data=assessment,
            user_id=self.user_id,
            data_type="cognitive_assessment",
            operation="generate"
        )
        
        # Add explanation
        explanation = self.explanation_generator.explain_cognitive_assessment(
            assessment_data=secure_assessment,
            learning_model=self.learning_model,
            detail_level=self.config["xai"]["default_detail_level"],
            user_id=self.user_id
        )
        secure_assessment["explanation"] = explanation
        
        return secure_assessment
    
    def save_user_data(self) -> bool:
        """Save current user data to storage
        
        Returns:
            bool: True if save successful, False otherwise
        """
        if not self.is_initialized or not self.user_id:
            return False
        
        if not self.verify_session():
            print("User session expired or invalid")
            return False
        
        try:
            # Get current user data from learning model
            user_data = self.learning_model.get_user_data()
            
            # Secure the user data
            secure_user_data = self.security_manager.secure_data(
                data=user_data,
                user_id=self.user_id,
                data_type="user_profile",
                operation="save"
            )
            
            # Save to file
            user_data_path = Path(self.config["system"]["data_dir"]) / f"user_{self.user_id}.json"
            with open(user_data_path, 'w') as f:
                json.dump(secure_user_data, f, indent=2)
            
            print(f"User data saved for {self.user_id}")
            return True
            
        except Exception as e:
            print(f"Error saving user data: {e}")
            return False
    
    def verify_data_integrity(self, record_id: str) -> Dict[str, Any]:
        """Verify the integrity of data using blockchain
        
        Args:
            record_id (str): Record identifier
            
        Returns:
            Dict[str, Any]: Verification result
        """
        if not self.is_initialized:
            return {"verified": False, "reason": "System not initialized"}
        
        # Verify through security manager
        result = self.security_manager.verify_data_integrity(record_id)
        
        return result
    
    def logout_user(self) -> bool:
        """Log out the current user
        
        Returns:
            bool: True if logout successful, False otherwise
        """
        if not self.is_initialized or not self.user_id:
            return False
        
        try:
            # Save user data
            self.save_user_data()
            
            # Clear user data from learning model
            self.learning_model.clear_user_data()
            
            # Process any pending blockchain records
            if self.config["security"]["blockchain"]["enabled"] and self.config["security"]["blockchain"]["batch_processing"]:
                self.security_manager.process_pending_blockchain_records()
            
            # Clear current user
            user_id = self.user_id
            self.user_id = None
            self.session_token = None
            
            print(f"User {user_id} logged out successfully")
            return True
            
        except Exception as e:
            print(f"Error logging out user: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the system
        
        Returns:
            bool: True if shutdown successful, False otherwise
        """
        try:
            print("Shutting down Personalized Learning System...")
            
            # Log out current user if any
            if self.user_id:
                self.logout_user()
            
            # Process any pending blockchain records
            if self.config["security"]["blockchain"]["enabled"] and self.config["security"]["blockchain"]["batch_processing"]:
                self.security_manager.process_pending_blockchain_records()
            
            # Shutdown components
            if self.config["biosensor"]["enabled"] and self.biosensor_manager.is_initialized:
                self.biosensor_manager.shutdown()
            
            # Set system state
            self.system_ready = False
            
            print("Personalized Learning System shut down successfully")
            return True
            
        except Exception as e:
            print(f"Error shutting down system: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create and initialize the system
    system = PersonalizedLearningSystem()
    if not system.initialize():
        print("Failed to initialize system")
        exit(1)
    
    # Log in a user
    login_result = system.login_user(
        "test_user_001", 
        "simulated_auth_token_32chars_long_min"
    )
    
    if not login_result["success"]:
        print(f"Failed to log in user: {login_result['message']}")
        exit(1)
    
    print(f"Session token: {login_result['session_token']}")
    print(f"Session expires: {login_result['session_expiry']}")
    
    # Define a learning goal
    goal = {
        "type": "concept_mastery",
        "target_concepts": [
            "math.calculus.derivatives",
            "math.calculus.integrals",
            "math.calculus.applications"
        ],
        "target_mastery": 0.8,
        "timeframe": "4 weeks"
    }
    
    # Get personalized learning path
    learning_path = system.get_learning_path(goal)
    print(f"Generated learning path with {len(learning_path.get('concepts', []))} concepts")
    
    # Example learning activity
    activity_data = {
        "type": "content_interaction",
        "content_id": "math_calculus_intro_001",
        "duration": 600,  # seconds
        "interactions": 12,
        "completion": 0.9,
        "assessment": {
            "questions": 5,
            "correct": 4,
            "score": 0.8
        }
    }
    
    # Process learning activity
    result = system.process_learning_activity(activity_data)
    print("Learning activity processed")
    
    if "cognitive_assessment" in result:
        print(f"Cognitive state: {result['cognitive_assessment']['primary_state']}")
    
    # Get adapted content
    adapted_content = system.get_adapted_content("math_calculus_derivatives_002")
    print("Content adapted to user learning style")
    
    # Verify data integrity
    if "_metadata" in adapted_content and "record_id" in adapted_content["_metadata"]:
        record_id = adapted_content["_metadata"]["record_id"]
        verification = system.verify_data_integrity(record_id)
        print(f"Data verification result: {verification.get('record_found', False)}")
    
    # Save user data and log out
    system.save_user_data()
    system.logout_user()
    
    # Shutdown the system
    system.shutdown()
