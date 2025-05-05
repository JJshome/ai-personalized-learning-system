"""Learning Model Module

This module implements the AI-driven learning model that tracks learner knowledge state,
learning style, cognitive profile, and provides predictions for learning outcomes.

Based on Ucaretron Inc.'s patent application for AI-based personalized learning systems.
"""

import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

class LearningModel:
    """AI-driven learning model for personalized education
    
    This class maintains the learner's model including:
    - Knowledge state (concept mastery levels)
    - Learning style preferences
    - Cognitive profile (attention span, memory, etc.)
    - Learning goals
    
    It uses this information to predict learning outcomes and
    support personalized learning path recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the learning model
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings
        """
        self.is_initialized = False
        self.user_id = None
        self.config = config or self._get_default_config()
        
        # Model components
        self.knowledge_state = {"concepts": {}}
        self.learning_style = {}
        self.cognitive_profile = {}
        self.goals = {"active_goals": [], "completed_goals": []}
        
        # Training data
        self.activity_history = []
        
        print("Learning Model created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "knowledge_tracking": {
                "enabled": True,
                "decay_rate": 0.01,  # Knowledge decay per day
                "min_evidence_threshold": 3  # Minimum evidence points for confident estimation
            },
            "learning_style": {
                "enabled": True,
                "adaptation_rate": 0.2  # How quickly style estimates update with new evidence
            },
            "cognitive_profile": {
                "enabled": True,
                "update_frequency": 3  # Sessions between major updates
            },
            "model_storage": {
                "enabled": True,
                "storage_path": "./data/learning_models",
                "auto_save": True,
                "save_interval": 300  # seconds
            },
            "training": {
                "enabled": True,
                "batch_size": 32,
                "learning_rate": 0.01,
                "optimizer": "adam",
                "regularization": 0.001
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the learning model
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Create storage directory if it doesn't exist
            if self.config["model_storage"]["enabled"]:
                storage_path = Path(self.config["model_storage"]["storage_path"])
                storage_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize model parameters with random weights if training is enabled
            if self.config["training"]["enabled"]:
                self._initialize_model_parameters()
            
            self.is_initialized = True
            print("Learning Model initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing learning model: {e}")
            return False
    
    def _initialize_model_parameters(self) -> None:
        """Initialize model neural network parameters"""
        # This is a simplified version - in a real implementation this would 
        # create the actual neural network parameters
        
        # Knowledge state prediction model (for predicting concept mastery from activity data)
        self.knowledge_prediction_weights = {
            "activity_type": np.random.randn(10, 5) * 0.1,  # 10 activity types, 5 hidden units
            "duration": np.random.randn(1, 5) * 0.1,  # Duration feature
            "correctness": np.random.randn(1, 5) * 0.1,  # Correctness feature
            "difficulty": np.random.randn(1, 5) * 0.1,  # Difficulty feature
            "hidden_to_output": np.random.randn(5, 1) * 0.1  # Hidden to output
        }
        
        # Learning style prediction model
        self.style_prediction_weights = {
            "visual_verbal": np.random.randn(15, 1) * 0.1,  # 15 input features
            "active_reflective": np.random.randn(15, 1) * 0.1,
            "sequential_global": np.random.randn(15, 1) * 0.1,
            "sensing_intuitive": np.random.randn(15, 1) * 0.1
        }
        
        # Cognitive profile prediction model
        self.cognitive_prediction_weights = {
            "attention_span": np.random.randn(20, 1) * 0.1,  # 20 input features
            "working_memory": np.random.randn(20, 1) * 0.1,
            "processing_speed": np.random.randn(20, 1) * 0.1,
            "optimal_difficulty": np.random.randn(20, 1) * 0.1
        }
    
    def load_user_data(self, user_data: Dict[str, Any]) -> bool:
        """Load user data into the learning model
        
        Args:
            user_data (Dict[str, Any]): User data including profile and learning history
        
        Returns:
            bool: True if loading successful, False otherwise
        """
        try:
            # Set user ID
            self.user_id = user_data.get("user_id", None)
            if not self.user_id:
                print("Warning: User ID not provided")
                return False
            
            # Load knowledge state if available
            if "knowledge_state" in user_data:
                self.knowledge_state = user_data["knowledge_state"]
            
            # Load learning style if available
            if "learning_style" in user_data:
                self.learning_style = user_data["learning_style"]
            
            # Load cognitive profile if available
            if "cognitive_profile" in user_data:
                self.cognitive_profile = user_data["cognitive_profile"]
            
            # Load goals if available
            if "goals" in user_data:
                self.goals = user_data["goals"]
            
            # Load activity history if available
            if "learning_history" in user_data and "activities" in user_data["learning_history"]:
                self.activity_history = user_data["learning_history"]["activities"]
            
            # If we have history data and training is enabled, update model
            if self.activity_history and self.config["training"]["enabled"]:
                self._train_model(self.activity_history)
            
            print(f"User data loaded for {self.user_id}")
            return True
        except Exception as e:
            print(f"Error loading user data: {e}")
            return False
    
    def _train_model(self, activity_data: List[Dict[str, Any]]) -> None:
        """Train the learning model on historical data
        
        Args:
            activity_data (List[Dict[str, Any]]): Learning activity data
        """
        print(f"Training model on {len(activity_data)} activities...")
        
        # In a real implementation, this would train the actual neural networks
        # Here we'll simulate training with a simple progress message
        
        # Process in batches
        batch_size = self.config["training"]["batch_size"]
        num_batches = (len(activity_data) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(activity_data))
            batch = activity_data[start_idx:end_idx]
            
            # Simulate training step
            print(f"Training batch {i+1}/{num_batches}...")
            
            # Update knowledge prediction weights
            # (In a real implementation, this would use gradient descent)
            for key in self.knowledge_prediction_weights:
                noise = np.random.randn(*self.knowledge_prediction_weights[key].shape) * 0.01
                self.knowledge_prediction_weights[key] += noise
            
            # Update style prediction weights
            for key in self.style_prediction_weights:
                noise = np.random.randn(*self.style_prediction_weights[key].shape) * 0.01
                self.style_prediction_weights[key] += noise
            
            # Update cognitive prediction weights
            for key in self.cognitive_prediction_weights:
                noise = np.random.randn(*self.cognitive_prediction_weights[key].shape) * 0.01
                self.cognitive_prediction_weights[key] += noise
        
        print("Model training completed")
    
    def update_from_activity(self, activity_data: Dict[str, Any]) -> bool:
        """Update the learning model based on a new learning activity
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
            
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.is_initialized:
            print("Learning model not initialized")
            return False
        
        try:
            # Add timestamp if not present
            if "timestamp" not in activity_data:
                activity_data["timestamp"] = time.time()
            
            # Add to activity history
            self.activity_history.append(activity_data)
            
            # Extract key information
            activity_type = activity_data.get("type", "unknown")
            concepts = activity_data.get("concepts", [])
            duration = activity_data.get("duration", 0)
            
            # Update knowledge state for involved concepts
            if self.config["knowledge_tracking"]["enabled"] and concepts:
                self._update_knowledge_state(activity_data)
            
            # Update learning style based on activity data
            if self.config["learning_style"]["enabled"]:
                self._update_learning_style(activity_data)
            
            # Update cognitive profile based on activity data
            if self.config["cognitive_profile"]["enabled"]:
                self._update_cognitive_profile(activity_data)
            
            # Update goals if relevant
            self._update_goals(activity_data)
            
            # Auto-save if enabled
            if self.config["model_storage"]["enabled"] and self.config["model_storage"]["auto_save"]:
                self._auto_save()
            
            return True
        except Exception as e:
            print(f"Error updating learning model: {e}")
            return False
    
    def _update_knowledge_state(self, activity_data: Dict[str, Any]) -> None:
        """Update knowledge state based on learning activity
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
        """
        # Extract relevant data
        concepts = activity_data.get("concepts", [])
        assessment = activity_data.get("assessment", {})
        score = assessment.get("score", None)
        
        # Convert string concepts to list if needed
        if isinstance(concepts, str):
            concepts = [concepts]
        
        # Process each concept
        for concept in concepts:
            # Get concept ID - handle both string and dict formats
            concept_id = concept if isinstance(concept, str) else concept.get("id", None)
            if not concept_id:
                continue
                
            # Initialize concept if not exists
            if concept_id not in self.knowledge_state["concepts"]:
                self.knowledge_state["concepts"][concept_id] = {
                    "mastery": 0.0,
                    "evidence_count": 0,
                    "last_updated": time.time()
                }
            
            concept_data = self.knowledge_state["concepts"][concept_id]
            
            # Calculate mastery change based on activity type and performance
            mastery_change = 0.0
            
            if activity_data["type"] == "content_interaction":
                # Reading/watching content provides small mastery increase
                mastery_change = 0.05 * min(1.0, duration / 300.0)  # Cap at 5 minutes
            
            elif activity_data["type"] == "practice":
                # Practice activities provide moderate mastery increase based on correctness
                if "questions" in assessment and "correct" in assessment:
                    questions = assessment["questions"]
                    correct = assessment["correct"]
                    if questions > 0:
                        correctness = correct / questions
                        mastery_change = 0.1 * correctness
            
            elif activity_data["type"] == "assessment":
                # Assessments provide larger mastery updates based on score
                if score is not None:
                    mastery_change = 0.2 * (score - concept_data["mastery"])
            
            # Apply mastery change with adaptation rate based on evidence count
            adaptation_rate = 1.0 / (1.0 + concept_data["evidence_count"] / 5.0)
            concept_data["mastery"] = min(1.0, max(0.0, 
                concept_data["mastery"] + adaptation_rate * mastery_change))
            
            # Update evidence count and timestamp
            concept_data["evidence_count"] += 1
            concept_data["last_updated"] = time.time()
        
        # Update last_updated timestamp on knowledge state
        self.knowledge_state["last_updated"] = time.time()
    
    def _update_learning_style(self, activity_data: Dict[str, Any]) -> None:
        """Update learning style based on learning activity
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
        """
        # Only update if we have relevant data
        if activity_data.get("learning_style_evidence") is None:
            return
            
        # Extract evidence
        evidence = activity_data["learning_style_evidence"]
        
        # Update visual-verbal preference if evidence exists
        if "visual_verbal" in evidence:
            value = evidence["visual_verbal"]
            if "visual_verbal_preference" not in self.learning_style:
                self.learning_style["visual_verbal_preference"] = value
            else:
                # Blend new evidence with existing value
                current = self.learning_style["visual_verbal_preference"]
                adaptation_rate = self.config["learning_style"]["adaptation_rate"]
                self.learning_style["visual_verbal_preference"] = (
                    current * (1 - adaptation_rate) + value * adaptation_rate
                )
        
        # Update active-reflective preference if evidence exists
        if "active_reflective" in evidence:
            value = evidence["active_reflective"]
            if "active_reflective_preference" not in self.learning_style:
                self.learning_style["active_reflective_preference"] = value
            else:
                # Blend new evidence with existing value
                current = self.learning_style["active_reflective_preference"]
                adaptation_rate = self.config["learning_style"]["adaptation_rate"]
                self.learning_style["active_reflective_preference"] = (
                    current * (1 - adaptation_rate) + value * adaptation_rate
                )
        
        # Update sequential-global preference if evidence exists
        if "sequential_global" in evidence:
            value = evidence["sequential_global"]
            if "sequential_global_preference" not in self.learning_style:
                self.learning_style["sequential_global_preference"] = value
            else:
                # Blend new evidence with existing value
                current = self.learning_style["sequential_global_preference"]
                adaptation_rate = self.config["learning_style"]["adaptation_rate"]
                self.learning_style["sequential_global_preference"] = (
                    current * (1 - adaptation_rate) + value * adaptation_rate
                )
    
    def _update_cognitive_profile(self, activity_data: Dict[str, Any]) -> None:
        """Update cognitive profile based on learning activity
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
        """
        # Only update if we have relevant data
        if activity_data.get("cognitive_evidence") is None:
            return
            
        # Extract evidence
        evidence = activity_data["cognitive_evidence"]
        
        # Update attention span if evidence exists
        if "attention_span" in evidence:
            value = evidence["attention_span"]
            if "attention_span" not in self.cognitive_profile:
                self.cognitive_profile["attention_span"] = value
            else:
                # Blend new evidence with existing value
                current = self.cognitive_profile["attention_span"]
                adaptation_rate = 0.2  # Fixed adaptation rate for now
                self.cognitive_profile["attention_span"] = (
                    current * (1 - adaptation_rate) + value * adaptation_rate
                )
        
        # Update working memory capacity if evidence exists
        if "working_memory_capacity" in evidence:
            value = evidence["working_memory_capacity"]
            if "working_memory_capacity" not in self.cognitive_profile:
                self.cognitive_profile["working_memory_capacity"] = value
            else:
                # Blend new evidence with existing value
                current = self.cognitive_profile["working_memory_capacity"]
                adaptation_rate = 0.1  # Use slower adaptation for this metric
                self.cognitive_profile["working_memory_capacity"] = (
                    current * (1 - adaptation_rate) + value * adaptation_rate
                )
        
        # Update processing speed if evidence exists
        if "processing_speed" in evidence:
            value = evidence["processing_speed"]
            if "processing_speed" not in self.cognitive_profile:
                self.cognitive_profile["processing_speed"] = value
            else:
                # Blend new evidence with existing value
                current = self.cognitive_profile["processing_speed"]
                adaptation_rate = 0.1  # Use slower adaptation for this metric
                self.cognitive_profile["processing_speed"] = (
                    current * (1 - adaptation_rate) + value * adaptation_rate
                )
        
        # Update optimal difficulty if evidence exists
        if "optimal_difficulty" in evidence:
            value = evidence["optimal_difficulty"]
            if "optimal_difficulty" not in self.cognitive_profile:
                self.cognitive_profile["optimal_difficulty"] = value
            else:
                # Blend new evidence with existing value
                current = self.cognitive_profile["optimal_difficulty"]
                adaptation_rate = 0.1  # Use slower adaptation for this metric
                self.cognitive_profile["optimal_difficulty"] = (
                    current * (1 - adaptation_rate) + value * adaptation_rate
                )
    
    def _update_goals(self, activity_data: Dict[str, Any]) -> None:
        """Update learning goals based on activity data
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
        """
        # Check if activity is relevant to any active goals
        for goal in self.goals["active_goals"]:
            # Check if goal is concept mastery type
            if goal.get("type") == "concept_mastery":
                target_concepts = goal.get("target_concepts", [])
                activity_concepts = activity_data.get("concepts", [])
                
                # Convert concepts to list of IDs if they're objects
                if activity_concepts and isinstance(activity_concepts[0], dict):
                    activity_concepts = [c["id"] for c in activity_concepts if "id" in c]
                
                # Check if any target concepts are involved in this activity
                goal_concepts_involved = any(tc in activity_concepts for tc in target_concepts)
                
                if goal_concepts_involved:
                    # Update goal progress
                    if "progress" not in goal:
                        goal["progress"] = 0.0
                    
                    # Calculate progress increment based on activity type
                    if activity_data["type"] == "content_interaction":
                        goal["progress"] += 0.05
                    elif activity_data["type"] == "practice":
                        goal["progress"] += 0.1
                    elif activity_data["type"] == "assessment":
                        assessment = activity_data.get("assessment", {})
                        score = assessment.get("score", 0.5)
                        goal["progress"] += 0.2 * score
                    
                    # Cap progress at 1.0
                    goal["progress"] = min(1.0, goal["progress"])
                    
                    # Check if goal is completed
                    if goal["progress"] >= 1.0:
                        goal["completed"] = True
                        goal["completion_date"] = time.time()
                        self.goals["completed_goals"].append(goal)
                        self.goals["active_goals"].remove(goal)
    
    def _auto_save(self) -> None:
        """Automatically save the learning model if conditions are met"""
        # Check if it's time to save
        current_time = time.time()
        last_save = getattr(self, "last_save_time", 0)
        save_interval = self.config["model_storage"]["save_interval"]
        
        if current_time - last_save >= save_interval:
            # Generate filename
            storage_path = Path(self.config["model_storage"]["storage_path"])
            filename = f"{self.user_id}_learning_model_{int(current_time)}.json"
            file_path = storage_path / filename
            
            # Get model data
            model_data = self.get_user_data()
            
            # Save to file
            try:
                with open(file_path, "w") as f:
                    json.dump(model_data, f, indent=2)
                self.last_save_time = current_time
                print(f"Learning model auto-saved to {filename}")
            except Exception as e:
                print(f"Error auto-saving learning model: {e}")
    
    def predict_learning_outcome(self, activity_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcome of a planned learning activity
        
        Args:
            activity_plan (Dict[str, Any]): Planned learning activity
            
        Returns:
            Dict[str, Any]: Predicted outcomes
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model not initialized")
        
        # Extract key information
        activity_type = activity_plan.get("type", "unknown")
        concepts = activity_plan.get("concepts", [])
        difficulty = activity_plan.get("difficulty", 0.5)
        duration = activity_plan.get("duration", 300)  # Default to 5 minutes
        
        # Convert string concepts to list if needed
        if isinstance(concepts, str):
            concepts = [concepts]
        
        # Initialize predictions
        predictions = {
            "engagement": 0.7,  # Default engagement prediction
            "knowledge_gain": {},
            "completion_probability": 0.9,
            "time_required": duration,
            "match_score": 0.0
        }
        
        # Predict knowledge gain for each concept
        for concept in concepts:
            # Get concept ID - handle both string and dict formats
            concept_id = concept if isinstance(concept, str) else concept.get("id", None)
            if not concept_id:
                continue
            
            # Get current mastery level if available
            current_mastery = 0.0
            if concept_id in self.knowledge_state["concepts"]:
                current_mastery = self.knowledge_state["concepts"][concept_id]["mastery"]
            
            # Calculate knowledge gain based on activity type and difficulty
            if activity_type == "content_interaction":
                # Reading/watching content provides modest gains
                knowledge_gain = 0.05 * (1.0 - current_mastery)
            
            elif activity_type == "practice":
                # Practice gives higher gains, especially at optimal difficulty
                optimal_difficulty = self.cognitive_profile.get("optimal_difficulty", 0.5)
                difficulty_match = 1.0 - abs(difficulty - optimal_difficulty)
                knowledge_gain = 0.1 * difficulty_match * (1.0 - current_mastery)
            
            elif activity_type == "assessment":
                # Assessments provide little direct gain
                knowledge_gain = 0.02 * (1.0 - current_mastery)
            
            else:
                # Default modest gain
                knowledge_gain = 0.03 * (1.0 - current_mastery)
            
            # Store prediction
            predictions["knowledge_gain"][concept_id] = knowledge_gain
        
        # Predict engagement based on learning style match
        if "content_style" in activity_plan:
            content_style = activity_plan["content_style"]
            
            # Calculate style match scores
            style_match_scores = []
            
            # Visual-verbal match
            if "visual_verbal" in content_style and "visual_verbal_preference" in self.learning_style:
                content_vv = content_style["visual_verbal"]
                user_vv = self.learning_style["visual_verbal_preference"]
                vv_match = 1.0 - abs(content_vv - user_vv)
                style_match_scores.append(vv_match)
            
            # Active-reflective match
            if "active_reflective" in content_style and "active_reflective_preference" in self.learning_style:
                content_ar = content_style["active_reflective"]
                user_ar = self.learning_style["active_reflective_preference"]
                ar_match = 1.0 - abs(content_ar - user_ar)
                style_match_scores.append(ar_match)
            
            # Calculate overall style match
            if style_match_scores:
                style_match = sum(style_match_scores) / len(style_match_scores)
                predictions["match_score"] = style_match
                
                # Adjust engagement prediction based on style match
                predictions["engagement"] = 0.4 + 0.6 * style_match
        
        # Predict completion probability based on difficulty and cognitive profile
        optimal_difficulty = self.cognitive_profile.get("optimal_difficulty", 0.5)
        attention_span = self.cognitive_profile.get("attention_span", 20.0)  # minutes
        
        # Adjust for difficulty match
        difficulty_match = 1.0 - abs(difficulty - optimal_difficulty)
        predictions["completion_probability"] *= difficulty_match
        
        # Adjust for duration vs. attention span
        duration_minutes = duration / 60.0
        if duration_minutes > attention_span:
            # Reduce probability for activities longer than attention span
            attention_factor = attention_span / duration_minutes
            predictions["completion_probability"] *= (0.7 + 0.3 * attention_factor)
        
        # Adjust time required based on processing speed
        processing_speed = self.cognitive_profile.get("processing_speed", 0.5)
        time_adjustment = 1.0 / (0.5 + processing_speed)  # Higher speed means lower time
        predictions["time_required"] = int(duration * time_adjustment)
        
        return predictions
    
    def get_knowledge_state(self) -> Dict[str, Any]:
        """Get the current knowledge state
        
        Returns:
            Dict[str, Any]: Knowledge state including concept mastery levels
        """
        return self.knowledge_state
    
    def get_learning_style(self) -> Dict[str, Any]:
        """Get the current learning style profile
        
        Returns:
            Dict[str, Any]: Learning style profile
        """
        return self.learning_style
    
    def get_cognitive_profile(self) -> Dict[str, Any]:
        """Get the current cognitive profile
        
        Returns:
            Dict[str, Any]: Cognitive profile
        """
        return self.cognitive_profile
    
    def get_goals(self) -> Dict[str, Any]:
        """Get the current learning goals
        
        Returns:
            Dict[str, Any]: Learning goals
        """
        return self.goals
    
    def get_user_data(self) -> Dict[str, Any]:
        """Get all user data from the learning model
        
        Returns:
            Dict[str, Any]: Complete user data
        """
        return {
            "user_id": self.user_id,
            "knowledge_state": self.knowledge_state,
            "learning_style": self.learning_style,
            "cognitive_profile": self.cognitive_profile,
            "goals": self.goals,
            "learning_history": {
                "activities": self.activity_history
            },
            "model_version": "1.0.0",
            "timestamp": time.time()
        }
    
    def clear_user_data(self) -> None:
        """Clear all user data from the learning model"""
        self.user_id = None
        self.knowledge_state = {"concepts": {}}
        self.learning_style = {}
        self.cognitive_profile = {}
        self.goals = {"active_goals": [], "completed_goals": []}
        self.activity_history = []
        print("User data cleared")
