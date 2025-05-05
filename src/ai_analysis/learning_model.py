"""Personalized Learning Model Module

This module implements the core AI-based personalized learning model that
analyzes and tracks a learner's knowledge state, learning style, and cognitive profile.
"""

import time
import json
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class PersonalizedLearningModel:
    """AI-based personalized learning model
    
    This class maintains and updates a comprehensive model of the learner,
    including knowledge state, learning style, preferences, and cognitive profile.
    It uses various AI techniques to adapt the model based on learning activities.
    """
    
    def __init__(self, user_id: str):
        """Initialize the personalized learning model
        
        Args:
            user_id (str): User identifier
        """
        self.user_id = user_id
        self.is_initialized = False
        self.model_data = {}
        self.last_updated = None
        self.version = "0.1.0"
        print(f"Personalized Learning Model created for user {user_id}")
    
    def initialize_model(self, data_path: Optional[str] = None) -> bool:
        """Initialize the learning model
        
        Args:
            data_path (Optional[str], optional): Path to load model data from.
                If None, initializes with default values. Defaults to None.
                
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # If data path is provided, load data from file
            if data_path and Path(data_path).exists():
                with open(data_path, "r") as f:
                    self.model_data = json.load(f)
                    self.is_initialized = True
                    self.last_updated = time.time()
                    print(f"Loaded model data from {data_path}")
                    return True
            
            # Otherwise, initialize with default values
            self._initialize_default_model()
            self.is_initialized = True
            self.last_updated = time.time()
            print(f"Initialized default model for user {self.user_id}")
            return True
            
        except Exception as e:
            print(f"Error initializing learning model: {e}")
            return False
    
    def _initialize_default_model(self) -> None:
        """Initialize model with default values"""
        # Basic user metadata
        self.model_data = {
            "user_id": self.user_id,
            "metadata": {
                "version": self.version,
                "created_at": time.time(),
                "last_updated": time.time(),
                "model_confidence": 0.3  # Low initial confidence
            },
            
            # Knowledge state (concepts, skills, etc.)
            "knowledge_state": {
                "confidence": 0.3,  # Overall confidence in knowledge assessment
                "concepts": {},  # Will store concept mastery data
                "skills": {},    # Will store skill proficiency data
                "domains": {},   # Will store domain familiarity data
                "last_updated": time.time()
            },
            
            # Learning style preferences
            "learning_style": {
                "confidence": 0.4,  # Confidence in learning style assessment
                "visual_verbal_preference": 0.5,  # 0 = visual, 1 = verbal
                "active_reflective_preference": 0.5,  # 0 = active, 1 = reflective
                "sensory_intuitive_preference": 0.5,  # 0 = sensory, 1 = intuitive
                "sequential_global_preference": 0.5,  # 0 = sequential, 1 = global
                "social_solitary_preference": 0.5,  # 0 = social, 1 = solitary
                "adaptability": 0.5,  # How easily the learner adapts to different styles
                "learning_style_stability": 0.7,  # How stable the learning style is
                "last_updated": time.time()
            },
            
            # User preferences
            "preferences": {
                "difficulty_preference": 0.5,  # 0 = easier, 1 = harder
                "content_types": {
                    "video": 0.25,
                    "text": 0.25,
                    "interactive": 0.25,
                    "audio": 0.25,
                    "simulation": 0.25
                },
                "feedback_frequency": 0.5,  # 0 = less frequent, 1 = more frequent
                "assessment_preference": 0.5,  # 0 = formative, 1 = summative
                "last_updated": time.time()
            },
            
            # Cognitive profile
            "cognitive_profile": {
                "confidence": 0.3,  # Confidence in cognitive assessment
                "attention_span": 20,  # minutes
                "working_memory_capacity": 0.5,  # 0-1 scale
                "processing_speed": 0.5,  # 0-1 scale
                "cognitive_load_threshold": 0.7,  # 0-1 scale
                "distraction_sensitivity": 0.5,  # 0-1 scale
                "fatigue_pattern": {
                    "onset_time": 30,  # minutes until fatigue onset
                    "recovery_rate": 0.1  # recovery per minute of rest
                },
                "optimal_difficulty": 0.6,  # 0-1 scale (flow state)
                "last_updated": time.time()
            },
            
            # Learning history
            "learning_history": {
                "sessions": [],
                "achievements": [],
                "challenges": [],
                "total_learning_time": 0,  # minutes
                "average_session_length": 0,  # minutes
                "completion_rate": 0.0,  # 0-1 scale
                "last_updated": time.time()
            },
            
            # Goals and motivations
            "goals": {
                "active_goals": [],
                "completed_goals": [],
                "motivation_level": 0.7,  # 0-1 scale
                "persistence": 0.5,  # 0-1 scale
                "last_updated": time.time()
            }
        }
        
        # Add some sample concept data for demonstration
        self._add_sample_concept_data()
    
    def _add_sample_concept_data(self) -> None:
        """Add sample concept data for demonstration purposes"""
        # Sample concepts from different domains
        sample_concepts = {
            "math.algebra.linear_equations": {
                "mastery": 0.7,
                "confidence": 0.8,
                "last_practiced": time.time() - 7 * 86400,  # 7 days ago
                "strength": 0.65,
                "importance": 0.9,
                "learning_count": 5
            },
            "math.algebra.quadratic_equations": {
                "mastery": 0.4,
                "confidence": 0.6,
                "last_practiced": time.time() - 14 * 86400,  # 14 days ago
                "strength": 0.35,
                "importance": 0.8,
                "learning_count": 3
            },
            "language.grammar.parts_of_speech": {
                "mastery": 0.8,
                "confidence": 0.9,
                "last_practiced": time.time() - 2 * 86400,  # 2 days ago
                "strength": 0.75,
                "importance": 0.7,
                "learning_count": 7
            },
            "science.physics.newtons_laws": {
                "mastery": 0.5,
                "confidence": 0.5,
                "last_practiced": time.time() - 30 * 86400,  # 30 days ago
                "strength": 0.4,
                "importance": 0.8,
                "learning_count": 2
            },
            "computer_science.programming.variables": {
                "mastery": 0.9,
                "confidence": 0.95,
                "last_practiced": time.time() - 3 * 86400,  # 3 days ago
                "strength": 0.85,
                "importance": 0.9,
                "learning_count": 10
            }
        }
        
        # Add to knowledge state
        self.model_data["knowledge_state"]["concepts"] = sample_concepts
        
        # Add some sample skills data
        sample_skills = {
            "critical_thinking": {
                "proficiency": 0.65,
                "confidence": 0.7,
                "importance": 0.9,
                "development_level": "intermediate"
            },
            "problem_solving": {
                "proficiency": 0.7,
                "confidence": 0.75,
                "importance": 0.9,
                "development_level": "intermediate"
            },
            "communication": {
                "proficiency": 0.8,
                "confidence": 0.85,
                "importance": 0.8,
                "development_level": "advanced"
            }
        }
        
        # Add to knowledge state
        self.model_data["knowledge_state"]["skills"] = sample_skills
    
    def save_model(self, data_path: str) -> bool:
        """Save model data to file
        
        Args:
            data_path (str): Path to save model data to
                
        Returns:
            bool: True if save successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Ensure directory exists
            directory = Path(data_path).parent
            directory.mkdir(parents=True, exist_ok=True)
            
            # Update timestamp
            self.model_data["metadata"]["last_updated"] = time.time()
            
            # Save to file
            with open(data_path, "w") as f:
                json.dump(self.model_data, f, indent=2)
            
            print(f"Saved model data to {data_path}")
            return True
            
        except Exception as e:
            print(f"Error saving learning model: {e}")
            return False
    
    def get_knowledge_state(self) -> Dict[str, Any]:
        """Get the learner's knowledge state
        
        Returns:
            Dict[str, Any]: Knowledge state
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        return self.model_data["knowledge_state"]
    
    def get_learning_style(self) -> Dict[str, Any]:
        """Get the learner's learning style
        
        Returns:
            Dict[str, Any]: Learning style
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        return self.model_data["learning_style"]
    
    def get_preferences(self) -> Dict[str, Any]:
        """Get the learner's preferences
        
        Returns:
            Dict[str, Any]: Preferences
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        return self.model_data["preferences"]
    
    def get_cognitive_profile(self) -> Dict[str, Any]:
        """Get the learner's cognitive profile
        
        Returns:
            Dict[str, Any]: Cognitive profile
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        return self.model_data["cognitive_profile"]
    
    def get_learning_history(self) -> Dict[str, Any]:
        """Get the learner's learning history
        
        Returns:
            Dict[str, Any]: Learning history
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        return self.model_data["learning_history"]
    
    def get_goals(self) -> Dict[str, Any]:
        """Get the learner's goals
        
        Returns:
            Dict[str, Any]: Goals
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        return self.model_data["goals"]
    
    def update_knowledge_state(self, updates: Dict[str, Any]) -> bool:
        """Update the learner's knowledge state
        
        Args:
            updates (Dict[str, Any]): Updates to apply
                
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Update concepts
            if "concepts" in updates:
                for concept_id, concept_data in updates["concepts"].items():
                    # Create concept if it doesn't exist
                    if concept_id not in self.model_data["knowledge_state"]["concepts"]:
                        self.model_data["knowledge_state"]["concepts"][concept_id] = {
                            "mastery": 0.0,
                            "confidence": 0.5,
                            "last_practiced": time.time(),
                            "strength": 0.0,
                            "importance": 0.5,
                            "learning_count": 0
                        }
                    
                    # Update existing concept
                    current_concept = self.model_data["knowledge_state"]["concepts"][concept_id]
                    for key, value in concept_data.items():
                        if key in current_concept:
                            # For mastery, apply special update logic with forgetting curve
                            if key == "mastery":
                                # Calculate time-based decay
                                time_since_last_practice = time.time() - current_concept["last_practiced"]
                                days_passed = time_since_last_practice / 86400  # Convert to days
                                strength = current_concept["strength"]
                                
                                # Simple forgetting curve model: mastery * e^(-days/strength)
                                current_mastery = current_concept["mastery"] * np.exp(-days_passed / (10 * strength + 1))
                                
                                # Apply new learning with diminishing returns
                                new_mastery = current_mastery + (1 - current_mastery) * value * 0.3
                                
                                # Update mastery and related values
                                current_concept["mastery"] = new_mastery
                                current_concept["strength"] = min(1.0, strength + value * 0.1)  # Strengthen with practice
                                current_concept["learning_count"] += 1
                                current_concept["last_practiced"] = time.time()
                            else:
                                # For other properties, simple update
                                current_concept[key] = value
            
            # Update skills
            if "skills" in updates:
                for skill_id, skill_data in updates["skills"].items():
                    # Create skill if it doesn't exist
                    if skill_id not in self.model_data["knowledge_state"]["skills"]:
                        self.model_data["knowledge_state"]["skills"][skill_id] = {
                            "proficiency": 0.0,
                            "confidence": 0.5,
                            "importance": 0.5,
                            "development_level": "beginner"
                        }
                    
                    # Update existing skill
                    current_skill = self.model_data["knowledge_state"]["skills"][skill_id]
                    for key, value in skill_data.items():
                        if key in current_skill:
                            current_skill[key] = value
            
            # Update domains
            if "domains" in updates:
                for domain_id, domain_data in updates["domains"].items():
                    # Create domain if it doesn't exist
                    if domain_id not in self.model_data["knowledge_state"]["domains"]:
                        self.model_data["knowledge_state"]["domains"][domain_id] = {
                            "familiarity": 0.0,
                            "confidence": 0.5,
                            "interest": 0.5
                        }
                    
                    # Update existing domain
                    current_domain = self.model_data["knowledge_state"]["domains"][domain_id]
                    for key, value in domain_data.items():
                        if key in current_domain:
                            current_domain[key] = value
            
            # Update overall knowledge state properties
            for key, value in updates.items():
                if key not in ["concepts", "skills", "domains"] and key in self.model_data["knowledge_state"]:
                    self.model_data["knowledge_state"][key] = value
            
            # Update timestamp
            self.model_data["knowledge_state"]["last_updated"] = time.time()
            return True
            
        except Exception as e:
            print(f"Error updating knowledge state: {e}")
            return False
    
    def update_learning_style(self, updates: Dict[str, Any]) -> bool:
        """Update the learner's learning style
        
        Args:
            updates (Dict[str, Any]): Updates to apply
                
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Get current learning style
            current_style = self.model_data["learning_style"]
            stability = current_style.get("learning_style_stability", 0.7)
            
            # Apply updates with stability factor
            # Higher stability means slower changes to learning style
            for key, value in updates.items():
                if key in current_style and key not in ["last_updated", "confidence", "learning_style_stability"]:
                    # Weighted average based on stability
                    current_style[key] = current_style[key] * stability + value * (1 - stability)
            
            # Update confidence and timestamp
            if "confidence" in updates:
                # Confidence increases faster than it decreases
                if updates["confidence"] > current_style["confidence"]:
                    current_style["confidence"] = current_style["confidence"] * 0.8 + updates["confidence"] * 0.2
                else:
                    current_style["confidence"] = current_style["confidence"] * 0.9 + updates["confidence"] * 0.1
            
            current_style["last_updated"] = time.time()
            return True
            
        except Exception as e:
            print(f"Error updating learning style: {e}")
            return False
    
    def update_preferences(self, updates: Dict[str, Any]) -> bool:
        """Update the learner's preferences
        
        Args:
            updates (Dict[str, Any]): Updates to apply
                
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Handle content type preferences specially
            if "content_types" in updates:
                if "content_types" not in self.model_data["preferences"]:
                    self.model_data["preferences"]["content_types"] = {}
                
                # Update each content type preference
                for content_type, preference in updates["content_types"].items():
                    self.model_data["preferences"]["content_types"][content_type] = preference
            
            # Update other preferences
            for key, value in updates.items():
                if key != "content_types" and key != "last_updated" and key in self.model_data["preferences"]:
                    self.model_data["preferences"][key] = value
            
            # Update timestamp
            self.model_data["preferences"]["last_updated"] = time.time()
            return True
            
        except Exception as e:
            print(f"Error updating preferences: {e}")
            return False
    
    def update_cognitive_profile(self, updates: Dict[str, Any]) -> bool:
        """Update the learner's cognitive profile
        
        Args:
            updates (Dict[str, Any]): Updates to apply
                
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Get current cognitive profile
            current_profile = self.model_data["cognitive_profile"]
            
            # Apply updates with stability factor (cognitive profile changes slowly)
            stability = 0.8  # Cognitive profile is fairly stable
            
            # Handle fatigue pattern specially
            if "fatigue_pattern" in updates:
                if "fatigue_pattern" not in current_profile:
                    current_profile["fatigue_pattern"] = {}
                
                for key, value in updates["fatigue_pattern"].items():
                    if key in current_profile["fatigue_pattern"]:
                        current_profile["fatigue_pattern"][key] = (
                            current_profile["fatigue_pattern"][key] * stability + value * (1 - stability)
                        )
                    else:
                        current_profile["fatigue_pattern"][key] = value
            
            # Update other profile attributes
            for key, value in updates.items():
                if key != "fatigue_pattern" and key != "last_updated" and key in current_profile:
                    if key == "confidence":
                        # Confidence increases faster than it decreases
                        if value > current_profile["confidence"]:
                            current_profile["confidence"] = current_profile["confidence"] * 0.8 + value * 0.2
                        else:
                            current_profile["confidence"] = current_profile["confidence"] * 0.9 + value * 0.1
                    else:
                        # Other attributes change slowly
                        current_profile[key] = current_profile[key] * stability + value * (1 - stability)
            
            # Update timestamp
            current_profile["last_updated"] = time.time()
            return True
            
        except Exception as e:
            print(f"Error updating cognitive profile: {e}")
            return False
    
    def record_learning_session(self, session_data: Dict[str, Any]) -> bool:
        """Record a learning session
        
        Args:
            session_data (Dict[str, Any]): Learning session data
                
        Returns:
            bool: True if recording successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Ensure required fields
            required_fields = ["start_time", "end_time", "concepts"]
            for field in required_fields:
                if field not in session_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Calculate session duration
            duration = (session_data["end_time"] - session_data["start_time"]) / 60  # Convert to minutes
            
            # Prepare session record
            session_record = {
                "id": f"session_{int(time.time())}",
                "start_time": session_data["start_time"],
                "end_time": session_data["end_time"],
                "duration": duration,
                "concepts": session_data["concepts"],
                "skills": session_data.get("skills", []),
                "domains": session_data.get("domains", []),
                "activities": session_data.get("activities", []),
                "performance": session_data.get("performance", {}),
                "focus_level": session_data.get("focus_level", 0.5),
                "engagement_level": session_data.get("engagement_level", 0.5),
                "fatigue_level": session_data.get("fatigue_level", 0.0),
                "notes": session_data.get("notes", "")
            }
            
            # Add to sessions list
            self.model_data["learning_history"]["sessions"].append(session_record)
            
            # Update aggregate statistics
            history = self.model_data["learning_history"]
            sessions = history["sessions"]
            
            # Total learning time
            history["total_learning_time"] += duration
            
            # Average session length
            if sessions:
                history["average_session_length"] = history["total_learning_time"] / len(sessions)
            
            # Update timestamp
            history["last_updated"] = time.time()
            return True
            
        except Exception as e:
            print(f"Error recording learning session: {e}")
            return False
    
    def add_goal(self, goal_data: Dict[str, Any]) -> bool:
        """Add a learning goal
        
        Args:
            goal_data (Dict[str, Any]): Goal data
                
        Returns:
            bool: True if addition successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Ensure required fields
            required_fields = ["name", "type"]
            for field in required_fields:
                if field not in goal_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Prepare goal record
            goal_record = {
                "id": f"goal_{int(time.time())}",
                "name": goal_data["name"],
                "type": goal_data["type"],
                "description": goal_data.get("description", ""),
                "target_concepts": goal_data.get("target_concepts", []),
                "target_skills": goal_data.get("target_skills", []),
                "mastery_threshold": goal_data.get("mastery_threshold", 0.8),
                "time_constraint": goal_data.get("time_constraint", None),
                "priority": goal_data.get("priority", "medium"),
                "created_at": time.time(),
                "status": "active",
                "progress": 0.0
            }
            
            # Add to active goals list
            self.model_data["goals"]["active_goals"].append(goal_record)
            
            # Update timestamp
            self.model_data["goals"]["last_updated"] = time.time()
            return True
            
        except Exception as e:
            print(f"Error adding goal: {e}")
            return False
    
    def update_goal_progress(self, goal_id: str, progress: float, status: Optional[str] = None) -> bool:
        """Update progress on a learning goal
        
        Args:
            goal_id (str): Goal identifier
            progress (float): Progress value (0-1)
            status (Optional[str], optional): Goal status. Defaults to None.
                
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Find goal in active goals
            found = False
            for goal in self.model_data["goals"]["active_goals"]:
                if goal["id"] == goal_id:
                    # Update progress
                    goal["progress"] = progress
                    
                    # Update status if provided
                    if status:
                        goal["status"] = status
                    
                    # If completed, move to completed goals
                    if progress >= 1.0 or status == "completed":
                        goal["status"] = "completed"
                        goal["completed_at"] = time.time()
                        self.model_data["goals"]["completed_goals"].append(goal)
                        self.model_data["goals"]["active_goals"].remove(goal)
                    
                    found = True
                    break
            
            if not found:
                print(f"Goal not found: {goal_id}")
                return False
            
            # Update timestamp
            self.model_data["goals"]["last_updated"] = time.time()
            return True
            
        except Exception as e:
            print(f"Error updating goal progress: {e}")
            return False
    
    def analyze_learning_patterns(self) -> Dict[str, Any]:
        """Analyze learning patterns to extract insights
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            results = {
                "timestamp": time.time(),
                "insights": [],
                "recommendations": [],
                "strengths": [],
                "weaknesses": [],
                "learning_efficiency": {}
            }
            
            # Analyze knowledge state
            if self.model_data["knowledge_state"]["concepts"]:
                # Identify strengths (high mastery concepts)
                strengths = []
                for concept_id, concept in self.model_data["knowledge_state"]["concepts"].items():
                    if concept["mastery"] > 0.8 and concept["confidence"] > 0.7:
                        strengths.append({
                            "concept_id": concept_id,
                            "mastery": concept["mastery"],
                            "confidence": concept["confidence"]
                        })
                
                # Sort by mastery (descending)
                strengths.sort(key=lambda x: x["mastery"], reverse=True)
                results["strengths"] = strengths[:5]  # Top 5 strengths
                
                # Identify weaknesses (low mastery concepts with high importance)
                weaknesses = []
                for concept_id, concept in self.model_data["knowledge_state"]["concepts"].items():
                    if concept["mastery"] < 0.5 and concept["importance"] > 0.7:
                        weaknesses.append({
                            "concept_id": concept_id,
                            "mastery": concept["mastery"],
                            "importance": concept["importance"]
                        })
                
                # Sort by importance (descending)
                weaknesses.sort(key=lambda x: x["importance"], reverse=True)
                results["weaknesses"] = weaknesses[:5]  # Top 5 weaknesses
                
                # Calculate knowledge gaps
                knowledge_gaps = []
                for concept_id, concept in self.model_data["knowledge_state"]["concepts"].items():
                    if concept["mastery"] < 0.7 and concept["importance"] > 0.6:
                        gap = (concept["importance"] * (1 - concept["mastery"]))
                        knowledge_gaps.append({
                            "concept_id": concept_id,
                            "gap": gap,
                            "mastery": concept["mastery"],
                            "importance": concept["importance"]
                        })
                
                # Sort by gap (descending)
                knowledge_gaps.sort(key=lambda x: x["gap"], reverse=True)
                results["knowledge_gaps"] = knowledge_gaps[:5]  # Top 5 gaps
            
            # Analyze learning history
            if self.model_data["learning_history"]["sessions"]:
                sessions = self.model_data["learning_history"]["sessions"]
                
                # Calculate learning efficiency over time
                if len(sessions) >= 3:  # Need at least 3 sessions for trend analysis
                    # Sort sessions by start time
                    sorted_sessions = sorted(sessions, key=lambda x: x["start_time"])
                    
                    # Calculate efficiency metrics
                    efficiency_data = []
                    for i, session in enumerate(sorted_sessions):
                        if "performance" in session and "efficiency" in session["performance"]:
                            efficiency_data.append({
                                "session_id": session["id"],
                                "timestamp": session["start_time"],
                                "efficiency": session["performance"]["efficiency"],
                                "focus_level": session.get("focus_level", None),
                                "duration": session["duration"]
                            })
                    
                    if efficiency_data:
                        results["learning_efficiency"] = {
                            "trend": efficiency_data,
                            "average": sum(item["efficiency"] for item in efficiency_data) / len(efficiency_data)
                        }
                
                # Identify optimal learning conditions
                if len(sessions) >= 5:  # Need sufficient data for meaningful analysis
                    # Find sessions with high focus and engagement
                    high_performance_sessions = []
                    for session in sessions:
                        if session.get("focus_level", 0) > 0.7 and session.get("engagement_level", 0) > 0.7:
                            high_performance_sessions.append(session)
                    
                    if high_performance_sessions:
                        # Extract common patterns
                        time_patterns = self._extract_time_patterns(high_performance_sessions)
                        content_patterns = self._extract_content_patterns(high_performance_sessions)
                        
                        results["optimal_conditions"] = {
                            "time_patterns": time_patterns,
                            "content_patterns": content_patterns
                        }
            
            # Generate insights
            insights = []
            
            # Learning style insights
            learning_style = self.model_data["learning_style"]
            if abs(learning_style["visual_verbal_preference"] - 0.5) > 0.2:
                preference = "visual" if learning_style["visual_verbal_preference"] < 0.5 else "verbal"
                strength = abs(learning_style["visual_verbal_preference"] - 0.5) * 2
                insights.append({
                    "type": "learning_style",
                    "description": f"Strong preference for {preference} learning materials (strength: {strength:.2f})",
                    "confidence": learning_style["confidence"]
                })
            
            if abs(learning_style["active_reflective_preference"] - 0.5) > 0.2:
                preference = "active" if learning_style["active_reflective_preference"] < 0.5 else "reflective"
                strength = abs(learning_style["active_reflective_preference"] - 0.5) * 2
                insights.append({
                    "type": "learning_style",
                    "description": f"Strong preference for {preference} learning approach (strength: {strength:.2f})",
                    "confidence": learning_style["confidence"]
                })
            
            if abs(learning_style["sequential_global_preference"] - 0.5) > 0.2:
                preference = "sequential" if learning_style["sequential_global_preference"] < 0.5 else "global"
                strength = abs(learning_style["sequential_global_preference"] - 0.5) * 2
                insights.append({
                    "type": "learning_style",
                    "description": f"Strong preference for {preference} learning progression (strength: {strength:.2f})",
                    "confidence": learning_style["confidence"]
                })
            
            # Cognitive profile insights
            cognitive_profile = self.model_data["cognitive_profile"]
            if cognitive_profile["attention_span"] < 15:
                insights.append({
                    "type": "cognitive_profile",
                    "description": f"Short attention span ({cognitive_profile['attention_span']} minutes) suggests need for shorter learning sessions",
                    "confidence": cognitive_profile["confidence"]
                })
            
            if cognitive_profile["distraction_sensitivity"] > 0.7:
                insights.append({
                    "type": "cognitive_profile",
                    "description": "High sensitivity to distractions suggests need for focused learning environment",
                    "confidence": cognitive_profile["confidence"]
                })
            
            # Add insights to results
            results["insights"] = insights
            
            # Generate recommendations
            recommendations = []
            
            # Knowledge gap recommendations
            if "knowledge_gaps" in results and results["knowledge_gaps"]:
                top_gap = results["knowledge_gaps"][0]
                recommendations.append({
                    "type": "knowledge_gap",
                    "description": f"Focus on improving {top_gap['concept_id'].split('.')[-1].replace('_', ' ')} to address largest knowledge gap",
                    "confidence": 0.8,
                    "priority": "high"
                })
            
            # Learning style recommendations
            if learning_style["visual_verbal_preference"] < 0.3:
                recommendations.append({
                    "type": "content_type",
                    "description": "Increase use of visual learning materials (videos, diagrams, charts)",
                    "confidence": learning_style["confidence"],
                    "priority": "medium"
                })
            elif learning_style["visual_verbal_preference"] > 0.7:
                recommendations.append({
                    "type": "content_type",
                    "description": "Increase use of verbal learning materials (texts, discussions, audio)",
                    "confidence": learning_style["confidence"],
                    "priority": "medium"
                })
            
            # Cognitive profile recommendations
            if cognitive_profile["attention_span"] < 15:
                recommendations.append({
                    "type": "session_structure",
                    "description": "Use shorter learning sessions (10-15 minutes) with frequent breaks",
                    "confidence": cognitive_profile["confidence"],
                    "priority": "high"
                })
            
            # Add recommendations to results
            results["recommendations"] = recommendations
            
            return results
            
        except Exception as e:
            print(f"Error analyzing learning patterns: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    def _extract_time_patterns(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract time-related patterns from learning sessions
        
        Args:
            sessions (List[Dict[str, Any]]): Learning sessions
                
        Returns:
            Dict[str, Any]: Time patterns
        """
        # Initialize counters
        hour_counts = {h: 0 for h in range(24)}
        weekday_counts = {d: 0 for d in range(7)}
        duration_total = 0
        session_count = len(sessions)
        
        # Process sessions
        for session in sessions:
            # Extract hour of day (0-23)
            start_time = session["start_time"]
            hour = time.localtime(start_time).tm_hour
            hour_counts[hour] += 1
            
            # Extract day of week (0-6, Monday is 0)
            weekday = time.localtime(start_time).tm_wday
            weekday_counts[weekday] += 1
            
            # Add duration
            duration_total += session["duration"]
        
        # Find optimal time of day
        best_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
        optimal_time_start = f"{best_hour:02d}:00"
        optimal_time_end = f"{(best_hour + 1) % 24:02d}:00"
        
        # Find optimal day of week
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        best_weekday = max(weekday_counts.items(), key=lambda x: x[1])[0]
        optimal_day = weekday_names[best_weekday]
        
        # Calculate optimal duration
        optimal_duration = duration_total / session_count if session_count > 0 else 0
        
        return {
            "optimal_time": f"{optimal_time_start} - {optimal_time_end}",
            "optimal_day": optimal_day,
            "optimal_duration": optimal_duration,
            "hour_distribution": hour_counts,
            "weekday_distribution": {weekday_names[d]: count for d, count in weekday_counts.items()}
        }
    
    def _extract_content_patterns(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract content-related patterns from learning sessions
        
        Args:
            sessions (List[Dict[str, Any]]): Learning sessions
                
        Returns:
            Dict[str, Any]: Content patterns
        """
        # Initialize counters
        content_type_counts = {}
        concept_performance = {}
        
        # Process sessions
        for session in sessions:
            # Extract content types from activities
            for activity in session.get("activities", []):
                content_type = activity.get("content_type")
                if content_type:
                    if content_type not in content_type_counts:
                        content_type_counts[content_type] = 0
                    content_type_counts[content_type] += 1
            
            # Extract concept performance
            performance = session.get("performance", {})
            for concept_id in session.get("concepts", []):
                if concept_id not in concept_performance:
                    concept_performance[concept_id] = {
                        "count": 0,
                        "total_performance": 0
                    }
                
                # Add performance if available for this concept
                if concept_id in performance:
                    concept_performance[concept_id]["count"] += 1
                    concept_performance[concept_id]["total_performance"] += performance[concept_id]
        
        # Find optimal content types
        optimal_content_types = []
        if content_type_counts:
            # Sort by count (descending)
            sorted_types = sorted(content_type_counts.items(), key=lambda x: x[1], reverse=True)
            optimal_content_types = [item[0] for item in sorted_types[:3]]  # Top 3
        
        # Find best performing concepts
        best_concepts = []
        for concept_id, data in concept_performance.items():
            if data["count"] > 0:
                avg_performance = data["total_performance"] / data["count"]
                best_concepts.append({
                    "concept_id": concept_id,
                    "average_performance": avg_performance,
                    "session_count": data["count"]
                })
        
        # Sort by average performance (descending)
        best_concepts.sort(key=lambda x: x["average_performance"], reverse=True)
        
        return {
            "optimal_content_types": optimal_content_types,
            "content_type_distribution": content_type_counts,
            "best_performing_concepts": best_concepts[:5]  # Top 5
        }
    
    def predict_learning_outcomes(self, learning_path: Dict[str, Any]) -> Dict[str, Any]:
        """Predict outcomes from following a learning path
        
        Args:
            learning_path (Dict[str, Any]): Learning path
                
        Returns:
            Dict[str, Any]: Predicted outcomes
        """
        if not self.is_initialized:
            raise RuntimeError("Learning model must be initialized first")
        
        try:
            # Extract concepts from learning path
            concepts = learning_path.get("concepts", [])
            if not concepts:
                return {
                    "error": "No concepts found in learning path",
                    "timestamp": time.time()
                }
            
            # Extract current knowledge state
            knowledge_state = self.model_data["knowledge_state"]
            
            # Predict knowledge gains
            knowledge_gains = {}
            total_gain = 0.0
            concept_count = 0
            
            for concept in concepts:
                concept_id = concept["id"] if isinstance(concept, dict) else concept
                
                # Get current mastery if available
                current_mastery = 0.0
                if concept_id in knowledge_state["concepts"]:
                    current_mastery = knowledge_state["concepts"][concept_id].get("mastery", 0.0)
                
                # Estimate gain (diminishing returns with higher mastery)
                estimated_gain = (1 - current_mastery) * 0.3  # Base gain of 30% of remaining mastery
                
                # Apply modifiers based on cognitive profile and learning style
                cognitive_profile = self.model_data["cognitive_profile"]
                learning_style = self.model_data["learning_style"]
                
                # Adjust for cognitive load
                if cognitive_profile["cognitive_load_threshold"] < 0.5:
                    # Lower threshold means less effective learning under heavy load
                    estimated_gain *= 0.8  # Reduce gain
                
                # Adjust for attention span
                path_duration = learning_path.get("estimated_duration", 60)  # minutes
                attention_span = cognitive_profile["attention_span"]
                if path_duration > attention_span * 1.5:
                    # Path is much longer than attention span
                    attention_factor = attention_span / path_duration
                    estimated_gain *= (0.7 + attention_factor * 0.3)  # Reduce gain based on ratio
                
                # Adjust for learning style match
                # This is a simplified approximation - real implementation would match path characteristics
                # with learning style preferences in detail
                style_match = 0.8  # Assume 80% match by default
                estimated_gain *= style_match
                
                # Store gain for this concept
                knowledge_gains[concept_id] = {
                    "current_mastery": current_mastery,
                    "estimated_gain": estimated_gain,
                    "predicted_mastery": min(1.0, current_mastery + estimated_gain)
                }
                
                total_gain += estimated_gain
                concept_count += 1
            
            # Calculate average gain
            average_gain = total_gain / concept_count if concept_count > 0 else 0.0
            
            # Estimate completion probability
            # Based on historical completion rate and path characteristics
            base_completion_prob = self.model_data["learning_history"].get("completion_rate", 0.7)
            
            # Adjust for path difficulty and duration
            path_difficulty = learning_path.get("difficulty", 0.5)  # 0-1 scale
            path_duration = learning_path.get("estimated_duration", 60)  # minutes
            
            difficulty_factor = 1.0 - path_difficulty * 0.3  # Higher difficulty reduces completion probability
            duration_factor = 1.0
            if path_duration > 90:  # Long paths have lower completion probability
                duration_factor = 0.9
            
            # Consider motivation level
            motivation = self.model_data["goals"].get("motivation_level", 0.7)
            motivation_factor = 0.7 + motivation * 0.3  # Motivation boosts completion probability
            
            # Calculate final completion probability
            completion_probability = base_completion_prob * difficulty_factor * duration_factor * motivation_factor
            completion_probability = max(0.1, min(0.95, completion_probability))  # Constrain to reasonable range
            
            # Put together prediction results
            prediction = {
                "timestamp": time.time(),
                "path_id": learning_path.get("id", str(int(time.time()))),
                "knowledge_gains": knowledge_gains,
                "average_gain": average_gain,
                "completion_probability": completion_probability,
                "estimated_mastery_increase": average_gain,
                "confidence": 0.7  # Confidence in prediction
            }
            
            return prediction
            
        except Exception as e:
            print(f"Error predicting learning outcomes: {e}")
            return {"error": str(e), "timestamp": time.time()}

# Example usage
if __name__ == "__main__":
    # Create and initialize a learning model
    model = PersonalizedLearningModel("user_123")
    model.initialize_model()
    
    # Update learning style
    model.update_learning_style({
        "visual_verbal_preference": 0.3,  # Slight preference for visual learning
        "sequential_global_preference": 0.7  # Slight preference for global learning
    })
    
    # Print learning style
    print("\nLearning Style:")
    learning_style = model.get_learning_style()
    for key, value in learning_style.items():
        if key not in ["last_updated"]:
            print(f"  {key}: {value}")
    
    # Print knowledge state
    print("\nKnowledge State (Concepts):")
    knowledge_state = model.get_knowledge_state()
    for concept_id, concept in list(knowledge_state["concepts"].items())[:3]:  # Print first 3 concepts
        print(f"  {concept_id}:")
        for key, value in concept.items():
            print(f"    {key}: {value}")