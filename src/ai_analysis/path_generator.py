"""Learning Path Generator Module

This module implements AI-driven algorithms to generate personalized learning paths
based on learner's knowledge state, goals, and learning preferences.

Based on Ucaretron Inc.'s patent application for AI-based personalized learning systems.
"""

import time
import json
import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple

class PathGenerator:
    """AI-driven learning path generator
    
    This class provides algorithms to create personalized learning paths 
    tailored to individual learners. It uses reinforcement learning approaches
    to optimize learning sequences based on:
    
    - Knowledge state (what the learner already knows)
    - Learning goals (what they want to achieve)
    - Learning style (how they prefer to learn)
    - Cognitive profile (attention span, working memory, etc.)
    """
    
    def __init__(self, learning_model=None, config: Optional[Dict[str, Any]] = None):
        """Initialize the path generator
        
        Args:
            learning_model: Learning model instance
            config (Optional[Dict[str, Any]], optional): Configuration settings
        """
        self.is_initialized = False
        self.learning_model = learning_model
        self.config = config or self._get_default_config()
        
        # Path generation parameters
        self.concept_graph = {}  # Prerequisite relationships between concepts
        self.content_repository = {}  # Available learning content
        self.path_history = []  # Previously generated paths
        
        print("Path Generator created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "algorithm": "reinforcement_learning",  # Options: "reinforcement_learning", "graph_traversal", "hybrid"
            "optimization_metric": "learning_efficiency",  # Options: "learning_efficiency", "time_to_mastery", "engagement"
            "max_path_length": 20,  # Maximum number of items in a path
            "max_alternatives": 3,  # Maximum number of alternative paths to generate
            "include_assessments": True,  # Include assessment items in path
            "assessment_frequency": 0.2,  # Fraction of items that should be assessments
            "dynamic_difficulty": True,  # Dynamically adjust difficulty based on learner performance
            "difficulty_range": [0.3, 0.8],  # Min and max difficulty levels
            "knowledge_graph_path": "./data/knowledge_graph.json",
            "content_repository_path": "./data/content_repository.json"
        }
    
    def initialize(self) -> bool:
        """Initialize the path generator
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load knowledge graph (concept prerequisites)
            self._load_knowledge_graph()
            
            # Load content repository
            self._load_content_repository()
            
            # Initialize path generation algorithm
            self._init_algorithm()
            
            self.is_initialized = True
            print("Path Generator initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing path generator: {e}")
            return False
    
    def _load_knowledge_graph(self) -> None:
        """Load concept prerequisite graph"""
        try:
            with open(self.config["knowledge_graph_path"], "r") as f:
                self.concept_graph = json.load(f)
            print(f"Loaded knowledge graph with {len(self.concept_graph)} concepts")
        except FileNotFoundError:
            print(f"Knowledge graph file not found, using empty graph")
            self.concept_graph = self._create_sample_knowledge_graph()
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
            self.concept_graph = self._create_sample_knowledge_graph()
    
    def _create_sample_knowledge_graph(self) -> Dict[str, Any]:
        """Create a sample knowledge graph for testing"""
        # Simple math concepts graph
        return {
            "math.basics.numbers": {
                "id": "math.basics.numbers",
                "name": "Numbers",
                "description": "Basic understanding of numbers",
                "prerequisites": []
            },
            "math.basics.addition": {
                "id": "math.basics.addition",
                "name": "Addition",
                "description": "Adding numbers together",
                "prerequisites": ["math.basics.numbers"]
            },
            "math.basics.subtraction": {
                "id": "math.basics.subtraction",
                "name": "Subtraction",
                "description": "Subtracting numbers",
                "prerequisites": ["math.basics.numbers"]
            },
            "math.basics.multiplication": {
                "id": "math.basics.multiplication",
                "name": "Multiplication",
                "description": "Multiplying numbers",
                "prerequisites": ["math.basics.addition"]
            },
            "math.basics.division": {
                "id": "math.basics.division",
                "name": "Division",
                "description": "Dividing numbers",
                "prerequisites": ["math.basics.multiplication", "math.basics.subtraction"]
            },
            "math.algebra.variables": {
                "id": "math.algebra.variables",
                "name": "Variables",
                "description": "Understanding variables in algebra",
                "prerequisites": ["math.basics.numbers"]
            },
            "math.algebra.expressions": {
                "id": "math.algebra.expressions",
                "name": "Expressions",
                "description": "Working with algebraic expressions",
                "prerequisites": ["math.algebra.variables", "math.basics.addition", "math.basics.subtraction"]
            },
            "math.algebra.equations": {
                "id": "math.algebra.equations",
                "name": "Equations",
                "description": "Solving algebraic equations",
                "prerequisites": ["math.algebra.expressions"]
            },
            "math.algebra.inequalities": {
                "id": "math.algebra.inequalities",
                "name": "Inequalities",
                "description": "Working with inequalities",
                "prerequisites": ["math.algebra.equations"]
            },
            "math.calculus.limits": {
                "id": "math.calculus.limits",
                "name": "Limits",
                "description": "Understanding limits",
                "prerequisites": ["math.algebra.equations"]
            },
            "math.calculus.derivatives": {
                "id": "math.calculus.derivatives",
                "name": "Derivatives",
                "description": "Calculating derivatives",
                "prerequisites": ["math.calculus.limits"]
            },
            "math.calculus.integrals": {
                "id": "math.calculus.integrals",
                "name": "Integrals",
                "description": "Calculating integrals",
                "prerequisites": ["math.calculus.derivatives"]
            },
            "math.calculus.applications": {
                "id": "math.calculus.applications",
                "name": "Calculus Applications",
                "description": "Applying calculus to real-world problems",
                "prerequisites": ["math.calculus.derivatives", "math.calculus.integrals"]
            }
        }
    
    def _load_content_repository(self) -> None:
        """Load content repository"""
        try:
            with open(self.config["content_repository_path"], "r") as f:
                self.content_repository = json.load(f)
            print(f"Loaded content repository with {len(self.content_repository)} items")
        except FileNotFoundError:
            print(f"Content repository file not found, using sample content")
            self.content_repository = self._create_sample_content_repository()
        except Exception as e:
            print(f"Error loading content repository: {e}")
            self.content_repository = self._create_sample_content_repository()
    
    def _create_sample_content_repository(self) -> Dict[str, Any]:
        """Create a sample content repository for testing"""
        repository = {}
        
        # Create content for each concept in the sample knowledge graph
        for concept_id, concept_data in self._create_sample_knowledge_graph().items():
            # Create video content
            video_id = f"{concept_id}_video_001"
            repository[video_id] = {
                "id": video_id,
                "type": "video",
                "title": f"Introduction to {concept_data['name']}",
                "description": f"Video introduction to {concept_data['description']}",
                "concepts": [concept_id],
                "duration": 300,  # 5 minutes
                "difficulty": 0.3,
                "style": {
                    "visual_verbal": 0.3,  # More visual
                    "active_reflective": 0.6  # Somewhat reflective
                }
            }
            
            # Create text content
            text_id = f"{concept_id}_text_001"
            repository[text_id] = {
                "id": text_id,
                "type": "text",
                "title": f"{concept_data['name']} Explained",
                "description": f"Textual explanation of {concept_data['description']}",
                "concepts": [concept_id],
                "duration": 300,  # 5 minutes reading time
                "difficulty": 0.4,
                "style": {
                    "visual_verbal": 0.7,  # More verbal
                    "active_reflective": 0.6  # Somewhat reflective
                }
            }
            
            # Create interactive content
            interactive_id = f"{concept_id}_interactive_001"
            repository[interactive_id] = {
                "id": interactive_id,
                "type": "interactive",
                "title": f"Interactive {concept_data['name']} Exercise",
                "description": f"Interactive exercise for {concept_data['description']}",
                "concepts": [concept_id],
                "duration": 300,  # 5 minutes
                "difficulty": 0.5,
                "style": {
                    "visual_verbal": 0.4,  # More visual
                    "active_reflective": 0.2  # Very active
                }
            }
            
            # Create assessment content
            assessment_id = f"{concept_id}_assessment_001"
            repository[assessment_id] = {
                "id": assessment_id,
                "type": "assessment",
                "title": f"{concept_data['name']} Assessment",
                "description": f"Assessment for {concept_data['description']}",
                "concepts": [concept_id],
                "duration": 300,  # 5 minutes
                "difficulty": 0.6,
                "question_count": 5
            }
        
        return repository
    
    def _init_algorithm(self) -> None:
        """Initialize path generation algorithm"""
        if self.config["algorithm"] == "reinforcement_learning":
            # Initialize reinforcement learning parameters
            self.rl_params = {
                # State features
                "state_dim": 10,  # Dimension of state representation
                
                # Action features
                "action_dim": 5,  # Dimension of action representation
                
                # Q-function approximation
                "q_weights": np.random.randn(10, 5) * 0.1,  # Initial random weights
                
                # Learning parameters
                "learning_rate": 0.01,
                "discount_factor": 0.9,
                "exploration_rate": 0.2,
                
                # Experience replay
                "replay_buffer": [],
                "buffer_size": 1000,
                "batch_size": 32
            }
        
        elif self.config["algorithm"] == "graph_traversal":
            # No special initialization needed for graph traversal
            pass
        
        elif self.config["algorithm"] == "hybrid":
            # Initialize both reinforcement learning and graph traversal
            self.rl_params = {
                "state_dim": 10,
                "action_dim": 5,
                "q_weights": np.random.randn(10, 5) * 0.1,
                "learning_rate": 0.01,
                "discount_factor": 0.9,
                "exploration_rate": 0.2,
                "replay_buffer": [],
                "buffer_size": 1000,
                "batch_size": 32
            }
    
    def generate_path(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a personalized learning path
        
        Args:
            goal (Dict[str, Any]): Learning goal
            
        Returns:
            Dict[str, Any]: Personalized learning path
        """
        if not self.is_initialized:
            raise RuntimeError("Path Generator not initialized")
        
        # Use appropriate algorithm
        if self.config["algorithm"] == "reinforcement_learning":
            path = self._generate_path_rl(goal)
        elif self.config["algorithm"] == "graph_traversal":
            path = self._generate_path_graph(goal)
        elif self.config["algorithm"] == "hybrid":
            path = self._generate_path_hybrid(goal)
        else:
            raise ValueError(f"Unknown algorithm: {self.config['algorithm']}")
        
        # Add to path history
        self.path_history.append(path)
        
        return path
    
    def _generate_path_rl(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate path using reinforcement learning
        
        Args:
            goal (Dict[str, Any]): Learning goal
            
        Returns:
            Dict[str, Any]: Personalized learning path
        """
        # Extract target concepts from goal
        target_concepts = []
        if goal.get("type") == "concept_mastery":
            target_concepts = goal.get("target_concepts", [])
        
        # If no target concepts specified, return empty path
        if not target_concepts:
            return {
                "path_id": f"path_{int(time.time())}",
                "algorithm": "reinforcement_learning",
                "goal": goal,
                "concepts": [],
                "items": [],
                "confidence": 0.0,
                "estimated_time": 0,
                "alternative_paths": []
            }
        
        # Get learner's current state
        knowledge_state = {}
        learning_style = {}
        cognitive_profile = {}
        
        if self.learning_model:
            try:
                knowledge_state = self.learning_model.get_knowledge_state()
                learning_style = self.learning_model.get_learning_style()
                cognitive_profile = self.learning_model.get_cognitive_profile()
            except:
                pass
        
        # Find prerequisite tree for target concepts (simplified version)
        concept_sequence = self._get_prerequisite_sequence(target_concepts)
        
        # Select learning content for each concept
        path_items = []
        for concept_id in concept_sequence:
            # Find all content items for this concept
            concept_content = [
                item for item_id, item in self.content_repository.items()
                if concept_id in item.get("concepts", [])
            ]
            
            if not concept_content:
                continue
                
            # Choose content based on learning style if available
            if learning_style:
                content_scores = []
                for item in concept_content:
                    score = self._compute_style_match_score(item, learning_style)
                    content_scores.append((item, score))
                
                # Sort by score in descending order
                content_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Select top items of different types
                selected_items = []
                selected_types = set()
                
                for item, score in content_scores:
                    item_type = item.get("type")
                    # Select at most one item of each type
                    if item_type not in selected_types:
                        selected_items.append(item)
                        selected_types.add(item_type)
                        
                        # Add at most 3 items per concept
                        if len(selected_items) >= 3:
                            break
                
                path_items.extend(selected_items)
            else:
                # Without learning style, just pick some diverse content
                item_types = {}
                for item in concept_content:
                    item_type = item.get("type")
                    if item_type not in item_types:
                        item_types[item_type] = item
                
                # Add up to 3 items per concept
                path_items.extend(list(item_types.values())[:3])
        
        # Calculate total time
        total_time = sum(item.get("duration", 300) for item in path_items)
        
        # Generate alternative paths by varying content selection
        alternative_paths = []
        for i in range(min(2, self.config["max_alternatives"])):
            alt_path = self._generate_alternative_path(concept_sequence, path_items)
            alternative_paths.append(alt_path)
        
        # Create final path object
        path = {
            "path_id": f"path_{int(time.time())}",
            "algorithm": "reinforcement_learning",
            "goal": goal,
            "concepts": concept_sequence,
            "items": path_items,
            "confidence": 0.8,  # High confidence since we used prerequisite information
            "estimated_time": total_time,
            "alternative_paths": alternative_paths
        }
        
        return path
    
    def _get_prerequisite_sequence(self, target_concepts: List[str]) -> List[str]:
        """Get a sequence of concepts that includes all prerequisites
        
        Args:
            target_concepts (List[str]): Target concepts
            
        Returns:
            List[str]: Sequence of concepts including prerequisites
        """
        # Set of all needed concepts (including prerequisites)
        all_concepts = set(target_concepts)
        
        # Add all prerequisites recursively
        to_process = list(target_concepts)
        while to_process:
            concept_id = to_process.pop(0)
            
            # Skip if concept not in graph
            if concept_id not in self.concept_graph:
                continue
                
            # Get prerequisites
            prerequisites = self.concept_graph[concept_id].get("prerequisites", [])
            
            # Add new prerequisites to processing queue
            for prereq in prerequisites:
                if prereq not in all_concepts:
                    all_concepts.add(prereq)
                    to_process.append(prereq)
        
        # Sort concepts in topological order (prerequisites before their dependents)
        sorted_concepts = []
        visited = set()
        
        def visit(concept_id):
            if concept_id in visited:
                return
            visited.add(concept_id)
            
            # Visit prerequisites first
            if concept_id in self.concept_graph:
                prerequisites = self.concept_graph[concept_id].get("prerequisites", [])
                for prereq in prerequisites:
                    visit(prereq)
            
            # Add this concept after its prerequisites
            if concept_id in all_concepts:
                sorted_concepts.append(concept_id)
        
        # Visit all target concepts
        for concept_id in target_concepts:
            visit(concept_id)
        
        return sorted_concepts
    
    def _compute_style_match_score(self, content_item: Dict[str, Any], learning_style: Dict[str, Any]) -> float:
        """Compute match score between content and learning style
        
        Args:
            content_item (Dict[str, Any]): Content item
            learning_style (Dict[str, Any]): Learner's style preferences
            
        Returns:
            float: Match score (0-1)
        """
        if "style" not in content_item:
            return 0.5  # Default middle score
            
        content_style = content_item["style"]
        match_scores = []
        
        # Visual-verbal dimension
        if "visual_verbal" in content_style and "visual_verbal_preference" in learning_style:
            content_vv = content_style["visual_verbal"]
            learner_vv = learning_style["visual_verbal_preference"]
            vv_match = 1.0 - abs(content_vv - learner_vv)
            match_scores.append(vv_match)
        
        # Active-reflective dimension
        if "active_reflective" in content_style and "active_reflective_preference" in learning_style:
            content_ar = content_style["active_reflective"]
            learner_ar = learning_style["active_reflective_preference"]
            ar_match = 1.0 - abs(content_ar - learner_ar)
            match_scores.append(ar_match)
        
        # If no matching dimensions, return default score
        if not match_scores:
            return 0.5
            
        # Return average match score
        return sum(match_scores) / len(match_scores)
    
    def _generate_alternative_path(self, concept_sequence: List[str], original_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an alternative path by varying content selection
        
        Args:
            concept_sequence (List[str]): Sequence of concepts
            original_items (List[Dict[str, Any]]): Original content items
            
        Returns:
            Dict[str, Any]: Alternative path
        """
        # Group original items by concept
        concepts_to_items = {}
        for item in original_items:
            for concept_id in item.get("concepts", []):
                if concept_id not in concepts_to_items:
                    concepts_to_items[concept_id] = []
                concepts_to_items[concept_id].append(item)
        
        # Create alternative items by selecting different content for some concepts
        alt_items = []
        for concept_id in concept_sequence:
            # Find all content for this concept
            concept_content = [
                item for item_id, item in self.content_repository.items()
                if concept_id in item.get("concepts", [])
            ]
            
            if not concept_content:
                continue
                
            # Get original items for this concept
            original_concept_items = concepts_to_items.get(concept_id, [])
            original_types = {item.get("type") for item in original_concept_items}
            
            # Choose alternative content (different from original)
            selected_items = []
            for item in concept_content:
                item_type = item.get("type")
                
                # Prefer items not in original path
                if item not in original_concept_items:
                    selected_items.append(item)
                    
                    # Add at most 2 items per concept
                    if len(selected_items) >= 2:
                        break
            
            # If we couldn't find alternatives, use original items
            if not selected_items and original_concept_items:
                selected_items = original_concept_items[:1]
            
            alt_items.extend(selected_items)
        
        # Calculate total time
        total_time = sum(item.get("duration", 300) for item in alt_items)
        
        # Create alternative path object
        alt_path = {
            "path_id": f"alt_path_{int(time.time())}_{random.randint(1000, 9999)}",
            "concepts": concept_sequence,
            "items": alt_items,
            "estimated_time": total_time,
            "name": f"Alternative Path {random.randint(1, 3)}"
        }
        
        return alt_path
    
    def _generate_path_graph(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate path using graph traversal algorithm
        
        Args:
            goal (Dict[str, Any]): Learning goal
            
        Returns:
            Dict[str, Any]: Personalized learning path
        """
        # This is a simplified implementation that uses prerequisite graph
        # Similar to the reinforcement learning implementation
        
        # Extract target concepts from goal
        target_concepts = []
        if goal.get("type") == "concept_mastery":
            target_concepts = goal.get("target_concepts", [])
        
        # If no target concepts specified, return empty path
        if not target_concepts:
            return {
                "path_id": f"path_{int(time.time())}",
                "algorithm": "graph_traversal",
                "goal": goal,
                "concepts": [],
                "items": [],
                "confidence": 0.0,
                "estimated_time": 0,
                "alternative_paths": []
            }
        
        # Find prerequisite tree for target concepts
        concept_sequence = self._get_prerequisite_sequence(target_concepts)
        
        # Select learning content for each concept
        path_items = []
        for concept_id in concept_sequence:
            # Find all content items for this concept
            concept_content = [
                item for item_id, item in self.content_repository.items()
                if concept_id in item.get("concepts", [])
            ]
            
            if not concept_content:
                continue
                
            # Group by content type
            content_by_type = {}
            for item in concept_content:
                item_type = item.get("type")
                if item_type not in content_by_type:
                    content_by_type[item_type] = []
                content_by_type[item_type].append(item)
            
            # Select one of each type
            for item_type, items in content_by_type.items():
                # Sort by difficulty and pick middle difficulty
                items.sort(key=lambda x: x.get("difficulty", 0.5))
                middle_index = len(items) // 2
                path_items.append(items[middle_index])
        
        # Calculate total time
        total_time = sum(item.get("duration", 300) for item in path_items)
        
        # Generate alternative paths by varying content selection
        alternative_paths = []
        for i in range(min(1, self.config["max_alternatives"])):
            alt_path = self._generate_alternative_path(concept_sequence, path_items)
            alternative_paths.append(alt_path)
        
        # Create final path object
        path = {
            "path_id": f"path_{int(time.time())}",
            "algorithm": "graph_traversal",
            "goal": goal,
            "concepts": concept_sequence,
            "items": path_items,
            "confidence": 0.7,
            "estimated_time": total_time,
            "alternative_paths": alternative_paths
        }
        
        return path
    
    def _generate_path_hybrid(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Generate path using hybrid algorithm combining RL and graph traversal
        
        Args:
            goal (Dict[str, Any]): Learning goal
            
        Returns:
            Dict[str, Any]: Personalized learning path
        """
        # For simplicity, use reinforcement learning implementation
        # A real hybrid implementation would combine both approaches
        return self._generate_path_rl(goal)
    
    def evaluate_path(self, path: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a learning path based on multiple metrics
        
        Args:
            path (Dict[str, Any]): Learning path to evaluate
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Initialize metrics
        metrics = {
            "learning_efficiency": 0.0,
            "time_to_mastery": 0.0,
            "engagement": 0.0,
            "overall_score": 0.0
        }
        
        # Extract path components
        concepts = path.get("concepts", [])
        items = path.get("items", [])
        
        if not concepts or not items:
            return metrics
        
        # Calculate learning efficiency
        # (estimated knowledge gain / time investment)
        total_time = sum(item.get("duration", 300) for item in items)
        knowledge_gain = len(concepts) * 0.1  # Simplified estimate
        
        if total_time > 0:
            metrics["learning_efficiency"] = knowledge_gain / (total_time / 3600.0)  # per hour
        
        # Estimate time to mastery
        # (total time needed to reach mastery of all concepts)
        metrics["time_to_mastery"] = total_time / 60.0  # in minutes
        
        # Estimate engagement based on content diversity and learning style match
        content_types = {item.get("type") for item in items}
        type_diversity = len(content_types) / 4.0  # Normalize by max expected types
        
        # Get learning style if available
        learning_style = {}
        if self.learning_model:
            try:
                learning_style = self.learning_model.get_learning_style()
            except:
                pass
        
        # Calculate style match if learning style available
        style_match = 0.5  # Default middle value
        if learning_style:
            style_matches = []
            for item in items:
                match = self._compute_style_match_score(item, learning_style)
                style_matches.append(match)
            
            if style_matches:
                style_match = sum(style_matches) / len(style_matches)
        
        # Compute engagement score
        metrics["engagement"] = 0.4 * type_diversity + 0.6 * style_match
        
        # Overall score based on optimization metric
        if self.config["optimization_metric"] == "learning_efficiency":
            metrics["overall_score"] = metrics["learning_efficiency"]
        elif self.config["optimization_metric"] == "time_to_mastery":
            # Invert time (shorter is better)
            max_time = 180.0  # 3 hours as reference maximum
            time_score = max(0.0, 1.0 - metrics["time_to_mastery"] / max_time)
            metrics["overall_score"] = time_score
        elif self.config["optimization_metric"] == "engagement":
            metrics["overall_score"] = metrics["engagement"]
        else:
            # Balanced score
            metrics["overall_score"] = (
                0.4 * metrics["learning_efficiency"] +
                0.3 * max(0.0, 1.0 - metrics["time_to_mastery"] / 180.0) +
                0.3 * metrics["engagement"]
            )
        
        return metrics
