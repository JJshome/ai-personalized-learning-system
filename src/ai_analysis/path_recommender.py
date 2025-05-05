"""Learning Path Recommender Module

This module generates personalized learning path recommendations based on
the learner's knowledge state, learning style, and goals.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class LearningPathRecommender:
    """Recommends personalized learning paths
    
    This class analyzes the learner's model and generates optimal learning paths
    tailored to their knowledge state, learning style, preferences, and goals.
    """
    
    def __init__(self, learning_model: Any):
        """Initialize the learning path recommender
        
        Args:
            learning_model (Any): Personalized learning model instance
        """
        self.learning_model = learning_model
        self.is_initialized = False
        self.content_repository = None
        self.concept_graph = None
        self.recommendation_parameters = {}
        print("Learning Path Recommender created")
    
    def initialize(self, content_repository: Any = None, concept_graph: Any = None) -> bool:
        """Initialize the path recommender
        
        Args:
            content_repository (Any, optional): Content repository instance.
                If None, uses an internal simulated repository. Defaults to None.
            concept_graph (Any, optional): Concept relationship graph instance.
                If None, uses an internal simulated graph. Defaults to None.
                
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Store repository and concept graph
            self.content_repository = content_repository
            self.concept_graph = concept_graph
            
            # Initialize recommendation parameters
            self.recommendation_parameters = {
                "algorithm": "reinforcement_learning",  # "reinforcement_learning", "bayesian", "hybrid"
                "optimization_goal": "balanced",  # "speed", "mastery", "engagement", "balanced"
                "max_path_length": 10,  # Maximum number of items in a path
                "min_confidence": 0.6,  # Minimum confidence for recommendations
                "difficulty_adjustment": 0.1,  # How much to adjust based on difficulty preference
                "learning_style_weight": 0.7,  # Weight of learning style in content selection
                "knowledge_state_weight": 0.9,  # Weight of knowledge state in concept selection
                "engagement_weight": 0.5,  # Weight of predicted engagement in selection
                "last_updated": time.time()
            }
            
            self.is_initialized = True
            print("Learning Path Recommender initialized successfully")
            return True
        
        except Exception as e:
            print(f"Error initializing Learning Path Recommender: {e}")
            return False
    
    def recommend_learning_path(
        self, 
        user_id: str, 
        goal: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Recommend a personalized learning path
        
        Args:
            user_id (str): User ID
            goal (Optional[Dict[str, Any]], optional): Learning goal specification.
                If None, infers goal from user model. Defaults to None.
                
        Returns:
            Dict[str, Any]: Recommended learning path
        """
        if not self.is_initialized:
            raise RuntimeError("Learning Path Recommender must be initialized first")
        
        try:
            # Get learner's current state
            knowledge_state = self.learning_model.get_knowledge_state()
            learning_style = self.learning_model.get_learning_style()
            preferences = self.learning_model.get_preferences()
            cognitive_profile = self.learning_model.get_cognitive_profile()
            
            # Process goal or infer if not provided
            processed_goal = self._process_goal(goal, knowledge_state)
            
            # Identify target concepts based on goal
            target_concepts = self._identify_target_concepts(processed_goal, knowledge_state)
            
            # Determine prerequisite concepts that need reinforcement
            prerequisite_concepts = self._identify_prerequisites(target_concepts, knowledge_state)
            
            # Generate optimal path through concepts based on algorithm
            concept_path = self._generate_concept_path(
                target_concepts, prerequisite_concepts, knowledge_state, learning_style, preferences
            )
            
            # Select optimal content for each concept
            path_items = self._select_content_for_concepts(
                concept_path, learning_style, preferences, cognitive_profile
            )
            
            # Estimate knowledge gains
            expected_knowledge_gain = self._estimate_knowledge_gain(path_items, knowledge_state)
            
            # Calculate confidence in recommendation
            confidence = self._calculate_recommendation_confidence(
                concept_path, path_items, knowledge_state, learning_style
            )
            
            # Generate alternative paths if confidence is below threshold
            alternative_paths = []
            if confidence < self.recommendation_parameters["min_confidence"]:
                alternative_paths = self._generate_alternative_paths(
                    processed_goal, knowledge_state, learning_style, preferences, cognitive_profile
                )
            
            # Compile final path recommendation
            learning_path = {
                "user_id": user_id,
                "goal": processed_goal,
                "concepts": concept_path,
                "items": path_items,
                "estimated_duration": sum(item.get("duration", 20) for item in path_items),  # minutes
                "difficulty": self._calculate_path_difficulty(path_items),  # 0-1 scale
                "expected_knowledge_gain": expected_knowledge_gain,
                "confidence": confidence,
                "alternative_paths": alternative_paths,
                "optimization_goal": self.recommendation_parameters["optimization_goal"],
                "algorithm": self.recommendation_parameters["algorithm"],
                "timestamp": time.time()
            }
            
            return learning_path
        
        except Exception as e:
            print(f"Error recommending learning path: {e}")
            return {
                "user_id": user_id,
                "error": str(e),
                "goal": goal,
                "timestamp": time.time()
            }
    
    def _process_goal(self, goal: Optional[Dict[str, Any]], knowledge_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process or infer learning goal
        
        Args:
            goal (Optional[Dict[str, Any]]): User-specified goal or None
            knowledge_state (Dict[str, Any]): Current knowledge state
                
        Returns:
            Dict[str, Any]: Processed goal
        """
        # If goal is provided, validate and process it
        if goal:
            processed_goal = {
                "type": goal.get("type", "concept_mastery"),
                "target_concepts": goal.get("target_concepts", []),
                "target_skills": goal.get("target_skills", []),
                "mastery_threshold": goal.get("mastery_threshold", 0.8),
                "time_constraint": goal.get("time_constraint", None),  # minutes or None
                "priority": goal.get("priority", "balanced")  # "speed", "depth", "balanced"
            }
            
            return processed_goal
        
        # If goal is not provided, infer from knowledge state
        # In a real implementation, this would use a sophisticated algorithm
        # to identify gaps and opportunities in the knowledge state
        
        # For demonstration, identify concepts with low mastery
        low_mastery_concepts = []
        for concept_id, concept_data in knowledge_state["concepts"].items():
            if concept_data.get("mastery", 0) < 0.5 and concept_data.get("confidence", 0) > 0.6:
                low_mastery_concepts.append(concept_id)
        
        # Prioritize a few concepts (in a real system, this would be more sophisticated)
        target_concepts = low_mastery_concepts[:3] if low_mastery_concepts else []
        
        # If still no target concepts, pick random ones for demonstration
        if not target_concepts and knowledge_state["concepts"]:
            concepts = list(knowledge_state["concepts"].keys())
            target_concepts = [concepts[0]] if concepts else ["math.algebra.quadratic_equations"]
        elif not target_concepts:
            target_concepts = ["math.algebra.quadratic_equations"]
        
        inferred_goal = {
            "type": "concept_mastery",
            "target_concepts": target_concepts,
            "target_skills": [],
            "mastery_threshold": 0.8,
            "time_constraint": None,
            "priority": "balanced",
            "inferred": True
        }
        
        return inferred_goal
    
    def _identify_target_concepts(self, goal: Dict[str, Any], knowledge_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify target concepts based on goal
        
        Args:
            goal (Dict[str, Any]): Learning goal
            knowledge_state (Dict[str, Any]): Current knowledge state
                
        Returns:
            List[Dict[str, Any]]: Target concepts with metadata
        """
        target_concepts = []
        
        # Process explicit target concepts from goal
        for concept_id in goal.get("target_concepts", []):
            # Get current mastery if available
            current_mastery = 0.0
            current_confidence = 0.5
            if concept_id in knowledge_state["concepts"]:
                current_mastery = knowledge_state["concepts"][concept_id].get("mastery", 0.0)
                current_confidence = knowledge_state["concepts"][concept_id].get("confidence", 0.5)
            
            # Calculate mastery gap
            mastery_threshold = goal.get("mastery_threshold", 0.8)
            mastery_gap = max(0, mastery_threshold - current_mastery)
            
            # Only include concept if there's a mastery gap
            if mastery_gap > 0.1 or current_confidence < 0.7:
                target_concepts.append({
                    "id": concept_id,
                    "current_mastery": current_mastery,
                    "target_mastery": mastery_threshold,
                    "mastery_gap": mastery_gap,
                    "confidence": current_confidence,
                    "importance": 1.0  # Primary goal concepts have maximum importance
                })
        
        # If goal is skill-based, identify concepts that contribute to those skills
        if goal.get("type") == "skill_development" and goal.get("target_skills"):
            # In a real implementation, this would use a concept-skill mapping
            # For demonstration, using dummy connections
            skill_concept_mapping = {
                "problem_solving": ["math.algebra.word_problems", "logic.critical_thinking"],
                "communication": ["language.grammar", "language.composition"],
                "creativity": ["arts.design", "language.creative_writing"],
                "critical_thinking": ["logic.argumentation", "logic.fallacies"],
                "data_analysis": ["math.statistics", "computer_science.data_science"]
            }
            
            for skill_id in goal.get("target_skills", []):
                if skill_id in skill_concept_mapping:
                    for concept_id in skill_concept_mapping[skill_id]:
                        # Check if concept already in target list
                        if not any(c["id"] == concept_id for c in target_concepts):
                            # Get current mastery if available
                            current_mastery = 0.0
                            current_confidence = 0.5
                            if concept_id in knowledge_state["concepts"]:
                                current_mastery = knowledge_state["concepts"][concept_id].get("mastery", 0.0)
                                current_confidence = knowledge_state["concepts"][concept_id].get("confidence", 0.5)
                            
                            # Calculate mastery gap
                            mastery_threshold = goal.get("mastery_threshold", 0.8)
                            mastery_gap = max(0, mastery_threshold - current_mastery)
                            
                            # Only include concept if there's a mastery gap
                            if mastery_gap > 0.1 or current_confidence < 0.7:
                                target_concepts.append({
                                    "id": concept_id,
                                    "current_mastery": current_mastery,
                                    "target_mastery": mastery_threshold,
                                    "mastery_gap": mastery_gap,
                                    "confidence": current_confidence,
                                    "importance": 0.8,  # Skill-supporting concepts have high importance
                                    "supports_skill": skill_id
                                })
        
        # Sort by importance and mastery gap (prioritize important concepts with larger gaps)
        target_concepts.sort(key=lambda c: (c["importance"], c["mastery_gap"]), reverse=True)
        
        return target_concepts
    
    def _identify_prerequisites(self, target_concepts: List[Dict[str, Any]], knowledge_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify prerequisite concepts that need reinforcement
        
        Args:
            target_concepts (List[Dict[str, Any]]): Target concepts
            knowledge_state (Dict[str, Any]): Current knowledge state
                
        Returns:
            List[Dict[str, Any]]: Prerequisite concepts with metadata
        """
        prerequisite_concepts = []
        
        # In a real implementation, this would use a concept graph to identify prerequisites
        # For demonstration, using a simplified relationship map
        prerequisite_map = {
            "math.algebra.quadratic_equations": [
                "math.algebra.linear_equations",
                "math.algebra.factoring"
            ],
            "math.calculus.derivatives": [
                "math.algebra.functions",
                "math.calculus.limits"
            ],
            "language.composition": [
                "language.grammar",
                "language.vocabulary"
            ],
            "computer_science.programming.python": [
                "computer_science.programming.basics",
                "computer_science.algorithms.basics"
            ]
        }
        
        # Check prerequisites for each target concept
        for target in target_concepts:
            concept_id = target["id"]
            prerequisites = prerequisite_map.get(concept_id, [])
            
            for prereq_id in prerequisites:
                # Get current mastery if available
                current_mastery = 0.0
                current_confidence = 0.5
                if prereq_id in knowledge_state["concepts"]:
                    current_mastery = knowledge_state["concepts"][prereq_id].get("mastery", 0.0)
                    current_confidence = knowledge_state["concepts"][prereq_id].get("confidence", 0.5)
                
                # Only include prerequisite if mastery is insufficient
                threshold = 0.7  # Prerequisites need good mastery but not necessarily target threshold
                if current_mastery < threshold or current_confidence < 0.6:
                    # Check if prerequisite already in list
                    if not any(p["id"] == prereq_id for p in prerequisite_concepts):
                        prerequisite_concepts.append({
                            "id": prereq_id,
                            "current_mastery": current_mastery,
                            "target_mastery": threshold,
                            "mastery_gap": max(0, threshold - current_mastery),
                            "confidence": current_confidence,
                            "importance": 0.9,  # Prerequisites have high importance
                            "prereq_for": concept_id
                        })
        
        # Sort by mastery gap (prioritize concepts with larger gaps)
        prerequisite_concepts.sort(key=lambda c: c["mastery_gap"], reverse=True)
        
        return prerequisite_concepts
    
    def _generate_concept_path(
        self,
        target_concepts: List[Dict[str, Any]],
        prerequisite_concepts: List[Dict[str, Any]],
        knowledge_state: Dict[str, Any],
        learning_style: Dict[str, Any],
        preferences: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate optimal path through concepts
        
        Args:
            target_concepts (List[Dict[str, Any]]): Target concepts
            prerequisite_concepts (List[Dict[str, Any]]): Prerequisite concepts
            knowledge_state (Dict[str, Any]): Current knowledge state
            learning_style (Dict[str, Any]): Learning style
            preferences (Dict[str, Any]): User preferences
                
        Returns:
            List[Dict[str, Any]]: Ordered concept path
        """
        # In a real implementation, this would use the specified algorithm
        # (reinforcement learning, Bayesian optimization, etc.) to find an optimal path
        # For demonstration, using a simplified sequencing approach
        
        # Combine all concepts
        all_concepts = prerequisite_concepts + target_concepts
        
        # Remove duplicates (keep the first occurrence, which will be from prerequisites if present)
        unique_concepts = []
        concept_ids = set()
        for concept in all_concepts:
            if concept["id"] not in concept_ids:
                unique_concepts.append(concept)
                concept_ids.add(concept["id"])
        
        # Build a simple dependency graph
        dependency_graph = {}
        for concept in unique_concepts:
            dependency_graph[concept["id"]] = []
        
        # Add prerequisite relationships
        for prereq in prerequisite_concepts:
            if "prereq_for" in prereq and prereq["prereq_for"] in dependency_graph:
                dependency_graph[prereq["prereq_for"]].append(prereq["id"])
        
        # Determine sequential vs. global learning style preference
        sequential_preference = learning_style.get("sequential_global_preference", 0.5) < 0.5
        
        # Generate sequence based on learning style
        if sequential_preference:
            # Sequential learners prefer a linear path through the concepts
            # Use topological sort to respect prerequisites
            path = self._topological_sort(unique_concepts, dependency_graph)
        else:
            # Global learners prefer getting the big picture first, then details
            # Start with high-level target concepts, then fill in prerequisites as needed
            path = self._global_concept_sequencing(unique_concepts, dependency_graph)
        
        # Limit path length based on recommendation parameters
        max_length = self.recommendation_parameters["max_path_length"]
        if len(path) > max_length:
            # Prioritize concepts with larger mastery gaps
            path.sort(key=lambda c: c["mastery_gap"], reverse=True)
            path = path[:max_length]
        
        return path
    
    def _topological_sort(self, concepts: List[Dict[str, Any]], dependency_graph: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Perform topological sort of concepts based on prerequisites
        
        Args:
            concepts (List[Dict[str, Any]]): Concepts to sort
            dependency_graph (Dict[str, List[str]]): Dependency graph
                
        Returns:
            List[Dict[str, Any]]: Topologically sorted concepts
        """
        # Create a map of concept_id to concept object
        concept_map = {concept["id"]: concept for concept in concepts}
        
        # Create adjacency list representation of dependency graph
        # Note: graph edges point from a concept to its prerequisites
        graph = {}
        for concept_id, prereqs in dependency_graph.items():
            graph[concept_id] = prereqs
        
        # Initialize variables for topological sort
        visited = set()
        temp_mark = set()
        ordered = []
        
        # Helper function for depth-first search
        def visit(node):
            if node in temp_mark:
                # Cyclic dependency, break it (in a real system, this would be handled better)
                return
            if node not in visited:
                temp_mark.add(node)
                for prereq in graph.get(node, []):
                    visit(prereq)
                temp_mark.remove(node)
                visited.add(node)
                ordered.append(node)
        
        # Visit each node
        for concept_id in graph:
            if concept_id not in visited:
                visit(concept_id)
        
        # Convert ordered concept IDs back to concept objects
        # Note: The list is in reverse topological order (prerequisites first)
        ordered_concepts = []
        for concept_id in ordered:
            if concept_id in concept_map:
                ordered_concepts.append(concept_map[concept_id])
        
        return ordered_concepts
    
    def _global_concept_sequencing(self, concepts: List[Dict[str, Any]], dependency_graph: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate concept sequence for global learners
        
        Args:
            concepts (List[Dict[str, Any]]): Concepts to sequence
            dependency_graph (Dict[str, List[str]]): Dependency graph
                
        Returns:
            List[Dict[str, Any]]: Sequenced concepts
        """
        # Create a map of concept_id to concept object
        concept_map = {concept["id"]: concept for concept in concepts}
        
        # Separate target concepts from prerequisites
        targets = [c for c in concepts if not c.get("prereq_for")]
        prereqs = [c for c in concepts if c.get("prereq_for")]
        
        # For global learners, start with a high-level overview of all target concepts
        # Then add prerequisites as needed
        sequenced = []
        
        # Add overview concepts first (those with high importance)
        # Sort targets by importance (descending) and mastery gap (descending)
        targets.sort(key=lambda c: (c.get("importance", 0), c.get("mastery_gap", 0)), reverse=True)
        sequenced.extend(targets)
        
        # Now add prerequisites that aren't already included
        added_ids = {c["id"] for c in sequenced}
        missing_prereqs = []
        
        # Find missing prerequisites for each target
        for target in targets:
            target_prereqs = dependency_graph.get(target["id"], [])
            for prereq_id in target_prereqs:
                if prereq_id not in added_ids and prereq_id in concept_map:
                    missing_prereqs.append(concept_map[prereq_id])
                    added_ids.add(prereq_id)
        
        # Sort prerequisites by mastery gap (descending)
        missing_prereqs.sort(key=lambda c: c.get("mastery_gap", 0), reverse=True)
        
        # Add missing prerequisites
        sequenced.extend(missing_prereqs)
        
        return sequenced
    
    def _select_content_for_concepts(
        self,
        concept_path: List[Dict[str, Any]],
        learning_style: Dict[str, Any],
        preferences: Dict[str, Any],
        cognitive_profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Select optimal content for each concept in the path
        
        Args:
            concept_path (List[Dict[str, Any]]): Ordered concept path
            learning_style (Dict[str, Any]): Learning style
            preferences (Dict[str, Any]): User preferences
            cognitive_profile (Dict[str, Any]): Cognitive profile
                
        Returns:
            List[Dict[str, Any]]: Path items with content details
        """
        # In a real implementation, this would query a content repository
        # For demonstration, generating simulated content items
        
        path_items = []
        
        # Extract learning style preferences
        visual_verbal = learning_style.get("visual_verbal_preference", 0.5)  # 0 = visual, 1 = verbal
        active_reflective = learning_style.get("active_reflective_preference", 0.5)  # 0 = active, 1 = reflective
        sensory_intuitive = learning_style.get("sensory_intuitive_preference", 0.5)  # 0 = sensory, 1 = intuitive
        
        # Extract content type preferences
        content_preferences = preferences.get("content_types", {
            "video": 0.25,
            "text": 0.25,
            "interactive": 0.25,
            "audio": 0.25
        })
        
        # Extract difficulty preference
        difficulty_preference = preferences.get("difficulty_preference", 0.5)  # 0 = easier, 1 = harder
        
        # Process each concept in the path
        for i, concept in enumerate(concept_path):
            concept_id = concept["id"]
            mastery_gap = concept.get("mastery_gap", 0.5)
            
            # Determine optimal content type based on learning style and preferences
            # Visual learners prefer videos and visualizations
            # Verbal learners prefer text and audio
            # Active learners prefer interactive content
            # Reflective learners prefer text and videos that allow reflection
            # Sensory learners prefer concrete examples and practical content
            # Intuitive learners prefer theoretical content and concepts
            
            # Calculate content type scores based on learning style and preferences
            content_scores = {
                "video": (1 - visual_verbal) * 0.8 + content_preferences.get("video", 0.25) * 0.2,
                "text": visual_verbal * 0.6 + active_reflective * 0.4 + content_preferences.get("text", 0.25) * 0.2,
                "interactive": (1 - active_reflective) * 0.8 + content_preferences.get("interactive", 0.25) * 0.2,
                "audio": visual_verbal * 0.7 + content_preferences.get("audio", 0.25) * 0.3,
                "simulation": (1 - sensory_intuitive) * 0.6 + (1 - active_reflective) * 0.4
            }
            
            # Determine optimal content type
            best_content_type = max(content_scores.items(), key=lambda x: x[1])[0]
            
            # Determine appropriate difficulty level
            # Base on difficulty preference and mastery gap
            base_difficulty = 0.3 + mastery_gap * 0.4  # 0.3-0.7 range based on mastery gap
            adjusted_difficulty = base_difficulty + (difficulty_preference - 0.5) * 0.2  # Adjust by preference
            difficulty = max(0.2, min(0.9, adjusted_difficulty))  # Ensure within reasonable range
            
            # Generate simulated content item
            concept_name = concept_id.split(".")[-1].replace("_", " ").title()
            content_type_names = {
                "video": "Video",
                "text": "Reading",
                "interactive": "Interactive Exercise",
                "audio": "Audio Lecture",
                "simulation": "Simulation"
            }
            
            # Estimate appropriate duration based on content type and cognitive profile
            attention_span = cognitive_profile.get("attention_span", 20)  # minutes
            base_durations = {
                "video": 10,
                "text": 15,
                "interactive": 20,
                "audio": 12,
                "simulation": 18
            }
            
            # Adjust duration based on attention span and difficulty
            base_duration = base_durations.get(best_content_type, 15)
            adjusted_duration = base_duration * (0.7 + difficulty * 0.6)  # Higher difficulty = longer duration
            # Cap at attention span
            duration = min(adjusted_duration, attention_span * 0.8)
            
            item = {
                "id": f"item_{i+1:03d}",
                "title": f"{content_type_names.get(best_content_type, 'Content')} on {concept_name}",
                "type": best_content_type,
                "concepts": [concept_id],
                "difficulty": difficulty,
                "duration": duration,  # minutes
                "mastery_gain": 0.1 + mastery_gap * 0.3,  # Estimated mastery gain (higher for larger gaps)
                "engagement_score": content_scores[best_content_type],  # Predicted engagement
                "prerequisites": [],
                "position": i
            }
            
            # Add prerequisites if this isn't the first item
            if i > 0:
                item["prerequisites"] = [path_items[i-1]["id"]]
            
            path_items.append(item)
        
        return path_items
    
    def _estimate_knowledge_gain(self, path_items: List[Dict[str, Any]], knowledge_state: Dict[str, Any]) -> Dict[str, float]:
        """Estimate knowledge gain from completing the learning path
        
        Args:
            path_items (List[Dict[str, Any]]): Path items
            knowledge_state (Dict[str, Any]): Current knowledge state
                
        Returns:
            Dict[str, float]: Estimated knowledge gain per concept
        """
        # In a real implementation, this would use a sophisticated model
        # that considers interdependencies between concepts
        
        # For demonstration, using a simple estimate
        estimated_gain = {}
        
        # Process each path item
        for item in path_items:
            for concept_id in item.get("concepts", []):
                # Get current mastery
                current_mastery = 0.0
                if concept_id in knowledge_state["concepts"]:
                    current_mastery = knowledge_state["concepts"][concept_id].get("mastery", 0.0)
                
                # Estimate gain for this item (diminishing returns with higher mastery)
                item_gain = item.get("mastery_gain", 0.2) * (1 - current_mastery * 0.7)
                
                # Add to total estimated gain for this concept
                if concept_id not in estimated_gain:
                    estimated_gain[concept_id] = 0.0
                
                # Apply diminishing returns for multiple items on same concept
                diminishing_factor = 1.0 if estimated_gain[concept_id] == 0 else 0.8
                estimated_gain[concept_id] += item_gain * diminishing_factor
                
                # Cap at reasonable maximum (can't go from 0 to 1 mastery in one path)
                estimated_gain[concept_id] = min(estimated_gain[concept_id], 0.6)
        
        return estimated_gain
    
    def _calculate_recommendation_confidence(self, 
                                             concept_path: List[Dict[str, Any]], 
                                             path_items: List[Dict[str, Any]],
                                             knowledge_state: Dict[str, Any],
                                             learning_style: Dict[str, Any]) -> float:
        """Calculate confidence in the recommendation
        
        Args:
            concept_path (List[Dict[str, Any]]): Ordered concept path
            path_items (List[Dict[str, Any]]): Path items with content details
            knowledge_state (Dict[str, Any]): Current knowledge state
            learning_style (Dict[str, Any]): Learning style
                
        Returns:
            float: Confidence score (0-1)
        """
        # In a real implementation, this would use a sophisticated model
        # that considers many factors
        
        # For demonstration, using a simple heuristic approach
        
        # Calculate average confidence in user's knowledge state
        knowledge_confidences = []
        for concept in concept_path:
            concept_id = concept["id"]
            if concept_id in knowledge_state["concepts"]:
                confidence = knowledge_state["concepts"][concept_id].get("confidence", 0.5)
                knowledge_confidences.append(confidence)
            else:
                knowledge_confidences.append(0.3)  # Low confidence for unknown concepts
        
        # Calculate average mastery gap (confidence is higher for larger gaps, as they're easier to identify)
        mastery_gaps = [concept.get("mastery_gap", 0.5) for concept in concept_path]
        avg_mastery_gap = sum(mastery_gaps) / len(mastery_gaps) if mastery_gaps else 0.5
        mastery_gap_confidence = min(0.9, 0.5 + avg_mastery_gap * 0.5)  # 0.5-0.9 based on avg gap
        
        # Calculate confidence in learning style assessment
        learning_style_confidence = learning_style.get("confidence", 0.7)
        
        # Calculate content match confidence
        content_match_scores = [item.get("engagement_score", 0.5) for item in path_items]
        avg_content_match = sum(content_match_scores) / len(content_match_scores) if content_match_scores else 0.5
        
        # Calculate overall confidence score as weighted average
        if knowledge_confidences:
            avg_knowledge_confidence = sum(knowledge_confidences) / len(knowledge_confidences)
        else:
            avg_knowledge_confidence = 0.5
        
        # Weights for different confidence components
        weights = {
            "knowledge_state": 0.3,
            "mastery_gap": 0.2,
            "learning_style": 0.2,
            "content_match": 0.3
        }
        
        overall_confidence = (
            avg_knowledge_confidence * weights["knowledge_state"] +
            mastery_gap_confidence * weights["mastery_gap"] +
            learning_style_confidence * weights["learning_style"] +
            avg_content_match * weights["content_match"]
        )
        
        return overall_confidence
    
    def _generate_alternative_paths(self,
                                    goal: Dict[str, Any],
                                    knowledge_state: Dict[str, Any],
                                    learning_style: Dict[str, Any],
                                    preferences: Dict[str, Any],
                                    cognitive_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative learning paths
        
        Args:
            goal (Dict[str, Any]): Learning goal
            knowledge_state (Dict[str, Any]): Current knowledge state
            learning_style (Dict[str, Any]): Learning style
            preferences (Dict[str, Any]): User preferences
            cognitive_profile (Dict[str, Any]): Cognitive profile
                
        Returns:
            List[Dict[str, Any]]: Alternative learning paths
        """
        # In a real implementation, this would generate truly different alternatives
        # For demonstration, creating simplified alternatives
        
        alternatives = []
        
        # Generate a shorter alternative (for time-constrained learning)
        if len(goal.get("target_concepts", [])) > 1:
            # Create a new goal with fewer target concepts
            short_goal = goal.copy()
            short_goal["target_concepts"] = goal["target_concepts"][:1]  # Take just the first concept
            short_goal["name"] = "Shorter focus on key concept"
            
            # Identify target concepts based on goal
            target_concepts = self._identify_target_concepts(short_goal, knowledge_state)
            
            # Determine prerequisite concepts that need reinforcement
            prerequisite_concepts = self._identify_prerequisites(target_concepts, knowledge_state)
            
            # Generate optimal path through concepts
            concept_path = self._generate_concept_path(
                target_concepts, prerequisite_concepts, knowledge_state, learning_style, preferences
            )
            
            # Select optimal content for each concept
            path_items = self._select_content_for_concepts(
                concept_path, learning_style, preferences, cognitive_profile
            )
            
            # Create the alternative path
            if path_items:
                alternatives.append({
                    "name": "Focused Path",
                    "description": "A shorter path focusing on the most important concept",
                    "concepts": [c["id"] for c in concept_path],
                    "items": [item["id"] for item in path_items],
                    "estimated_duration": sum(item.get("duration", 20) for item in path_items)  # minutes
                })
        
        # Generate an alternative with different content types
        # Invert content type preferences
        inverted_preferences = preferences.copy()
        if "content_types" in preferences:
            content_types = preferences["content_types"].copy()
            max_type = max(content_types.items(), key=lambda x: x[1])[0]
            min_type = min(content_types.items(), key=lambda x: x[1])[0]
            
            # Swap highest and lowest preferences
            if max_type in content_types and min_type in content_types:
                inverted_preferences["content_types"] = content_types.copy()
                inverted_preferences["content_types"][max_type] = content_types[min_type]
                inverted_preferences["content_types"][min_type] = content_types[max_type]
        
        # Identify target concepts based on goal
        target_concepts = self._identify_target_concepts(goal, knowledge_state)
        
        # Determine prerequisite concepts that need reinforcement
        prerequisite_concepts = self._identify_prerequisites(target_concepts, knowledge_state)
        
        # Generate optimal path through concepts
        concept_path = self._generate_concept_path(
            target_concepts, prerequisite_concepts, knowledge_state, learning_style, inverted_preferences
        )
        
        # Select optimal content for each concept
        path_items = self._select_content_for_concepts(
            concept_path, learning_style, inverted_preferences, cognitive_profile
        )
        
        # Create the alternative path
        if path_items:
            content_types = [item["type"] for item in path_items]
            common_type = max(set(content_types), key=content_types.count)
            
            alternatives.append({
                "name": f"{common_type.title()}-focused Path",
                "description": f"An alternative path emphasizing {common_type} content",
                "concepts": [c["id"] for c in concept_path],
                "items": [item["id"] for item in path_items],
                "estimated_duration": sum(item.get("duration", 20) for item in path_items)  # minutes
            })
        
        return alternatives
    
    def _calculate_path_difficulty(self, path_items: List[Dict[str, Any]]) -> float:
        """Calculate overall difficulty of the learning path
        
        Args:
            path_items (List[Dict[str, Any]]): Path items
                
        Returns:
            float: Difficulty score (0-1)
        """
        # Simple weighted average of item difficulties
        if not path_items:
            return 0.5
        
        total_difficulty = sum(item.get("difficulty", 0.5) * item.get("duration", 20) for item in path_items)
        total_duration = sum(item.get("duration", 20) for item in path_items)
        
        return total_difficulty / total_duration if total_duration > 0 else 0.5
    
    def update_recommendation_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Update recommendation parameters
        
        Args:
            parameters (Dict[str, Any]): New parameter values
                
        Returns:
            bool: True if update successful, False otherwise
        """
        if not self.is_initialized:
            raise RuntimeError("Learning Path Recommender must be initialized first")
        
        try:
            # Update parameters
            for key, value in parameters.items():
                if key in self.recommendation_parameters:
                    self.recommendation_parameters[key] = value
            
            # Update timestamp
            self.recommendation_parameters["last_updated"] = time.time()
            
            print("Updated recommendation parameters")
            return True
        
        except Exception as e:
            print(f"Error updating recommendation parameters: {e}")
            return False
    
    def get_recommendation_parameters(self) -> Dict[str, Any]:
        """Get current recommendation parameters
        
        Returns:
            Dict[str, Any]: Recommendation parameters
        """
        if not self.is_initialized:
            raise RuntimeError("Learning Path Recommender must be initialized first")
        
        return self.recommendation_parameters.copy()