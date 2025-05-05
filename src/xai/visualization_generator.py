"""Visualization Generator Module

This module generates visualizations to help explain AI decisions,
making the system's reasoning more transparent and understandable.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import time

class VisualizationGenerator:
    """Generates visualizations to explain AI decisions
    
    This class creates various types of visualizations including charts, graphs,
    and diagrams to illustrate how and why the AI system made specific decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the visualization generator
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings.
                If None, uses default settings. Defaults to None.
        """
        self.is_initialized = False
        self.config = config or self._get_default_config()
        print("Visualization Generator created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "visualization_types": {
                "learning_path": [
                    "path_flow",
                    "concept_map",
                    "knowledge_radar",
                    "difficulty_progression",
                    "time_estimate"
                ],
                "content_adaptation": [
                    "adaptation_comparison",
                    "learning_style_match",
                    "cognitive_load_adjustment",
                    "content_structure"
                ],
                "cognitive_assessment": [
                    "attention_timeline",
                    "engagement_heatmap",
                    "cognitive_load_chart",
                    "focus_pattern"
                ],
                "knowledge_estimation": [
                    "knowledge_map",
                    "mastery_progression",
                    "concept_relationship",
                    "learning_curve"
                ]
            },
            "rendering": {
                "format": "svg",  # svg, png, jpg
                "resolution": "medium",  # low, medium, high
                "color_scheme": "accessible",  # accessible, grayscale, vibrant
                "interactive": True,
                "animation": True
            },
            "accessibility": {
                "high_contrast": True,
                "alternative_text": True,
                "large_font": False,
                "color_blind_friendly": True
            },
            "performance": {
                "caching": True,
                "cache_size": 20,  # Number of visualizations to cache
                "timeout": 3.0  # Seconds before timing out visualization generation
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the visualization generator
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize visualization backend
            # In a real implementation, this might initialize a plotting library
            # or set up connections to visualization services
            
            self.is_initialized = True
            print("Visualization Generator initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing visualization generator: {e}")
            return False
    
    def generate_path_flow_visualization(self, path_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a flowchart visualization of a learning path
        
        Args:
            path_data (Dict[str, Any]): Learning path data
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        if not self.is_initialized:
            raise RuntimeError("Visualization generator must be initialized first")
        
        try:
            # Extract concepts and items from path data
            concepts = path_data.get("concepts", [])
            items = path_data.get("items", [])
            
            # Create nodes and edges for the flowchart
            nodes = []
            edges = []
            
            # Create a start node
            nodes.append({
                "id": "start",
                "type": "start",
                "label": "Start Here"
            })
            
            # Process concepts and items to create nodes and edges
            # This is a simplified implementation
            for i, concept in enumerate(concepts):
                concept_id = concept.get("id", f"concept_{i}") if isinstance(concept, dict) else f"concept_{i}"
                concept_name = concept.get("name", f"Concept {i}") if isinstance(concept, dict) else str(concept)
                
                # Add concept node
                nodes.append({
                    "id": concept_id,
                    "type": "concept",
                    "label": concept_name
                })
                
                # Add edge from previous node to this concept
                if i == 0:
                    # Connect start to first concept
                    edges.append({
                        "source": "start",
                        "target": concept_id,
                        "type": "path"
                    })
                else:
                    # Connect previous concept to this one
                    prev_concept = concepts[i-1]
                    prev_id = prev_concept.get("id", f"concept_{i-1}") if isinstance(prev_concept, dict) else f"concept_{i-1}"
                    edges.append({
                        "source": prev_id,
                        "target": concept_id,
                        "type": "path"
                    })
            
            # Add a finish node
            nodes.append({
                "id": "finish",
                "type": "finish",
                "label": "Learning Goal Achieved"
            })
            
            # Add edge from last concept to finish
            if concepts:
                last_concept = concepts[-1]
                last_id = last_concept.get("id", f"concept_{len(concepts)-1}") if isinstance(last_concept, dict) else f"concept_{len(concepts)-1}"
                edges.append({
                    "source": last_id,
                    "target": "finish",
                    "type": "path"
                })
            
            # Create the visualization result
            visualization = {
                "visualization_id": f"path_flow_{int(time.time())}",
                "type": "path_flow",
                "title": "Your Learning Path Flow",
                "description": "A flowchart showing the progression of concepts in this learning path",
                "data": {
                    "nodes": nodes,
                    "edges": edges
                },
                "format": "flowchart",
                "rendering_options": {
                    "layout": "vertical",
                    "node_size": "medium",
                    "show_labels": True,
                    "highlight_current": True
                }
            }
            
            return visualization
            
        except Exception as e:
            print(f"Error generating path flow visualization: {e}")
            return {
                "visualization_id": f"error_{int(time.time())}",
                "type": "path_flow",
                "error": str(e),
                "description": "Unable to generate path flow visualization due to an error",
                "data": {},
                "format": "error"
            }
    
    def generate_knowledge_radar_visualization(self, knowledge_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a radar chart visualization of a learner's knowledge state
        
        Args:
            knowledge_state (Dict[str, Any]): Knowledge state data
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        if not self.is_initialized:
            raise RuntimeError("Visualization generator must be initialized first")
        
        try:
            # Extract concept mastery data
            concepts = []
            values = []
            
            if "concepts" in knowledge_state:
                for concept_id, concept_data in knowledge_state["concepts"].items():
                    # Format concept name for display
                    concept_name = concept_id.split(".")[-1].replace("_", " ").title() if "." in concept_id else concept_id
                    
                    # Get mastery value
                    mastery = concept_data.get("mastery", 0.0)
                    
                    concepts.append(concept_name)
                    values.append(mastery)
            
            # Ensure we have at least some data to display
            if not concepts:
                # Add placeholder data
                concepts = ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"]
                values = [0.2, 0.4, 0.1, 0.3, 0.5]
            
            # Create the visualization result
            visualization = {
                "visualization_id": f"knowledge_radar_{int(time.time())}",
                "type": "knowledge_radar",
                "title": "Your Knowledge Profile",
                "description": "A radar chart showing your current knowledge across key concepts",
                "data": {
                    "categories": concepts,
                    "values": values,
                    "scale": [0, 1],
                    "labels": ["Beginning", "Developing", "Proficient", "Mastered"]
                },
                "format": "radar_chart",
                "rendering_options": {
                    "fill_opacity": 0.5,
                    "line_thickness": 2,
                    "show_grid": True,
                    "show_labels": True
                }
            }
            
            return visualization
            
        except Exception as e:
            print(f"Error generating knowledge radar visualization: {e}")
            return {
                "visualization_id": f"error_{int(time.time())}",
                "type": "knowledge_radar",
                "error": str(e),
                "description": "Unable to generate knowledge radar visualization due to an error",
                "data": {},
                "format": "error"
            }
    
    def generate_attention_timeline_visualization(self, cognitive_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a timeline visualization of attention levels during a learning session
        
        Args:
            cognitive_data (Dict[str, Any]): Cognitive data including attention measurements
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        if not self.is_initialized:
            raise RuntimeError("Visualization generator must be initialized first")
        
        try:
            # Extract attention data
            attention_timeline = []
            
            if "attention_data" in cognitive_data and isinstance(cognitive_data["attention_data"], list):
                attention_timeline = cognitive_data["attention_data"]
            else:
                # Generate placeholder data if actual data not available
                # In a real implementation, this would be replaced with actual data
                import random
                
                # Generate 20 minutes of simulated attention data (1 point per minute)
                base_attention = 0.7  # Starting attention level
                attention_timeline = []
                for i in range(20):
                    # Add some natural variation
                    attention = base_attention + (random.random() - 0.5) * 0.2
                    # Apply a natural decline over time
                    base_attention -= 0.01
                    # Ensure value stays in range [0, 1]
                    attention = max(0, min(1, attention))
                    
                    attention_timeline.append({
                        "timestamp": i * 60,  # seconds
                        "value": attention
                    })
            
            # Create the visualization result
            visualization = {
                "visualization_id": f"attention_timeline_{int(time.time())}",
                "type": "attention_timeline",
                "title": "Your Attention During Learning",
                "description": "A timeline showing your attention levels throughout your learning session",
                "data": {
                    "timeline": attention_timeline,
                    "thresholds": {
                        "high": 0.8,
                        "medium": 0.5,
                        "low": 0.3
                    }
                },
                "format": "line_chart",
                "rendering_options": {
                    "x_axis_label": "Time (minutes)",
                    "y_axis_label": "Attention Level",
                    "show_thresholds": True,
                    "highlight_peaks": True,
                    "highlight_valleys": True
                }
            }
            
            return visualization
            
        except Exception as e:
            print(f"Error generating attention timeline visualization: {e}")
            return {
                "visualization_id": f"error_{int(time.time())}",
                "type": "attention_timeline",
                "error": str(e),
                "description": "Unable to generate attention timeline visualization due to an error",
                "data": {},
                "format": "error"
            }
    
    def generate_learning_style_match_visualization(self, adaptation_data: Dict[str, Any], learning_style: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a visualization showing how content adaptations match learning style
        
        Args:
            adaptation_data (Dict[str, Any]): Content adaptation data
            learning_style (Dict[str, Any]): Learner's learning style data
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        if not self.is_initialized:
            raise RuntimeError("Visualization generator must be initialized first")
        
        try:
            # Extract learning style dimensions
            style_dimensions = [
                "visual_verbal",
                "active_reflective",
                "sequential_global",
                "sensing_intuitive"
            ]
            
            # Extract values for each dimension
            style_values = []
            adaptation_values = []
            
            for dimension in style_dimensions:
                # Get learner's style preference (0-1 scale)
                preference_key = f"{dimension}_preference"
                if preference_key in learning_style:
                    style_values.append(learning_style[preference_key])
                else:
                    # Default to middle value if not available
                    style_values.append(0.5)
                
                # Get adaptation strength for this dimension (0-1 scale)
                adaptation_key = f"{dimension}_adaptation"
                if adaptation_key in adaptation_data:
                    adaptation_values.append(adaptation_data[adaptation_key])
                else:
                    # Default to match learner's style if not available
                    adaptation_values.append(style_values[-1])
            
            # Create formatted dimension labels
            dimension_labels = [
                "Visual-Verbal",
                "Active-Reflective",
                "Sequential-Global",
                "Sensing-Intuitive"
            ]
            
            # Create the visualization result
            visualization = {
                "visualization_id": f"learning_style_match_{int(time.time())}",
                "type": "learning_style_match",
                "title": "Learning Style Alignment",
                "description": "A chart showing how the adaptations align with your learning style",
                "data": {
                    "dimensions": dimension_labels,
                    "learner_values": style_values,
                    "adaptation_values": adaptation_values,
                    "scale": [0, 1],
                    "interpretation": {
                        "visual_verbal": ["Visual", "Balanced", "Verbal"],
                        "active_reflective": ["Active", "Balanced", "Reflective"],
                        "sequential_global": ["Sequential", "Balanced", "Global"],
                        "sensing_intuitive": ["Sensing", "Balanced", "Intuitive"]
                    }
                },
                "format": "radar_chart",
                "rendering_options": {
                    "show_both_series": True,
                    "fill_opacity": 0.5,
                    "show_legend": True,
                    "legend_labels": ["Your Style", "Content Adaptation"]
                }
            }
            
            return visualization
            
        except Exception as e:
            print(f"Error generating learning style match visualization: {e}")
            return {
                "visualization_id": f"error_{int(time.time())}",
                "type": "learning_style_match",
                "error": str(e),
                "description": "Unable to generate learning style match visualization due to an error",
                "data": {},
                "format": "error"
            }
    
    def generate_mastery_progression_visualization(self, learning_history: Dict[str, Any], concept_id: str) -> Dict[str, Any]:
        """Generate a visualization showing a learner's mastery progression for a concept
        
        Args:
            learning_history (Dict[str, Any]): Learner's learning history data
            concept_id (str): ID of the concept to visualize
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        if not self.is_initialized:
            raise RuntimeError("Visualization generator must be initialized first")
        
        try:
            # Extract mastery progression data for the concept
            progression = []
            
            if "concept_history" in learning_history and concept_id in learning_history["concept_history"]:
                concept_history = learning_history["concept_history"][concept_id]
                if "mastery_progression" in concept_history:
                    progression = concept_history["mastery_progression"]
            
            if not progression:
                # Generate placeholder data if actual data not available
                # In a real implementation, this would be replaced with actual data
                progression = [
                    {"timestamp": 1619844000, "value": 0.1, "event": "Introduction"},
                    {"timestamp": 1620103200, "value": 0.25, "event": "First Exercise"},
                    {"timestamp": 1620362400, "value": 0.4, "event": "Video Lesson"},
                    {"timestamp": 1620621600, "value": 0.55, "event": "Practice Quiz"},
                    {"timestamp": 1620880800, "value": 0.7, "event": "Application Project"},
                    {"timestamp": 1621140000, "value": 0.85, "event": "Final Assessment"}
                ]
            
            # Format concept name for display
            concept_name = concept_id.split(".")[-1].replace("_", " ").title() if "." in concept_id else concept_id
            
            # Extract values and timestamps for the chart
            timestamps = [entry["timestamp"] for entry in progression]
            values = [entry["value"] for entry in progression]
            events = [entry.get("event", "") for entry in progression]
            
            # Create the visualization result
            visualization = {
                "visualization_id": f"mastery_progression_{int(time.time())}",
                "type": "mastery_progression",
                "title": f"Your Learning Journey: {concept_name}",
                "description": f"A chart showing your progress in mastering {concept_name} over time",
                "data": {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "timestamps": timestamps,
                    "values": values,
                    "events": events,
                    "thresholds": {
                        "beginning": 0.3,
                        "developing": 0.6,
                        "proficient": 0.8,
                        "mastered": 0.95
                    }
                },
                "format": "line_chart",
                "rendering_options": {
                    "x_axis_label": "Time",
                    "y_axis_label": "Mastery Level",
                    "show_events": True,
                    "show_thresholds": True,
                    "threshold_labels": ["Beginning", "Developing", "Proficient", "Mastered"]
                }
            }
            
            return visualization
            
        except Exception as e:
            print(f"Error generating mastery progression visualization: {e}")
            return {
                "visualization_id": f"error_{int(time.time())}",
                "type": "mastery_progression",
                "error": str(e),
                "description": "Unable to generate mastery progression visualization due to an error",
                "data": {},
                "format": "error"
            }
