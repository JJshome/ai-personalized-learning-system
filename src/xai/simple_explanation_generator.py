"""Simple Explanation Generator Module

This module generates human-understandable explanations for the AI system's
decisions, recommendations, and adaptations, to build trust and transparency.

Based on Ucaretron Inc.'s patent application for AI-based personalized learning systems.
"""

import time
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

class SimpleExplanationGenerator:
    """Generates simple explanations for AI decisions
    
    Creates understandable explanations for the system's recommendations, content adaptations,
    and cognitive assessments to make the AI's decision-making transparent to users.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the explanation generator
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings.
        """
        self.is_initialized = False
        self.config = config or self._get_default_config()
        print("Simple Explanation Generator created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "detail_level": "medium",  # "simple", "medium", "detailed"
            "language_style": "conversational",  # "conversational", "technical", "educational"
            "visualization": True,
            "explanation_storage": {
                "store_explanations": True,
                "storage_path": "./data/explanations"
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the explanation generator"""
        try:
            # Create storage directory if it doesn't exist and storage is enabled
            if self.config["explanation_storage"]["store_explanations"]:
                storage_path = Path(self.config["explanation_storage"]["storage_path"])
                storage_path.mkdir(parents=True, exist_ok=True)
            
            self.is_initialized = True
            print("Simple Explanation Generator initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing explanation generator: {e}")
            return False
    
    def explain_learning_path(self, path_data: Dict[str, Any], learning_model: Any) -> Dict[str, Any]:
        """Generate explanation for a learning path recommendation
        
        Args:
            path_data: Learning path data
            learning_model: Learning model with user data
            
        Returns:
            Dict[str, Any]: Explanation with text and visualizations
        """
        if not self.is_initialized:
            raise RuntimeError("SimpleExplanationGenerator must be initialized first")
        
        # Extract key information from path data
        concepts = path_data.get("concepts", [])
        goal = path_data.get("goal", {})
        confidence = path_data.get("confidence", 0.7)
        
        # Format concept names
        concept_names = []
        for c in concepts[:5]:  # Limit to top 5 concepts
            if isinstance(c, dict) and "id" in c:
                name = c["id"].split(".")[-1].replace("_", " ").title()
                concept_names.append(name)
            elif isinstance(c, str):
                name = c.split(".")[-1].replace("_", " ").title()
                concept_names.append(name)
        
        # Get goal name
        goal_name = "mastering these concepts"
        if isinstance(goal, dict):
            if "name" in goal:
                goal_name = goal["name"]
            elif "target_concepts" in goal:
                targets = [t.split(".")[-1].replace("_", " ").title() for t in goal["target_concepts"][:3]]
                if targets:
                    goal_name = f"mastering {', '.join(targets)}"
        
        # Generate explanation text based on detail level
        confidence_level = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "modest"
        confidence_statement = f"I have {confidence_level} confidence ({confidence:.0%}) in this recommendation."
        
        if self.config["detail_level"] == "simple":
            explanation_text = f"I've suggested this learning path because it covers {', '.join(concept_names)} that will help you reach your goal of {goal_name}. {confidence_statement}"
        else:
            # Get learning style info if available
            style_info = ""
            try:
                if hasattr(learning_model, "get_learning_style"):
                    style = learning_model.get_learning_style()
                    if "visual_verbal_preference" in style:
                        pref = style["visual_verbal_preference"]
                        if pref < 0.4:
                            style_info = "I've included more visual content to match your learning preferences."
                        elif pref > 0.6:
                            style_info = "I've included more textual explanations to match your learning preferences."
            except:
                pass
            
            explanation_text = f"I've created this learning path specifically for you to help you with {goal_name}. It covers key concepts including {', '.join(concept_names)} in a structured sequence designed for optimal learning. {style_info} {confidence_statement}"
        
        # Create simple visualization info
        visualizations = []
        if self.config["visualization"]:
            visualizations.append({
                "type": "path_flow",
                "title": "Your Learning Path Flow",
                "description": "A flowchart showing the progression of concepts in this learning path",
                "format": "flowchart"
            })
        
        # Return explanation
        return {
            "explanation_id": f"path_rec_exp_{int(time.time())}",
            "text": explanation_text,
            "visualizations": visualizations,
            "timestamp": time.time()
        }
    
    def explain_content_adaptation(self, content_data: Dict[str, Any], learning_model: Any) -> Dict[str, Any]:
        """Generate explanation for content adaptation
        
        Args:
            content_data: Adapted content data
            learning_model: Learning model with user data
            
        Returns:
            Dict[str, Any]: Explanation with text and visualizations
        """
        if not self.is_initialized:
            raise RuntimeError("SimpleExplanationGenerator must be initialized first")
        
        # Determine primary adaptation type
        primary_adaptation = "personalized"
        adaptation_details = []
        
        # Try to extract learning style
        try:
            if hasattr(learning_model, "get_learning_style"):
                style = learning_model.get_learning_style()
                if "visual_verbal_preference" in style:
                    pref = style["visual_verbal_preference"]
                    if pref < 0.4:
                        primary_adaptation = "visual"
                        adaptation_details.append("Added more visual elements to match your learning style")
                    elif pref > 0.6:
                        primary_adaptation = "verbal"
                        adaptation_details.append("Added more detailed explanations to match your learning style")
                
                if "active_reflective_preference" in style:
                    pref = style["active_reflective_preference"]
                    if pref < 0.4:
                        adaptation_details.append("Included interactive elements for active learning")
                    elif pref > 0.6:
                        adaptation_details.append("Added reflection prompts to encourage deeper thinking")
        except:
            pass
        
        # Fallback if no details found
        if not adaptation_details:
            adaptation_details = [
                "Adjusted content based on your learning preferences",
                "Modified examples to match your interests",
                "Optimized content length based on your attention profile"
            ]
        
        # Generate explanation text
        if self.config["detail_level"] == "simple":
            explanation_text = f"I've adapted this content to match your {primary_adaptation} learning style."
        else:
            explanation_text = "I've customized this content in several ways to match your learning preferences:\n\n"
            for detail in adaptation_details[:3]:
                explanation_text += f"- {detail}\n"
            explanation_text += "\nThese changes should help you learn more effectively."
        
        # Create simple visualization info
        visualizations = []
        if self.config["visualization"]:
            visualizations.append({
                "type": "adaptation_comparison",
                "title": "Content Adaptation Overview",
                "description": "A comparison of the original content and adapted content",
                "format": "side_by_side"
            })
        
        # Return explanation
        return {
            "explanation_id": f"content_adapt_exp_{int(time.time())}",
            "text": explanation_text,
            "visualizations": visualizations,
            "timestamp": time.time()
        }
    
    def explain_cognitive_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for a cognitive assessment
        
        Args:
            assessment_data: Cognitive assessment data
            
        Returns:
            Dict[str, Any]: Explanation with text and visualizations
        """
        if not self.is_initialized:
            raise RuntimeError("SimpleExplanationGenerator must be initialized first")
        
        # Extract key information
        attention = assessment_data.get("attention", {}).get("level", 0.5)
        engagement = assessment_data.get("engagement", {}).get("level", 0.5)
        cognitive_load = assessment_data.get("cognitive_load", {}).get("level", 0.5)
        
        # Determine state descriptions
        attention_level = "high" if attention > 0.7 else "moderate" if attention > 0.4 else "lower"
        engagement_level = "high" if engagement > 0.7 else "moderate" if engagement > 0.4 else "lower"
        cognitive_load_level = "well-balanced" if 0.4 < cognitive_load < 0.7 else "high" if cognitive_load >= 0.7 else "low"
        
        # Generate primary state description
        if attention > 0.7 and engagement > 0.7:
            primary_state = "in an optimal learning state with strong focus and engagement"
        elif attention < 0.4 and engagement < 0.4:
            primary_state = "experiencing some difficulty maintaining focus and engagement"
        else:
            primary_state = f"showing {attention_level} attention and {engagement_level} engagement with {cognitive_load_level} cognitive load"
        
        # Generate suggestions
        suggestions = []
        if attention < 0.5:
            suggestions.append("Consider removing distractions or taking a short break")
        if engagement < 0.5:
            suggestions.append("Try different learning materials or more interactive content")
        if cognitive_load > 0.7:
            suggestions.append("The content might be too challenging - breaking it into smaller parts may help")
        elif cognitive_load < 0.3:
            suggestions.append("You might benefit from more challenging material")
        
        # Generate explanation text
        if self.config["detail_level"] == "simple":
            explanation_text = f"Based on your learning activity, I've noticed you're {primary_state}."
            if suggestions:
                explanation_text += f" {suggestions[0]}."
        else:
            explanation_text = f"I've analyzed your learning session and noticed these patterns:\n\n"
            explanation_text += f"- Attention level: {attention_level}\n"
            explanation_text += f"- Engagement: {engagement_level}\n"
            explanation_text += f"- Cognitive load: {cognitive_load_level}\n\n"
            explanation_text += f"These suggest you're {primary_state}."
            
            if suggestions:
                explanation_text += "\n\nSuggestions:\n"
                for suggestion in suggestions:
                    explanation_text += f"- {suggestion}\n"
        
        # Create simple visualization info
        visualizations = []
        if self.config["visualization"]:
            visualizations.append({
                "type": "cognitive_state",
                "title": "Your Cognitive Learning State",
                "description": "A visualization of your attention, engagement, and cognitive load",
                "format": "gauge_chart"
            })
        
        # Return explanation
        return {
            "explanation_id": f"cognitive_assess_exp_{int(time.time())}",
            "text": explanation_text,
            "visualizations": visualizations,
            "timestamp": time.time()
        }
