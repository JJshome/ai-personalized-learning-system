"""XAI Module for AI Personalized Learning System

This module provides explainable AI capabilities for the learning system,
making the system's decision-making process transparent and understandable
to users, educators, and other stakeholders.

Based on Ucaretron Inc.'s patent application for AI-based personalized learning systems.
"""

from .simple_explanation_generator import SimpleExplanationGenerator
from .visualization_generator import VisualizationGenerator

__all__ = ['SimpleExplanationGenerator', 'VisualizationGenerator', 'XAIManager']

class XAIManager:
    """Manager class for Explainable AI components
    
    This class coordinates the different XAI components to provide
    comprehensive explanations for the AI system's decisions and adaptations.
    """
    
    def __init__(self, config=None):
        """Initialize the XAI manager
        
        Args:
            config: Configuration settings for XAI components
        """
        self.is_initialized = False
        self.config = config or {}
        
        # Initialize explanation generator
        self.explanation_generator = SimpleExplanationGenerator(
            self.config.get("explanation_generator", None)
        )
        
        # Initialize visualization generator
        self.visualization_generator = VisualizationGenerator(
            self.config.get("visualization_generator", None)
        )
        
        print("XAI Manager created")
    
    def initialize(self):
        """Initialize all XAI components
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize explanation generator
            if not self.explanation_generator.initialize():
                print("Failed to initialize explanation generator")
                return False
            
            # Initialize visualization generator
            if not self.visualization_generator.initialize():
                print("Failed to initialize visualization generator")
                return False
            
            self.is_initialized = True
            print("XAI Manager initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing XAI manager: {e}")
            return False
    
    def explain_learning_path(self, path_data, learning_model, detail_level=None, language_style=None):
        """Generate explanation for a learning path recommendation
        
        Args:
            path_data: Learning path data to explain
            learning_model: Learning model instance
            detail_level: Desired level of detail in explanation
            language_style: Desired language style for explanation
            
        Returns:
            dict: Explanation with text and visualizations
        """
        if not self.is_initialized:
            raise RuntimeError("XAI Manager must be initialized first")
        
        # Generate explanation text
        explanation = self.explanation_generator.explain_learning_path(path_data, learning_model)
        
        # Generate visualizations if not already included
        if not explanation.get("visualizations"):
            visualizations = []
            
            # Add path flow visualization
            path_flow = self.visualization_generator.generate_path_flow_visualization(path_data)
            visualizations.append(path_flow)
            
            # Add knowledge radar visualization if learning model is available
            try:
                knowledge_state = learning_model.get_knowledge_state()
                knowledge_radar = self.visualization_generator.generate_knowledge_radar_visualization(knowledge_state)
                visualizations.append(knowledge_radar)
            except:
                # Skip if learning model doesn't have knowledge state
                pass
            
            explanation["visualizations"] = visualizations
        
        return explanation
    
    def explain_content_adaptation(self, content_data, learning_model, detail_level=None, language_style=None):
        """Generate explanation for content adaptation
        
        Args:
            content_data: Adapted content data to explain
            learning_model: Learning model instance
            detail_level: Desired level of detail in explanation
            language_style: Desired language style for explanation
            
        Returns:
            dict: Explanation with text and visualizations
        """
        if not self.is_initialized:
            raise RuntimeError("XAI Manager must be initialized first")
        
        # Generate explanation text
        explanation = self.explanation_generator.explain_content_adaptation(content_data, learning_model)
        
        # Generate visualizations if needed
        if not explanation.get("visualizations"):
            visualizations = []
            
            # Add learning style match visualization if learning model is available
            try:
                learning_style = learning_model.get_learning_style()
                style_match = self.visualization_generator.generate_learning_style_match_visualization(
                    content_data, learning_style
                )
                visualizations.append(style_match)
            except:
                # Skip if learning model doesn't have learning style
                pass
            
            explanation["visualizations"] = visualizations
        
        return explanation
    
    def explain_cognitive_assessment(self, assessment_data, learning_model=None):
        """Generate explanation for cognitive assessment
        
        Args:
            assessment_data: Cognitive assessment data to explain
            learning_model: Optional learning model instance
            
        Returns:
            dict: Explanation with text and visualizations
        """
        if not self.is_initialized:
            raise RuntimeError("XAI Manager must be initialized first")
        
        # Generate explanation text
        explanation = self.explanation_generator.explain_cognitive_assessment(assessment_data)
        
        # Generate visualizations if needed
        if not explanation.get("visualizations"):
            visualizations = []
            
            # Add attention timeline visualization
            attention_timeline = self.visualization_generator.generate_attention_timeline_visualization(
                assessment_data
            )
            visualizations.append(attention_timeline)
            
            explanation["visualizations"] = visualizations
        
        return explanation
    
    def generate_model_card(self, model_data):
        """Generate a model card for the AI system
        
        Args:
            model_data: Data about the AI model
            
        Returns:
            dict: Model card with detailed information about the AI system
        """
        if not self.is_initialized:
            raise RuntimeError("XAI Manager must be initialized first")
        
        # Simple model card implementation
        model_card = {
            "model_name": model_data.get("name", "AI Personalized Learning System"),
            "version": model_data.get("version", "1.0.0"),
            "description": model_data.get("description", "An AI system for personalized learning path recommendation and management"),
            "purpose": "To provide personalized learning experiences through adaptive path recommendations and content adaptations",
            "architecture": model_data.get("architecture", "Hybrid AI system combining reinforcement learning, neural networks, and knowledge graphs"),
            "performance_metrics": model_data.get("performance_metrics", {
                "recommendation_accuracy": 0.85,
                "content_adaptation_relevance": 0.82,
                "cognitive_assessment_accuracy": 0.78
            }),
            "limitations": [
                "Requires sufficient learning history data for optimal performance",
                "Performance may vary across different subject domains",
                "Cognitive assessments work best with supported biometric sensors"
            ],
            "ethical_considerations": [
                "Designed to protect user privacy and data security",
                "Aims to promote learning efficacy while respecting learner autonomy",
                "Continually evaluated for potential biases in recommendations"
            ],
            "feedback_mechanism": "System incorporates user feedback to improve recommendations and explanations"
        }
        
        return model_card
