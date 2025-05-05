"""Content Generator Module

This module handles the generation of personalized learning content
based on the learner's knowledge state, learning style, and cognitive profile.
"""

import time
import random
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class ContentGenerator:
    """Generates personalized learning content
    
    This class creates and adapts learning content to match the learner's
    characteristics, using various AI generation techniques for different content types,
    including text, images, audio, interactive exercises, and more.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the content generator
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings.
                If None, uses default settings. Defaults to None.
        """
        self.is_initialized = False
        self.content_models = {}
        self.content_templates = {}
        self.content_cache = {}
        self.config = config or self._get_default_config()
        print("Content Generator created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration
        
        Returns:
            Dict[str, Any]: Default configuration
        """
        return {
            "content_types": {
                "text": {
                    "enabled": True,
                    "model": "gpt",  # "gpt", "llama", etc.
                    "temperature": 0.7,
                    "max_length": 2000,
                    "formats": ["explanation", "summary", "example", "question", "answer"]
                },
                "image": {
                    "enabled": True,
                    "model": "stable_diffusion",  # "stable_diffusion", "dalle", etc.
                    "resolution": "512x512",
                    "formats": ["diagram", "illustration", "chart", "infographic"]
                },
                "audio": {
                    "enabled": True,
                    "model": "elevenlabs",  # "elevenlabs", "gtts", etc.
                    "voice": "neutral",  # "neutral", "male", "female", etc.
                    "formats": ["narration", "pronunciation"]
                },
                "video": {
                    "enabled": True,
                    "model": "synthesis",  # "synthesis", "replicate", etc.
                    "resolution": "720p",
                    "max_duration": 120,  # seconds
                    "formats": ["demonstration", "animation", "lecture"]
                },
                "interactive": {
                    "enabled": True,
                    "frameworks": ["html5", "react"],
                    "complexity": "medium",  # "low", "medium", "high"
                    "formats": ["quiz", "simulation", "game", "exercise"]
                },
                "ar_vr": {
                    "enabled": True,
                    "frameworks": ["unity", "unreal"],
                    "complexity": "medium",  # "low", "medium", "high"
                    "formats": ["3d_model", "virtual_environment", "interactive_simulation"]
                }
            },
            "adaptation": {
                "learning_style": True,
                "knowledge_state": True,
                "cognitive_profile": True,
                "emotional_state": True,
                "difficulty_levels": 5,  # Number of difficulty levels
                "complexity_levels": 3  # Number of complexity levels
            },
            "performance": {
                "caching": True,
                "cache_size": 100,  # Number of content items to cache
                "generation_timeout": 30,  # seconds
                "batch_generation": True
            },
            "resources": {
                "template_directory": "./resources/templates",
                "asset_directory": "./resources/assets",
                "output_directory": "./output/content"
            },
            "api_keys": {
                "openai": "${OPENAI_API_KEY}",  # Environment variable placeholder
                "elevenlabs": "${ELEVENLABS_API_KEY}",
                "replicate": "${REPLICATE_API_KEY}"
            },
            "simulation": {
                "enabled": True,  # Use simulated content generation for development
                "generation_delay": 1.0,  # seconds - simulate API latency
                "quality_variation": 0.2  # Random variation in simulated quality
            }
        }
    
    def initialize(self) -> bool:
        """Initialize content generation models and resources
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load content templates
            self._load_templates()
            
            # Initialize content generation models
            if not self.config["simulation"]["enabled"]:
                self._initialize_models()
            else:
                print("Initializing in simulation mode (no actual generation models)")
            
            # Create output directory if it doesn't exist
            output_dir = Path(self.config["resources"]["output_directory"])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.is_initialized = True
            print("Content Generator initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing content generator: {e}")
            return False
    
    def _load_templates(self) -> None:
        """Load content templates from the template directory
        
        In a real implementation, this would load template files.
        For this reference implementation, we'll create some built-in templates.
        """
        # Create dictionaries for each content type
        for content_type in self.config["content_types"]:
            self.content_templates[content_type] = {}
            
            # Add templates for each format
            formats = self.config["content_types"][content_type].get("formats", [])
            for format_type in formats:
                self.content_templates[content_type][format_type] = self._get_default_templates(content_type, format_type)
        
        print(f"Loaded templates for {len(self.content_templates)} content types")
    
    def _get_default_templates(self, content_type: str, format_type: str) -> Dict[str, Any]:
        """Get default templates for a content type and format
        
        Args:
            content_type (str): Content type (e.g., "text", "image")
            format_type (str): Format type (e.g., "explanation", "diagram")
            
        Returns:
            Dict[str, Any]: Default templates
        """
        templates = {}
        
        # Text templates
        if content_type == "text":
            if format_type == "explanation":
                templates["basic"] = {
                    "template": "In this lesson, we will learn about {{concept}}. {{concept}} is {{definition}}. {{additional_details}}",
                    "variables": ["concept", "definition", "additional_details"],
                    "difficulty_params": {
                        "easy": {"max_length": 200, "complexity": "simple", "examples": 2},
                        "medium": {"max_length": 400, "complexity": "moderate", "examples": 1},
                        "hard": {"max_length": 600, "complexity": "advanced", "examples": 0}
                    }
                }
                templates["visual_learner"] = {
                    "template": "Let's visualize {{concept}}. Imagine {{visual_metaphor}}. {{concept}} works similarly to {{analogy}}. {{additional_details}}",
                    "variables": ["concept", "visual_metaphor", "analogy", "additional_details"],
                    "difficulty_params": {
                        "easy": {"max_length": 200, "visuals": 3, "analogies": 2},
                        "medium": {"max_length": 400, "visuals": 2, "analogies": 1},
                        "hard": {"max_length": 600, "visuals": 1, "analogies": 1}
                    }
                }
            elif format_type == "question":
                templates["multiple_choice"] = {
                    "template": "{{question}}\n\nA) {{option_a}}\nB) {{option_b}}\nC) {{option_c}}\nD) {{option_d}}",
                    "variables": ["question", "option_a", "option_b", "option_c", "option_d", "correct_answer"],
                    "difficulty_params": {
                        "easy": {"distractors": "obvious", "complexity": "recall"},
                        "medium": {"distractors": "plausible", "complexity": "application"},
                        "hard": {"distractors": "subtle", "complexity": "analysis"}
                    }
                }
                templates["open_ended"] = {
                    "template": "{{question}}",
                    "variables": ["question", "rubric", "sample_answer"],
                    "difficulty_params": {
                        "easy": {"complexity": "recall", "word_limit": 50},
                        "medium": {"complexity": "application", "word_limit": 100},
                        "hard": {"complexity": "synthesis", "word_limit": 200}
                    }
                }
        
        # Image templates
        elif content_type == "image":
            if format_type == "diagram":
                templates["concept_map"] = {
                    "template": "A concept map showing the relationship between {{main_concept}} and {{related_concepts}}.",
                    "variables": ["main_concept", "related_concepts", "relationships", "style"],
                    "difficulty_params": {
                        "easy": {"num_nodes": 5, "complexity": "linear"},
                        "medium": {"num_nodes": 10, "complexity": "branching"},
                        "hard": {"num_nodes": 15, "complexity": "network"}
                    }
                }
                templates["process_flow"] = {
                    "template": "A diagram showing the steps of {{process_name}}: {{steps}}.",
                    "variables": ["process_name", "steps", "style"],
                    "difficulty_params": {
                        "easy": {"num_steps": 4, "details": "minimal"},
                        "medium": {"num_steps": 7, "details": "moderate"},
                        "hard": {"num_steps": 10, "details": "detailed"}
                    }
                }
        
        # Interactive templates
        elif content_type == "interactive":
            if format_type == "quiz":
                templates["knowledge_check"] = {
                    "template": "A quiz to test knowledge of {{concept}}.",
                    "variables": ["concept", "num_questions", "question_types", "feedback"],
                    "difficulty_params": {
                        "easy": {"num_questions": 5, "time_limit": 300, "feedback": "detailed"},
                        "medium": {"num_questions": 10, "time_limit": 600, "feedback": "moderate"},
                        "hard": {"num_questions": 15, "time_limit": 900, "feedback": "minimal"}
                    }
                }
            elif format_type == "simulation":
                templates["interactive_model"] = {
                    "template": "An interactive simulation of {{concept}} showing {{interactions}}.",
                    "variables": ["concept", "interactions", "parameters", "visualization"],
                    "difficulty_params": {
                        "easy": {"parameters": 2, "complexity": "simple", "guidance": "step_by_step"},
                        "medium": {"parameters": 4, "complexity": "moderate", "guidance": "hints"},
                        "hard": {"parameters": 6, "complexity": "complex", "guidance": "minimal"}
                    }
                }
        
        # Default empty template if not specifically defined
        if not templates:
            templates["default"] = {
                "template": f"Default template for {content_type}/{format_type}",
                "variables": ["content"],
                "difficulty_params": {
                    "easy": {},
                    "medium": {},
                    "hard": {}
                }
            }
        
        return templates
    
    def _initialize_models(self) -> None:
        """Initialize content generation models
        
        In a real implementation, this would load the actual AI models.
        For this reference implementation, we'll just print the steps that would be taken.
        """
        # Initialize models for each enabled content type
        for content_type, settings in self.config["content_types"].items():
            if settings["enabled"]:
                model_name = settings.get("model")
                print(f"Initializing {model_name} model for {content_type} generation")
                
                # In a real implementation, this would load the actual models
                # For this reference, just store placeholder model info
                self.content_models[content_type] = {
                    "name": model_name,
                    "initialized": True,
                    "settings": settings
                }
        
        print(f"Initialized {len(self.content_models)} content generation models")
    
    def generate_content(self, 
                       content_type: str, 
                       format_type: str, 
                       content_params: Dict[str, Any],
                       learner_profile: Optional[Dict[str, Any]] = None,
                       difficulty: str = "medium") -> Dict[str, Any]:
        """Generate personalized content
        
        Args:
            content_type (str): Type of content to generate (e.g., "text", "image")
            format_type (str): Format of content (e.g., "explanation", "diagram")
            content_params (Dict[str, Any]): Content parameters
            learner_profile (Optional[Dict[str, Any]], optional): Learner profile for personalization.
                If None, uses default settings. Defaults to None.
            difficulty (str, optional): Difficulty level. Defaults to "medium".
                
        Returns:
            Dict[str, Any]: Generated content
        """
        if not self.is_initialized:
            raise RuntimeError("Content generator must be initialized first")
        
        try:
            # Check if content type and format are supported
            if content_type not in self.config["content_types"] or not self.config["content_types"][content_type]["enabled"]:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            if content_type not in self.content_templates or format_type not in self.content_templates[content_type]:
                raise ValueError(f"Unsupported format type: {format_type} for content type: {content_type}")
            
            # Generate cache key
            cache_key = self._generate_cache_key(content_type, format_type, content_params, difficulty)
            
            # Check cache if enabled
            if self.config["performance"]["caching"] and cache_key in self.content_cache:
                print(f"Using cached content for {content_type}/{format_type}")
                return self.content_cache[cache_key]
            
            # Select template based on learner profile if available
            template = self._select_template(content_type, format_type, learner_profile)
            
            # Adapt content parameters based on difficulty
            adapted_params = self._adapt_params(content_params, template, difficulty, learner_profile)
            
            # Generate content
            if self.config["simulation"]["enabled"]:
                content = self._simulate_content_generation(content_type, format_type, adapted_params, template)
            else:
                content = self._generate_content_with_model(content_type, format_type, adapted_params, template)
            
            # Apply adaptations based on learner profile
            if learner_profile and self.config["adaptation"]["learning_style"]:
                content = self._adapt_to_learning_style(content, content_type, learner_profile)
            
            if learner_profile and self.config["adaptation"]["cognitive_profile"]:
                content = self._adapt_to_cognitive_profile(content, content_type, learner_profile)
            
            # Add metadata
            content_with_metadata = {
                "content_id": f"{content_type}_{format_type}_{int(time.time())}",
                "content_type": content_type,
                "format_type": format_type,
                "difficulty": difficulty,
                "template_used": template["name"],
                "generated_at": time.time(),
                "content": content,
                "metadata": {
                    "params": adapted_params,
                    "personalized": learner_profile is not None,
                    "template_variables": template["variables"]
                }
            }
            
            # Cache the result if caching is enabled
            if self.config["performance"]["caching"]:
                self.content_cache[cache_key] = content_with_metadata
                
                # Manage cache size
                if len(self.content_cache) > self.config["performance"]["cache_size"]:
                    # Remove oldest entries
                    oldest_keys = sorted(self.content_cache.keys(), 
                                        key=lambda k: self.content_cache[k]["generated_at"])[:len(self.content_cache) - self.config["performance"]["cache_size"]]
                    for key in oldest_keys:
                        del self.content_cache[key]
            
            return content_with_metadata
            
        except Exception as e:
            print(f"Error generating content: {e}")
            return {
                "content_id": f"error_{int(time.time())}",
                "content_type": content_type,
                "format_type": format_type,
                "error": str(e),
                "generated_at": time.time()
            }
    
    def _generate_cache_key(self, content_type: str, format_type: str, params: Dict[str, Any], difficulty: str) -> str:
        """Generate a cache key for content
        
        Args:
            content_type (str): Content type
            format_type (str): Format type
            params (Dict[str, Any]): Content parameters
            difficulty (str): Difficulty level
                
        Returns:
            str: Cache key
        """
        # Sort params to ensure consistent keys regardless of param order
        sorted_params = json.dumps(params, sort_keys=True)
        return f"{content_type}_{format_type}_{sorted_params}_{difficulty}"
    
    def _select_template(self, content_type: str, format_type: str, learner_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the most appropriate template for the learner
        
        Args:
            content_type (str): Content type
            format_type (str): Format type
            learner_profile (Optional[Dict[str, Any]]): Learner profile
                
        Returns:
            Dict[str, Any]: Selected template with name
        """
        templates = self.content_templates[content_type][format_type]
        
        # If no learner profile or only one template, use the first available template
        if not learner_profile or len(templates) == 1:
            template_name = next(iter(templates))
            return {"name": template_name, **templates[template_name]}
        
        # Select template based on learning style
        learning_style = learner_profile.get("learning_style", {})
        visual_verbal = learning_style.get("visual_verbal_preference", 0.5)
        active_reflective = learning_style.get("active_reflective_preference", 0.5)
        
        # For text content, choose based on visual/verbal preference
        if content_type == "text" and format_type == "explanation":
            if visual_verbal < 0.4 and "visual_learner" in templates:
                # Visual learner
                return {"name": "visual_learner", **templates["visual_learner"]}
            else:
                # Verbal learner or balanced
                return {"name": "basic", **templates["basic"]}
        
        # For questions, choose based on active/reflective preference
        elif content_type == "text" and format_type == "question":
            if active_reflective < 0.4 and "multiple_choice" in templates:
                # Active learner
                return {"name": "multiple_choice", **templates["multiple_choice"]}
            elif active_reflective > 0.6 and "open_ended" in templates:
                # Reflective learner
                return {"name": "open_ended", **templates["open_ended"]}
            else:
                # Balanced
                template_name = next(iter(templates))
                return {"name": template_name, **templates[template_name]}
        
        # Default: use the first available template
        template_name = next(iter(templates))
        return {"name": template_name, **templates[template_name]}
    
    def _adapt_params(self, 
                      params: Dict[str, Any], 
                      template: Dict[str, Any], 
                      difficulty: str, 
                      learner_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Adapt content parameters based on difficulty and learner profile
        
        Args:
            params (Dict[str, Any]): Content parameters
            template (Dict[str, Any]): Selected template
            difficulty (str): Difficulty level
            learner_profile (Optional[Dict[str, Any]]): Learner profile
                
        Returns:
            Dict[str, Any]: Adapted parameters
        """
        adapted_params = params.copy()
        
        # Apply difficulty parameters from template
        difficulty_level = "medium"  # Default
        if difficulty in ["very_easy", "easy"]:
            difficulty_level = "easy"
        elif difficulty in ["medium"]:
            difficulty_level = "medium"
        elif difficulty in ["hard", "very_hard"]:
            difficulty_level = "hard"
        
        # Apply template difficulty parameters
        if "difficulty_params" in template and difficulty_level in template["difficulty_params"]:
            difficulty_params = template["difficulty_params"][difficulty_level]
            for param, value in difficulty_params.items():
                # Only override if not explicitly provided in original params
                if param not in adapted_params:
                    adapted_params[param] = value
        
        # Apply adaptations based on learner profile if available
        if learner_profile:
            # Adapt based on cognitive profile
            cognitive_profile = learner_profile.get("cognitive_profile", {})
            
            # Adjust length based on attention span
            if "max_length" in adapted_params and "attention_span" in cognitive_profile:
                attention_span = cognitive_profile["attention_span"]  # minutes
                max_length_factor = min(1.0, attention_span / 30.0)  # Normalize to 0-1 range based on 30min span
                adapted_params["max_length"] = int(adapted_params["max_length"] * max_length_factor)
            
            # Adjust complexity based on cognitive load threshold
            if "complexity" in adapted_params and "cognitive_load_threshold" in cognitive_profile:
                load_threshold = cognitive_profile["cognitive_load_threshold"]  # 0-1 scale
                complexity_levels = ["simple", "moderate", "advanced"]
                
                if load_threshold < 0.5 and adapted_params["complexity"] == "advanced":
                    adapted_params["complexity"] = "moderate"
                elif load_threshold < 0.3 and adapted_params["complexity"] == "moderate":
                    adapted_params["complexity"] = "simple"
        
        return adapted_params
    
    def _simulate_content_generation(self, 
                                    content_type: str, 
                                    format_type: str, 
                                    params: Dict[str, Any], 
                                    template: Dict[str, Any]) -> Any:
        """Simulate content generation for development
        
        Args:
            content_type (str): Content type
            format_type (str): Format type
            params (Dict[str, Any]): Content parameters
            template (Dict[str, Any]): Selected template
                
        Returns:
            Any: Simulated generated content
        """
        # Simulate generation delay
        time.sleep(self.config["simulation"]["generation_delay"])
        
        # Generate content based on content type
        if content_type == "text":
            return self._simulate_text_generation(format_type, params, template)
        elif content_type == "image":
            return self._simulate_image_generation(format_type, params, template)
        elif content_type == "interactive":
            return self._simulate_interactive_generation(format_type, params, template)
        else:
            # Generic simulation for other content types
            template_str = template.get("template", f"Simulated {content_type}/{format_type} content")
            
            # Fill in template variables with param values
            for var in template.get("variables", []):
                if var in params:
                    template_str = template_str.replace(f"{{{{${var}}}}}", str(params[var]))
            
            return {
                "simulated": True,
                "content": template_str,
                "params": params
            }
    
    def _simulate_text_generation(self, format_type: str, params: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate text content generation
        
        Args:
            format_type (str): Format type
            params (Dict[str, Any]): Content parameters
            template (Dict[str, Any]): Selected template
                
        Returns:
            Dict[str, Any]: Simulated text content
        """
        template_str = template.get("template", "")
        
        # Fill in template variables with param values
        for var in template.get("variables", []):
            placeholder = f"{{{{{var}}}}}"
            if var in params:
                template_str = template_str.replace(placeholder, str(params[var]))
            else:
                # Replace with a generic placeholder
                template_str = template_str.replace(placeholder, f"[{var} content]")
        
        # For explanation format, add some generated content
        if format_type == "explanation":
            concept = params.get("concept", "concept")
            complexity = params.get("complexity", "moderate")
            
            # Generate different content based on complexity
            if complexity == "simple":
                additional_content = f"{concept} is important because it helps us understand the world around us. Here's a simple example: imagine a ball rolling down a hill. {concept} explains why this happens and how we can predict where the ball will go."
            elif complexity == "advanced":
                additional_content = f"The theoretical underpinnings of {concept} are rooted in fundamental principles of mathematics and logic. Consider the implications when applied to complex systems: emergent properties become predictable, and non-linear relationships can be modeled with precision."
            else:  # moderate
                additional_content = f"{concept} has several important applications. First, it helps us solve everyday problems. Second, it provides a framework for understanding related concepts. Finally, it serves as a foundation for more advanced topics."
            
            # Append or replace in template string
            if "[additional_details content]" in template_str:
                template_str = template_str.replace("[additional_details content]", additional_content)
            elif "{{additional_details}}" in template_str:
                template_str = template_str.replace("{{additional_details}}", additional_content)
            else:
                template_str += "\n\n" + additional_content
        
        # For question format, generate appropriate questions
        elif format_type == "question":
            if "multiple_choice" in template["name"].lower():
                # Ensure we have a question and options
                if "[question content]" in template_str:
                    concept = params.get("concept", "topic")
                    template_str = template_str.replace("[question content]", f"What is the primary purpose of {concept}?")
                
                # Add options if placeholders exist
                option_placeholders = [
                    "[option_a content]", "[option_b content]", 
                    "[option_c content]", "[option_d content]"
                ]
                
                options = [
                    "To explain natural phenomena",
                    "To predict future events",
                    "To organize knowledge systematically",
                    "To communicate ideas effectively"
                ]
                
                for i, placeholder in enumerate(option_placeholders):
                    if placeholder in template_str and i < len(options):
                        template_str = template_str.replace(placeholder, options[i])
        
        return {
            "simulated": True,
            "content": template_str,
            "format": "plain_text",
            "word_count": len(template_str.split()),
            "reading_time": len(template_str.split()) / 200  # Approx. reading time in minutes
        }
    
    def _simulate_image_generation(self, format_type: str, params: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate image content generation
        
        Args:
            format_type (str): Format type
            params (Dict[str, Any]): Content parameters
            template (Dict[str, Any]): Selected template
                
        Returns:
            Dict[str, Any]: Simulated image content
        """
        # Create a description of what the image would contain
        template_str = template.get("template", "")
        
        # Fill in template variables
        for var in template.get("variables", []):
            placeholder = f"{{{{{var}}}}}"
            if var in params:
                template_str = template_str.replace(placeholder, str(params[var]))
            else:
                template_str = template_str.replace(placeholder, f"[{var}]")
        
        # Generate additional details based on format type
        additional_details = ""
        if format_type == "diagram":
            main_concept = params.get("main_concept", "concept")
            related_concepts = params.get("related_concepts", ["related concept 1", "related concept 2"])
            
            if isinstance(related_concepts, str):
                related_concepts = [related_concepts]
            
            # Create a textual description of what the diagram would contain
            additional_details = f"The diagram would show '{main_concept}' as the central node, "
            additional_details += f"with connections to {', '.join(related_concepts)}. "
            additional_details += "Each connection would be labeled with the relationship type. "
            additional_details += "The diagram would use a clean, modern style with a color-coded hierarchy."
        
        elif format_type == "chart":
            chart_type = params.get("chart_type", "bar")
            data_series = params.get("data_series", ["Series A", "Series B"])
            
            if isinstance(data_series, str):
                data_series = [data_series]
            
            additional_details = f"The chart would be a {chart_type} chart displaying data for {', '.join(data_series)}. "
            additional_details += "It would include proper axes labels, a legend, and data points. "
            additional_details += "The chart would use a clear, accessible color scheme with a title and subtitle."
        
        # Add the additional details to the description
        image_description = template_str + "\n\n" + additional_details
        
        return {
            "simulated": True,
            "content": {
                "description": image_description,
                "url": "https://via.placeholder.com/512x512.png?text=Simulated+Image",
                "alt_text": template_str
            },
            "format": "image_url",
            "dimensions": "512x512"
        }
    
    def _simulate_interactive_generation(self, format_type: str, params: Dict[str, Any], template: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate interactive content generation
        
        Args:
            format_type (str): Format type
            params (Dict[str, Any]): Content parameters
            template (Dict[str, Any]): Selected template
                
        Returns:
            Dict[str, Any]: Simulated interactive content
        """
        # Create a description of what the interactive content would contain
        template_str = template.get("template", "")
        
        # Fill in template variables
        for var in template.get("variables", []):
            placeholder = f"{{{{{var}}}}}"
            if var in params:
                template_str = template_str.replace(placeholder, str(params[var]))
            else:
                template_str = template_str.replace(placeholder, f"[{var}]")
        
        # Generate code or description based on format type
        interactive_description = ""
        interactive_code = ""
        
        if format_type == "quiz":
            concept = params.get("concept", "topic")
            num_questions = params.get("num_questions", 5)
            
            interactive_description = f"An interactive quiz about {concept} with {num_questions} questions. "
            interactive_description += "The quiz would include multiple choice and true/false questions, "
            interactive_description += "with immediate feedback after each answer and a final score at the end."
            
            # Sample HTML/JS code for quiz
            interactive_code = f"""<!DOCTYPE html>
<html>
<head>
    <title>Quiz on {concept}</title>
    <style>
        /* CSS styling would go here */
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .question {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .options {{ margin-top: 10px; }}
        .feedback {{ margin-top: 10px; padding: 10px; border-radius: 5px; display: none; }}
        .correct {{ background-color: #dff0d8; color: #3c763d; }}
        .incorrect {{ background-color: #f2dede; color: #a94442; }}
        .button {{ background-color: #4CAF50; color: white; padding: 10px 15px; border: none; 
                 border-radius: 4px; cursor: pointer; margin-top: 10px; }}
        .results {{ margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; display: none; }}
    </style>
</head>
<body>
    <h1>Quiz: {concept}</h1>
    <div id="quiz-container">
        <!-- Questions would be dynamically generated here -->
        <div class="question" id="q1">
            <h3>Question 1: What is the main purpose of {concept}?</h3>
            <div class="options">
                <label><input type="radio" name="q1" value="a"> To explain natural phenomena</label><br>
                <label><input type="radio" name="q1" value="b"> To predict future events</label><br>
                <label><input type="radio" name="q1" value="c"> To organize knowledge</label><br>
                <label><input type="radio" name="q1" value="d"> To communicate ideas</label><br>
            </div>
            <div class="feedback correct" id="q1-correct">Correct! {concept} is primarily used to organize knowledge.</div>
            <div class="feedback incorrect" id="q1-incorrect">Incorrect. The main purpose of {concept} is to organize knowledge.</div>
            <button class="button" onclick="checkAnswer('q1', 'c')">Submit</button>
        </div>
        <!-- More questions would follow -->
        
        <div class="results" id="quiz-results">
            <h2>Quiz Results</h2>
            <p>You scored: <span id="score">0</span> out of {num_questions}</p>
            <p id="feedback-message"></p>
            <button class="button" onclick="resetQuiz()">Try Again</button>
        </div>
    </div>
    
    <script>
        // JavaScript for quiz functionality would go here
        let score = 0;
        
        function checkAnswer(questionId, correctAnswer) {{
            const selected = document.querySelector(`input[name="${questionId}"]:checked`);
            
            if (!selected) {{
                alert("Please select an answer.");
                return;
            }}
            
            const selectedValue = selected.value;
            const correctFeedback = document.getElementById(`${questionId}-correct`);
            const incorrectFeedback = document.getElementById(`${questionId}-incorrect`);
            
            // Disable inputs after answering
            const inputs = document.querySelectorAll(`input[name="${questionId}"]`);
            inputs.forEach(input => input.disabled = true);
            
            // Show feedback
            if (selectedValue === correctAnswer) {{
                correctFeedback.style.display = "block";
                score++;
            }} else {{
                incorrectFeedback.style.display = "block";
            }}
            
            // Hide submit button
            document.querySelector(`#${questionId} .button`).style.display = "none";
            
            // Show results if this is the last question
            // In a real implementation, we would check if all questions are answered
            document.getElementById("score").textContent = score;
            const feedbackMessage = document.getElementById("feedback-message");
            
            if (score === {num_questions}) {{
                feedbackMessage.textContent = "Excellent! Perfect score!";
            }} else if (score >= {num_questions} * 0.7) {{
                feedbackMessage.textContent = "Great job! You have a good understanding of the topic.";
            }} else {{
                feedbackMessage.textContent = "You might want to review the material and try again.";
            }}
            
            document.getElementById("quiz-results").style.display = "block";
        }}
        
        function resetQuiz() {{
            // In a real implementation, this would reset all questions
            location.reload();
        }}
    </script>
</body>
</html>"""
        
        elif format_type == "simulation":
            concept = params.get("concept", "concept")
            complexity = params.get("complexity", "moderate")
            
            interactive_description = f"An interactive simulation of {concept}. "
            interactive_description += "Users would be able to adjust parameters and see the effects in real-time. "
            
            if complexity == "simple":
                interactive_description += "The simulation includes basic controls and a simple visualization."
            elif complexity == "complex":
                interactive_description += "The simulation includes advanced controls, multiple visualization modes, and data export capabilities."
            else:  # moderate
                interactive_description += "The simulation includes standard controls, a detailed visualization, and explanatory tooltips."
        
        return {
            "simulated": True,
            "content": {
                "description": interactive_description,
                "code": interactive_code,
                "parameters": params
            },
            "format": "interactive_html" if interactive_code else "description",
            "complexity": params.get("complexity", "moderate")
        }
    
    def _generate_content_with_model(self, 
                                   content_type: str, 
                                   format_type: str, 
                                   params: Dict[str, Any], 
                                   template: Dict[str, Any]) -> Any:
        """Generate content using AI models
        
        In a real implementation, this would call appropriate AI APIs.
        For this reference implementation, we'll fall back to simulation.
        
        Args:
            content_type (str): Content type
            format_type (str): Format type
            params (Dict[str, Any]): Content parameters
            template (Dict[str, Any]): Selected template
                
        Returns:
            Any: Generated content
        """
        # In a real implementation, this would call the appropriate AI model API
        # For this reference implementation, just use the simulation
        print(f"Note: Using simulation as fallback for {content_type}/{format_type} generation")
        return self._simulate_content_generation(content_type, format_type, params, template)
    
    def _adapt_to_learning_style(self, content: Any, content_type: str, learner_profile: Dict[str, Any]) -> Any:
        """Adapt content to match learner's learning style
        
        Args:
            content (Any): Generated content
            content_type (str): Content type
            learner_profile (Dict[str, Any]): Learner profile
                
        Returns:
            Any: Adapted content
        """
        # Get learning style preferences
        learning_style = learner_profile.get("learning_style", {})
        visual_verbal = learning_style.get("visual_verbal_preference", 0.5)  # 0 = visual, 1 = verbal
        active_reflective = learning_style.get("active_reflective_preference", 0.5)  # 0 = active, 1 = reflective
        
        # For text content
        if content_type == "text" and isinstance(content, dict) and "content" in content:
            text_content = content["content"]
            
            # For visual learners, add visual cues and structure
            if visual_verbal < 0.4 and "format" in content and content["format"] == "plain_text":
                # Add more structure with headers, lists, etc.
                if not text_content.startswith("#"):
                    # Add a title if none exists
                    text_content = "# " + text_content.split("\n")[0] + "\n\n" + text_content
                
                # Add suggestion for visual aids
                content["visual_aid_suggestions"] = [
                    "Consider adding diagrams to illustrate key concepts",
                    "Use color coding for important information",
                    "Add spatial organization with mind maps"
                ]
            
            # For active learners, add interactive elements
            if active_reflective < 0.4 and "format" in content and content["format"] == "plain_text":
                # Add interactive suggestions
                content["interactive_suggestions"] = [
                    "Try the hands-on exercises for this concept",
                    "Experiment with the simulation to apply these ideas",
                    "Form a study group to discuss these concepts"
                ]
                
                # Add practice questions
                if "Try these questions:" not in text_content:
                    text_content += "\n\n## Try these questions:\n\n1. Question 1 about the concept\n2. Question 2 about the concept"
            
            # For reflective learners, add thinking prompts
            if active_reflective > 0.6 and "format" in content and content["format"] == "plain_text":
                # Add reflection suggestions
                content["reflection_suggestions"] = [
                    "Take time to think about how this connects to what you already know",
                    "Consider the implications of these concepts",
                    "Write a summary in your own words to deepen understanding"
                ]
                
                # Add reflection prompts
                if "Reflection questions:" not in text_content:
                    text_content += "\n\n## Reflection questions:\n\n* How does this concept relate to what you already know?\n* What are the broader implications of this concept?\n* How might you apply this in different contexts?"
            
            # Update content
            content["content"] = text_content
        
        return content
    
    def _adapt_to_cognitive_profile(self, content: Any, content_type: str, learner_profile: Dict[str, Any]) -> Any:
        """Adapt content to match learner's cognitive profile
        
        Args:
            content (Any): Generated content
            content_type (str): Content type
            learner_profile (Dict[str, Any]): Learner profile
                
        Returns:
            Any: Adapted content
        """
        # Get cognitive profile
        cognitive_profile = learner_profile.get("cognitive_profile", {})
        attention_span = cognitive_profile.get("attention_span", 20)  # minutes
        working_memory = cognitive_profile.get("working_memory_capacity", 0.5)  # 0-1 scale
        processing_speed = cognitive_profile.get("processing_speed", 0.5)  # 0-1 scale
        
        # For text content
        if content_type == "text" and isinstance(content, dict) and "content" in content:
            # Adjust content based on attention span
            if attention_span < 15 and "format" in content and content["format"] == "plain_text":
                # Add pacing suggestions
                content["pacing_suggestions"] = [
                    f"Break this content into {max(2, int(content.get('reading_time', 5) / (attention_span / 60)))} sessions",
                    "Take a 5-minute break every 10 minutes",
                    "Use the Pomodoro technique: 25 minutes of focus followed by a 5-minute break"
                ]
            
            # Adjust for working memory capacity
            if working_memory < 0.4 and "format" in content and content["format"] == "plain_text":
                # Add memory aid suggestions
                content["memory_aid_suggestions"] = [
                    "Create flashcards for key concepts",
                    "Use mnemonic devices to remember important points",
                    "Create a concept map to visualize relationships"
                ]
            
            # Adjust for processing speed
            if processing_speed < 0.4 and "format" in content and content["format"] == "plain_text":
                # Add processing aid suggestions
                content["processing_aid_suggestions"] = [
                    "Read through the material multiple times",
                    "Highlight key points for easier review",
                    "Use the pre-reading questions to focus your attention"
                ]
        
        # For interactive content
        elif content_type == "interactive" and isinstance(content, dict) and "content" in content:
            # Adjust interactive content based on cognitive profile
            if "parameters" in content["content"]:
                params = content["content"]["parameters"]
                
                # Adjust timing for processing speed
                if "time_limit" in params and processing_speed < 0.5:
                    # Increase time limit for slower processing speed
                    params["time_limit"] = int(params["time_limit"] * (1.5 - processing_speed))
                    content["content"]["parameters"] = params
        
        return content

# Example usage
if __name__ == "__main__":
    # Create and initialize content generator
    generator = ContentGenerator()
    generator.initialize()
    
    # Example learner profile
    learner_profile = {
        "learning_style": {
            "visual_verbal_preference": 0.3,  # Prefers visual learning
            "active_reflective_preference": 0.6,  # Balanced active/reflective
            "sequential_global_preference": 0.7  # Slightly prefers global learning
        },
        "cognitive_profile": {
            "attention_span": 15,  # minutes
            "working_memory_capacity": 0.6,
            "processing_speed": 0.7,
            "cognitive_load_threshold": 0.7
        }
    }
    
    # Generate a text explanation
    text_content = generator.generate_content(
        "text", 
        "explanation", 
        {
            "concept": "Quadratic Equations",
            "definition": "second-degree polynomial equations in the form ax² + bx + c = 0"
        },
        learner_profile,
        "medium"
    )
    
    # Print the generated content
    print("\nGenerated Text Content:")
    if "error" in text_content:
        print(f"Error: {text_content['error']}")
    else:
        print(f"Content Type: {text_content['content_type']}/{text_content['format_type']}")
        print(f"Template Used: {text_content['template_used']}")
        print(f"Content:\n{text_content['content']['content']}")
        
        # Print adaptations if any
        for key in text_content['content']:
            if key.endswith('_suggestions'):
                print(f"\n{key.replace('_', ' ').title()}:")
                for suggestion in text_content['content'][key]:
                    print(f"- {suggestion}")
    
    # Generate an interactive quiz
    interactive_content = generator.generate_content(
        "interactive",
        "quiz",
        {
            "concept": "Quadratic Equations",
            "num_questions": 5
        },
        learner_profile,
        "easy"
    )
    
    # Print the interactive content description
    print("\nGenerated Interactive Content:")
    if "error" in interactive_content:
        print(f"Error: {interactive_content['error']}")
    else:
        print(f"Content Type: {interactive_content['content_type']}/{interactive_content['format_type']}")
        print(f"Description: {interactive_content['content']['description']}")
        print(f"Format: {interactive_content['format']}")
        print(f"Complexity: {interactive_content['content'].get('complexity', 'unknown')}")
