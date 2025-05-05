"""Data Processor Module

This module processes raw sensor data and learning interactions, extracting
meaningful features for the AI system's learning model.

Based on Ucaretron Inc.'s patent application for AI-based personalized learning systems.
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class DataProcessor:
    """Data processor for multimodal learning data
    
    This class processes:
    - Biosensor data (EEG, eye tracking, heart rate, etc.)
    - Learning interactions (clicks, time spent, content navigation)
    - Assessment results (answers, scores, response times)
    
    And extracts features related to:
    - Cognitive state (attention, engagement, cognitive load)
    - Learning style (visual/verbal, active/reflective, etc.)
    - Knowledge state (concept mastery, misconceptions)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the data processor
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings
        """
        self.is_initialized = False
        self.config = config or self._get_default_config()
        print("Data Processor created")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "processing": {
                "cognitive_state": True,
                "learning_style": True,
                "knowledge_state": True
            },
            "feature_extraction": {
                "eeg": {
                    "frequency_bands": True,  # Extract alpha, beta, etc. bands
                    "asymmetry": True,  # Hemisphere asymmetry
                    "coherence": True  # Inter-region coherence
                },
                "eye_tracking": {
                    "fixations": True,  # Fixation count and duration
                    "saccades": True,  # Rapid eye movements
                    "blink_rate": True,  # Blinks per minute
                    "pupil_dilation": True  # Pupil size changes
                },
                "heart_rate": {
                    "heart_rate_variability": True,  # HRV analysis
                    "baseline_comparison": True  # Compare to baseline
                },
                "interaction": {
                    "timing": True,  # Response times
                    "patterns": True,  # Click/navigation patterns
                    "content_focus": True  # Content viewing patterns
                },
                "advanced": {
                    "multi_modal_fusion": True,  # Combine data from multiple sources
                    "temporal_patterns": True  # Time-based patterns
                }
            },
            "machine_learning": {
                "pretrained_models": {
                    "attention_model": "./models/attention_model.pkl",
                    "engagement_model": "./models/engagement_model.pkl",
                    "cognitive_load_model": "./models/cognitive_load_model.pkl",
                    "learning_style_model": "./models/learning_style_model.pkl"
                },
                "online_learning": True,  # Update models with new data
                "transfer_learning": True  # Adapt models to individual users
            },
            "privacy": {
                "anonymization": True,  # Remove identifying information
                "edge_computing": True,  # Process sensitive data on device
                "data_minimization": True  # Only extract necessary features
            }
        }
    
    def initialize(self) -> bool:
        """Initialize the data processor
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Load machine learning models if configured
            if self.config["machine_learning"]["pretrained_models"]:
                self._load_models()
                
            self.is_initialized = True
            print("Data Processor initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing data processor: {e}")
            return False
    
    def _load_models(self) -> None:
        """Load pretrained machine learning models"""
        # This is a simplified implementation - in a real system this would
        # actually load ML models from files
        
        # Simply log that we would load models here
        model_paths = self.config["machine_learning"]["pretrained_models"]
        print("Would load these models in a real implementation:")
        for model_name, model_path in model_paths.items():
            print(f"  - {model_name}: {model_path}")
        
        # In a real implementation, we would load the models here:
        # self.attention_model = load_model(model_paths["attention_model"])
        # self.engagement_model = load_model(model_paths["engagement_model"])
        # etc.
    
    def process_activity_data(self, activity_data: Dict[str, Any], 
                            biosensor_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process learning activity data with optional biosensor data
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
            biosensor_data (Optional[Dict[str, Any]]): Biosensor data
            
        Returns:
            Dict[str, Any]: Processed activity data with extracted features
        """
        if not self.is_initialized:
            raise RuntimeError("Data Processor not initialized")
        
        # Copy input data to avoid modifying the original
        processed_data = activity_data.copy()
        
        # Add timestamp if not present
        if "timestamp" not in processed_data:
            processed_data["timestamp"] = time.time()
        
        # Extract cognitive state features if enabled and biosensor data available
        if self.config["processing"]["cognitive_state"] and biosensor_data:
            cognitive_features = self._extract_cognitive_features(activity_data, biosensor_data)
            processed_data["cognitive_evidence"] = cognitive_features
        
        # Extract learning style evidence if enabled
        if self.config["processing"]["learning_style"]:
            style_evidence = self._extract_learning_style_evidence(activity_data)
            processed_data["learning_style_evidence"] = style_evidence
        
        # Extract knowledge state evidence if enabled
        if self.config["processing"]["knowledge_state"]:
            knowledge_evidence = self._extract_knowledge_evidence(activity_data)
            processed_data["knowledge_evidence"] = knowledge_evidence
            
        return processed_data
    
    def _extract_cognitive_features(self, activity_data: Dict[str, Any], 
                                  biosensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cognitive state features from biosensor and activity data
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
            biosensor_data (Dict[str, Any]): Biosensor data
            
        Returns:
            Dict[str, Any]: Cognitive features
        """
        # This is a simplified implementation - in a real system this would
        # use actual machine learning models to extract cognitive features
        
        features = {}
        
        # Get raw biosensor features if available
        attention = biosensor_data.get("attention", 0.5)
        focus = biosensor_data.get("focus", 0.5)
        engagement = biosensor_data.get("engagement", 0.5)
        cognitive_load = biosensor_data.get("cognitive_load", 0.5)
        stress = biosensor_data.get("stress", 0.3)
        
        # Extract activity type and duration
        activity_type = activity_data.get("type", "unknown")
        duration = activity_data.get("duration", 0)
        
        # Estimate attention span based on combination of biosensor data and activity
        # Adjust attention based on activity duration relative to typical attention span
        typical_span = 1200  # 20 minutes in seconds
        duration_factor = min(1.0, duration / typical_span)
        
        # Attention span (in minutes) - combining biosensor data with activity patterns
        attention_span = 10 + 40 * (0.7 * attention + 0.3 * focus) * (1 - 0.5 * duration_factor)
        features["attention_span"] = max(5, min(50, attention_span))
        
        # Working memory capacity (0-1 scale)
        # Cognitive load is inversely related to available working memory
        features["working_memory_capacity"] = max(0.1, min(0.9, 1.0 - cognitive_load * 0.8))
        
        # Processing speed (0-1 scale)
        # Combine attention, cognitive load (inverted), and stress (inverted)
        features["processing_speed"] = max(0.1, min(0.9, 
            0.4 * attention + 
            0.4 * (1.0 - cognitive_load) + 
            0.2 * (1.0 - stress)
        ))
        
        # Optimal difficulty (0-1 scale)
        # Estimate the best difficulty level based on cognitive state
        if engagement > 0.7 and cognitive_load < 0.7:
            # Highly engaged and not overloaded - can handle higher difficulty
            features["optimal_difficulty"] = 0.7
        elif engagement < 0.3 or cognitive_load > 0.8:
            # Low engagement or overloaded - need lower difficulty
            features["optimal_difficulty"] = 0.3
        else:
            # Moderate state - middling difficulty
            features["optimal_difficulty"] = 0.5
        
        return features
    
    def _extract_learning_style_evidence(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning style evidence from activity data
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
            
        Returns:
            Dict[str, Any]: Learning style evidence
        """
        # This is a simplified implementation - in a real system this would
        # analyze detailed interaction patterns
        
        evidence = {}
        
        # Get activity type and content information
        activity_type = activity_data.get("type", "unknown")
        content_type = activity_data.get("content_type", "unknown")
        
        # For content interaction activities, extract style preferences
        if activity_type == "content_interaction":
            # Look at what type of content the user engaged with
            if content_type == "video":
                evidence["visual_verbal"] = 0.3  # More visual
            elif content_type == "text":
                evidence["visual_verbal"] = 0.7  # More verbal
            elif content_type == "interactive":
                evidence["visual_verbal"] = 0.4  # Somewhat visual
                evidence["active_reflective"] = 0.3  # More active
            
            # Look at interaction patterns
            interactions = activity_data.get("interactions", 0)
            duration = activity_data.get("duration", 300)
            
            # Calculate interaction rate (interactions per minute)
            if duration > 0:
                interaction_rate = interactions / (duration / 60)
                
                # High interaction rate suggests active learning style
                if interaction_rate > 5:
                    evidence["active_reflective"] = 0.2  # Very active
                elif interaction_rate > 2:
                    evidence["active_reflective"] = 0.4  # Somewhat active
                elif interaction_rate < 0.5:
                    evidence["active_reflective"] = 0.8  # Very reflective
        
        # For assessment activities, look at response patterns
        elif activity_type == "assessment":
            assessment = activity_data.get("assessment", {})
            
            # Look at response times
            avg_response_time = assessment.get("avg_response_time", 15)
            
            # Longer response times suggest reflective style
            if avg_response_time > 30:
                evidence["active_reflective"] = 0.8  # Very reflective
            elif avg_response_time > 20:
                evidence["active_reflective"] = 0.6  # Somewhat reflective
            elif avg_response_time < 10:
                evidence["active_reflective"] = 0.3  # More active
            
            # Look at sequential vs. global pattern
            question_sequence = assessment.get("question_sequence", [])
            
            # If user jumped around between questions, suggest global style
            if question_sequence and len(question_sequence) > len(set(question_sequence)):
                evidence["sequential_global"] = 0.7  # More global
            else:
                evidence["sequential_global"] = 0.3  # More sequential
        
        return evidence
    
    def _extract_knowledge_evidence(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge state evidence from activity data
        
        Args:
            activity_data (Dict[str, Any]): Learning activity data
            
        Returns:
            Dict[str, Any]: Knowledge state evidence
        """
        # This is a simplified implementation - in a real system this would
        # perform detailed analysis of assessment results
        
        evidence = {}
        
        # Get relevant activity data
        activity_type = activity_data.get("type", "unknown")
        concepts = activity_data.get("concepts", [])
        
        # Convert string concepts to list if needed
        if isinstance(concepts, str):
            concepts = [concepts]
        
        # Process each concept
        for concept in concepts:
            # Get concept ID - handle both string and dict formats
            concept_id = concept if isinstance(concept, str) else concept.get("id", None)
            if not concept_id:
                continue
                
            # Initialize concept evidence if not exists
            if concept_id not in evidence:
                evidence[concept_id] = {
                    "mastery_evidence": 0.0,
                    "confidence": 0.0
                }
            
            # For assessment activities, analyze results
            if activity_type == "assessment":
                assessment = activity_data.get("assessment", {})
                
                # Check if we have a score for this concept
                score = assessment.get("score", None)
                if score is not None:
                    evidence[concept_id]["mastery_evidence"] = score
                    evidence[concept_id]["confidence"] = assessment.get("confidence", 0.5)
                    
                # Look for misconceptions in answer patterns
                misconceptions = assessment.get("misconceptions", {})
                if misconceptions and concept_id in misconceptions:
                    evidence[concept_id]["misconceptions"] = misconceptions[concept_id]
            
            # For content interaction activities, analyze engagement
            elif activity_type == "content_interaction":
                # Check if we have a completion percentage
                completion = activity_data.get("completion", 0.0)
                
                # Higher completion suggests higher mastery/understanding
                if completion > 0.9:
                    # Near-complete might indicate high understanding
                    evidence[concept_id]["mastery_evidence"] = 0.8
                    evidence[concept_id]["confidence"] = 0.6
                elif completion > 0.7:
                    # Substantial completion indicates decent understanding
                    evidence[concept_id]["mastery_evidence"] = 0.6
                    evidence[concept_id]["confidence"] = 0.5
                elif completion < 0.3:
                    # Very low completion might indicate difficulty
                    evidence[concept_id]["mastery_evidence"] = 0.3
                    evidence[concept_id]["confidence"] = 0.4
        
        return evidence
    
    def process_biosensor_data(self, biosensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw biosensor data to extract cognitive features
        
        Args:
            biosensor_data (Dict[str, Any]): Raw biosensor data
            
        Returns:
            Dict[str, Any]: Processed biosensor data with extracted features
        """
        if not self.is_initialized:
            raise RuntimeError("Data Processor not initialized")
        
        # Copy input data to avoid modifying the original
        processed_data = biosensor_data.copy()
        
        # Add timestamp if not present
        if "timestamp" not in processed_data:
            processed_data["timestamp"] = time.time()
        
        # Extract EEG features if available and enabled
        if "eeg" in biosensor_data and self.config["feature_extraction"]["eeg"]["frequency_bands"]:
            eeg_features = self._extract_eeg_features(biosensor_data["eeg"])
            processed_data["eeg_features"] = eeg_features
        
        # Extract eye tracking features if available and enabled
        if "eye_tracking" in biosensor_data and self.config["feature_extraction"]["eye_tracking"]["fixations"]:
            eye_features = self._extract_eye_tracking_features(biosensor_data["eye_tracking"])
            processed_data["eye_tracking_features"] = eye_features
        
        # Extract heart rate features if available and enabled
        if "heart_rate" in biosensor_data and self.config["feature_extraction"]["heart_rate"]["heart_rate_variability"]:
            hr_features = self._extract_heart_rate_features(biosensor_data["heart_rate"])
            processed_data["heart_rate_features"] = hr_features
        
        # Perform multimodal fusion if enabled and enough data available
        if (self.config["feature_extraction"]["advanced"]["multi_modal_fusion"] and
            any(key in processed_data for key in ["eeg_features", "eye_tracking_features", "heart_rate_features"])):
            
            fused_features = self._perform_multimodal_fusion(processed_data)
            processed_data["fused_cognitive_state"] = fused_features
        
        return processed_data
    
    def _extract_eeg_features(self, eeg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from EEG data
        
        Args:
            eeg_data (Dict[str, Any]): Raw EEG data
            
        Returns:
            Dict[str, Any]: Extracted EEG features
        """
        features = {}
        
        # This is a simplified implementation - in a real system this would
        # perform proper EEG signal processing (filtering, FFT, etc.)
        
        # Get frequency band powers if available
        if "band_powers" in eeg_data:
            band_powers = eeg_data["band_powers"]
            
            # Store band powers directly
            features["band_powers"] = band_powers
            
            # Calculate attention index (beta / (theta + alpha))
            if all(band in band_powers for band in ["alpha", "beta", "theta"]):
                numerator = band_powers["beta"]
                denominator = band_powers["theta"] + band_powers["alpha"]
                
                if denominator > 0:
                    features["attention_index"] = numerator / denominator
                    # Normalize to 0-1 range with sigmoid function
                    features["attention"] = 1 / (1 + np.exp(-2 * (features["attention_index"] - 1)))
                    
            # Calculate relaxation index (alpha / beta)
            if all(band in band_powers for band in ["alpha", "beta"]):
                if band_powers["beta"] > 0:
                    features["relaxation_index"] = band_powers["alpha"] / band_powers["beta"]
                    # Normalize to 0-1 range with sigmoid function
                    features["relaxation"] = 1 / (1 + np.exp(-2 * (features["relaxation_index"] - 1)))
        
        # Calculate hemispheric asymmetry if available and configured
        if "channel_data" in eeg_data and self.config["feature_extraction"]["eeg"]["asymmetry"]:
            channels = eeg_data["channel_data"]
            
            # Simplified asymmetry calculation (would be more sophisticated in real implementation)
            if "left_frontal" in channels and "right_frontal" in channels:
                features["frontal_asymmetry"] = channels["left_frontal"] - channels["right_frontal"]
            
            if "left_temporal" in channels and "right_temporal" in channels:
                features["temporal_asymmetry"] = channels["left_temporal"] - channels["right_temporal"]
        
        return features
    
    def _extract_eye_tracking_features(self, eye_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from eye tracking data
        
        Args:
            eye_data (Dict[str, Any]): Raw eye tracking data
            
        Returns:
            Dict[str, Any]: Extracted eye tracking features
        """
        features = {}
        
        # Extract fixation data if available
        if "fixations" in eye_data and self.config["feature_extraction"]["eye_tracking"]["fixations"]:
            fixations = eye_data["fixations"]
            
            # Calculate number of fixations
            features["fixation_count"] = len(fixations)
            
            # Calculate average fixation duration
            if fixations:
                durations = [fix.get("duration", 0) for fix in fixations]
                features["avg_fixation_duration"] = sum(durations) / len(durations)
                
                # Long fixations suggest deep processing
                features["deep_processing_ratio"] = sum(1 for d in durations if d > 300) / len(durations)
        
        # Extract saccade data if available
        if "saccades" in eye_data and self.config["feature_extraction"]["eye_tracking"]["saccades"]:
            saccades = eye_data["saccades"]
            
            # Calculate number of saccades
            features["saccade_count"] = len(saccades)
            
            # Calculate average saccade velocity
            if saccades:
                velocities = [sacc.get("velocity", 0) for sacc in saccades]
                features["avg_saccade_velocity"] = sum(velocities) / len(velocities)
        
        # Extract blink data if available
        if "blinks" in eye_data and self.config["feature_extraction"]["eye_tracking"]["blink_rate"]:
            blinks = eye_data["blinks"]
            duration = eye_data.get("duration", 60)  # Default to 1 minute if not provided
            
            # Calculate blink rate (blinks per minute)
            features["blink_rate"] = len(blinks) / (duration / 60)
            
            # High blink rate may indicate fatigue
            features["fatigue_indicator"] = 1 if features["blink_rate"] > 20 else 0
        
        # Extract pupil data if available
        if "pupil" in eye_data and self.config["feature_extraction"]["eye_tracking"]["pupil_dilation"]:
            pupil = eye_data["pupil"]
            
            # Store pupil diameter directly
            features["pupil_diameter"] = pupil.get("diameter", 0)
            
            # Calculate pupil diameter change from baseline
            baseline = pupil.get("baseline", features["pupil_diameter"])
            if baseline > 0:
                features["pupil_change_ratio"] = features["pupil_diameter"] / baseline - 1
                
                # Cognitive load estimation from pupil dilation
                features["cognitive_load_indicator"] = 1 / (1 + np.exp(-5 * features["pupil_change_ratio"]))
        
        return features
    
    def _extract_heart_rate_features(self, hr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from heart rate data
        
        Args:
            hr_data (Dict[str, Any]): Raw heart rate data
            
        Returns:
            Dict[str, Any]: Extracted heart rate features
        """
        features = {}
        
        # Extract heart rate if available
        if "rate" in hr_data:
            features["heart_rate"] = hr_data["rate"]
            
            # Compare to baseline if available and configured
            if "baseline" in hr_data and self.config["feature_extraction"]["heart_rate"]["baseline_comparison"]:
                baseline = hr_data["baseline"]
                features["heart_rate_change"] = features["heart_rate"] - baseline
                features["heart_rate_ratio"] = features["heart_rate"] / baseline if baseline > 0 else 1.0
                
                # Stress indicator based on heart rate increase
                features["stress_indicator"] = max(0, min(1, (features["heart_rate_ratio"] - 1) * 3))
        
        # Extract heart rate variability if available and configured
        if "hrv" in hr_data and self.config["feature_extraction"]["heart_rate"]["heart_rate_variability"]:
            hrv = hr_data["hrv"]
            
            # Store HRV metrics directly
            features["hrv"] = hrv
            
            # Calculate cognitive load indicator from HRV
            # (Lower HRV often correlates with higher cognitive load)
            if "rmssd" in hrv:
                rmssd = hrv["rmssd"]
                features["cognitive_load_from_hrv"] = max(0, min(1, 1 - rmssd / 100))
        
        return features
    
    def _perform_multimodal_fusion(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform multimodal fusion of different biosensor features
        
        Args:
            processed_data (Dict[str, Any]): Processed data with individual features
            
        Returns:
            Dict[str, Any]: Fused cognitive state features
        """
        fused_features = {}
        
        # This is a simplified implementation - in a real system this would
        # use sophisticated fusion algorithms (e.g., Kalman filters, Bayesian nets)
        
        # Simple weighted average fusion for attention
        attention_sources = []
        
        # Add EEG-based attention if available
        if "eeg_features" in processed_data and "attention" in processed_data["eeg_features"]:
            attention_sources.append((processed_data["eeg_features"]["attention"], 0.6))  # Higher weight
        
        # Add eye tracking-based attention if available
        if "eye_tracking_features" in processed_data:
            et_features = processed_data["eye_tracking_features"]
            
            # Use deep processing ratio as a proxy for attention
            if "deep_processing_ratio" in et_features:
                attention_sources.append((et_features["deep_processing_ratio"], 0.3))
            
            # Use cognitive load from pupil as inverse attention contributor
            if "cognitive_load_indicator" in et_features:
                # Higher cognitive load might indicate focused attention (up to a point)
                cognitive_load = et_features["cognitive_load_indicator"]
                
                # Inverted U-shape relationship: moderate load is optimal
                attention_from_cl = 1 - 2 * abs(cognitive_load - 0.5)
                attention_sources.append((attention_from_cl, 0.2))
        
        # Add heart rate-based attention if available
        if "heart_rate_features" in processed_data:
            hr_features = processed_data["heart_rate_features"]
            
            # Use stress as inverse attention contributor (high stress -> lower attention)
            if "stress_indicator" in hr_features:
                attention_from_stress = 1 - hr_features["stress_indicator"]
                attention_sources.append((attention_from_stress, 0.1))  # Lower weight
        
        # Calculate weighted average attention
        if attention_sources:
            total_weight = sum(weight for _, weight in attention_sources)
            if total_weight > 0:
                fused_features["attention"] = sum(value * weight for value, weight in attention_sources) / total_weight
        
        # Similar fusion for other cognitive states (engagement, cognitive load, etc.)
        # ... (implementation for other cognitive states would follow similar pattern)
        
        # Estimate overall cognitive state
        if "attention" in fused_features:
            # Very simplified example - would be more sophisticated in real implementation
            fused_features["optimal_learning_state"] = fused_features["attention"] > 0.7
        
        return fused_features
    
    def get_last_processed_timestamp(self) -> float:
        """Get timestamp of last processed data
        
        Returns:
            float: Timestamp in seconds since epoch, or 0 if no data processed
        """
        # This is a placeholder - in a real implementation this would track
        # the timestamps of processed data
        return time.time()
    
    def apply_privacy_settings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy settings to the data based on configuration
        
        Args:
            data (Dict[str, Any]): Data to process
            
        Returns:
            Dict[str, Any]: Privacy-processed data
        """
        # Copy data to avoid modifying the original
        private_data = data.copy()
        
        # Apply anonymization if configured
        if self.config["privacy"]["anonymization"]:
            # Remove any identifying information
            for identifiable_field in ["user_id", "name", "email", "device_id"]:
                if identifiable_field in private_data:
                    private_data[identifiable_field] = f"anonymized-{hash(str(private_data[identifiable_field])) % 10000}"
        
        # Apply data minimization if configured
        if self.config["privacy"]["data_minimization"]:
            # Keep only necessary fields
            necessary_fields = [
                "timestamp", "cognitive_evidence", "learning_style_evidence", 
                "knowledge_evidence", "fused_cognitive_state"
            ]
            
            # Add feature fields if present
            if "eeg_features" in private_data:
                necessary_fields.append("eeg_features")
            if "eye_tracking_features" in private_data:
                necessary_fields.append("eye_tracking_features")
            if "heart_rate_features" in private_data:
                necessary_fields.append("heart_rate_features")
            
            # Filter to only necessary fields
            private_data = {k: v for k, v in private_data.items() if k in necessary_fields}
        
        return private_data
    
    def export_data(self, data: Dict[str, Any], format: str = "json") -> str:
        """Export processed data in the specified format
        
        Args:
            data (Dict[str, Any]): Data to export
            format (str, optional): Export format (json, csv). Defaults to "json".
            
        Returns:
            str: Exported data as string
        """
        if format.lower() == "json":
            return json.dumps(data, indent=2)
        elif format.lower() == "csv":
            # This is a simplified CSV export - would be more sophisticated in real implementation
            headers = list(data.keys())
            values = []
            
            for key in headers:
                value = data[key]
                if isinstance(value, dict):
                    # Flatten dictionaries
                    value = str(value)
                values.append(str(value))
            
            return ",".join(headers) + "\n" + ",".join(values)
        else:
            raise ValueError(f"Unsupported export format: {format}")
