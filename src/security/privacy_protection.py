"""Privacy Protection Module

This module provides privacy protection features for the AI-based personalized
learning system, implementing the security components described in the patent.

Features include:
- Homomorphic encryption for processing encrypted data
- Differential privacy for statistical data analysis while protecting individuals
- Data anonymization and pseudonymization
- Minimization of data collection and retention
"""

import os
import json
import time
import hashlib
import secrets
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrivacyProtection:
    """Privacy protection for learning data
    
    This class implements various privacy-enhancing technologies:
    - Homomorphic encryption (simulated)
    - Differential privacy
    - Data anonymization
    - Data minimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the privacy protection module
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings
        """
        self.config = config or self._get_default_config()
        self.encryption_keys = {}
        logger.info("Privacy Protection module initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "encryption": {
                "enabled": True,
                "key_size": 2048,  # Simulated key size
                "scheme": "simulated_bfv",  # Simulated BFV scheme
                "attributes_to_encrypt": [
                    "eeg_data", "heart_rate", "eye_tracking", 
                    "personal_info", "assessment_results"
                ]
            },
            "differential_privacy": {
                "enabled": True,
                "epsilon": 1.0,  # Privacy budget (smaller = more privacy)
                "delta": 0.00001,  # Probability of privacy breach
                "mechanism": "laplace",  # Noise mechanism (laplace, gaussian)
                "attributes_to_protect": [
                    "learning_performance", "attention_metrics", 
                    "engagement_metrics", "cognitive_load"
                ]
            },
            "anonymization": {
                "enabled": True,
                "strategy": "pseudonymization",  # pseudonymization, k-anonymity
                "k_value": 5,  # For k-anonymity
                "attributes_to_anonymize": [
                    "user_id", "name", "email", "device_id", "location"
                ]
            },
            "data_minimization": {
                "enabled": True,
                "retention_period_days": 90,  # Default data retention period
                "essential_attributes": [
                    "learning_progress", "cognitive_state", "learning_style",
                    "knowledge_state", "timestamp"
                ]
            },
            "edge_computing": {
                "enabled": True,
                "sensitive_processing": [
                    "raw_eeg_analysis", "emotion_detection", 
                    "attention_calculation", "raw_biometric_processing"
                ]
            }
        }
    
    def generate_encryption_keys(self, user_id: str) -> Dict[str, str]:
        """Generate encryption keys for a user (simulated)
        
        In a real implementation, this would generate proper homomorphic
        encryption keys using a library like SEAL or HElib
        
        Args:
            user_id (str): User identifier
            
        Returns:
            Dict[str, str]: Dictionary containing public and private keys
        """
        if not self.config["encryption"]["enabled"]:
            logger.warning("Encryption is disabled, returning dummy keys")
            return {"public_key": "dummy_public_key", "private_key": "dummy_private_key"}
        
        # Simulate key generation with random tokens
        # In a real implementation, this would use proper HE libraries
        public_key = secrets.token_hex(32)
        private_key = secrets.token_hex(64)
        
        # Store keys (in a real implementation, private key would be securely stored)
        self.encryption_keys[user_id] = {
            "public_key": public_key,
            "private_key": private_key,
            "generated_at": time.time()
        }
        
        logger.info(f"Generated encryption keys for user {user_id}")
        
        # Return only the public key for normal operations
        return {"public_key": public_key}
    
    def homomorphic_encrypt(self, data: Dict[str, Any], 
                          user_id: str, 
                          public_key: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt data using homomorphic encryption (simulated)
        
        In a real implementation, this would use a library like SEAL or HElib
        to perform actual homomorphic encryption
        
        Args:
            data (Dict[str, Any]): Data to encrypt
            user_id (str): User identifier
            public_key (Optional[str], optional): Public key for encryption
            
        Returns:
            Dict[str, Any]: Encrypted data
        """
        if not self.config["encryption"]["enabled"]:
            logger.warning("Encryption is disabled, returning original data")
            return data
        
        # Get key if not provided
        if not public_key:
            if user_id not in self.encryption_keys:
                self.generate_encryption_keys(user_id)
            public_key = self.encryption_keys[user_id]["public_key"]
        
        # Copy input data to avoid modifying original
        encrypted_data = data.copy()
        
        # Get attributes to encrypt
        attributes_to_encrypt = self.config["encryption"]["attributes_to_encrypt"]
        
        # Encrypt each attribute (simulated)
        for attr in attributes_to_encrypt:
            if attr in encrypted_data:
                # In a real implementation, this would use proper HE
                # Here we just simulate with a hash-based approach
                value = encrypted_data[attr]
                
                if isinstance(value, (int, float)):
                    # Encode numeric values with a reversible transformation
                    # This is a very simplified simulation - real HE would be more complex
                    salt = hashlib.sha256(public_key.encode()).digest()[:8]
                    encoded = str(value).encode() + salt
                    encrypted_data[attr] = {
                        "encrypted": True,
                        "value": hashlib.sha256(encoded).hexdigest(),
                        "type": "numeric",
                        "schema": "simulated_bfv"
                    }
                elif isinstance(value, str):
                    # Encode string values
                    salt = hashlib.sha256(public_key.encode()).digest()[:8]
                    encoded = value.encode() + salt
                    encrypted_data[attr] = {
                        "encrypted": True,
                        "value": hashlib.sha256(encoded).hexdigest(),
                        "type": "string",
                        "schema": "simulated_bfv"
                    }
                elif isinstance(value, dict):
                    # Recursively encrypt nested dictionaries
                    encrypted_data[attr] = self.homomorphic_encrypt(value, user_id, public_key)
                elif isinstance(value, list):
                    # Encrypt lists item by item
                    if all(isinstance(item, (int, float)) for item in value):
                        # Numeric list
                        salt = hashlib.sha256(public_key.encode()).digest()[:8]
                        encrypted_data[attr] = {
                            "encrypted": True,
                            "value": [hashlib.sha256((str(v).encode() + salt)).hexdigest() for v in value],
                            "type": "numeric_list",
                            "schema": "simulated_bfv"
                        }
                    else:
                        # Mixed list - simplified approach
                        salt = hashlib.sha256(public_key.encode()).digest()[:8]
                        encrypted_data[attr] = {
                            "encrypted": True,
                            "value": hashlib.sha256((str(value).encode() + salt)).hexdigest(),
                            "type": "list",
                            "schema": "simulated_bfv"
                        }
        
        return encrypted_data
    
    def homomorphic_decrypt(self, data: Dict[str, Any], 
                          user_id: str) -> Dict[str, Any]:
        """Decrypt homomorphically encrypted data (simulated)
        
        In a real implementation, this would use a library like SEAL or HElib
        
        Args:
            data (Dict[str, Any]): Encrypted data
            user_id (str): User identifier
            
        Returns:
            Dict[str, Any]: Decrypted data
        """
        if not self.config["encryption"]["enabled"]:
            logger.warning("Encryption is disabled, returning original data")
            return data
        
        # Check if we have the private key
        if user_id not in self.encryption_keys or "private_key" not in self.encryption_keys[user_id]:
            raise ValueError(f"No private key available for user {user_id}")
        
        # Get private key
        private_key = self.encryption_keys[user_id]["private_key"]
        
        # Copy input data to avoid modifying original
        decrypted_data = data.copy()
        
        # Decrypt each attribute (simulated)
        for attr, value in list(decrypted_data.items()):
            if isinstance(value, dict) and value.get("encrypted") == True:
                # This is an encrypted value
                # In a real implementation, we would use proper HE decryption
                # Here we just acknowledge that we found encrypted data
                
                # Simplified simulation just returns a placeholder
                # In a real system, we would decrypt with the private key
                if value.get("type") == "numeric":
                    decrypted_data[attr] = 0.0  # Placeholder
                elif value.get("type") == "string":
                    decrypted_data[attr] = "decrypted_value"  # Placeholder
                elif value.get("type") == "numeric_list":
                    decrypted_data[attr] = [0.0] * len(value.get("value", []))  # Placeholder
                else:
                    decrypted_data[attr] = None  # Placeholder
            elif isinstance(value, dict) and not value.get("encrypted"):
                # Recursively decrypt nested dictionaries
                decrypted_data[attr] = self.homomorphic_decrypt(value, user_id)
        
        return decrypted_data
    
    def compute_on_encrypted(self, data: Dict[str, Any], 
                           operation: str, 
                           operands: List[str],
                           result_field: str) -> Dict[str, Any]:
        """Perform operations on encrypted data (simulated)
        
        This simulates homomorphic operations (addition, multiplication) on
        encrypted data without decrypting.
        
        Args:
            data (Dict[str, Any]): Encrypted data
            operation (str): Operation to perform (add, multiply)
            operands (List[str]): Fields to operate on
            result_field (str): Field to store the result
            
        Returns:
            Dict[str, Any]: Data with the operation result
        """
        if not self.config["encryption"]["enabled"]:
            logger.warning("Encryption is disabled, performing on plaintext")
            return self._compute_on_plaintext(data, operation, operands, result_field)
        
        # Copy input data to avoid modifying original
        result_data = data.copy()
        
        # Check if operands are encrypted
        encrypted_operands = all(
            isinstance(data.get(op), dict) and 
            data.get(op, {}).get("encrypted") == True
            for op in operands
        )
        
        if not encrypted_operands:
            logger.warning("Some operands are not encrypted, performing on plaintext")
            return self._compute_on_plaintext(data, operation, operands, result_field)
        
        # Simulate homomorphic operation
        # In a real implementation, this would use HE library operations
        
        # For this simulation, we just create a new encrypted field with the operation recorded
        result_data[result_field] = {
            "encrypted": True,
            "type": "numeric",
            "schema": "simulated_bfv",
            "value": f"hom_operation_{operation}_on_" + "_".join(operands),
            "operation": {
                "type": operation,
                "operands": operands
            }
        }
        
        return result_data
    
    def _compute_on_plaintext(self, data: Dict[str, Any], 
                            operation: str, 
                            operands: List[str],
                            result_field: str) -> Dict[str, Any]:
        """Helper to perform operations on plaintext data
        
        Args:
            data (Dict[str, Any]): Data
            operation (str): Operation to perform (add, multiply)
            operands (List[str]): Fields to operate on
            result_field (str): Field to store the result
            
        Returns:
            Dict[str, Any]: Data with the operation result
        """
        # Copy input data to avoid modifying original
        result_data = data.copy()
        
        # Extract values, defaulting to 0 if not present
        values = [float(data.get(op, 0)) for op in operands]
        
        # Perform operation
        if operation == "add":
            result_data[result_field] = sum(values)
        elif operation == "multiply":
            # Start with 1 for multiplication
            product = 1
            for value in values:
                product *= value
            result_data[result_field] = product
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return result_data
    
    def add_differential_privacy_noise(self, data: Dict[str, Any], 
                                    user_id: Optional[str] = None) -> Dict[str, Any]:
        """Add noise to data for differential privacy
        
        Args:
            data (Dict[str, Any]): Data to add noise to
            user_id (Optional[str], optional): User identifier
            
        Returns:
            Dict[str, Any]: Data with noise added
        """
        if not self.config["differential_privacy"]["enabled"]:
            logger.warning("Differential privacy is disabled, returning original data")
            return data
        
        # Copy input data to avoid modifying original
        noisy_data = data.copy()
        
        # Get privacy parameters
        epsilon = self.config["differential_privacy"]["epsilon"]
        delta = self.config["differential_privacy"]["delta"]
        mechanism = self.config["differential_privacy"]["mechanism"]
        attributes = self.config["differential_privacy"]["attributes_to_protect"]
        
        # Add noise to each attribute
        for attr in attributes:
            if attr in noisy_data and isinstance(noisy_data[attr], (int, float)):
                # Calculate sensitivity (assumed 1.0 for simplicity)
                # In a real implementation, this would be carefully calculated
                sensitivity = 1.0
                
                # Add noise based on the selected mechanism
                if mechanism == "laplace":
                    # Laplace noise with scale = sensitivity / epsilon
                    scale = sensitivity / epsilon
                    noise = self._generate_laplace_noise(scale)
                    noisy_data[attr] += noise
                elif mechanism == "gaussian":
                    # Gaussian noise with std = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
                    # Simplified version
                    import math
                    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
                    noise = self._generate_gaussian_noise(sigma)
                    noisy_data[attr] += noise
                else:
                    logger.warning(f"Unknown mechanism {mechanism}, not adding noise")
            elif attr in noisy_data and isinstance(noisy_data[attr], dict):
                # Recursively add noise to nested dictionaries
                noisy_data[attr] = self.add_differential_privacy_noise(noisy_data[attr], user_id)
        
        return noisy_data
    
    def _generate_laplace_noise(self, scale: float) -> float:
        """Generate Laplace distributed noise
        
        Args:
            scale (float): Scale parameter
            
        Returns:
            float: Random noise value
        """
        import random
        import math
        
        # Generate uniform random number in (0, 1)
        u = random.random()
        while u == 0 or u == 1:
            u = random.random()
        
        # Convert to Laplace distribution
        if u < 0.5:
            return scale * math.log(2 * u)
        else:
            return -scale * math.log(2 * (1 - u))
    
    def _generate_gaussian_noise(self, sigma: float) -> float:
        """Generate Gaussian distributed noise
        
        Args:
            sigma (float): Standard deviation
            
        Returns:
            float: Random noise value
        """
        import random
        import math
        
        # Box-Muller transform to generate Gaussian noise
        u1 = random.random()
        u2 = random.random()
        
        while u1 == 0:
            u1 = random.random()
        
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return z0 * sigma
    
    def anonymize_data(self, data: Dict[str, Any], 
                     user_id: Optional[str] = None) -> Dict[str, Any]:
        """Anonymize data by removing or transforming identifying information
        
        Args:
            data (Dict[str, Any]): Data to anonymize
            user_id (Optional[str], optional): User identifier
            
        Returns:
            Dict[str, Any]: Anonymized data
        """
        if not self.config["anonymization"]["enabled"]:
            logger.warning("Anonymization is disabled, returning original data")
            return data
        
        # Copy input data to avoid modifying original
        anon_data = data.copy()
        
        # Get anonymization settings
        strategy = self.config["anonymization"]["strategy"]
        attributes = self.config["anonymization"]["attributes_to_anonymize"]
        
        # Process each attribute to anonymize
        for attr in attributes:
            if attr in anon_data:
                if strategy == "pseudonymization":
                    # Replace with pseudonym (hashed value)
                    salt = str(time.time()).encode()
                    
                    if isinstance(anon_data[attr], str):
                        # Hash string values
                        anon_data[attr] = hashlib.sha256(
                            anon_data[attr].encode() + salt
                        ).hexdigest()[:12]
                    else:
                        # Convert to string and hash for non-string values
                        anon_data[attr] = hashlib.sha256(
                            str(anon_data[attr]).encode() + salt
                        ).hexdigest()[:12]
                    
                    # Add pseudonym prefix for clarity
                    anon_data[attr] = f"pseudo_{anon_data[attr]}"
                    
                elif strategy == "k-anonymity":
                    # For k-anonymity, we would group similar records
                    # This is a simplified implementation as proper k-anonymity
                    # requires looking at the entire dataset, not just one record
                    
                    # For now, we'll just generalize the values
                    if isinstance(anon_data[attr], str):
                        # Keep only first character for strings
                        if len(anon_data[attr]) > 0:
                            anon_data[attr] = anon_data[attr][0] + "*" * (len(anon_data[attr]) - 1)
                    elif isinstance(anon_data[attr], (int, float)):
                        # Round numeric values to reduce precision
                        anon_data[attr] = round(anon_data[attr], -1)  # Round to nearest 10
        
        return anon_data
    
    def apply_data_minimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data minimization by keeping only essential attributes
        
        Args:
            data (Dict[str, Any]): Data to minimize
            
        Returns:
            Dict[str, Any]: Minimized data
        """
        if not self.config["data_minimization"]["enabled"]:
            logger.warning("Data minimization is disabled, returning original data")
            return data
        
        # Get essential attributes
        essential_attrs = self.config["data_minimization"]["essential_attributes"]
        
        # Keep only essential attributes
        minimized_data = {
            attr: data[attr] for attr in essential_attrs 
            if attr in data
        }
        
        # Always keep timestamp if present
        if "timestamp" in data:
            minimized_data["timestamp"] = data["timestamp"]
        
        return minimized_data
    
    def check_retention_policy(self, data: Dict[str, Any]) -> bool:
        """Check if data should be retained based on retention policy
        
        Args:
            data (Dict[str, Any]): Data to check
            
        Returns:
            bool: True if data should be retained, False if it should be deleted
        """
        if not self.config["data_minimization"]["enabled"]:
            # If data minimization is disabled, always retain
            return True
        
        # Get retention period in seconds
        retention_period = self.config["data_minimization"]["retention_period_days"] * 24 * 60 * 60
        
        # Check if timestamp is present
        if "timestamp" not in data:
            # If no timestamp, default to retain
            logger.warning("No timestamp in data, defaulting to retain")
            return True
        
        # Calculate age of data
        data_age = time.time() - data["timestamp"]
        
        # Retain if within retention period
        return data_age <= retention_period
    
    def get_edge_computing_config(self) -> Dict[str, Any]:
        """Get configuration for edge computing
        
        Returns:
            Dict[str, Any]: Edge computing configuration
        """
        if not self.config["edge_computing"]["enabled"]:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "sensitive_processing": self.config["edge_computing"]["sensitive_processing"]
        }
    
    def create_blockchain_record(self, data: Dict[str, Any], 
                               user_id: str) -> Dict[str, Any]:
        """Create a blockchain record for immutable learning history (simulated)
        
        In a real implementation, this would interact with an actual blockchain
        
        Args:
            data (Dict[str, Any]): Data to record
            user_id (str): User identifier
            
        Returns:
            Dict[str, Any]: Blockchain record
        """
        # This is a simplified simulation - a real implementation would use
        # an actual blockchain library/platform
        
        # Create a hash of the data
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Create a timestamp
        timestamp = time.time()
        
        # Create a blockchain record
        record = {
            "user_id": user_id,
            "data_hash": data_hash,
            "timestamp": timestamp,
            "block_id": "sim_" + hashlib.sha256(f"{user_id}_{timestamp}".encode()).hexdigest()[:16]
        }
        
        logger.info(f"Created blockchain record for user {user_id}: {record['block_id']}")
        
        return record
    
    def verify_blockchain_record(self, data: Dict[str, Any], 
                               record: Dict[str, Any]) -> bool:
        """Verify data against a blockchain record (simulated)
        
        In a real implementation, this would interact with an actual blockchain
        
        Args:
            data (Dict[str, Any]): Data to verify
            record (Dict[str, Any]): Blockchain record
            
        Returns:
            bool: True if data matches record, False otherwise
        """
        # Calculate hash of the data
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()
        
        # Compare with record hash
        matches = data_hash == record["data_hash"]
        
        if matches:
            logger.info(f"Verified blockchain record {record['block_id']}: VALID")
        else:
            logger.warning(f"Verified blockchain record {record['block_id']}: INVALID")
        
        return matches

# Example usage (for documentation purposes only)
if __name__ == "__main__":
    # Create privacy protection module
    privacy = PrivacyProtection()
    
    # Sample data
    user_data = {
        "user_id": "user123",
        "name": "John Doe",
        "email": "john@example.com",
        "eeg_data": [0.5, 0.6, 0.7, 0.8],
        "heart_rate": 75,
        "engagement_metrics": 0.8,
        "attention_metrics": 0.7,
        "cognitive_load": 0.4,
        "learning_performance": 0.85,
        "timestamp": time.time()
    }
    
    # Generate encryption keys
    keys = privacy.generate_encryption_keys("user123")
    
    # Encrypt data
    encrypted_data = privacy.homomorphic_encrypt(user_data, "user123")
    
    # Perform computation on encrypted data
    result = privacy.compute_on_encrypted(
        encrypted_data, "add", 
        ["engagement_metrics", "attention_metrics"], 
        "combined_metrics"
    )
    
    # Add differential privacy noise
    noisy_data = privacy.add_differential_privacy_noise(user_data)
    
    # Anonymize data
    anon_data = privacy.anonymize_data(user_data)
    
    # Apply data minimization
    min_data = privacy.apply_data_minimization(user_data)
    
    # Create blockchain record
    record = privacy.create_blockchain_record(user_data, "user123")
    
    # Verify blockchain record
    is_valid = privacy.verify_blockchain_record(user_data, record)
