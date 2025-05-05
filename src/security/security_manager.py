"""Security Manager Module

This module serves as the primary interface to the security components of the
AI-based personalized learning system. It integrates privacy protection and
blockchain ledger functionality, providing a unified security layer for the system.

Features:
- Secure data processing with privacy protections
- Record integrity verification via blockchain
- Encrypted data storage and transmission
- User authentication and authorization
- Audit logging for security events
"""

import time
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Callable

from src.security.privacy_protection import PrivacyProtection
from src.security.blockchain_ledger import BlockchainLedger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    """Security manager for the AI-based personalized learning system
    
    This class integrates various security components to provide a comprehensive
    security layer for the system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the security manager
        
        Args:
            config (Optional[Dict[str, Any]], optional): Configuration settings
        """
        self.config = config or self._get_default_config()
        
        # Initialize privacy protection module
        privacy_config = self.config.get("privacy", {})
        self.privacy = PrivacyProtection(privacy_config)
        
        # Initialize blockchain ledger
        blockchain_config = self.config.get("blockchain", {})
        self.blockchain = BlockchainLedger(
            difficulty=blockchain_config.get("difficulty", 4)
        )
        
        # Initialize authentication cache
        self.auth_cache = {}
        
        # Initialize audit log
        self.audit_log = []
        
        logger.info("Security Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "privacy": {
                "encryption": {
                    "enabled": True,
                    "key_size": 2048,
                    "scheme": "simulated_bfv"
                },
                "differential_privacy": {
                    "enabled": True,
                    "epsilon": 1.0
                },
                "anonymization": {
                    "enabled": True
                },
                "data_minimization": {
                    "enabled": True
                }
            },
            "blockchain": {
                "enabled": True,
                "difficulty": 4,
                "batch_processing": True,
                "batch_size": 10
            },
            "authentication": {
                "session_timeout_minutes": 60,
                "max_failed_attempts": 5,
                "lockout_duration_minutes": 30
            },
            "audit": {
                "enabled": True,
                "log_level": "INFO",
                "events_to_log": [
                    "authentication", "data_access", "encryption", 
                    "blockchain", "privacy", "security_config"
                ]
            }
        }
    
    def secure_data(self, data: Dict[str, Any], 
                  user_id: str, 
                  data_type: str,
                  operation: str = "process") -> Dict[str, Any]:
        """Process data through the security pipeline
        
        This is the main method for securing data in the system. It applies
        privacy protections, logs to blockchain, and ensures data protection.
        
        Args:
            data (Dict[str, Any]): Data to secure
            user_id (str): User identifier
            data_type (str): Type of data (e.g., learning_data, assessment, biometric)
            operation (str, optional): Operation being performed. Defaults to "process".
            
        Returns:
            Dict[str, Any]: Secured data
        """
        secured_data = data.copy()
        
        # Apply privacy protections
        if self.config["privacy"]["encryption"]["enabled"]:
            # Generate keys if needed
            if user_id not in self.privacy.encryption_keys:
                self.privacy.generate_encryption_keys(user_id)
            
            # Apply homomorphic encryption
            secured_data = self.privacy.homomorphic_encrypt(secured_data, user_id)
            self._log_audit_event("encryption", f"Encrypted {data_type} data for user {user_id}")
        
        # Apply differential privacy
        if self.config["privacy"]["differential_privacy"]["enabled"]:
            secured_data = self.privacy.add_differential_privacy_noise(secured_data, user_id)
            self._log_audit_event("privacy", f"Applied differential privacy to {data_type} data")
        
        # Apply anonymization
        if self.config["privacy"]["anonymization"]["enabled"]:
            secured_data = self.privacy.anonymize_data(secured_data, user_id)
            self._log_audit_event("privacy", f"Anonymized {data_type} data")
        
        # Apply data minimization
        if self.config["privacy"]["data_minimization"]["enabled"]:
            secured_data = self.privacy.apply_data_minimization(secured_data)
            self._log_audit_event("privacy", f"Applied data minimization to {data_type} data")
        
        # Record to blockchain if enabled
        if self.config["blockchain"]["enabled"]:
            if self.config["blockchain"]["batch_processing"]:
                # Add to pending records for batch processing
                record_id = self.blockchain.add_pending_record(user_id, data_type, {
                    "operation": operation,
                    "data_hash": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                    "timestamp": time.time()
                })
                secured_data["_metadata"] = secured_data.get("_metadata", {})
                secured_data["_metadata"]["record_id"] = record_id
                self._log_audit_event("blockchain", f"Added {data_type} data to pending blockchain records")
            else:
                # Add directly to blockchain
                record = self.blockchain.add_learning_record(user_id, data_type, {
                    "operation": operation,
                    "data_hash": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                    "timestamp": time.time()
                })
                secured_data["_metadata"] = secured_data.get("_metadata", {})
                secured_data["_metadata"]["record_id"] = record["record_id"]
                secured_data["_metadata"]["block_hash"] = record["block_hash"]
                secured_data["_metadata"]["block_index"] = record["block_index"]
                self._log_audit_event("blockchain", f"Added {data_type} data to blockchain")
        
        return secured_data
    
    def process_pending_blockchain_records(self) -> Optional[Dict[str, Any]]:
        """Process pending blockchain records
        
        Returns:
            Optional[Dict[str, Any]]: Processing result or None if no records
        """
        if not self.config["blockchain"]["enabled"] or not self.config["blockchain"]["batch_processing"]:
            return None
        
        result = self.blockchain.process_pending_records(
            batch_size=self.config["blockchain"]["batch_size"]
        )
        
        if result:
            self._log_audit_event(
                "blockchain", 
                f"Processed {result['record_count']} pending blockchain records"
            )
        
        return result
    
    def verify_data_integrity(self, record_id: str) -> Dict[str, Any]:
        """Verify the integrity of data using blockchain
        
        Args:
            record_id (str): Record identifier
            
        Returns:
            Dict[str, Any]: Verification result
        """
        if not self.config["blockchain"]["enabled"]:
            return {"verified": False, "reason": "Blockchain verification disabled"}
        
        result = self.blockchain.verify_record(record_id)
        
        if result["record_found"]:
            self._log_audit_event(
                "blockchain", 
                f"Verified record {record_id}: {'VALID' if result['block_valid'] else 'INVALID'}"
            )
        else:
            self._log_audit_event(
                "blockchain", 
                f"Record not found: {record_id}"
            )
        
        return result
    
    def decrypt_data(self, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Decrypt secured data
        
        Args:
            data (Dict[str, Any]): Encrypted data
            user_id (str): User identifier
            
        Returns:
            Dict[str, Any]: Decrypted data
        """
        if not self.config["privacy"]["encryption"]["enabled"]:
            return data
        
        # Decrypt data
        decrypted_data = self.privacy.homomorphic_decrypt(data, user_id)
        self._log_audit_event("encryption", f"Decrypted data for user {user_id}")
        
        return decrypted_data
    
    def authenticate_user(self, user_id: str, 
                        auth_token: str, 
                        client_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Authenticate a user
        
        This is a simplified authentication method. In a real system,
        this would validate credentials against a secure user database.
        
        Args:
            user_id (str): User identifier
            auth_token (str): Authentication token (e.g., password hash)
            client_info (Optional[Dict[str, Any]], optional): Client information
            
        Returns:
            Dict[str, Any]: Authentication result
        """
        # This is a simulated authentication
        # In a real system, this would validate against a secure database
        
        # Simulated authentication - always succeeds with correct format
        # In a real system, would check credentials properly
        is_valid = len(auth_token) >= 32
        
        if is_valid:
            # Create session
            session_token = hashlib.sha256(f"{user_id}_{time.time()}_{auth_token}".encode()).hexdigest()
            session_expiry = time.time() + (self.config["authentication"]["session_timeout_minutes"] * 60)
            
            # Store in cache
            self.auth_cache[session_token] = {
                "user_id": user_id,
                "expiry": session_expiry,
                "client_info": client_info or {}
            }
            
            self._log_audit_event("authentication", f"User {user_id} authenticated successfully")
            
            return {
                "authenticated": True,
                "session_token": session_token,
                "expiry": session_expiry
            }
        else:
            self._log_audit_event("authentication", f"Authentication failed for user {user_id}")
            
            return {
                "authenticated": False,
                "reason": "Invalid credentials"
            }
    
    def validate_session(self, session_token: str) -> Dict[str, Any]:
        """Validate a user session
        
        Args:
            session_token (str): Session token
            
        Returns:
            Dict[str, Any]: Validation result
        """
        if session_token not in self.auth_cache:
            return {"valid": False, "reason": "Invalid session"}
        
        session = self.auth_cache[session_token]
        
        # Check if session has expired
        if session["expiry"] < time.time():
            # Clean up expired session
            del self.auth_cache[session_token]
            self._log_audit_event("authentication", f"Session expired for user {session['user_id']}")
            return {"valid": False, "reason": "Session expired"}
        
        # Session is valid
        self._log_audit_event("authentication", f"Session validated for user {session['user_id']}")
        return {
            "valid": True,
            "user_id": session["user_id"],
            "expiry": session["expiry"]
        }
    
    def authorize_access(self, user_id: str, 
                       resource_type: str, 
                       operation: str) -> Dict[str, Any]:
        """Authorize a user's access to a resource
        
        This is a simplified authorization method. In a real system,
        this would check against defined permissions and roles.
        
        Args:
            user_id (str): User identifier
            resource_type (str): Type of resource
            operation (str): Operation to perform
            
        Returns:
            Dict[str, Any]: Authorization result
        """
        # This is a simulated authorization
        # In a real system, this would check against defined permissions
        
        # Simulated authorization rules
        # Students can read their own data
        # Teachers can read and write student data
        # Admins can do anything
        
        # Extract role from user_id (simplified approach)
        role = "student"  # Default role
        if user_id.startswith("teacher_"):
            role = "teacher"
        elif user_id.startswith("admin_"):
            role = "admin"
        
        # Check permission based on role
        is_authorized = False
        reason = "Permission denied"
        
        if role == "admin":
            # Admins can do anything
            is_authorized = True
        elif role == "teacher":
            # Teachers can read and write student data
            if resource_type.startswith("student_") and operation in ["read", "write"]:
                is_authorized = True
            elif operation == "read":
                is_authorized = True
        elif role == "student":
            # Students can read their own data
            if resource_type.startswith(f"student_{user_id}") and operation == "read":
                is_authorized = True
            elif resource_type == "content" and operation == "read":
                is_authorized = True
        
        if is_authorized:
            self._log_audit_event("authorization", f"User {user_id} authorized for {operation} on {resource_type}")
        else:
            self._log_audit_event("authorization", f"User {user_id} denied {operation} on {resource_type}: {reason}")
        
        return {
            "authorized": is_authorized,
            "reason": reason if not is_authorized else None
        }
    
    def _log_audit_event(self, event_type: str, message: str) -> None:
        """Log a security audit event
        
        Args:
            event_type (str): Type of event
            message (str): Event message
        """
        if not self.config["audit"]["enabled"]:
            return
        
        # Check if this event type should be logged
        if event_type not in self.config["audit"]["events_to_log"]:
            return
        
        # Create audit log entry
        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "message": message
        }
        
        # Add to in-memory log
        self.audit_log.append(audit_entry)
        
        # Log to file (in a real system)
        logger.info(f"AUDIT: [{event_type}] {message}")
    
    def get_audit_log(self, 
                    event_type: Optional[str] = None,
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get filtered audit log entries
        
        Args:
            event_type (Optional[str], optional): Filter by event type
            start_time (Optional[float], optional): Filter by start time
            end_time (Optional[float], optional): Filter by end time
            
        Returns:
            List[Dict[str, Any]]: Filtered audit log entries
        """
        filtered_log = self.audit_log
        
        # Apply filters
        if event_type:
            filtered_log = [entry for entry in filtered_log if entry["event_type"] == event_type]
        
        if start_time:
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] >= start_time]
        
        if end_time:
            filtered_log = [entry for entry in filtered_log if entry["timestamp"] <= end_time]
        
        return filtered_log
    
    def is_blockchain_valid(self) -> bool:
        """Check if the blockchain is valid
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not self.config["blockchain"]["enabled"]:
            return True
        
        is_valid = self.blockchain.is_chain_valid()
        
        self._log_audit_event(
            "blockchain", 
            f"Blockchain validation: {'VALID' if is_valid else 'INVALID'}"
        )
        
        return is_valid
    
    def get_user_records(self, user_id: str, 
                       record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get blockchain records for a user
        
        Args:
            user_id (str): User identifier
            record_type (Optional[str], optional): Filter by record type
            
        Returns:
            List[Dict[str, Any]]: User records
        """
        if not self.config["blockchain"]["enabled"]:
            return []
        
        records = self.blockchain.get_user_records(user_id, record_type)
        
        self._log_audit_event(
            "blockchain", 
            f"Retrieved {len(records)} records for user {user_id}"
        )
        
        return records
    
    def export_security_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export security configuration
        
        Args:
            include_sensitive (bool, optional): Include sensitive settings. Defaults to False.
            
        Returns:
            Dict[str, Any]: Security configuration
        """
        config = self.config.copy()
        
        # Remove sensitive information if not requested
        if not include_sensitive:
            if "privacy" in config and "encryption" in config["privacy"]:
                # Replace key settings with indicator
                config["privacy"]["encryption"]["key_details"] = "[REDACTED]"
            
            if "authentication" in config:
                # Replace sensitive authentication settings
                for sensitive_key in ["password_hash_algo", "salt_length"]:
                    if sensitive_key in config["authentication"]:
                        config["authentication"][sensitive_key] = "[REDACTED]"
        
        self._log_audit_event(
            "security_config", 
            f"Exported security configuration (sensitive info: {include_sensitive})"
        )
        
        return config

# Example usage (for documentation purposes only)
if __name__ == "__main__":
    # Create security manager
    security = SecurityManager()
    
    # Authenticate a user
    auth_result = security.authenticate_user(
        "student_123",
        "simulated_auth_token_32chars_long_min"
    )
    
    if auth_result["authenticated"]:
        # Secure some learning data
        learning_data = {
            "user_id": "student_123",
            "course_id": "math_101",
            "assessment_score": 85,
            "attention_level": 0.9,
            "timestamp": time.time()
        }
        
        secured_data = security.secure_data(
            learning_data,
            "student_123",
            "assessment_result"
        )
        
        # Verify data
        record_id = secured_data["_metadata"]["record_id"]
        verification = security.verify_data_integrity(record_id)
        
        # Check authorization
        auth_check = security.authorize_access(
            "student_123",
            "student_student_123_data",
            "read"
        )
