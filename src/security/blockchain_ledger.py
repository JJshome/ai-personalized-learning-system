"""Blockchain Ledger Module

This module implements a simulated blockchain ledger for maintaining the
integrity of learning records. It provides immutable storage for learning
progress, achievements, and assessment results.

In a production environment, this would integrate with an actual blockchain
platform, but this implementation provides a simplified simulation for
demonstration and development purposes.
"""

import time
import json
import hashlib
import datetime
import logging
from typing import Dict, List, Any, Optional, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Block:
    """Represents a single block in the blockchain"""
    
    def __init__(self, index: int, timestamp: float, data: Dict[str, Any], 
                previous_hash: str, difficulty: int = 4):
        """Initialize a new block
        
        Args:
            index (int): Block index in the chain
            timestamp (float): Block creation timestamp
            data (Dict[str, Any]): Block data (learning records)
            previous_hash (str): Hash of the previous block
            difficulty (int, optional): Mining difficulty (proof of work). Defaults to 4.
        """
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.difficulty = difficulty
        self.hash = self.calculate_hash()
        
    def calculate_hash(self) -> str:
        """Calculate the hash of this block
        
        Returns:
            str: SHA-256 hash of the block
        """
        # Convert block data to string and hash it
        block_string = f"{self.index}{self.timestamp}{json.dumps(self.data, sort_keys=True)}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self) -> None:
        """Mine the block by finding a hash with the required difficulty"""
        target = "0" * self.difficulty
        
        while self.hash[:self.difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        
        logger.info(f"Block mined: {self.hash}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary
        
        Returns:
            Dict[str, Any]: Block as dictionary
        """
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }
    
    @classmethod
    def from_dict(cls, block_dict: Dict[str, Any]) -> 'Block':
        """Create block from dictionary
        
        Args:
            block_dict (Dict[str, Any]): Block as dictionary
            
        Returns:
            Block: Block instance
        """
        block = cls(
            block_dict["index"],
            block_dict["timestamp"],
            block_dict["data"],
            block_dict["previous_hash"]
        )
        block.nonce = block_dict["nonce"]
        block.hash = block_dict["hash"]
        return block

class BlockchainLedger:
    """Blockchain ledger for learning records
    
    This class implements a simplified blockchain to store learning records
    with immutability and integrity verification.
    """
    
    def __init__(self, difficulty: int = 4):
        """Initialize the blockchain ledger
        
        Args:
            difficulty (int, optional): Mining difficulty. Defaults to 4.
        """
        self.chain = []
        self.difficulty = difficulty
        self.pending_records = []
        
        # Create genesis block
        self._create_genesis_block()
        
        logger.info("Blockchain Ledger initialized")
    
    def _create_genesis_block(self) -> None:
        """Create the first block in the chain (genesis block)"""
        genesis_block = Block(
            0,
            time.time(),
            {"message": "Genesis Block for Learning Records", "created_at": str(datetime.datetime.now())},
            "0"  # Previous hash is 0 for genesis block
        )
        genesis_block.mine_block()
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain
        
        Returns:
            Block: Latest block
        """
        return self.chain[-1]
    
    def add_block(self, data: Dict[str, Any]) -> Block:
        """Add a new block to the chain
        
        Args:
            data (Dict[str, Any]): Data to store in the block
            
        Returns:
            Block: Newly created block
        """
        latest_block = self.get_latest_block()
        new_block = Block(
            latest_block.index + 1,
            time.time(),
            data,
            latest_block.hash,
            self.difficulty
        )
        
        # Mine the block
        new_block.mine_block()
        
        # Add to chain
        self.chain.append(new_block)
        logger.info(f"Added block {new_block.index} to chain")
        
        return new_block
    
    def is_chain_valid(self) -> bool:
        """Verify the integrity of the entire blockchain
        
        Returns:
            bool: True if chain is valid, False otherwise
        """
        # Start from block 1 (after genesis)
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block hash is correct
            if current_block.hash != current_block.calculate_hash():
                logger.warning(f"Block {current_block.index} hash is invalid")
                return False
            
            # Check if this block points to the previous block's hash
            if current_block.previous_hash != previous_block.hash:
                logger.warning(f"Block {current_block.index} has invalid previous hash")
                return False
        
        logger.info("Blockchain validation successful")
        return True
    
    def add_learning_record(self, user_id: str, record_type: str, 
                         record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add a learning record to the blockchain
        
        Args:
            user_id (str): User identifier
            record_type (str): Type of record (e.g., assessment, progress, achievement)
            record_data (Dict[str, Any]): Record data
            
        Returns:
            Dict[str, Any]: Block data with record information
        """
        # Create a record object
        record = {
            "user_id": user_id,
            "record_type": record_type,
            "data": record_data,
            "timestamp": time.time(),
            "record_id": hashlib.sha256(f"{user_id}_{record_type}_{time.time()}".encode()).hexdigest()[:16]
        }
        
        # Add to blockchain
        block_data = {
            "records": [record],
            "created_at": str(datetime.datetime.now())
        }
        
        # Add the block to the chain
        block = self.add_block(block_data)
        
        # Return record info with block reference
        result = record.copy()
        result["block_hash"] = block.hash
        result["block_index"] = block.index
        
        return result
    
    def add_pending_record(self, user_id: str, record_type: str, 
                        record_data: Dict[str, Any]) -> str:
        """Add a learning record to the pending list (for batch processing)
        
        Args:
            user_id (str): User identifier
            record_type (str): Type of record
            record_data (Dict[str, Any]): Record data
            
        Returns:
            str: Record ID
        """
        # Create a record object
        record = {
            "user_id": user_id,
            "record_type": record_type,
            "data": record_data,
            "timestamp": time.time(),
            "record_id": hashlib.sha256(f"{user_id}_{record_type}_{time.time()}".encode()).hexdigest()[:16]
        }
        
        # Add to pending records
        self.pending_records.append(record)
        
        return record["record_id"]
    
    def process_pending_records(self, batch_size: int = 10) -> Optional[Dict[str, Any]]:
        """Process pending records in batches
        
        Args:
            batch_size (int, optional): Records per block. Defaults to 10.
            
        Returns:
            Optional[Dict[str, Any]]: Block data or None if no records
        """
        if not self.pending_records:
            return None
        
        # Take up to batch_size records
        batch = self.pending_records[:batch_size]
        self.pending_records = self.pending_records[batch_size:]
        
        # Add to blockchain
        block_data = {
            "records": batch,
            "created_at": str(datetime.datetime.now()),
            "record_count": len(batch)
        }
        
        # Add the block to the chain
        block = self.add_block(block_data)
        
        # Return block info with record references
        result = block_data.copy()
        result["block_hash"] = block.hash
        result["block_index"] = block.index
        
        return result
    
    def verify_record(self, record_id: str) -> Dict[str, Any]:
        """Verify a learning record in the blockchain
        
        Args:
            record_id (str): Record identifier
            
        Returns:
            Dict[str, Any]: Verification result
        """
        # Search for the record in all blocks
        for block in self.chain:
            if "records" in block.data:
                for record in block.data["records"]:
                    if record.get("record_id") == record_id:
                        # Found the record, verify block integrity
                        is_valid = self._verify_block(block)
                        
                        return {
                            "record_found": True,
                            "block_valid": is_valid,
                            "record": record,
                            "block_index": block.index,
                            "block_hash": block.hash,
                            "verification_time": time.time()
                        }
        
        # Record not found
        return {
            "record_found": False,
            "verification_time": time.time()
        }
    
    def _verify_block(self, block: Block) -> bool:
        """Verify a single block's integrity
        
        Args:
            block (Block): Block to verify
            
        Returns:
            bool: True if block is valid, False otherwise
        """
        # Check if block hash is correct
        if block.hash != block.calculate_hash():
            return False
        
        # If not genesis block, check if it points to correct previous block
        if block.index > 0:
            previous_block = self.chain[block.index - 1]
            if block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_user_records(self, user_id: str, 
                       record_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all learning records for a user
        
        Args:
            user_id (str): User identifier
            record_type (Optional[str], optional): Filter by record type. Defaults to None.
            
        Returns:
            List[Dict[str, Any]]: List of records
        """
        records = []
        
        # Search for records in all blocks
        for block in self.chain:
            if "records" in block.data:
                for record in block.data["records"]:
                    if record.get("user_id") == user_id:
                        # Filter by record type if specified
                        if record_type is None or record.get("record_type") == record_type:
                            # Add block reference to record
                            record_with_ref = record.copy()
                            record_with_ref["block_index"] = block.index
                            record_with_ref["block_hash"] = block.hash
                            records.append(record_with_ref)
        
        return records
    
    def export_chain(self) -> List[Dict[str, Any]]:
        """Export the entire blockchain
        
        Returns:
            List[Dict[str, Any]]: List of blocks as dictionaries
        """
        return [block.to_dict() for block in self.chain]
    
    def import_chain(self, chain_data: List[Dict[str, Any]]) -> bool:
        """Import a blockchain
        
        Args:
            chain_data (List[Dict[str, Any]]): List of blocks as dictionaries
            
        Returns:
            bool: True if import successful, False otherwise
        """
        try:
            # Convert dictionaries to blocks
            new_chain = [Block.from_dict(block_dict) for block_dict in chain_data]
            
            # Verify the new chain
            self.chain = new_chain
            if not self.is_chain_valid():
                logger.error("Imported chain is invalid")
                self._create_genesis_block()  # Reset to fresh chain
                return False
            
            logger.info(f"Successfully imported blockchain with {len(new_chain)} blocks")
            return True
        except Exception as e:
            logger.error(f"Error importing blockchain: {e}")
            self._create_genesis_block()  # Reset to fresh chain
            return False

# Example usage (for documentation purposes only)
if __name__ == "__main__":
    # Create blockchain ledger
    ledger = BlockchainLedger(difficulty=2)  # Lower difficulty for quick testing
    
    # Add some sample learning records
    record1 = ledger.add_learning_record(
        "user123",
        "assessment",
        {
            "assessment_id": "math_101",
            "score": 85,
            "duration_minutes": 45,
            "questions_answered": 20,
            "date": str(datetime.datetime.now())
        }
    )
    
    record2 = ledger.add_learning_record(
        "user123",
        "progress",
        {
            "course_id": "math_101",
            "module": "algebra",
            "completion_percentage": 75,
            "time_spent_minutes": 120,
            "date": str(datetime.datetime.now())
        }
    )
    
    # Verify a record
    verification = ledger.verify_record(record1["record_id"])
    
    # Get user records
    user_records = ledger.get_user_records("user123")
    
    # Validate the blockchain
    is_valid = ledger.is_chain_valid()
