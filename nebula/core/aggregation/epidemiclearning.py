import gc
import logging
import random
import torch
import numpy as np
from typing import Dict, Set, Tuple

from nebula.core.aggregation.aggregator import Aggregator


class EpidemicLearning(Aggregator):
    """
    Aggregator: Epidemic Learning (Gossip-based Decentralized Federated Learning)
    
    This implements a synchronous epidemic learning algorithm for decentralized federated learning.
    The algorithm is inspired by epidemic protocols where "infection" (model updates) spreads
    through the network via gossip communication.
    
    Key characteristics:
    - Gossip-based: Nodes randomly select neighbors to communicate with
    - Probabilistic: Communication happens with configurable probability
    - Fanout-controlled: Limited number of neighbors contacted per round
    - Epidemic rounds: Multiple gossip rounds ensure network-wide propagation
    - Robust: Natural fault tolerance through redundant communication paths
    
    The algorithm works in phases:
    1. Local Training: Each node trains on its local data
    2. Gossip Phase: Multiple sub-rounds of probabilistic model exchange
    3. Aggregation: Received models are aggregated using weighted averaging
    4. Synchronization: All nodes complete before next training round
    
    Authors: Based on epidemic protocols literature and gossip algorithms
    References: 
    - Kempe et al. "Gossip-based computation of aggregate information" (2003)
    - Boyd et al. "Randomized gossip algorithms" (2006)
    - Jin et al. "Local SGD for Decentralized Federated Learning" (2019)
    """

    def __init__(self, config=None, **kwargs):
        # Epidemic Learning specific configuration
        epidemic_config = config.participant.get("epidemic_config", {})
        aggregator_args = config.participant.get("aggregator_args", {})
        
        # Core epidemic parameters
        self.gossip_probability = epidemic_config.get("gossip_probability", aggregator_args.get("gossip_probability", 0.8))
        self.fanout = epidemic_config.get("fanout", aggregator_args.get("fanout", 3))
        self.epidemic_rounds = epidemic_config.get("epidemic_rounds", aggregator_args.get("epidemic_rounds", 2))
        self.sync_timeout = epidemic_config.get("sync_timeout", 120)
        
        # Advanced parameters
        self.adaptive_fanout = epidemic_config.get("adaptive_fanout", False)
        self.infection_threshold = epidemic_config.get("infection_threshold", 0.1)
        self.recovery_probability = epidemic_config.get("recovery_probability", 0.0)
        self.topology_awareness = epidemic_config.get("topology_awareness", True)
        
        # Aggregation parameters
        self.weighted_averaging = epidemic_config.get("weighted_averaging", True)
        self.age_decay_factor = epidemic_config.get("age_decay_factor", 0.95)
        
        # Initialize parent - temporarily set federation to EpidemicLearning for correct handler
        original_federation = config.participant["scenario_args"]["federation"]
        config.participant["scenario_args"]["federation"] = "EpidemicLearning"
        
        super().__init__(config, **kwargs)
        
        # Restore original federation type
        config.participant["scenario_args"]["federation"] = original_federation
        
        # Initialize epidemic state
        self.current_epidemic_round = 0
        self.infection_history = {}  # Track which nodes we've "infected"
        self.immunity_status = {}    # Track node immunity/recovery status
        self.gossip_statistics = {
            'total_gossips': 0,
            'successful_infections': 0,
            'failed_contacts': 0,
            'rounds_completed': 0
        }
        
        logging.info(f"[Epidemic Learning] Initialized with configuration:")
        logging.info(f"  - Gossip probability: {self.gossip_probability}")
        logging.info(f"  - Fanout: {self.fanout}")
        logging.info(f"  - Epidemic rounds: {self.epidemic_rounds}")
        logging.info(f"  - Sync timeout: {self.sync_timeout}s")
        logging.info(f"  - Adaptive fanout: {self.adaptive_fanout}")
        logging.info(f"  - Weighted averaging: {self.weighted_averaging}")
        logging.info("[Epidemic Learning] Using custom EpidemicLearning update handler")

    def run_aggregation(self, models: Dict[str, Tuple]) -> Dict[str, torch.Tensor]:
        """
        Perform epidemic learning aggregation with gossip-based model averaging.
        
        In epidemic learning, we aggregate models received through gossip communication.
        The aggregation considers:
        - Model age (older models get lower weight)
        - Source reliability (based on past successful communications)
        - Infection strength (how many times we've seen updates from this source)
        
        Args:
            models (dict): Dictionary mapping node_id -> (model_parameters, metadata)
                          where metadata contains weight, timestamp, gossip_round, etc.
        
        Returns:
            dict: Averaged model parameters after epidemic aggregation
        """
        super().run_aggregation(models)
        
        if not models:
            raise ValueError("No models received for epidemic learning aggregation")
        
        models_list = list(models.values())
        num_models = len(models_list)
        
        logging.info(f"[Epidemic Learning] Aggregating {num_models} models from gossip communication")
        
        # Initialize accumulator
        first_model_params = models_list[0][0]
        accum = {layer: torch.zeros_like(param, dtype=torch.float32) 
                for layer, param in first_model_params.items()}
        
        # Compute aggregation weights based on epidemic learning principles
        if self.weighted_averaging:
            weights = self._compute_epidemic_weights(models)
        else:
            # Simple equal weighting
            weights = {node_id: 1.0 / num_models for node_id in models.keys()}
        
        # Perform weighted aggregation
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Total aggregation weight is zero")
        
        with torch.no_grad():
            for node_id, (model_parameters, metadata) in models.items():
                node_weight = weights[node_id] / total_weight
                
                for layer in accum:
                    accum[layer].add_(
                        model_parameters[layer].to(accum[layer].dtype),
                        alpha=node_weight
                    )
        
        # Update epidemic statistics
        self.gossip_statistics['rounds_completed'] += 1
        self.gossip_statistics['successful_infections'] += num_models
        
        # Clean up memory
        del models_list
        gc.collect()
        
        logging.info(f"[Epidemic Learning] Completed aggregation with {num_models} models")
        logging.info(f"[Epidemic Learning] Total epidemic rounds completed: {self.gossip_statistics['rounds_completed']}")
        
        return accum

    def _compute_epidemic_weights(self, models) -> dict:
        """
        Compute aggregation weights based on epidemic learning principles.
        
        Weights are based on:
        1. Recency: More recent models get higher weight
        2. Infection strength: Models from frequently communicating nodes
        3. Network position: Central nodes may get higher weight
        4. Reliability: Nodes with successful past communications
        
        Args:
            models: Dictionary of received models with metadata
            
        Returns:
            Dict mapping node_id to aggregation weight
        """
        weights = {}
        current_time = self._get_current_time()
        
        # Find the most recent timestamp to normalize age calculations
        timestamps = []
        for node_id, (model_params, metadata) in models.items():
            if isinstance(metadata, dict) and 'timestamp' in metadata:
                timestamps.append(metadata['timestamp'])
        
        # Use relative age instead of absolute age to avoid numerical issues
        max_timestamp = max(timestamps) if timestamps else current_time
        
        for node_id, (model_params, metadata) in models.items():
            weight = 1.0
            
            # Age-based weighting (recent models preferred)
            if isinstance(metadata, dict) and 'timestamp' in metadata:
                # Use relative age in seconds, capped at reasonable values
                relative_age = max_timestamp - metadata['timestamp']
                # Convert to a reasonable scale (e.g., minutes) and cap
                age_in_minutes = min(relative_age / 60.0, 60.0)  # Cap at 1 hour
                age_weight = self.age_decay_factor ** age_in_minutes
                weight *= age_weight
            
            # Infection history weighting
            if node_id in self.infection_history:
                infection_count = self.infection_history[node_id]
                # Nodes we've successfully communicated with more get slightly higher weight
                infection_weight = 1.0 + 0.1 * min(infection_count / 10.0, 1.0)
                weight *= infection_weight
            
            # Gossip round weighting (models from later gossip rounds may be more refined)
            if isinstance(metadata, dict) and 'gossip_round' in metadata:
                round_weight = 1.0 + 0.05 * metadata['gossip_round']
                weight *= round_weight
            
            # Ensure minimum weight to avoid zero weights
            weight = max(weight, 0.01)
            weights[node_id] = weight
        
        return weights

    def _get_current_time(self) -> float:
        """Get current timestamp for age-based weighting."""
        import time
        return time.time()

    def select_gossip_targets(self, available_neighbors: Set[str]) -> Set[str]:
        """
        Select which neighbors to gossip with in epidemic learning.
        
        Selection strategy:
        1. Random selection with fanout limit
        2. Probability-based filtering
        3. Avoid recently contacted nodes (if configured)
        4. Adaptive fanout based on network conditions
        
        Args:
            available_neighbors: Set of available neighbor addresses
            
        Returns:
            Set of selected neighbor addresses for gossiping
        """
        if not available_neighbors:
            return set()
        
        neighbors_list = list(available_neighbors)
        targets = set()
        
        # Determine effective fanout
        effective_fanout = self.fanout
        if self.adaptive_fanout:
            # Adapt fanout based on network size and epidemic progress
            network_size = len(available_neighbors)
            adaptation_factor = min(1.5, max(0.5, network_size / 10.0))
            effective_fanout = max(1, int(self.fanout * adaptation_factor))
        
        # Select targets with probability filtering
        attempts = 0
        max_attempts = min(len(neighbors_list), effective_fanout * 3)  # Avoid infinite loops
        
        while len(targets) < effective_fanout and attempts < max_attempts:
            # Random selection
            candidate = random.choice(neighbors_list)
            
            # Probability check
            if random.random() < self.gossip_probability:
                # Additional checks for epidemic learning
                if self._should_gossip_with(candidate):
                    targets.add(candidate)
            
            attempts += 1
        
        logging.info(f"[Epidemic Learning] Selected {len(targets)} gossip targets from {len(available_neighbors)} neighbors")
        return targets

    def _should_gossip_with(self, neighbor: str) -> bool:
        """
        Determine if we should gossip with a specific neighbor.
        
        Considers:
        - Recent communication history
        - Immunity/recovery status
        - Infection threshold
        
        Args:
            neighbor: Address of potential gossip target
            
        Returns:
            bool: Whether to proceed with gossip
        """
        # Check if neighbor is in recovery phase
        if neighbor in self.immunity_status:
            immunity_info = self.immunity_status[neighbor]
            if immunity_info.get('immune', False):
                if random.random() < self.recovery_probability:
                    # Overcome immunity with small probability
                    return True
                else:
                    return False
        
        # Check infection threshold
        if neighbor in self.infection_history:
            infection_count = self.infection_history[neighbor]
            if infection_count > 0:
                # Already infected - lower probability of re-infection
                re_infection_prob = max(0.1, 1.0 - (infection_count * self.infection_threshold))
                return random.random() < re_infection_prob
        
        return True  # Default: allow gossip

    def update_epidemic_state(self, successful_gossips: Dict[str, bool]):
        """
        Update epidemic state based on gossip results.
        
        Args:
            successful_gossips: Dict mapping neighbor_id -> success_status
        """
        for neighbor, success in successful_gossips.items():
            if success:
                # Update infection history
                if neighbor not in self.infection_history:
                    self.infection_history[neighbor] = 0
                self.infection_history[neighbor] += 1
                
                # Update statistics
                self.gossip_statistics['successful_infections'] += 1
            else:
                self.gossip_statistics['failed_contacts'] += 1
        
        self.gossip_statistics['total_gossips'] += len(successful_gossips)

    def get_epidemic_statistics(self) -> Dict:
        """Get current epidemic learning statistics."""
        stats = self.gossip_statistics.copy()
        stats['infection_coverage'] = len(self.infection_history)
        stats['immunity_nodes'] = len(self.immunity_status)
        stats['current_epidemic_round'] = self.current_epidemic_round
        
        if stats['total_gossips'] > 0:
            stats['success_rate'] = stats['successful_infections'] / stats['total_gossips']
        else:
            stats['success_rate'] = 0.0
        
        return stats

    def reset_epidemic_state(self):
        """Reset epidemic state for new training round."""
        self.current_epidemic_round = 0
        self.infection_history.clear()
        self.immunity_status.clear()
        
        # Keep cumulative statistics but reset round-specific ones
        logging.info("[Epidemic Learning] Reset epidemic state for new training round")

    def advance_epidemic_round(self):
        """Advance to next epidemic round."""
        self.current_epidemic_round += 1
        logging.info(f"[Epidemic Learning] Advanced to epidemic round {self.current_epidemic_round}")

    def is_epidemic_complete(self) -> bool:
        """Check if epidemic has completed all configured rounds."""
        return self.current_epidemic_round >= self.epidemic_rounds

    def get_configuration_summary(self) -> Dict:
        """Get summary of epidemic learning configuration."""
        return {
            'algorithm': 'EpidemicLearning',
            'gossip_probability': self.gossip_probability,
            'fanout': self.fanout,
            'epidemic_rounds': self.epidemic_rounds,
            'adaptive_fanout': self.adaptive_fanout,
            'weighted_averaging': self.weighted_averaging,
            'topology_awareness': self.topology_awareness,
            'sync_timeout': self.sync_timeout
        }
