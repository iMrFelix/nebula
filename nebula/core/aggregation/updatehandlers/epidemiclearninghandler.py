import asyncio
import logging
import random
import time
from collections import deque, defaultdict
from typing import TYPE_CHECKING, Dict, Set, List, Tuple

from nebula.core.aggregation.updatehandlers.updatehandler import UpdateHandler
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateNeighborEvent, UpdateReceivedEvent
from nebula.core.utils.locker import Locker

if TYPE_CHECKING:
    from nebula.core.aggregation.aggregator import Aggregator


class EpidemicUpdate:
    """
    Represents a model update in epidemic learning with gossip metadata.
    
    Attributes:
        model (object): The model object or weights received.
        weight (float): The weight or importance of the update.
        source (str): Identifier of the source node.
        round (int): Training round this update belongs to.
        epidemic_round (int): Which epidemic sub-round this update is from.
        gossip_path (list): Path of gossip transmission.
        timestamp (float): When the update was received.
        infection_strength (float): Strength of the epidemic infection.
    """
    def __init__(self, model, weight, source, round, epidemic_round=0, 
                 gossip_path=None, timestamp=None, infection_strength=1.0):
        self.model = model
        self.weight = weight
        self.source = source
        self.round = round
        self.epidemic_round = epidemic_round
        self.gossip_path = gossip_path or [source]
        self.timestamp = timestamp or time.time()
        self.infection_strength = infection_strength

    def __eq__(self, other):
        """Check if two updates are from the same source and epidemic round."""
        return (self.source == other.source and 
                self.round == other.round and 
                self.epidemic_round == other.epidemic_round)

    def get_metadata(self) -> Dict:
        """Get update metadata for aggregation."""
        return {
            'weight': self.weight,
            'timestamp': self.timestamp,
            'gossip_round': self.epidemic_round,
            'infection_strength': self.infection_strength,
            'gossip_path_length': len(self.gossip_path)
        }


MAX_EPIDEMIC_BUFFER_SIZE = 5  # Keep multiple epidemic rounds per node


class EpidemicLearningUpdateHandler(UpdateHandler):
    """
    Epidemic Learning Update Handler for Gossip-based Decentralized Federated Learning.

    This handler manages the epidemic-style gossip communication protocol:
    - Random neighbor selection for gossip
    - Probabilistic model exchange
    - Multiple epidemic rounds per training round
    - Infection tracking and immunity management
    - Adaptive fanout based on network conditions
    
    Key differences from other update handlers:
    - Supports multiple sub-rounds (epidemic rounds) per training round
    - Implements gossip probability and fanout limiting
    - Tracks infection/immunity status of nodes
    - Handles asynchronous gossip communication patterns
    """
    
    def __init__(self, aggregator, addr, buffersize=MAX_EPIDEMIC_BUFFER_SIZE):
        """
        Initialize the Epidemic Learning update handler.

        Args:
            aggregator (Aggregator): EpidemicLearning aggregator instance.
            addr (str): Address of the local node.
            buffersize (int): Maximum number of updates to keep per node.
        """
        self._addr = addr
        self._aggregator: Aggregator = aggregator
        self._buffersize = buffersize
        
        # Storage for epidemic updates
        self._updates_storage: Dict[str, Tuple[EpidemicUpdate, deque[EpidemicUpdate]]] = {}
        self._updates_storage_lock = Locker(name="epidemic_updates_storage_lock", async_lock=True)
        
        # Epidemic-specific tracking
        self._all_neighbors = set()  # All potential neighbors
        self._gossip_targets = set()  # Current gossip targets
        self._active_gossips = set()  # Currently active gossip communications
        self._completed_gossips = defaultdict(list)  # Completed gossips per epidemic round
        
        # Synchronization and state management
        self._current_training_round = 0
        self._current_epidemic_round = 0
        self._epidemic_complete_event = asyncio.Event()
        self._gossip_round_lock = Locker(name="epidemic_gossip_round_lock", async_lock=True)
        self._neighbor_management_lock = Locker(name="epidemic_neighbor_lock", async_lock=True)
        self._notification_lock = Locker(name="epidemic_notification_lock", async_lock=True)
        
        # Epidemic state
        self._epidemic_active = False
        self._gossip_statistics = {
            'rounds_initiated': 0,
            'successful_gossips': 0,
            'failed_gossips': 0,
            'total_infections': 0
        }
        
        # Timing and synchronization
        self._round_start_time = None
        self._gossip_timeouts = {}  # Track gossip timeouts
        
        logging.info(f"[Epidemic UpdateHandler] Initialized for node {addr}")

    @property
    def us(self):
        """Returns the internal updates storage dictionary."""
        return self._updates_storage

    @property
    def agg(self):
        """Returns the EpidemicLearning aggregator instance."""
        return self._aggregator

    async def init(self, config=None):
        """
        Subscribe to update-related events from the event manager.
        """
        await EventManager.get_instance().subscribe_node_event(UpdateNeighborEvent, self.notify_epidemic_update)
        await EventManager.get_instance().subscribe_node_event(UpdateReceivedEvent, self.storage_update)
        logging.info("[Epidemic UpdateHandler] Event subscriptions initialized")

    async def round_expected_updates(self, federation_nodes: set):
        """
        Initialize epidemic learning round with all potential neighbors.
        
        In epidemic learning, we start with all federation nodes as potential
        gossip targets, then select subsets randomly for each epidemic round.

        Args:
            federation_nodes (set): Set of all federation nodes for potential gossip.
        """
        await self._neighbor_management_lock.acquire_async()
        await self._updates_storage_lock.acquire_async()
        
        self._all_neighbors = federation_nodes.copy()
        self._gossip_targets.clear()
        self._active_gossips.clear()
        self._completed_gossips.clear()
        
        logging.info(f"[Epidemic UpdateHandler] Round setup - {len(federation_nodes)} potential gossip neighbors")

        # Initialize storage for all potential neighbors
        for neighbor in federation_nodes:
            if neighbor not in self.us:
                self.us[neighbor] = (None, deque(maxlen=self._buffersize))

        # Remove nodes no longer in federation
        removed_nodes = [node for node in self._updates_storage.keys() if node not in federation_nodes]
        for node in removed_nodes:
            del self._updates_storage[node]
            logging.info(f"[Epidemic UpdateHandler] Removed node {node} from storage")

        # Initialize epidemic state
        self._current_epidemic_round = 0
        self._epidemic_active = True
        self._epidemic_complete_event.clear()
        self._round_start_time = time.time()
        
        await self._updates_storage_lock.release_async()
        await self._neighbor_management_lock.release_async()

        logging.info("[Epidemic UpdateHandler] Epidemic learning round initialized")

    async def start_epidemic_round(self) -> bool:
        """
        Start a new epidemic gossip round.
        
        Returns:
            bool: True if epidemic round started, False if epidemic is complete
        """
        await self._gossip_round_lock.acquire_async()
        
        try:
            if not self._epidemic_active or self.agg.is_epidemic_complete():
                logging.info("[Epidemic UpdateHandler] Epidemic learning complete")
                return False
            
            # Select gossip targets for this epidemic round
            available_neighbors = self._all_neighbors - {self._addr}  # Exclude self
            self._gossip_targets = self.agg.select_gossip_targets(available_neighbors)
            
            if not self._gossip_targets:
                logging.warning("[Epidemic UpdateHandler] No gossip targets selected")
                return False
            
            self._active_gossips = self._gossip_targets.copy()
            self._current_epidemic_round += 1
            self.agg.advance_epidemic_round()
            
            logging.info(f"[Epidemic UpdateHandler] Started epidemic round {self._current_epidemic_round}")
            logging.info(f"[Epidemic UpdateHandler] Gossip targets: {self._gossip_targets}")
            
            # Update statistics
            self._gossip_statistics['rounds_initiated'] += 1
            
            return True
            
        finally:
            await self._gossip_round_lock.release_async()

    async def storage_update(self, updt_received_event: UpdateReceivedEvent):
        """
        Store an incoming epidemic update from gossip communication.

        Args:
            updt_received_event (UpdateReceivedEvent): Event with model update data.
        """
        time_received = time.time()
        (model, weight, source, round, extra_data) = await updt_received_event.get_event_data()
        
        # Extract epidemic-specific metadata
        epidemic_round = extra_data.get('epidemic_round', 0) if extra_data else 0
        gossip_path = extra_data.get('gossip_path', [source]) if extra_data else [source]
        infection_strength = extra_data.get('infection_strength', 1.0) if extra_data else 1.0
        
        # Only accept updates from known neighbors during active epidemic
        if source in self._all_neighbors and self._epidemic_active:
            epidemic_update = EpidemicUpdate(
                model=model,
                weight=weight,
                source=source,
                round=round,
                epidemic_round=epidemic_round,
                gossip_path=gossip_path,
                timestamp=time_received,
                infection_strength=infection_strength
            )
            
            await self._updates_storage_lock.acquire_async()
            
            try:
                # Check for duplicate updates
                if epidemic_update in self.us[source][1]:
                    logging.info(f"[Epidemic UpdateHandler] Discarding duplicate epidemic update from {source}")
                else:
                    # Store the epidemic update
                    last_update_used = self.us[source][0]
                    self.us[source][1].append(epidemic_update)
                    self.us[source] = (last_update_used, self.us[source][1])
                    
                    logging.info(f"[Epidemic UpdateHandler] Stored epidemic update | source={source} | "
                               f"round={round} | epidemic_round={epidemic_round} | "
                               f"path_length={len(gossip_path)} | strength={infection_strength:.2f}")

                    # Track completed gossip
                    self._completed_gossips[epidemic_round].append(source)
                    
                    # Update gossip statistics
                    self._gossip_statistics['successful_gossips'] += 1
                    self._gossip_statistics['total_infections'] += 1
                    
                    # Remove from active gossips if this was a target
                    if source in self._active_gossips:
                        self._active_gossips.remove(source)
                    
                    # Check if current epidemic round is complete
                    await self._check_epidemic_round_complete()
                    
            finally:
                await self._updates_storage_lock.release_async()
        else:
            if not self._epidemic_active:
                logging.debug(f"[Epidemic UpdateHandler] Discarding update - epidemic not active")
            else:
                logging.info(f"[Epidemic UpdateHandler] Discarding update from unknown neighbor: {source}")

    async def _check_epidemic_round_complete(self):
        """Check if current epidemic round is complete and trigger next round or completion."""
        if not self._active_gossips:  # All gossip targets have responded
            logging.info(f"[Epidemic UpdateHandler] Epidemic round {self._current_epidemic_round} complete")
            
            # Start next epidemic round or complete epidemic
            if not self.agg.is_epidemic_complete():
                # Schedule next epidemic round
                asyncio.create_task(self._schedule_next_epidemic_round())
            else:
                # Epidemic is complete
                await self._complete_epidemic()

    async def _schedule_next_epidemic_round(self):
        """Schedule the next epidemic round with a small delay."""
        await asyncio.sleep(0.1)  # Small delay between epidemic rounds
        await self.start_epidemic_round()

    async def _complete_epidemic(self):
        """Complete the epidemic learning process."""
        self._epidemic_active = False
        self._epidemic_complete_event.set()
        
        # Update aggregator epidemic state
        successful_gossips = {}
        for round_gossips in self._completed_gossips.values():
            for neighbor in round_gossips:
                successful_gossips[neighbor] = True
        
        self.agg.update_epidemic_state(successful_gossips)
        
        logging.info("[Epidemic UpdateHandler] Epidemic learning process completed")
        logging.info(f"[Epidemic UpdateHandler] Statistics: {self._gossip_statistics}")

    async def get_round_updates(self) -> Dict[str, Tuple]:
        """
        Retrieve epidemic updates for aggregation.
        
        Returns the most recent and relevant updates from the epidemic process,
        considering infection strength, gossip path, and recency.

        Returns:
            dict: A dictionary mapping node_id to (model, metadata) tuples.
        """
        await self._updates_storage_lock.acquire_async()
        
        try:
            updates = {}
            
            # Collect best update from each source
            for source, (last_used, update_history) in self._updates_storage.items():
                if len(update_history) > 0:
                    # Select best update (most recent, highest infection strength)
                    best_update = max(update_history, 
                                    key=lambda u: (u.timestamp, u.infection_strength, u.epidemic_round))
                    
                    updates[source] = (best_update.model, best_update.get_metadata())
                    
                    # Mark as used
                    self.us[source] = (best_update, update_history)
                    
                    logging.info(f"[Epidemic UpdateHandler] Retrieved update from {source} "
                               f"(epidemic_round={best_update.epidemic_round}, "
                               f"strength={best_update.infection_strength:.2f})")

            # Add self-update (own model)
            if hasattr(self.agg, 'engine') and hasattr(self.agg.engine, 'trainer'):
                try:
                    own_model = self.agg.engine.trainer.model.state_dict()
                    own_metadata = {
                        'weight': 1.0,
                        'timestamp': time.time(),
                        'gossip_round': 0,
                        'infection_strength': 1.0,
                        'gossip_path_length': 0
                    }
                    updates[self._addr] = (own_model, own_metadata)
                    logging.info("[Epidemic UpdateHandler] Added own model to updates")
                except Exception as e:
                    logging.warning(f"[Epidemic UpdateHandler] Could not add own model: {e}")

            logging.info(f"[Epidemic UpdateHandler] Retrieved {len(updates)} updates for aggregation")
            return updates
            
        finally:
            await self._updates_storage_lock.release_async()

    async def wait_for_epidemic_completion(self, timeout=None):
        """
        Wait for the epidemic learning process to complete.
        
        Args:
            timeout: Maximum time to wait for completion
            
        Returns:
            bool: True if completed successfully, False if timeout
        """
        try:
            await asyncio.wait_for(self._epidemic_complete_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logging.warning("[Epidemic UpdateHandler] Epidemic completion timeout reached")
            return False

    # Abstract method implementations

    async def get_round_missing_nodes(self) -> Set[str]:
        """
        Get nodes that haven't participated in the epidemic yet.
        
        Returns:
            set: Set of neighbor identifiers that haven't sent any epidemic updates.
        """
        participating_nodes = set()
        for round_gossips in self._completed_gossips.values():
            participating_nodes.update(round_gossips)
        
        return self._all_neighbors - participating_nodes - {self._addr}

    async def notify_epidemic_update(self, updt_nei_event: UpdateNeighborEvent):
        """Handle epidemic update notifications."""
        (round,) = await updt_nei_event.get_event_data()
        logging.info(f"[Epidemic UpdateHandler] Epidemic update notification for round {round}")
        
        if round > self._current_training_round:
            self._current_training_round = round

    async def notify_federation_update(self, updt_nei_event: UpdateNeighborEvent):
        """Handle federation update notifications (same as epidemic update)."""
        await self.notify_epidemic_update(updt_nei_event)

    async def notify_if_all_updates_received(self):
        """Notify if all expected epidemic updates have been received."""
        if not self._epidemic_active:
            await self._complete_epidemic()

    async def stop_notifying_updates(self):
        """Stop epidemic learning and reset state."""
        self._epidemic_active = False
        self._epidemic_complete_event.set()
        self._gossip_targets.clear()
        self._active_gossips.clear()
        self._completed_gossips.clear()
        
        logging.info("[Epidemic UpdateHandler] Stopped epidemic learning process")

    def get_epidemic_statistics(self) -> Dict:
        """Get current epidemic learning statistics."""
        stats = self._gossip_statistics.copy()
        stats['current_epidemic_round'] = self._current_epidemic_round
        stats['active_gossips'] = len(self._active_gossips)
        stats['completed_gossips_total'] = sum(len(gossips) for gossips in self._completed_gossips.values())
        stats['epidemic_active'] = self._epidemic_active
        
        if stats['rounds_initiated'] > 0:
            stats['avg_gossips_per_round'] = stats['successful_gossips'] / stats['rounds_initiated']
        else:
            stats['avg_gossips_per_round'] = 0.0
        
        return stats
