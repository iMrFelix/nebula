import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

from nebula.core.aggregation.updatehandlers.updatehandler import UpdateHandler
from nebula.core.eventmanager import EventManager
from nebula.core.nebulaevents import UpdateNeighborEvent, UpdateReceivedEvent
from nebula.core.utils.locker import Locker

if TYPE_CHECKING:
    from nebula.core.aggregation.aggregator import Aggregator


class Update:
    """
    Represents a model update received from a neighbor node in D-PSGD.
    
    Attributes:
        model (object): The model object or weights received.
        weight (float): The weight or importance of the update.
        source (str): Identifier of the neighbor node that sent the update.
        round (int): Training round this update belongs to.
        time_received (float): Timestamp when the update was received.
    """
    def __init__(self, model, weight, source, round, time_received):
        self.model = model
        self.weight = weight
        self.source = source
        self.round = round
        self.time_received = time_received

    def __eq__(self, other):
        """
        Checks if two updates belong to the same round.
        """
        return self.round == other.round


MAX_UPDATE_BUFFER_SIZE = 1  # Keep only the latest update per neighbor


class DPSGDUpdateHandler(UpdateHandler):
    """
    D-PSGD Update Handler for Synchronous Decentralized Federated Learning.

    This handler manages model updates exchange between direct neighbors in D-PSGD.
    Key differences from vanilla Nebula DFL strategy:
    - Only communicates with direct network neighbors (as opposed to all nodes)
    - Enforces synchronous rounds (all neighbors must participate)
    - Supports mixing matrix-based weighted averaging
    - Handles topology-aware neighbor selection
    """
    
    def __init__(self, aggregator, addr, buffersize=MAX_UPDATE_BUFFER_SIZE):
        """
        Initialize the D-PSGD update handler.

        Args:
            aggregator (Aggregator): DPSGD aggregator instance.
            addr (str): Address of the local node.
            buffersize (int): Maximum number of historical updates to keep per neighbor.
        """
        self._addr = addr
        self._aggregator: Aggregator = aggregator
        self._buffersize = buffersize
        self._updates_storage: dict[str, tuple[Update, deque[Update]]] = {}
        self._updates_storage_lock = Locker(name="dpsgd_updates_storage_lock", async_lock=True)
        
        # D-PSGD specific: track only direct neighbors
        self._direct_neighbors = set()
        self._neighbors_received = set()
        self._round_updates_lock = Locker(name="dpsgd_round_updates_lock", async_lock=True)
        self._update_neighbors_lock = Locker(name="dpsgd_update_neighbors_lock", async_lock=True)
        self._notification_sent_lock = Locker(name="dpsgd_notification_sent_lock", async_lock=True)
        self._notification = False
        self._missing_neighbors = set()
        
        # Synchronization for D-PSGD rounds
        self._current_round = 0
        self._round_sync_event = asyncio.Event()
        
        logging.info(f"[D-PSGD UpdateHandler] Initialized for node {addr}")

    @property
    def us(self):
        """Returns the internal updates storage dictionary."""
        return self._updates_storage

    @property
    def agg(self):
        """Returns the DPSGD aggregator instance."""
        return self._aggregator

    async def init(self, config=None):
        """
        Subscribe to update-related events from the event manager.
        """
        await EventManager.get_instance().subscribe_node_event(UpdateNeighborEvent, self.notify_neighbor_update)
        await EventManager.get_instance().subscribe_node_event(UpdateReceivedEvent, self.storage_update)
        logging.info("[D-PSGD UpdateHandler] Event subscriptions initialized")

    async def round_expected_updates(self, federation_nodes: set):
        """
        For D-PSGD, we only expect updates from direct neighbors, not all federation nodes.
        This method discovers and sets up communication with direct neighbors.

        Args:
            federation_nodes (set): Set of all federation nodes (ignored in D-PSGD).
        """
        await self._update_neighbors_lock.acquire_async()
        await self._updates_storage_lock.acquire_async()
        
        # Get direct neighbors from the communications manager
        direct_neighbors = await self._get_direct_neighbors()
        self._direct_neighbors = direct_neighbors
        self._neighbors_received.clear()
        
        logging.info(f"[D-PSGD UpdateHandler] Round setup - expecting updates from {len(direct_neighbors)} direct neighbors: {direct_neighbors}")

        # Initialize storage for direct neighbors only
        for neighbor in direct_neighbors:
            if neighbor not in self.us:
                self.us[neighbor] = (None, deque(maxlen=self._buffersize))

        # Remove nodes that are no longer neighbors
        removed_nodes = [node for node in self._updates_storage.keys() if node not in direct_neighbors]
        for rn in removed_nodes:
            del self._updates_storage[rn]
            logging.info(f"[D-PSGD UpdateHandler] Removed non-neighbor node {rn} from storage")

        # Check for already received updates from this round
        await self._check_updates_already_received()

        await self._updates_storage_lock.release_async()
        await self._update_neighbors_lock.release_async()

        # Prepare for new round synchronization
        if self._round_updates_lock.locked():
            await self._round_updates_lock.release_async()

        self._notification = False
        self._round_sync_event.clear()

    async def _get_direct_neighbors(self):
        """
        Get the list of direct neighbors from the communications manager.
        In D-PSGD, we only communicate with nodes we have direct connections to.
        """
        neighbors = set()
        if self.agg.engine and hasattr(self.agg.engine, 'cm'):
            # Get current direct connections
            try:
                current_connections = await self.agg.engine.cm.get_addrs_current_connections(only_direct=True)
                neighbors = set(current_connections)
                logging.info(f"[D-PSGD UpdateHandler] Found {len(neighbors)} direct neighbors")
            except Exception as e:
                logging.warning(f"[D-PSGD UpdateHandler] Could not get direct neighbors: {e}")
        
        return neighbors

    async def _check_updates_already_received(self):
        """
        Scan storage for updates already received in this round from direct neighbors.
        """
        for neighbor in self._direct_neighbors:
            if neighbor in self._updates_storage:
                (last_updt, node_storage) = self._updates_storage[neighbor]
                if len(node_storage) > 0:
                    try:
                        if (last_updt and node_storage[-1] and last_updt != node_storage[-1]) or (
                            node_storage[-1] and not last_updt
                        ):
                            self._neighbors_received.add(neighbor)
                            logging.info(
                                f"[D-PSGD UpdateHandler] Update already received from neighbor: {neighbor} | ({len(self._neighbors_received)}/{len(self._direct_neighbors)}) Updates received"
                            )
                    except Exception as e:
                        logging.warning(f"[D-PSGD UpdateHandler] Error checking existing update from {neighbor}: {e}")

    async def storage_update(self, updt_received_event: UpdateReceivedEvent):
        """
        Store an incoming update from a direct neighbor and trigger aggregation if all updates are received.

        Args:
            updt_received_event (UpdateReceivedEvent): Event with model update data.
        """
        time_received = time.time()
        (model, weight, source, round, _) = await updt_received_event.get_event_data()
        
        # D-PSGD only accepts updates from direct neighbors
        if source in self._direct_neighbors:
            updt = Update(model, weight, source, round, time_received)
            await self._updates_storage_lock.acquire_async()
            
            if updt in self.us[source][1]:
                logging.info(f"[D-PSGD UpdateHandler] Discarding duplicate update from neighbor: {source} for round: {round}")
            else:
                last_update_used = self.us[source][0]
                self.us[source][1].append(updt)
                self.us[source] = (last_update_used, self.us[source][1])
                
                logging.info(f"[D-PSGD UpdateHandler] Stored update | neighbor={source} | round={round} | weight={weight}")

                self._neighbors_received.add(source)
                updates_left = self._direct_neighbors.difference(self._neighbors_received)
                logging.info(f"[D-PSGD UpdateHandler] Updates received ({len(self._neighbors_received)}/{len(self._direct_neighbors)}) | Missing neighbors: {updates_left}")
                
                # Check if all neighbor updates received
                if self._round_updates_lock.locked() and not updates_left:
                    all_received = await self._all_updates_received()
                    if all_received:
                        await self._notify()
            
            await self._updates_storage_lock.release_async()
        else:
            logging.info(f"[D-PSGD UpdateHandler] Discarding update from non-neighbor: {source}")

    async def get_round_updates(self):
        """
        Retrieve the most recent valid updates from direct neighbors for D-PSGD aggregation.

        Returns:
            dict: A dictionary mapping neighbor_id to (model, weight) tuples.
        """
        await self._updates_storage_lock.acquire_async()
        
        updates_missing = self._direct_neighbors.difference(self._neighbors_received)
        if updates_missing:
            self._missing_neighbors = updates_missing
            logging.warning(f"[D-PSGD UpdateHandler] Missing updates from neighbors: {updates_missing}")
        else:
            self._missing_neighbors.clear()

        updates = {}
        
        # Collect updates from neighbors that sent them
        for neighbor in self._neighbors_received:
            if neighbor in self.us:
                source_historic = self.us[neighbor][1]
                if len(source_historic) > 0:
                    latest_update = source_historic[-1]
                    updates[neighbor] = (latest_update.model, latest_update.weight)
                    # Mark as used
                    self.us[neighbor] = (latest_update, source_historic)
                    logging.info(f"[D-PSGD UpdateHandler] Retrieved update from neighbor {neighbor}")

        # Add self-update (own model) - important for D-PSGD
        # The aggregator should have access to its own model through the engine
        if hasattr(self.agg, 'engine') and hasattr(self.agg.engine, 'trainer'):
            try:
                own_model = self.agg.engine.trainer.model.state_dict()
                updates[self._addr] = (own_model, 1.0)  # Own model with weight 1.0
                logging.info("[D-PSGD UpdateHandler] Added own model to updates")
            except Exception as e:
                logging.warning(f"[D-PSGD UpdateHandler] Could not add own model: {e}")

        await self._updates_storage_lock.release_async()
        
        logging.info(f"[D-PSGD UpdateHandler] Retrieved {len(updates)} updates for aggregation")
        return updates

    async def _all_updates_received(self):
        """
        Check if updates from all direct neighbors have been received.
        In synchronous D-PSGD, we need all neighbors to participate.
        """
        return len(self._neighbors_received) == len(self._direct_neighbors)

    async def _notify(self):
        """
        Notify that all neighbor updates have been received and aggregation can proceed.
        """
        await self._notification_sent_lock.acquire_async()
        if not self._notification:
            self._notification = True
            self._round_sync_event.set()
            logging.info("[D-PSGD UpdateHandler] All neighbor updates received - proceeding with aggregation")
        await self._notification_sent_lock.release_async()

    async def notify_neighbor_update(self, update_neighbor_event: UpdateNeighborEvent):
        """
        Handle neighbor update notifications for D-PSGD synchronization.
        """
        (round,) = await update_neighbor_event.get_event_data()
        logging.info(f"[D-PSGD UpdateHandler] Neighbor update notification for round {round}")
        
        # In D-PSGD, we could use this to trigger synchronization checks
        if round > self._current_round:
            self._current_round = round
            
    async def wait_for_sync(self, timeout=None):
        """
        Wait for synchronization in D-PSGD rounds.
        All neighbors must complete their updates before proceeding.
        """
        try:
            await asyncio.wait_for(self._round_sync_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logging.warning("[D-PSGD UpdateHandler] Synchronization timeout reached")
            return False

    async def get_round_missing_nodes(self) -> set[str]:
        """
        Identifies neighbors that have not yet provided updates in the current round.

        Returns:
            set: A set of neighbor identifiers that are expected to send updates but have not yet been received.
        """
        return self._direct_neighbors.difference(self._neighbors_received)

    async def notify_federation_update(self, updt_nei_event: UpdateNeighborEvent):
        """
        Notifies about federation updates. In D-PSGD, this is used for neighbor updates.
        """
        (round,) = await updt_nei_event.get_event_data()
        logging.info(f"[D-PSGD UpdateHandler] Federation update notification for round {round}")
        
        # In D-PSGD, we could use this to trigger synchronization checks
        if round > self._current_round:
            self._current_round = round

    async def notify_if_all_updates_received(self):
        """
        Notifies the system when all expected neighbor updates have been received.
        """
        if await self._all_updates_received():
            await self._notify()

    async def stop_notifying_updates(self):
        """
        Stops notifications related to update reception.
        Resets synchronization state for D-PSGD.
        """
        self._notification = False
        self._round_sync_event.clear()
        self._neighbors_received.clear()
        logging.info("[D-PSGD UpdateHandler] Stopped update notifications")
