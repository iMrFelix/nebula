import gc
import logging
import torch

from nebula.core.aggregation.aggregator import Aggregator


class DPSGD(Aggregator):
    """
    Aggregator: Decentralized Parallel Stochastic Gradient Descent (D-PSGD)
    
    This implements the synchronous D-PSGD algorithm for decentralized federated learning.
    In D-PSGD, each node communicates only with its direct neighbors in the network topology.
    After local training, nodes exchange their models with neighbors and perform averaging.
    The algorithm is synchronous: nodes perform local training, share updated weights with
    neighbors, and perform local arithmetic mean averaging.
        
    Note: https://arxiv.org/abs/1705.09056
    """

    def __init__(self, config=None, **kwargs):
        # D-PSGD specific configuration - check multiple possible locations
        dpsgd_config = config.participant.get("dpsgd_config", {})
        aggregator_args = config.participant.get("aggregator_args", {})
        
        self.mixing_matrix_enabled = dpsgd_config.get("mixing_matrix", aggregator_args.get("mixing_matrix", False))
        self.topology_type = dpsgd_config.get("topology_type", aggregator_args.get("topology", "ring"))
        self.sync_timeout = dpsgd_config.get("sync_timeout", 60)
        self.neighbor_selection = dpsgd_config.get("neighbor_selection", "direct")
        self.convergence_threshold = dpsgd_config.get("convergence_threshold", 0.001)
        
        # Initialize parent - but first temporarily change the federation type to DPSGD
        # so it uses the correct update handler
        original_federation = config.participant["scenario_args"]["federation"]
        config.participant["scenario_args"]["federation"] = "DPSGD"
        
        super().__init__(config, **kwargs)
        
        # Restore original federation type
        config.participant["scenario_args"]["federation"] = original_federation
        
        logging.info(f"[D-PSGD] Initialized with configuration:")
        logging.info(f"  - Topology: {self.topology_type}")
        logging.info(f"  - Mixing matrix: {self.mixing_matrix_enabled}")
        logging.info(f"  - Sync timeout: {self.sync_timeout}s")
        logging.info(f"  - Neighbor selection: {self.neighbor_selection}")
        logging.info("[D-PSGD] Using custom DPSGD update handler for neighbor communication")

    def run_aggregation(self, models):
        """
        Perform D-PSGD aggregation by averaging models from neighbors.
        
        In D-PSGD, we perform a simple average of all received models (including our own).
        This simulates the consensus averaging step in decentralized optimization.
        
        Args:
            models (dict): Dictionary mapping node_id -> (model_parameters, weight)
        
        Returns:
            dict: Averaged model parameters
        """

        # Check whether models is not empty (has length 0)
        super().run_aggregation(models)
        if not models:
            raise ValueError("No models received for D-PSGD aggregation")
        
        models_list = list(models.values())
        logging.info(f"[D-PSGD] Aggregating {len(models_list)} models from neighbors")
        
        # In standard D-PSGD, we use equal weights (simple averaging)
        # Each node (including self) contributes equally to the new (merged) model
        num_models = len(models_list)
        
        # Initialize accumulator with the structure of the first model
        first_model_params = models_list[0][0]
        accum = {layer: torch.zeros_like(param, dtype=torch.float32)
                for layer, param in first_model_params.items()}
        
        # Perform averaging (consensus step in D-PSGD)
        with torch.no_grad():
            for model_parameters, weight in models_list:
                # In D-PSGD, we use equal mixing weights (1/num_neighbors, assuming every neighbor csent a model)
                mixing_weight = 1.0 / num_models
                
                # Merge layer by layer into accum, weighing each layer with mixing_weight
                for layer in accum:
                    accum[layer].add_(
                        model_parameters[layer].to(accum[layer].dtype),
                        alpha=mixing_weight
                    )
        
        # Clean up memory (borriwed from preexisting fedavg.py code)
        del models_list
        gc.collect()
        
        logging.info(f"[D-PSGD] Completed model averaging with {num_models} neighbors")
        return accum

    # TODO: Complete function to perform weighted averaging, e.g. based on layer importance (?)
    #       Not really relevent for the project though, so lower priority!
    # def _compute_mixing_weights(self, models):
    #     """
    #     Compute mixing weights for more sophisticated D-PSGD variants.
        
    #     This could implement:
    #     - Metropolis-Hastings weights
    #     - Degree-based weights
    #     - Distance-based weights
        
    #     For now, implements equal weighting (standard D-PSGD).
    #     """
    #     num_models = len(models)
    #     return {node_id: 1.0 / num_models for node_id in models.keys()}

    # def get_topology_info(self):
    #     """
    #     Get information about the current network topology for D-PSGD.
    #     This can be used for advanced mixing matrix computation.
    #     """
    #     # Get current connections from the communications manager
    #     if self.engine and hasattr(self.engine, 'cm'):
    #         return {
    #             'neighbors': len(self.engine.cm.connections),
    #             'topology': self.topology_type
    #         }
    #     return {'neighbors': 0, 'topology': self.topology_type}
