# D-PSGD Implementation for Nebula

This document describes the implementation of the synchronous Decentralized Parallel Stochastic Gradient Descent (D-PSGD) algorithm in the Nebula federated learning framework.

## Overview

D-PSGD is a decentralized federated learning algorithm where nodes communicate only with their direct neighbors in a network topology. Unlike centralized approaches (like FedAvg), D-PSGD does not require a central aggregator and enables truly peer-to-peer federated learning.

### Key Features

- **Decentralized**: No central aggregator - all nodes are equal participants
- **Synchronous**: All nodes must complete local training before neighbor communication
- **Topology-aware**: Works with any connected graph topology (ring, grid, random, etc.)
- **Memory efficient**: Only stores updates from direct neighbors
- **Fault tolerant**: Can handle neighbor failures gracefully

## Algorithm Details

1. **Local Training**: Each node performs local SGD on its private dataset
2. **Model Exchange**: Nodes send their updated models to direct neighbors
3. **Consensus Averaging**: Each node averages received models (including its own)
4. **Synchronization**: Process repeats synchronously for all nodes

## Implementation Components

### 1. DPSGD Aggregator (`dpsgd.py`)

The main aggregation algorithm that performs model averaging:

```python
from nebula.core.aggregation.dpsgd import DPSGD

# Automatically used when agg_algorithm="DPSGD" in configuration
```

### 2. DPSGD Update Handler (`dpsgupdatehandler.py`)

Handles neighbor discovery and model exchange:

- Discovers direct neighbors from network topology
- Manages synchronous model exchange
- Enforces timeout policies for failed neighbors

### 3. Configuration

D-PSGD specific configuration parameters:

```json
{
  "federation": "DPSGD",
  "agg_algorithm": "DPSGD",
  "dpsgd_config": {
    "sync_timeout": 60,
    "mixing_matrix": false,
    "topology_type": "ring",
    "neighbor_selection": "direct",
    "convergence_threshold": 0.001
  }
}
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `sync_timeout` | Maximum time to wait for neighbor updates (seconds) | 60 |
| `mixing_matrix` | Enable sophisticated mixing weights (future feature) | false |
| `topology_type` | Network topology type | "ring" |
| `neighbor_selection` | How to select neighbors | "direct" |
| `convergence_threshold` | Convergence threshold for stopping | 0.001 |

## Usage

### 1. Basic Ring Topology

Use the provided example configuration:

```bash
# Use the example D-PSGD configuration
cp config_dpsgd_example.json my_dpsgd_experiment.json
```

### 2. Custom Topology

Modify the `nodes` and `matrix` sections in the configuration:

```json
{
  "topology": "Custom",
  "nodes": {
    "0": {"neighbors": ["1", "2"]},
    "1": {"neighbors": ["0", "3"]},
    "2": {"neighbors": ["0", "3"]},
    "3": {"neighbors": ["1", "2"]}
  },
  "matrix": [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
  ]
}
```

### 3. Node Roles

For D-PSGD, all nodes should be `trainer_aggregator` since each node both trains and aggregates:

```json
{
  "role": "trainer_aggregator"
}
```

## Example Topologies

### Ring Topology (4 nodes)
```
0 -- 1
|    |
3 -- 2
```

### Grid Topology (4 nodes)
```
0 -- 1
|    |
2 -- 3
```

### Star Topology (4 nodes)
```
  1
  |
2-0-3
```

## Implementation Notes

### Synchronization

D-PSGD requires synchronous operation:
- All nodes must complete local training before exchanging models
- Timeout mechanisms prevent indefinite waiting for failed nodes
- Missing neighbor updates are handled gracefully

### Memory Efficiency

The implementation is designed for efficiency:
- Only stores the latest update from each neighbor
- Automatic garbage collection of old models
- Configurable buffer sizes

### Fault Tolerance

The system handles various failure scenarios:
- Neighbor timeout: Continue with available neighbors
- Network partitions: Partial consensus within connected components
- Node failures: Dynamic neighbor discovery and adaptation

## Performance Considerations

### Convergence

D-PSGD convergence depends on:
- Network topology connectivity
- Data distribution across nodes (IID vs non-IID)
- Learning rate and local training epochs
- Mixing matrix design (if enabled)

### Communication

- Communication is peer-to-peer only
- No central bottleneck
- Communication overhead scales with node degree, not total nodes

### Scalability

- Naturally scales to large networks
- Communication complexity O(degree) per node
- No single point of failure

## Future Enhancements

1. **Adaptive Mixing Matrices**: Topology-specific weight computation
2. **Asynchronous Variant**: Support for asynchronous D-PSGD
3. **Dynamic Topologies**: Handle node joins/leaves during training
4. **Compression**: Model compression for bandwidth efficiency
5. **Privacy**: Integration with differential privacy mechanisms

## References

1. Lian, X., Zhang, C., Zhang, H., Hsieh, C. J., Zhang, W., & Liu, J. (2017). Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent. Advances in Neural Information Processing Systems, 30.

2. Koloskova, A., Loizou, N., Boreiri, S., Jaggi, M., & Stich, S. U. (2020). A unified theory of decentralized sgd with changing topology and local updates. International Conference on Machine Learning.

## Troubleshooting

### Common Issues

1. **Nodes not finding neighbors**: Check network connectivity and topology configuration
2. **Synchronization timeouts**: Increase `sync_timeout` or check node performance
3. **Convergence issues**: Verify topology connectivity and data distribution
4. **Memory errors**: Reduce buffer sizes or model complexity

### Debug Mode

Enable detailed logging:

```json
{
  "logginglevel": true,
  "report_status_data_queue": true
}
```

This will provide detailed information about:
- Neighbor discovery
- Model exchange timing
- Aggregation statistics
- Synchronization events
