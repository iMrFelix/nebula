# Epidemic Learning Implementation for Nebula

This document describes the implementation of the **Epidemic Learning** algorithm for gossip-based decentralized federated learning in the Nebula framework.

## Overview

Epidemic Learning is a gossip-based decentralized federated learning algorithm inspired by epidemic protocols. It uses probabilistic communication patterns where nodes randomly select neighbors to exchange models with, mimicking how diseases spread through populations.

### Key Features

- **Gossip-based Communication**: Nodes randomly select subsets of neighbors for model exchange
- **Probabilistic Exchange**: Communication happens with configurable probability
- **Multi-round Epidemic**: Multiple gossip rounds per training round ensure network-wide propagation
- **Adaptive Fanout**: Dynamic adjustment of communication partners based on network conditions
- **Fault Tolerance**: Natural robustness through redundant communication paths
- **Infection Tracking**: Maintains history of successful communications for intelligent routing

## Algorithm Details

### Epidemic Phases

1. **Local Training**: Each node performs SGD on local data
2. **Gossip Initiation**: Node selects random neighbors based on fanout and probability
3. **Epidemic Rounds**: Multiple sub-rounds of model exchange within each training round
4. **Model Aggregation**: Weighted averaging of received models with age/infection weighting  
5. **Synchronization**: Coordination across all nodes before next training round

### Gossip Protocol

```
For each training round:
  For epidemic_rounds iterations:
    1. Select fanout neighbors randomly
    2. With gossip_probability, send model to each selected neighbor
    3. Receive and store models from other nodes
    4. Track infection status and communication success
  End epidemic rounds
  5. Aggregate all received models using epidemic weights
  6. Synchronize with other nodes
End training round
```

## Implementation Components

### 1. EpidemicLearning Aggregator (`epidemiclearning.py`)

The main aggregation algorithm with epidemic-specific features:

- **Gossip Target Selection**: Random neighbor selection with fanout control
- **Epidemic Weighting**: Age-based and infection-strength-based model weighting
- **Infection Tracking**: Maintains history of successful communications
- **Adaptive Behavior**: Dynamic fanout adjustment based on network conditions

### 2. EpidemicLearningUpdateHandler (`epidemiclearninghandler.py`)

Handles the gossip protocol communication:

- **Multi-round Management**: Coordinates multiple epidemic rounds per training round
- **Probabilistic Communication**: Implements gossip probability filtering
- **Asynchronous Coordination**: Manages timing and synchronization of gossip rounds
- **Metadata Tracking**: Stores gossip path, infection strength, and timing information

## Configuration Parameters

### Core Epidemic Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `gossip_probability` | Probability of gossiping with selected neighbor | 0.8 | 0.0-1.0 |
| `fanout` | Number of neighbors to contact per epidemic round | 3 | 1-10 |
| `epidemic_rounds` | Number of gossip rounds per training round | 3 | 1-10 |
| `sync_timeout` | Maximum time to wait for epidemic completion (seconds) | 120 | 30-300 |

### Advanced Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `adaptive_fanout` | Enable dynamic fanout adjustment | true | boolean |
| `infection_threshold` | Threshold for re-infection probability | 0.1 | 0.0-1.0 |
| `recovery_probability` | Probability of overcoming immunity | 0.05 | 0.0-1.0 |
| `weighted_averaging` | Enable epidemic-based weighting | true | boolean |
| `age_decay_factor` | Decay factor for model age weighting | 0.95 | 0.5-1.0 |

### Configuration Structure

```json
{
  "federation": "EpidemicLearning",
  "agg_algorithm": "EpidemicLearning",
  "epidemic_config": {
    "gossip_probability": 0.8,
    "fanout": 3,
    "epidemic_rounds": 3,
    "sync_timeout": 120,
    "adaptive_fanout": true,
    "infection_threshold": 0.1,
    "recovery_probability": 0.05,
    "topology_awareness": true,
    "weighted_averaging": true,
    "age_decay_factor": 0.95
  }
}
```

## Usage Examples

### 1. Basic Epidemic Learning

Use the provided example configuration:

```bash
cp config_epidemic_example.json my_epidemic_experiment.json
# Edit parameters as needed
```

### 2. High Gossip Activity

For faster convergence with more communication:

```json
{
  "epidemic_config": {
    "gossip_probability": 0.9,
    "fanout": 4,
    "epidemic_rounds": 4,
    "adaptive_fanout": true
  }
}
```

### 3. Conservative Gossip

For bandwidth-limited scenarios:

```json
{
  "epidemic_config": {
    "gossip_probability": 0.5,
    "fanout": 2,
    "epidemic_rounds": 2,
    "adaptive_fanout": false
  }
}
```

### 4. Large Network

For networks with many nodes:

```json
{
  "epidemic_config": {
    "gossip_probability": 0.7,
    "fanout": 5,
    "epidemic_rounds": 4,
    "adaptive_fanout": true,
    "infection_threshold": 0.05
  }
}
```

## Network Topologies

Epidemic Learning works well with various network topologies:

### Random Topology (Recommended)
```
High connectivity, random connections
- Best convergence properties
- Natural fault tolerance
- Efficient information spreading
```

### Small World
```
Regular structure with random shortcuts
- Good balance of efficiency and robustness
- Realistic network modeling
```

### Scale-Free
```
Hub-based structure with power-law degree distribution
- Models real-world networks
- Robust to random failures
- Vulnerable to targeted attacks on hubs
```

## Performance Characteristics

### Convergence

Epidemic Learning convergence depends on:
- **Gossip Probability**: Higher probability → faster convergence, more communication
- **Fanout**: More neighbors → faster spreading, higher overhead
- **Epidemic Rounds**: More rounds → better mixing, longer latency
- **Network Topology**: Better connectivity → faster convergence

### Communication Complexity

- **Per Node**: O(fanout × epidemic_rounds) per training round
- **Network Total**: O(n × fanout × epidemic_rounds)
- **Comparison to Centralized**: Lower than centralized FL for large networks
- **Comparison to D-PSGD**: Higher due to multiple epidemic rounds

### Fault Tolerance

- **Node Failures**: Naturally robust due to redundant paths
- **Communication Failures**: Handled through probabilistic retry mechanisms
- **Network Partitions**: Partial consensus within connected components
- **Recovery**: Automatic healing through continued gossip

## Algorithm Variants

### 1. Push-Pull Epidemic
Current implementation uses "push" (nodes send to neighbors).
Can be extended to "pull" (nodes request from neighbors).

### 2. Adaptive Probability
Gossip probability can adapt based on:
- Network congestion
- Model convergence rate
- Communication success rate

### 3. Topology-Aware Gossip
Enhanced neighbor selection based on:
- Network centrality
- Communication history
- Geographical proximity

## Monitoring and Statistics

The implementation provides detailed statistics:

```python
# Get epidemic statistics
stats = aggregator.get_epidemic_statistics()
print(f"Infection coverage: {stats['infection_coverage']}")
print(f"Success rate: {stats['success_rate']:.2f}")
print(f"Epidemic rounds completed: {stats['rounds_completed']}")
```

### Key Metrics

- **Infection Coverage**: Number of nodes successfully contacted
- **Success Rate**: Ratio of successful to attempted gossips
- **Epidemic Rounds**: Total epidemic rounds completed
- **Average Gossips per Round**: Communication efficiency metric

## Troubleshooting

### Common Issues

1. **Slow Convergence**
   - Increase gossip_probability
   - Increase fanout or epidemic_rounds
   - Check network connectivity

2. **High Communication Overhead**
   - Decrease fanout
   - Reduce epidemic_rounds
   - Enable adaptive_fanout

3. **Synchronization Timeouts**
   - Increase sync_timeout
   - Reduce epidemic_rounds
   - Check network latency

4. **Uneven Model Propagation**
   - Increase gossip_probability
   - Enable topology_awareness
   - Check network topology balance

### Debug Configuration

```json
{
  "logginglevel": true,
  "report_status_data_queue": true,
  "epidemic_config": {
    "gossip_probability": 1.0,
    "fanout": 2,
    "epidemic_rounds": 2,
    "sync_timeout": 300
  }
}
```

## Comparison with Other Algorithms

| Algorithm | Communication | Fault Tolerance | Convergence | Scalability |
|-----------|--------------|-----------------|-------------|-------------|
| **EpidemicLearning** | Gossip-based | Excellent | Good | Excellent |
| **D-PSGD** | Direct neighbors | Good | Excellent | Good |
| **FedAvg** | Centralized | Poor | Excellent | Poor |
| **DFL** | All-to-all | Good | Good | Poor |

## Future Enhancements

1. **Compression Integration**: Model compression for bandwidth efficiency
2. **Security**: Byzantine fault tolerance for malicious nodes
3. **Privacy**: Differential privacy integration
4. **Dynamic Networks**: Support for node joins/leaves during training
5. **Hierarchical Gossip**: Multi-level epidemic spreading

## References

1. Kempe, D., Dobra, A., & Gehrke, J. (2003). Gossip-based computation of aggregate information. *FOCS 2003*.

2. Boyd, S., Ghosh, A., Prabhakar, B., & Shah, D. (2006). Randomized gossip algorithms. *IEEE Transactions on Information Theory*.

3. Nedić, A., & Ozdaglar, A. (2009). Distributed subgradient methods for multi-agent optimization. *IEEE Transactions on Automatic Control*.

4. Jin, R., He, L., Dai, T., & Bai, B. (2019). Towards communication efficient and privacy preserving federated learning. *ArXiv preprint*.

5. Koloskova, A., Loizou, N., Boreiri, S., Jaggi, M., & Stich, S. U. (2020). A unified theory of decentralized SGD with changing topology and local updates. *ICML 2020*.
