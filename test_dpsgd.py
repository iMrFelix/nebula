#!/usr/bin/env python3
"""
Simple test script for D-PSGD implementation in Nebula.

This script tests the basic functionality of the D-PSGD aggregator
and update handler without requiring a full Nebula deployment.
"""

import sys
import os
import logging
import torch
import asyncio
from unittest.mock import MagicMock, AsyncMock

# Add the nebula package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def setup_logging():
    """Setup logging for test output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_mock_config():
    """Create a mock configuration for testing."""
    config = MagicMock()
    config.participant = {
        "network_args": {
            "addr": "127.0.0.1:45001"
        },
        "scenario_args": {
            "federation": "DPSGD"
        },
        "aggregator_args": {
            "algorithm": "DPSGD",
            "aggregation_timeout": 60
        },
        "dpsgd_config": {
            "sync_timeout": 30,
            "mixing_matrix": False,
            "topology_type": "ring",
            "neighbor_selection": "direct",
            "convergence_threshold": 0.001
        }
    }
    return config

def create_mock_engine():
    """Create a mock engine for testing."""
    engine = MagicMock()
    engine.get_addr.return_value = "127.0.0.1:45001"
    engine.get_config.return_value = create_mock_config()
    
    # Mock communications manager
    cm = MagicMock()
    cm.get_addrs_current_connections = AsyncMock(return_value=["127.0.0.1:45002", "127.0.0.1:45003"])
    cm.connections = {"127.0.0.1:45002": MagicMock(), "127.0.0.1:45003": MagicMock()}
    engine.cm = cm
    
    # Mock trainer with model
    trainer = MagicMock()
    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {
        'layer1.weight': torch.randn(10, 5),
        'layer1.bias': torch.randn(10),
        'layer2.weight': torch.randn(1, 10),
        'layer2.bias': torch.randn(1)
    }
    engine.trainer = trainer
    
    return engine

def create_test_models():
    """Create sample models for testing aggregation."""
    models = {}
    
    # Create 3 different models (simulating updates from 2 neighbors + self)
    for i, addr in enumerate(["127.0.0.1:45001", "127.0.0.1:45002", "127.0.0.1:45003"]):
        model_params = {
            'layer1.weight': torch.randn(10, 5) + i * 0.1,  # Slightly different models
            'layer1.bias': torch.randn(10) + i * 0.1,
            'layer2.weight': torch.randn(1, 10) + i * 0.1,
            'layer2.bias': torch.randn(1) + i * 0.1
        }
        models[addr] = (model_params, 1.0)  # (model, weight)
    
    return models

def test_dpsgd_aggregator():
    """Test the D-PSGD aggregator functionality."""
    print("\n=== Testing D-PSGD Aggregator ===")
    
    try:
        from nebula.core.aggregation.dpsgd import DPSGD
        
        # Create mock objects
        config = create_mock_config()
        engine = create_mock_engine()
        
        # Initialize D-PSGD aggregator
        print("Creating D-PSGD aggregator...")
        dpsgd = DPSGD(config=config, engine=engine)
        
        print(f"✓ D-PSGD aggregator created successfully")
        print(f"  - Topology: {dpsgd.topology_type}")
        print(f"  - Mixing matrix: {dpsgd.mixing_matrix_enabled}")
        print(f"  - Sync timeout: {dpsgd.sync_timeout}")
        
        # Test aggregation
        print("\nTesting model aggregation...")
        test_models = create_test_models()
        print(f"  - Number of models to aggregate: {len(test_models)}")
        
        # Perform aggregation
        aggregated_model = dpsgd.run_aggregation(test_models)
        
        print("✓ Model aggregation completed successfully")
        print(f"  - Aggregated model layers: {list(aggregated_model.keys())}")
        
        # Verify aggregation results
        for layer_name, aggregated_params in aggregated_model.items():
            print(f"  - {layer_name}: shape {aggregated_params.shape}")
            
            # Verify that aggregation produces reasonable results
            # (should be average of input models)
            input_tensors = [model[0][layer_name] for model in test_models.values()]
            expected_avg = torch.stack(input_tensors).mean(dim=0)
            
            if torch.allclose(aggregated_params, expected_avg, atol=1e-6):
                print(f"    ✓ Correct averaging for {layer_name}")
            else:
                print(f"    ✗ Incorrect averaging for {layer_name}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_dpsgd_update_handler():
    """Test the D-PSGD update handler functionality."""
    print("\n=== Testing D-PSGD Update Handler ===")
    
    try:
        from nebula.core.aggregation.updatehandlers.dpsgupdatehandler import DPSGDUpdateHandler, Update
        
        # Create mock aggregator
        aggregator = MagicMock()
        aggregator.engine = create_mock_engine()
        
        # Initialize update handler
        print("Creating D-PSGD update handler...")
        handler = DPSGDUpdateHandler(aggregator, "127.0.0.1:45001")
        
        print("✓ D-PSGD update handler created successfully")
        
        # Test neighbor discovery
        print("\nTesting neighbor discovery...")
        neighbors = await handler._get_direct_neighbors()
        print(f"✓ Found neighbors: {neighbors}")
        
        # Test update storage setup
        print("\nTesting update storage setup...")
        federation_nodes = {"127.0.0.1:45001", "127.0.0.1:45002", "127.0.0.1:45003"}
        await handler.round_expected_updates(federation_nodes)
        print(f"✓ Update storage setup completed")
        print(f"  - Direct neighbors: {handler._direct_neighbors}")
        
        # Test update creation
        print("\nTesting update creation...")
        model_params = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10)
        }
        update = Update(model_params, 1.0, "127.0.0.1:45002", 1, 1234567890)
        print(f"✓ Update created: source={update.source}, round={update.round}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test D-PSGD configuration handling."""
    print("\n=== Testing D-PSGD Configuration ===")
    
    try:
        # Test configuration loading
        config_file = "config_dpsgd_example.json"
        if os.path.exists(config_file):
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            print(f"✓ Configuration file loaded: {config_file}")
            print(f"  - Federation: {config_data.get('federation')}")
            print(f"  - Algorithm: {config_data.get('agg_algorithm')}")
            print(f"  - Topology: {config_data.get('topology')}")
            print(f"  - Nodes: {config_data.get('n_nodes')}")
            
            # Verify D-PSGD specific configuration
            dpsgd_config = config_data.get('dpsgd_config', {})
            print(f"  - D-PSGD config: {dpsgd_config}")
            
            return True
        else:
            print(f"✗ Configuration file not found: {config_file}")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_factory_integration():
    """Test integration with Nebula's factory system."""
    print("\n=== Testing Factory Integration ===")
    
    try:
        from nebula.core.aggregation.aggregator import create_aggregator
        
        # Create config for D-PSGD
        config = create_mock_config()
        engine = create_mock_engine()
        
        print("Testing aggregator factory...")
        aggregator = create_aggregator(config, engine)
        
        if aggregator.__class__.__name__ == "DPSGD":
            print("✓ Factory correctly created D-PSGD aggregator")
            return True
        else:
            print(f"✗ Factory created wrong aggregator: {aggregator.__class__.__name__}")
            return False
            
    except Exception as e:
        print(f"✗ Factory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all D-PSGD tests."""
    print("Starting D-PSGD Implementation Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("D-PSGD Aggregator", test_dpsgd_aggregator),
        ("D-PSGD Update Handler", test_dpsgd_update_handler),
        ("Factory Integration", test_factory_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! D-PSGD implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    setup_logging()
    
    # Suppress some noisy logs during testing
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
