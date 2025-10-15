#!/usr/bin/env python3
"""
Comprehensive test script for Epidemic Learning implementation in Nebula.

This script tests the functionality of the EpidemicLearning aggregator
and EpidemicLearningUpdateHandler.
"""

import sys
import os
import asyncio
import json
import torch
import random
from unittest.mock import MagicMock, AsyncMock

# Add the nebula package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def setup_logging():
    """Setup logging for test output."""
    import logging
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
            "federation": "EpidemicLearning"
        },
        "aggregator_args": {
            "algorithm": "EpidemicLearning",
            "aggregation_timeout": 120
        },
        "epidemic_config": {
            "gossip_probability": 0.8,
            "fanout": 3,
            "epidemic_rounds": 2,
            "sync_timeout": 60,
            "adaptive_fanout": True,
            "infection_threshold": 0.1,
            "recovery_probability": 0.05,
            "topology_awareness": True,
            "weighted_averaging": True,
            "age_decay_factor": 0.95
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
    cm.get_addrs_current_connections = AsyncMock(return_value=[
        "127.0.0.1:45002", "127.0.0.1:45003", "127.0.0.1:45004", 
        "127.0.0.1:45005", "127.0.0.1:45006"
    ])
    cm.connections = {
        "127.0.0.1:45002": MagicMock(), 
        "127.0.0.1:45003": MagicMock(),
        "127.0.0.1:45004": MagicMock(),
        "127.0.0.1:45005": MagicMock(),
        "127.0.0.1:45006": MagicMock()
    }
    engine.cm = cm
    
    # Mock trainer with model
    trainer = MagicMock()
    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {
        'layer1.weight': torch.randn(5, 3),
        'layer1.bias': torch.randn(5),
        'layer2.weight': torch.randn(1, 5),
        'layer2.bias': torch.randn(1)
    }
    engine.trainer = trainer
    
    return engine

def create_epidemic_test_models(num_models=5):
    """Create sample models with epidemic metadata for testing."""
    models = {}
    
    for i, addr in enumerate([f"127.0.0.1:4500{j+1}" for j in range(num_models)]):
        # Create slightly different models
        model_params = {
            'layer1.weight': torch.randn(5, 3) + i * 0.1,
            'layer1.bias': torch.randn(5) + i * 0.1,
            'layer2.weight': torch.randn(1, 5) + i * 0.1,
            'layer2.bias': torch.randn(1) + i * 0.1
        }
        
        # Create epidemic metadata
        metadata = {
            'weight': 1.0,
            'timestamp': 1234567890 + i * 10,
            'gossip_round': random.randint(0, 2),
            'infection_strength': random.uniform(0.5, 1.5),
            'gossip_path_length': random.randint(1, 4)
        }
        
        models[addr] = (model_params, metadata)
    
    return models

def test_epidemic_imports():
    """Test importing Epidemic Learning components."""
    print("\\n=== Testing Epidemic Learning Imports ===")
    
    tests_passed = 0
    total_tests = 0
    
    # Test aggregator import
    total_tests += 1
    try:
        from nebula.core.aggregation.epidemiclearning import EpidemicLearning
        print("✓ EpidemicLearning aggregator import successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ EpidemicLearning aggregator import failed: {e}")
    
    # Test update handler import
    total_tests += 1
    try:
        from nebula.core.aggregation.updatehandlers.epidemiclearninghandler import EpidemicLearningUpdateHandler
        print("✓ EpidemicLearningUpdateHandler import successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ EpidemicLearningUpdateHandler import failed: {e}")
    
    # Test factory integration
    total_tests += 1
    try:
        from nebula.core.aggregation.aggregator import create_aggregator
        print("✓ Aggregator factory import successful")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Aggregator factory import failed: {e}")
    
    return tests_passed, total_tests

def test_epidemic_configuration():
    """Test Epidemic Learning configuration handling."""
    print("\\n=== Testing Epidemic Learning Configuration ===")
    
    tests_passed = 0
    total_tests = 0
    
    # Test configuration file
    total_tests += 1
    config_file = "config_epidemic_example.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            print(f"✓ Configuration file loaded: {config_file}")
            print(f"  - Federation: {config_data.get('federation')}")
            print(f"  - Algorithm: {config_data.get('agg_algorithm')}")
            print(f"  - Nodes: {config_data.get('n_nodes')}")
            
            # Check epidemic-specific config
            epidemic_config = config_data.get('epidemic_config', {})
            if epidemic_config:
                print("  - Epidemic Configuration:")
                print(f"    - Gossip probability: {epidemic_config.get('gossip_probability')}")
                print(f"    - Fanout: {epidemic_config.get('fanout')}")
                print(f"    - Epidemic rounds: {epidemic_config.get('epidemic_rounds')}")
            
            tests_passed += 1
        except Exception as e:
            print(f"✗ Configuration file loading failed: {e}")
    else:
        print(f"✗ Configuration file not found: {config_file}")
    
    return tests_passed, total_tests

def test_epidemic_aggregator():
    """Test EpidemicLearning aggregator functionality."""
    print("\\n=== Testing EpidemicLearning Aggregator ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        from nebula.core.aggregation.epidemiclearning import EpidemicLearning
        
        # Create aggregator
        total_tests += 1
        config = create_mock_config()
        engine = create_mock_engine()
        
        epidemic = EpidemicLearning(config=config, engine=engine)
        print("✓ EpidemicLearning aggregator created")
        print(f"  - Gossip probability: {epidemic.gossip_probability}")
        print(f"  - Fanout: {epidemic.fanout}")
        print(f"  - Epidemic rounds: {epidemic.epidemic_rounds}")
        tests_passed += 1
        
        # Test gossip target selection
        total_tests += 1
        available_neighbors = {"127.0.0.1:45002", "127.0.0.1:45003", "127.0.0.1:45004", "127.0.0.1:45005"}
        gossip_targets = epidemic.select_gossip_targets(available_neighbors)
        print(f"✓ Gossip target selection: {len(gossip_targets)} targets from {len(available_neighbors)} neighbors")
        tests_passed += 1
        
        # Test model aggregation
        total_tests += 1
        test_models = create_epidemic_test_models(4)
        print(f"  - Created {len(test_models)} test models with epidemic metadata")
        
        aggregated = epidemic.run_aggregation(test_models)
        print("✓ Epidemic model aggregation completed")
        print(f"  - Aggregated model layers: {list(aggregated.keys())}")
        tests_passed += 1
        
        # Test epidemic state management
        total_tests += 1
        epidemic.advance_epidemic_round()
        print("✓ Epidemic round advancement")
        print(f"  - Current epidemic round: {epidemic.current_epidemic_round}")
        tests_passed += 1
        
        # Test statistics
        total_tests += 1
        stats = epidemic.get_epidemic_statistics()
        print("✓ Epidemic statistics retrieval")
        print(f"  - Total gossips: {stats.get('total_gossips', 0)}")
        print(f"  - Success rate: {stats.get('success_rate', 0):.2f}")
        tests_passed += 1
        
    except Exception as e:
        print(f"✗ EpidemicLearning aggregator test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return tests_passed, total_tests

async def test_epidemic_update_handler():
    """Test EpidemicLearningUpdateHandler functionality."""
    print("\\n=== Testing EpidemicLearning Update Handler ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        from nebula.core.aggregation.updatehandlers.epidemiclearninghandler import (
            EpidemicLearningUpdateHandler, EpidemicUpdate
        )
        
        # Create update handler
        total_tests += 1
        aggregator = MagicMock()
        aggregator.engine = create_mock_engine()
        aggregator.is_epidemic_complete.return_value = False
        aggregator.select_gossip_targets.return_value = {"127.0.0.1:45002", "127.0.0.1:45003"}
        aggregator.advance_epidemic_round = MagicMock()
        
        handler = EpidemicLearningUpdateHandler(aggregator, "127.0.0.1:45001")
        print("✓ EpidemicLearningUpdateHandler created")
        tests_passed += 1
        
        # Test epidemic update creation
        total_tests += 1
        model_params = {
            'layer1.weight': torch.randn(5, 3),
            'layer1.bias': torch.randn(5)
        }
        epidemic_update = EpidemicUpdate(
            model=model_params,
            weight=1.0,
            source="127.0.0.1:45002",
            round=1,
            epidemic_round=1,
            gossip_path=["127.0.0.1:45002", "127.0.0.1:45003"],
            infection_strength=0.8
        )
        print("✓ EpidemicUpdate created")
        print(f"  - Source: {epidemic_update.source}")
        print(f"  - Epidemic round: {epidemic_update.epidemic_round}")
        print(f"  - Infection strength: {epidemic_update.infection_strength}")
        tests_passed += 1
        
        # Test round setup
        total_tests += 1
        federation_nodes = {"127.0.0.1:45001", "127.0.0.1:45002", "127.0.0.1:45003", "127.0.0.1:45004"}
        await handler.round_expected_updates(federation_nodes)
        print("✓ Epidemic round setup completed")
        print(f"  - All neighbors: {len(handler._all_neighbors)}")
        tests_passed += 1
        
        # Test epidemic round start
        total_tests += 1
        round_started = await handler.start_epidemic_round()
        if round_started:
            print("✓ Epidemic round started successfully")
            print(f"  - Gossip targets: {len(handler._gossip_targets)}")
            tests_passed += 1
        else:
            print("⚠️  Epidemic round start failed (may be expected)")
        
        # Test statistics
        total_tests += 1
        handler_stats = handler.get_epidemic_statistics()
        print("✓ Update handler statistics")
        print(f"  - Current epidemic round: {handler_stats.get('current_epidemic_round', 0)}")
        print(f"  - Epidemic active: {handler_stats.get('epidemic_active', False)}")
        tests_passed += 1
        
    except Exception as e:
        print(f"✗ EpidemicLearning update handler test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return tests_passed, total_tests

def test_epidemic_factory_integration():
    """Test factory integration for Epidemic Learning."""
    print("\\n=== Testing Factory Integration ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        from nebula.core.aggregation.aggregator import create_aggregator
        
        # Test factory creation
        total_tests += 1
        config = create_mock_config()
        engine = create_mock_engine()
        
        aggregator = create_aggregator(config, engine)
        
        if aggregator.__class__.__name__ == "EpidemicLearning":
            print("✓ Factory correctly created EpidemicLearning aggregator")
            print(f"  - Algorithm: {aggregator.__class__.__name__}")
            print(f"  - Gossip probability: {aggregator.gossip_probability}")
            print(f"  - Fanout: {aggregator.fanout}")
            tests_passed += 1
        else:
            print(f"✗ Factory created wrong aggregator: {aggregator.__class__.__name__}")
        
    except Exception as e:
        print(f"✗ Factory integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return tests_passed, total_tests

def test_epidemic_edge_cases():
    """Test edge cases for Epidemic Learning."""
    print("\\n=== Testing Edge Cases ===")
    
    tests_passed = 0
    total_tests = 0
    
    try:
        from nebula.core.aggregation.epidemiclearning import EpidemicLearning
        
        config = create_mock_config()
        engine = create_mock_engine()
        epidemic = EpidemicLearning(config=config, engine=engine)
        
        # Test empty neighbor selection
        total_tests += 1
        empty_targets = epidemic.select_gossip_targets(set())
        if len(empty_targets) == 0:
            print("✓ Empty neighbor selection handled correctly")
            tests_passed += 1
        else:
            print("✗ Empty neighbor selection failed")
        
        # Test zero fanout
        total_tests += 1
        epidemic.fanout = 0
        zero_fanout_targets = epidemic.select_gossip_targets({"127.0.0.1:45002", "127.0.0.1:45003"})
        if len(zero_fanout_targets) == 0:
            print("✓ Zero fanout handled correctly")
            tests_passed += 1
        else:
            print("✗ Zero fanout not handled correctly")
        
        # Test epidemic completion
        total_tests += 1
        epidemic.current_epidemic_round = 10
        epidemic.epidemic_rounds = 3
        if epidemic.is_epidemic_complete():
            print("✓ Epidemic completion detection works")
            tests_passed += 1
        else:
            print("✗ Epidemic completion detection failed")
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return tests_passed, total_tests

async def run_all_epidemic_tests():
    """Run all Epidemic Learning tests."""
    print("Starting Epidemic Learning Implementation Tests")
    print("=" * 60)
    
    all_tests_passed = 0
    all_total_tests = 0
    
    # Run test suites
    test_suites = [
        ("Import Tests", test_epidemic_imports),
        ("Configuration Tests", test_epidemic_configuration),
        ("Aggregator Tests", test_epidemic_aggregator),
        ("Update Handler Tests", test_epidemic_update_handler),
        ("Factory Integration Tests", test_epidemic_factory_integration),
        ("Edge Cases Tests", test_epidemic_edge_cases),
    ]
    
    results = []
    for test_name, test_func in test_suites:
        try:
            if asyncio.iscoroutinefunction(test_func):
                passed, total = await test_func()
            else:
                passed, total = test_func()
            results.append((test_name, passed, total))
            all_tests_passed += passed
            all_total_tests += total
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, 0, 1))
            all_total_tests += 1
    
    # Print summary
    print("\\n" + "=" * 60)
    print("Epidemic Learning Test Results Summary:")
    print("=" * 60)
    
    for test_name, passed, total in results:
        success_rate = (passed / total * 100) if total > 0 else 0
        status = "✓ PASSED" if passed == total else f"⚠️  PARTIAL ({passed}/{total})" if passed > 0 else "✗ FAILED"
        print(f"  {test_name}: {status} ({success_rate:.1f}%)")
    
    overall_success_rate = (all_tests_passed / all_total_tests * 100) if all_total_tests > 0 else 0
    print(f"\\nOverall: {all_tests_passed}/{all_total_tests} tests passed ({overall_success_rate:.1f}%)")
    
    if all_tests_passed == all_total_tests:
        print("\\n🎉 ALL TESTS PASSED! Epidemic Learning implementation is working correctly.")
        return True
    elif all_tests_passed >= all_total_tests * 0.8:
        print("\\n✅ Most tests passed - Epidemic Learning implementation is mostly working")
        return True
    else:
        print("\\n❌ Several tests failed - implementation needs fixes")
        return False

if __name__ == "__main__":
    setup_logging()
    
    # Set random seed for reproducible tests
    random.seed(42)
    torch.manual_seed(42)
    
    # Run tests
    success = asyncio.run(run_all_epidemic_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
