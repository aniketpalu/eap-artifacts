"""
RBAC Test Script for Call Center Feature Views
Tests feature retrieval operations with RBAC permissions
Designed to run in Jupyter notebook cells with pre-initialized FeatureStore

Usage in Jupyter:
    # In a notebook cell, after initializing fs for a specific user:
    from test_rbac_call_center import run_rbac_tests
    run_rbac_tests(fs, username="user1")
    
    # Or if fs is already in scope:
    run_rbac_tests(fs, username="user5")
"""

import pandas as pd
from datetime import datetime, timedelta
from feast.errors import FeastPermissionError

from feast import FeatureStore

# Import Feast objects for write operations
from feast import FeatureView, Field, Entity
from feast.types import Int64
from feast.value_type import ValueType

fs = FeatureStore(repo_path='../client_feature_repo')

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"🔐 {title}")
    print("=" * 80)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n📋 {title}")
    print("-" * 80)

def print_result(operation: str, success: bool, message: str = ""):
    """Print operation result"""
    status = "✅ SUCCESS" if success else "❌ FAILED"
    print(f"  {status}: {operation}")
    if message:
        print(f"    {message}")

# ============================================================================
# Test Operations
# ============================================================================

def test_list_feature_views(store: "FeatureStore"):
    """Test listing feature views"""
    try:
        feature_views = store.list_feature_views()
        call_center_fvs = [fv for fv in feature_views if 'call_center' in fv.name]
        print_result("List Feature Views", True, f"Found {len(call_center_fvs)} call center feature views")
        for fv in call_center_fvs:
            print(f"    - {fv.name}")
        return True, call_center_fvs
    except FeastPermissionError as e:
        print_result("List Feature Views", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("List Feature Views", False, f"Error: {e}")
        return False, None

def test_list_entities(store: "FeatureStore"):
    """Test listing entities"""
    try:
        entities = store.list_entities()
        customer_entity = [e for e in entities if e.name == "customer"]
        print_result("List Entities", True, f"Found customer entity")
        return True, entities
    except FeastPermissionError as e:
        print_result("List Entities", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("List Entities", False, f"Error: {e}")
        return False, None

def test_historical_features_call_center_90d(store: "FeatureStore"):
    """Test get_historical_features with call_center_90d"""
    customer_ids = ["CUST_000752", "CUST_000284", "CUST_000737"]
    entity_df = pd.DataFrame({
        "customer_id": customer_ids,
        "event_timestamp": [datetime.now()] * len(customer_ids)
    })
    
    try:
        historical = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "call_center_90d:call_type",
                "call_center_90d:call_duration_minutes",
                "call_center_90d:is_resolved",
                "call_center_90d:customer_satisfaction_score"
            ]
        )
        df = historical.to_df()
        print_result("Get Historical Features (call_center_90d)", True, 
                    f"Retrieved {len(df)} records with {len(df.columns)} columns")
        print(f"    Sample data shape: {df.shape}")
        return True, df
    except FeastPermissionError as e:
        print_result("Get Historical Features (call_center_90d)", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Get Historical Features (call_center_90d)", False, f"Error: {e}")
        return False, None

def test_online_features_call_center_90d(store: "FeatureStore"):
    """Test get_online_features with call_center_90d"""
    customer_ids = ["CUST_000752", "CUST_000284", "CUST_000737"]
    
    try:
        online = store.get_online_features(
            entity_rows=[{"customer_id": cid} for cid in customer_ids],
            features=[
                "call_center_90d:call_type",
                "call_center_90d:call_duration_minutes",
                "call_center_90d:is_resolved",
                "call_center_90d:customer_satisfaction_score"
            ]
        )
        df = online.to_df()
        print_result("Get Online Features (call_center_90d)", True,
                    f"Retrieved features for {len(df)} customers")
        print(f"    Sample data shape: {df.shape}")
        return True, df
    except FeastPermissionError as e:
        print_result("Get Online Features (call_center_90d)", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Get Online Features (call_center_90d)", False, f"Error: {e}")
        return False, None

def test_historical_features_call_center_predictive(store: "FeatureStore"):
    """Test get_historical_features with call_center_predictive"""
    customer_ids = ["CUST_000752", "CUST_000284", "CUST_000737"]
    entity_df = pd.DataFrame({
        "customer_id": customer_ids,
        "event_timestamp": [datetime.now()] * len(customer_ids)
    })
    
    try:
        historical = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "call_center_predictive:call_type",
                "call_center_predictive:call_duration_minutes",
                "call_center_predictive:is_resolved",
                "call_center_predictive:customer_satisfaction_score"
            ]
        )
        df = historical.to_df()
        print_result("Get Historical Features (call_center_predictive)", True,
                    f"Retrieved {len(df)} records with {len(df.columns)} columns")
        print(f"    Sample data shape: {df.shape}")
        return True, df
    except FeastPermissionError as e:
        print_result("Get Historical Features (call_center_predictive)", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Get Historical Features (call_center_predictive)", False, f"Error: {e}")
        return False, None

def test_online_features_call_center_predictive(store: "FeatureStore"):
    """Test get_online_features with call_center_predictive"""
    customer_ids = ["CUST_000752", "CUST_000284", "CUST_000737"]
    
    try:
        online = store.get_online_features(
            entity_rows=[{"customer_id": cid} for cid in customer_ids],
            features=[
                "call_center_predictive:call_type",
                "call_center_predictive:call_duration_minutes",
                "call_center_predictive:is_resolved",
                "call_center_predictive:customer_satisfaction_score"
            ]
        )
        df = online.to_df()
        print_result("Get Online Features (call_center_predictive)", True,
                    f"Retrieved features for {len(df)} customers")
        print(f"    Sample data shape: {df.shape}")
        return True, df
    except FeastPermissionError as e:
        print_result("Get Online Features (call_center_predictive)", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Get Online Features (call_center_predictive)", False, f"Error: {e}")
        return False, None

def test_list_feature_services(store: "FeatureStore"):
    """Test listing feature services"""
    try:
        services = store.list_feature_services()
        call_services = [s for s in services if 'call' in s.name.lower()]
        print_result("List Feature Services", True, f"Found {len(call_services)} call-related services")
        return True, services
    except FeastPermissionError as e:
        print_result("List Feature Services", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("List Feature Services", False, f"Error: {e}")
        return False, None

def test_list_data_sources(store: "FeatureStore"):
    """Test listing data sources - should fail for data scientists and data engineers"""
    try:
        data_sources = store.list_data_sources()
        print_result("List Data Sources", True, f"Found {len(data_sources)} data sources")
        for ds in data_sources[:5]:  # Show first 5
            print(f"    - {ds.name}")
        if len(data_sources) > 5:
            print(f"    ... and {len(data_sources) - 5} more")
        return True, data_sources
    except FeastPermissionError as e:
        print_result("List Data Sources", False, f"RBAC blocked (expected for data scientists/engineers): {e}")
        return False, None
    except Exception as e:
        print_result("List Data Sources", False, f"Error: {e}")
        return False, None

def test_get_data_source(store: "FeatureStore", data_source_name: str = "customer_data_source"):
    """Test getting a specific data source - should fail for data scientists and data engineers"""
    try:
        data_source = store.get_data_source(data_source_name)
        print_result("Get Data Source", True, f"Found data source: {data_source.name}")
        return True, data_source
    except FeastPermissionError as e:
        print_result("Get Data Source", False, f"RBAC blocked (expected for data scientists/engineers): {e}")
        return False, None
    except Exception as e:
        print_result("Get Data Source", False, f"Error: {e}")
        return False, None

def test_historical_features_transaction(store: "FeatureStore", fv_name: str = "transaction_30d_aggregations"):
    """Test get_historical_features with transaction-related feature view - should fail for read-only analysts"""
    customer_ids = ["CUST_000001", "CUST_000002", "CUST_000003"]
    entity_df = pd.DataFrame({
        "customer_id": customer_ids,
        "event_timestamp": [datetime.now()] * len(customer_ids)
    })
    
    try:
        # First, try to get the feature view to see what features are available
        try:
            fv = store.get_feature_view(fv_name)
            # Get first few feature names from the schema
            feature_names = [f.name for f in fv.schema[:2]] if hasattr(fv, 'schema') and fv.schema else ["amount", "transaction_type"]
        except:
            # If we can't get the feature view, use default feature names
            feature_names = ["amount", "transaction_type"]
        
        # Build feature list
        features = [f"{fv_name}:{name}" for name in feature_names]
        
        # Try to get features from transaction feature view
        historical = store.get_historical_features(
            entity_df=entity_df,
            features=features
        )
        df = historical.to_df()
        print_result(f"Get Historical Features ({fv_name})", True, 
                    f"Retrieved {len(df)} records with {len(df.columns)} columns")
        if not df.empty:
            print(f"    First 5 rows:\n{df.head(5)}")
        return True, df
    except FeastPermissionError as e:
        print_result(f"Get Historical Features ({fv_name})", False, 
                    f"RBAC blocked (expected for read-only analysts): {e}")
        return False, None
    except Exception as e:
        error_msg = str(e)
        # If feature doesn't exist, that's okay - we're testing permissions
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print_result(f"Get Historical Features ({fv_name})", False, 
                        f"Feature view not found: {error_msg}")
        else:
            print_result(f"Get Historical Features ({fv_name})", False, f"Error: {e}")
        return False, None

def test_online_features_transaction(store: "FeatureStore", fv_name: str = "transaction_30d_aggregations"):
    """Test get_online_features with transaction-related feature view - should fail for read-only analysts"""
    customer_ids = ["CUST_000001", "CUST_000002", "CUST_000003"]
    
    try:
        # First, try to get the feature view to see what features are available
        try:
            fv = store.get_feature_view(fv_name)
            # Get first few feature names from the schema
            feature_names = [f.name for f in fv.schema[:2]] if hasattr(fv, 'schema') and fv.schema else ["amount", "transaction_type"]
        except:
            # If we can't get the feature view, use default feature names
            feature_names = ["amount", "transaction_type"]
        
        # Build feature list
        features = [f"{fv_name}:{name}" for name in feature_names]
        
        online = store.get_online_features(
            entity_rows=[{"customer_id": cid} for cid in customer_ids],
            features=features
        )
        df = online.to_df()
        print_result(f"Get Online Features ({fv_name})", True,
                    f"Retrieved features for {len(df)} customers")
        if not df.empty:
            print(f"    First 5 rows:\n{df.head(5)}")
        return True, df
    except FeastPermissionError as e:
        print_result(f"Get Online Features ({fv_name})", False, 
                    f"RBAC blocked (expected for read-only analysts): {e}")
        return False, None
    except Exception as e:
        error_msg = str(e)
        # If feature doesn't exist, that's okay - we're testing permissions
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print_result(f"Get Online Features ({fv_name})", False, 
                        f"Feature view not found: {error_msg}")
        else:
            print_result(f"Get Online Features ({fv_name})", False, f"Error: {e}")
        return False, None

# ============================================================================
# Write Operations Tests
# ============================================================================

def test_get_or_create_entity(store: "FeatureStore", entity_name: str = "customer"):
    """Test getting or creating an entity"""
    try:
        # Try to get existing entity
        try:
            entity = store.get_entity(entity_name)
            print_result("Get Entity", True, f"Found existing entity: {entity.name}")
            return True, entity
        except Exception:
            # Entity doesn't exist, try to create it
            print(f"    Entity '{entity_name}' not found, attempting to create...")
            entity = Entity(
                name=entity_name,
                join_keys=["customer_id"],
                value_type=ValueType.STRING,
                description=f"Customer entity for RBAC testing"
            )
            store.apply([entity])
            print_result("Create Entity", True, f"Created entity: {entity.name}")
            return True, entity
    except FeastPermissionError as e:
        print_result("Get/Create Entity", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Get/Create Entity", False, f"Error: {e}")
        return False, None

def test_get_or_create_data_source(store: "FeatureStore", data_source_name: str = "customer_data_source"):
    """Test getting or creating a data source"""
    try:
        # Try to get existing data source
        try:
            data_source = store.get_data_source(data_source_name)
            print_result("Get Data Source", True, f"Found existing data source: {data_source.name}")
            return True, data_source
        except Exception:
            # Data source doesn't exist - this shouldn't happen in production
            # but we'll handle it gracefully
            print_result("Get Data Source", False, f"Data source '{data_source_name}' not found and cannot be created without source definition")
            return False, None
    except FeastPermissionError as e:
        print_result("Get Data Source", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Get Data Source", False, f"Error: {e}")
        return False, None

def test_create_feature_view(store: "FeatureStore", test_name_suffix: str = None):
    """Test creating a FeatureView programmatically"""
    if test_name_suffix is None:
        import uuid
        test_name_suffix = str(uuid.uuid4())[:8]
    
    test_fv_name = f"rbac_test_customer_simple_fv_{test_name_suffix}"
    
    try:
        # Get or create customer entity
        entity_success, customer_entity = test_get_or_create_entity(store, "customer")
        if not entity_success or not customer_entity:
            print_result("Create FeatureView", False, "Could not get/create customer entity")
            return False, None, None
        
        # Get existing data source
        ds_success, customer_data_source = test_get_or_create_data_source(store, "customer_data_source")
        if not ds_success or not customer_data_source:
            print_result("Create FeatureView", False, "Could not get customer_data_source")
            return False, None, None
        
        # Create FeatureView definition
        test_feature_view = FeatureView(
            name=test_fv_name,
            entities=[customer_entity],
            ttl=timedelta(days=30),
            schema=[
                Field(
                    name="age",
                    dtype=Int64,
                    description="Customer age for RBAC testing"
                ),
                Field(
                    name="credit_score",
                    dtype=Int64,
                    description="Customer credit score for RBAC testing"
                ),
            ],
            source=customer_data_source,
            description="Simple test feature view created programmatically for RBAC testing"
        )
        
        print_result("Define FeatureView", True, f"FeatureView '{test_fv_name}' defined")
        print(f"    - Entities: {[e.name if hasattr(e, 'name') else str(e) for e in test_feature_view.entities]}")
        print(f"    - Features: {[f.name for f in test_feature_view.schema]}")
        
        return True, test_feature_view, test_fv_name
        
    except FeastPermissionError as e:
        print_result("Create FeatureView", False, f"RBAC blocked: {e}")
        return False, None, None
    except Exception as e:
        print_result("Create FeatureView", False, f"Error: {e}")
        return False, None, None

def test_apply_feature_view(store: "FeatureStore", feature_view: FeatureView):
    """Test applying a FeatureView to the feature store"""
    try:
        store.apply([feature_view])
        print_result("Apply FeatureView", True, f"FeatureView '{feature_view.name}' applied successfully")
        return True
    except FeastPermissionError as e:
        print_result("Apply FeatureView", False, f"RBAC blocked: {e}")
        return False
    except Exception as e:
        print_result("Apply FeatureView", False, f"Error: {e}")
        return False

def test_verify_feature_view(store: "FeatureStore", fv_name: str):
    """Test verifying that a FeatureView exists in registry"""
    try:
        store.refresh_registry()
        fv = store.get_feature_view(fv_name)
        print_result("Verify FeatureView", True, f"FeatureView '{fv.name}' found in registry")
        return True, fv
    except FeastPermissionError as e:
        print_result("Verify FeatureView", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Verify FeatureView", False, f"Error: {e}")
        return False, None

def test_retrieve_from_created_feature_view(store: "FeatureStore", fv_name: str):
    """Test retrieving historical features from a newly created FeatureView"""
    customer_id = "CUST_000001"
    entity_df = pd.DataFrame({
        "customer_id": [customer_id],
        "event_timestamp": [datetime.now()]
    })
    
    try:
        historical = store.get_historical_features(
            entity_df=entity_df,
            features=[
                f"{fv_name}:age",
                f"{fv_name}:credit_score"
            ]
        )
        df = historical.to_df()
        print_result("Retrieve from Created FeatureView", True, 
                    f"Retrieved {len(df)} records with {len(df.columns)} columns")
        if not df.empty:
            print(f"    First row:\n{df.head(1)}")
        return True, df
    except FeastPermissionError as e:
        print_result("Retrieve from Created FeatureView", False, f"RBAC blocked: {e}")
        return False, None
    except Exception as e:
        print_result("Retrieve from Created FeatureView", False, f"Error: {e}")
        return False, None

def test_delete_feature_view(store: "FeatureStore", fv_name: str):
    """Test deleting a FeatureView (cleanup)"""
    try:
        store.delete_feature_view(fv_name)
        print_result("Delete FeatureView", True, f"FeatureView '{fv_name}' deleted successfully")
        return True
    except FeastPermissionError as e:
        print_result("Delete FeatureView", False, f"RBAC blocked: {e}")
        return False
    except Exception as e:
        # FeatureView might not exist or already deleted
        error_msg = str(e)
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            print_result("Delete FeatureView", True, f"FeatureView '{fv_name}' already deleted or doesn't exist")
            return True
        print_result("Delete FeatureView", False, f"Error: {e}")
        return False

# ============================================================================
# Main Test Runner
# ============================================================================

def run_rbac_tests(fs, username: str = None):
    """
    Run all RBAC tests for the specified user
    
    Args:
        fs: FeatureStore object (pre-initialized in notebook)
        username: Username to test as (e.g., "user1", "user5")
                 If None, will try to extract from fs or default to "user1"
    
    Usage in Jupyter:
        # Option 1: Explicit username
        run_rbac_tests(fs, username="user1")
        
        # Option 2: Username from fs context (if available)
        run_rbac_tests(fs)
    """
    # Determine username for display
    if username is None:
        # Try to get from fs if it has user context
        if hasattr(fs, 'user') and fs.user:
            username = fs.user.username
        else:
            username = "unknown"  # Default if not specified
    
    print_header(f"RBAC TEST - Call Center Features")
    print(f"\n👤 Testing as user: {username}")
    print(f"\n✅ Using provided FeatureStore object")
    
    # Track results
    results = {}
    
    # Run tests
    print_section("READ OPERATIONS - List Resources")
    results["list_feature_views"], _ = test_list_feature_views(fs)
    results["list_entities"], _ = test_list_entities(fs)
    results["list_feature_services"], _ = test_list_feature_services(fs)
    
    # Test DataSource access (Scenario 1 & 2: Should fail for data scientists and data engineers)
    print_section("READ OPERATIONS - DataSource Access (Permission Test)")
    results["list_data_sources"], _ = test_list_data_sources(fs)
    results["get_data_source"], _ = test_get_data_source(fs, "customer_data_source")
    
    print_section("READ OPERATIONS - Historical Features (Call Center)")
    results["historical_90d"], _ = test_historical_features_call_center_90d(fs)
    results["historical_predictive"], _ = test_historical_features_call_center_predictive(fs)
    
    # Test transaction-related feature views (Scenario 3: Should fail for read-only analysts)
    print_section("READ OPERATIONS - Transaction Features (Permission Test)")
    results["historical_transaction"], _ = test_historical_features_transaction(fs, "transaction_30d_aggregations")
    
    print_section("READ OPERATIONS - Online Features (Call Center)")
    results["online_90d"], _ = test_online_features_call_center_90d(fs)
    results["online_predictive"], _ = test_online_features_call_center_predictive(fs)
    
    # Test transaction online features (Scenario 3: Should fail for read-only analysts)
    print_section("READ OPERATIONS - Online Transaction Features (Permission Test)")
    results["online_transaction"], _ = test_online_features_transaction(fs, "transaction_30d_aggregations")
    
    # Write operations
    print_section("WRITE OPERATIONS - Create FeatureView")
    test_fv_success, test_feature_view, test_fv_name = test_create_feature_view(fs, username)
    
    if test_fv_success and test_feature_view:
        results["create_feature_view"] = True
        
        # Apply the feature view
        results["apply_feature_view"] = test_apply_feature_view(fs, test_feature_view)
        
        if results["apply_feature_view"]:
            # Verify it exists
            verify_success, _ = test_verify_feature_view(fs, test_fv_name)
            results["verify_feature_view"] = verify_success
            
            # Test retrieving from it
            retrieve_success, _ = test_retrieve_from_created_feature_view(fs, test_fv_name)
            results["retrieve_from_created_fv"] = retrieve_success
            
            # Cleanup: Delete the test feature view
            print_section("WRITE OPERATIONS - Cleanup")
            results["delete_feature_view"] = test_delete_feature_view(fs, test_fv_name)
        else:
            results["verify_feature_view"] = False
            results["retrieve_from_created_fv"] = False
            results["delete_feature_view"] = False
    else:
        results["create_feature_view"] = False
        results["apply_feature_view"] = False
        results["verify_feature_view"] = False
        results["retrieve_from_created_fv"] = False
        results["delete_feature_view"] = False
    
    # Print summary
    print_section("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if isinstance(v, bool) and v)
    failed_tests = total_tests - passed_tests
    
    print(f"\n📊 Results for {username}:")
    print(f"   ✅ Passed: {passed_tests}/{total_tests}")
    print(f"   ❌ Failed: {failed_tests}/{total_tests}")
    if total_tests > 0:
        print(f"   📈 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    print("\n📋 Detailed Results:")
    for test_name, success in results.items():
        if isinstance(success, bool):
            status = "✅" if success else "❌"
            # Add permission context for specific tests
            permission_note = ""
            if "data_source" in test_name.lower():
                permission_note = " (Should fail for data scientists/engineers)"
            elif "transaction" in test_name.lower():
                permission_note = " (Should fail for read-only analysts)"
            
            print(f"   {status} {test_name}{permission_note}")
    
    # Permission scenario summary
    print("\n🔐 Permission Scenario Validation:")
    print("   Note: For permission tests, 'Failed' means RBAC correctly blocked access")
    print("   Scenario 1: Data Scientists - No DataSource access")
    ds_blocked = not results.get('list_data_sources', True) and not results.get('get_data_source', True)
    print(f"      DataSource access: {'✅ Blocked (correct)' if ds_blocked else '❌ Allowed (should be blocked)'}")
    print("   Scenario 2: Data Engineers - No DataSource access")
    print(f"      DataSource access: {'✅ Blocked (correct)' if ds_blocked else '❌ Allowed (should be blocked)'}")
    print("   Scenario 3: Read-Only Analysts - No transaction feature views")
    trans_blocked = not results.get('historical_transaction', True) and not results.get('online_transaction', True)
    print(f"      Transaction features: {'✅ Blocked (correct)' if trans_blocked else '❌ Allowed (should be blocked)'}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    
    return results

run_rbac_tests(fs)