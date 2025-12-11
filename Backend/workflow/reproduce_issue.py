import sys
import os

# Adjust path to make imports work like in main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Attempting to import DocumentComplianceAgent...")
    from workflow.doc_analyzer import DocumentComplianceAgent
    print("Import successful.")

    print("Attempting to initialize DocumentComplianceAgent...")
    agent = DocumentComplianceAgent()
    print("Initialization successful.")

    print("Checking RuleManager configuration...")
    # Verify that we can call filter with the new argument
    rules = agent.rule_manager._filter_applicable_rules(only_category='structurelle')
    print(f"Filtered rules count (structurelle): {len(rules)}")
    
except Exception as e:
    print(f"\n❌ CAUGHT EXCEPTION:\n{e}")
    import traceback
    traceback.print_exc()
