#!/usr/bin/env python3
"""
known_bills_status.py

Reads known_bills_fixed.json and updates existing known_bills_visualize.json.
Merges new bills and updates existing ones while preserving clean status fields.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

def map_status(original_status):
    """Map LegiScan status codes to clean display text."""
    
    # Direct mapping for numeric codes
    status_mapping = {
        "0": "Inactive",           # Pre-filed
        "1": "Active",             # Introduced  
        "2": "Active",             # Engrossed
        "3": "Active",             # Enrolled
        "4": "Signed Into Law",    # Passed
        "5": "Vetoed",             # Vetoed
        "6": "Inactive",           # Failed
        "7": "Signed Into Law",    # Override
        "8": "Signed Into Law",    # Chaptered
        "9": "Active",             # Refer
        "10": "Active",            # Report Pass
        "11": "Inactive",          # Report DNP
        "12": "Active",            # Draft
        
        # Integer versions
        0: "Inactive", 1: "Active", 2: "Active", 3: "Active",
        4: "Signed Into Law", 5: "Vetoed", 6: "Inactive",
        7: "Signed Into Law", 8: "Signed Into Law", 9: "Active",
        10: "Active", 11: "Inactive", 12: "Active"
    }
    
    # Try direct mapping first
    if original_status in status_mapping:
        return status_mapping[original_status]
    
    # Handle text statuses
    if original_status:
        status_str = str(original_status).lower()
        if "pass" in status_str or "signed" in status_str or "enacted" in status_str:
            return "Signed Into Law"
        elif "veto" in status_str:
            return "Vetoed"
        elif "fail" in status_str or "dead" in status_str or "killed" in status_str:
            return "Inactive"
        elif "active" in status_str or "intro" in status_str or "pending" in status_str:
            return "Active"
    
    # Default fallback
    return "Inactive"

def create_bill_key(bill):
    """Create a unique key for each bill."""
    return f"{bill.get('state', 'Unknown')}_{bill.get('bill_number', 'Unknown')}"

def merge_bill_data(new_bill, existing_bill=None):
    """Merge new bill data with existing bill, preserving processed fields."""
    if not existing_bill:
        # New bill - create clean version
        merged_bill = new_bill.copy()
        original_status = merged_bill.get('status')
        merged_bill['original_status'] = original_status
        merged_bill['status'] = map_status(original_status)
        merged_bill['status_updated_at'] = datetime.now(timezone.utc).isoformat()
        return merged_bill
    
    # Existing bill - merge carefully
    merged_bill = existing_bill.copy()
    
    # Update with new data from source (except status fields)
    for key, value in new_bill.items():
        if key not in ['status', 'original_status', 'status_updated_at']:
            merged_bill[key] = value
    
    # Check if original status actually changed
    new_original_status = new_bill.get('status')
    old_original_status = existing_bill.get('original_status')
    
    # Convert both to strings for comparison to handle int vs string
    new_status_str = str(new_original_status) if new_original_status is not None else None
    old_status_str = str(old_original_status) if old_original_status is not None else None
    
    if new_status_str != old_status_str:
        # Real change in underlying data
        new_clean_status = map_status(new_original_status)
        merged_bill['original_status'] = new_original_status
        merged_bill['status'] = new_clean_status
        merged_bill['status_updated_at'] = datetime.now(timezone.utc).isoformat()
        return merged_bill
    
    # No change - keep existing clean status but ensure it's properly mapped
    if 'status' not in merged_bill or merged_bill['status'] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
        # Only remap if status is still numeric (needs cleaning)
        merged_bill['status'] = map_status(old_original_status)
    
    return merged_bill

def main():
    
    # File paths
    input_file = Path("data/known_bills_fixed.json")
    output_file = Path("data/known_bills_visualize.json")
    
    print(f"Reading source bills from: {input_file}")
    
    # Load source bills data
    with open(input_file, 'r', encoding='utf-8') as f:
        source_bills = json.load(f)
    
    print(f"Loaded {len(source_bills)} bills from source")
    
    # Load existing visualization data if it exists
    existing_bills = []
    if output_file.exists():
        print(f"Reading existing visualization data from: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_bills = json.load(f)
        print(f"Loaded {len(existing_bills)} existing bills")
    else:
        print("No existing visualization data found - will create new file")
    
    # Create lookup dictionary for existing bills
    existing_bills_dict = {}
    for bill in existing_bills:
        key = create_bill_key(bill)
        existing_bills_dict[key] = bill
    
    # Process and merge bills
    merged_bills = []
    new_bills_count = 0
    updated_bills_count = 0
    unchanged_bills_count = 0
    
    print(f"\nProcessing {len(source_bills)} bills...")
    
    for source_bill in source_bills:
        bill_key = create_bill_key(source_bill)
        existing_bill = existing_bills_dict.get(bill_key)
        
        if existing_bill:
            # Check if anything actually changed
            old_original_status = existing_bill.get('original_status')
            new_original_status = source_bill.get('status')
            
            if old_original_status != new_original_status:
                updated_bills_count += 1
            else:
                unchanged_bills_count += 1
        else:
            new_bills_count += 1
        
        merged_bill = merge_bill_data(source_bill, existing_bill)
        merged_bills.append(merged_bill)
    
    # Check for bills that exist in visualization but not in source (removed bills)
    source_keys = {create_bill_key(bill) for bill in source_bills}
    existing_keys = set(existing_bills_dict.keys())
    removed_keys = existing_keys - source_keys
    
    # Save updated bills
    print(f"\nSaving updated bills to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_bills, f, indent=2, ensure_ascii=False)
    
    # Show status distribution
    status_counts = {}
    for bill in merged_bills:
        status = bill['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Summary
    print(f"\n✅ Update complete!")
    print(f"  📊 Total bills: {len(merged_bills)}")
    if new_bills_count > 0:
        print(f"  🆕 New bills: {new_bills_count}")
    if updated_bills_count > 0:
        print(f"  🔄 Updated bills: {updated_bills_count}")
    if unchanged_bills_count > 0:
        print(f"  ✅ Unchanged bills: {unchanged_bills_count}")
    if removed_keys:
        print(f"  🗑️  Removed bills: {len(removed_keys)}")
    
    if new_bills_count == 0 and updated_bills_count == 0:
        print(f"  🎉 All bills are up to date - no changes needed!")
    
    print(f"\n📈 Status distribution:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    print(f"\n📁 Clean data saved to: {output_file}")
    print("Now run: streamlit run scripts/visualize-MIT.py")

if __name__ == "__main__":
    main()