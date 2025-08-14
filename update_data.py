import subprocess
import sys

# List of all scripts
all_scripts = [
    "data_updating_scripts/get_data.py",
    "data_updating_scripts/fix_pdf_bills.py",
    "data_updating_scripts/migrate_iapp_categories.py",
    "data_updating_scripts/mark_no_text_bills.py",
    "data_updating_scripts/known_bills_status.py",
    "data_updating_scripts/generate_summaries.py",
    "data_updating_scripts/generate_suggested_questions.py",
    "data_updating_scripts/generate_reports.py",
    "data_updating_scripts/eu_vectorstore.py"
]

# Ask user if they want to pull new data
print("Do you want to pull new data from LegiScan?")
print("Enter 'y' or 'yes' to pull new data, or 'n' or 'no' to skip and use existing data:")
response = input().strip().lower()

# Determine which scripts to run
if response in ['y', 'yes']:
    print("\n✓ Will pull new data from LegiScan")
    scripts_to_run = all_scripts  # Run all scripts including get_data.py
elif response in ['n', 'no']:
    print("\n✓ Skipping data pull, using existing data")
    scripts_to_run = all_scripts[2:]  # Skip get_data.py and fix_pdf_bills.py, start from migrate_iapp
else:
    print(f"\n✗ Invalid response '{response}'. Please run the script again and enter 'y' or 'n'.")
    sys.exit(1)

# Run the selected scripts
print(f"\nWill run {len(scripts_to_run)} scripts:")
for script in scripts_to_run:
    print(f"  - {script}")

print("\n" + "="*50)

for script in scripts_to_run:
    print(f"\n--- Running {script} ---")
    print("="*50)
    
    # Run without capturing output - this allows real-time display
    result = subprocess.run(["python", script])
    
    # Check if the script failed
    if result.returncode != 0:
        print(f"\n✗ Script {script} failed with return code {result.returncode}")
        print("Do you want to continue with the remaining scripts? (y/n):")
        continue_response = input().strip().lower()
        if continue_response not in ['y', 'yes']:
            print("Stopping pipeline execution.")
            sys.exit(1)
    else:
        print(f"✓ {script} completed successfully")

print("\n" + "="*50)
print("✓ Pipeline execution completed!")