"""
EcoFusion 3.0 - Application Launcher
Run this script to choose between Vendor or Customer portal
"""

import subprocess
import sys

def main():
    print("=" * 60)
    print("🌿 ECOFUSION 3.0 - Application Launcher")
    print("=" * 60)
    print()
    print("Please select which portal to launch:")
    print()
    print("1. 🔧 VENDOR PORTAL - Full analytics and business intelligence")
    print("2. 🌱 CUSTOMER PORTAL - Simple and user-friendly interface")
    print("3. ❌ Exit")
    print()
    
    while True:
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🔧 Launching Vendor Portal...")
            print("Opening in your browser...\n")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app_vendor.py"])
            break
            
        elif choice == "2":
            print("\n🌱 Launching Customer Portal...")
            print("Opening in your browser...\n")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app_customer.py"])
            break
            
        elif choice == "3":
            print("\n👋 Thank you for using EcoFusion 3.0!")
            sys.exit(0)
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application closed. Thank you!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)