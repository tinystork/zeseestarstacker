"""
This script deeply inspects the drizzle module to understand its proper usage,
particularly for the Drizzle class and how to properly set up input/output WCS.
"""

import inspect
import sys
import warnings
from pprint import pprint

# Try importing drizzle
print("Attempting to import drizzle components...")
try:
    from drizzle.resample import Drizzle
    print("✓ Successfully imported Drizzle from drizzle.resample")
    
    # Check if dodrizzle module exists
    try:
        from drizzle import dodrizzle
        print("✓ Successfully imported dodrizzle module")
        for name in dir(dodrizzle):
            if not name.startswith('_'):
                print(f"  - Found in dodrizzle: {name}")
                
                # If it's a function, show its signature
                if callable(getattr(dodrizzle, name)):
                    try:
                        sig = inspect.signature(getattr(dodrizzle, name))
                        print(f"    Signature: {sig}")
                    except Exception as e:
                        print(f"    Could not inspect signature: {e}")
    except ImportError:
        print("✗ Could not import dodrizzle module")
    
    # Check for find_center function
    try:
        from drizzle.util import find_center
        print("✓ Found utility function find_center")
        sig = inspect.signature(find_center)
        print(f"  Signature: {sig}")
    except (ImportError, AttributeError):
        print("✗ Could not find find_center utility function")
    
    # Detailed Drizzle class inspection
    print("\n=== Drizzle Class Inspection ===")
    
    # Check initialization parameters
    print("* Drizzle.__init__ parameters:")
    sig = inspect.signature(Drizzle.__init__)
    print(f"  {sig}")
    
    # Check if the class has a docstring
    if Drizzle.__doc__:
        print("\n* Drizzle class docstring:")
        print(f"  {Drizzle.__doc__}")
    
    # Check add_image method
    if hasattr(Drizzle, 'add_image'):
        print("\n* Drizzle.add_image method:")
        sig = inspect.signature(Drizzle.add_image)
        print(f"  Signature: {sig}")
        if Drizzle.add_image.__doc__:
            print(f"  Docstring: {Drizzle.add_image.__doc__}")
    
    # Create a minimal Drizzle instance to inspect
    print("\n* Creating minimal Drizzle instance to inspect attributes...")
    try:
        driz = Drizzle(kernel='square')
        print("  ✓ Created Drizzle instance")
        
        # List all attributes and their types
        print("\n* Drizzle instance attributes:")
        for attr_name in dir(driz):
            if not attr_name.startswith('_'):
                attr = getattr(driz, attr_name)
                if not callable(attr):
                    print(f"  - {attr_name}: {type(attr).__name__} = {attr}")
        
        # List all methods
        print("\n* Drizzle instance methods:")
        for attr_name in dir(driz):
            if not attr_name.startswith('_'):
                attr = getattr(driz, attr_name)
                if callable(attr):
                    try:
                        sig = inspect.signature(attr)
                        print(f"  - {attr_name}{sig}")
                    except Exception:
                        print(f"  - {attr_name}()")
    except Exception as e:
        print(f"  ✗ Failed to create Drizzle instance: {e}")
    
    # Look for outwcs or output_wcs handling
    print("\n* Searching for WCS handling functionality...")
    for name in dir(Drizzle):
        if 'wcs' in name.lower() and not name.startswith('_'):
            attr = getattr(Drizzle, name)
            print(f"  - Found WCS-related attribute/method: {name}")
            if callable(attr):
                print(f"    Signature: {inspect.signature(attr)}")
    
except ImportError as e:
    print(f"✗ Failed to import Drizzle: {e}")
    print("Please ensure the drizzle module is installed.")
    sys.exit(1)

# Try to find the cdrizzle module which might contain lower-level functions
try:
    from drizzle import cdrizzle
    print("\n=== cdrizzle Module Inspection ===")
    for name in dir(cdrizzle):
        if not name.startswith('_'):
            print(f"- Found: {name}")
except ImportError:
    print("Note: Could not import cdrizzle module (may be internal implementation)")

print("\nInspection complete.")