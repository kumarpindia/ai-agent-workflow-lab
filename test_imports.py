#!/usr/bin/env python3
"""
Test script to verify that all required packages are installed correctly.
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")

        import openai
        print("✓ OpenAI imported successfully")

        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")

        import PyPDF2
        print("✓ PyPDF2 imported successfully")

        import docx2txt
        print("✓ docx2txt imported successfully")

        print("\n🎉 All packages imported successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    print("Testing package imports...")
    success = test_imports()
    sys.exit(0 if success else 1)