#!/bin/bash
# Quick test script to verify the restructured installation

echo "🧪 Testing BOM Tools Installation..."
echo ""

# Test imports
echo "1️⃣ Testing Python imports..."
python3 << 'PYEOF'
import sys
sys.path.insert(0, 'src')

try:
    from aikaboom.core.processors import AIBOMProcessor, DATABOMProcessor
    print("   ✅ Processors imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import processors: {e}")
    sys.exit(1)

try:
    from aikaboom.core.agentic_rag import AgenticRAG, DirectLLM
    print("   ✅ RAG modules imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import RAG modules: {e}")
    sys.exit(1)

try:
    from aikaboom.utils.link_fallback import LinkFallbackFinder
    print("   ✅ Link fallback imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import link fallback: {e}")
    sys.exit(1)

try:
    from aikaboom.web.app import app
    print("   ✅ Web app imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import web app: {e}")
    sys.exit(1)

print("\n✅ All imports successful!")
PYEOF

if [ $? -eq 0 ]; then
    echo ""
    echo "2️⃣ Checking files..."
    
    # Check key files exist
    files=(
        "src/aikaboom/__init__.py"
        "src/aikaboom/web/app.py"
        "requirements.txt"
        "setup.py"
        "pyproject.toml"
        ".env.example"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            echo "   ✅ $file"
        else
            echo "   ❌ Missing: $file"
        fi
    done
    
    echo ""
    echo "✅ Structure verification complete!"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Install: pip install -e ."
    echo "   2. Configure: cp .env.example .env (and edit with your keys)"
    echo "   3. Run: python -m aikaboom.web.app"
    echo ""
else
    echo ""
    echo "❌ Import test failed. Please check the error messages above."
    exit 1
fi
