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
    from bom_tools.core.processors import AIBOMProcessor, DATABOMProcessor
    print("   ✅ Processors imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import processors: {e}")
    sys.exit(1)

try:
    from bom_tools.core.agentic_rag import AgenticRAG, DirectLLM
    print("   ✅ RAG modules imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import RAG modules: {e}")
    sys.exit(1)

try:
    from bom_tools.utils.link_fallback import LinkFallbackFinder
    print("   ✅ Link fallback imported successfully")
except Exception as e:
    print(f"   ❌ Failed to import link fallback: {e}")
    sys.exit(1)

try:
    from bom_tools.web.app import app
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
        "src/bom_tools/__init__.py"
        "src/bom_tools/web/app.py"
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
    echo "   3. Run: python -m bom_tools.web.app"
    echo ""
else
    echo ""
    echo "❌ Import test failed. Please check the error messages above."
    exit 1
fi
