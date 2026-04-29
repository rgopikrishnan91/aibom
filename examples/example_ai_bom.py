#!/usr/bin/env python3
"""
Example script for processing an AI model BOM
"""
from aikaboom.core.processors import AIBOMProcessor
import json

def main():
    # Initialize processor with LOCAL embeddings (default, free, no API key needed)
    processor = AIBOMProcessor(
        model="gpt-4o",
        mode="rag",
        llm_provider="openai",
        use_case="complete",
        embedding_provider="local",  # Use local HuggingFace embeddings (default)
        embedding_model="BAAI/bge-small-en-v1.5"  # Good balance of quality and speed
    )
    
    # Alternative: Use OpenAI embeddings (requires OPENAI_API_KEY, costs money)
    # processor = AIBOMProcessor(
    #     model="gpt-4o",
    #     mode="rag",
    #     llm_provider="openai",
    #     use_case="complete",
    #     embedding_provider="openai"
    # )
    
    # Process a model
    metadata = processor.process_ai_model(
        repo_id="microsoft/DialoGPT-medium",
        arxiv_url="https://arxiv.org/abs/1911.00536",
        github_url="https://github.com/microsoft/DialoGPT"
    )
    
    # Save results
    with open("example_ai_bom_output.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ AI BOM generated successfully!")
    print(f"Model ID: {metadata['model_id']}")
    print(f"Direct fields: {len(metadata.get('direct_fields', {}))}")
    print(f"RAG fields: {len(metadata.get('rag_fields', {}))}")

if __name__ == "__main__":
    main()
