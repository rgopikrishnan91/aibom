#!/usr/bin/env python3
"""
Example script for processing a dataset BOM
"""
from aikaboom.core.processors import DATABOMProcessor
import json

def main():
    # Initialize processor with LOCAL embeddings (default, free, no API key needed)
    processor = DATABOMProcessor(
        model="gpt-4o",
        mode="rag",
        llm_provider="openai",
        use_case="complete",
        embedding_provider="local",  # Use local HuggingFace embeddings (default)
        embedding_model="BAAI/bge-small-en-v1.5"  # Good balance of quality and speed
    )
    
    # Alternative: Use OpenAI embeddings (requires OPENAI_API_KEY, costs money)
    # processor = DATABOMProcessor(
    #     model="gpt-4o",
    #     mode="rag",
    #     llm_provider="openai",
    #     use_case="complete",
    #     embedding_provider="openai"
    # )
    
    # Process a dataset
    metadata = processor.process_dataset(
        arxiv_url="https://arxiv.org/abs/1606.05250",
        github_url="https://github.com/rajpurkar/SQuAD-explorer",
        hf_url="https://huggingface.co/datasets/squad"
    )
    
    # Save results
    with open("example_data_bom_output.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Dataset BOM generated successfully!")
    print(f"Dataset ID: {metadata['dataset_id']}")
    print(f"Direct metadata: {len(metadata.get('direct_metadata', {}))}")
    print(f"RAG metadata: {len(metadata.get('rag_metadata', {}))}")

if __name__ == "__main__":
    main()
