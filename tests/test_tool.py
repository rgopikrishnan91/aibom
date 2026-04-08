#!/usr/bin/env python3
"""
Process Golden Set CSV files and generate BOMs

This script processes AIBOM and DataBOM golden set CSV files to generate
Bill of Materials (BOM) with configurable options:

Features:
- Supports both AIBOM and DataBOM golden sets
- Configurable processing modes: RAG or Direct
- Multiple LLM providers: Ollama or OpenRouter
- Checkpoint support for resuming interrupted processing
- Outputs to results-golden-set/ folder in JSON and CSV formats

Usage:
    python tests/test_tool.py --mode rag --provider ollama
    python tests/test_tool.py --mode direct --provider ollama --limit 5
    python tests/test_tool.py --mode rag --provider openrouter --model qwen/qwen3-coder:free
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
base_dir = Path(__file__).parent.parent
env_file = base_dir / ".env"
if env_file.exists():
    load_dotenv(env_file)

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bom_tools.core.processors import AIBOMProcessor, DATABOMProcessor


def setup_directories():
    """Create necessary directories for results"""
    base_dir = Path(__file__).parent.parent
    results_dir = base_dir / "results-golden-set"
    results_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different output types
    (results_dir / "json").mkdir(exist_ok=True)
    (results_dir / "csv").mkdir(exist_ok=True)
    (results_dir / "logs").mkdir(exist_ok=True)
    
    return results_dir


def detect_bom_type(df):
    """Detect if CSV is AIBOM or DataBOM based on columns"""
    # AIBOM-specific columns
    aibom_indicators = ['autonomyType', 'typeOfModel', 'modelExplainability']
    # DataBOM-specific columns
    databom_indicators = ['anonymizationMethodUsed', 'datasetType', 'dataCollectionProcess']
    
    # Check which type of columns are present
    has_aibom = any(col in df.columns for col in aibom_indicators)
    has_databom = any(col in df.columns for col in databom_indicators)
    
    if has_aibom and not has_databom:
        return 'aibom'
    elif has_databom and not has_aibom:
        return 'databom'
    else:
        # Fallback: check filename
        return None


def load_csv_data(csv_path, start_row=51):
    """Load the CSV file and return data starting from specified row"""
    df = pd.read_csv(csv_path)
    
    # Detect BOM type
    bom_type = detect_bom_type(df)
    if bom_type is None:
        # Fallback: detect from filename
        if 'AIBOM' in str(csv_path):
            bom_type = 'aibom'
        elif 'DataBOM' in str(csv_path):
            bom_type = 'databom'
        else:
            bom_type = 'aibom'  # Default to AIBOM
    
    print(f"\n{'='*70}")
    print(f"LOADED CSV DATA")
    print(f"{'='*70}")
    print(f"BOM Type:       {bom_type.upper()}")
    print(f"Total rows in CSV: {len(df)}")
    print(f"Starting from row: {start_row} (index {start_row-1})")
    print(f"Rows to process: {len(df) - start_row + 1}")
    print(f"{'='*70}\n")
    
    # Return rows from start_row onwards (convert to 0-based index)
    return df.iloc[start_row-1:].reset_index(drop=True), df, bom_type


def get_user_configuration():
    """Get user configuration via command line arguments"""
    parser = argparse.ArgumentParser(
        description='Process Golden Set CSV files and generate BOMs with configurable options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use RAG mode with Ollama (FREE - no API keys needed for LLM or embeddings)
    python tests/test_tool.py --mode rag --provider ollama
  
  # Use RAG mode with OpenRouter (requires credits)
    python tests/test_tool.py --mode rag --provider openrouter --model qwen/qwen-2.5-72b-instruct
  
  # Use Direct mode with Ollama
    python tests/test_tool.py --mode direct --provider ollama
  
  # Process only first 5 rows for testing
    python tests/test_tool.py --limit 5
  
  # Start from a different row (default is 1)
    python tests/test_tool.py --start-row 100
  
  # Force restart from beginning (ignore checkpoint)
    python tests/test_tool.py --force-restart
  
  # Use custom input CSV
    python tests/test_tool.py --input-csv Golden_Set/DataBOM_Golden-Set_main-version.csv
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['rag', 'direct'],
        default='rag',
        help='Processing mode: "rag" for RAG-based retrieval, "direct" for direct LLM (default: rag)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        choices=['ollama', 'openrouter'],
        default='ollama',
        help='LLM provider: "ollama" for a local Ollama server, or "openrouter" for OpenRouter (default: ollama)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name (default: llama3:70b for ollama, qwen/qwen-2.5-72b-instruct for openrouter)'
    )
    
    parser.add_argument(
        '--ollama-url',
        type=str,
        default=None,
        help='Ollama base URL (default: http://localhost:11434/v1/)'
    )
    
    parser.add_argument(
        '--start-row',
        type=int,
        default=1,
        help='Starting row number (1-based index, default: 1)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of rows to process (for testing, default: process all)'
    )
    
    parser.add_argument(
        '--use-case',
        type=str,
        default='complete',
        help='BOM use case type (default: complete)'
    )
    
    parser.add_argument(
        '--input-csv',
        type=str,
        default='Golden_Set/AIBOM_Golden-Set_main-version.csv',
        help='Input CSV path relative to repo root (default: Golden_Set/AIBOM_Golden-Set_main-version.csv)'
    )
    
    parser.add_argument(
        '--force-restart',
        action='store_true',
        help='Force restart from beginning, ignoring any checkpoints'
    )
    
    parser.add_argument(
        '--embedding-provider',
        type=str,
        choices=['local'],
        default='local',
        help='Embedding provider: "local" for free HuggingFace models (default and only supported option in this script)'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Embedding model name (only for local provider, default: sentence-transformers/all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    # Set default model based on provider if not specified
    if args.model is None:
        if args.provider == 'ollama':
            args.model = 'llama3:70b'
        else:
            args.model = 'qwen/qwen-2.5-72b-instruct'
    
    # Display configuration
    print(f"\n{'='*70}")
    print(f"CONFIGURATION")
    print(f"{'='*70}")
    print(f"Processing Mode:  {args.mode.upper()}")
    print(f"LLM Provider:     {args.provider.upper()}")
    print(f"Model:            {args.model}")
    if args.provider == 'ollama':
        print(f"Ollama URL:       {args.ollama_url or 'default (http://localhost:11434/v1/)'}")
    elif args.provider == 'openrouter':
        print(f"OpenRouter URL:   {os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')}")
    print(f"Embedding:        {args.embedding_provider.upper()}")
    if args.embedding_provider == 'local':
        print(f"Embedding Model:  {args.embedding_model}")
    print(f"Use Case:         {args.use_case}")
    print(f"Input CSV:        {args.input_csv}")
    print(f"Start Row:        {args.start_row}")
    print(f"Row Limit:        {args.limit if args.limit else 'None (process all)'}")
    print(f"Checkpoint:       {'DISABLED (force restart)' if args.force_restart else 'ENABLED (auto-resume)'}")
    print(f"{'='*70}\n")
    
    return args


def process_row(row, row_index, ai_processor, data_processor, config, bom_type):
    """Process a single row from the CSV"""
    repo_id = row['repo_id']
    github_link = row['github_link'] if pd.notna(row['github_link']) else None
    arxiv_link = row['arxiv_paper'] if pd.notna(row['arxiv_paper']) else None
    
    # Set item type based on BOM type
    item_type = 'model' if bom_type == 'aibom' else 'dataset'
    
    print(f"\n{'='*70}")
    print(f"PROCESSING ROW {row_index + config.start_row}")
    print(f"{'='*70}")
    print(f"Repo ID:        {repo_id}")
    print(f"Type:           {item_type} ({bom_type.upper()})")
    print(f"GitHub:         {github_link or 'N/A'}")
    print(f"ArXiv:          {arxiv_link or 'N/A'}")
    print(f"{'='*70}")

    try:
        used_model = config.model
        if item_type == 'model':
            # Process as AI model (AIBOM)
            result = ai_processor.process_ai_model(
                repo_id=repo_id,
                arxiv_url=arxiv_link,
                github_url=github_link
            )
        else:
            # Process as dataset (DataBOM)
            # Construct HuggingFace dataset URL
            hf_url = f"https://huggingface.co/datasets/{repo_id}" if repo_id else None
            result = data_processor.process_dataset(
                arxiv_url=arxiv_link,
                github_url=github_link,
                hf_url=hf_url
            )
        
        return {
            'row_index': row_index + config.start_row,
            'repo_id': repo_id,
            'type': item_type,
            'github_link': github_link,
            'arxiv_link': arxiv_link,
            'model': used_model,
            'result': result,
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"\n❌ ERROR processing row {row_index + config.start_row}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'row_index': row_index + config.start_row,
            'repo_id': repo_id,
            'type': item_type,
            'github_link': github_link,
            'arxiv_link': arxiv_link,
            'model': config.model,
            'result': None,
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def format_output_like_ui(result, config):
    """Format output to match UI format (like the example JSON)"""
    item_type = result['type']
    raw_result = result['result']
    result_model = result.get('model', config.model)
    
    # Generate config name like UI does
    config_name = f"{config.provider}-{config.mode}_{config.use_case}_{result_model.replace(':', '_').replace('/', '_')}"
    
    if item_type == 'model':
        # For AI models
        formatted = {
            "uid": raw_result.get('model_id'),
            "model_id": raw_result.get('model_id'),
            "model_name": result['repo_id'],
            "use_case": config.use_case,
            "direct_fields": raw_result.get('direct_fields', {}),
            "rag_fields": raw_result.get('rag_fields', {}),
            "processing_mode": config.mode,
            "llm_provider": config.provider,
            "model": result_model,
            "config_name": config_name,
            "timestamp": result['timestamp']
        }
    else:
        # For datasets - match the example format
        direct_metadata = raw_result.get('direct_metadata', {})
        rag_metadata = raw_result.get('rag_metadata', {})
        
        # Convert direct_metadata to triplet format if needed
        direct_fields = {}
        for key, value in direct_metadata.items():
            if not key.endswith('_source') and not key.endswith('_conflicts'):
                direct_fields[key] = {
                    "value": value,
                    "conflict": direct_metadata.get(f"{key}_conflicts"),
                    "source": direct_metadata.get(f"{key}_source")
                }
        
        # Convert rag_metadata to triplet format if needed
        rag_fields = {}
        for key, value in rag_metadata.items():
            if not key.endswith('_sources') and not key.endswith('_conflicts'):
                rag_fields[key] = {
                    "value": value,
                    "conflict": rag_metadata.get(f"{key}_conflicts", "No conflicts detected"),
                    "source": rag_metadata.get(f"{key}_sources", "")
                }
        
        dataset_id = raw_result.get('dataset_id', result['repo_id'].replace('/', '_'))
        
        formatted = {
            "uid": dataset_id,
            "dataset_id": dataset_id,
            "dataset_name": result['repo_id'],
            "use_case": config.use_case,
            "direct_fields": direct_fields,
            "rag_fields": rag_fields,
            "processing_mode": config.mode,
            "llm_provider": config.provider,
            "model": result_model,
            "config_name": config_name,
            "timestamp": result['timestamp']
        }
    
    return formatted


def build_wide_csv_row(result, config):
    """Build a wide-format CSV row matching golden set structure"""
    # Define AIBOM fields
    aibom_fields = [
        'autonomyType', 'domain', 'energyConsumption', 'hyperparameter',
        'informationAboutApplication', 'informationAboutTraining', 'limitation',
        'metric', 'metricDecisionThreshold', 'modelDataPreprocessing',
        'modelExplainability', 'safetyRiskAssessment', 'standardCompliance',
        'typeOfModel', 'useSensitivePersonalInformation'
    ]
    
    # Define DataBOM fields
    databom_fields = [
        'anonymizationMethodUsed', 'confidentialityLevel', 'dataCollectionProcess',
        'dataPreprocessing', 'datasetAvailability', 'datasetNoise', 'datasetSize',
        'datasetType', 'datasetUpdateMechanism', 'hasSensitivePersonalInformation',
        'intendedUse', 'knownBias', 'sensorUsed'
    ]
    
    # Start with basic metadata
    row = {
        'repo_id': result['repo_id'],
        'arxiv_paper': result.get('arxiv_link', ''),
        'github_link': result.get('github_link', '')
    }
    
    # Get the appropriate fields list based on type
    fields_list = aibom_fields if result['type'] == 'model' else databom_fields
    
    # Extract fields from result
    # FIXED: Handle different structures for AIBOM and DataBOM
    raw_result = result.get('result', {})
    
    if result['type'] == 'dataset':
        # For DataBOM: processor returns direct_metadata and rag_metadata
        # We need to convert rag_metadata to triplet format
        rag_metadata = raw_result.get('rag_metadata', {})
        
        # Convert rag_metadata to triplet format
        rag_fields = {}
        for key, value in rag_metadata.items():
            if not key.endswith('_sources') and not key.endswith('_conflicts'):
                rag_fields[key] = {
                    "value": value,
                    "conflict": rag_metadata.get(f"{key}_conflicts", "No conflicts detected"),
                    "source": rag_metadata.get(f"{key}_sources", "")
                }
        
        fields_data = rag_fields
    else:
        # For AIBOM: processor returns direct_fields and rag_fields already in triplet format
        # Always use rag_fields for AIBOM-specific fields
        fields_data = raw_result.get('rag_fields', {})
    
    # Add each field and its sources to the row
    for field_name in fields_list:
        field_info = fields_data.get(field_name, {})
        
        # Handle both dictionary format and direct value
        if isinstance(field_info, dict):
            value = field_info.get('value', 'Not found')
            source = field_info.get('source', 'N/A')
        else:
            value = field_info if field_info else 'Not found'
            source = 'N/A'
        
        # Clean up the value and source
        if value is None or value == '':
            value = 'Not found'
        if source is None or source == '':
            source = 'N/A'
        
        row[field_name] = value
        row[f"{field_name}_sources"] = source
    
    # Add processing metadata at the end
    row['processing_status'] = result['status']
    row['result_file'] = ''
    row['processing_timestamp'] = result['timestamp']
    
    return row


def save_results(results, results_dir, config):
    """Save results in multiple formats (UI-compatible format)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = f"{config.mode}_{config.provider}_{config.model.replace(':', '_').replace('/', '_')}"
    
    # Format each result to match UI output
    formatted_results = []
    for result in results:
        if result['status'] == 'success':
            formatted = format_output_like_ui(result, config)
            formatted_results.append(formatted)
            
            # Save individual BOM files (like UI does)
            item_id = formatted.get('dataset_id') or formatted.get('model_id')
            item_type = 'databom' if result['type'] == 'dataset' else 'aibom'
            individual_file = results_dir / "json" / f"{item_id}_{config.provider}-{config.mode}_{config.use_case}_{config.model.replace(':', '_').replace('/', '_')}_{item_type}.json"
            
            with open(individual_file, 'w', encoding='utf-8') as f:
                json.dump(formatted, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {item_type.upper()} to: {individual_file}")
    
    # Save combined results
    json_file = results_dir / "json" / f"results_{mode_str}_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n✓ Saved combined JSON results to: {json_file}")
    
    # Save as CSV in golden set format (separate AIBOM and DataBOM)
    aibom_rows = []
    databom_rows = []
    
    for result in results:
        if result['status'] == 'success':
            wide_row = build_wide_csv_row(result, config)
            
            if result['type'] == 'model':
                aibom_rows.append(wide_row)
            else:
                databom_rows.append(wide_row)
    
    # Save AIBOM CSV if we have any AI models
    if aibom_rows:
        aibom_csv_file = results_dir / "csv" / f"AIBOM_{mode_str}_{timestamp}.csv"
        df_aibom = pd.DataFrame(aibom_rows)
        df_aibom.to_csv(aibom_csv_file, index=False)
        print(f"✓ Saved AIBOM CSV (golden-set format) to: {aibom_csv_file}")
    
    # Save DataBOM CSV if we have any datasets
    if databom_rows:
        databom_csv_file = results_dir / "csv" / f"DataBOM_{mode_str}_{timestamp}.csv"
        df_databom = pd.DataFrame(databom_rows)
        df_databom.to_csv(databom_csv_file, index=False)
        print(f"✓ Saved DataBOM CSV (golden-set format) to: {databom_csv_file}")
    
    csv_file = aibom_csv_file if aibom_rows else (databom_csv_file if databom_rows else None)
    return json_file, csv_file


def update_input_csv_with_links(original_df, results, csv_path):
    """Update the original CSV with finding links"""
    # Add new columns if they don't exist
    if 'processing_status' not in original_df.columns:
        original_df['processing_status'] = ''
    if 'result_file' not in original_df.columns:
        original_df['result_file'] = ''
    if 'processing_timestamp' not in original_df.columns:
        original_df['processing_timestamp'] = ''
    
    # Update rows with processing results
    for result in results:
        row_idx = result['row_index'] - 1  # Convert to 0-based index
        original_df.at[row_idx, 'processing_status'] = result['status']
        original_df.at[row_idx, 'processing_timestamp'] = result['timestamp']
        
        if result['status'] == 'error':
            original_df.at[row_idx, 'result_file'] = f"ERROR: {result.get('error', 'Unknown')}"
    
    # Save updated CSV
    backup_path = csv_path.replace('.csv', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    original_df.to_csv(backup_path, index=False)
    print(f"\n✓ Created backup of original CSV: {backup_path}")
    
    original_df.to_csv(csv_path, index=False)
    print(f"✓ Updated original CSV with processing status: {csv_path}")


def print_processing_summary(results, config):
    """Print a summary of processing results"""
    print(f"\n{'='*70}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total rows processed: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"\nConfiguration:")
    print(f"  Mode:     {config.mode}")
    print(f"  Provider: {config.provider}")
    print(f"  Model:    {config.model}")
    print(f"{'='*70}\n")


def get_checkpoint_filename(results_dir, config):
    """Generate config-specific checkpoint filename"""
    mode_str = f"{config.mode}_{config.provider}_{config.model.replace(':', '_').replace('/', '_')}"
    checkpoint_file = results_dir / "json" / f"checkpoint_{mode_str}.json"
    return checkpoint_file


def save_checkpoint(results_dir, config, results):
    """Save checkpoint with configuration metadata"""
    checkpoint_file = get_checkpoint_filename(results_dir, config)
    
    checkpoint_data = {
        'config': {
            'mode': config.mode,
            'provider': config.provider,
            'model': config.model,
            'use_case': config.use_case,
            'input_csv': config.input_csv
        },
        'results': results,
        'last_updated': datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2, default=str)


def check_for_checkpoint(results_dir, config):
    """Check for existing checkpoint and return last processed row"""
    checkpoint_file = get_checkpoint_filename(results_dir, config)
    
    if not checkpoint_file.exists():
        return None, []
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Validate checkpoint matches current config
        saved_config = checkpoint_data.get('config', {})
        if (saved_config.get('mode') != config.mode or 
            saved_config.get('provider') != config.provider or 
            saved_config.get('model') != config.model):
            print(f"\n{'='*70}")
            print("  ⚠️  CHECKPOINT CONFIG MISMATCH")
            print(f"{'='*70}")
            print(f"Checkpoint config: {saved_config.get('mode')}/{saved_config.get('provider')}/{saved_config.get('model')}")
            print(f"Current config:    {config.mode}/{config.provider}/{config.model}")
            print(f"Starting fresh with current config...")
            print(f"{'='*70}\n")
            return None, []
        
        existing_results = checkpoint_data.get('results', [])
        
        if not existing_results:
            return None, []
        
        # Find the last successfully processed row
        successful_rows = [r['row_index'] for r in existing_results if r.get('status') == 'success']
        if not successful_rows:
            return None, []
        
        last_row = max(successful_rows)
        
        print(f"\n{'='*70}")
        print("  ✓ CHECKPOINT FOUND")
        print(f"{'='*70}")
        print(f"Config: {config.mode}/{config.provider}/{config.model}")
        print(f"📂 Found existing results: {len(existing_results)} rows processed")
        print(f"✓  Last successful row: {last_row}")
        print(f"▶️  Will resume from row: {last_row + 1}")
        print(f"{'='*70}\n")
        
        return last_row, existing_results
    
    except Exception as e:
        print(f"⚠️  Warning: Could not read checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def main():
    """Main execution function"""
    # Get user configuration
    config = get_user_configuration()
    
    # Setup directories
    results_dir = setup_directories()
    
    # Check for checkpoint (unless force restart)
    last_processed_row, existing_results = None, []
    
    if config.force_restart:
        print(f"\n{'='*70}")
        print("  FORCE RESTART")
        print(f"{'='*70}")
        print("🔄 Starting from beginning (--force-restart flag set)")
        print(f"{'='*70}\n")
        # Delete existing checkpoint for this config if it exists
        checkpoint_file = get_checkpoint_filename(results_dir, config)
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"🗑️  Deleted existing checkpoint: {checkpoint_file.name}\n")
    else:
        last_processed_row, existing_results = check_for_checkpoint(results_dir, config)
        
        # Update start_row if resuming from checkpoint
        if last_processed_row is not None:
            resume_from = last_processed_row + 1
            print(f"🔄 RESUMING FROM CHECKPOINT")
            print(f"   Original start_row: {config.start_row}")
            print(f"   Resuming from row:  {resume_from}\n")
            config.start_row = resume_from
    
    # Load CSV data
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / config.input_csv
    
    if not csv_path.exists():
        print(f"❌ ERROR: CSV file not found at {csv_path}")
        print(f"   Available CSV files in data folder:")
        data_dir = base_dir / "data"
        if data_dir.exists():
            for csv_file in data_dir.glob("*.csv"):
                print(f"   - {csv_file.name}")
        return
    
    data_to_process, original_df, bom_type = load_csv_data(str(csv_path), start_row=config.start_row)
    
    # Apply limit if specified
    if config.limit:
        data_to_process = data_to_process.head(config.limit)
        print(f"⚠️  Limiting processing to first {config.limit} rows (for testing)")
    
    # Initialize processors
    print(f"\nInitializing processors...")
    print(f"  Mode: {config.mode}")
    print(f"  Provider: {config.provider}")
    print(f"  Model: {config.model}\n")
    
    ai_processor = AIBOMProcessor(
        model=config.model,
        mode=config.mode,
        llm_provider=config.provider,
        ollama_base_url=config.ollama_url,
        use_case=config.use_case,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model
    )
    
    data_processor = DATABOMProcessor(
        model=config.model,
        mode=config.mode,
        llm_provider=config.provider,
        ollama_base_url=config.ollama_url,
        use_case=config.use_case,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model
    )
    
    print("\n✓ Processors initialized successfully\n")
    
    # Process each row
    # Start with existing results if resuming from checkpoint
    all_results = existing_results if existing_results else []
    
    if existing_results:
        print(f"📋 Loaded {len(existing_results)} existing results from checkpoint\n")
    
    for idx, row in data_to_process.iterrows():
        result = process_row(row, idx, ai_processor, data_processor, config, bom_type)
        if result:
            all_results.append(result)
            
            # Save checkpoint after each row (config-specific, in case of crashes)
            save_checkpoint(results_dir, config, all_results)
        
        print(f"\n{'='*70}")
        print(f"Progress: {idx + 1}/{len(data_to_process)} rows processed")
        print(f"Total successful: {len([r for r in all_results if r['status'] == 'success'])}")
        print(f"Checkpoint saved: {get_checkpoint_filename(results_dir, config).name}")
        print(f"{'='*70}\n")
    
    # Save final results
    if all_results:
        json_file, csv_file = save_results(all_results, results_dir, config)
        
        # Update original CSV with links - DISABLED for golden-set (keep source CSV clean)
        # update_input_csv_with_links(original_df, all_results, str(csv_path))
        
        # Print summary
        print_processing_summary(all_results, config)
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"Results saved in: {results_dir}")
        
        # Clean up checkpoint after successful completion
        checkpoint_file = get_checkpoint_filename(results_dir, config)
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            print(f"✓ Checkpoint cleaned up: {checkpoint_file.name}")
    else:
        print(f"\n⚠️  No results to save")


if __name__ == '__main__':
    main()
