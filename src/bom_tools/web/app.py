"""
Unified Web Interface for BOM Generator
Supports both AI Model BOM and Data BOM generation
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from bom_tools.utils.link_fallback import LinkFallbackFinder
from bom_tools.core.processors import AIBOMProcessor, DATABOMProcessor
from bom_tools.core.agentic_rag import get_fixed_questions, FIXED_QUESTIONS_AI, FIXED_QUESTIONS_DATA

# Load environment variables
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()

# Use case presets for AI
USE_CASE_PRESETS_AI = {
    'complete': {
        'label': 'Complete AI BOM',
        'description': 'Process every inference-backed field plus direct metadata.',
        'fields': None
    },
    'safety': {
        'label': 'Safety & Compliance',
        'description': 'Focus on safety risks, standards, limitations, and lifecycle transparency.',
        'fields': ['standardCompliance', 'limitation', 'modelExplainability', 'informationAboutApplication', 'informationAboutTraining', 'modelDataPreprocessing', 'domain']
    },
    'security': {
        'label': 'Security & Risk',
        'description': 'Highlight security posture, sensitive data usage, and operational context.',
        'fields': ['safetyRiskAssessment', 'informationAboutApplication', 'domain', 'useSensitivePersonalInformation', 'autonomyType']
    },
    'lineage': {
        'label': 'Model Lineage',
        'description': 'Capture training provenance, preprocessing, and supporting context.',
        'fields': ['informationAboutTraining', 'modelDataPreprocessing', 'hyperparameter', 'metric', 'metricDecisionThreshold']
    },
    'license': {
        'label': 'License Compliance',
        'description': 'Surface licensing and compliance-adjacent information.',
        'fields': ['standardCompliance', 'license']
    }
}

# Use case presets for Data
USE_CASE_PRESETS_DATA = {
    'complete': {
        'label': 'Complete Data BOM',
        'description': 'Process every inference-backed field plus direct metadata.',
        'fields': None
    },
    'safety': {
        'label': 'Safety & Bias',
        'description': 'Focus on bias, noise, and data quality aspects.',
        'fields': ['knownBias', 'datasetNoise', 'datasetUpdateMechanism', 'dataPreprocessing', 'dataCollectionProcess', 'intendedUse']
    },
    'security': {
        'label': 'Security & Privacy',
        'description': 'Focus on intended use, purpose, and anonymization.',
        'fields': ['intendedUse', 'primaryPurpose', 'anonymizationMethodUsed']
    },
    'lineage': {
        'label': 'Data Lineage',
        'description': 'Capture data origin, collection, and preprocessing information.',
        'fields': ['datasetAvailability', 'dataPreprocessing', 'dataCollectionProcess', 'releaseTime', 'originatedBy']
    },
    'license': {
        'label': 'License & Rights',
        'description': 'Focus on licensing and usage rights information.',
        'fields': ['license']
    }
}


def normalize_use_case(use_case: str, bom_type: str = 'ai') -> str:
    """Normalize use case string to a valid preset key"""
    key = (use_case or 'complete').strip().lower()
    presets = USE_CASE_PRESETS_AI if bom_type == 'ai' else USE_CASE_PRESETS_DATA
    return key if key in presets else 'complete'


def get_use_case_label(use_case: str, bom_type: str = 'ai') -> str:
    """Get the display label for a use case"""
    presets = USE_CASE_PRESETS_AI if bom_type == 'ai' else USE_CASE_PRESETS_DATA
    return presets.get(use_case, presets['complete'])['label']


def count_fields(metadata_dict: dict) -> int:
    """Count non-null fields in a metadata dictionary (triplet structure)"""
    if not metadata_dict:
        return 0
    count = 0
    for key, value in metadata_dict.items():
        # Handle triplet structure {"value": ..., "source": ..., "conflict": ...}
        if isinstance(value, dict) and 'value' in value:
            if value['value'] is not None and value['value'] != '':
                count += 1
        # Handle direct values
        elif value is not None and value != '':
            count += 1
    return count


app = Flask(__name__)
# Use absolute paths so Flask can always find the files regardless of CWD
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
app.config['UPLOAD_FOLDER'] = os.path.join(_PROJECT_ROOT, 'results')
app.config['REPO_RESULTS_FOLDER'] = os.path.join(_PROJECT_ROOT, 'data', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPO_RESULTS_FOLDER'], exist_ok=True)

# Initialize processors cache
processors_cache = {}


def get_processor(bom_type: str, mode: str = "rag", llm_provider: str = "openai", 
                  model: str = "gpt-4o", ollama_model: str = "llama3:70b",
                  openrouter_model: str = "qwen/qwen-2.5-72b-instruct",
                  ollama_url: str = None, use_case: str = 'complete'):
    """Get or create a processor for the specified configuration"""
    global processors_cache
    
    # Normalize inputs
    bom_type = bom_type.lower()
    normalized_mode = mode if mode in ['rag', 'direct'] else 'rag'
    normalized_use_case = normalize_use_case(use_case, bom_type)
    
    # Determine which model to use based on provider
    if llm_provider == 'ollama':
        model_to_use = ollama_model
    elif llm_provider == 'openrouter':
        model_to_use = openrouter_model
    else:
        model_to_use = model
    
    # Create a unique key for this configuration
    cache_key = f"{bom_type}_{llm_provider}_{normalized_mode}_{model_to_use}_{normalized_use_case}"
    
    if cache_key not in processors_cache:
        # Get questions for use case
        questions = get_fixed_questions(bom_type)
        if normalized_use_case != 'complete':
            presets = USE_CASE_PRESETS_AI if bom_type == 'ai' else USE_CASE_PRESETS_DATA
            preset_config = presets.get(normalized_use_case, {})
            requested_fields = preset_config.get('fields', [])
            if requested_fields:
                questions = {k: v for k, v in questions.items() if k in requested_fields}
        
        if bom_type == 'ai':
            processors_cache[cache_key] = AIBOMProcessor(
                model=model_to_use,
                mode=normalized_mode,
                llm_provider=llm_provider,
                ollama_base_url=ollama_url,
                questions_config=questions,
                use_case=normalized_use_case,
                embedding_provider="local",  # Use free local embeddings
                embedding_model="BAAI/bge-small-en-v1.5"
            )
        else:
            processors_cache[cache_key] = DATABOMProcessor(
                model=model_to_use,
                mode=normalized_mode,
                llm_provider=llm_provider,
                ollama_base_url=ollama_url,
                questions_config=questions,
                use_case=normalized_use_case,
                embedding_provider="local",  # Use free local embeddings
                embedding_model="BAAI/bge-small-en-v1.5"
            )
    
    return processors_cache[cache_key]


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/find_links', methods=['POST'])
def find_links():
    """Run link fallback and return found links immediately, before full BOM generation."""
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
        bom_type = data.get('bom_type', 'ai').strip().lower()
        repo_id = data.get('repo_id', '').strip() or None
        hf_repo_id = data.get('hf_repo_id', '').strip() or repo_id or None
        arxiv_url = data.get('arxiv_url', '').strip() or None
        github_url = data.get('github_url', '').strip() or None

        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if not gemini_api_key:
            return jsonify({
                'status': 'skipped',
                'reason': 'no_gemini_key',
                'message': 'Link Fallback Agent inactive — no GEMINI_API_KEY set',
                'hf_repo_id': hf_repo_id,
                'arxiv_url': arxiv_url,
                'github_url': github_url,
                'link_status': {}
            })

        # Verify google-genai is installed (it's a hard dep now, but fall back gracefully)
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return jsonify({
                'status': 'skipped',
                'reason': 'genai_not_installed',
                'message': 'Link Fallback Agent unavailable — google-genai package not installed. Run: pip install google-genai',
                'hf_repo_id': hf_repo_id,
                'arxiv_url': arxiv_url,
                'github_url': github_url,
                'link_status': {}
            })

        # Compute which links were missing before the call, so we can tell
        # the UI both what was found and what we tried but couldn't find.
        missing_before = {
            'hf': not hf_repo_id,
            'arxiv': not arxiv_url,
            'github': not github_url,
        }

        fallback_finder = LinkFallbackFinder()
        final_repo_id, final_arxiv_url, final_github_url, link_status = fallback_finder.find_missing_links(
            repo_id=hf_repo_id,
            hf_repo_id=hf_repo_id,
            arxiv_url=arxiv_url,
            github_url=github_url
        )
        link_status = link_status or {}
        # For each link that was missing, mark whether the agent recovered it.
        not_found = []
        if missing_before['hf'] and not final_repo_id:
            not_found.append('huggingface')
        if missing_before['arxiv'] and not final_arxiv_url:
            not_found.append('arxiv')
        if missing_before['github'] and not final_github_url:
            not_found.append('github')

        return jsonify({
            'status': 'success',
            'hf_repo_id': final_repo_id,
            'arxiv_url': final_arxiv_url,
            'github_url': final_github_url,
            'link_status': link_status,
            'not_found': not_found,
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/config', methods=['GET'])
def get_config():
    """Return available configuration options"""
    bom_type = request.args.get('bom_type', 'ai').lower()
    
    use_cases_info = {}
    presets = USE_CASE_PRESETS_AI if bom_type == 'ai' else USE_CASE_PRESETS_DATA
    for key, preset in presets.items():
        use_cases_info[key] = {
            'label': preset['label'],
            'description': preset['description'],
            'has_fields': preset['fields'] is not None and len(preset['fields']) > 0
        }
    
    config = {
        'bom_types': ['ai', 'data'],
        'modes': ['rag', 'direct'],
        'llm_providers': ['openai', 'ollama', 'openrouter'],
        'default_openai_models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'default_ollama_models': ['llama3:70b', 'llama3:8b', 'mixtral:8x7b', 'codellama:34b'],
        'default_openrouter_models': ['qwen/qwen-2.5-72b-instruct', 'meta-llama/llama-3.3-70b-instruct', 'mistralai/mistral-medium-3.1', 'openai/gpt-oss-120b', 'qwen/qwen3-coder'],
        'default_mode': 'rag',
        'default_provider': 'openai',
        'default_model': 'gpt-4o',
        'use_cases': use_cases_info,
        'default_use_case': 'complete'
    }
    return jsonify(config)


@app.route('/process', methods=['POST'])
def process():
    """Handle processing request for both AI and Data BOMs"""
    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
        bom_type = data.get('bom_type', 'ai').strip().lower()
        
        # Get processing options
        mode = data.get('mode', 'rag').strip().lower()
        llm_provider = data.get('llm_provider', 'openrouter').strip().lower()
        model = data.get('model', 'gpt-4o').strip()
        ollama_model = data.get('ollama_model', 'llama3:70b').strip()
        openrouter_model = data.get('openrouter_model', 'qwen/qwen-2.5-72b-instruct').strip() or 'qwen/qwen-2.5-72b-instruct'
        ollama_url = data.get('ollama_url', '').strip() or None
        use_case = normalize_use_case(data.get('use_case', 'complete'), bom_type)
        
        # Validate mode and provider
        if mode not in ['rag', 'direct']:
            mode = 'rag'
        if llm_provider not in ['openai', 'ollama', 'openrouter']:
            llm_provider = 'openai'
        
        if bom_type == 'ai':
            # AI BOM processing
            repo_id = data.get('repo_id', '').strip() or None
            arxiv_url = data.get('arxiv_url', '').strip() or None
            github_url = data.get('github_url', '').strip() or None
            
            if not any([repo_id, arxiv_url, github_url]):
                return jsonify({
                    'status': 'error',
                    'message': 'Provide at least a repo ID or one URL!'
                }), 400
            
            # Use link fallback to find missing links (skip if frontend already ran /find_links)
            skip_fallback = data.get('skip_fallback', False)
            link_fallback_info = {'attempted': False}
            if not skip_fallback:
                print("\n" + "="*70)
                print("LINK FALLBACK: Checking for missing links...")
                print("="*70)
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    link_fallback_info = {
                        'attempted': False,
                        'reason': 'no_gemini_key',
                        'message': 'Link Fallback Agent inactive — no GEMINI_API_KEY set',
                    }
                else:
                    missing_before = {
                        'hf': not repo_id,
                        'arxiv': not arxiv_url,
                        'github': not github_url,
                    }
                    try:
                        fallback_finder = LinkFallbackFinder()
                        final_repo_id, final_arxiv_url, final_github_url, link_status = fallback_finder.find_missing_links(
                            repo_id=repo_id,
                            hf_repo_id=repo_id,
                            arxiv_url=arxiv_url,
                            github_url=github_url
                        )
                        # Always apply discovered links — overwrite missing fields if found.
                        if missing_before['hf'] and final_repo_id:
                            repo_id = final_repo_id
                        if missing_before['arxiv'] and final_arxiv_url:
                            arxiv_url = final_arxiv_url
                        if missing_before['github'] and final_github_url:
                            github_url = final_github_url
                        not_found = []
                        if missing_before['hf'] and not final_repo_id:
                            not_found.append('huggingface')
                        if missing_before['arxiv'] and not final_arxiv_url:
                            not_found.append('arxiv')
                        if missing_before['github'] and not final_github_url:
                            not_found.append('github')
                        link_fallback_info = {
                            'attempted': True,
                            'link_status': link_status or {},
                            'not_found': not_found,
                        }
                    except Exception as e:
                        print(f"⚠️ Link fallback failed: {e}")
                        link_fallback_info = {
                            'attempted': True,
                            'error': str(e),
                        }
            
            proc = get_processor(
                bom_type='ai',
                mode=mode,
                llm_provider=llm_provider,
                model=model,
                ollama_model=ollama_model,
                openrouter_model=openrouter_model,
                ollama_url=ollama_url,
                use_case=use_case
            )
            
            metadata = proc.process_ai_model(
                repo_id=repo_id,
                arxiv_url=arxiv_url,
                github_url=github_url
            )
            
            # Save to file (download folder + persistent repo copy)
            filename = f"{metadata['model_id']}_aibom.json"
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            repo_copy = os.path.join(app.config['REPO_RESULTS_FOLDER'], filename)
            with open(repo_copy, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Count fields
            direct_count = count_fields(metadata.get('direct_fields', {}))
            rag_count = count_fields(metadata.get('rag_fields', {}))
            
            response_data = {
                'status': 'success',
                'message': 'AI model processed successfully!',
                'metadata': metadata,
                'bom_type': 'ai',
                'use_case': proc.use_case,
                'use_case_label': get_use_case_label(proc.use_case, 'ai'),
                'download_url': f'/download/{os.path.basename(output_file)}',
                'field_counts': {
                    'direct': direct_count,
                    'rag': rag_count,
                    'total': direct_count + rag_count
                },
                'found_links': {
                    'hf_repo_id': repo_id,
                    'arxiv_url': arxiv_url,
                    'github_url': github_url
                },
                'link_fallback': link_fallback_info,
            }

        else:
            # Data BOM processing
            arxiv_url = data.get('arxiv_url', '').strip() or None
            github_url = data.get('github_url', '').strip() or None
            hf_repo_id = data.get('hf_repo_id', '').strip() or None
            
            if not any([arxiv_url, github_url, hf_repo_id]):
                return jsonify({
                    'status': 'error',
                    'message': 'At least one repo ID or URL must be provided!'
                }), 400
            
            # Construct HuggingFace URL from repo_id if provided
            hf_url = None
            if hf_repo_id:
                hf_url = f"https://huggingface.co/datasets/{hf_repo_id}"
                print(f"Constructed HuggingFace dataset URL: {hf_url}")
            
            # Try link fallback (skip if frontend already ran /find_links)
            skip_fallback = data.get('skip_fallback', False)
            link_fallback_info = {'attempted': False}
            if not skip_fallback:
                gemini_api_key = os.getenv("GEMINI_API_KEY")
                if not gemini_api_key:
                    link_fallback_info = {
                        'attempted': False,
                        'reason': 'no_gemini_key',
                        'message': 'Link Fallback Agent inactive — no GEMINI_API_KEY set',
                    }
                else:
                    missing_before = {
                        'hf': not hf_repo_id,
                        'arxiv': not arxiv_url,
                        'github': not github_url,
                    }
                    try:
                        fallback_finder = LinkFallbackFinder()
                        final_repo_id, final_arxiv_url, final_github_url, link_status = fallback_finder.find_missing_links(
                            repo_id=hf_repo_id,
                            hf_repo_id=hf_repo_id,
                            arxiv_url=arxiv_url,
                            github_url=github_url
                        )
                        if missing_before['hf'] and final_repo_id and not hf_url:
                            hf_url = f"https://huggingface.co/datasets/{final_repo_id}"
                            hf_repo_id = final_repo_id
                        if missing_before['arxiv'] and final_arxiv_url:
                            arxiv_url = final_arxiv_url
                        if missing_before['github'] and final_github_url:
                            github_url = final_github_url
                        not_found = []
                        if missing_before['hf'] and not final_repo_id:
                            not_found.append('huggingface')
                        if missing_before['arxiv'] and not final_arxiv_url:
                            not_found.append('arxiv')
                        if missing_before['github'] and not final_github_url:
                            not_found.append('github')
                        link_fallback_info = {
                            'attempted': True,
                            'link_status': link_status or {},
                            'not_found': not_found,
                        }
                    except Exception as e:
                        print(f"⚠️ Link fallback failed: {e}")
                        link_fallback_info = {
                            'attempted': True,
                            'error': str(e),
                        }
            
            proc = get_processor(
                bom_type='data',
                mode=mode,
                llm_provider=llm_provider,
                model=model,
                ollama_model=ollama_model,
                openrouter_model=openrouter_model,
                ollama_url=ollama_url,
                use_case=use_case
            )
            
            metadata = proc.process_dataset(
                arxiv_url=arxiv_url,
                github_url=github_url,
                hf_url=hf_url
            )
            
            # Save to file (download folder + persistent repo copy)
            filename = f"{metadata['dataset_id']}_databom.json"
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            repo_copy = os.path.join(app.config['REPO_RESULTS_FOLDER'], filename)
            with open(repo_copy, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Count fields
            direct_count = count_fields(metadata.get('direct_fields', {}))
            rag_count = count_fields(metadata.get('rag_fields', {}))
            
            response_data = {
                'status': 'success',
                'message': 'Dataset processed successfully!',
                'metadata': metadata,
                'bom_type': 'data',
                'use_case': proc.use_case,
                'use_case_label': get_use_case_label(proc.use_case, 'data'),
                'download_url': f'/download/{os.path.basename(output_file)}',
                'field_counts': {
                    'direct': direct_count,
                    'rag': rag_count,
                    'total': direct_count + rag_count
                },
                'found_links': {
                    'hf_repo_id': hf_repo_id,
                    'arxiv_url': arxiv_url,
                    'github_url': github_url
                },
                'link_fallback': link_fallback_info,
            }

        # Always generate SPDX 3.0.1 output — it's the headline value prop.
        # If conversion fails, log it but don't break the rest of the response.
        try:
            from bom_tools.utils.spdx_validator import SPDXValidator
            validator = SPDXValidator(bom_type=bom_type)
            spdx_output = validator.validate_and_convert(metadata)
            spdx_filename = filename.replace('.json', '.spdx.json')
            spdx_path = os.path.join(app.config['UPLOAD_FOLDER'], spdx_filename)
            with open(spdx_path, 'w', encoding='utf-8') as f:
                json.dump(spdx_output, f, indent=2, ensure_ascii=False)
            response_data['spdx_download_url'] = f'/download/{spdx_filename}'
            response_data['spdx_data'] = spdx_output
        except Exception as spdx_exc:
            import traceback
            print(f"⚠️ SPDX conversion failed: {spdx_exc}")
            print(traceback.format_exc())
            response_data['spdx_error'] = str(spdx_exc)

        return jsonify(response_data)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(error_trace)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'trace': error_trace
        }), 500


@app.route('/download/<filename>')
def download(filename):
    """Download the generated JSON file"""
    try:
        safe_name = secure_filename(filename)
        if not safe_name:
            return jsonify({'status': 'error', 'message': 'Invalid filename'}), 400
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
        return send_file(file_path, as_attachment=True, download_name=safe_name)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 404


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 Unified BOM Generator - WEB INTERFACE")
    print("="*70)
    print("\nServer starting...")
    print("Access the web interface at: http://localhost:5000")
    print("Or from another machine: http://<server-ip>:5000")
    print("\n⚠️  To stop the server, press Ctrl+C")
    print("="*70 + "\n")
    
    # Run Flask app
    host = os.getenv('BOM_HOST', '127.0.0.1')
    port = int(os.getenv('BOM_PORT', '5000'))
    app.run(host=host, port=port, debug=False, threaded=True)


def main():
    """Entry point for console script"""
    print("\n" + "="*70)
    print("🌐 Unified BOM Generator - WEB INTERFACE")
    print("="*70)
    print("\nServer starting...")
    print("Access the web interface at: http://localhost:5000")
    print("Or from another machine: http://<server-ip>:5000")
    print("\n⚠️  To stop the server, press Ctrl+C")
    print("="*70 + "\n")
    
    # Run Flask app
    host = os.getenv('BOM_HOST', '127.0.0.1')
    port = int(os.getenv('BOM_PORT', '5000'))
    app.run(host=host, port=port, debug=False, threaded=True)
