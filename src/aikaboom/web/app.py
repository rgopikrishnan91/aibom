"""
Unified Web Interface for BOM Generator
Supports both AI Model BOM and Data BOM generation
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import os
import io
import json
import logging
import queue
import threading
from dotenv import load_dotenv

from aikaboom.utils.link_fallback import LinkFallbackFinder

# Load environment variables
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    load_dotenv(_env_path)
else:
    load_dotenv()

# Use-case presets + question filtering live in a shared helper so the CLI
# and the web UI use the same source of truth.
from aikaboom.utils.use_case import (  # noqa: E402
    USE_CASE_PRESETS_AI,
    USE_CASE_PRESETS_DATA,
    filter_questions_by_use_case,
    get_use_case_label,
    normalize_use_case,
)


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

# ---------- Live log streaming ----------
# Thread-safe queue; the SSE endpoint drains it, /process pushes to it.
_log_subscribers = []  # list of queue.Queue, one per SSE listener
_log_lock = threading.Lock()


class _QueueLogHandler(logging.Handler):
    """Forward log records to all connected SSE subscribers."""
    def emit(self, record):
        msg = self.format(record)
        with _log_lock:
            for q in _log_subscribers:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass


class _PrintCapture(io.TextIOBase):
    """Intercept print() calls and mirror them to the log queue."""
    def __init__(self, original):
        self._original = original

    def write(self, text):
        self._original.write(text)
        if text.strip():
            with _log_lock:
                for q in _log_subscribers:
                    try:
                        q.put_nowait(text.rstrip())
                    except queue.Full:
                        pass
        return len(text)

    def flush(self):
        self._original.flush()


import sys
sys.stdout = _PrintCapture(sys.stdout)


def _extract_conflicts(metadata: dict) -> list:
    """Walk the BOM metadata and return a list of human-readable conflict dicts."""
    conflicts = []
    for section in ('direct_fields', 'rag_fields'):
        fields = metadata.get(section, {})
        if not isinstance(fields, dict):
            continue
        for field, triplet in fields.items():
            if not isinstance(triplet, dict):
                continue
            c = triplet.get('conflict')
            if c and isinstance(c, dict):
                trace = triplet.get('trace') or {}
                conflicts.append({
                    'field': field,
                    'section': section.replace('_fields', ''),
                    'chosen_value': triplet.get('value'),
                    'chosen_source': triplet.get('source'),
                    'conflict_value': c.get('value'),
                    'conflict_source': c.get('source'),
                    'conflict_type': c.get('type'),
                    # Phase 4: full per-source claim trace; renderer uses
                    # this to show every source's claim alongside the
                    # legacy single-conflict summary.
                    'claims': trace.get('claims') or {},
                })
    return conflicts


@app.route('/logs')
def stream_logs():
    """SSE endpoint — streams server logs to the browser in real time."""
    q = queue.Queue(maxsize=500)
    with _log_lock:
        _log_subscribers.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _log_lock:
                _log_subscribers.remove(q)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def get_processor(bom_type: str, mode: str = "rag", llm_provider: str = "openai", 
                  model: str = "gpt-4o", ollama_model: str = "llama3:70b",
                  openrouter_model: str = "qwen/qwen-2.5-72b-instruct",
                  ollama_url: str = None, use_case: str = 'complete'):
    """Get or create a processor for the specified configuration"""
    from aikaboom.core.processors import AIBOMProcessor, DATABOMProcessor
    from aikaboom.core.agentic_rag import get_fixed_questions

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
        # Filter the question set per the use-case preset (shared helper —
        # CLI uses the same one).
        questions = filter_questions_by_use_case(
            normalized_use_case, bom_type, get_fixed_questions(bom_type),
        )

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


@app.route('/models', methods=['GET'])
def list_models_endpoint():
    """Return a model catalog for the requested provider.

    Query params:
        provider: 'openrouter' (default). Other providers return [] for now.
        free_only: 'true' to only return free models.
        force_refresh: 'true' to bypass the in-memory cache.
    """
    provider = request.args.get('provider', 'openrouter').lower()
    free_only = request.args.get('free_only', 'false').lower() == 'true'
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'

    if provider != 'openrouter':
        return jsonify({'provider': provider, 'models': []})

    from aikaboom.utils.openrouter_models import (
        list_free_openrouter_models,
        list_openrouter_models,
    )
    fn = list_free_openrouter_models if free_only else list_openrouter_models
    try:
        models = fn(force_refresh=force_refresh)
        return jsonify({'provider': provider, 'free_only': free_only, 'models': models})
    except Exception as exc:
        return jsonify({'provider': provider, 'models': [], 'error': str(exc)}), 500


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
        validate_spdx = data.get('validate_spdx', True) is not False
        strict_spdx_validation = bool(data.get('strict_spdx_validation', False))
        recursive_bom = bool(data.get('recursive_bom', False))
        recursive_safety_cap = max(1, int(data.get('recursive_safety_cap', 50)))
        raw_depth = data.get('recursive_depth', 1)
        if isinstance(raw_depth, str) and raw_depth.strip().lower() in ('all', 'exhaust'):
            from aikaboom.utils.recursive_bom import EXHAUST_DEPTH
            recursive_depth = EXHAUST_DEPTH
        else:
            try:
                # Integer mode keeps the existing 5-level UI cap.
                recursive_depth = max(0, min(int(raw_depth), 5))
            except (TypeError, ValueError):
                recursive_depth = 1
        
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
                # beta-feature labels populated below as features run
                'beta_fields': [],
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
                'conflicts': _extract_conflicts(metadata),
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
                # beta-feature labels populated below as features run
                'beta_fields': [],
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
                'conflicts': _extract_conflicts(metadata),
            }

        # Always generate SPDX 3.0.1 output — it's the headline value prop.
        # If conversion fails, log it but don't break the rest of the response.
        try:
            from aikaboom.utils.spdx_validator import SPDXValidator, validate_spdx_export
            validator = SPDXValidator(bom_type=bom_type)
            spdx_output = validator.validate_and_convert(metadata)
            spdx_filename = filename.replace('.json', '.spdx.json')
            spdx_path = os.path.join(app.config['UPLOAD_FOLDER'], spdx_filename)
            with open(spdx_path, 'w', encoding='utf-8') as f:
                json.dump(spdx_output, f, indent=2, ensure_ascii=False)
            response_data['spdx_download_url'] = f'/download/{spdx_filename}'
            response_data['spdx_data'] = spdx_output
            if validate_spdx:
                response_data['spdx_validation'] = validate_spdx_export(
                    spdx_output,
                    strict=strict_spdx_validation,
                    bom_type=bom_type,
                )
            else:
                response_data['spdx_validation'] = {
                    'valid': None,
                    'strict': strict_spdx_validation,
                    'beta': strict_spdx_validation,
                    'validator': 'disabled',
                    'errors': [],
                }
        except Exception as spdx_exc:
            import traceback
            print(f"⚠️ SPDX conversion failed: {spdx_exc}")
            print(traceback.format_exc())
            response_data['spdx_error'] = str(spdx_exc)
            response_data['spdx_validation'] = {
                'valid': False,
                'strict': strict_spdx_validation,
                'beta': strict_spdx_validation,
                'validator': 'jsonschema+shacl' if strict_spdx_validation else 'jsonschema',
                'errors': [str(spdx_exc)],
            }

        # Always generate CycloneDX 1.7 beta output
        try:
            from aikaboom.utils.cyclonedx_exporter import CycloneDXExporter
            cdx_exporter = CycloneDXExporter(bom_type=bom_type)
            cdx_output = cdx_exporter.validate_and_convert(metadata)
            cdx_filename = filename.replace('.json', '.cyclonedx.json')
            cdx_path = os.path.join(app.config['UPLOAD_FOLDER'], cdx_filename)
            with open(cdx_path, 'w', encoding='utf-8') as f:
                json.dump(cdx_output, f, indent=2, ensure_ascii=False)
            response_data['cyclonedx_download_url'] = f'/download/{cdx_filename}'
            response_data['cyclonedx_data'] = cdx_output
            response_data['cyclonedx_beta'] = True
            response_data.setdefault('beta_fields', []).append('cyclonedx')

            # Authoritative validation via sbom-utility if available
            from aikaboom.utils.cyclonedx_validator import validate_cyclonedx
            response_data['cyclonedx_validation'] = validate_cyclonedx(cdx_path)
        except Exception as cdx_exc:
            import traceback
            print(f"⚠️ CycloneDX conversion failed: {cdx_exc}")
            print(traceback.format_exc())
            response_data['cyclonedx_error'] = str(cdx_exc)
            response_data['cyclonedx_beta'] = True

        if recursive_bom:
            try:
                from aikaboom.utils.recursive_bom import (
                    build_linked_spdx_bundle,
                    generate_recursive_boms,
                    linked_bundle_summary,
                )
                from aikaboom.utils.recursive_enrich import build_enrich_fn

                enrich_fn = build_enrich_fn(
                    use_case=use_case,
                    mode=mode,
                    llm_provider=llm_provider,
                    model=model,
                )
                recursive_output = generate_recursive_boms(
                    metadata,
                    bom_type=bom_type,
                    max_depth=recursive_depth,
                    safety_cap=recursive_safety_cap,
                    validate_spdx=validate_spdx,
                    strict_spdx=strict_spdx_validation,
                    enrich_fn=enrich_fn,
                )
                recursive_filename = filename.replace('.json', '.recursive.json')
                recursive_path = os.path.join(app.config['UPLOAD_FOLDER'], recursive_filename)
                with open(recursive_path, 'w', encoding='utf-8') as f:
                    json.dump(recursive_output, f, indent=2, ensure_ascii=False)
                response_data['recursive_bom'] = recursive_output
                response_data['recursive_bom_download_url'] = f'/download/{recursive_filename}'
                response_data.setdefault('beta_fields', []).append('recursive_bom')

                try:
                    linked_bundle = build_linked_spdx_bundle(
                        metadata, recursive_output, bom_type=bom_type
                    )
                    linked_filename = filename.replace('.json', '.linked.spdx.json')
                    linked_path = os.path.join(app.config['UPLOAD_FOLDER'], linked_filename)
                    with open(linked_path, 'w', encoding='utf-8') as f:
                        json.dump(linked_bundle, f, indent=2, ensure_ascii=False)
                    summary = linked_bundle_summary(linked_bundle, recursive_output)
                    if validate_spdx:
                        summary['validation'] = validate_spdx_export(
                            linked_bundle,
                            strict=strict_spdx_validation,
                            bom_type=bom_type,
                        )
                    response_data['linked_bom'] = summary
                    response_data['linked_bom_download_url'] = f'/download/{linked_filename}'
                    response_data.setdefault('beta_fields', []).append('linked_spdx_bundle')
                except Exception as linked_exc:
                    response_data['linked_bom'] = {'error': str(linked_exc), 'beta': True}
            except Exception as recursive_exc:
                import traceback
                print(f"⚠️ Recursive BOM beta generation failed: {recursive_exc}")
                print(traceback.format_exc())
                response_data['recursive_bom'] = {
                    'beta': True,
                    'enabled': True,
                    'max_depth': recursive_depth,
                    'generated_count': 0,
                    'generated': [],
                    'errors': [str(recursive_exc)],
                }
        else:
            response_data['recursive_bom'] = {
                'beta': True,
                'enabled': False,
                'max_depth': recursive_depth,
                'generated_count': 0,
                'generated': [],
            }

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
