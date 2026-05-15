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
app.config['HISTORY_FOLDER'] = os.path.join(_PROJECT_ROOT, 'bom-history')
app.config['HISTORY_INDEX'] = os.path.join(app.config['HISTORY_FOLDER'], 'index.json')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPO_RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['HISTORY_FOLDER'], exist_ok=True)

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

# Turn on structured pipeline-event emission (the [[BOM_EVENT]] lines the
# frontend parses out of the log stream to drive the Pipeline tab). Off by
# default so CLI users don't see the noise. Shared by agentic_rag and the
# recursive walker — both emit through aikaboom.utils.pipeline_events.
from aikaboom.utils import pipeline_events as _pipeline_events  # noqa: E402
_pipeline_events.EMIT_EVENTS = True


# ---------- BOM history ledger ----------
# Every successful /process run drops its artifacts under bom-history/{hash}
# and appends a row to bom-history/index.json so the History tab can show
# past runs and re-load any of them back into the viewer tabs without
# re-generating.

import hashlib  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

_history_lock = threading.Lock()


def _bom_hash(metadata: dict) -> str:
    """Stable short hash for a BOM. SHA-256 of canonical JSON, first 12 chars."""
    payload = json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:12]


def _history_load() -> list:
    path = app.config['HISTORY_INDEX']
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _history_save(rows: list) -> None:
    path = app.config['HISTORY_INDEX']
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _history_record(bom_type: str, metadata: dict, artifacts: dict) -> dict:
    """Persist a run's artifacts and append a row to the ledger.

    ``artifacts`` is a {label: source_filename_in_UPLOAD_FOLDER} mapping. The
    canonical BOM file is required ('bom'); SPDX, CycloneDX, recursive, and
    linked exports are optional. Files are copied into bom-history/ keyed by
    the stable BOM hash so the index can later serve them independently of
    UPLOAD_FOLDER (which gets overwritten on every re-run of the same subject).
    """
    import shutil
    hash_id = _bom_hash(metadata)
    subject = metadata.get('model_id') or metadata.get('dataset_id') or 'unknown'
    row = {
        'hash': hash_id,
        'subject': subject,
        'bom_type': bom_type,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'artifacts': {},
    }
    hist_dir = app.config['HISTORY_FOLDER']
    for label, src_filename in (artifacts or {}).items():
        if not src_filename:
            continue
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], src_filename)
        if not os.path.exists(src_path):
            continue
        ext = src_filename.split('_', 1)[-1] if '_' in src_filename else src_filename
        dst_name = f"{hash_id}_{ext}"
        dst_path = os.path.join(hist_dir, dst_name)
        try:
            shutil.copy2(src_path, dst_path)
            row['artifacts'][label] = dst_name
        except OSError as exc:
            print(f"⚠️ history copy failed for {label}: {exc}")
    with _history_lock:
        rows = _history_load()
        rows = [r for r in rows if r.get('hash') != hash_id]
        rows.insert(0, row)
        _history_save(rows)
    return row


def _conflict_summary(metadata: dict) -> dict:
    """Return the canonical conflict summary, recomputed live.

    BOMs cached on disk before the Phase-12 ``summarise_conflicts`` fix
    carry inflated counts in ``metadata['conflict_summary']`` (counted
    every triplet envelope, not just real verdicts). Recomputing on every
    response makes sure the headline matches the actual conflict list.
    """
    from aikaboom.core.conflict_routing import summarise_conflicts
    return summarise_conflicts(metadata)


def _extract_conflicts(metadata: dict) -> list:
    """Walk the BOM metadata and return a list of human-readable conflict dicts.

    Surfaces three classes of conflict:
    1. Direct/intra triplet conflicts (`triplet.conflict`) — deterministic;
       no LLM grounding score (``grounding`` is ``None``, ``confidence``
       is ``"n/a"``).
    2. RAG `trace.internal_conflicts` — auditor-LLM-detected
       contradictions WITHIN one source. Carries a Phase 12B grounding
       score (cosine vs the chunks); ``confidence`` lands as
       ``"high"`` / ``"low"`` per ``HIGH_CONFIDENCE_THRESHOLD``.
    3. RAG `trace.external_conflicts` — auditor-LLM-detected
       contradictions BETWEEN sources. Same grounding shape as (2).
    """
    from aikaboom.core.conflict_routing import HIGH_CONFIDENCE_THRESHOLD, _confidence_label
    from aikaboom import get_direct_priority, get_rag_priority

    bom_type = 'data' if 'dataset_id' in (metadata or {}) else 'ai'

    def _priority_for(field_name, section):
        if section == 'direct':
            return get_direct_priority(field_name) or []
        return get_rag_priority(field_name, bom_type) or []

    conflicts = []
    for section in ('direct_fields', 'rag_fields'):
        fields = metadata.get(section, {})
        if not isinstance(fields, dict):
            continue
        for field, triplet in fields.items():
            if not isinstance(triplet, dict):
                continue
            section_label = section.replace('_fields', '')
            trace = triplet.get('trace') or {}
            priority = _priority_for(field, section_label)
            field_class = 'direct' if section_label == 'direct' else 'inferred'

            # Every source that contributed a claim for this field — the
            # universe of evidence the auditor saw. ``trace.claims`` is
            # populated for every speaking source (silent sources are
            # omitted upstream), so its keys are the "sources considered"
            # display list. ``trace.selected_sources`` is the post-consensus
            # winners and shipped separately so the UI can mark which of
            # those considered sources survived the routing step.
            sources_considered = list((trace.get('claims') or {}).keys())
            selected_sources = trace.get('selected_sources') or sources_considered

            # 1. Direct-field conflict (deterministic). Only direct-field
            # triplets carry a ``conflict`` shaped as ``{value, source,
            # type}``. RAG triplets carry a ``conflict`` envelope shaped
            # as ``{internal: ..., external: ...}`` even when nothing
            # disagrees — those must NOT trigger a phantom row here. The
            # presence of a ``type`` key (only direct conflicts have one)
            # is the discriminator. Same fix class as the
            # ``summarise_conflicts`` Mistral-27 bug.
            c = triplet.get('conflict')
            if (
                isinstance(c, dict)
                and 'type' in c
                and 'value' in c
                and 'internal' not in c
            ):
                conflicts.append({
                    'field': field,
                    'section': section_label,
                    'field_class': field_class,
                    'priority': priority,
                    'kind': c.get('type') or 'inter',
                    'chosen_value': triplet.get('value'),
                    'chosen_source': triplet.get('source'),
                    'conflict_value': c.get('value'),
                    'conflict_source': c.get('source'),
                    'conflict_type': c.get('type'),
                    'grounding': None,
                    'confidence': 'n/a',
                    'narrative': None,
                    'claims': trace.get('claims') or {},
                    'sources': [c.get('source')] if c.get('source') else [],
                    'sources_considered': sources_considered,
                    'selected_sources': selected_sources,
                })

            # 2. RAG auditor — internal (intra-source) LLM-detected.
            for src, entry in (trace.get('internal_conflicts') or {}).items():
                if not isinstance(entry, dict):
                    entry = {'narrative': str(entry)}
                grounding = entry.get('grounding')
                conflicts.append({
                    'field': field,
                    'section': section_label,
                    'field_class': field_class,
                    'priority': priority,
                    'kind': 'rag-internal',
                    'chosen_value': triplet.get('value'),
                    'chosen_source': triplet.get('source'),
                    'conflict_value': None,
                    'conflict_source': src,
                    'conflict_type': 'rag-internal',
                    'grounding': grounding,
                    'confidence': _confidence_label(grounding),
                    'narrative': entry.get('narrative'),
                    'claims': trace.get('claims') or {},
                    'sources': [src],
                    'sources_considered': sources_considered,
                    'selected_sources': selected_sources,
                })

            # 3. RAG auditor — external (cross-source) LLM-detected.
            for entry in (trace.get('external_conflicts') or []):
                if not isinstance(entry, dict):
                    continue
                sources = entry.get('sources') or []
                grounding = entry.get('grounding')
                conflicts.append({
                    'field': field,
                    'section': section_label,
                    'field_class': field_class,
                    'priority': priority,
                    'kind': 'rag-external',
                    'chosen_value': triplet.get('value'),
                    'chosen_source': triplet.get('source'),
                    'conflict_value': None,
                    'conflict_source': ' vs '.join(sources) if sources else None,
                    'conflict_type': 'rag-external',
                    'grounding': grounding,
                    'confidence': _confidence_label(grounding),
                    'narrative': entry.get('description'),
                    'claims': trace.get('claims') or {},
                    'sources': list(sources),
                    'sources_considered': sources_considered,
                    'selected_sources': selected_sources,
                })
    return conflicts


def _open_graph_store():
    """Open the graph store for read routes, or None if unavailable/disabled."""
    if os.environ.get('AIKABOOM_GRAPH_DISABLE') == '1':
        return None
    try:
        from aikaboom.store.store import BomStore
        return BomStore.open()
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ graph store unavailable for worldofBOMs: {e}")
        return None


def _try_resolve_cache(
    bom_type: str,
    cache_policy_str: str,
    mode: str,
    use_case: str,
    planned_llm: str,
    identifiers_kwargs: dict,
):
    """worldofBOMs cache-resolution wrap for the /process route.

    Mirrors ``cli.cmd_generate``'s wrap but with ``interactive=False``:
    headless POSTs have no TTY, so the ``prompt`` policy degrades to
    ``use`` per ``cache_resolver.decide``'s contract.

    Returns a ``(store_or_none, identifiers, claim_iri_or_none)`` tuple.
    The route uses ``claim_iri`` to short-circuit to a cached BOM and
    ``store + identifiers`` to persist a freshly generated claim. When
    ``AIKABOOM_GRAPH_DISABLE=1`` is set or the backend fails to open,
    all three are returned in a "no cache, no persist" shape so the
    caller falls through cleanly.
    """
    if os.environ.get('AIKABOOM_GRAPH_DISABLE') == '1':
        return None, [], None

    try:
        from aikaboom.store.cache_resolver import CachePolicy, decide
        from aikaboom.store.naming import Identifier
        from aikaboom.store.store import BomStore
    except Exception as e:
        print(f"⚠️ graph store imports unavailable, skipping cache: {e}")
        return None, [], None

    try:
        store = BomStore.open()
    except Exception as e:
        print(f"⚠️ graph store unavailable, skipping cache: {e}")
        return None, [], None

    idents: list = []
    repo_id = identifiers_kwargs.get('repo_id')
    arxiv_url = identifiers_kwargs.get('arxiv_url')
    github_url = identifiers_kwargs.get('github_url')
    if repo_id:
        idents.append(Identifier('huggingface', repo_id))
    if arxiv_url:
        idents.append(Identifier('arxiv', arxiv_url))
    if github_url:
        idents.append(Identifier('github', github_url))

    if not idents:
        return store, [], None

    try:
        resolved = store.resolve(
            identifiers=idents,
            use_case=use_case,
            mode=mode,
        )
    except Exception as e:
        print(f"⚠️ cache resolve failed, generating: {e}")
        return store, idents, None

    try:
        policy = CachePolicy(cache_policy_str)
    except ValueError:
        policy = CachePolicy.USE

    # Server-side requests have no TTY for the prompt path to fire on;
    # ``decide`` degrades PROMPT to USE under ``interactive=False``.
    decision = decide(
        resolved, policy, interactive=False, planned_llm=planned_llm,
    )
    if decision == 'use' and resolved.matching_claims:
        return store, idents, resolved.matching_claims[0]['iri']
    return store, idents, None


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
    
    # Auto-detect the configured provider so the UI defaults to the one the
    # user actually has credentials for, instead of always offering OpenAI.
    from aikaboom.utils.provider_resolver import detect_default_provider
    detected_provider, detected_model = detect_default_provider()

    config = {
        'bom_types': ['ai', 'data'],
        'modes': ['rag', 'direct'],
        'llm_providers': ['openai', 'ollama', 'openrouter'],
        'default_openai_models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        'default_ollama_models': ['llama3:70b', 'llama3:8b', 'mixtral:8x7b', 'codellama:34b'],
        'default_openrouter_models': ['qwen/qwen-2.5-72b-instruct', 'meta-llama/llama-3.3-70b-instruct', 'mistralai/mistral-medium-3.1', 'openai/gpt-oss-120b', 'qwen/qwen3-coder'],
        'default_mode': 'rag',
        'default_provider': detected_provider or 'openai',
        'default_model': detected_model or 'gpt-4o',
        'use_cases': use_cases_info,
        'default_use_case': 'complete'
    }
    return jsonify(config)


@app.route('/models', methods=['GET'])
def list_models_endpoint():
    """Return a model catalog for the requested provider.

    Query params:
        provider: 'openrouter' (default). Other providers return [] for now.
        force_refresh: 'true' to bypass the in-memory cache.

    Phase 10 retired the ``free_only`` filter (Findings #6, #12). The
    free-tier path was a usability trap; the catalog now always returns
    the full list and users pick paid model ids explicitly.
    """
    provider = request.args.get('provider', 'openrouter').lower()
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'

    if provider != 'openrouter':
        return jsonify({'provider': provider, 'models': []})

    from aikaboom.utils.openrouter_models import list_openrouter_models
    try:
        models = list_openrouter_models(force_refresh=force_refresh)
        return jsonify({'provider': provider, 'models': models})
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
        # Phase 10 / Finding #22: optional cap on the standalone SPDX
        # relationship-children list. ``None`` (default, or empty form
        # input, or ``0``/negative) means unbounded — same behaviour as
        # the recursive walker and linked bundle.
        raw_cap = data.get('spdx_relationship_cap')
        try:
            spdx_relationship_cap = int(raw_cap) if raw_cap not in (None, '', 'unbounded') else None
        except (TypeError, ValueError):
            spdx_relationship_cap = None
        if spdx_relationship_cap is not None and spdx_relationship_cap < 1:
            spdx_relationship_cap = None
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

        # worldofBOMs cache policy. Default is ``use`` because headless POSTs
        # have no TTY for the prompt path to fire on; ``force_refresh: true``
        # remains supported as a shorthand for ``cache_policy=regen``.
        cache_policy_raw = (
            data.get('cache_policy')
            or ('regen' if data.get('force_refresh') else 'use')
        )
        cache_policy_str = str(cache_policy_raw).strip().lower() or 'use'

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

            # Be lenient: users routinely paste the full HF URL into the
            # repo_id field. Normalise so downstream code only ever sees the
            # canonical ``namespace/repo`` form. (The dataset BOM path
            # already does this in core/processors.py for hf_url; this
            # mirrors that behaviour at the AI entry point.)
            if repo_id and "huggingface.co" in repo_id:
                from aikaboom.utils.metadata_fetcher import MetadataFetcher
                normalised = MetadataFetcher.extract_repo_id_from_hf_url(repo_id)
                if normalised:
                    repo_id = normalised

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
            
            # worldofBOMs cache resolution (Task 16). Build identifiers from
            # the request and ask the BOM store whether we already have a
            # matching claim; on a "use" decision, reconstruct the cached
            # BOM and skip the processor call. ``AIKABOOM_GRAPH_DISABLE=1``
            # bypasses the entire wrap to keep headless deployments
            # lightweight, matching the CLI's bypass semantics.
            _store, _idents, _cache_claim_iri = _try_resolve_cache(
                bom_type='ai',
                cache_policy_str=cache_policy_str,
                mode=mode,
                use_case=use_case,
                planned_llm=model,
                identifiers_kwargs={
                    'repo_id': repo_id,
                    'arxiv_url': arxiv_url,
                    'github_url': github_url,
                },
            )

            if _cache_claim_iri is not None:
                # Reconstruct the cached BOM from the graph and record an
                # implicit-use vote. Mirrors cli.cmd_generate's cache-hit
                # path. ``rdf_to_bom`` only populates ``repo_id`` from the
                # graph; the AI branch also needs a ``model_id`` for the
                # output filename, so derive one if it's missing (same
                # rule as ``AIBOMProcessor.generate_model_id``).
                try:
                    from aikaboom.store.trust import VoteKind
                    metadata = _store.reconstruct_bom(_cache_claim_iri)
                    if not metadata.get('model_id'):
                        rid = metadata.get('repo_id') or repo_id or ''
                        metadata['model_id'] = (
                            rid.replace('/', '_').replace(' ', '_') or 'ai_model_cached'
                        )
                    metadata['_cached'] = True
                    metadata['_claim_iri'] = _cache_claim_iri
                    _store.record_trust_vote(_cache_claim_iri, VoteKind.IMPLICIT_USE)
                except Exception as e:
                    print(f"⚠️ cache reconstruct failed, regenerating: {e}")
                    _cache_claim_iri = None  # fall through to generation

            if _cache_claim_iri is None:
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

                # Best-effort persistence to the graph store. The LLM cost
                # has already been paid; never let a backend write failure
                # destroy the generated BOM before the JSON output is
                # written.
                if _store is not None and _idents:
                    try:
                        _store.save_claim(
                            metadata,
                            run_meta={
                                'provider': llm_provider, 'llm_model': model,
                                'prompt_version': 'v1', 'code_version': 'head',
                                'mode': mode, 'use_case': use_case,
                            },
                            identifiers=_idents,
                        )
                    except Exception as e:
                        print(f"⚠️ failed to persist claim to graph store: {e}")

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
            
            # Fall back to the normalized request use_case when the
            # processor was bypassed by a cache hit.
            effective_use_case = (
                getattr(proc, 'use_case', None)
                if _cache_claim_iri is None else (metadata.get('use_case') or use_case)
            ) or use_case

            response_data = {
                'status': 'success',
                'message': 'AI model processed successfully!',
                'metadata': metadata,
                'bom_type': 'ai',
                'use_case': effective_use_case,
                'use_case_label': get_use_case_label(effective_use_case, 'ai'),
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
                'conflict_summary': _conflict_summary(metadata),
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
            
            # worldofBOMs cache resolution (Task 16). See the AI branch
            # above for rationale; the data path uses ``hf_repo_id`` as the
            # huggingface identifier.
            _store, _idents, _cache_claim_iri = _try_resolve_cache(
                bom_type='data',
                cache_policy_str=cache_policy_str,
                mode=mode,
                use_case=use_case,
                planned_llm=model,
                identifiers_kwargs={
                    'repo_id': hf_repo_id,
                    'arxiv_url': arxiv_url,
                    'github_url': github_url,
                },
            )

            if _cache_claim_iri is not None:
                # See the AI branch above. The data branch uses
                # ``dataset_id`` instead of ``model_id`` for the output
                # filename.
                try:
                    from aikaboom.store.trust import VoteKind
                    metadata = _store.reconstruct_bom(_cache_claim_iri)
                    if not metadata.get('dataset_id'):
                        rid = metadata.get('repo_id') or hf_repo_id or ''
                        metadata['dataset_id'] = (
                            rid.replace('/', '_').replace(' ', '_') or 'dataset_cached'
                        )
                    metadata['_cached'] = True
                    metadata['_claim_iri'] = _cache_claim_iri
                    _store.record_trust_vote(_cache_claim_iri, VoteKind.IMPLICIT_USE)
                except Exception as e:
                    print(f"⚠️ cache reconstruct failed, regenerating: {e}")
                    _cache_claim_iri = None  # fall through to generation

            if _cache_claim_iri is None:
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

                # Best-effort persistence to the graph store.
                if _store is not None and _idents:
                    try:
                        _store.save_claim(
                            metadata,
                            run_meta={
                                'provider': llm_provider, 'llm_model': model,
                                'prompt_version': 'v1', 'code_version': 'head',
                                'mode': mode, 'use_case': use_case,
                            },
                            identifiers=_idents,
                        )
                    except Exception as e:
                        print(f"⚠️ failed to persist claim to graph store: {e}")

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
                'use_case': (
                    getattr(proc, 'use_case', None)
                    if _cache_claim_iri is None else (metadata.get('use_case') or use_case)
                ) or use_case,
                'use_case_label': get_use_case_label(
                    (
                        getattr(proc, 'use_case', None)
                        if _cache_claim_iri is None else (metadata.get('use_case') or use_case)
                    ) or use_case,
                    'data',
                ),
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
                'conflict_summary': _conflict_summary(metadata),
            }

        # Always generate SPDX 3.0.1 output — it's the headline value prop.
        # If conversion fails, log it but don't break the rest of the response.
        try:
            from aikaboom.utils.spdx_validator import SPDXValidator, validate_spdx_export
            validator = SPDXValidator(
                bom_type=bom_type,
                relationship_cap=spdx_relationship_cap,
            )
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

        # Always generate CycloneDX 1.6 beta output (1.6 is the latest spec
        # version sbom-utility v0.18.x can validate; revisit when sbom-utility
        # ships 1.7 schemas).
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

        # Snapshot this run into the on-disk history so it survives the next
        # /process call (which would otherwise overwrite UPLOAD_FOLDER files
        # of the same subject). Failure here must never break the response.
        try:
            history_row = _history_record(
                bom_type=bom_type,
                metadata=metadata,
                artifacts={
                    'bom':        filename,
                    'spdx':       locals().get('spdx_filename'),
                    'cyclonedx':  locals().get('cdx_filename'),
                    'recursive':  locals().get('recursive_filename'),
                    'linked':     locals().get('linked_filename'),
                },
            )
            response_data['history'] = history_row
        except Exception as hist_exc:
            print(f"⚠️ history record failed: {hist_exc}")

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


@app.route('/history', methods=['GET'])
def list_history():
    """Return the BOM history ledger (most recent first)."""
    return jsonify({'rows': _history_load()})


@app.route('/history/<hash_id>', methods=['GET'])
def get_history_entry(hash_id):
    """Return one history row plus the parsed BOM/SPDX/CycloneDX JSON so the
    frontend can rehydrate the viewer tabs without a separate round-trip."""
    safe_hash = ''.join(c for c in hash_id if c.isalnum())[:32]
    if not safe_hash:
        return jsonify({'status': 'error', 'message': 'invalid hash'}), 400
    rows = _history_load()
    row = next((r for r in rows if r.get('hash') == safe_hash), None)
    if not row:
        return jsonify({'status': 'error', 'message': 'not found'}), 404
    artifacts = row.get('artifacts') or {}
    payload = {'row': row, 'data': {}}
    for label, fname in artifacts.items():
        if not fname:
            continue
        path = os.path.join(app.config['HISTORY_FOLDER'], secure_filename(fname))
        if not os.path.exists(path):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload['data'][label] = json.load(f)
        except (OSError, ValueError) as exc:
            payload['data'][label] = {'error': str(exc)}
    return jsonify(payload)


@app.route('/history/<hash_id>/<label>', methods=['GET'])
def download_history(hash_id, label):
    """Download a specific artifact from a history row."""
    safe_hash = ''.join(c for c in hash_id if c.isalnum())[:32]
    rows = _history_load()
    row = next((r for r in rows if r.get('hash') == safe_hash), None)
    if not row:
        return jsonify({'status': 'error', 'message': 'not found'}), 404
    fname = (row.get('artifacts') or {}).get(label)
    if not fname:
        return jsonify({'status': 'error', 'message': 'artifact not present'}), 404
    path = os.path.join(app.config['HISTORY_FOLDER'], secure_filename(fname))
    if not os.path.exists(path):
        return jsonify({'status': 'error', 'message': 'file missing on disk'}), 404
    return send_file(path, as_attachment=True, download_name=fname)


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


@app.route('/worldofboms/graph', methods=['GET'])
def worldofboms_graph():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'nodes': [], 'edges': [], 'store_unavailable': True})
    try:
        return jsonify(graph_view.full_graph(store))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms graph failed: {e}")
        return jsonify({'nodes': [], 'edges': [], 'store_unavailable': True})


@app.route('/worldofboms/stats', methods=['GET'])
def worldofboms_stats():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'artifacts': 0, 'versions': 0, 'claims': 0,
                        'edges': 0, 'store_unavailable': True})
    try:
        stats = dict(store.stats())
        stats['edges'] = graph_view.edge_count(store)
        return jsonify(stats)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms stats failed: {e}")
        return jsonify({'artifacts': 0, 'edges': 0, 'store_unavailable': True})


@app.route('/worldofboms/ego/<path:artifact>', methods=['GET'])
def worldofboms_ego(artifact):
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'nodes': [], 'edges': [], 'focus': artifact,
                        'store_unavailable': True})
    direction = request.args.get('direction', 'both')
    if direction not in ('up', 'down', 'both'):
        direction = 'both'
    depth_arg = request.args.get('depth')
    depth = int(depth_arg) if depth_arg and depth_arg.isdigit() else None
    try:
        return jsonify(graph_view.ego_graph(store, artifact,
                                            direction=direction, depth=depth))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms ego failed: {e}")
        return jsonify({'nodes': [], 'edges': [], 'focus': artifact,
                        'store_unavailable': True})


@app.route('/worldofboms/bom/<path:artifact>', methods=['GET'])
def worldofboms_bom(artifact):
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'store_unavailable': True})
    try:
        claim = graph_view.canonical_claim_iri(store, artifact)
        if not claim:
            return jsonify({'error': 'no claim for artifact'}), 404
        return jsonify(store.reconstruct_bom(claim))
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms bom failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/worldofboms/query', methods=['POST'])
def worldofboms_query():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'rows': [], 'store_unavailable': True})
    data = request.get_json(silent=True) or {}
    try:
        if data.get('sparql'):
            rows = graph_view.raw_query(store, data['sparql'])
        else:
            rows = graph_view.lineage_query(
                store, data.get('artifact', ''),
                preset=data.get('preset', 'datasets'),
                direction=data.get('direction', 'both'),
            )
        return jsonify({'rows': rows})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms query failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/worldofboms/export', methods=['GET'])
def worldofboms_export():
    from aikaboom.store import graph_view
    store = _open_graph_store()
    if store is None:
        return jsonify({'@context': None, '@graph': [], 'store_unavailable': True})
    scope = request.args.get('scope', 'full')
    artifact = request.args.get('artifact') if scope == 'ego' else None
    direction = request.args.get('direction', 'both')
    if scope == 'ego' and not artifact:
        return jsonify({'error': 'scope=ego requires an artifact parameter'}), 400
    try:
        bundle = graph_view.ego_spdx_bundle(store, artifact, direction=direction)
        resp = jsonify(bundle)
        fname = 'worldofboms-graph.spdx.json' if scope == 'full' \
            else 'worldofboms-ego.spdx.json'
        resp.headers['Content-Disposition'] = f'attachment; filename="{fname}"'
        return resp
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ worldofboms export failed: {e}")
        return jsonify({'error': str(e)}), 500


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
