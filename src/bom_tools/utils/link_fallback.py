"""
Link Fallback Module - Uses Gemini to find missing AI model links
Finds missing HuggingFace, ArXiv, or GitHub links when user provides incomplete information

Required Environment Variables:
    GEMINI_API_KEY: Your Google Gemini API key (required)
    
Optional Environment Variables:
    DEBUG_FALLBACK: Set to "true" to enable verbose debugging output
    
Example .env file:
    GEMINI_API_KEY=your_api_key_here
    DEBUG_FALLBACK=false
"""

import ssl
import os
import re
from typing import Dict, Optional, Tuple, List
from dotenv import load_dotenv

# Import optional dependencies with fallbacks
try:
    import httpx
    from httpx import ConnectError, ConnectTimeout
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    # Create placeholder exceptions
    ConnectError = ConnectionError
    ConnectTimeout = TimeoutError
    httpx = None

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    types = None

# Load environment variables - try to find .env file explicitly
# This ensures it works both when run directly and when imported by Flask
_env_path = None
_current_dir = os.path.dirname(os.path.abspath(__file__))
_env_file = os.path.join(_current_dir, '.env')
if os.path.exists(_env_file):
    _env_path = _env_file
load_dotenv(_env_path)


class LinkFallbackFinder:
    """Finds missing links using Gemini with web search"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with SSL verification disabled"""
        # Check if required dependencies are available
        if not GENAI_AVAILABLE:
            print("⚠️ Warning: google-genai not installed. Link fallback will be disabled.")
            print("   Install with: pip install google-genai")
            self.client = None
            self.config = None
            return
        
        if not HTTPX_AVAILABLE:
            print("⚠️ Warning: httpx not installed. Using simple Gemini client initialization.")
            print("   For better network handling, install with: pip install httpx")
        
        # Get API key from parameter, environment variable, or raise error
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            print("⚠️ Warning: GEMINI_API_KEY not found! Link fallback will be disabled.")
            print("   Set it in your .env file: GEMINI_API_KEY=your_api_key_here")
            self.client = None
            self.config = None
            return
        
        # HTTPX transport (only if httpx is available)
        if HTTPX_AVAILABLE:
            timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        else:
            timeout = None
        
        try:
            # Ensure API key is available
            if not self.api_key or not self.api_key.strip():
                raise ValueError("GEMINI_API_KEY is empty or invalid")
            
            # Debug: Print API key status (masked)
            if os.getenv("DEBUG_FALLBACK", "false").lower() == "true":
                print(f"   [DEBUG] Using API key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else '****'}")
            
            # Initialize client with simplified configuration
            # The Gemini client may not fully support all httpx options, so we use minimal config
            try:
                # Try with HTTP options if httpx is available
                if HTTPX_AVAILABLE and timeout:
                    try:
                        self.client = genai.Client(
                            api_key=self.api_key,
                            http_options=types.HttpOptions(
                                client_args={
                                    "timeout": timeout,
                                    "trust_env": True,
                                }
                            ),
                        )
                    except Exception as http_config_error:
                        # Fallback to simple init if HTTP config fails
                        print(f"   ⚠️ Warning: HTTP options failed, using simple init: {http_config_error}")
                        self.client = genai.Client(api_key=self.api_key)
                else:
                    # Simple initialization without httpx
                    self.client = genai.Client(api_key=self.api_key)
            except Exception as client_init_error:
                print(f"   ❌ Failed to initialize Gemini client: {client_init_error}")
                print(f"   Link fallback will be disabled.")
                self.client = None
                self.config = None
                return
            
            # Configure grounding tool for web search
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            self.config = types.GenerateContentConfig(tools=[grounding_tool])
            
            print("✓ Initialized LinkFallbackFinder with Gemini")
            
            # Test connection (optional but helpful for debugging)
            if os.getenv("TEST_CONNECTION", "false").lower() == "true":
                try:
                    import socket
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(5)
                    result = test_socket.connect_ex(("generativelanguage.googleapis.com", 443))
                    test_socket.close()
                    if result == 0:
                        print("   ✓ Network connectivity check passed")
                    else:
                        print(f"   ⚠️ Network connectivity check failed (code: {result})")
                except Exception as conn_err:
                    print(f"   ⚠️ Network connectivity check error: {conn_err}")
                    
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"❌ Failed to initialize Gemini client ({error_type}): {error_msg}")
            if os.getenv("DEBUG_FALLBACK", "false").lower() == "true":
                import traceback
                traceback.print_exc()
            raise RuntimeError(f"Failed to initialize Gemini client: {e}") from e
    
    def _extract_model_name(self, repo_id: Optional[str], url: Optional[str]) -> str:
        """Extract model name from repo_id or URL"""
        if repo_id:
            return repo_id.split('/')[-1] if '/' in repo_id else repo_id
        if url:
            # Extract from URL
            parts = url.rstrip('/').split('/')
            return parts[-1] if parts else "model"
        return "model"
    
    def _is_valid_url(self, url: str, url_type: str) -> bool:
        """Validate URL format"""
        if not url:
            return False
        
        url = url.strip().lower()
        
        if url_type == "huggingface":
            return "huggingface.co" in url or url.startswith("hf.co/")
        elif url_type == "arxiv":
            return "arxiv.org" in url and ("abs/" in url or "pdf/" in url)
        elif url_type == "github":
            return "github.com" in url
        
        return False
    
    def _extract_url_from_text(self, text: str, url_type: str) -> Optional[str]:
        """Extract URL from Gemini response text"""
        if not text:
            return None
        
        # Look for URLs in the text - more flexible patterns
        url_patterns = {
            "huggingface": [
                r'https?://[^\s]*huggingface\.co/[^\s\)"]+',
                r'https?://[^\s]*hf\.co/[^\s\)"]+',
                r'huggingface\.co/[^\s\)"]+',
                r'hf\.co/[^\s\)"]+',
            ],
            "arxiv": [
                r'https?://[^\s]*arxiv\.org/[^\s\)"]+',
                r'arxiv\.org/[^\s\)"]+',
            ],
            "github": [
                r'https?://[^\s]*github\.com/[^\s\)"]+',
                r'github\.com/[^\s\)"]+',
            ]
        }
        
        patterns = url_patterns.get(url_type, [])
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                url = matches[0].rstrip('.,;!?)\'"')
                # Add https:// if missing
                if not url.startswith('http'):
                    url = f"https://{url}"
                
                if self._is_valid_url(url, url_type):
                    return url
        
        # Fallback: look for any URL that might contain the domain
        fallback_patterns = {
            "huggingface": r'https?://[^\s]*huggingface[^\s\)"]+',
            "arxiv": r'https?://[^\s]*arxiv[^\s\)"]+',
            "github": r'https?://[^\s]*github[^\s\)"]+',
        }
        
        if url_type in fallback_patterns:
            matches = re.findall(fallback_patterns[url_type], text, re.IGNORECASE)
            if matches:
                url = matches[0].rstrip('.,;!?)\'"')
                if self._is_valid_url(url, url_type):
                    return url
        
        return None
    
    def _find_missing_link(self, missing_type: str, available_info: Dict[str, Optional[str]]) -> Optional[str]:
        """Use Gemini to find a missing link"""
        repo_id = available_info.get('repo_id')
        hf_repo_id = available_info.get('hf_repo_id')
        arxiv_url = available_info.get('arxiv_url')
        github_url = available_info.get('github_url')
        
        # Extract model identifier from available info
        model_identifier = None
        if repo_id:
            model_identifier = repo_id
        elif hf_repo_id:
            model_identifier = hf_repo_id
        elif github_url:
            model_identifier = github_url.split('/')[-1]
        elif arxiv_url:
            model_identifier = arxiv_url.split('/')[-1]
        
        if not model_identifier:
            print(f"⚠️ Cannot find {missing_type} link: No model identifier available")
            return None
        
        # Build prompt based on missing link type
        prompts = {
            "huggingface": f"""Find the HuggingFace model page for this AI model: {model_identifier}

Available information:
{f"- HuggingFace repo ID: {hf_repo_id}" if hf_repo_id else ""}
{f"- ArXiv paper: {arxiv_url}" if arxiv_url else ""}
{f"- GitHub repository: {github_url}" if github_url else ""}

Please search for and provide the exact HuggingFace model URL (huggingface.co or hf.co). The URL should be for the model page, not the organization page.
Respond with ONLY the URL, nothing else.""",

            "arxiv": f"""Find the ArXiv paper URL for this AI model: {model_identifier}

Available information:
{f"- HuggingFace repo ID: {hf_repo_id}" if hf_repo_id else ""}
{f"- GitHub repository: {github_url}" if github_url else ""}

Please search for and provide the exact ArXiv paper URL (arxiv.org/abs/...). 
Respond with ONLY the URL, nothing else.""",

            "github": f"""Find the GitHub repository URL for this AI model: {model_identifier}

Available information:
{f"- HuggingFace repo ID: {hf_repo_id}" if hf_repo_id else ""}
{f"- ArXiv paper: {arxiv_url}" if arxiv_url else ""}

Please search for and provide the exact GitHub repository URL (github.com/owner/repo).
Respond with ONLY the URL, nothing else."""
        }
        
        prompt = prompts.get(missing_type)
        if not prompt:
            return None
        
        try:
            print(f"🔍 Searching for {missing_type} link using Gemini...")
            print(f"   Prompt length: {len(prompt)} characters")
            
            # Verify client is initialized
            if not hasattr(self, 'client') or self.client is None:
                raise RuntimeError("Gemini client not initialized")
            
            if not hasattr(self, 'config') or self.config is None:
                raise RuntimeError("Gemini config not initialized")
            
            # Check network connectivity first (optional - can be disabled for speed)
            if os.getenv("CHECK_NETWORK", "true").lower() == "true":
                try:
                    import socket
                    # Test DNS resolution
                    try:
                        hostname = "generativelanguage.googleapis.com"
                        port = 443
                        print(f"   Testing network connectivity to {hostname}:{port}...")
                        socket.create_connection((hostname, port), timeout=5)
                        print(f"   ✓ Network connectivity check passed")
                    except socket.gaierror as dns_err:
                        print(f"   ❌ DNS resolution failed: {dns_err}")
                        print(f"   Error details: {dns_err.args}")
                        print(f"   This indicates the hostname cannot be resolved.")
                        print(f"   Possible causes: No internet, DNS issues, or firewall blocking.")
                        raise RuntimeError(f"DNS resolution failed for {hostname}: {dns_err}") from dns_err
                    except (socket.timeout, OSError) as net_err:
                        print(f"   ⚠️ Network connectivity check failed: {net_err}")
                        print(f"   This may indicate network connectivity issues.")
                        print(f"   The fallback will still attempt to proceed...")
                except RuntimeError:
                    # Re-raise DNS errors immediately
                    raise
            
            # Debug: Print client info if enabled
            if os.getenv("DEBUG_FALLBACK", "false").lower() == "true":
                print(f"   [DEBUG] Client type: {type(self.client)}")
                print(f"   [DEBUG] Config type: {type(self.config)}")
            
            print(f"   Making API request to Gemini...")
            # Try with grounding tool first (web search enabled)
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=self.config,
                )
            except (ConnectError, ConnectTimeout, OSError, ConnectionError) as api_error:
                # If connection fails, try without grounding tool (simpler request)
                error_msg = str(api_error)
                error_type = type(api_error).__name__
                print(f"   ⚠️ API call with grounding tool failed ({error_type}): {error_msg}")
                print(f"   Trying without grounding tool (web search disabled)...")
                try:
                    # Try without the grounding tool config - simpler request
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                    )
                except Exception as retry_error:
                    print(f"   ❌ Retry without grounding tool also failed: {retry_error}")
                    raise api_error  # Raise original error
            except Exception as api_error:
                # For other errors, just re-raise
                raise
            
            if not response:
                raise RuntimeError("Gemini API returned no response")
            
            response_text = response.text.strip() if response.text else ""
            
            if not response_text:
                print(f"   ⚠️ Warning: Gemini returned empty response")
                return None
            
            # Debug: Print full response if verbose
            if os.getenv("DEBUG_FALLBACK", "false").lower() == "true":
                print(f"   [DEBUG] Full Gemini response: {response_text}")
            
            # Extract URL from response
            found_url = self._extract_url_from_text(response_text, missing_type)
            
            if found_url:
                print(f"✓ Found {missing_type} link: {found_url}")
                return found_url
            else:
                print(f"⚠️ Could not extract valid {missing_type} URL from Gemini response")
                print(f"   Response preview: {response_text[:200]}...")
                # Try to find URL even if pattern doesn't match exactly
                if response_text:
                    # Look for any URL that might be related
                    all_urls = re.findall(r'https?://[^\s\)"]+', response_text)
                    if all_urls:
                        print(f"   Found potential URLs in response (first 3):")
                        for url in all_urls[:3]:
                            print(f"     - {url}")
                        
                        # Try to match any URL that looks like the target type
                        for url in all_urls:
                            url_clean = url.rstrip('.,;!?)\'"')
                            if self._is_valid_url(url_clean, missing_type):
                                print(f"   ✓ Found valid {missing_type} URL after fallback: {url_clean}")
                                return url_clean
                return None
                
        except (OSError, ConnectionError, ConnectError, ConnectTimeout) as e:
            # Handle network/DNS errors specifically
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Check for various connection error patterns
            is_connection_error = (
                "Name or service not known" in error_msg or 
                "Name resolution failed" in error_msg or
                "ConnectError" in error_type or
                "ConnectionError" in error_type or
                "Connection refused" in error_msg or
                "Network is unreachable" in error_msg
            )
            
            if is_connection_error:
                print(f"❌ Connection error when connecting to Gemini API ({error_type})")
                print(f"   Error: {e}")
                print(f"   This usually indicates:")
                print(f"   - Network connectivity issues")
                print(f"   - DNS resolution problems")
                print(f"   - Firewall/proxy blocking the API")
                print(f"   - The Gemini API endpoint may be unreachable from this network")
                print(f"   Skipping {missing_type} link search...")
            else:
                print(f"❌ Network error finding {missing_type} link ({error_type}): {e}")
                print(f"   This may be a connectivity issue. Skipping...")
            return None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"❌ Error finding {missing_type} link ({error_type}): {error_msg}")
            
            # Provide more specific error messages
            if "timeout" in error_msg.lower():
                print(f"   The request timed out. This may indicate slow network or API issues.")
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                print(f"   Network connectivity issue detected.")
            elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                print(f"   API key authentication failed. Please check your GEMINI_API_KEY.")
            
            if os.getenv("DEBUG_FALLBACK", "false").lower() == "true":
                print(f"   [DEBUG] Full traceback:")
                import traceback
                traceback.print_exc()
            return None
    
    def find_missing_links(
        self,
        repo_id: Optional[str] = None,
        hf_repo_id: Optional[str] = None,
        arxiv_url: Optional[str] = None,
        github_url: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Dict[str, bool]]:
        """
        Find missing links and return complete set
        
        Args:
            repo_id: HuggingFace repo ID (e.g., "org/model-name")
            hf_repo_id: Alternative HuggingFace repo ID
            arxiv_url: ArXiv paper URL
            github_url: GitHub repository URL
        
        Returns:
            Tuple of (final_hf_repo_id, final_arxiv_url, final_github_url, status_dict)
            status_dict indicates which links were found vs provided
        """
        # Normalize inputs - convert empty strings to None
        def normalize(value):
            if value is None:
                return None
            if isinstance(value, str):
                value = value.strip()
                return value if value else None
            return value
        
        repo_id = normalize(repo_id)
        hf_repo_id = normalize(hf_repo_id)
        arxiv_url = normalize(arxiv_url)
        github_url = normalize(github_url)
        
        # Normalize inputs
        final_hf_repo_id = hf_repo_id or repo_id
        final_arxiv_url = arxiv_url
        final_github_url = github_url
        
        # Track what we have
        has_hf = bool(final_hf_repo_id)
        has_arxiv = bool(final_arxiv_url)
        has_github = bool(final_github_url)
        
        missing_count = sum([not has_hf, not has_arxiv, not has_github])
        provided_count = 3 - missing_count
        
        print(f"\n{'='*70}")
        print(f"LINK FALLBACK ANALYSIS")
        print(f"{'='*70}")
        print(f"Provided links: {provided_count}/3")
        print(f"  - HuggingFace: {'✓' if has_hf else '✗'} {'(provided)' if has_hf else '(missing)'}")
        print(f"  - ArXiv: {'✓' if has_arxiv else '✗'} {'(provided)' if has_arxiv else '(missing)'}")
        print(f"  - GitHub: {'✓' if has_github else '✗'} {'(provided)' if has_github else '(missing)'}")
        print(f"Missing links: {missing_count}/3")
        print(f"{'='*70}\n")
        
        if missing_count == 0:
            print("✓ All links provided, no fallback needed")
            return final_hf_repo_id, final_arxiv_url, final_github_url, {
                'hf_provided': True,
                'arxiv_provided': True,
                'github_provided': True,
                'hf_found': False,
                'arxiv_found': False,
                'github_found': False
            }
        
        # Prepare available info for Gemini
        available_info = {
            'repo_id': repo_id or hf_repo_id,
            'hf_repo_id': final_hf_repo_id,
            'arxiv_url': final_arxiv_url,
            'github_url': final_github_url
        }
        
        status = {
            'hf_provided': has_hf,
            'arxiv_provided': has_arxiv,
            'github_provided': has_github,
            'hf_found': False,
            'arxiv_found': False,
            'github_found': False
        }
        
        found_count = 0
        
        # Find missing HuggingFace link
        if not has_hf:
            found_hf = self._find_missing_link("huggingface", available_info)
            if found_hf:
                # Extract repo ID from URL if full URL provided
                if "huggingface.co" in found_hf:
                    parts = found_hf.split("huggingface.co/")[-1].split("/")
                    if len(parts) >= 2:
                        final_hf_repo_id = f"{parts[0]}/{parts[1]}"
                    else:
                        final_hf_repo_id = found_hf
                elif "hf.co" in found_hf:
                    final_hf_repo_id = found_hf.split("hf.co/")[-1]
                else:
                    final_hf_repo_id = found_hf
                status['hf_found'] = True
                found_count += 1
                available_info['hf_repo_id'] = final_hf_repo_id
        
        # Find missing ArXiv link
        if not has_arxiv:
            found_arxiv = self._find_missing_link("arxiv", available_info)
            if found_arxiv:
                final_arxiv_url = found_arxiv
                status['arxiv_found'] = True
                found_count += 1
                available_info['arxiv_url'] = final_arxiv_url
        
        # Find missing GitHub link
        if not has_github:
            found_github = self._find_missing_link("github", available_info)
            if found_github:
                final_github_url = found_github
                status['github_found'] = True
                found_count += 1
                available_info['github_url'] = final_github_url
        
        print(f"\n{'='*70}")
        print(f"LINK FALLBACK RESULTS")
        print(f"{'='*70}")
        print(f"Links provided: {provided_count}/3")
        print(f"Links found: {found_count}/{missing_count}")
        print(f"  - HuggingFace: {'✓ FOUND' if status['hf_found'] else '✗ NOT FOUND' if not has_hf else '✓ PROVIDED'}")
        print(f"  - ArXiv: {'✓ FOUND' if status['arxiv_found'] else '✗ NOT FOUND' if not has_arxiv else '✓ PROVIDED'}")
        print(f"  - GitHub: {'✓ FOUND' if status['github_found'] else '✗ NOT FOUND' if not has_github else '✓ PROVIDED'}")
        print(f"{'='*70}\n")
        
        return final_hf_repo_id, final_arxiv_url, final_github_url, status
