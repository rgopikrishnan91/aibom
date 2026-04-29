# metadata_utils.py
import os
import requests
from urllib.parse import urlparse
from huggingface_hub import HfApi
import fitz  # PyMuPDF

def _get_github_headers():
    """Build GitHub API headers with token from environment (if available)."""
    headers = {"Accept": "application/vnd.github.v3.raw"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


class MetadataFetcher:
    @staticmethod
    def extract_github_user_repo(url):
        parts = urlparse(url).path.strip("/").split("/")
        return (parts[0], parts[1]) if len(parts) >= 2 else (None, None)

    @staticmethod
    def extract_repo_path(github_url):
        """
        Extracts 'owner/repo' from a full GitHub URL, even if it includes branches or subfolders.
        """
        parsed = urlparse(github_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        else:
            raise ValueError(f"Invalid GitHub repository URL: {github_url}")

    @staticmethod
    def extract_repo_id_from_hf_url(url):
        """
        Given any HF URL (dataset landing page, blob link to a file, etc.),
        return the "namespace/repo" part that you can feed to api.dataset_info().
        """
        if not url or not isinstance(url, str):
            return None

        parsed = urlparse(url)
        # split & drop empty segments
        parts = [p for p in parsed.path.split("/") if p]

        # 1) If it's a datasets URL, e.g.
        #    /datasets/<namespace>/<repo>/...  
        if "datasets" in parts:
            idx = parts.index("datasets")
            if len(parts) > idx + 2:
                namespace = parts[idx + 1]
                repo      = parts[idx + 2]
                return f"{namespace}/{repo}"

        # 2) Otherwise, maybe it's just https://huggingface.co/<namespace>/<repo>
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"

        # 3) Give up
        return None

    @staticmethod
    def fetch_github_repo_license(user, repo):
        url = f"https://api.github.com/repos/{user}/{repo}/license"
        try:
            r = requests.get(url, headers=_get_github_headers(), timeout=30)
            if r.status_code == 200:
                data = r.json()
                return {
                    "license_name": data["license"]["name"] if data.get("license") else None,
                    "license_spdx_id": data["license"]["spdx_id"] if data.get("license") else None,
                    "license_url": data["html_url"]
                }
        except Exception:
            pass
        return {"license_name": None, "license_spdx_id": None, "license_url": None}

    @staticmethod
    def get_pdf_text_from_arxiv(arxiv_url):
        import tempfile
        arxiv_id = arxiv_url.split('/')[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=30)
        if response.status_code == 200:
            temp_pdf_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    temp_pdf_path = tmp.name
                    tmp.write(response.content)
                doc = fitz.open(temp_pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            finally:
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
        else:
            print(f"Failed to download PDF from {pdf_url}")
            return ""
    
    @staticmethod
    def inspect_github_BOM_Fields(github_repo, bom_type='ai'):
        """
        Extract BOM-related metadata fields from a GitHub repository.
        Supports both AI and Data BOM types.
        """
        github_metadata = {}

        try:
            repo = github_repo

            if bom_type == 'ai':
                # AI BOM fields
                github_metadata["releaseTime"] = repo.pushed_at.isoformat() if repo.pushed_at else (repo.created_at.isoformat() if repo.created_at else None)
                github_metadata["suppliedBy"] = repo.owner.login if repo.owner and repo.owner.login else None
                github_metadata["downloadLocation"] = repo.clone_url if repo.clone_url else repo.html_url
                
                # Package Version
                try:
                    releases = repo.get_releases()
                    latest_release = releases[0] if releases.totalCount > 0 else None
                    if latest_release:
                        github_metadata["packageVersion"] = latest_release.tag_name
                    else:
                        tags = repo.get_tags()
                        github_metadata["packageVersion"] = tags[0].name if tags.totalCount > 0 else None
                except Exception:
                    github_metadata["packageVersion"] = None
                
                # Primary Purpose
                purpose_indicators = []
                if repo.description:
                    purpose_indicators.append(repo.description)
                try:
                    topics = repo.get_topics()
                    if topics:
                        purpose_indicators.extend(topics)
                except Exception:
                    pass
                github_metadata["primaryPurpose"] = "; ".join(purpose_indicators) if purpose_indicators else None
                
                # License
                try:
                    license_info = repo.get_license()
                    github_metadata["license"] = license_info.license.name if license_info else None
                except Exception:
                    try:
                        github_metadata["license"] = repo.license.name if repo.license else None
                    except Exception:
                        github_metadata["license"] = None
            else:
                # Data BOM fields
                github_metadata["builtTime"] = repo.created_at.isoformat() if repo.created_at else None
                github_metadata["originatedBy"] = repo.owner.login if repo.owner and repo.owner.login else None
                github_metadata["software_downloadLocation"] = repo.clone_url if repo.clone_url else None
                github_metadata["releaseTime"] = repo.pushed_at.isoformat() if repo.pushed_at else None
                github_metadata["name"] = repo.name if repo.name else None
                github_metadata["datasetAvailability"] = "public" if not repo.private else "private"
                # Primary Purpose (description + topics, same as AI BOM)
                purpose_indicators = []
                if repo.description:
                    purpose_indicators.append(repo.description)
                try:
                    topics = repo.get_topics()
                    if topics:
                        purpose_indicators.extend(topics)
                except Exception:
                    pass
                github_metadata["primaryPurpose"] = "; ".join(purpose_indicators) if purpose_indicators else None

                # License
                try:
                    license_info = repo.get_license()
                    github_metadata["license"] = license_info.license.name if license_info else None
                except Exception:
                    try:
                        github_metadata["license"] = repo.license.name if repo.license else None
                    except Exception:
                        github_metadata["license"] = None

        except Exception as e:
            print(f"Error retrieving metadata from {repo.full_name}: {e}")
            return {}

        return github_metadata

    @staticmethod
    def inspect_huggingface_BOM_Fields(hf_repo, bom_type='ai'):
        """
        Extract BOM-style metadata from a Hugging Face repository.
        Supports both AI and Data BOM types.
        """
        hf_metadata = {}

        try:
            repo_info = hf_repo

            if bom_type == 'ai':
                # AI BOM fields
                hf_metadata["releaseTime"] = repo_info.last_modified.isoformat() if repo_info.last_modified else None
                hf_metadata["suppliedBy"] = repo_info.author if repo_info.author else (repo_info.id.split('/')[0] if '/' in repo_info.id else None)
                hf_metadata["downloadLocation"] = f"https://huggingface.co/{repo_info.id}"
                
                # Package Version
                try:
                    if hasattr(repo_info, 'sha') and repo_info.sha:
                        hf_metadata["packageVersion"] = repo_info.sha[:8]
                    else:
                        hf_metadata["packageVersion"] = None
                except Exception:
                    hf_metadata["packageVersion"] = None
                
                # Primary Purpose
                # First try: $.components[0].modelCard.modelParameters.task
                purpose_from_components = None
                try:
                    components = None
                    if hasattr(repo_info, 'components') and repo_info.components:
                        components = repo_info.components
                    elif repo_info.cardData and isinstance(repo_info.cardData.get('components'), list):
                        components = repo_info.cardData.get('components')
                    if components:
                        first_comp = components[0]
                        if isinstance(first_comp, dict):
                            task = (first_comp
                                    .get('modelCard', {})
                                    .get('modelParameters', {})
                                    .get('task'))
                        else:
                            task = (getattr(getattr(getattr(first_comp, 'modelCard', None), 'modelParameters', None), 'task', None))
                        if task:
                            purpose_from_components = str(task) if not isinstance(task, list) else "; ".join(task)
                except Exception:
                    pass

                if purpose_from_components:
                    hf_metadata["primaryPurpose"] = purpose_from_components
                else:
                    # Fallback: task_categories + pipeline_tag + first 3 tags
                    purpose_indicators = []
                    if repo_info.cardData:
                        if repo_info.cardData.get("task_categories"):
                            task_cats = repo_info.cardData.get("task_categories")
                            if isinstance(task_cats, list):
                                purpose_indicators.extend(task_cats)
                            else:
                                purpose_indicators.append(str(task_cats))
                        if repo_info.cardData.get("pipeline_tag"):
                            purpose_indicators.append(repo_info.cardData.get("pipeline_tag"))
                    if hasattr(repo_info, 'tags') and repo_info.tags:
                        purpose_indicators.extend(repo_info.tags[:3])
                    hf_metadata["primaryPurpose"] = "; ".join(purpose_indicators) if purpose_indicators else None
                
                # License
                hf_metadata["license"] = repo_info.cardData.get("license") if repo_info.cardData else None

                # Model-tree relationship hints: structured signal extracted
                # from HF metadata that mirrors the SPDX trainedOn / testedOn /
                # dependsOn relationships. These contribute a separate source
                # so the RAG pipeline's conflict detector can compare them
                # against the README/arxiv/github text.
                tree = MetadataFetcher.extract_huggingface_model_tree(repo_info)
                if tree.get("trainedOnDatasets"):
                    hf_metadata["trainedOnDatasets"] = ", ".join(tree["trainedOnDatasets"])
                if tree.get("testedOnDatasets"):
                    hf_metadata["testedOnDatasets"] = ", ".join(tree["testedOnDatasets"])
                if tree.get("modelLineage"):
                    hf_metadata["modelLineage"] = ", ".join(tree["modelLineage"])
            else:
                # Data BOM fields
                hf_metadata["releaseTime"] = repo_info.last_modified.isoformat() if repo_info.last_modified else None
                hf_metadata["originatedBy"] = repo_info.author if repo_info.author else None
                hf_metadata["software_downloadLocation"] = f"https://huggingface.co/{repo_info.id}"
                hf_metadata["builtTime"] = repo_info.created_at.isoformat() if repo_info.created_at else None
                hf_metadata["name"] = repo_info.id if repo_info.id else None
                hf_metadata["datasetAvailability"] = "public" if repo_info.private is False else "private"
                hf_metadata["primaryPurpose"] = repo_info.cardData.get("task_categories") if repo_info.cardData else None
                hf_metadata["license"] = repo_info.cardData.get("license") if repo_info.cardData else None
                hf_metadata["sourceInfo"] = repo_info.cardData.get("source_datasets") if repo_info.cardData else None

        except Exception as e:
            print(f"Error retrieving Hugging Face metadata for {repo_info.id}: {e}")
            return {}

        return hf_metadata

    @staticmethod
    def extract_huggingface_model_tree(repo_info):
        """Extract structured trainedOn / testedOn / modelLineage hints from
        an HF ``model_info`` response.

        HuggingFace exposes datasets and base models in several places:
        ``cardData.datasets``, ``cardData.base_model``, ``model_index[*]
        .results[*].dataset.name``, and ``tags`` like ``dataset:squad`` or
        ``base_model:meta-llama/Llama-3``. This helper normalises them into
        three deduplicated lists so the RAG conflict detector can compare
        them against the README / arXiv / GitHub text.
        """
        result = {"trainedOnDatasets": [], "testedOnDatasets": [], "modelLineage": []}
        if repo_info is None:
            return result

        def _add(bucket, value):
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    _add(bucket, item)
                return
            text = str(value).strip()
            if not text or text.lower() in {"none", "unknown", "n/a"}:
                return
            if text not in result[bucket]:
                result[bucket].append(text)

        card = getattr(repo_info, "cardData", None) or {}
        if isinstance(card, dict):
            _add("trainedOnDatasets", card.get("datasets"))
            _add("modelLineage", card.get("base_model"))

            model_index = card.get("model_index") or card.get("model-index")
            if isinstance(model_index, list):
                for entry in model_index:
                    if not isinstance(entry, dict):
                        continue
                    for r in entry.get("results", []) or []:
                        ds = (r or {}).get("dataset") or {}
                        if isinstance(ds, dict):
                            _add("testedOnDatasets", ds.get("name") or ds.get("type"))

        for tag in (getattr(repo_info, "tags", None) or []):
            if not isinstance(tag, str):
                continue
            if tag.startswith("dataset:"):
                _add("trainedOnDatasets", tag.split(":", 1)[1])
            elif tag.startswith("base_model:"):
                _add("modelLineage", tag.split(":", 1)[1])

        return result
