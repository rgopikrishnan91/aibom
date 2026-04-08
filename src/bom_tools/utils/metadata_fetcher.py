# metadata_utils.py
import os
import requests
from urllib.parse import urlparse
from huggingface_hub import HfApi
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import inspect

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_HEADERS = {
    "Accept": "application/vnd.github.v3.raw"
}
if GITHUB_TOKEN:
    GITHUB_HEADERS["Authorization"] = f"token {GITHUB_TOKEN}"
class MetadataFetcher:
    @staticmethod
    def extract_hf_repo_id(url):
        parts = urlparse(url).path.strip("/").split("/")
        if "datasets" in parts:
            parts.remove("datasets")
        return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None

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
            r = requests.get(url, headers=GITHUB_HEADERS, timeout=30)
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
    def inspect_huggingface_dataset(url):
        repo_id = MetadataFetcher.extract_hf_repo_id(url)
        if not repo_id:
            return {}

        try:
            api = HfApi()
            info = api.dataset_info(repo_id)
            files = api.list_repo_files(repo_id, repo_type="dataset")
            license_tag = info.cardData.get('license') if info.cardData else None

            matched_files = {f for f in files if any(k in f.upper() for k in ["LICENSE", "COPYING", "NOTICE", "README"])}
            content = {}
            for f in matched_files:
                r = requests.get(f"https://huggingface.co/datasets/{repo_id}/resolve/main/{f}", timeout=30)
                if r.status_code == 200:
                    content[f] = r.text.strip()

            return {
                "hf_repo_id": repo_id,
                "hf_license_tag": license_tag,
                "hf_file_contents": content,
                "hf_license_files": [f for f in matched_files if "LICENSE" in f.upper() or "COPYING" in f.upper()],
                "hf_notice_files": [f for f in matched_files if "NOTICE" in f.upper()],
                "hf_readme_files": [f for f in matched_files if "README" in f.upper()]
            }
        except Exception:
            return {}

    @staticmethod
    def inspect_github_repo(url):
        user, repo = MetadataFetcher.extract_github_user_repo(url)
        if not user or not repo:
            return {}

        try:
            base = f"https://raw.githubusercontent.com/{user}/{repo}/main"
            license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "NOTICE", "NOTICE.txt"]
            readme_files = ["README.md", "README.txt", "README"]

            found_licenses = {}
            found_readmes = {}

            for f in license_files:
                r = requests.get(f"{base}/{f}", headers=GITHUB_HEADERS, timeout=30)
                if r.status_code == 200:
                    found_licenses[f] = r.text.strip()

            for f in readme_files:
                r = requests.get(f"{base}/{f}", headers=GITHUB_HEADERS, timeout=30)
                if r.status_code == 200:
                    found_readmes[f] = r.text.strip()
                    break

            license_info = MetadataFetcher.fetch_github_repo_license(user, repo)

            return {
                "gh_repo": f"{user}/{repo}",
                "gh_license_files": list(found_licenses.keys()),
                "gh_readme_files": list(found_readmes.keys()),
                "gh_license_contents": found_licenses,
                "gh_readme_contents": found_readmes,
                "gh_license_name": license_info["license_name"],
                "gh_license_spdx_id": license_info["license_spdx_id"],
                "gh_license_url": license_info["license_url"]
            }
        except Exception:
            return {}

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
    def retrieve_paper_by_uid(uid: str):
        """
        Retrieve top-k most relevant chunks from a specific paper by UID using semantic search.
        """
        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection(name="papers")

        try:
            result = collection.get(ids=[uid])
            if result and result['documents']:
                return {
                    'id': result['ids'][0],
                    'document': result['documents'][0],
                }
            return None
        except Exception as e:
            print(f"Error retrieving paper for UID {uid}: {e}")
            return None

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
    def inspect_kaggle_BOM_Fields(kaggle_repo):
        """
        Extract BOM-style metadata from a Kaggle dataset.
        """
        kaggle_metadata = {}

        try:
            repo_info = kaggle_repo
            kaggle_metadata["software_downloadLocation"] = repo_info.url if repo_info.url else None
            kaggle_metadata["builtTime"] = None
            kaggle_metadata["releaseTime"] = None
            kaggle_metadata["name"] = repo_info.title if repo_info.title else None
            kaggle_metadata["originatedBy"] = repo_info.creator_name if repo_info.creator_name else None
            kaggle_metadata["datasetAvailability"] = "public" if repo_info._is_private is False else "private"
            kaggle_metadata["dataset description"] = repo_info.subtitle if repo_info.subtitle else None
            kaggle_metadata["license"] = repo_info._license_name if repo_info._license_name else None
            kaggle_metadata["sourceInfo"] = None

        except Exception as e:
            print(f"Error retrieving Kaggle metadata: {e}")
            return {}

        return kaggle_metadata
