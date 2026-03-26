"""
Unified BOM Processors
Contains AIBOMProcessor and DATABOMProcessor classes for processing AI models and datasets
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Tuple
from github import Github
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from requests.exceptions import HTTPError

from bom_tools.core.agentic_rag import AgenticRAG, DirectLLM, get_fixed_questions, FIXED_QUESTIONS_AI, FIXED_QUESTIONS_DATA
from bom_tools.utils.metadata_fetcher import MetadataFetcher
from bom_tools.core.source_handler import SourceHandler


def _clean_value(value):
    return None if pd.isna(value) else value


def _build_triplet_payload(mapping, source_suffix, conflict_suffix, skip_keys=None):
    payload = {}
    skip_keys = set(skip_keys or [])
    for key, value in mapping.items():
        if key in skip_keys or key.endswith(source_suffix) or key.endswith(conflict_suffix):
            continue
        triplet = {
            "value": _clean_value(value),
            "conflict": _clean_value(mapping.get(f"{key}{conflict_suffix}")),
            "source": _clean_value(mapping.get(f"{key}{source_suffix}"))
        }
        payload[key] = triplet
    return payload


class AIBOMProcessor:
    """Processes AI models using either RAG or Direct LLM mode with Conflict Detection"""
    
    def __init__(self, model: str = "gpt-4o", mode: str = "rag", llm_provider: str = "openai", 
                 ollama_base_url: str = None, questions_config: Dict[str, Dict] = None, use_case: str = 'complete',
                 embedding_provider: str = "local", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize processing system based on mode
        
        Args:
            embedding_provider: 'local' (default, free HuggingFace) or 'openai'
            embedding_model: Model for local embeddings (only used if embedding_provider='local')
        """
        self.mode = mode
        self.model = model
        self.llm_provider = llm_provider
        self.ollama_base_url = ollama_base_url
        self.questions_config = questions_config if questions_config is not None else FIXED_QUESTIONS_AI
        self.use_case = use_case
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        
        if mode == 'rag':
            self.processor = AgenticRAG(
                model=model, 
                llm_provider=llm_provider,
                ollama_base_url=ollama_base_url,
                questions=self.questions_config,
                bom_type='ai',
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
            print(f"✓ Initialized AIBOMProcessor in RAG mode with model: {model}")
        else:
            self.processor = DirectLLM(
                model=model,
                llm_provider=llm_provider,
                ollama_base_url=ollama_base_url,
                questions=self.questions_config,
                bom_type='ai'
            )
            print(f"✓ Initialized AIBOMProcessor in DIRECT mode with model: {model}")
        
        # Initialize API clients
        gh_token = os.getenv("GITHUB_TOKEN")
        if gh_token:
            try:
                self.github_client = Github(gh_token)
            except Exception as exc:
                print(f"Warning: Github client unavailable ({exc})")
                self.github_client = None
        else:
            self.github_client = None

        hf_token = os.getenv("hug_token") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            try:
                self.hf_api = HfApi(token=hf_token)
            except Exception as exc:
                print(f"Warning: Hugging Face client unavailable ({exc})")
                self.hf_api = None
        else:
            self.hf_api = None

    def generate_model_id(self, repo_id: str, github_url: str) -> str:
        """Generate a unique model identifier"""
        if repo_id:
            return repo_id.replace("/", "_").replace(" ", "_")
        
        if github_url and "github.com" in str(github_url):
            parts = github_url.rstrip('/').split('/')
            if len(parts) >= 2:
                return f"{parts[-2]}_{parts[-1]}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ai_model_{timestamp}"
    
    def pivot_results_to_wide_format(self, results: list, model_id: str) -> dict:
        """Convert results from long format to wide format with answer, sources, and conflicts"""
        wide_result = {"model_id": model_id}
        
        for result in results:
            question_type = result['question_type']
            answer = result['answer']
            sources_used = ', '.join(result['sources_used']) if result['sources_used'] else 'None'
            conflicts = result.get('conflicts', 'No conflicts detected')
            
            wide_result[question_type] = answer
            wide_result[f"{question_type}_sources"] = sources_used
            wide_result[f"{question_type}_conflicts"] = conflicts
        
        return wide_result
    
    def fetch_direct_metadata(self, github_url: str, hf_repo_id: str = None) -> dict:
        """Fetch direct metadata from GitHub and HuggingFace"""
        direct_metadata = {}
        github_metadata = {}
        if self.github_client and github_url and "github.com" in str(github_url):
            try:
                repo_path = MetadataFetcher.extract_repo_path(github_url)
                github_repo = self.github_client.get_repo(repo_path)
                github_metadata = MetadataFetcher.inspect_github_BOM_Fields(github_repo, bom_type='ai')
            except Exception as e:
                print(f"Error fetching GitHub metadata: {e}")

        huggingface_metadata = {}
        if self.hf_api and hf_repo_id:
            repo_id = hf_repo_id.strip()
            if repo_id:
                info = None
                try:
                    info = self.hf_api.dataset_info(repo_id)
                except RepositoryNotFoundError:
                    try:
                        info = self.hf_api.model_info(repo_id)
                    except (RepositoryNotFoundError, HTTPError) as e:
                        print(f"Error fetching Hugging Face metadata for {repo_id}: {e}")
                except HTTPError as e:
                    print(f"Error fetching Hugging Face metadata for {repo_id}: {e}")
                if info:
                    huggingface_metadata = MetadataFetcher.inspect_huggingface_BOM_Fields(info, bom_type='ai')

        direct_metadata["releaseTime"], direct_metadata["releaseTime_source"] = SourceHandler.get_field(
            "releaseTime", huggingface_metadata, github_metadata, mode="latest"
        )
        direct_metadata["suppliedBy"], direct_metadata["suppliedBy_source"], direct_metadata["suppliedBy_conflicts"] = SourceHandler.get_field_conflict(
            "suppliedBy", huggingface_metadata, github_metadata
        )
        direct_metadata["downloadLocation"], direct_metadata["downloadLocation_source"], direct_metadata["downloadLocation_conflicts"] = SourceHandler.get_field_conflict(
            "software_downloadLocation", huggingface_metadata, github_metadata
        )
        direct_metadata["packageVersion"], direct_metadata["packageVersion_source"], direct_metadata["packageVersion_conflicts"] = SourceHandler.get_field_conflict(
            "packageVersion", huggingface_metadata, github_metadata
        )
        direct_metadata["primaryPurpose"], direct_metadata["primaryPurpose_source"], direct_metadata["primaryPurpose_conflicts"] = SourceHandler.get_field_conflict(
            "primaryPurpose", huggingface_metadata, github_metadata
        )
        direct_metadata["license"], direct_metadata["license_source"], direct_metadata["license_conflicts"] = SourceHandler.get_field_conflict(
            "license", huggingface_metadata, github_metadata
        )
        return direct_metadata
    
    def process_ai_model(self, repo_id: str, arxiv_url: str, github_url: str) -> tuple:
        """Main processing function for a single AI model"""
        model_id = self.generate_model_id(repo_id, github_url)
        
        print(f"\nProcessing model: {model_id}")
        print(f"Use case: {self.use_case}")
        print(f"Questions to ask: {len(self.questions_config)}")
        
        # Construct HuggingFace URL if repo_id is provided
        huggingface_url = None
        if repo_id and '/' in repo_id:  # repo_id should be in format 'owner/model'
            huggingface_url = f"https://huggingface.co/{repo_id}"
            print(f"Constructed HuggingFace URL: {huggingface_url}")
        
        rag_results = self.processor.process_ai_model(
            repo_id=model_id,
            arxiv_url=arxiv_url,
            github_url=github_url,
            huggingface_url=huggingface_url
        )
        
        wide_metadata = self.pivot_results_to_wide_format(rag_results, model_id)
        rag_fields = _build_triplet_payload(
            wide_metadata,
            source_suffix='_sources',
            conflict_suffix='_conflicts',
            skip_keys={'model_id'}
        )
        
        direct_metadata_raw = self.fetch_direct_metadata(
            github_url=github_url,
            hf_repo_id=repo_id
        )
        
        direct_fields = _build_triplet_payload(
            direct_metadata_raw,
            source_suffix='_source',
            conflict_suffix='_conflicts'
        )

        complete_metadata = {
            "repo_id": repo_id or model_id,
            "model_id": model_id,
            "use_case": self.use_case,
            "direct_fields": direct_fields,
            "rag_fields": rag_fields
        }

        return complete_metadata


class DATABOMProcessor:
    """Processes datasets using either RAG or Direct LLM mode"""
    
    def __init__(self, model: str = "gpt-4o", mode: str = "rag", llm_provider: str = "openai", 
                 ollama_base_url: str = None, questions_config: Dict[str, Dict] = None, use_case: str = 'complete',
                 embedding_provider: str = "local", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize processing system based on mode
        
        Args:
            embedding_provider: 'local' (default, free HuggingFace) or 'openai'
            embedding_model: Model for local embeddings (only used if embedding_provider='local')
        """
        self.mode = mode
        self.model = model
        self.llm_provider = llm_provider
        self.ollama_base_url = ollama_base_url
        self.questions_config = questions_config if questions_config is not None else FIXED_QUESTIONS_DATA
        self.use_case = use_case
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        
        if mode == 'rag':
            self.processor = AgenticRAG(
                model=model,
                llm_provider=llm_provider,
                ollama_base_url=ollama_base_url,
                questions=self.questions_config,
                bom_type='data',
                embedding_provider=embedding_provider,
                embedding_model=embedding_model
            )
            print(f"✓ Initialized DATABOMProcessor in RAG mode with model: {model}")
        else:
            self.processor = DirectLLM(
                model=model,
                llm_provider=llm_provider,
                ollama_base_url=ollama_base_url,
                questions=self.questions_config,
                bom_type='data'
            )
            print(f"✓ Initialized DATABOMProcessor in DIRECT mode with model: {model}")
        
        # Initialize API clients
        gh_token = os.getenv("GITHUB_TOKEN")
        if gh_token:
            try:
                self.github_client = Github(gh_token)
            except Exception as exc:
                print(f"Warning: Github client unavailable ({exc})")
                self.github_client = None
        else:
            self.github_client = None

        hf_token = os.getenv("hug_token") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            try:
                self.hf_api = HfApi(token=hf_token)
            except Exception as exc:
                print(f"Warning: Hugging Face client unavailable ({exc})")
                self.hf_api = None
        else:
            self.hf_api = None

    
    def generate_dataset_id(self, arxiv_url: str, github_url: str, hf_url: str) -> str:
        """Generate a unique dataset identifier from URLs"""
        if hf_url and "huggingface.co" in str(hf_url):
            repo_id = MetadataFetcher.extract_repo_id_from_hf_url(hf_url)
            if repo_id:
                return repo_id.replace("/", "_")
        
        if github_url and "github.com" in str(github_url):
            try:
                repo_path = MetadataFetcher.extract_repo_path(github_url)
                return repo_path.replace("/", "_")
            except:
                pass
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"dataset_{timestamp}"
    
    def fetch_direct_metadata(self, github_url: str, hf_url: str) -> dict:
        """Perform direct metadata fetching from GitHub and Hugging Face"""
        direct_metadata = {}
        
        # GitHub Metadata
        github_metadata = {}
        if github_url and "github.com" in str(github_url):
            try:
                repo_path = MetadataFetcher.extract_repo_path(github_url)
                github_repo = self.github_client.get_repo(repo_path)
                github_metadata = MetadataFetcher.inspect_github_BOM_Fields(github_repo, bom_type='data')
            except Exception as e:
                print(f"Error fetching GitHub metadata: {e}")
        
        # Hugging Face Metadata
        huggingface_metadata = {}
        if hf_url:
            try:
                repo_id = MetadataFetcher.extract_repo_id_from_hf_url(hf_url)
                if repo_id:
                    info = self.hf_api.dataset_info(repo_id)
                    huggingface_metadata = MetadataFetcher.inspect_huggingface_BOM_Fields(info, bom_type='data')
            except Exception as e:
                print(f"Error fetching Hugging Face metadata: {e}")
        
        # Aggregate metadata using SourceHandler
        direct_metadata["builtTime"], direct_metadata["builtTime_source"] = SourceHandler.get_field(
            "builtTime", huggingface_metadata, github_metadata, mode='earliest'
        )
        
        direct_metadata["originatedBy"], direct_metadata["originatedBy_source"], direct_metadata["originatedBy_conflicts"] = SourceHandler.get_field_conflict(
            "originatedBy", huggingface_metadata, github_metadata
        )
        
        direct_metadata["releaseTime"], direct_metadata["releaseTime_source"] = SourceHandler.get_field(
            "releaseTime", huggingface_metadata, github_metadata, mode='latest'
        )
        
        direct_metadata["downloadLocation"], direct_metadata["downloadLocation_source"], direct_metadata["downloadLocation_conflicts"] = SourceHandler.get_field_conflict(
            "software_downloadLocation", huggingface_metadata, github_metadata
        )
        
        direct_metadata["primaryPurpose"], direct_metadata["primaryPurpose_source"], direct_metadata["primaryPurpose_conflicts"] = SourceHandler.get_field_conflict(
            "primaryPurpose", huggingface_metadata, github_metadata
        )
        
        direct_metadata["license"], direct_metadata["license_source"], direct_metadata["license_conflicts"] = SourceHandler.get_field_conflict(
            "license", huggingface_metadata, github_metadata
        )
        
        direct_metadata["sourceInfo"], direct_metadata["sourceInfo_source"], direct_metadata["sourceInfo_conflicts"] = SourceHandler.get_field_conflict(
            "sourceInfo", huggingface_metadata, github_metadata
        )
        
        direct_metadata["datasetAvailability"], direct_metadata["datasetAvailability_source"], direct_metadata["datasetAvailability_conflicts"] = SourceHandler.get_field_conflict(
            "datasetAvailability", huggingface_metadata, github_metadata
        )
        
        return direct_metadata
    
    def fetch_rag_metadata(self, dataset_id: str, arxiv_url: str, github_url: str, hf_url: str) -> dict:
        """Perform agentic RAG to extract detailed information from documents"""
        rag_results = self.processor.process_dataset(
            dataset_id=dataset_id,
            arxiv_url=arxiv_url,
            github_url=github_url,
            huggingface_url=hf_url
        )
        
        rag_metadata = {}
        for result in rag_results:
            question_type = result['question_type']
            answer = result['answer']
            sources_used = result.get('sources_used', [])
            conflicts = result.get('conflicts', 'No conflicts detected')
            
            rag_metadata[question_type] = answer
            rag_metadata[f"{question_type}_sources"] = ', '.join(sources_used) if sources_used else 'None'
            rag_metadata[f"{question_type}_conflicts"] = conflicts
        
        return rag_metadata
    
    def process_dataset(self, arxiv_url: str, github_url: str, hf_url: str) -> tuple:
        """Main processing function that combines direct fetching and RAG"""
        dataset_id = self.generate_dataset_id(arxiv_url, github_url, hf_url)
        
        # Step 1: Direct Metadata Fetching
        direct_metadata = self.fetch_direct_metadata(github_url, hf_url)
        
        # Step 2: Agentic RAG Processing
        rag_metadata = self.fetch_rag_metadata(dataset_id, arxiv_url, github_url, hf_url)
        
        # Combine all metadata
        complete_metadata = {
            "dataset_id": dataset_id,
            "use_case": self.use_case,
            "urls": {
                "arxiv": arxiv_url,
                "github": github_url,
                "huggingface": hf_url
            },
            "processing_timestamp": datetime.now().isoformat(),
            "direct_metadata": direct_metadata,
            "rag_metadata": rag_metadata
        }
        
        return complete_metadata
