"""
Agentic RAG Core Module - Global Top-K Retrieval with Conflict Detection
Modified to retrieve top 6 chunks globally and detect conflicts between sources
"""

import os
from typing import TypedDict, List, Dict, Tuple, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Ollama support - create LLM instance based on provider
def create_llm(model: str, temperature: float = 0, llm_provider: str = "openai", ollama_base_url: str = None):
    """
    Create an LLM instance based on the provider.
    
    Args:
        model: Model name (e.g., 'gpt-4o' for OpenAI, 'llama3:70b' for Ollama)
        temperature: Temperature for generation
        llm_provider: 'openai' or 'ollama'
        ollama_base_url: Base URL for Ollama server (defaults to env var or localhost)
    
    Returns:
        LLM instance compatible with LangChain
    """
    if llm_provider == "ollama":
        if ollama_base_url is None:
            ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://10.218.163.118:11434/v1/')
        # ChatOpenAI is compatible with Ollama's OpenAI-compatible API
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=ollama_base_url,
            api_key="ollama"  # Required but ignored by Ollama
        )
    else:
        # Default to OpenAI
        return ChatOpenAI(model=model, temperature=temperature)

# LangChain relocated some helpers into their own packages; prefer the current locations.
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document
import httpx
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dotenv import load_dotenv
import urllib3
import ssl
import certifi
from github import Github
from huggingface_hub import HfApi
import fitz  # PyMuPDF
import re

# Load environment variables
load_dotenv()

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Monkey patch httpx for insecure connections
orig_init = httpx.Client.__init__
def insecure_init(self, *args, **kwargs):
    kwargs["verify"] = False
    return orig_init(self, *args, **kwargs)
httpx.Client.__init__ = insecure_init


# FIXED AI MODEL QUESTIONS WITH PRE-DEFINED SOURCE PRIORITY
FIXED_QUESTIONS_AI = {
    'autonomyType': {
        'question': 'Does the system can perform a decision or action without human involvement or guidance (yes, no, or noAssertion)?',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'autonomous automation automated decision-making human involvement guidance manual intervention oversight supervision fully automated system self-governing independent unsupervised human-in-the-loop semi-autonomous user-controlled operator-assisted',
        'description': 'Indicates if the system is fully automated or a human is involved in any of the decisions of the AI system. yes: Indicates that the system is fully automated. no: Indicates that a human is involved in any of the decisions of the AI system. noAssertion: Makes no assertion about the autonomy.'
    },
    'domain': {
        'question': 'What is the domain in which the AI package can be used?',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'domain application area field sector industry vertical computer vision natural language processing NLP machine learning classification regression detection segmentation speech recognition image processing text analysis audio processing video analysis time series forecasting recommendation systems robotics healthcare finance automotive agriculture education retail manufacturing cybersecurity gaming entertainment',
        'description': 'A free-form text that describes the domain where the AI model contained in the AI software can be expected to operate successfully. Examples include computer vision, natural language processing, etc.'
    },
    'energyConsumption': {
        'question': 'Indicates the amount of energy consumption incurred by an AI model.',
        'priority': ['huggingface', 'arxiv', 'github'],
        'keywords': 'energy consumption power usage electricity computational resources GPU CPU training inference kWh joules watts',
        'description': 'Captures the energy consumption of an AI model, either known or estimated. In the absence of direct measurements, an SPDX data creator may choose to estimate the energy consumption based on information about computational resources (e.g., number of floating-point operations), training time, and other relevant training details.'
    },
    # 'energyQuantity': {
    #     'question': 'Represents the energy quantity.',
    #     'priority': ['github', 'huggingface', 'arxiv'],
    #     'keywords': 'energy quantity measurement value estimation calculation',
    #     'description': 'Provides the quantity information of the energy.'
    # },
    # 'energyUnit': {
    #     'question': 'Specifies the unit in which energy is measured.',
    #     'priority': ['arxiv', 'huggingface', 'github'],
    #     'keywords': 'energy unit measurement kWh kilowatt-hour kilowatt-hours kw-hr kwh joules joule J kilojoule kJ megajoule MJ GJ megawatt MW  british thermal unit BTU btu  calorie cal kcal kilocalorie foot-pound ftlb therm therms quad quadrillion electron-volt eV keV MeV GeV TeV newton-meter Nm horsepower-hour hph power consumption electricity electrical energy thermal energy mechanical energy unit of measurement units measurement scale metric imperial SI international system base unit derived unit energy efficiency power rating electrical consumption carbon footprint CO2 equivalent emissions per unit computational energy training energy inference energy GPU-hours CPU-hours TPU-hours floating point operations FLOPS energy per FLOP',
    #     'description': 'Provides the unit information of the energy.'
    # },
    # 'finetuningEnergyConsumption': {
    #     'question': 'Specifies the amount of energy consumed when finetuning the AI model that is being used in the AI system.',
    #     'priority': ['huggingface', 'arxiv', 'github'],
    #     'keywords': 'finetuning energy consumption power usage computational resources',
    #     'description': 'The field specifies the amount of energy consumed when finetuning the AI model that is being used in the AI system.'
    # },
    'hyperparameter': {
        'question': 'Records a hyperparameter used to build the AI model contained in the AI package.',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'hyperparameter learning rate batch size layers tuning settings',
        'description': 'Records a hyperparameter value. Hyperparameters are settings defined before the training process that control the learning algorithms behavior. They differ from model parameters, which are learned from the data during training. Developers typically set hyperparameters manually or through a process of hyperparameter tuning (also known as trial and error). Examples of hyperparameters include learning rate, batch size, and the number of layers in a neural network.'
    },
    # 'inferenceEnergyConsumption': {
    #     'question': 'Provides relevant information about the AI software, not including the model description.',
    #     'priority': ['huggingface', 'github', 'arxiv'],
    #     'keywords': 'inference energy consumption runtime power usage efficiency',
    #     'description': 'The field specifies the amount of energy consumed during inference time by an AI model that is being used in the AI system.'
    # },
    'informationAboutApplication': {
        'question': 'Provides relevant information about the AI software, not including the model description.',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'application functionality pre-processing APIs dependencies',
        'description': 'A free-form text description of how the AI model is used within the software. It should include any relevant information, such as pre-processing steps, third-party APIs, and other pertinent details. It can also include: Functionality provided by the AI model within the software application, including: any specific tasks or decisions it is designed to perform; any pre-processing steps that are applied to the input data before it is fed into the AI model for inference, such as data cleaning, normalization, or feature extraction; and any third-party APIs or services that are used in conjunction with the AI model, such as data sources, cloud services, or other AI models. Description of any dependencies or requirements needed to run the AI model within the software application, including: specific hardware, software libraries, and operating systems.'
    },
    'informationAboutTraining': {
        'question': 'Describes relevant information about different steps of the training process.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'training process data algorithms techniques evaluation metrics',
        'description': 'A detailed explanation of the training process, including the specific techniques, algorithms, and methods employed. Examples include: training data used to train the AI model, along with any relevant details about its source, quality, and pre-processing steps; specific training algorithms employed, including stochastic gradient descent, backpropagation, and reinforcement learning; specific training techniques used to improve the performance or accuracy of the AI model, such as transfer learning, fine-tuning, or active learning; and any evaluation metrics used to assess the performance of the AI model during the training process, including accuracy, precision, recall, and F1 score.'
    },
    'limitation': {
        'question': 'Captures a limitation of the AI software.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'limitations constraints challenges restrictions',
        'description': 'A free-form text that captures a limitation of the AI package (or of the AI models present in the AI package). Note that this is not guaranteed to be exhaustive. For instance, a limitation might be that the AI package cannot be used on datasets from a certain demography.'
    },
    'metric': {
        'question': 'Records the measurement of prediction quality of the AI model.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'metrics evaluation accuracy precision recall F1-score',
        'description': 'Records the measurement with which the AI model was evaluated. This makes statements about the prediction quality including uncertainty, accuracy, characteristics of the tested population, quality, fairness, explainability, robustness etc.'
    },
    'metricDecisionThreshold': {
        'question': 'Captures the threshold that was used for computation of a metric described in the metric field.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'decision threshold metric computation evaluation',
        'description': 'Each metric might be computed based on a decision threshold. For instance, precision or recall is typically computed by checking if the probability of the outcome is larger than 0.5. Each decision threshold should match with a metric field defined in the AI package.'
    },
    'modelDataPreprocessing': {
        'question': 'Describes all the preprocessing steps applied to the training data before the model training.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'data preprocessing cleaning normalization transformation',
        'description': 'A free-form text that describes the preprocessing steps applied to the training data before training of the model(s) contained in the AI software.'
    },
    'modelExplainability': {
        'question': 'Describes methods that can be used to explain the results from the AI model.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'explainability interpretability model results explanation',
        'description': 'A free-form text that lists the different explainability mechanisms and how they can be used to explain the results from the AI model.'
    },
    'safetyRiskAssessment': {
        'question': 'Records the results of general safety risk assessment of the AI system.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'safety risk assessment compliance evaluation',
        'description': 'Records the results of general safety risk assessment of the AI system. Using categorization according to the EU general risk assessment methodology. The methodology implements Article 20 of Regulation (EC) No 765/2008 and is intended to assist authorities when they assess general product safety compliance. It is important to note that this categorization differs from the one proposed in the EU AI Acts provisional agreement.'
    },
    'standardCompliance': {
        'question': 'Captures a standard that is being complied with.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'standards compliance ISO IEEE ETSI regulations',
        'description': 'A free-form text that captures a standard that the AI software complies with. This includes both published and unpublished standards, such as those developed by ISO, IEEE, and ETSI. The standard may, but is not necessarily required to, satisfy a legal or regulatory requirement.'
    },
    # 'trainingEnergyConsumption': {
    #     'question': 'Specifies the amount of energy consumed when training the AI model that is being used in the AI system.',
    #     'priority': ['arxiv', 'github', 'huggingface'],
    #     'keywords': 'training energy consumption computational resources power usage',
    #     'description': 'The field specifies the amount of energy consumed when training the AI model that is being used in the AI system.'
    # },
    'typeOfModel': {
        'question': 'Records the type of the model used in the AI software.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'model type supervised unsupervised reinforcement learning',
        'description': 'A free-form text that records the type of the AI model(s) used in the software. For instance, if it is a supervised model, unsupervised model, reinforcement learning model or a combination of those.'
    },
    'useSensitivePersonalInformation': {
        'question': 'Records if sensitive personal information is used during model training or could be used during the inference.',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'sensitive personal information PII privacy data protection',
        'description': 'Notes if sensitive personal information is used in the training or inference of the AI models. This might include biometric data, addresses or other data that can be used to infer a persons identity.'
    }
}

# FIXED DATASET QUESTIONS WITH PRE-DEFINED SOURCE PRIORITY
FIXED_QUESTIONS_DATA = {
    'anonymizationMethodUsed': {
        'question': 'What is the anonymization methods used in this dataset?',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'anonymization anonymous anonymized privacy protection de-identification pseudonymization masking',
        'description': 'A free-form text that describes the methods used to anonymize the dataset or fields in the dataset.'
    },
    'confidentialityLevel': {
        'question': 'What is the confidentiality level of the data points contained in the dataset?',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'confidential confidentiality sensitive privacy classification level public private restricted',
        'description': 'Describes the levels of confidentiality of the data points contained in the dataset.'
    },
    'dataCollectionProcess': {
        'question': 'How the dataset was collected?',
        'priority': ['huggingface', 'arxiv', 'github'],
        'keywords': 'collected collection gathering scraping crawling sources method methodology acquisition obtained compiled',
        'description': 'A free-form text that describes how a dataset was collected.'
    },
    'dataPreprocessing': {
        'question': 'What is the preprocessing steps that were applied to the raw data to create the given dataset.',
        'priority': ['github', 'huggingface', 'arxiv'],
        'keywords': 'preprocessing cleaning normalization standardization tokenization filtering transformation deduplication removal',
        'description': 'A free-form text that describes the various preprocessing steps applied to the raw data.'
    },
    'datasetAvailability': {
        'question': 'Does the dataset publicaly available or not?',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'available availability download access public private registration clickthrough restricted open',
        'description': 'Describes the dataset availability from accessibility perspective.'
    },
    'datasetNoise': {
        'question': 'What is the potentially noisy elements of the dataset?',
        'priority': ['huggingface', 'arxiv', 'github'],
        'keywords': 'noise noisy quality errors artifacts outliers inaccuracies inconsistencies corrupted',
        'description': 'Describes what kinds of noises a dataset might encompass.'
    },
    'datasetSize': {
        'question': 'What is the size of the dataset.',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'size samples examples instances records entries GB MB TB bytes volume count number',
        'description': 'Captures how large a dataset is, measured in bytes.'
    },
    'datasetType': {
        'question': 'What is the type of the given dataset.',
        'priority': ['huggingface', 'github', 'arxiv'],
        'keywords': 'type format datatype image text audio video multimodal tabular structured unstructured',
        'description': 'Describes the datatype contained in the dataset.'
    },
    'datasetUpdateMechanism': {
        'question': 'What is the mechanism to update the dataset.',
        'priority': ['arxiv', 'huggingface', 'github'],
        'keywords': 'update updated updating version maintenance versioning refresh dynamic static mechanism',
        'description': 'A free-form text that describes a mechanism to update the dataset.'
    },
    'hasSensitivePersonalInformation': {
        'question': 'Does any sensitive personal information is present in the dataset?',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'sensitive personal information PII privacy identifying identity protected health financial',
        'description': 'Indicates the presence of sensitive personal data or information.'
    },
    'intendedUse': {
        'question': 'For what the given dataset should be used for?',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'intended use purpose application task designed goal objective research training evaluation',
        'description': 'A free-form text that describes what the given dataset should be used for.'
    },
    'knownBias': {
        'question': 'What is the biases that the dataset is known to encompass?',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'bias biased fairness demographic representation imbalance skewed prejudice limitation',
        'description': 'A free-form text that describes the different biases that the dataset encompasses.'
    },
    'sensorUsed': {
        'question': 'What is the sensor used for collecting the data?',
        'priority': ['arxiv', 'github', 'huggingface'],
        'keywords': 'sensor device instrument equipment camera microphone accelerometer calibration hardware',
        'description': 'Describes a sensor that was used for collecting the data and its calibration value.'
    }
}

# Unified function to get questions based on BOM type
def get_fixed_questions(bom_type='ai'):
    """Get FIXED_QUESTIONS based on BOM type"""
    if bom_type.lower() == 'data':
        return FIXED_QUESTIONS_DATA
    return FIXED_QUESTIONS_AI

# Default to AI for backward compatibility
FIXED_QUESTIONS = FIXED_QUESTIONS_AI

# Extract questions and create reverse lookup
QUESTIONS_LIST = [config['question'] for config in FIXED_QUESTIONS.values()]
QUESTION_TO_KEY = {config['question']: key for key, config in FIXED_QUESTIONS.items()}


class AgentState(TypedDict):
    """State management for the RAG workflow"""
    question: str
    documents: List[Document]
    answer: str
    source_priority: List[str]
    sources_used: List[str]
    question_type: str
    row_index: int
    all_results: List[Dict]
    row_retrievers: Dict
    available_sources: List[str]
    conflicts: str


class HeaderAwareTextSplitter:
    """Custom text splitter that respects markdown headers and document structure"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Header patterns in order of priority (H1 > H2 > H3 > etc.)
        self.header_patterns = [
            r'^# .+$',      # H1
            r'^## .+$',     # H2  
            r'^### .+$',    # H3
            r'^#### .+$',   # H4
            r'^##### .+$',  # H5
            r'^###### .+$'  # H6
        ]
    
    def split_text(self, text: str) -> List[str]:
        """Split text based on headers while maintaining context"""
        if not text:
            return []
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        header_context = []  # Stack to maintain header hierarchy
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # Check if line is a header
            header_level = self._get_header_level(line)
            
            if header_level is not None:
                # Update header context - remove headers at same or lower level
                header_context = [h for h in header_context if h['level'] < header_level]
                header_context.append({'level': header_level, 'text': line})
                
                # If current chunk is getting large and we hit a header, finalize it
                if current_size > self.chunk_size * 0.7 and current_chunk:
                    chunk_text = self._build_chunk_with_context(current_chunk, header_context[:-1])
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_size = 0
            
            # Add line to current chunk
            current_chunk.append(line)
            current_size += line_size
            
            # If chunk exceeds size limit, try to split at a good point
            if current_size > self.chunk_size:
                # Look for a good split point (paragraph break, sentence end, etc.)
                split_point = self._find_split_point(current_chunk)
                
                if split_point > 0:
                    # Create chunk up to split point
                    chunk_lines = current_chunk[:split_point]
                    chunk_text = self._build_chunk_with_context(chunk_lines, header_context)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, split_point - self._calculate_overlap_lines(current_chunk[split_point:]))
                    current_chunk = current_chunk[overlap_start:]
                    current_size = sum(len(line) + 1 for line in current_chunk)
                else:
                    # Force split if no good point found
                    chunk_text = self._build_chunk_with_context(current_chunk[:-1], header_context)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = current_chunk[-1:]
                    current_size = len(current_chunk[0]) + 1
        
        # Add remaining content
        if current_chunk:
            chunk_text = self._build_chunk_with_context(current_chunk, header_context)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(chunk_text)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _get_header_level(self, line: str) -> Optional[int]:
        """Determine if line is a header and return its level (1-6)"""
        line = line.strip()
        for i, pattern in enumerate(self.header_patterns):
            if re.match(pattern, line):
                return i + 1
        return None
    
    def _build_chunk_with_context(self, chunk_lines: List[str], header_context: List[Dict]) -> str:
        """Build chunk with relevant header context"""
        context_lines = []
        
        # Add header context (but not if the chunk already starts with headers)
        if header_context and chunk_lines:
            first_line = chunk_lines[0].strip()
            if not any(re.match(pattern, first_line) for pattern in self.header_patterns):
                # Add relevant headers for context
                for header in header_context[-2:]:  # Last 2 levels of context
                    context_lines.append(header['text'])
                context_lines.append("")  # Empty line after headers
        
        # Add the actual chunk content
        context_lines.extend(chunk_lines)
        
        return '\n'.join(context_lines)
    
    def _find_split_point(self, lines: List[str]) -> int:
        """Find a good point to split the chunk"""
        # Look for split points in reverse order (prefer later splits)
        for i in range(len(lines) - 1, max(0, len(lines) - 10), -1):
            line = lines[i].strip()
            
            # Good split points in order of preference
            if not line:  # Empty line (paragraph break)
                return i + 1
            elif line.endswith('.') or line.endswith('!') or line.endswith('?'):  # Sentence end
                return i + 1
            elif self._get_header_level(line) is not None:  # Header
                return i
        
        # If no good split point found, split at 80% of current position
        return int(len(lines) * 0.8)
    
    def _calculate_overlap_lines(self, remaining_lines: List[str]) -> int:
        """Calculate how many lines to overlap based on chunk_overlap setting"""
        if not remaining_lines:
            return 0
        
        overlap_chars = 0
        overlap_lines = 0
        
        for line in remaining_lines:
            if overlap_chars >= self.chunk_overlap:
                break
            overlap_chars += len(line) + 1
            overlap_lines += 1
        
        return min(overlap_lines, len(remaining_lines) // 2)

class AgenticRAG:
    """Main RAG class that handles document processing and question answering"""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0, llm_provider: str = "openai", ollama_base_url: str = None, questions: Optional[Dict[str, Dict]] = None, bom_type: str = 'ai', embedding_provider: str = "local", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the RAG system with LLM and embeddings
        
        Args:
            model: Model name (e.g., 'gpt-4o' for OpenAI, 'llama3:70b' for Ollama)
            temperature: Temperature for generation
            llm_provider: 'openai' or 'ollama'
            ollama_base_url: Base URL for Ollama server (only used if llm_provider='ollama')
            questions: Optional subset of FIXED_QUESTIONS to answer in this run
            bom_type: 'ai' or 'data' - determines which question set to use if questions not provided
            embedding_provider: 'local' (default, HuggingFace) or 'openai' - provider for embeddings
            embedding_model: Model name for embeddings (default: 'sentence-transformers/all-MiniLM-L6-v2')
                            Other good options: 'sentence-transformers/all-mpnet-base-v2' (higher quality),
                                                'BAAI/bge-small-en-v1.5' (good balance)
        """
        self.llm = create_llm(model, temperature, llm_provider, ollama_base_url)
        
        # Initialize embeddings based on provider
        if embedding_provider == "local":
            print(f"✓ Using LOCAL embeddings: {embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},  # Use 'cuda' if you have GPU
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            print(f"✓ Using OpenAI embeddings (requires OPENAI_API_KEY)")
            self.embeddings = OpenAIEmbeddings()
        
        self.g = Github(os.environ.get("GITHUB_TOKEN"))
        self.hf_api = HfApi(token=os.environ.get("hug_token"))
        self.bom_type = bom_type.lower()
        if questions is None:
            questions = get_fixed_questions(self.bom_type)
        self.questions = questions
        self.workflow = self._build_workflow()
        provider_info = f"Ollama ({ollama_base_url or 'default'})" if llm_provider == "ollama" else "OpenAI"
        print(f"✓ Initialized AgenticRAG with model: {model} (provider: {provider_info}, BOM type: {self.bom_type})")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        workflow.add_node("global_retrieve_top_k", self._global_retrieve_top_k)
        workflow.add_node("generate", self._generate_answer)
        workflow.set_entry_point("global_retrieve_top_k")
        workflow.add_edge("global_retrieve_top_k", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def _normalize_markdown(self, content: str, source: str) -> str:
        """Ensure all source content is normalized as markdown"""
        if not content:
            return ""
        normalized = content.replace("\r\n", "\n").strip()
        if not normalized:
            return ""
        if normalized.lstrip().startswith("#"):
            return normalized
        header = f"# {source.capitalize()} Source\n\n"
        return f"{header}{normalized}"

    def extract_repo_path(self, github_url: str) -> str:
        """Extract 'owner/repo' from a full GitHub URL"""
        parsed = urlparse(github_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        raise ValueError(f"Invalid GitHub repository URL: {github_url}")
    
    def extract_repo_id_from_hf_url(self, url: str) -> Optional[str]:
        """Extract HuggingFace repo ID from URL"""
        if not url or not isinstance(url, str):
            return None
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if "datasets" in parts:
            idx = parts.index("datasets")
            if len(parts) > idx + 2:
                return f"{parts[idx + 1]}/{parts[idx + 2]}"
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    
    def fetch_github_readme(self, github_url: str) -> str:
        """Fetch README from GitHub repository"""
        try:
            repo_path = self.extract_repo_path(github_url)
            repo = self.g.get_repo(repo_path)
            readme_content = repo.get_readme().decoded_content.decode('utf-8')
            print(f"  ✓ Fetched README from GitHub repo: {repo_path}")
            return readme_content
        except Exception as e:
            print(f"  ⚠️ Error fetching GitHub README: {e}")
            return ""
    
    # def fetch_huggingface_readme(self, hf_url: str) -> str:
    #     """Fetch README from HuggingFace dataset"""
    #     try:
    #         repo_id = self.extract_repo_id_from_hf_url(hf_url)
    #         if not repo_id:
    #             return ""
    #         info = self.hf_api.dataset_info(repo_id)
    #         readme_content = info.cardData.get('readme', '')
    #         print(f"  ✓ Fetched README from HuggingFace: {repo_id}")
    #         return readme_content
    #     except Exception as e:
    #         print(f"  ⚠️ Error fetching HuggingFace README: {e}")
    #         return ""
    

    def fetch_huggingface_readme(self, hf_url: str) -> str:
        try:
            repo_id = self.extract_repo_id_from_hf_url(hf_url)
            if not repo_id:
                print(f"  ⚠️ Could not extract repo_id from HuggingFace URL: {hf_url}")
                return ""
            
            headers = {}
            if getattr(self, "hf_token", None):
                headers["Authorization"] = f"Bearer {self.hf_token}"
            
            # Try model URL first (most common for AI models)
            url_model = f"https://huggingface.co/{repo_id}/raw/main/README.md"
            try:
                r = requests.get(url_model, headers=headers, timeout=30, verify=False)
                r.raise_for_status()
                print(f"  ✓ Fetched README from HuggingFace model: {repo_id}")
                return r.text
            except Exception as model_error:
                # Try dataset URL as fallback
                url_dataset = f"https://huggingface.co/datasets/{repo_id}/raw/main/README.md"
                try:
                    r = requests.get(url_dataset, headers=headers, timeout=30, verify=False)
                    r.raise_for_status()
                    print(f"  ✓ Fetched README from HuggingFace dataset: {repo_id}")
                    return r.text
                except Exception as dataset_error:
                    print(f"  ⚠️ Failed to fetch README from both model and dataset URLs:")
                    print(f"      Model error: {model_error}")
                    print(f"      Dataset error: {dataset_error}")
                    return ""
        except Exception as e:
            print(f"  ⚠️ Error in fetch_huggingface_readme: {e}")
            return ""




    def fetch_arxiv_pdf_text(self, arxiv_url: str) -> str:
        """Fetch and extract text from arXiv PDF using PyMuPDF with markdown conversion"""
        try:
            arxiv_id = arxiv_url.split('/')[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(pdf_url, verify=False, timeout=30)
            
            if response.status_code == 200:
                temp_pdf_path = "temp.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(response.content)
                
                # Use PyMuPDF to extract text and convert to markdown
                markdown_content = self._pdf_to_markdown(temp_pdf_path)
                
                # Clean up temporary file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                
                print(f"  ✓ Fetched and converted arXiv paper to markdown ({len(markdown_content)} chars)")
                return markdown_content
            else:
                print(f"  ⚠️ Failed to download PDF from {pdf_url}")
                return ""
        except Exception as e:
            print(f"  ⚠️ Error fetching arXiv content: {e}")
            # Clean up temporary file in case of error
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
            return ""
    
    def _pdf_to_markdown(self, pdf_path: str) -> str:
        """Convert PDF to markdown using PyMuPDF with structure preservation"""
        try:
            doc = fitz.open(pdf_path)
            markdown_content = []
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get text blocks with formatting information
                blocks = page.get_text("dict")["blocks"]
                
                page_content = []
                current_section = None
                
                for block in blocks:
                    if "lines" in block:  # Text block
                        block_text = ""
                        block_font_size = 0
                        block_flags = 0
                        
                        # Extract text and formatting from block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text += text + " "
                                    # Track largest font size and flags in block
                                    if span["size"] > block_font_size:
                                        block_font_size = span["size"]
                                        block_flags = span["flags"]
                            
                            if line_text.strip():
                                block_text += line_text.strip() + "\n"
                        
                        block_text = block_text.strip()
                        if not block_text:
                            continue
                        
                        # Determine if this is likely a header based on formatting
                        formatted_text = self._format_text_block(block_text, block_font_size, block_flags)
                        page_content.append(formatted_text)
                
                # Add page content
                if page_content:
                    if page_num == 0:
                        markdown_content.append(f"# ArXiv Paper Content\n")
                    else:
                        markdown_content.append(f"\n## Page {page_num + 1}\n")
                    
                    markdown_content.extend(page_content)
                    markdown_content.append("")  # Empty line between pages
            
            doc.close()
            
            # Clean up and structure the markdown
            final_content = "\n".join(markdown_content)
            final_content = self._clean_markdown_content(final_content)
            
            return final_content
        
        except Exception as e:
            print(f"  ⚠️ Error converting PDF to markdown: {e}")
            return ""
    
    def _format_text_block(self, text: str, font_size: float, font_flags: int) -> str:
        """Format text block based on font properties"""
        # Font flags: 16=bold, 2=italic, 20=bold+italic
        is_bold = font_flags & 16
        is_italic = font_flags & 2
        
        # Heuristics for header detection
        if len(text) < 100 and (is_bold or font_size > 12):
            # Likely a header - determine level based on font size and content
            if font_size > 16 or any(keyword in text.lower() for keyword in ['abstract', 'introduction', 'conclusion']):
                return f"## {text}\n"
            elif font_size > 14 or is_bold:
                return f"### {text}\n"
            else:
                return f"#### {text}\n"
        else:
            # Regular paragraph text
            formatted_text = text
            if is_bold and len(text) < 200:
                formatted_text = f"**{text}**"
            elif is_italic and len(text) < 200:
                formatted_text = f"*{text}*"
            
            return formatted_text + "\n"
    
    def _clean_markdown_content(self, content: str) -> str:
        """Clean and structure markdown content"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip very short lines that are likely artifacts
            if len(line) < 3:
                if not line:  # Keep empty lines for structure
                    cleaned_lines.append(line)
                continue
            
            # Clean up common PDF artifacts
            line = re.sub(r'\s+', ' ', line)  # Multiple spaces to single space
            line = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\{\}\"\'\/\\\#\*\_\`\|\~\@\#\$\%\^\&\+\=]', '', line)
            
            # Skip lines that are mostly numbers (page numbers, etc.)
            if re.match(r'^\d+[\s\d]*$', line):
                continue
            
            cleaned_lines.append(line)
        
        # Remove excessive empty lines
        final_lines = []
        empty_count = 0
        
        for line in cleaned_lines:
            if not line.strip():
                empty_count += 1
                if empty_count <= 2:  # Max 2 consecutive empty lines
                    final_lines.append(line)
            else:
                empty_count = 0
                final_lines.append(line)
        
        return '\n'.join(final_lines)
    
    def create_vector_stores(self, content_dict: Dict[str, str]) -> Dict[str, any]:
        """Create vector stores for each source with header-aware chunking"""
        retrievers = {}
        
        # Use header-aware text splitter for better document structure
        text_splitter = HeaderAwareTextSplitter(
            chunk_size=1200,  # Slightly larger chunks for better context
            chunk_overlap=300,  # More overlap to preserve context across chunks
            min_chunk_size=150  # Minimum viable chunk size
        )
        
        for source, content in content_dict.items():
            if not content:
                print(f"    ⚠️ Skipping {source}: Content is empty or None")
                continue
            if len(content) <= 50:
                print(f"    ⚠️ Skipping {source}: Content too short ({len(content)} chars)")
                continue
                
            normalized_content = self._normalize_markdown(content, source)
            if normalized_content and len(normalized_content) > 50:
                # Split content into header-aware chunks
                chunks = text_splitter.split_text(normalized_content)
                
                if not chunks:
                    print(f"    ⚠️ Skipping {source}: No chunks created after splitting")
                    continue
                
                docs = []
                for i, chunk in enumerate(chunks):
                    # Extract header information for metadata
                    header_info = self._extract_header_info(chunk)
                    
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": source,
                            "format": "markdown",
                            "chunk_index": i,
                            "headers": header_info,
                            "chunk_size": len(chunk)
                        }
                    )
                    docs.append(doc)
                
                vectorstore = FAISS.from_documents(docs, self.embeddings)
                retrievers[source] = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                )
                print(f"    Created retriever for {source}: {len(docs)} header-aware chunks")
        
        return retrievers
    
    def _extract_header_info(self, chunk: str) -> List[str]:
        """Extract header information from a chunk for metadata"""
        headers = []
        lines = chunk.split('\n')
        
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if re.match(r'^#{1,6}\s+.+$', line):
                headers.append(line)
        
        return headers
    
    def get_question_priority(self, question: str) -> Tuple[str, List[str], str]:
        """Get pre-defined priority and keywords for a question"""
        question_key = QUESTION_TO_KEY.get(question)
        if question_key:
            config = self.questions.get(question_key) or FIXED_QUESTIONS.get(question_key)
            if config:
                return question_key, config['priority'], config['keywords']
        return 'unknown', ['huggingface', 'arxiv', 'github'], question
    
    def _global_retrieve_top_k(self, state: AgentState) -> AgentState:
        """
        Retrieve top K chunks GLOBALLY across all sources based on similarity scores.
        Uses question + description + keywords for comprehensive retrieval.
        Retrieves 20 candidates from each source, then selects top 12 globally.
        Returns the top 12 most similar chunks regardless of source.
        """
        question = state["question"]
        TOP_K_PER_SOURCE = 20  # Number of candidates to retrieve from each source
        FINAL_TOP_K = 12  # Final number of chunks to select globally
        MIN_CHUNK_LENGTH = 100  # Minimum characters for a chunk to be valid
        
        # Get retrievers for current row
        row_retrievers = state.get("row_retrievers", {})
        available_sources = list(row_retrievers.keys())
        
        print(f"\n  📊 Available sources: {available_sources}")
        
        # Get the question type, enhanced keywords, and description
        question_type, _, enhanced_keywords = self.get_question_priority(question)
        question_config = self.questions.get(question_type, FIXED_QUESTIONS.get(question_type, {}))
        description = question_config.get('description', '')
        
        print(f"  🎯 Question type: {question_type}")
        print(f"  📝 Original question: {question[:80]}...")
        print(f"  📋 Description: {description[:80]}...")
        print(f"  🔑 Enhanced keywords: {enhanced_keywords[:80]}...")
        
        # IMPROVED: Combine question, description, and keywords for comprehensive retrieval
        # This gives: semantic understanding + context + keyword coverage
        combined_query = f"{question} {description} {enhanced_keywords}"
        print(f"  🔍 Combined query length: {len(combined_query)} characters")
        
        # Collect ALL chunks from ALL sources with their similarity scores
        all_chunks_with_scores = []
        
        print(f"\n  🔎 Retrieving {TOP_K_PER_SOURCE} candidates from each source using combined query...")
        
        for source in available_sources:
            try:
                print(f"  📖 Retrieving from {source}...")
                
                # Get the underlying vectorstore from the retriever
                vectorstore = row_retrievers[source].vectorstore
                
                # Retrieve top 20 candidates from this source
                docs_with_scores = vectorstore.similarity_search_with_score(
                    combined_query,
                    k=TOP_K_PER_SOURCE  # Get 20 candidates from each source
                )
                
                if not docs_with_scores:
                    print(f"    ⚠️ No documents retrieved from {source}")
                    continue
                
                # Convert FAISS distances to similarity scores (0-1 scale, higher=better)
                # FAISS returns L2 distance, so smaller distance = more similar
                for doc, distance in docs_with_scores:
                    similarity = 1 / (1 + distance)
                    chunk_length = len(doc.page_content.strip())
                    
                    # Only include chunks that meet minimum length requirement
                    if chunk_length >= MIN_CHUNK_LENGTH:
                        all_chunks_with_scores.append({
                            'document': doc,
                            'similarity': similarity,
                            'source': source,
                            'length': chunk_length
                        })
                
                print(f"    ✓ Retrieved {len([c for c in all_chunks_with_scores if c['source'] == source])} valid chunks")
                
            except Exception as e:
                print(f"    ✗ Error retrieving from {source}: {e}")
        
        # Sort all chunks by similarity score (highest first)
        all_chunks_with_scores.sort(key=lambda x: x['similarity'], reverse=True)
        
        print(f"\n  📦 Total chunks collected across all sources: {len(all_chunks_with_scores)}")
        
        # Select top K chunks
        top_k_chunks = all_chunks_with_scores[:FINAL_TOP_K]
        
        # Extract documents and track sources used
        selected_documents = []
        sources_used = []
        
        print(f"\n  🏆 Top {FINAL_TOP_K} chunks selected:")
        for i, chunk_info in enumerate(top_k_chunks, 1):
            doc = chunk_info['document']
            similarity = chunk_info['similarity']
            source = chunk_info['source']
            length = chunk_info['length']
            
            selected_documents.append(doc)
            
            if source not in sources_used:
                sources_used.append(source)
            
            # Show preview of content
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"    {i}. {source:12s} | Sim: {similarity:.4f} | Len: {length:4d} | Preview: {preview}...")
        
        # Count chunks per source
        source_counts = {}
        for chunk_info in top_k_chunks:
            source = chunk_info['source']
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print(f"\n  📈 Chunks per source in top {FINAL_TOP_K}:")
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {source}: {count} chunks")
        
        return {
            **state, 
            "documents": selected_documents,
            "sources_used": sources_used,
            "source_priority": available_sources,  # Not used for retrieval anymore
            "question_type": question_type
        }

    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate an answer using retrieved documents and detect conflicts between sources"""
        question = state["question"]
        documents = state.get("documents", [])
        sources_used = state.get("sources_used", [])
        question_type = state.get("question_type")
        
        if documents:
            # Group documents by source
            source_contexts = {}
            for doc in documents:
                source = doc.metadata.get('source', 'unknown')
                if source not in source_contexts:
                    source_contexts[source] = []
                source_contexts[source].append(doc.page_content)
            
            # Build context with source labels
            context_parts = []
            for source in sources_used:
                if source in source_contexts:
                    context_parts.append(f"\n{'='*60}")
                    context_parts.append(f"SOURCE: {source.upper()}")
                    context_parts.append(f"{'='*60}")
                    for content in source_contexts[source]:
                        context_parts.append(content.strip())
                        context_parts.append("")
            
            context = "\n".join(context_parts)

            question_config = self.questions.get(question_type, FIXED_QUESTIONS.get(question_type, {}))
            field_name = question_type
            question_summary = question_config.get('question', question)
            field_description = question_config.get('description', '')
            
            prompt = f"""You are an AI model documentation expert. Your task is to extract specific information about an AI model or package from the provided sources AND detect any conflicts between sources.

                You are looking for information related to the following field:

                FIELD_NAME: {field_name}
                QUESTION_SUMMARY: {question_summary}
                FIELD_DESCRIPTION: {field_description}

                AVAILABLE SOURCES:
                {context}

                INSTRUCTIONS:
                1. Carefully read through ALL the provided source materials.
                2. Look for information that directly answers the question based on the field name and description.
                3. CONFLICT DETECTION - THIS IS CRITICAL:
                - If multiple sources provide information about the same aspect but with DIFFERENT or CONTRADICTORY details, this is a CONFLICT.
                - For example:
                    * If arxiv says "the model is a transformer" but github says "the model is a CNN" → CONFLICT
                    * If huggingface says "trained on ImageNet" but arxiv says "trained on COCO" → CONFLICT
                    * If one source says "learning rate is 0.01" but another says "learning rate is 0.001" → CONFLICT
                - Pay attention to: different methods, different values, contradictory statements, incompatible descriptions.
                - Minor differences in wording are NOT conflicts if they describe the same thing.

                4. If you find the answer:
                - Provide a clear, specific, and detailed response.
                - If information comes from multiple sources and they AGREE, synthesize them.
                - If sources DISAGREE or provide CONFLICTING information, note this explicitly in your answer.
                - Mention which source(s) contained each piece of information.

                5. If information is partially available:
                - Provide what you found.
                - Clearly state what information is missing.

                6. If no relevant information is found:
                - State: "Not found."
                - DO NOT make up information.

                YOUR RESPONSE MUST BE IN THIS EXACT FORMAT:

                ANSWER: [Your detailed answer here, incorporating information from all sources]

                CONFLICT_STATUS: [Either "No conflicts detected" OR "CONFLICT DETECTED: [describe the specific conflict between sources]"]

                IMPORTANT: 
                - Be thorough - the information might be phrased differently than the question.
                - Look for synonyms and related concepts.
                - Be VERY CAREFUL to identify genuine conflicts where sources contradict each other.
                - Do not report conflicts for information that is complementary or describes different aspects.

                RESPONSE:"""
            
            print(f"\n  📝 Context length: {len(context)} characters")
            print(f"  📚 Sources in context: {list(source_contexts.keys())}")
        else:
            prompt = f"""You are analyzing information about an AI model, but no relevant source documents were retrieved.

Question: {question}

YOUR RESPONSE MUST BE IN THIS EXACT FORMAT:

ANSWER: Not found.

CONFLICT_STATUS: No conflicts detected

RESPONSE:"""
        
        response = self.llm.invoke(prompt)
        response_text = response.content
        
        # Parse the response to extract answer and conflict status
        answer = ""
        conflicts = "No conflicts detected"
        
        # Split by the format markers
        parts = response_text.split("CONFLICT_STATUS:")
        
        if len(parts) == 2:
            # Extract answer (remove "ANSWER:" prefix if present)
            answer_part = parts[0].strip()
            if answer_part.startswith("ANSWER:"):
                answer = answer_part[7:].strip()
            else:
                answer = answer_part
            
            # Extract conflict status
            conflicts = parts[1].strip()
        else:
            # Fallback if format is not followed
            answer = response_text
            conflicts = "No conflicts detected"
        
        print(f"  ⚠️  Conflicts: {conflicts[:100]}...")
        
        return {**state, "answer": answer, "conflicts": conflicts}
    
    def process_ai_model(self, repo_id: str, arxiv_url: str, github_url: str, huggingface_url: str) -> List[Dict]:
        """Process a single AI model and answer all questions"""
        return self.process(repo_id, arxiv_url, github_url, huggingface_url, 'ai')
    
    def process_dataset(self, dataset_id: str, arxiv_url: str, github_url: str, huggingface_url: str) -> List[Dict]:
        """Process a single dataset and answer all questions"""
        return self.process(dataset_id, arxiv_url, github_url, huggingface_url, 'data')
    
    def process(self, item_id: str, arxiv_url: str, github_url: str, huggingface_url: str, item_type: str = None) -> List[Dict]:
        """Unified process method that handles both AI models and datasets"""
        if item_type is None:
            item_type = self.bom_type
        item_type = item_type.lower()
        
        item_label = "AI MODEL" if item_type == 'ai' else "DATASET"
        print(f"\n{'='*70}")
        print(f"PROCESSING {item_label}: {item_id}")
        print(f"{'='*70}")
        print(f"  Input URLs:")
        print(f"    ArXiv: {arxiv_url or 'None'}")
        print(f"    GitHub: {github_url or 'None'}")
        print(f"    HuggingFace: {huggingface_url or 'None'}")
        
        # Fetch content from all sources
        content_dict = {}
        
        if arxiv_url and "arxiv.org" in str(arxiv_url):
            print(f"\n  Fetching ArXiv content...")
            content_dict['arxiv'] = self.fetch_arxiv_pdf_text(arxiv_url)
        else:
            print(f"\n  Skipping ArXiv (no valid URL)")
        
        if github_url and "github.com" in str(github_url):
            print(f"  Fetching GitHub content...")
            content_dict['github'] = self.fetch_github_readme(github_url)
        else:
            print(f"  Skipping GitHub (no valid URL)")
        
        if huggingface_url and "huggingface.co" in str(huggingface_url):
            print(f"  Fetching HuggingFace content...")
            content_dict['huggingface'] = self.fetch_huggingface_readme(huggingface_url)
        else:
            print(f"  Skipping HuggingFace (no valid URL: {huggingface_url})")
        
        # Print content statistics
        print("\n  📊 Content Statistics:")
        for source, content in content_dict.items():
            if content:
                print(f"    {source}: {len(content):,} characters")
            else:
                print(f"    {source}: NO CONTENT")
        
        # Create vector stores
        print("\n  Creating vector stores...")
        retrievers = self.create_vector_stores(content_dict)
        
        if not retrievers:
            print("  ❌ ERROR: No retrievers created! Check if sources have content.")
            return []
        
        print(f"  ✓ Created retrievers for: {list(retrievers.keys())}")
        
        # Process all questions
        results = []
        for question_type, config in self.questions.items():
            question = config['question']
            print(f"\n  ❓ Question: {question}")
            
            initial_state = {
                "question": question,
                "documents": [],
                "answer": "",
                "source_priority": [],
                "sources_used": [],
                "question_type": question_type,
                "row_index": 0,
                "row_retrievers": retrievers,
                "all_results": [],
                "conflicts": ""
            }
            
            result = self.workflow.invoke(initial_state)
            
            if item_type == 'ai':
                qa_result = {
                    'repo_id': item_id,
                    'question': question,
                    'answer': result['answer'],
                    'sources_used': result.get('sources_used', []),
                    'conflicts': result.get('conflicts', 'No conflicts detected'),
                    'source_priority': result.get('source_priority', []),
                    'question_type': result.get('question_type', 'general'),
                    'num_documents': len(result['documents'])
                }
            else:
                qa_result = {
                    'dataset_id': item_id,
                    'question': question,
                    'answer': result['answer'],
                    'sources_used': result.get('sources_used', []),
                    'conflicts': result.get('conflicts', 'No conflicts detected'),
                    'source_priority': result.get('source_priority', []),
                    'question_type': result.get('question_type', 'general'),
                    'num_documents': len(result['documents'])
                }
            
            results.append(qa_result)
            
            print(f"  🏷️  Question Type: {result.get('question_type', 'general')}")
            print(f"  💡 Answer: {result['answer'][:150]}...")
            print(f"  📚 Sources Used: {result.get('sources_used', [])}")
            print(f"  ⚠️  Conflicts: {result.get('conflicts', 'No conflicts detected')[:100]}...")
            print(f"  📄 Documents Retrieved: {len(result['documents'])}")
        
        return results


class DirectLLM:
    """
    Direct LLM class that calls the LLM directly with full source content.
    No chunking or RAG retrieval - sends all fetched content directly to the LLM.
    """
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0, llm_provider: str = "openai", ollama_base_url: str = None, questions: Optional[Dict[str, Dict]] = None, bom_type: str = 'ai'):
        """Initialize the Direct LLM system
        
        Args:
            model: Model name (e.g., 'gpt-4o' for OpenAI, 'llama3:70b' for Ollama)
            temperature: Temperature for generation
            llm_provider: 'openai' or 'ollama'
            ollama_base_url: Base URL for Ollama server (only used if llm_provider='ollama')
            questions: Optional subset of FIXED_QUESTIONS to answer in this run
            bom_type: 'ai' or 'data' - determines which question set to use if questions not provided
        """
        self.llm = create_llm(model, temperature, llm_provider, ollama_base_url)
        self.g = Github(os.environ.get("GITHUB_TOKEN"))
        self.hf_api = HfApi(token=os.environ.get("hug_token"))
        self.questions = questions or FIXED_QUESTIONS
        provider_info = f"Ollama ({ollama_base_url or 'default'})" if llm_provider == "ollama" else "OpenAI"
        print(f"✓ Initialized DirectLLM with model: {model} (provider: {provider_info})")
    
    def extract_repo_path(self, github_url: str) -> str:
        """Extract 'owner/repo' from a full GitHub URL"""
        parsed = urlparse(github_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2:
            return f"{path_parts[0]}/{path_parts[1]}"
        raise ValueError(f"Invalid GitHub repository URL: {github_url}")
    
    def extract_repo_id_from_hf_url(self, url: str) -> Optional[str]:
        """Extract HuggingFace repo ID from URL"""
        if not url or not isinstance(url, str):
            return None
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if "datasets" in parts:
            idx = parts.index("datasets")
            if len(parts) > idx + 2:
                return f"{parts[idx + 1]}/{parts[idx + 2]}"
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return None
    
    def fetch_github_readme(self, github_url: str) -> str:
        """Fetch README from GitHub repository"""
        try:
            repo_path = self.extract_repo_path(github_url)
            repo = self.g.get_repo(repo_path)
            readme_content = repo.get_readme().decoded_content.decode('utf-8')
            print(f"  ✓ Fetched README from GitHub repo: {repo_path}")
            return readme_content
        except Exception as e:
            print(f"  ⚠️ Error fetching GitHub README: {e}")
            return ""
    
    def fetch_huggingface_readme(self, hf_url: str) -> str:
        """Fetch README from HuggingFace"""
        try:
            repo_id = self.extract_repo_id_from_hf_url(hf_url)
            if not repo_id:
                print(f"  ⚠️ Could not extract repo_id from HuggingFace URL: {hf_url}")
                return ""
            
            headers = {}
            hf_token = os.environ.get("hug_token")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            
            # Try model URL first (most common for AI models)
            url_model = f"https://huggingface.co/{repo_id}/raw/main/README.md"
            try:
                r = requests.get(url_model, headers=headers, timeout=30, verify=False)
                r.raise_for_status()
                print(f"  ✓ Fetched README from HuggingFace model: {repo_id}")
                return r.text
            except Exception as model_error:
                # Try dataset URL as fallback
                url_dataset = f"https://huggingface.co/datasets/{repo_id}/raw/main/README.md"
                try:
                    r = requests.get(url_dataset, headers=headers, timeout=30, verify=False)
                    r.raise_for_status()
                    print(f"  ✓ Fetched README from HuggingFace dataset: {repo_id}")
                    return r.text
                except Exception as dataset_error:
                    print(f"  ⚠️ Failed to fetch README from both model and dataset URLs:")
                    print(f"      Model error: {model_error}")
                    print(f"      Dataset error: {dataset_error}")
                    return ""
        except Exception as e:
            print(f"  ⚠️ Error in fetch_huggingface_readme: {e}")
            return ""
    
    def fetch_arxiv_pdf_text(self, arxiv_url: str) -> str:
        """Fetch and extract text from arXiv PDF"""
        try:
            arxiv_id = arxiv_url.split('/')[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(pdf_url, verify=False, timeout=30)
            
            if response.status_code == 200:
                temp_pdf_path = "temp_direct.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(response.content)
                
                # Extract text using PyMuPDF
                doc = fitz.open(temp_pdf_path)
                text_content = []
                for page in doc:
                    text_content.append(page.get_text())
                doc.close()
                
                # Clean up temporary file
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                
                full_text = "\n".join(text_content)
                print(f"  ✓ Fetched arXiv paper ({len(full_text):,} chars)")
                return full_text
            else:
                print(f"  ⚠️ Failed to download PDF from {pdf_url}")
                return ""
        except Exception as e:
            print(f"  ⚠️ Error fetching arXiv content: {e}")
            if os.path.exists("temp_direct.pdf"):
                os.remove("temp_direct.pdf")
            return ""
    
    def _truncate_content(self, content: str, max_chars: int = 50000) -> str:
        """Truncate content to fit within context window limits"""
        if len(content) <= max_chars:
            return content
        # Truncate and add indicator
        return content[:max_chars] + f"\n\n[... Content truncated at {max_chars:,} characters ...]"
    
    def _generate_answer_direct(self, question: str, question_type: str, content_dict: Dict[str, str]) -> Dict:
        """Generate answer by sending full content directly to LLM"""
        
        # Build context from all sources
        context_parts = []
        sources_used = []
        
        for source, content in content_dict.items():
            if content and len(content.strip()) > 50:
                sources_used.append(source)
                # Truncate very long content to avoid context window issues
                truncated_content = self._truncate_content(content, max_chars=40000)
                context_parts.append(f"\n{'='*60}")
                context_parts.append(f"SOURCE: {source.upper()}")
                context_parts.append(f"{'='*60}")
                context_parts.append(truncated_content)
                context_parts.append("")
        
        context = "\n".join(context_parts)
        
        if not context.strip():
            return {
                'answer': 'Not found.',
                'conflicts': 'No conflicts detected',
                'sources_used': []
            }
        
        question_config = self.questions.get(question_type, FIXED_QUESTIONS.get(question_type, {}))
        field_name = question_type
        question_summary = question_config.get('question', question)
        field_description = question_config.get('description', '')
        
        prompt = f"""You are an AI model documentation expert. Your task is to extract specific information about an AI model or package from the provided sources AND detect any conflicts between sources.

You are looking for information related to the following field:

FIELD_NAME: {field_name}
QUESTION_SUMMARY: {question_summary}
FIELD_DESCRIPTION: {field_description}

AVAILABLE SOURCES:
{context}

INSTRUCTIONS:
1. Carefully read through ALL the provided source materials.
2. Look for information that directly answers the question based on the field name and description.
3. CONFLICT DETECTION - THIS IS CRITICAL:
   - If multiple sources provide information about the same aspect but with DIFFERENT or CONTRADICTORY details, this is a CONFLICT.
   - Pay attention to: different methods, different values, contradictory statements, incompatible descriptions.
   - Minor differences in wording are NOT conflicts if they describe the same thing.

4. If you find the answer:
   - Provide a clear, specific, and detailed response.
   - If information comes from multiple sources and they AGREE, synthesize them.
   - If sources DISAGREE or provide CONFLICTING information, note this explicitly in your answer.
   - Mention which source(s) contained each piece of information.

5. If information is partially available:
   - Provide what you found.
   - Clearly state what information is missing.

6. If no relevant information is found:
   - State: "Not found."
   - DO NOT make up information.

YOUR RESPONSE MUST BE IN THIS EXACT FORMAT:

ANSWER: [Your detailed answer here, incorporating information from all sources]

CONFLICT_STATUS: [Either "No conflicts detected" OR "CONFLICT DETECTED: [describe the specific conflict between sources]"]

RESPONSE:"""
        
        print(f"\n  📝 Full context length: {len(context):,} characters (Direct mode - no chunking)")
        print(f"  📚 Sources in context: {sources_used}")
        
        response = self.llm.invoke(prompt)
        response_text = response.content
        
        # Parse the response
        answer = ""
        conflicts = "No conflicts detected"
        
        parts = response_text.split("CONFLICT_STATUS:")
        
        if len(parts) == 2:
            answer_part = parts[0].strip()
            if answer_part.startswith("ANSWER:"):
                answer = answer_part[7:].strip()
            else:
                answer = answer_part
            conflicts = parts[1].strip()
        else:
            answer = response_text
            conflicts = "No conflicts detected"
        
        return {
            'answer': answer,
            'conflicts': conflicts,
            'sources_used': sources_used
        }
    
    def process_ai_model(self, repo_id: str, arxiv_url: str, github_url: str, huggingface_url: str) -> List[Dict]:
        """Process a single AI model and answer all questions using Direct LLM (no RAG)"""
        return self.process(repo_id, arxiv_url, github_url, huggingface_url, 'ai')
    
    def process_dataset(self, dataset_id: str, arxiv_url: str, github_url: str, huggingface_url: str) -> List[Dict]:
        """Process a single dataset and answer all questions using Direct LLM (no RAG)"""
        return self.process(dataset_id, arxiv_url, github_url, huggingface_url, 'data')
    
    def process(self, item_id: str, arxiv_url: str, github_url: str, huggingface_url: str, item_type: str = None) -> List[Dict]:
        """Unified process method that handles both AI models and datasets using Direct LLM"""
        if item_type is None:
            item_type = self.bom_type
        item_type = item_type.lower()
        
        item_label = "AI MODEL" if item_type == 'ai' else "DATASET"
        print(f"\n{'='*70}")
        print(f"PROCESSING {item_label} (DIRECT MODE - NO RAG): {item_id}")
        print(f"{'='*70}")
        print(f"  Input URLs:")
        print(f"    ArXiv: {arxiv_url or 'None'}")
        print(f"    GitHub: {github_url or 'None'}")
        print(f"    HuggingFace: {huggingface_url or 'None'}")
        
        # Fetch content from all sources
        content_dict = {}
        
        if arxiv_url and "arxiv.org" in str(arxiv_url):
            print(f"\n  Fetching ArXiv content...")
            content_dict['arxiv'] = self.fetch_arxiv_pdf_text(arxiv_url)
        else:
            print(f"\n  Skipping ArXiv (no valid URL)")
        
        if github_url and "github.com" in str(github_url):
            print(f"  Fetching GitHub content...")
            content_dict['github'] = self.fetch_github_readme(github_url)
        else:
            print(f"  Skipping GitHub (no valid URL)")
        
        if huggingface_url and "huggingface.co" in str(huggingface_url):
            print(f"  Fetching HuggingFace content...")
            content_dict['huggingface'] = self.fetch_huggingface_readme(huggingface_url)
        else:
            print(f"  Skipping HuggingFace (no valid URL: {huggingface_url})")
        
        # Print content statistics
        print("\n  📊 Content Statistics (Direct Mode):")
        total_chars = 0
        for source, content in content_dict.items():
            if content:
                chars = len(content)
                total_chars += chars
                print(f"    {source}: {chars:,} characters")
            else:
                print(f"    {source}: NO CONTENT")
        print(f"    TOTAL: {total_chars:,} characters")
        
        if not any(content_dict.values()):
            print("  ❌ ERROR: No content fetched from any source!")
            return []
        
        # Process all questions - send full content for each question
        results = []
        for question_type, config in self.questions.items():
            question = config['question']
            print(f"\n  ❓ Question: {question}")
            
            # Generate answer directly (no retrieval step)
            result = self._generate_answer_direct(question, question_type, content_dict)
            
            if item_type == 'ai':
                qa_result = {
                    'repo_id': item_id,
                    'question': question,
                    'answer': result['answer'],
                    'sources_used': result['sources_used'],
                    'conflicts': result.get('conflicts', 'No conflicts detected'),
                    'source_priority': list(content_dict.keys()),
                    'question_type': question_type,
                    'num_documents': 0  # No chunking in direct mode
                }
            else:
                qa_result = {
                    'dataset_id': item_id,
                    'question': question,
                    'answer': result['answer'],
                    'sources_used': result['sources_used'],
                    'conflicts': result.get('conflicts', 'No conflicts detected'),
                    'source_priority': list(content_dict.keys()),
                    'question_type': question_type,
                    'num_documents': 0  # No chunking in direct mode
                }
            
            results.append(qa_result)
            
            print(f"  🏷️  Question Type: {question_type}")
            print(f"  💡 Answer: {result['answer'][:150]}...")
            print(f"  📚 Sources Used: {result['sources_used']}")
            print(f"  ⚠️  Conflicts: {result.get('conflicts', 'No conflicts detected')[:100]}...")
            print(f"  📄 Mode: Direct LLM (no chunking/retrieval)")
        
        return results