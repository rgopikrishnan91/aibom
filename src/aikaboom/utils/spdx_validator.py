"""
Unified SPDX 3.0.1 BOM Validator
Converts AI and Dataset BOM metadata to SPDX 3.0.1 compliant format
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


class SPDXValidator:
    """Unified validator that converts both AI and Dataset BOM data to SPDX 3.0.1 format"""
    
    # Field mappings for AI BOM
    AI_FIELD_MAPPING = {
        # Direct fields
        "releaseTime": "releaseTime",
        "suppliedBy": "suppliedBy",
        "downloadLocation": "downloadLocation",
        "packageVersion": "packageVersion",
        "primaryPurpose": "primaryPurpose",
        "license": "license",
        
        # RAG fields for AI models
        "model_name": "AI-Model-Name",
        "autonomy_type": "autonomyType",
        "domain": "domain",
        "energy_consumption": "energyConsumption",
        "hyperparameters": "hyperparameter",
        "intended_use": "informationAboutApplication",
        "training_information": "informationAboutTraining",
        "limitations": "limitation",
        "performance_metrics": "metric",
        "decision_threshold": "metricDecisionThreshold",
        "data_preprocessing": "modelDataPreprocessing",
        "model_explainability": "modelExplainability",
        "safety_risk_assessment": "safetyRiskAssessment",
        "standard_compliance": "standardCompliance",
        "model_type": "typeOfModel",
        "sensitive_personal_information": "useSensitivePersonalInformation"
    }
    
    # Field mappings for Dataset BOM
    DATASET_FIELD_MAPPING = {
        # Direct fields
        "name": "dataset_name",
        "originatedBy": "originatedBy",
        "builtTime": "builtTime",
        "releaseTime": "releaseTime",
        "downloadLocation": "downloadLocation",
        "primaryPurpose": "primaryPurpose",
        "license": "license",
        
        # RAG fields for datasets
        "dataPreprocessing": "dataPreprocessing",
        "datasetAvailability": "datasetAvailability",
        "dataCollectionProcess": "dataCollectionProcess",
        "datasetSize": "datasetSize",
        "datasetType": "datasetType",
        "datasetUpdateMechanism": "datasetUpdateMechanism",
        "hasSensitivePersonalInformation": "hasSensitivePersonalInformation",
        "intendedUse": "intendedUse",
        "knownBias": "knownBias",
        "anonymizationMethodUsed": "anonymizationMethodUsed",
        "confidentialityLevel": "confidentialityLevel",
        "datasetNoise": "datasetNoise",
        "sensorUsed": "sensorUsed"
    }
    
    def __init__(self, template_path: str = None, bom_type: str = 'ai'):
        """
        Initialize validator with optional SPDX template
        
        Args:
            template_path: Optional path to SPDX template file
            bom_type: Type of BOM to generate ('ai' or 'data')
        """
        self.template_path = template_path
        self.bom_type = bom_type.lower()
        self.template = self._load_template() if template_path else None
    
    def _load_template(self) -> Optional[Dict[str, Any]]:
        """Load SPDX template from file"""
        try:
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Warning: SPDX template not found at {self.template_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"⚠️ Warning: Invalid JSON in SPDX template: {e}")
            return None
    
    def _create_minimal_template(self) -> Dict[str, Any]:
        """Create minimal SPDX 3.0.1 template"""
        return {
            "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
            "@graph": []
        }
    
    def _extract_value(self, field_data: Any) -> Any:
        """Extract value from triplet structure or return as-is"""
        if isinstance(field_data, dict) and "value" in field_data:
            return field_data["value"]
        return field_data
    
    def _generate_uuid(self) -> str:
        """Generate UUID for SPDX IDs"""
        return str(uuid.uuid4())
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def validate_and_convert(self, bom_data: Dict[str, Any], bom_type: str = None) -> Dict[str, Any]:
        """
        Convert BOM data to SPDX 3.0.1 format
        
        Args:
            bom_data: Dictionary containing metadata
            bom_type: Type of BOM ('ai' or 'data'), overrides constructor setting
            
        Returns:
            SPDX 3.0.1 compliant dictionary
        """
        # Use provided bom_type or fall back to constructor setting
        bom_type = (bom_type or self.bom_type).lower()
        
        if bom_type == 'ai':
            return self._convert_ai_bom(bom_data)
        elif bom_type in ['data', 'dataset']:
            return self._convert_dataset_bom(bom_data)
        else:
            raise ValueError(f"Invalid bom_type: {bom_type}. Must be 'ai' or 'data'")
    
    def _convert_ai_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert AI BOM data to SPDX 3.0.1 AI Package format"""
        # Extract fields
        direct_fields = bom_data.get("direct_fields", {})
        rag_fields = bom_data.get("rag_fields", {})
        repo_id = bom_data.get("repo_id", "unknown")
        
        # Generate UUIDs
        doc_uuid = self._generate_uuid()
        creation_uuid = self._generate_uuid()
        person_uuid = self._generate_uuid()
        org_uuid = self._generate_uuid()
        bom_uuid = self._generate_uuid()
        package_uuid = self._generate_uuid()
        license_uuid = self._generate_uuid()
        
        # Get timestamp
        timestamp = self._get_current_timestamp()
        
        # Extract organization info
        supplied_by = self._extract_value(direct_fields.get("suppliedBy", "Unknown"))
        
        # Extract license
        license_expr = self._extract_value(direct_fields.get("license", "NOASSERTION"))
        
        # Build SPDX document
        spdx_doc = {
            "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
            "@graph": [
                # 1. CreationInfo
                {
                    "type": "CreationInfo",
                    "@id": f"_:creationinfo-{creation_uuid}",
                    "specVersion": "3.0.1",
                    "createdBy": [f"urn:spdx:Person-{person_uuid}"],
                    "created": timestamp
                },
                # 2. Person
                {
                    "type": "Person",
                    "spdxId": f"urn:spdx:Person-{person_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "name": "AI BOM Generator",
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "email",
                        "identifier": "bom-generator@example.com"
                    }]
                },
                # 3. Organization
                {
                    "type": "Organization",
                    "spdxId": f"urn:spdx:Organization-{org_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "name": supplied_by or "Unknown",
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "other",
                        "issuingAuthority": "GitHub",
                        "identifier": repo_id,
                        "identifierLocator": ["NOASSERTION"]
                    }]
                },
                # 4. SpdxDocument
                {
                    "type": "SpdxDocument",
                    "spdxId": f"urn:spdx:Document-{doc_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "profileConformance": ["core", "AI"],
                    "rootElement": [f"urn:spdx:Bom-{bom_uuid}"]
                },
                # 5. Bom
                {
                    "type": "Bom",
                    "spdxId": f"urn:spdx:Bom-{bom_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "profileConformance": ["core", "AI"],
                    "rootElement": [f"urn:spdx:AIPackage-{package_uuid}"]
                },
                # 6. AIPackage
                self._build_ai_package(package_uuid, creation_uuid, org_uuid, direct_fields, rag_fields, repo_id),
                # 7. LicenseExpression
                {
                    "type": "simplelicensing_LicenseExpression",
                    "spdxId": f"urn:spdx:License-{license_uuid}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "simplelicensing_licenseExpression": license_expr or "NOASSERTION",
                    "simplelicensing_licenseListVersion": "3.25.0",
                    "comment": "License information extracted from AI BOM metadata"
                },
                # 8. Relationship - concludedLicense
                {
                    "type": "Relationship",
                    "spdxId": f"urn:spdx:Relationship-concludedLicense-{self._generate_uuid()}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "relationshipType": "hasConcludedLicense",
                    "from": f"urn:spdx:AIPackage-{package_uuid}",
                    "to": [f"urn:spdx:License-{license_uuid}"],
                    "description": "Concluded license for AI package"
                },
                # 9. Relationship - declaredLicense
                {
                    "type": "Relationship",
                    "spdxId": f"urn:spdx:Relationship-declaredLicense-{self._generate_uuid()}",
                    "creationInfo": f"_:creationinfo-{creation_uuid}",
                    "relationshipType": "hasDeclaredLicense",
                    "from": f"urn:spdx:AIPackage-{package_uuid}",
                    "to": [f"urn:spdx:License-{license_uuid}"],
                    "description": "Declared license for AI package"
                }
            ]
        }
        
        return spdx_doc
    
    def _build_ai_package(
        self, package_uuid: str, creation_uuid: str, org_uuid: str,
        direct_fields: Dict, rag_fields: Dict, repo_id: str
    ) -> Dict[str, Any]:
        """Build AI Package element with all mapped fields"""
        ai_package = {
            "type": "AI_AIPackage",
            "spdxId": f"urn:spdx:AIPackage-{package_uuid}",
            "creationInfo": f"_:creationinfo-{creation_uuid}",
            "originatedBy": [f"urn:spdx:Organization-{org_uuid}"]
        }
        
        # Map direct fields
        direct_mapping = {
            "releaseTime": "releaseTime",
            "suppliedBy": "suppliedBy",
            "downloadLocation": "downloadLocation",
            "packageVersion": "packageVersion",
            "primaryPurpose": "primaryPurpose"
        }
        
        for our_field, spdx_field in direct_mapping.items():
            value = self._extract_value(direct_fields.get(our_field))
            if value is not None and value != "":
                ai_package[spdx_field] = value
            else:
                if spdx_field == "downloadLocation":
                    ai_package[spdx_field] = "NOASSERTION"
                elif spdx_field in ["suppliedBy", "packageVersion"]:
                    ai_package[spdx_field] = ""
                elif spdx_field == "primaryPurpose":
                    ai_package[spdx_field] = "data"
        
        # Set AI-Model-Name (required field)
        model_name_value = self._extract_value(rag_fields.get("model_name"))
        ai_package["AI-Model-Name"] = model_name_value or repo_id or "AI Model Name Placeholder"
        
        # Map RAG fields
        rag_mapping = {
            "autonomy_type": "autonomyType",
            "domain": "domain",
            "energy_consumption": "energyConsumption",
            "hyperparameters": "hyperparameter",
            "intended_use": "informationAboutApplication",
            "training_information": "informationAboutTraining",
            "limitations": "limitation",
            "performance_metrics": "metric",
            "decision_threshold": "metricDecisionThreshold",
            "data_preprocessing": "modelDataPreprocessing",
            "model_explainability": "modelExplainability",
            "safety_risk_assessment": "safetyRiskAssessment",
            "standard_compliance": "standardCompliance",
            "model_type": "typeOfModel",
            "sensitive_personal_information": "useSensitivePersonalInformation"
        }
        
        for our_field, spdx_field in rag_mapping.items():
            value = self._extract_value(rag_fields.get(our_field))
            if value is not None and value != "":
                ai_package[spdx_field] = value
        
        # Set builtTime if not present
        if "builtTime" not in ai_package:
            built_time = self._extract_value(direct_fields.get("builtTime"))
            ai_package["builtTime"] = built_time or self._get_current_timestamp()
        
        ai_package["comment"] = "This results are generated by AI tools."
        
        return ai_package
    
    def _convert_dataset_bom(self, bom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Dataset BOM data to SPDX 3.0.1 Dataset Package format"""
        # Extract metadata
        direct = bom_data.get('direct_metadata', bom_data.get('direct_fields', {}))
        rag = bom_data.get('rag_metadata', bom_data.get('rag_fields', {}))
        urls = bom_data.get('urls', {})
        dataset_id = bom_data.get('dataset_id', bom_data.get('repo_id', 'unknown'))
        
        # Generate UUIDs
        person_uuid = self._generate_uuid()
        org_uuid = self._generate_uuid()
        doc_uuid = self._generate_uuid()
        bom_uuid = self._generate_uuid()
        dataset_uuid = self._generate_uuid()
        rel_concluded_uuid = self._generate_uuid()
        rel_declared_uuid = self._generate_uuid()
        license_uuid = self._generate_uuid()
        
        # Get timestamp
        created_time = self._get_current_timestamp()
        
        # Extract values with fallbacks
        dataset_name = self._extract_value(direct.get('name') or rag.get('name') or dataset_id)
        originated_by = self._extract_value(direct.get('originatedBy') or rag.get('originatedBy') or "Unknown")
        built_time = self._extract_value(direct.get('builtTime') or rag.get('builtTime') or created_time)
        release_time = self._extract_value(direct.get('releaseTime') or rag.get('releaseTime') or created_time)
        download_location = self._extract_value(direct.get('downloadLocation') or urls.get('github') or urls.get('huggingface') or "NOASSERTION")
        primary_purpose = self._extract_value(direct.get('primaryPurpose') or rag.get('primaryPurpose') or "data")
        license_expr = self._extract_value(direct.get('license') or rag.get('license') or "NOASSERTION")
        dataset_availability = self._extract_value(direct.get('datasetAvailability') or rag.get('datasetAvailability') or "directDownload")
        
        # RAG-specific fields with type conversion
        data_preprocessing = self._extract_value(rag.get('dataPreprocessing', []))
        if isinstance(data_preprocessing, str):
            data_preprocessing = [data_preprocessing] if data_preprocessing else []
        
        data_collection = self._extract_value(rag.get('dataCollectionProcess') or "")
        
        dataset_size = self._extract_value(rag.get('datasetSize', 0))
        if isinstance(dataset_size, str):
            try:
                dataset_size = int(dataset_size)
            except (ValueError, TypeError):
                dataset_size = 0
        
        dataset_type = self._extract_value(rag.get('datasetType', []))
        if isinstance(dataset_type, str):
            dataset_type = [dataset_type] if dataset_type else []
        
        dataset_update = self._extract_value(rag.get('datasetUpdateMechanism') or "")
        has_pii = self._extract_value(rag.get('hasSensitivePersonalInformation') or "no")
        intended_use = self._extract_value(rag.get('intendedUse') or "")
        
        known_bias = self._extract_value(rag.get('knownBias', []))
        if isinstance(known_bias, str):
            known_bias = [known_bias] if known_bias else []
        
        # Build SPDX document
        spdx_doc = {
            "@context": "https://spdx.org/rdf/3.0.1/spdx-context.jsonld",
            "@graph": [
                # 1. CreationInfo
                {
                    "type": "CreationInfo",
                    "@id": "_:creationinfo",
                    "specVersion": "3.0.1",
                    "createdBy": [f"https://spdx.org/spdxdocs/Person1-{person_uuid}"],
                    "created": created_time
                },
                # 2. Person
                {
                    "type": "Person",
                    "spdxId": f"https://spdx.org/spdxdocs/Person1-{person_uuid}",
                    "creationInfo": "_:creationinfo",
                    "name": "Dataset BOM Generator",
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "email",
                        "identifier": "bom-generator@example.com"
                    }]
                },
                # 3. Organization
                {
                    "type": "Organization",
                    "spdxId": f"https://spdx.org/spdxdocs/Organization1-{org_uuid}",
                    "creationInfo": "_:creationinfo",
                    "name": originated_by,
                    "externalIdentifier": [{
                        "type": "ExternalIdentifier",
                        "externalIdentifierType": "other",
                        "issuingAuthority": "GitHub",
                        "identifier": originated_by,
                        "identifierLocator": [download_location]
                    }]
                },
                # 4. SpdxDocument
                {
                    "type": "SpdxDocument",
                    "spdxId": f"https://spdx.org/spdxdocs/Document1-{doc_uuid}",
                    "creationInfo": "_:creationinfo",
                    "profileConformance": ["core", "dataset"],
                    "rootElement": [f"https://spdx.org/spdxdocs/BOM1-{bom_uuid}"]
                },
                # 5. Bom
                {
                    "type": "Bom",
                    "spdxId": f"https://spdx.org/spdxdocs/BOM1-{bom_uuid}",
                    "creationInfo": "_:creationinfo",
                    "profileConformance": ["core", "dataset"],
                    "rootElement": [f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}"]
                },
                # 6. DatasetPackage
                {
                    "type": "dataset_DatasetPackage",
                    "spdxId": f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}",
                    "creationInfo": "_:creationinfo",
                    "dataset_name": dataset_name,
                    "originatedBy": [f"https://spdx.org/spdxdocs/Organization1-{org_uuid}"],
                    "builtTime": built_time,
                    "releaseTime": release_time,
                    "downloadLocation": download_location,
                    "primaryPurpose": primary_purpose,
                    "anonymizationMethodUsed": self._extract_value(rag.get('anonymizationMethodUsed', "")),
                    "confidentialityLevel": self._extract_value(rag.get('confidentialityLevel', "clear")),
                    "dataPreprocessing": data_preprocessing,
                    "datasetAvailability": dataset_availability,
                    "dataCollectionProcess": data_collection,
                    "datasetNoise": self._extract_value(rag.get('datasetNoise', "")),
                    "datasetSize": dataset_size,
                    "datasetType": dataset_type,
                    "datasetUpdateMechanism": dataset_update,
                    "hasSensitivePersonalInformation": has_pii,
                    "intendedUse": intended_use,
                    "knownBias": known_bias,
                    "sensorUsed": self._extract_value(rag.get('sensorUsed', "")),
                    "comment": "This results are generated by AI tools."
                },
                # 7. LicenseExpression
                {
                    "type": "simplelicensing_LicenseExpression",
                    "spdxId": f"https://spdx.org/licenses/{license_uuid}",
                    "creationInfo": "_:creationinfo",
                    "simplelicensing_licenseExpression": license_expr,
                    "simplelicensing_licenseListVersion": "3.25.0",
                    "comment": "License information extracted from Dataset BOM metadata"
                },
                # 8. Relationship - concludedLicense
                {
                    "type": "Relationship",
                    "spdxId": f"https://spdx.org/spdxdocs/Relationship/concludedLicense-{rel_concluded_uuid}",
                    "creationInfo": "_:creationinfo",
                    "relationshipType": "hasConcludedLicense",
                    "from": f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}",
                    "to": [f"https://spdx.org/licenses/{license_uuid}"],
                    "description": "Concluded license for dataset package"
                },
                # 9. Relationship - declaredLicense
                {
                    "type": "Relationship",
                    "spdxId": f"https://spdx.org/spdxdocs/Relationship/declaredLicense-{rel_declared_uuid}",
                    "creationInfo": "_:creationinfo",
                    "relationshipType": "hasDeclaredLicense",
                    "from": f"https://spdx.org/spdxdocs/DatasetPackage1-{dataset_uuid}",
                    "to": [f"https://spdx.org/licenses/{license_uuid}"],
                    "description": "Declared license for dataset package"
                }
            ]
        }
        
        return spdx_doc
    
    def save_spdx(self, spdx_data: Dict[str, Any], output_path: str) -> str:
        """Save SPDX document to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(spdx_data, f, indent=2, ensure_ascii=False)
        print(f"✅ SPDX BOM saved to: {output_path}")
        return output_path
    
    def validate_spdx_bom(self, spdx_bom: Dict) -> tuple:
        """
        Basic validation of SPDX BOM structure
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for @graph array (SPDX 3.0 structure)
        if '@graph' not in spdx_bom or not isinstance(spdx_bom['@graph'], list):
            errors.append("Missing or invalid '@graph' array")
        
        # Check for context
        if '@context' not in spdx_bom:
            errors.append("Missing '@context'")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            print("✅ SPDX BOM validation passed")
        else:
            print(f"❌ SPDX BOM validation failed with {len(errors)} errors")
            for error in errors:
                print(f"  ⚠️ {error}")
        
        return is_valid, errors


def validate_bom_to_spdx(
    bom_data: Dict[str, Any], 
    bom_type: str = 'ai',
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to validate and convert BOM data to SPDX format
    
    Args:
        bom_data: BOM metadata dictionary
        bom_type: Type of BOM ('ai' or 'data')
        output_path: Optional path to save SPDX JSON
        
    Returns:
        SPDX 3.0.1 compliant dictionary
    """
    validator = SPDXValidator(bom_type=bom_type)
    spdx_data = validator.validate_and_convert(bom_data, bom_type=bom_type)
    
    if output_path:
        validator.save_spdx(spdx_data, output_path)
    
    return spdx_data


if __name__ == "__main__":
    print("🧪 Testing Unified SPDX Validator")
    print("=" * 70)
    
    # Test AI BOM
    print("\n📦 Testing AI BOM conversion...")
    ai_bom_data = {
        "repo_id": "test/ai-model",
        "direct_fields": {
            "suppliedBy": "Test AI Lab",
            "license": "MIT",
            "downloadLocation": "https://huggingface.co/test/ai-model",
            "releaseTime": "2024-01-15T00:00:00Z"
        },
        "rag_fields": {
            "model_name": "Test AI Model",
            "model_type": "transformer",
            "intended_use": "Text generation"
        }
    }
    
    ai_validator = SPDXValidator(bom_type='ai')
    ai_spdx = ai_validator.validate_and_convert(ai_bom_data)
    ai_validator.validate_spdx_bom(ai_spdx)
    
    # Test Dataset BOM
    print("\n📊 Testing Dataset BOM conversion...")
    dataset_bom_data = {
        "dataset_id": "test/dataset",
        "direct_metadata": {
            "name": "Test Dataset",
            "license": "Apache-2.0",
            "originatedBy": "Test Data Lab"
        },
        "rag_metadata": {
            "intendedUse": "Research purposes",
            "datasetSize": 10000
        },
        "urls": {
            "huggingface": "https://huggingface.co/datasets/test/dataset"
        }
    }
    
    dataset_validator = SPDXValidator(bom_type='data')
    dataset_spdx = dataset_validator.validate_and_convert(dataset_bom_data)
    dataset_validator.validate_spdx_bom(dataset_spdx)
    
    print("\n" + "=" * 70)
    print("✅ Unified SPDX Validator test completed successfully!")
