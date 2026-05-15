"""RDF namespaces and predicate constants for the worldofBOMs graph.

Single source of truth. If you add a predicate here, also add it to
docs/worldofboms/SCHEMA.md (a CI test enforces parity).
"""

from rdflib import Namespace, URIRef

AIBOM = Namespace("https://aikaboom.dev/aibom#")
BOM = Namespace("bom:")
SPDX = Namespace("https://spdx.org/rdf/3.0.1/terms/")
PROV = Namespace("http://www.w3.org/ns/prov#")
DCAT = Namespace("http://www.w3.org/ns/dcat#")


# Core classes
Artifact = AIBOM.Artifact
Model = AIBOM.Model
Dataset = AIBOM.Dataset
Paper = AIBOM.Paper
CodeRepo = AIBOM.CodeRepo
ArtifactVersion = AIBOM.ArtifactVersion
BOMClaim = AIBOM.BOMClaim
GenerationRun = AIBOM.GenerationRun
TrustVote = AIBOM.TrustVote
Agent = AIBOM.Agent
License = AIBOM.License
Supplier = AIBOM.Supplier
Person = AIBOM.Person
Source = AIBOM.Source


# Predicates: tier edges
hasVersion = AIBOM.hasVersion
hasClaim = AIBOM.hasClaim
canonicalClaim = AIBOM.canonicalClaim
generatedBy = AIBOM.generatedBy
supersedes = AIBOM.supersedes


# Predicates: claim properties
trustScore = AIBOM.trustScore
useCase = AIBOM.useCase
mode = AIBOM.mode
createdAt = AIBOM.createdAt
schemaVersion = AIBOM.schemaVersion


# Predicates: trust votes
trustVoteFor = AIBOM.trustVoteFor
votedBy = AIBOM.votedBy
voteKind = AIBOM.voteKind
votedAt = AIBOM.votedAt
comment = AIBOM.comment


# Predicates: identifier model
identifier = AIBOM.identifier
primaryIdentifier = AIBOM.primaryIdentifier
canonicalLabel = AIBOM.canonicalLabel
canonRuleVersion = AIBOM.canonRuleVersion
platform = AIBOM.platform
value = AIBOM.value
alias = AIBOM.alias
isPlaceholder = AIBOM.isPlaceholder
potentialDuplicateOf = AIBOM.potentialDuplicateOf


# Predicates: per-field RDF-star annotations
assertedBy = AIBOM.assertedBy
conflictKind = AIBOM.conflictKind
conflictsWith = AIBOM.conflictsWith


# Predicates: BOM-domain edges
trainedOn = AIBOM.trainedOn
describedIn = AIBOM.describedIn
hostedAt = AIBOM.hostedAt
hasLicense = AIBOM.hasLicense
suppliedBy = AIBOM.suppliedBy
authoredBy = AIBOM.authoredBy


# Predicates: GenerationRun properties
provider = AIBOM.provider
llmModel = AIBOM.llmModel
promptVersion = AIBOM.promptVersion
codeVersion = AIBOM.codeVersion


# Vote-kind individuals
trusted = AIBOM.trusted
flagged = AIBOM.flagged
disputed = AIBOM.disputed
implicit_use = URIRef(str(AIBOM) + "implicit-use")
implicit_validate = URIRef(str(AIBOM) + "implicit-validate")


# Conflict-kind individuals
noConflict = AIBOM.noConflict
interSourceConflict = AIBOM.interSourceConflict
intraSourceConflict = AIBOM.intraSourceConflict


SCHEMA_VERSION = "1.0"
CANON_RULE_VERSION = "1"
