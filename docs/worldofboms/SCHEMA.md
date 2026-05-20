# worldofBOMs — RDF Schema Reference

Canonical reference for every class and predicate in the worldofBOMs
vocabulary. The CI parity test (`tests/store/test_docs_schema_parity.py`)
ensures this doc and `src/aikaboom/store/vocab.py` stay in sync.

## Namespaces

| Prefix | URI |
|---|---|
| `aibom:` | `https://aikaboom.dev/aibom#` |
| `bom:` | `bom:` (instance IRIs use this scheme) |
| `spdx:` | `https://spdx.org/rdf/3.0.1/terms/` |
| `prov:` | `http://www.w3.org/ns/prov#` |
| `dcat:` | `http://www.w3.org/ns/dcat#` |

## Classes

### aibom:Artifact

The upstream thing a BOM is about. Subtype-specific subclasses (`Model`,
`Dataset`, `Paper`, `CodeRepo`) inherit from this.

```turtle
bom:artifact/<hash>
    a aibom:Model ;
    aibom:identifier [ aibom:platform "huggingface" ; aibom:value "mistralai/mistral-7b" ] ;
    aibom:primaryIdentifier "huggingface:mistralai/mistral-7b" ;
    aibom:canonicalLabel "Mistral 7B v0.1" ;
    aibom:canonRuleVersion "1" .
```

### aibom:Model, aibom:Dataset, aibom:Paper, aibom:CodeRepo

Subclasses of `aibom:Artifact` indicating the artifact's kind. Selected at
mint time from the primary identifier's platform and the existing aikaboom
artifact-type heuristics.

### aibom:ArtifactVersion

A specific commit / version of an Artifact.

```turtle
bom:version/<artifact-hash>/27d67f1b
    a aibom:ArtifactVersion ;
    aibom:canonicalClaim bom:claim/<uuid> .
```

### aibom:BOMClaim

One generation event's claim about an ArtifactVersion. Many BOMClaims may
exist per ArtifactVersion (one per generation run).

```turtle
bom:claim/<uuid>
    a aibom:BOMClaim ;
    aibom:useCase "license" ;
    aibom:mode "rag" ;
    aibom:createdAt "2026-05-14T10:00:00Z"^^xsd:dateTime ;
    aibom:schemaVersion "1.0" ;
    aibom:trustScore 0.0 ;
    aibom:generatedBy bom:run/<hash> .
```

### aibom:GenerationRun

The (provider, LLM model, prompt version, code version, mode, use case)
combination that produced a claim. Deterministic IRI hash — identical
parameters share a single run node.

```turtle
bom:run/<hash>
    a aibom:GenerationRun ;
    aibom:provider "openrouter" ;
    aibom:llmModel "anthropic/claude-3-haiku" ;
    aibom:promptVersion "v12" ;
    aibom:codeVersion "abc1234" ;
    aibom:mode "rag" ;
    aibom:useCase "license" .
```

### aibom:TrustVote

A vote on a BOMClaim.

```turtle
bom:vote/<uuid>
    a aibom:TrustVote ;
    aibom:trustVoteFor bom:claim/<claim-uuid> ;
    aibom:votedBy bom:agent/<agent-hash> ;
    aibom:voteKind aibom:trusted ;
    aibom:votedAt "2026-05-14T..."^^xsd:dateTime .
```

### aibom:Agent

A user or automated generator.

### aibom:License, aibom:Supplier, aibom:Person, aibom:Source

Reused where SPDX/PROV-O don't already cover them.

## Predicates

### Tier edges

| Predicate | Domain → Range | Purpose |
|---|---|---|
| `aibom:hasVersion` | Artifact → ArtifactVersion | An artifact has this version. |
| `aibom:hasClaim` | ArtifactVersion → BOMClaim | A version has this claim made about it. |
| `aibom:canonicalClaim` | ArtifactVersion → BOMClaim | Highest-trust claim pointer; recomputed on every vote/claim. |
| `aibom:generatedBy` | BOMClaim → GenerationRun | Which run produced this claim. |
| `aibom:supersedes` | BOMClaim → BOMClaim | This claim replaces an older one (rarely used in v1 — claims accumulate). |

### Claim properties

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:trustScore` | xsd:decimal | Aggregate score in `[-1, +1]`. |
| `aibom:useCase` | xsd:string | `license` / `complete` / etc. |
| `aibom:mode` | xsd:string | `rag` / `direct`. |
| `aibom:createdAt` | xsd:dateTime | Generation timestamp. |
| `aibom:schemaVersion` | xsd:string | Vocab version this claim was written under. |

### Vote properties

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:trustVoteFor` | BOMClaim | The claim being voted on. |
| `aibom:votedBy` | Agent | The voter. |
| `aibom:voteKind` | `aibom:trusted`/`flagged`/`disputed`/`implicit-use`/`implicit-validate` | The vote's type. |
| `aibom:votedAt` | xsd:dateTime | When the vote was cast. |
| `aibom:comment` | xsd:string | Optional free-text reason. |

### Identifier model

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:identifier` | blank node | Platform/value pair. |
| `aibom:primaryIdentifier` | xsd:string | The `platform:value` form chosen as primary. |
| `aibom:canonicalLabel` | xsd:string | Human-readable display name. |
| `aibom:canonRuleVersion` | xsd:string | Version of the canonicalization rules used. |
| `aibom:platform` | xsd:string | `huggingface` / `github` / `arxiv` / `doi` / `url` / `name-only`. |
| `aibom:value` | xsd:string | Canonicalized identifier value. |
| `aibom:alias` | xsd:string | Original pre-canonical input string. |
| `aibom:isPlaceholder` | xsd:boolean | True for unresolvable references. |
| `aibom:potentialDuplicateOf` | Artifact | Soft-collision marker. |

### Per-field RDF-star annotations

| Predicate | Annotation of | Purpose |
|---|---|---|
| `aibom:assertedBy` | `<< claim pred value >>` → Source | Which data source asserted this field. |
| `aibom:conflictKind` | `<< claim pred value >>` → vocab individual | `noConflict` / `interSourceConflict` / `intraSourceConflict`. |
| `aibom:conflictsWith` | `<< claim pred value >>` → quoted triple | Pointer to the conflicting claim triple. |

### BOM-domain edges

| Predicate | Domain → Range | Purpose |
|---|---|---|
| `aibom:trainedOn` | Model → Dataset | Training data dependency. |
| `aibom:testedOn` | Model → Dataset | Evaluation/test dataset dependency. |
| `aibom:dependsOn` | Model → Model / Dataset → Dataset | General dependency between artifacts. |
| `aibom:describedIn` | Model → Paper | Paper that describes the model. |
| `aibom:hostedAt` | Model → CodeRepo | Code repo hosting the model. |
| `aibom:hasLicense` | Artifact → License (also `spdx:license`) | License attached to the artifact. |
| `aibom:suppliedBy` | Artifact → Supplier | Org or individual supplying the artifact. |
| `aibom:authoredBy` | Paper → Person | Author of a paper. |

### GenerationRun properties

| Predicate | Range | Purpose |
|---|---|---|
| `aibom:provider` | xsd:string | LLM provider key (`openrouter`, `openai`, `ollama`). |
| `aibom:llmModel` | xsd:string | The LLM model id. |
| `aibom:promptVersion` | xsd:string | Internal prompt version tag. |
| `aibom:codeVersion` | xsd:string | Code version (git SHA short form). |

## Vocabulary individuals

These are named IRIs used as the **object** of `aibom:voteKind` and
`aibom:conflictKind`. They are not predicates; they are constants.

### Vote-kind individuals

| Individual | Meaning |
|---|---|
| `aibom:trusted` | Explicit positive vote: "this claim looks correct." |
| `aibom:flagged` | Explicit negative vote: "this claim looks wrong." |
| `aibom:disputed` | Explicit ambiguity vote: "this claim is contested." |
| `aibom:implicit-use` | Implicit positive: the claim was used downstream (e.g. resolved by `bom show`). Python attribute: `implicit_use`. |
| `aibom:implicit-validate` | Implicit positive: validator confirmed the claim's structure. Python attribute: `implicit_validate`. |

### Conflict-kind individuals

| Individual | Meaning |
|---|---|
| `aibom:noConflict` | No other source asserts a competing value for this field. |
| `aibom:interSourceConflict` | Two distinct sources assert different values. |
| `aibom:intraSourceConflict` | A single source asserts inconsistent values (rare). |

## Constants

- `SCHEMA_VERSION = "1.0"` — current vocab version. Bump on any
  backward-incompatible predicate change.
- `CANON_RULE_VERSION = "1"` — current canonicalization rule version. Bump
  on any change that would split or merge previously-distinct nodes.
