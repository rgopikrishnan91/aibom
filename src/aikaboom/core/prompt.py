"""
Prompt templates for AgenticRAG LLM calls.

Two-step workflow:
  1. prompt_detect_conflicts  — group-anonymized consistency audit
  2. prompt_generate_answer   — answer generation from (possibly filtered) chunks

The conflict-detection prompt buckets chunks by source into groups
labelled A/B/C/... and asks the LLM to compare groups without telling
it which is which. This eliminates source-name bias and gives a
deterministic line-based output (``CLAIM_A``, ``CONFLICT_WITHIN_A``,
``CONFLICT_A_VS_B``, ...) that parses cleanly. The caller maps groups
back to source names after parsing.

The answer-generation prompt receives a three-part extraction spec
(``instruction`` / ``field_spec`` / ``output_guidance``) sourced from
the per-field question-bank ``extraction`` block.
"""

from collections import OrderedDict


def prompt_detect_conflicts(field_spec, group_chunks):
    """Group-anonymized consistency auditor prompt.

    Args:
        field_spec: The SPDX/field-spec text the auditor compares
            against. Drawn from the question-bank ``extraction.field_spec``.
        group_chunks: ``OrderedDict[str, list[str]]`` mapping group letter
            (``"A"``, ``"B"``, ...) to a list of chunk text strings already
            bucketed by source. Group letters DO NOT carry source identity
            into the prompt — the caller maps groups → sources after parse.

    Output format the LLM follows:
        CLAIM_<L>: <one sentence or "No relevant information">
        CONFLICT_WITHIN_<L>: No | Yes: "<stmt 1>" vs "<stmt 2>"
        CONFLICT_<A>_VS_<B>: No | Yes: A says "..." vs B says "..."
    """
    group_letters = list(group_chunks.keys())

    group_blocks = []
    for letter, chunks in group_chunks.items():
        if chunks:
            body = "\n".join(f"---\n{c}\n---" for c in chunks)
        else:
            body = "---\n(no chunks)\n---"
        group_blocks.append(f"GROUP {letter}:\n{body}")
    groups_section = "\n\n".join(group_blocks)

    step1_lines = "\n".join(
        f'CLAIM_{L}: <one sentence or "No relevant information">'
        for L in group_letters
    ) or "(no groups)"

    step2_lines = "\n".join(
        f'CONFLICT_WITHIN_{L}: No | Yes: "<statement 1>" vs "<statement 2>"'
        for L in group_letters
    ) or "(no groups)"

    pairs = [
        (a, b)
        for i, a in enumerate(group_letters)
        for b in group_letters[i + 1 :]
    ]
    step3_lines = "\n".join(
        f'CONFLICT_{a}_VS_{b}: No | Yes: {a} says "<claim>" vs {b} says "<claim>"'
        for a, b in pairs
    ) or "(only one group present — no pairwise comparison)"

    return f"""You are a consistency auditor. You compare independent documentation
sources about the same AI artifact and identify where they agree
or contradict each other. You do not judge whether claims are true
or false — only whether sources are consistent with each other.

FIELD BEING AUDITED:
{field_spec}

{groups_section}

EXAMPLE OUTPUT (for a different field, showing the expected format):

CLAIM_A: The model uses a decoder-only transformer with 7B parameters
CLAIM_B: The architecture is an encoder-decoder transformer with 6.7B parameters
CLAIM_C: No relevant information

CONFLICT_WITHIN_A: No
CONFLICT_WITHIN_B: No
CONFLICT_WITHIN_C: No

CONFLICT_A_VS_B: Yes: A says "decoder-only transformer" vs B says "encoder-decoder transformer"
CONFLICT_A_VS_C: No
CONFLICT_B_VS_C: No

NOW AUDIT THE FIELD ABOVE. Follow the three steps below.

STEP 1 — Extract each group's claim about this field in one sentence.
If a group contains no information about this field, write
"No relevant information."

{step1_lines}

STEP 2 — Check for contradictions WITHIN each group.
Two statements contradict if they cannot both be true about the same
aspect. Different details about different aspects are consistent,
not contradictory.
If no contradiction, write "No".
If contradiction, write "Yes" with the two conflicting statements.

{step2_lines}

STEP 3 — Check for contradictions BETWEEN each pair of groups.
Two groups contradict if they assert incompatible facts about the
same aspect of this field. One group having more detail than another
is not a contradiction.
If no contradiction, write "No".
If contradiction, write "Yes" with each group's conflicting claim.

{step3_lines}
"""


def format_chunks_for_answer(documents):
    """Render documents as plain ``---``-separated blocks for the answer prompt.

    No source labels, no chunk numbers. By the time chunks reach the answer
    prompt they have already been routed through consensus filtering — by
    construction they agree, so source attribution adds noise to the
    format-conversion task.
    """
    parts = ["---"]
    for doc in documents:
        parts.append(doc.page_content.strip())
        parts.append("---")
    return "\n".join(parts)


def prompt_generate_answer(instruction, field_spec, output_guidance, context):
    """Step 2: generate the field value from pre-filtered, source-agnostic chunks.

    The chunks reaching this prompt have already been consensus-filtered, so
    the answerer's job is format conversion + edge-case handling per
    ``field_spec`` / ``output_guidance``, not multi-source synthesis. The
    template follows the CTF (Context, Task, Format) shape with three
    universal rules; per-field behaviour comes from the question-bank
    extraction block, not the template.
    """
    return f"""FIELD:
{field_spec}

TASK:
{instruction}

RULES:
1. Use ONLY information from the context below. Do not add facts from your
   own knowledge.
2. Preserve exact names, numbers, and formatting from the source.
3. If the context contains no relevant information, return noAssertion.

FIELD-SPECIFIC GUIDANCE:
{output_guidance or "(No additional guidance.)"}

CONTEXT:
{context}

ANSWER:"""


def prompt_no_documents(field_name, instruction):
    """Fallback when no source documents were retrieved."""
    return f"""You are analyzing information about an AI model, but no relevant source documents were retrieved.

FIELD NAME: {field_name}
INSTRUCTION: {instruction}

YOUR RESPONSE MUST BE IN THIS EXACT FORMAT:

ANSWER: noAssertion

RESPONSE:"""


def prompt_direct_llm(field_name, instruction, field_spec, output_guidance, context):
    """Used by DirectLLM._generate_answer_direct — full source content, no chunking."""
    return f"""You are an AI model or dataset documentation expert. Your task is to extract specific information about an AI model or package from the provided sources AND detect any conflicts between sources.

You are looking for information related to the following field:

FIELD_NAME: {field_name}
INSTRUCTION: {instruction}
FIELD_SPEC: {field_spec}
OUTPUT_GUIDANCE: {output_guidance}

AVAILABLE SOURCES:
{context}

INSTRUCTIONS:
1. Carefully read through ALL the provided source materials.
2. Follow the INSTRUCTION: precisely; honour the FIELD_SPEC: contract.
3. Apply OUTPUT_GUIDANCE: for edge cases.
4. CONFLICT DETECTION - THIS IS CRITICAL:
   - If multiple sources provide information about the same aspect but with DIFFERENT or CONTRADICTORY details, this is a CONFLICT.
   - Pay attention to: different methods, different values, contradictory statements, incompatible descriptions.
   - Minor differences in wording are NOT conflicts if they describe the same thing.

5. If you find the answer:
   - Provide a clear, specific, and detailed response.
   - If information comes from multiple sources and they AGREE, synthesize them.
   - If sources DISAGREE or provide CONFLICTING information, note this explicitly.

6. If information is partially available:
   - Provide what you found.
   - Clearly state what information is missing.

7. If no relevant information is found:
   - State: "Not found."
   - DO NOT make up information.

YOUR RESPONSE MUST BE IN THIS EXACT FORMAT:

ANSWER: [Your detailed answer here, incorporating information from all sources]

CONFLICT_STATUS: [Either "No conflicts detected" OR "CONFLICT DETECTED: [describe the specific conflict between sources]"]

RESPONSE:"""
