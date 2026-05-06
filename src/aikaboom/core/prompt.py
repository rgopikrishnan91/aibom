"""
Prompt templates for AgenticRAG LLM calls.

Two-step workflow:
  1. prompt_detect_conflicts  — conflict detection only (no answer)
  2. prompt_generate_answer   — answer generation from (possibly filtered) chunks

Each prompt receives a three-part extraction spec
(``instruction`` / ``field_spec`` / ``output_guidance``) sourced from
the per-field question-bank ``extraction`` block. The structure
separates "what to extract" from "the SPDX/internal contract" from
"edge-case decision rules" so the LLM gets each piece in its own role.
"""


def prompt_detect_conflicts(field_name, instruction, field_spec, output_guidance, context):
    """Step 1: Detect conflicts between chunks. No answer generation."""
    return f"""You are a strict fact-checker.

Your ONLY job is to detect CONTRADICTIONS between text chunks about a specific field.
Do NOT generate an answer — only report conflicts.

FIELD NAME: {field_name}
INSTRUCTION: {instruction}
FIELD SPEC: {field_spec}
OUTPUT GUIDANCE: {output_guidance}

CHUNKS (each chunk is labelled with its source):
{context}

INSTRUCTIONS:
1. Read every chunk and identify statements directly relevant to "{field_name}".
2. Compare each pair of chunks that both mention "{field_name}".
3. For each contradictory pair, note the chunk numbers and their source labels.
4. Classify each contradiction:
   - INTERNAL CONFLICT: The two conflicting chunks come from the SAME source label.
   - EXTERNAL CONFLICT: The two conflicting chunks come from DIFFERENT source labels.

RULES:
- Only flag a conflict if BOTH statements are directly about "{field_name}".
- Complementary information (different aspects of the same field) is NOT a conflict.
- Minor wording differences that describe the same fact are NOT a conflict.
- If no chunks mention the field at all → both conflicts are "No".

YOUR RESPONSE MUST USE THIS EXACT FORMAT:

INTERNAL_CONFLICT: [No  |  Yes: <source> — Chunk <X> says "..." vs Chunk <Y> says "..."]
EXTERNAL_CONFLICT: [No  |  Yes: <source A> (Chunk <X>) says "..." while <source B> (Chunk <Y>) says "..."]

RESPONSE:"""


def prompt_generate_answer(field_name, instruction, field_spec, output_guidance, context):
    """Step 2: Generate answer from the provided chunks (may already be filtered)."""
    return f"""You are an AI model and dataset documentation expert.

Your task is to extract specific information from the provided source chunks.

FIELD NAME: {field_name}
INSTRUCTION: {instruction}
FIELD SPEC: {field_spec}
OUTPUT GUIDANCE: {output_guidance}

CHUNKS:
{context}

INSTRUCTIONS:
1. Read all provided chunks carefully.
2. Follow the INSTRUCTION: above precisely.
3. Honour the FIELD SPEC: legal values, format, and units.
4. Apply the OUTPUT GUIDANCE: for edge cases (missing info, conflicts, multi-value).
5. If multiple chunks provide information, synthesize them into a complete answer.
6. If no relevant information is found, respond with "Not found." — do NOT fabricate.

YOUR RESPONSE MUST USE THIS EXACT FORMAT:

ANSWER: [Your detailed answer synthesizing all provided chunks]

RESPONSE:"""


def prompt_no_documents(field_name, instruction):
    """Fallback when no source documents were retrieved."""
    return f"""You are analyzing information about an AI model, but no relevant source documents were retrieved.

FIELD NAME: {field_name}
INSTRUCTION: {instruction}

YOUR RESPONSE MUST BE IN THIS EXACT FORMAT:

ANSWER: Not found.

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
