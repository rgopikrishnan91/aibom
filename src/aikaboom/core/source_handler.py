import inspect
from collections import defaultdict


def _tag_similarity(v1, v2):
    """Jaccard similarity between two tag/category strings.

    Splits on ';', ',', and whitespace, lowercases each token, then computes
    |intersection| / |union|.  Returns a float in [0, 1].
    """
    import re
    def _tokenize(s):
        return set(t.strip().lower() for t in re.split(r'[;,\s]+', str(s)) if t.strip())
    t1, t2 = _tokenize(v1), _tokenize(v2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


class SourceHandler:
    @staticmethod
    def get_field_conflict(key, *sources, fuzzy=False, fuzzy_threshold=0.5):
        """
        Return (value, source_name, conflict) based on priority and conflict resolution rules.
        
        Rules:
        - If exactly one unique value exists across non-null sources → choose it, no conflict.
        - If two sources agree (same normalized value) → choose the majority value (ignore priority), 
        record the third source with its value in conflict.
        - If three different values (or 2 vs 1 but different due to normalization) → use priority mapping,
        record non-chosen sources with their values in conflict.
        - If one is null, choose from other two based on priority, record conflict source with value.
        - If all are null, return (None, None, None) and call OpenAI model.
        
        Returns:
            tuple: (value, source_name, conflict) where conflict is a string like "source1: value1, source2: value2"
                or None if no conflict exists.
        """
        # Inspect caller frame to map object IDs to variable names
        id_to_name = {}
        frame = inspect.currentframe()
        if frame and frame.f_back:
            id_to_name = {id(val): name for name, val in frame.f_back.f_locals.items()}

        # Collect all non-null values with their sources
        collected = []  # List of tuples: (value, source_name, source_dict)
        
        for src in sources:
            # Unpack explicit name if provided
            if isinstance(src, tuple) and len(src) == 2 and isinstance(src[1], dict):
                src_name, source = src
            else:
                source = src
                src_name = id_to_name.get(id(source))
                # Normalize variable names by dropping metadata suffix
                if isinstance(src_name, str) and '_' in src_name:
                    src_name = src_name.split('_', 1)[0]
            
            # Only consider dict-like sources
            if isinstance(source, dict) and key in source:
                val = source[key]
                # Treat None or empty string as missing
                if val is not None and val != "":
                    # Override LLM_Result name to 'paper'
                    if src_name and src_name.lower().startswith('llm'):
                        src_name = 'paper'
                    collected.append((val, src_name, source))
        
        # No valid values found
        if len(collected) == 0:
            return None, None, None
        
        # Exactly one source has a value
        if len(collected) == 1:
            return collected[0][0], collected[0][1], None
        
        # Normalize values for comparison (strip whitespace, lowercase)
        def normalize(v):
            if isinstance(v, str):
                return v.strip().lower()
            return str(v).strip().lower()

        # When fuzzy=True, merge collected entries whose tag similarity exceeds the
        # threshold into the same group instead of requiring an exact string match.
        if fuzzy and len(collected) >= 2:
            # Build groups by fuzzy similarity; greedy first-match
            groups = []  # list of lists of (val, src_name) pairs
            for val, src_name, _ in collected:
                placed = False
                for grp in groups:
                    rep_val = grp[0][0]  # representative value of the group
                    if _tag_similarity(val, rep_val) >= fuzzy_threshold:
                        grp.append((val, src_name))
                        placed = True
                        break
                if not placed:
                    groups.append([(val, src_name)])

            if len(groups) == 1:
                # All sources are similar enough — no conflict
                return collected[0][0], collected[0][1], None

            # Multiple dissimilar groups — pick largest (or first on tie), flag rest
            groups.sort(key=lambda g: len(g), reverse=True)
            chosen_val, chosen_src = groups[0][0]
            conflict_parts = [
                f"{src}: {val}" for grp in groups[1:] for val, src in grp
            ]
            conflict = ", ".join(conflict_parts) if conflict_parts else None
            return chosen_val, chosen_src, conflict

        # Group by normalized values (exact match path)
        value_groups = defaultdict(list)
        for val, src_name, _ in collected:
            norm_val = normalize(val)
            value_groups[norm_val].append((val, src_name))
        
        # Count unique normalized values
        unique_values = list(value_groups.keys())
        
        # Case 1: All sources agree (one unique normalized value)
        if len(unique_values) == 1:
            # Pick the first occurrence's original value
            return collected[0][0], collected[0][1], None
        
        # Case 2: Two sources agree (2 vs 1)
        if len(collected) == 3 and len(unique_values) == 2:
            # Find majority
            majority_norm = max(value_groups.keys(), key=lambda k: len(value_groups[k]))
            minority_norm = [k for k in unique_values if k != majority_norm][0]
            
            if len(value_groups[majority_norm]) == 2:
                # Two sources agree - choose majority, ignore priority
                chosen_val = value_groups[majority_norm][0][0]  # original value
                chosen_src = value_groups[majority_norm][0][1]  # source name
                
                # Conflict is the minority source
                minority_val, minority_src = value_groups[minority_norm][0]
                conflict = f"{minority_src}: {minority_val}"
                
                return chosen_val, chosen_src, conflict
        
        # Case 3: Multiple different values - use priority
        # Choose based on order (priority)
        chosen_val, chosen_src, _ = collected[0]
        
        # Build conflict string with all non-chosen sources
        conflict_parts = []
        for val, src_name, _ in collected[1:]:
            conflict_parts.append(f"{src_name}: {val}")
        
        conflict = ", ".join(conflict_parts) if conflict_parts else None
        
        return chosen_val, chosen_src, conflict
    
    @staticmethod
    def get_field(key, *sources, mode='priority'):
        """
        Return (value, source_name) based on the specified mode.
        
        Parameters:
        - key: The field key to retrieve
        - sources: Variable number of sources (dicts or tuples of (name, dict))
        - mode: 'priority' (default), 'earliest', or 'latest'
                - 'priority': Return first non-null value based on source order
                - 'earliest': Return the earliest date among all sources
                - 'latest': Return the latest date among all sources
        
        Returns:
        - (value, source_name) tuple
        - If the valid source is the LLM_Result, returns 'paper' as the source_name.
        - If no source has a valid value, returns (None, None).
        """
        from datetime import datetime
        
        # Inspect caller frame to map object IDs to variable names
        id_to_name = {}
        frame = inspect.currentframe()
        if frame and frame.f_back:
            id_to_name = {id(val): name for name, val in frame.f_back.f_locals.items()}

        # Helper function to parse dates
        def parse_date(val):
            """Try to parse various date formats"""
            if val is None or val == "":
                return None
            
            # If already a datetime object
            if isinstance(val, datetime):
                return val
            
            # Try common date formats
            date_formats = [
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%d/%m/%Y',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z',  # With timezone
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y',  # Just year
                '%Y-%m',  # Year-month
            ]
            
            if isinstance(val, str):
                # First try parsing with timezone info
                for fmt in date_formats:
                    try:
                        return datetime.strptime(val, fmt)
                    except ValueError:
                        continue
            
            return None
        
        # Priority mode - return first non-null value
        if mode == 'priority':
            for src in sources:
                # Unpack explicit name if provided
                if isinstance(src, tuple) and len(src) == 2 and isinstance(src[1], dict):
                    src_name, source = src
                else:
                    source = src
                    src_name = id_to_name.get(id(source))
                    # Normalize variable names by dropping metadata suffix
                    if isinstance(src_name, str) and '_' in src_name:
                        src_name = src_name.split('_', 1)[0]
                
                # Only consider dict-like sources
                if isinstance(source, dict) and key in source:
                    val = source[key]
                    # Treat None or empty string as missing
                    if val is not None and val != "":
                        # Override LLM_Result name to 'paper'
                        if src_name and src_name.lower().startswith('llm'):
                            src_name = 'paper'
                        return val, src_name
            
            # No source provided a valid value
            return None, None
        
        # Date comparison modes - collect all valid dates
        elif mode in ['earliest', 'latest']:
            collected = []  # List of tuples: (original_value, parsed_date, source_name)
            
            for src in sources:
                # Unpack explicit name if provided
                if isinstance(src, tuple) and len(src) == 2 and isinstance(src[1], dict):
                    src_name, source = src
                else:
                    source = src
                    src_name = id_to_name.get(id(source))
                    # Normalize variable names by dropping metadata suffix
                    if isinstance(src_name, str) and '_' in src_name:
                        src_name = src_name.split('_', 1)[0]
                
                # Only consider dict-like sources
                if isinstance(source, dict) and key in source:
                    val = source[key]
                    # Treat None or empty string as missing
                    if val is not None and val != "":
                        parsed = parse_date(val)
                        if parsed:
                            # Override LLM_Result name to 'paper'
                            if src_name and src_name.lower().startswith('llm'):
                                src_name = 'paper'
                            collected.append((val, parsed, src_name))
                        else:
                            print(f"⚠️ Could not parse date '{val}' from {src_name} for key '{key}'")
            
            # No valid dates found
            if not collected:
                return None, None
            
            # Find earliest or latest
            if mode == 'earliest':
                result = min(collected, key=lambda x: x[1])
            else:  # latest
                result = max(collected, key=lambda x: x[1])
            
            return result[0], result[2]  # Return original value and source name
        
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'priority', 'earliest', or 'latest'")
