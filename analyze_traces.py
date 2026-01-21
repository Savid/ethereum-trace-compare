#!/usr/bin/env python3
"""Analyze trace comparison data and generate client quirk reports."""

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path


def format_value(val, max_len=100):
    """Format a value for display, truncating if necessary."""
    if val is None:
        return "None"
    if isinstance(val, dict):
        s = json.dumps(val, sort_keys=True)
    elif isinstance(val, list):
        s = str(val)
    else:
        s = str(val)

    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def describe_difference(client_val, majority_val) -> str:
    """Describe what's different between two values."""
    if majority_val is None:
        return "no majority value"

    # Handle lists
    if isinstance(client_val, list) and isinstance(majority_val, list):
        if len(client_val) != len(majority_val):
            return f"list length differs ({len(client_val)} vs {len(majority_val)})"

        # Find first differing index
        for i, (c, m) in enumerate(zip(client_val, majority_val)):
            if c != m:
                c_short = str(c)[:20] + "..." if len(str(c)) > 20 else str(c)
                m_short = str(m)[:20] + "..." if len(str(m)) > 20 else str(m)
                return f"item[{i}] differs: `{c_short}` vs `{m_short}`"

        return "values differ (unknown reason)"

    # Handle dicts
    if isinstance(client_val, dict) and isinstance(majority_val, dict):
        client_keys = set(client_val.keys())
        majority_keys = set(majority_val.keys())

        if client_keys != majority_keys:
            missing = majority_keys - client_keys
            extra = client_keys - majority_keys
            parts = []
            if missing:
                parts.append(f"missing keys: {missing}")
            if extra:
                parts.append(f"extra keys: {extra}")
            return "; ".join(parts)

        # Same keys, find differing value
        for key in client_keys:
            if client_val[key] != majority_val[key]:
                return f"key `{key}` differs"

        return "values differ (unknown reason)"

    # Handle strings (common case: hex formatting)
    if isinstance(client_val, str) and isinstance(majority_val, str):
        if client_val.lower() == majority_val.lower():
            return "case differs (e.g., hex casing)"
        if client_val.lstrip('0x') == majority_val.lstrip('0x'):
            return "0x prefix differs"
        if len(client_val) != len(majority_val):
            return f"string length differs ({len(client_val)} vs {len(majority_val)})"
        return "string content differs"

    # Different types
    if type(client_val) != type(majority_val):
        return f"type differs ({type(client_val).__name__} vs {type(majority_val).__name__})"

    return "values differ"


def load_comparisons(traces_dir: Path) -> list[dict]:
    """Load all _comparison.json files."""
    comparisons = []
    for comp_file in traces_dir.rglob("_comparison.json"):
        try:
            with open(comp_file) as f:
                data = json.load(f)
                data["_source_file"] = str(comp_file)
                comparisons.append(data)
        except Exception as e:
            print(f"Error loading {comp_file}: {e}")
    return comparisons


def load_opcodes_for_tx(traces_dir: Path, block: str, tx_hash: str) -> list[str]:
    """Load opcode list from a raw trace file (any client).

    Returns a list of opcodes indexed by structLog index.
    """
    tx_dir = traces_dir / block / tx_hash
    if not tx_dir.exists():
        return []

    for client_file in tx_dir.glob("*.json"):
        if client_file.name == "_comparison.json":
            continue
        try:
            with open(client_file) as f:
                data = json.load(f)
            if "structLogs" in data:
                return [log.get("op", "unknown") for log in data["structLogs"]]
        except Exception:
            continue
    return []


def analyze_differences(comparisons: list[dict], traces_dir: Path) -> dict:
    """Analyze all differences and categorize by client."""
    analysis = {
        "summary": {
            "total_comparisons": len(comparisons),
            "skipped_due_to_errors": 0,
            "analyzed_comparisons": 0,
            "diff_types": defaultdict(int),
        },
        "client_quirks": defaultdict(lambda: {
            "missing_keys": defaultdict(lambda: defaultdict(int)),  # key -> op -> count
            "has_extra_keys": defaultdict(lambda: defaultdict(int)),  # key -> op -> count
            "type_mismatches": defaultdict(list),
            "value_mismatches": defaultdict(list),
            "structlog_count_outlier": [],
            "gas_differences": [],
            "return_value_differences": [],
        }),
        "errors_by_client": defaultdict(int),
        # New: aggregated format quirks (not bugs, just formatting differences)
        "format_quirks": {
            "returnValue_encoding": {
                "by_client": defaultdict(lambda: defaultdict(int)),  # {client: {format: count}}
                "examples": [],  # up to 3 examples with per-client values
            },
            "memory_format": {
                "by_client": defaultdict(lambda: defaultdict(int)),  # {client: {format: count}}
                "examples": [],  # up to 3 examples with per-client values
            },
            "storage_format": {
                "by_client": defaultdict(lambda: defaultdict(int)),  # {client: {format: count}}
                "examples": [],  # up to 3 examples with per-client values
            },
            "optional_fields": {
                "by_client": defaultdict(lambda: defaultdict(int)),  # {client: {field: count}}
                "examples": defaultdict(list),  # {field: [examples]}
            },
        },
    }

    # Cache for opcodes per transaction
    opcodes_cache = {}

    for comp in comparisons:
        # Parse block and tx from source file path: traces/BLOCK/TX/_comparison.json
        source_file = comp.get("_source_file", "")
        parts = Path(source_file).parts
        block_number = parts[-3] if len(parts) >= 3 else "unknown"
        tx_hash = parts[-2] if len(parts) >= 2 else "unknown"

        # Track errors and skip comparisons where any client errored
        errors = comp.get("errors", {})
        if errors:
            for client, error in errors.items():
                analysis["errors_by_client"][client] += 1
            analysis["summary"]["skipped_due_to_errors"] += 1
            continue  # Skip this transaction - incomplete data

        analysis["summary"]["analyzed_comparisons"] += 1

        # Load opcodes for this transaction (cached)
        cache_key = f"{block_number}/{tx_hash}"
        if cache_key not in opcodes_cache:
            opcodes_cache[cache_key] = load_opcodes_for_tx(traces_dir, block_number, tx_hash)
        opcodes = opcodes_cache[cache_key]

        # Track differences
        for diff in comp.get("differences", []):
            diff_type = diff.get("type")
            analysis["summary"]["diff_types"][diff_type] += 1

            if diff_type == "missing_key":
                key = diff.get("key")
                index = diff.get("index", 0)
                op = opcodes[index] if index is not None and index < len(opcodes) else "unknown"
                for client in diff.get("missing_in", []):
                    analysis["client_quirks"][client]["missing_keys"][key][op] += 1
                for client in diff.get("present_in", []):
                    analysis["client_quirks"][client]["has_extra_keys"][key][op] += 1

            elif diff_type == "type_mismatch":
                types = diff.get("types", {})
                key = diff.get("key")
                # Find which client is the outlier
                type_counts = Counter(types.values())
                if len(type_counts) > 1:
                    majority_type = type_counts.most_common(1)[0][0]
                    for client, t in types.items():
                        if t != majority_type:
                            analysis["client_quirks"][client]["type_mismatches"][key].append({
                                "client_type": t,
                                "majority_type": majority_type,
                                "index": diff.get("index"),
                                "block": block_number,
                                "tx_hash": tx_hash,
                            })

            elif diff_type == "value_mismatch":
                values = diff.get("values", {})
                key = diff.get("key")

                # Normalize values for comparison
                normalized = {}
                for client, val in values.items():
                    if isinstance(val, dict):
                        normalized[client] = json.dumps(val, sort_keys=True)
                    else:
                        normalized[client] = str(val)

                value_counts = Counter(normalized.values())
                if len(value_counts) > 1:
                    majority_val = value_counts.most_common(1)[0][0]
                    majority_clients = [c for c, v in normalized.items() if v == majority_val]

                    for client, val in normalized.items():
                        if val != majority_val:
                            analysis["client_quirks"][client]["value_mismatches"][key].append({
                                "client_value": values[client],
                                "majority_value": [v for c, v in values.items() if c in majority_clients][0] if majority_clients else None,
                                "majority_clients": majority_clients,
                                "index": diff.get("index"),
                                "block": block_number,
                                "tx_hash": tx_hash,
                            })

            elif diff_type == "structlog_count_mismatch":
                details = diff.get("details", {})
                count_freq = Counter(details.values())
                if len(count_freq) > 1:
                    majority_count = count_freq.most_common(1)[0][0]
                    majority_clients = [c for c, cnt in details.items() if cnt == majority_count]
                    for client, count in details.items():
                        if count != majority_count:
                            analysis["client_quirks"][client]["structlog_count_outlier"].append({
                                "block": block_number,
                                "tx_hash": tx_hash,
                                "client_count": count,
                                "majority_count": majority_count,
                                "majority_clients": majority_clients,
                                "all_counts": details,
                            })

            elif diff_type == "top_level_mismatch":
                key = diff.get("key")
                values = diff.get("values", {})

                # Normalize
                normalized = {}
                for client, val in values.items():
                    if isinstance(val, dict):
                        normalized[client] = json.dumps(val, sort_keys=True)
                    else:
                        normalized[client] = str(val)

                value_counts = Counter(normalized.values())
                if len(value_counts) > 1:
                    majority_val = value_counts.most_common(1)[0][0]
                    majority_clients = [c for c, v in normalized.items() if v == majority_val]

                    for client, val in normalized.items():
                        if val != majority_val:
                            if key == "gas":
                                analysis["client_quirks"][client]["gas_differences"].append({
                                    "block": block_number,
                                    "tx_hash": tx_hash,
                                    "client_value": values[client],
                                    "majority_value": [v for c, v in values.items() if c in majority_clients][0] if majority_clients else None,
                                    "majority_clients": majority_clients,
                                })
                            elif key == "returnValue":
                                analysis["client_quirks"][client]["return_value_differences"].append({
                                    "block": block_number,
                                    "tx_hash": tx_hash,
                                    "client_value": values[client],
                                    "majority_value": [v for c, v in values.items() if c in majority_clients][0] if majority_clients else None,
                                    "majority_clients": majority_clients,
                                })

        # Aggregate format quirks from comparison
        format_quirks = comp.get("format_quirks", {})

        # returnValue encoding quirks
        rv_data = format_quirks.get("returnValue_encoding", {})
        for client, data in rv_data.items():
            if isinstance(data, dict):
                fmt = data.get("format", "unknown")
                example = data.get("example")
            else:
                fmt = data  # backward compat with old format
                example = None
            analysis["format_quirks"]["returnValue_encoding"]["by_client"][client][fmt] += 1

        # Collect example for returnValue encoding if we have fewer than 3
        if len(analysis["format_quirks"]["returnValue_encoding"]["examples"]) < 3 and rv_data:
            example_data = {
                "block": block_number,
                "tx_hash": tx_hash,
                "values": {}
            }
            for client, data in rv_data.items():
                if isinstance(data, dict):
                    example_data["values"][client] = data.get("example")
                else:
                    example_data["values"][client] = data
            if any(v is not None for v in example_data["values"].values()):
                analysis["format_quirks"]["returnValue_encoding"]["examples"].append(example_data)

        # memory format quirks
        mem_data = format_quirks.get("memory_format", {})
        for client, data in mem_data.items():
            if isinstance(data, dict):
                fmt = data.get("format", "unknown")
                example = data.get("example")
            else:
                fmt = data  # backward compat
                example = None
            analysis["format_quirks"]["memory_format"]["by_client"][client][fmt] += 1

        # Collect example for memory format if we have fewer than 3
        if len(analysis["format_quirks"]["memory_format"]["examples"]) < 3 and mem_data:
            example_data = {
                "block": block_number,
                "tx_hash": tx_hash,
                "values": {}
            }
            for client, data in mem_data.items():
                if isinstance(data, dict):
                    example_data["values"][client] = data.get("example")
                else:
                    example_data["values"][client] = None
            if any(v is not None for v in example_data["values"].values()):
                analysis["format_quirks"]["memory_format"]["examples"].append(example_data)

        # storage format quirks
        stor_data = format_quirks.get("storage_format", {})
        for client, data in stor_data.items():
            if isinstance(data, dict):
                fmt = data.get("format", "unknown")
            else:
                fmt = data  # backward compat
            analysis["format_quirks"]["storage_format"]["by_client"][client][fmt] += 1

        # Collect example for storage format if we have fewer than 3
        if len(analysis["format_quirks"]["storage_format"]["examples"]) < 3 and stor_data:
            example_data = {
                "block": block_number,
                "tx_hash": tx_hash,
                "values": {}
            }
            for client, data in stor_data.items():
                if isinstance(data, dict):
                    example_data["values"][client] = {
                        "key": data.get("example_key"),
                        "value": data.get("example_value"),
                    }
                else:
                    example_data["values"][client] = None
            if any(v is not None for v in example_data["values"].values()):
                analysis["format_quirks"]["storage_format"]["examples"].append(example_data)

        # optional fields quirks - handle both old format (backward compat) and new format
        opt_data = format_quirks.get("optional_fields", {})
        # Check if this is the old format (direct client -> fields mapping) or new format (with by_client key)
        # In old format, keys would be client names. In new format, keys would be "by_client" and "examples"
        if "by_client" not in opt_data and opt_data:
            # Old format - backward compat
            for client, fields in opt_data.items():
                if isinstance(fields, dict):
                    for field, count in fields.items():
                        analysis["format_quirks"]["optional_fields"]["by_client"][client][field] += count
        else:
            # New format with by_client structure (nothing to do - examples are stored separately)
            pass

        # Collect examples from optional_fields_examples (new format from compare_traces)
        opt_examples = format_quirks.get("optional_fields_examples", {})
        for field, examples in opt_examples.items():
            # Only collect up to 3 examples per field
            if len(analysis["format_quirks"]["optional_fields"]["examples"][field]) < 3:
                for ex in examples:
                    if len(analysis["format_quirks"]["optional_fields"]["examples"][field]) >= 3:
                        break
                    ex_with_context = {
                        "block": block_number,
                        "tx_hash": tx_hash,
                        "client_values": ex.get("client_values", {})
                    }
                    analysis["format_quirks"]["optional_fields"]["examples"][field].append(ex_with_context)

    # Convert nested defaultdicts to regular dicts for JSON serialization
    analysis["format_quirks"]["returnValue_encoding"]["by_client"] = {
        k: dict(v) for k, v in analysis["format_quirks"]["returnValue_encoding"]["by_client"].items()
    }
    analysis["format_quirks"]["memory_format"]["by_client"] = {
        k: dict(v) for k, v in analysis["format_quirks"]["memory_format"]["by_client"].items()
    }
    analysis["format_quirks"]["storage_format"]["by_client"] = {
        k: dict(v) for k, v in analysis["format_quirks"]["storage_format"]["by_client"].items()
    }
    analysis["format_quirks"]["optional_fields"] = {
        "by_client": {k: dict(v) for k, v in analysis["format_quirks"]["optional_fields"]["by_client"].items()},
        "examples": dict(analysis["format_quirks"]["optional_fields"]["examples"]),
    }

    return analysis


def generate_client_report(client: str, quirks: dict, all_clients: list[str],
                           total_txs: int, client_tx_count: int, error_count: int,
                           format_quirks: dict | None = None) -> str:
    """Generate a markdown report for a single client."""
    lines = [f"# {client.upper()} - debug_traceTransaction Quirks Report\n"]
    lines.append("## Summary\n")
    lines.append(f"- **Total transactions analyzed**: {total_txs:,}\n")

    if client_tx_count < total_txs:
        lines.append(f"- **{client} successful traces**: {client_tx_count:,} ({100*client_tx_count/total_txs:.1f}%)\n")
        lines.append(f"- **{client} errors**: {error_count:,}\n")
    else:
        lines.append(f"- **{client} successful traces**: {client_tx_count:,} (100%)\n")

    lines.append(f"\nThis report documents differences in `debug_traceTransaction` output from **{client}** ")
    lines.append(f"compared to the majority behavior of other clients ({', '.join(c for c in all_clients if c != client)}).\n")

    has_quirks = False

    # Missing keys
    if quirks["missing_keys"]:
        has_quirks = True
        lines.append("\n## Missing Keys\n")
        lines.append(f"{client} omits these keys that other clients include:\n")

        # Sort by total occurrences across all opcodes
        sorted_keys = sorted(
            quirks["missing_keys"].items(),
            key=lambda x: -sum(x[1].values())
        )
        for key, op_counts in sorted_keys:
            total = sum(op_counts.values())
            lines.append(f"\n### `{key}` ({total:,} occurrences)\n")
            lines.append("| Opcode | Count |\n|--------|-------|\n")
            for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
                lines.append(f"| {op} | {count:,} |\n")

    # Extra keys
    if quirks["has_extra_keys"]:
        has_quirks = True
        lines.append("\n## Extra Keys (Present When Others Omit)\n")
        lines.append(f"{client} includes these keys that other clients omit:\n")

        # Sort by total occurrences across all opcodes
        sorted_keys = sorted(
            quirks["has_extra_keys"].items(),
            key=lambda x: -sum(x[1].values())
        )
        for key, op_counts in sorted_keys:
            total = sum(op_counts.values())
            lines.append(f"\n### `{key}` ({total:,} occurrences)\n")
            lines.append("| Opcode | Count |\n|--------|-------|\n")
            for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
                lines.append(f"| {op} | {count:,} |\n")

    # Type mismatches
    if quirks["type_mismatches"]:
        has_quirks = True
        lines.append("\n## Type Mismatches\n")
        lines.append(f"{client} returns different types for these keys:\n")
        for key, mismatches in quirks["type_mismatches"].items():
            count = len(mismatches)
            lines.append(f"\n### `{key}` ({count:,} occurrences)\n")

            # Show a few examples
            examples = mismatches[:3]
            for i, ex in enumerate(examples, 1):
                block = ex.get("block", "?")
                tx = ex.get("tx_hash", "?")

                lines.append(f"\n**Example {i}** (block {block}, tx {tx}, index {ex.get('index')}):\n")
                lines.append(f"- **{client}**: `{ex['client_type']}`\n")
                lines.append(f"- **Others**: `{ex['majority_type']}`\n")

    # Value mismatches
    if quirks["value_mismatches"]:
        has_quirks = True
        lines.append("\n## Value Mismatches\n")
        lines.append(f"{client} returns different values for these keys:\n")

        for key, mismatches in sorted(quirks["value_mismatches"].items(), key=lambda x: -len(x[1])):
            count = len(mismatches)
            lines.append(f"\n### `{key}` ({count:,} occurrences)\n")

            # Show a few examples
            examples = mismatches[:3]
            for i, ex in enumerate(examples, 1):
                # Format client value - no truncation for full visibility
                client_val = format_value(ex["client_value"], max_len=10000)

                # Format majority value - no truncation for full visibility
                majority_val = format_value(ex.get("majority_value"), max_len=10000)

                # Describe what's different
                diff_desc = describe_difference(ex["client_value"], ex.get("majority_value"))

                # Get block and tx
                block = ex.get("block", "?")
                tx = ex.get("tx_hash", "?")

                lines.append(f"\n**Example {i}** (block {block}, tx `{tx}`, index {ex.get('index')}):\n")
                lines.append(f"- **Difference**: {diff_desc}\n")
                lines.append("| Client | Value |\n|--------|-------|\n")
                lines.append(f"| {client} | `{client_val}` |\n")
                majority_clients = ex['majority_clients']
                if len(majority_clients) == 1:
                    lines.append(f"| {majority_clients[0]} | `{majority_val}` |\n")
                else:
                    lines.append(f"| Majority ({', '.join(majority_clients)}) | `{majority_val}` |\n")

    # Gas differences
    if quirks["gas_differences"]:
        has_quirks = True
        lines.append("\n## Top-Level `gas` Differences\n")
        lines.append(f"{client} reports different gas values at the top level:\n")
        count = len(quirks["gas_differences"])
        lines.append(f"- **Occurrences**: {count}\n")

        # Show up to 3 examples
        examples = quirks["gas_differences"][:3]
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n**Example {i}** (block {ex['block']}, tx {ex['tx_hash']}):\n")
            lines.append(f"- **{client}**: `{ex['client_value']}`\n")
            majority_clients = ex.get('majority_clients', [])
            if len(majority_clients) == 1:
                lines.append(f"- **{majority_clients[0]}**: `{ex['majority_value']}`\n")
            else:
                lines.append(f"- **Majority** ({', '.join(majority_clients)}): `{ex['majority_value']}`\n")

        if count > 3:
            lines.append(f"\n... and {count - 3} more\n")

    # Return value differences
    if quirks["return_value_differences"]:
        has_quirks = True
        lines.append("\n## Top-Level `returnValue` Differences\n")
        lines.append(f"{client} formats returnValue differently:\n")
        count = len(quirks["return_value_differences"])
        lines.append(f"- **Occurrences**: {count}\n")

        # Show up to 3 examples
        examples = quirks["return_value_differences"][:3]
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n**Example {i}** (block {ex['block']}, tx {ex['tx_hash']}):\n")
            lines.append(f"- **{client}**: `{repr(ex['client_value'])}`\n")
            majority_clients = ex.get('majority_clients', [])
            if len(majority_clients) == 1:
                lines.append(f"- **{majority_clients[0]}**: `{repr(ex['majority_value'])}`\n")
            else:
                lines.append(f"- **Majority** ({', '.join(majority_clients)}): `{repr(ex['majority_value'])}`\n")

        if count > 3:
            lines.append(f"\n... and {count - 3} more\n")

    # Structlog count outlier
    if quirks["structlog_count_outlier"]:
        has_quirks = True
        count = len(quirks["structlog_count_outlier"])
        lines.append("\n## StructLog Count Mismatches\n")
        lines.append(f"{client} returned a different number of structLog entries {count:,} times.\n")

        # Show up to 5 examples
        examples = quirks["structlog_count_outlier"][:5]
        for i, ex in enumerate(examples, 1):
            lines.append(f"\n**Example {i}** (block {ex['block']}, tx {ex['tx_hash']}):\n")
            lines.append(f"- **{client}**: {ex['client_count']:,} entries\n")
            lines.append(f"- **Majority** ({', '.join(ex['majority_clients'])}): {ex['majority_count']:,} entries\n")
            # Show all counts for full picture
            lines.append("- All clients: ")
            lines.append(", ".join(f"{c}={cnt:,}" for c, cnt in sorted(ex["all_counts"].items())))
            lines.append("\n")

        if count > 5:
            lines.append(f"\n... and {count - 5:,} more occurrences\n")

    # Format Quirks Section (non-bug formatting differences)
    if format_quirks:
        has_format_quirks_for_client = False
        format_quirks_lines = []

        # Check returnValue encoding
        rv_by_client = format_quirks.get("returnValue_encoding", {}).get("by_client", {})
        rv_examples = format_quirks.get("returnValue_encoding", {}).get("examples", [])
        if client in rv_by_client:
            has_format_quirks_for_client = True
            encodings = rv_by_client[client]
            format_quirks_lines.append("### returnValue Encoding\n")
            format_quirks_lines.append("How empty return values are represented. These are semantically equivalent.\n\n")
            format_quirks_lines.append("| Format | Count |\n|--------|-------|\n")
            for fmt, cnt in sorted(encodings.items(), key=lambda x: -x[1]):
                format_quirks_lines.append(f"| `{fmt}` | {cnt:,} |\n")

            # Show examples
            if rv_examples:
                for i, ex in enumerate(rv_examples[:3], 1):
                    tx_hash = ex['tx_hash']
                    format_quirks_lines.append(f"\n**Example {i}** (block {ex['block']}, tx `{tx_hash}`):\n")
                    format_quirks_lines.append("| Client | Value |\n|--------|-------|\n")
                    for c, val in sorted(ex.get("values", {}).items()):
                        val_str = str(val) if val is not None else "null"
                        val_display = f"`{val_str}`"
                        format_quirks_lines.append(f"| {c} | {val_display} |\n")
            format_quirks_lines.append("\n")

        # Check memory format
        mem_by_client = format_quirks.get("memory_format", {}).get("by_client", {})
        mem_examples = format_quirks.get("memory_format", {}).get("examples", [])
        if client in mem_by_client:
            has_format_quirks_for_client = True
            formats = mem_by_client[client]
            format_quirks_lines.append("### Memory Format\n")
            format_quirks_lines.append("How memory words are formatted in structLogs.\n\n")
            format_quirks_lines.append("| Format | Count |\n|--------|-------|\n")
            for fmt, cnt in sorted(formats.items(), key=lambda x: -x[1]):
                format_quirks_lines.append(f"| `{fmt}` | {cnt:,} |\n")

            # Show examples
            if mem_examples:
                for i, ex in enumerate(mem_examples[:3], 1):
                    tx_hash = ex['tx_hash']
                    format_quirks_lines.append(f"\n**Example {i}** (block {ex['block']}, tx `{tx_hash}`):\n")
                    format_quirks_lines.append("| Client | Value |\n|--------|-------|\n")
                    for c, val in sorted(ex.get("values", {}).items()):
                        val_str = str(val) if val is not None else "null"
                        val_display = f"`{val_str}`"
                        format_quirks_lines.append(f"| {c} | {val_display} |\n")
            format_quirks_lines.append("\n")

        # Check storage format - simplify by categorizing
        stor_by_client = format_quirks.get("storage_format", {}).get("by_client", {})
        stor_examples = format_quirks.get("storage_format", {}).get("examples", [])
        if client in stor_by_client:
            has_format_quirks_for_client = True
            formats = stor_by_client[client]
            format_quirks_lines.append("### Storage Format\n")
            format_quirks_lines.append("How storage keys and values are formatted. Clients differ in zero-padding.\n\n")

            # Categorize the formats
            uses_short_keys = 0
            uses_padded_keys = 0
            uses_short_vals = 0
            uses_padded_vals = 0
            total = 0

            for fmt, cnt in formats.items():
                total += cnt
                # Parse format like "key:short_1char,val:short_60char" or "key:64char_no_prefix,val:64char_no_prefix"
                if "key:short_" in fmt or "key:with_0x" in fmt:
                    uses_short_keys += cnt
                elif "key:64char" in fmt:
                    uses_padded_keys += cnt
                if "val:short_" in fmt or "val:with_0x" in fmt:
                    uses_short_vals += cnt
                elif "val:64char" in fmt:
                    uses_padded_vals += cnt

            format_quirks_lines.append("| Aspect | Style | Count |\n|--------|-------|-------|\n")
            if uses_short_keys > 0:
                format_quirks_lines.append(f"| Keys | Compact (e.g., `0`, `1`) | {uses_short_keys:,} |\n")
            if uses_padded_keys > 0:
                format_quirks_lines.append(f"| Keys | 64-char padded (e.g., `0000...0000`) | {uses_padded_keys:,} |\n")
            if uses_short_vals > 0:
                format_quirks_lines.append(f"| Values | Compact (e.g., `0x1`, `abc`) | {uses_short_vals:,} |\n")
            if uses_padded_vals > 0:
                format_quirks_lines.append(f"| Values | 64-char padded | {uses_padded_vals:,} |\n")

            # Show examples
            if stor_examples:
                for i, ex in enumerate(stor_examples[:3], 1):
                    tx_hash = ex['tx_hash']
                    format_quirks_lines.append(f"\n**Example {i}** (block {ex['block']}, tx `{tx_hash}`):\n")
                    format_quirks_lines.append("| Client | Key | Value |\n|--------|-----|-------|\n")
                    for c, data in sorted(ex.get("values", {}).items()):
                        if data and isinstance(data, dict):
                            key_str = str(data.get("key", ""))
                            val_str = str(data.get("value", ""))
                            key_display = f"`{key_str}`"
                            val_display = f"`{val_str}`"
                            format_quirks_lines.append(f"| {c} | {key_display} | {val_display} |\n")
            format_quirks_lines.append("\n")

        # Check optional fields - with proper descriptions for each field
        opt_by_client = format_quirks.get("optional_fields", {}).get("by_client", {})
        opt_examples = format_quirks.get("optional_fields", {}).get("examples", {})

        if client in opt_by_client:
            has_format_quirks_for_client = True
            fields = opt_by_client[client]
            format_quirks_lines.append("### Optional Fields\n")
            format_quirks_lines.append("Fields this client includes that others may omit.\n\n")

            field_descriptions = {
                "error": "Includes `error: null` on every step (others omit when no error)",
                "storage": "Includes `storage: {}` on every step (others omit when empty)",
                "returnData": "Includes `returnData` field on every step",
                "structLog_for_simple_transfer": "Emits a structLog entry for simple ETH transfers (no EVM execution), while other clients emit 0 entries",
            }

            # Show each field with its examples grouped together
            for field, cnt in sorted(fields.items(), key=lambda x: -x[1]):
                desc = field_descriptions.get(field, "—")
                format_quirks_lines.append(f"#### `{field}` ({cnt:,} occurrences)\n")
                format_quirks_lines.append(f"{desc}\n\n")

                # Show examples for this specific field
                if field in opt_examples and opt_examples[field]:
                    for i, ex in enumerate(opt_examples[field][:3], 1):
                        format_quirks_lines.append(f"**Example {i}** (block {ex['block']}, tx `{ex['tx_hash']}`):\n")
                        format_quirks_lines.append("| Client | Value |\n|--------|-------|\n")
                        for c, val in sorted(ex.get("client_values", {}).items()):
                            format_quirks_lines.append(f"| {c} | `{val}` |\n")
                        format_quirks_lines.append("\n")
            format_quirks_lines.append("\n")

        if has_format_quirks_for_client:
            lines.append("\n## Format Quirks (Not Bugs)\n")
            lines.append("These are formatting differences that don't affect trace correctness:\n\n")
            lines.extend(format_quirks_lines)

    if not has_quirks:
        lines.append("\nNo significant quirks detected for this client.\n")

    return "".join(lines)


async def ai_analyze_differences(analysis: dict) -> tuple[str, str | None]:
    """Use Claude Agent SDK to provide intelligent analysis of differences.

    Returns:
        Tuple of (result_text, error_message). error_message is None on success.
    """
    from claude_agent_sdk import query, ClaudeAgentOptions, CLINotFoundError, ClaudeSDKError

    # Prepare a summary of the data for Claude
    summary_data = {
        "total_transactions": analysis["summary"]["total_comparisons"],
        "difference_types": dict(analysis["summary"]["diff_types"]),
        "client_quirks": {},
        "format_quirks": analysis.get("format_quirks", {}),
    }

    for client, quirks in analysis["client_quirks"].items():
        # Convert nested defaultdicts to regular dicts for JSON serialization
        missing_keys_data = {key: dict(op_counts) for key, op_counts in quirks["missing_keys"].items()}
        extra_keys_data = {key: dict(op_counts) for key, op_counts in quirks["has_extra_keys"].items()}

        # Get type mismatch examples
        type_mismatch_examples = {}
        for key, mismatches in quirks["type_mismatches"].items():
            if mismatches:
                ex = mismatches[0]
                type_mismatch_examples[key] = {
                    "client_type": ex["client_type"],
                    "majority_type": ex["majority_type"],
                }

        # Get returnValue example if exists
        return_value_example = None
        if quirks["return_value_differences"]:
            ex = quirks["return_value_differences"][0]
            return_value_example = {
                "client_value": repr(ex["client_value"]),
                "majority_value": repr(ex["majority_value"]),
            }

        summary_data["client_quirks"][client] = {
            "missing_keys": missing_keys_data,
            "extra_keys": extra_keys_data,
            "value_mismatch_keys": list(quirks["value_mismatches"].keys()),
            "value_mismatch_count": sum(len(v) for v in quirks["value_mismatches"].values()),
            "type_mismatch_count": sum(len(v) for v in quirks["type_mismatches"].values()),
            "type_mismatch_examples": type_mismatch_examples,
            "gas_differences": len(quirks["gas_differences"]),
            "return_value_differences": len(quirks["return_value_differences"]),
            "return_value_example": return_value_example,
            "structlog_count_mismatches": len(quirks["structlog_count_outlier"]),
        }

    prompt = f"""Analyze these debug_traceTransaction differences across Ethereum execution clients.

Data:
{json.dumps(summary_data, indent=2)}

Note: The data includes both:
- "client_quirks": Semantic differences (potential bugs)
- "format_quirks": Formatting differences that don't affect correctness (e.g., returnValue encoding, memory format)

Provide:
1. A summary of the key behavioral differences between clients
2. Highlight which items in "client_quirks" are actual semantic differences vs formatting noise
3. Any patterns that suggest potential bugs or spec violations
4. For differences that matter, note what each client does without assuming any client is the "reference" or "correct" implementation
5. Notable quirks for each client

Be concise but thorough. Focus on actionable insights. Do not assume any client (including Geth) is the reference implementation - treat all clients equally."""

    options = ClaudeAgentOptions(
        max_turns=1,
    )

    try:
        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        result_text += block.text
        return result_text, None
    except CLINotFoundError:
        return "", "Claude Code CLI not found. Install with: curl -fsSL https://claude.ai/install.sh | bash"
    except ClaudeSDKError as e:
        return "", f"Claude SDK error: {e}"


def check_claude_sdk_available() -> bool:
    """Check if claude-agent-sdk is available."""
    try:
        import claude_agent_sdk  # noqa: F401
        import anyio  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(description="Analyze trace comparison data and generate client quirk reports")
    parser.add_argument("--input-dir", type=str, default="./traces", help="Input directory containing traces (default: ./traces)")
    parser.add_argument("--output-dir", type=str, default="./reports", help="Output directory for reports (default: ./reports)")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI analysis even if claude-agent-sdk is available")
    args = parser.parse_args()

    traces_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading comparison files...")
    comparisons = load_comparisons(traces_dir)
    print(f"Loaded {len(comparisons)} comparison files")

    print("Analyzing differences...")
    analysis = analyze_differences(comparisons, traces_dir)

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Total comparisons: {analysis['summary']['total_comparisons']}")
    if analysis["summary"]["skipped_due_to_errors"] > 0:
        print(f"Skipped (client errors): {analysis['summary']['skipped_due_to_errors']}")
    print(f"Analyzed: {analysis['summary']['analyzed_comparisons']}")
    print("\nDifference types:")
    for dtype, count in sorted(analysis["summary"]["diff_types"].items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count:,}")

    if analysis["errors_by_client"]:
        print("\nErrors by client (these transactions excluded from analysis):")
        for client, count in sorted(analysis["errors_by_client"].items()):
            print(f"  {client}: {count}")

    # Get all clients
    all_clients = list(analysis["client_quirks"].keys())

    # Generate reports for each client
    print("\nGenerating client reports...")
    analyzed_txs = analysis["summary"]["analyzed_comparisons"]
    for client, quirks in analysis["client_quirks"].items():
        error_count = analysis["errors_by_client"].get(client, 0)

        report = generate_client_report(
            client, quirks, all_clients,
            analyzed_txs, analyzed_txs, error_count,  # All analyzed txs have all clients
            format_quirks=analysis.get("format_quirks")
        )
        report_path = output_dir / f"{client}.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Written: {report_path}")

    # Generate overall summary report
    summary_lines = ["# Ethereum Client debug_traceTransaction Comparison Summary\n"]
    summary_lines.append("\nThis report compares `debug_traceTransaction` outputs across Ethereum execution clients to identify behavioral differences and formatting quirks.\n")
    summary_lines.append("\n## Overview\n")
    summary_lines.append(f"- **Total transactions**: {analysis['summary']['total_comparisons']}\n")
    if analysis["summary"]["skipped_due_to_errors"] > 0:
        summary_lines.append(f"- **Skipped (client errors)**: {analysis['summary']['skipped_due_to_errors']}\n")
    summary_lines.append(f"- **Analyzed**: {analysis['summary']['analyzed_comparisons']}\n")
    summary_lines.append(f"- **Clients**: {', '.join(sorted(all_clients))}\n")
    summary_lines.append("\n---\n")
    summary_lines.append("\n## Difference Types Found\n")
    summary_lines.append("\nThese are semantic differences detected in trace outputs. High counts indicate systematic differences in how clients report EVM execution.\n\n")
    diff_type_descriptions = {
        "missing_key": "Field present in some clients but not others",
        "value_mismatch": "Same field has different values",
        "type_mismatch": "Same field has different types",
        "structlog_count_mismatch": "Different number of structLog entries",
        "top_level_mismatch": "Top-level trace fields differ",
    }
    summary_lines.append("| Type | Count | Description |\n|------|-------|-------------|\n")
    for dtype, count in sorted(analysis["summary"]["diff_types"].items(), key=lambda x: -x[1]):
        desc = diff_type_descriptions.get(dtype, "—")
        summary_lines.append(f"| {dtype} | {count:,} | {desc} |\n")

    summary_lines.append("\n---\n")
    summary_lines.append("\n## Client Quirk Summary\n")
    summary_lines.append("\nThis table shows where each client differs from the majority behavior. Lower numbers indicate better conformance.\n\n")
    summary_lines.append("| Client | Missing Keys | Extra Keys | Type Mismatches | Value Mismatches | Gas Diffs | ReturnValue Diffs |\n")
    summary_lines.append("|--------|--------------|------------|-----------------|------------------|-----------|-------------------|\n")

    for client in sorted(all_clients):
        quirks = analysis["client_quirks"][client]
        # Sum nested dicts: key -> op -> count
        missing = sum(sum(op_counts.values()) for op_counts in quirks["missing_keys"].values())
        extra = sum(sum(op_counts.values()) for op_counts in quirks["has_extra_keys"].values())
        type_mm = sum(len(v) for v in quirks["type_mismatches"].values())
        value_mm = sum(len(v) for v in quirks["value_mismatches"].values())
        gas = len(quirks["gas_differences"])
        ret = len(quirks["return_value_differences"])
        summary_lines.append(f"| {client} | {missing:,} | {extra:,} | {type_mm:,} | {value_mm:,} | {gas} | {ret} |\n")

    summary_lines.append("\n---\n")
    summary_lines.append("\n## Per-Client Reports\n")
    summary_lines.append("\nDetailed reports for each client:\n\n")
    for client in sorted(all_clients):
        summary_lines.append(f"- [{client.upper()}](./{client}.md)\n")

    # Add Format Quirks summary section
    format_quirks = analysis.get("format_quirks", {})
    rv_by_client = format_quirks.get("returnValue_encoding", {}).get("by_client", {})
    rv_examples = format_quirks.get("returnValue_encoding", {}).get("examples", [])
    mem_by_client = format_quirks.get("memory_format", {}).get("by_client", {})
    mem_examples = format_quirks.get("memory_format", {}).get("examples", [])
    stor_by_client = format_quirks.get("storage_format", {}).get("by_client", {})
    stor_examples = format_quirks.get("storage_format", {}).get("examples", [])
    has_format_quirks = (
        rv_by_client
        or mem_by_client
        or stor_by_client
        or format_quirks.get("optional_fields")
    )

    if has_format_quirks:
        summary_lines.append("\n---\n")
        summary_lines.append("\n## Format Quirks (Not Bugs)\n")
        summary_lines.append("\nThese are formatting differences that don't affect trace correctness. ")
        summary_lines.append("The comparison tool normalizes these before semantic comparison.\n\n")

        # returnValue encoding
        if rv_by_client:
            summary_lines.append("### Return Value Encoding\n")
            summary_lines.append("How empty return values are represented. These are semantically equivalent.\n\n")
            summary_lines.append("| Client | Format | Count |\n|--------|--------|-------|\n")
            for client, encodings in sorted(rv_by_client.items()):
                for encoding, count in sorted(encodings.items(), key=lambda x: -x[1]):
                    summary_lines.append(f"| {client} | `{encoding}` | {count:,} |\n")

            # Show examples
            if rv_examples:
                for i, ex in enumerate(rv_examples[:1], 1):
                    tx_hash = ex['tx_hash']
                    summary_lines.append(f"\n**Example** (block {ex['block']}, tx `{tx_hash}`):\n")
                    summary_lines.append("| Client | Value |\n|--------|-------|\n")
                    for c, val in sorted(ex.get("values", {}).items()):
                        val_str = str(val) if val is not None else "null"
                        val_display = f"`{val_str}`"
                        summary_lines.append(f"| {c} | {val_display} |\n")
            summary_lines.append("\n")

        # memory format
        if mem_by_client:
            summary_lines.append("### Memory Format\n")
            summary_lines.append("How memory words are formatted in structLogs.\n\n")
            summary_lines.append("| Client | Format | Count |\n|--------|--------|-------|\n")
            for client, formats in sorted(mem_by_client.items()):
                for fmt, count in sorted(formats.items(), key=lambda x: -x[1]):
                    summary_lines.append(f"| {client} | `{fmt}` | {count:,} |\n")

            # Show examples
            if mem_examples:
                for i, ex in enumerate(mem_examples[:1], 1):
                    tx_hash = ex['tx_hash']
                    summary_lines.append(f"\n**Example** (block {ex['block']}, tx `{tx_hash}`):\n")
                    summary_lines.append("| Client | Value |\n|--------|-------|\n")
                    for c, val in sorted(ex.get("values", {}).items()):
                        val_str = str(val) if val is not None else "null"
                        val_display = f"`{val_str}`"
                        summary_lines.append(f"| {c} | {val_display} |\n")
            summary_lines.append("\n")

        # storage format - summarize by client
        if stor_by_client:
            summary_lines.append("### Storage Format\n")
            summary_lines.append("How storage keys and values are formatted. Clients differ in zero-padding.\n\n")
            summary_lines.append("| Client | Keys | Values | Observations |\n|--------|------|--------|-------------|\n")

            for client, formats in sorted(stor_by_client.items()):
                uses_short_keys = 0
                uses_padded_keys = 0
                uses_short_vals = 0
                uses_padded_vals = 0

                for fmt, cnt in formats.items():
                    if "key:short_" in fmt or "key:with_0x" in fmt:
                        uses_short_keys += cnt
                    elif "key:64char" in fmt:
                        uses_padded_keys += cnt
                    if "val:short_" in fmt or "val:with_0x" in fmt:
                        uses_short_vals += cnt
                    elif "val:64char" in fmt:
                        uses_padded_vals += cnt

                key_style = []
                if uses_short_keys > 0:
                    key_style.append(f"compact ({uses_short_keys})")
                if uses_padded_keys > 0:
                    key_style.append(f"64-char ({uses_padded_keys})")

                val_style = []
                if uses_short_vals > 0:
                    val_style.append(f"compact ({uses_short_vals})")
                if uses_padded_vals > 0:
                    val_style.append(f"64-char ({uses_padded_vals})")

                total = sum(formats.values())
                summary_lines.append(f"| {client} | {', '.join(key_style) or '—'} | {', '.join(val_style) or '—'} | {total:,} txs |\n")

            # Show examples
            if stor_examples:
                for i, ex in enumerate(stor_examples[:1], 1):
                    tx_hash = ex['tx_hash']
                    summary_lines.append(f"\n**Example** (block {ex['block']}, tx `{tx_hash}`):\n")
                    summary_lines.append("| Client | Key | Value |\n|--------|-----|-------|\n")
                    for c, data in sorted(ex.get("values", {}).items()):
                        if data and isinstance(data, dict):
                            key_str = str(data.get("key", ""))
                            val_str = str(data.get("value", ""))
                            key_display = f"`{key_str}`"
                            val_display = f"`{val_str}`"
                            summary_lines.append(f"| {c} | {key_display} | {val_display} |\n")
            summary_lines.append("\n")

        # optional fields
        opt_by_client = format_quirks.get("optional_fields", {}).get("by_client", {})
        opt_examples = format_quirks.get("optional_fields", {}).get("examples", {})

        if opt_by_client:
            summary_lines.append("### Optional Fields\n")
            summary_lines.append("Fields some clients include that others may omit.\n\n")

            field_descriptions = {
                "error": "Includes `error: null` on every step",
                "storage": "Includes `storage: {}` on every step",
                "returnData": "Includes `returnData` field on every step",
                "structLog_for_simple_transfer": "Emits structLog for simple ETH transfers (others emit 0 entries)",
            }

            summary_lines.append("| Client | Field | Count | Description |\n|--------|-------|-------|-------------|\n")
            for client, fields in sorted(opt_by_client.items()):
                for field, count in sorted(fields.items(), key=lambda x: -x[1]):
                    desc = field_descriptions.get(field, "—")
                    summary_lines.append(f"| {client} | `{field}` | {count:,} | {desc} |\n")

            # Show examples for each field
            if opt_examples:
                for field, examples in sorted(opt_examples.items()):
                    if examples:
                        summary_lines.append(f"\n**`{field}` example:**\n")
                        for i, ex in enumerate(examples[:1], 1):
                            summary_lines.append(f"\n*Example* (block {ex['block']}, tx `{ex['tx_hash']}`):\n")
                            summary_lines.append("| Client | Value |\n|--------|-------|\n")
                            for c, val in sorted(ex.get("client_values", {}).items()):
                                summary_lines.append(f"| {c} | `{val}` |\n")
            summary_lines.append("\n")

    # Add AI analysis if available and not disabled
    if args.no_ai:
        print("\nAI analysis disabled via --no-ai flag")
    elif not check_claude_sdk_available():
        print("\nAI analysis skipped (claude-agent-sdk not installed)")
        print("  Install with: pip install claude-agent-sdk")
    else:
        import anyio
        print("\nGenerating AI analysis...")
        ai_analysis, ai_error = anyio.run(ai_analyze_differences, analysis)

        if ai_error:
            print(f"  AI analysis failed: {ai_error}")
        else:
            summary_lines.append("\n## AI Analysis\n")
            summary_lines.append(ai_analysis)
            summary_lines.append("\n")

            # Write standalone AI analysis file
            ai_report_path = output_dir / "ai_analysis.md"
            with open(ai_report_path, "w") as f:
                f.write("# AI Analysis of Client Differences\n\n")
                f.write(ai_analysis)
            print(f"  Written: {ai_report_path}")

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("".join(summary_lines))
    print(f"  Written: {summary_path}")


if __name__ == "__main__":
    main()
