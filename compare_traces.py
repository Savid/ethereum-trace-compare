#!/usr/bin/env python3
"""
Compare debug_traceTransaction payloads across multiple Ethereum execution clients.

This script:
1. Queries each node for head block number
2. Finds min(block_number) if max-min < 16
3. Traces all transactions in that block on each node
4. Stores traces in ./traces/$block_number/$tx_hash/$node.json
5. Compares payloads to find differences
"""

import argparse
import json
import signal
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

SUPPORTED_CLIENTS = ["geth", "besu", "reth", "nethermind", "erigon", "ethrex"]

# Known optional fields that some clients include but others don't
# These are tracked as format quirks, not semantic differences
KNOWN_OPTIONAL_FIELDS = {
    "error": {"nethermind"},      # Nethermind always includes error: null
    "storage": {"nethermind"},    # Nethermind always includes storage: {}
    "returnData": {"reth"},       # Reth includes returnData at every step
}

# Global state
_shutdown_requested = False
_request_headers: dict[str, str] = {}


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    if _shutdown_requested:
        print(f"\n\nForced exit (received {sig_name} again)")
        sys.exit(1)
    print(f"\n\nShutdown requested ({sig_name}), finishing current transaction...")
    _shutdown_requested = True


def rpc_call(url: str, method: str, params: list | None = None, timeout: int = 120) -> dict:
    """Make a JSON-RPC call to an Ethereum node."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params or [],
        "id": 1,
    }
    resp = requests.post(url, json=payload, headers=_request_headers or None, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        raise Exception(f"RPC error: {result['error']}")
    return result.get("result")


def get_block_number(url: str) -> int:
    """Get the current block number from a node."""
    result = rpc_call(url, "eth_blockNumber", timeout=10)
    return int(result, 16)


def get_block_with_transactions(url: str, block_number: int) -> dict:
    """Get a block with full transaction objects."""
    block_hex = hex(block_number)
    return rpc_call(url, "eth_getBlockByNumber", [block_hex, True])


def get_transaction_receipt(url: str, tx_hash: str) -> dict:
    """Get transaction receipt to check gas usage."""
    return rpc_call(url, "eth_getTransactionReceipt", [tx_hash], timeout=10)


def trace_transaction(url: str, tx_hash: str) -> dict:
    """Call debug_traceTransaction on a transaction."""
    tracer_config = {
        "enableMemory": True,
        "disableStack": False,
        "disableStorage": False,
        "enableReturnData": True,
    }
    return rpc_call(url, "debug_traceTransaction", [tx_hash, tracer_config])


def save_trace(trace_dir: Path, node_name: str, trace_data: dict | None, error: str | None = None) -> None:
    """Save a trace result to a JSON file."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    filepath = trace_dir / f"{node_name}.json"

    if error:
        data = {"error": error}
    else:
        data = trace_data

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def normalize_memory_word(value: str) -> int | str:
    """
    Normalize memory word to integer for semantic comparison.

    Handles different formats:
    - "0x60" (besu short hex with prefix)
    - "0000...0060" (64-char padded hex without prefix)
    """
    if isinstance(value, str):
        clean = value[2:] if value.startswith("0x") else value
        try:
            return int(clean, 16) if clean else 0
        except ValueError:
            return value
    return value


def normalize_return_value(val: str | None) -> str:
    """
    Normalize return value for semantic comparison.

    Treats "", "0x", and None as equivalent empty values.
    For non-empty values, strips 0x prefix and normalizes to lowercase.
    """
    if val in ("", "0x", None):
        return ""  # Treat all empty as equivalent
    if isinstance(val, str):
        # Strip 0x prefix and normalize to lowercase
        clean = val[2:] if val.startswith("0x") else val
        return clean.lower()
    return val


def detect_memory_format(value: str) -> str:
    """Detect the memory format style used by a client."""
    if isinstance(value, str):
        if value.startswith("0x"):
            return "short_hex_with_0x"  # e.g., "0x60"
        elif len(value) == 64:
            return "64char_no_prefix"  # e.g., "0000...0060"
        else:
            return f"other_len_{len(value)}"
    return "non_string"


def normalize_value(value: Any) -> tuple[str, Any]:
    """
    Normalize a value and return its type category and normalized form.

    Returns (type_category, normalized_value) where type_category is one of:
    - "number": integer values
    - "hex_string": hex strings (with or without 0x prefix)
    - "bytes_array": list of integers (byte arrays)
    - "string": other strings
    - "other": other types
    """
    if isinstance(value, int):
        return ("number", value)

    if isinstance(value, str):
        # Handle 0x-prefixed hex strings
        if value.startswith("0x"):
            try:
                int_val = int(value, 16)
                return ("hex_string", int_val)
            except ValueError:
                return ("hex_string", value)

        # Handle hex strings without 0x prefix (common in memory words)
        clean = value
        if clean and all(c in "0123456789abcdefABCDEF" for c in clean):
            try:
                int_val = int(clean, 16)
                return ("hex_string", int_val)
            except ValueError:
                pass

        return ("string", value)

    if isinstance(value, list):
        if all(isinstance(x, int) and 0 <= x <= 255 for x in value):
            return ("bytes_array", bytes(value).hex())
        return ("list", value)

    if isinstance(value, bool):
        return ("bool", value)

    if isinstance(value, dict):
        return ("dict", value)

    return ("other", value)


def normalize_memory_array(memory: list) -> list[int]:
    """Normalize a memory array to list of integers for semantic comparison."""
    return [normalize_memory_word(word) for word in memory]


def normalize_storage_dict(storage: dict) -> dict:
    """
    Normalize storage dict keys and values for semantic comparison.

    Handles different formats:
    - Keys: "0" vs "0000000000000000000000000000000000000000000000000000000000000000"
    - Values: "44000100..." vs "000044000100..." (with or without 0x prefix)

    Normalizes both keys and values to 64-char lowercase hex strings.
    """
    if not isinstance(storage, dict):
        return storage
    normalized = {}
    for key, value in storage.items():
        # Normalize key: strip 0x, convert to int, then back to 64-char hex
        if isinstance(key, str):
            clean_key = key[2:] if key.startswith("0x") else key
            try:
                key_int = int(clean_key, 16) if clean_key else 0
                norm_key = f"{key_int:064x}"
            except ValueError:
                norm_key = key
        else:
            norm_key = str(key)

        # Normalize value similarly
        if isinstance(value, str):
            clean_val = value[2:] if value.startswith("0x") else value
            try:
                val_int = int(clean_val, 16) if clean_val else 0
                norm_val = f"{val_int:064x}"
            except ValueError:
                norm_val = value
        else:
            norm_val = str(value)

        normalized[norm_key] = norm_val
    return normalized


def detect_storage_format(storage: dict) -> str:
    """Detect the storage format style used by a client."""
    if not isinstance(storage, dict) or not storage:
        return "empty"

    # Check first key's format
    first_key = next(iter(storage.keys()))
    first_val = storage[first_key]

    key_format = "unknown"
    if isinstance(first_key, str):
        if first_key.startswith("0x"):
            key_format = "with_0x_prefix"
        elif len(first_key) == 64:
            key_format = "64char_no_prefix"
        else:
            key_format = f"short_{len(first_key)}char"

    val_format = "unknown"
    if isinstance(first_val, str):
        if first_val.startswith("0x"):
            val_format = "with_0x_prefix"
        elif len(first_val) == 64:
            val_format = "64char_no_prefix"
        else:
            val_format = f"short_{len(first_val)}char"

    return f"key:{key_format},val:{val_format}"


def is_known_optional_field(key: str, present_nodes: list[str], missing_nodes: list[str]) -> bool:
    """
    Check if a key difference is a known optional field quirk.

    Returns True if all present_nodes are known to include this optional field
    AND all missing_nodes are known to NOT include it.
    """
    if key not in KNOWN_OPTIONAL_FIELDS:
        return False

    known_includers = KNOWN_OPTIONAL_FIELDS[key]

    # All present nodes should be known includers
    # AND all missing nodes should NOT be known includers
    for node in present_nodes:
        if node not in known_includers:
            return False  # Non-known client has it - this is a real difference

    return True


def compare_struct_logs(logs_by_node: dict[str, list]) -> tuple[list[dict], dict]:
    """
    Compare structLogs across nodes.

    Returns:
        tuple: (differences, format_quirks)
            - differences: list of semantic differences (actual bugs)
            - format_quirks: dict tracking formatting differences (not bugs)
    """
    differences = []
    format_quirks = {
        "memory_format": defaultdict(int),  # {client: {format: count}}
        "storage_format": defaultdict(int),  # {client: {format: count}}
        "optional_fields": {
            "by_client": defaultdict(lambda: defaultdict(int)),  # {client: {field: count}}
            "examples": defaultdict(list),  # {field: [{client_values: {client: value}}]}
        },
    }

    # Check for length differences
    lengths = {node: len(logs) for node, logs in logs_by_node.items()}
    unique_lengths = set(lengths.values())

    if len(unique_lengths) > 1:
        # Check if this is just a "minimal vs empty" quirk (e.g., 0 vs 1 for simple transfers)
        counts = list(lengths.values())
        if all(c <= 1 for c in counts):
            # Track as format quirk, not semantic difference
            for node, count in lengths.items():
                if count == 1:
                    format_quirks["optional_fields"]["by_client"][node]["structLog_for_simple_transfer"] += 1
            # Add example showing all client values
            format_quirks["optional_fields"]["examples"]["structLog_for_simple_transfer"].append({
                "client_values": {node: f"{cnt} structLog entries" for node, cnt in lengths.items()}
            })
        else:
            differences.append({
                "type": "structlog_count_mismatch",
                "details": lengths,
            })

    # Find minimum length for comparison
    min_length = min(lengths.values()) if lengths else 0

    # Track memory and storage formats seen (only need to detect once per client)
    memory_formats_detected = set()
    storage_formats_detected = set()

    # Compare each struct log entry
    for i in range(min_length):
        entries = {node: logs[i] for node, logs in logs_by_node.items()}

        # Get all keys across all nodes
        all_keys = set()
        for entry in entries.values():
            if isinstance(entry, dict):
                all_keys.update(entry.keys())

        # Compare each key
        for key in all_keys:
            values_by_node = {}
            types_by_node = {}
            normalized_by_node = {}

            for node, entry in entries.items():
                if isinstance(entry, dict) and key in entry:
                    type_cat, norm_val = normalize_value(entry[key])
                    values_by_node[node] = entry[key]
                    types_by_node[node] = type_cat
                    normalized_by_node[node] = norm_val
                else:
                    values_by_node[node] = None
                    types_by_node[node] = "missing"
                    normalized_by_node[node] = None

            # Check for missing keys
            missing_nodes = [n for n, t in types_by_node.items() if t == "missing"]
            present_nodes = [n for n, t in types_by_node.items() if t != "missing"]

            if missing_nodes and present_nodes:
                # Check if this is a known optional field
                if is_known_optional_field(key, present_nodes, missing_nodes):
                    # Track as format quirk, not difference
                    for node in present_nodes:
                        format_quirks["optional_fields"]["by_client"][node][key] += 1
                    # Add example with client values (only first occurrence per index)
                    def format_val(val):
                        """Format value for display, using JSON-style null."""
                        if val is None:
                            return "null"
                        if isinstance(val, dict):
                            return json.dumps(val)[:50]
                        if isinstance(val, str) and len(val) > 50:
                            return val[:50] + "..."
                        return repr(val)

                    format_quirks["optional_fields"]["examples"][key].append({
                        "index": i,
                        "client_values": {
                            node: f"present: {format_val(values_by_node[node])}" if node in present_nodes else "missing"
                            for node in values_by_node.keys()
                        }
                    })
                else:
                    differences.append({
                        "type": "missing_key",
                        "index": i,
                        "key": key,
                        "missing_in": missing_nodes,
                        "present_in": present_nodes,
                    })
                continue

            # Special handling for memory arrays
            if key == "memory":
                # Track memory format differences as quirks
                for node, val in values_by_node.items():
                    if isinstance(val, list) and val and node not in memory_formats_detected:
                        fmt = detect_memory_format(val[0])
                        format_quirks["memory_format"][node] = {
                            "format": fmt,
                            "example": val[0] if val else None,
                        }
                        memory_formats_detected.add(node)

                # Normalize memory arrays for semantic comparison
                normalized_memory = {}
                for node, val in values_by_node.items():
                    if isinstance(val, list):
                        normalized_memory[node] = normalize_memory_array(val)
                    else:
                        normalized_memory[node] = val

                # Compare normalized memory
                unique_memory = set()
                for v in normalized_memory.values():
                    if v is not None:
                        if isinstance(v, list):
                            unique_memory.add(tuple(v))
                        else:
                            unique_memory.add(v)

                if len(unique_memory) > 1:
                    differences.append({
                        "type": "value_mismatch",
                        "index": i,
                        "key": key,
                        "values": values_by_node,
                    })
                continue

            # Special handling for storage dicts
            if key == "storage":
                # Track storage format differences as quirks
                # First, find a common key across all nodes for examples
                common_key = None
                if all(isinstance(values_by_node.get(n), dict) and values_by_node.get(n) for n in values_by_node):
                    # Normalize keys to find common ones
                    key_sets = []
                    for node, val in values_by_node.items():
                        if isinstance(val, dict):
                            normalized_keys = set()
                            for k in val.keys():
                                clean_k = k[2:] if isinstance(k, str) and k.startswith("0x") else k
                                try:
                                    norm_k = int(clean_k, 16) if clean_k else 0
                                    normalized_keys.add(norm_k)
                                except ValueError:
                                    pass
                            key_sets.append(normalized_keys)
                    if key_sets:
                        common_keys = key_sets[0].intersection(*key_sets[1:]) if len(key_sets) > 1 else key_sets[0]
                        if common_keys:
                            common_key = next(iter(common_keys))  # Pick first common key

                for node, val in values_by_node.items():
                    if isinstance(val, dict) and val and node not in storage_formats_detected:
                        fmt = detect_storage_format(val)
                        # Find the raw key/value for this node that matches common_key
                        example_key = None
                        example_value = None
                        if common_key is not None:
                            for k, v in val.items():
                                clean_k = k[2:] if isinstance(k, str) and k.startswith("0x") else k
                                try:
                                    if int(clean_k, 16) == common_key:
                                        example_key = k
                                        example_value = v
                                        break
                                except ValueError:
                                    pass
                        # Fallback to first key if no common key found
                        if example_key is None:
                            example_key = next(iter(val.keys())) if val else None
                            example_value = next(iter(val.values())) if val else None
                        format_quirks["storage_format"][node] = {
                            "format": fmt,
                            "example_key": example_key,
                            "example_value": example_value,
                        }
                        storage_formats_detected.add(node)

                # Normalize storage dicts for semantic comparison
                normalized_storage = {}
                for node, val in values_by_node.items():
                    if isinstance(val, dict):
                        normalized_storage[node] = normalize_storage_dict(val)
                    else:
                        normalized_storage[node] = val

                # Compare normalized storage
                unique_storage = set()
                for v in normalized_storage.values():
                    if v is not None:
                        if isinstance(v, dict):
                            unique_storage.add(json.dumps(v, sort_keys=True))
                        else:
                            unique_storage.add(v)

                if len(unique_storage) > 1:
                    differences.append({
                        "type": "value_mismatch",
                        "index": i,
                        "key": key,
                        "values": values_by_node,
                    })
                continue

            # Check for type differences
            present_types = {n: t for n, t in types_by_node.items() if t != "missing"}
            unique_types = set(present_types.values())

            if len(unique_types) > 1:
                differences.append({
                    "type": "type_mismatch",
                    "index": i,
                    "key": key,
                    "types": present_types,
                    "values": values_by_node,
                })
                continue

            # Check for value differences (after normalization)
            unique_normalized = set()
            for v in normalized_by_node.values():
                if v is not None:
                    if isinstance(v, (list, dict)):
                        unique_normalized.add(json.dumps(v, sort_keys=True))
                    else:
                        unique_normalized.add(v)

            if len(unique_normalized) > 1:
                differences.append({
                    "type": "value_mismatch",
                    "index": i,
                    "key": key,
                    "values": values_by_node,
                })

    # Convert nested defaultdicts to regular dicts for JSON serialization
    format_quirks["memory_format"] = dict(format_quirks["memory_format"])
    format_quirks["storage_format"] = dict(format_quirks["storage_format"])
    format_quirks["optional_fields"] = {
        "by_client": {k: dict(v) for k, v in format_quirks["optional_fields"]["by_client"].items()},
        "examples": dict(format_quirks["optional_fields"]["examples"]),
    }

    return differences, format_quirks


def compare_traces(traces_by_node: dict[str, dict]) -> dict:
    """
    Compare trace results across all nodes.

    Returns a summary of differences found, plus format quirks.
    """
    comparison = {
        "nodes_compared": list(traces_by_node.keys()),
        "errors": {},
        "differences": [],  # Semantic differences (actual bugs)
        "format_quirks": {   # Formatting differences (not bugs)
            "returnValue_encoding": {},     # e.g., {"besu": '""', "others": '"0x"'}
            "memory_format": {},            # e.g., {"besu": "short_hex_with_0x", "others": "64char_no_prefix"}
            "storage_format": {},           # e.g., {"besu": "key:short_1char,val:short_16char", "others": "key:64char_no_prefix,val:64char_no_prefix"}
            "optional_fields": {},          # e.g., {"nethermind": {"error": 1, "storage": 1}}
        },
    }

    # Separate successful traces from errors
    successful = {}
    for node, trace in traces_by_node.items():
        if trace is None or (isinstance(trace, dict) and "error" in trace):
            comparison["errors"][node] = trace.get("error") if trace else "No trace data"
        else:
            successful[node] = trace

    if len(successful) < 2:
        comparison["differences"].append({
            "type": "insufficient_successful_traces",
            "successful_count": len(successful),
        })
        return comparison

    # Compare top-level keys
    all_keys = set()
    for trace in successful.values():
        if isinstance(trace, dict):
            all_keys.update(trace.keys())

    for key in all_keys:
        if key == "structLogs":
            continue  # Handle separately

        values = {}
        for node, trace in successful.items():
            if isinstance(trace, dict):
                values[node] = trace.get(key)

        # Special handling for returnValue - use semantic comparison
        if key == "returnValue":
            normalized_values = {node: normalize_return_value(val) for node, val in values.items()}
            unique_normalized = set(normalized_values.values())

            if len(unique_normalized) > 1:
                # Actual semantic difference
                comparison["differences"].append({
                    "type": "top_level_mismatch",
                    "key": key,
                    "values": values,
                })
            else:
                # Check for encoding differences (format quirk, not semantic)
                unique_raw = set()
                for v in values.values():
                    unique_raw.add(repr(v))

                if len(unique_raw) > 1:
                    # Track encoding differences as format quirk
                    # Store format STYLE (not actual value) + example
                    for node, val in values.items():
                        normalized = normalized_values[node]
                        if normalized == "":
                            if val == "0x":
                                style = "empty_as_0x"
                            elif val == "":
                                style = "empty_as_empty"
                            elif val is None:
                                style = "empty_as_null"
                            else:
                                style = f"empty_as_{repr(val)}"
                        else:
                            style = "with_0x" if isinstance(val, str) and val.startswith("0x") else "no_0x"
                        comparison["format_quirks"]["returnValue_encoding"][node] = {
                            "format": style,
                            "example": val[:50] + "..." if isinstance(val, str) and len(val) > 50 else val,
                        }
            continue

        unique_values = set()
        for v in values.values():
            if isinstance(v, (list, dict)):
                unique_values.add(json.dumps(v, sort_keys=True))
            else:
                unique_values.add(v)

        if len(unique_values) > 1:
            comparison["differences"].append({
                "type": "top_level_mismatch",
                "key": key,
                "values": values,
            })

    # Compare structLogs
    struct_logs = {}
    for node, trace in successful.items():
        if isinstance(trace, dict) and "structLogs" in trace:
            struct_logs[node] = trace["structLogs"]

    if struct_logs:
        log_diffs, struct_log_quirks = compare_struct_logs(struct_logs)
        comparison["differences"].extend(log_diffs)

        # Merge struct log quirks into comparison format_quirks
        comparison["format_quirks"]["memory_format"].update(struct_log_quirks.get("memory_format", {}))
        comparison["format_quirks"]["storage_format"].update(struct_log_quirks.get("storage_format", {}))
        # Merge optional_fields with new structure
        opt_quirks = struct_log_quirks.get("optional_fields", {})
        for client, fields in opt_quirks.get("by_client", {}).items():
            if client not in comparison["format_quirks"]["optional_fields"]:
                comparison["format_quirks"]["optional_fields"][client] = {}
            comparison["format_quirks"]["optional_fields"][client].update(fields)
        # Store examples in format_quirks
        comparison["format_quirks"]["optional_fields_examples"] = opt_quirks.get("examples", {})

    return comparison


def print_comparison_summary(tx_hash: str, comparison: dict) -> None:
    """Print a human-readable summary of comparison results."""
    print(f"\n{'='*80}")
    print(f"Transaction: {tx_hash}")
    print(f"Nodes compared: {', '.join(comparison['nodes_compared'])}")

    if comparison["errors"]:
        print(f"\nErrors:")
        for node, error in comparison["errors"].items():
            print(f"  {node}: {error}")

    diffs = comparison["differences"]
    quirks = comparison.get("format_quirks", {})

    # Count non-empty quirks
    quirk_count = 0
    if quirks.get("returnValue_encoding"):
        quirk_count += 1
    if quirks.get("memory_format"):
        quirk_count += 1
    if quirks.get("storage_format"):
        quirk_count += 1
    if quirks.get("optional_fields"):
        quirk_count += sum(len(fields) for fields in quirks["optional_fields"].values())

    if not diffs and not quirk_count:
        print("\nNo differences found!")
        return

    if diffs:
        print(f"\nFound {len(diffs)} semantic difference(s):")

        # Group differences by type
        by_type = defaultdict(list)
        for diff in diffs:
            by_type[diff["type"]].append(diff)

        for diff_type, type_diffs in by_type.items():
            print(f"\n  [{diff_type}] ({len(type_diffs)} occurrences)")

            if diff_type == "structlog_count_mismatch":
                for d in type_diffs:
                    print(f"    Counts: {d['details']}")

            elif diff_type == "type_mismatch":
                # Show first few examples
                for d in type_diffs[:5]:
                    print(f"    Index {d['index']}, key '{d['key']}': {d['types']}")
                if len(type_diffs) > 5:
                    print(f"    ... and {len(type_diffs) - 5} more")

            elif diff_type == "value_mismatch":
                for d in type_diffs[:5]:
                    print(f"    Index {d['index']}, key '{d['key']}':")
                    for node, val in d["values"].items():
                        val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
                        print(f"      {node}: {val_str}")
                if len(type_diffs) > 5:
                    print(f"    ... and {len(type_diffs) - 5} more")

            elif diff_type == "missing_key":
                for d in type_diffs[:5]:
                    print(f"    Index {d['index']}, key '{d['key']}' missing in: {d['missing_in']}")
                if len(type_diffs) > 5:
                    print(f"    ... and {len(type_diffs) - 5} more")

            else:
                for d in type_diffs[:3]:
                    print(f"    {d}")
    else:
        print("\nNo semantic differences found!")

    # Show format quirks summary
    if quirk_count:
        print(f"\nFormat quirks (not bugs): {quirk_count}")
        if quirks.get("returnValue_encoding"):
            encodings = quirks["returnValue_encoding"]
            print(f"  returnValue encoding: {encodings}")
        if quirks.get("memory_format"):
            formats = quirks["memory_format"]
            print(f"  memory format: {formats}")
        if quirks.get("storage_format"):
            formats = quirks["storage_format"]
            print(f"  storage format: {formats}")
        if quirks.get("optional_fields"):
            for client, fields in quirks["optional_fields"].items():
                print(f"  {client} optional fields: {list(fields.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Compare debug_traceTransaction across Ethereum clients")
    parser.add_argument("--block", type=int, help="Specific block number to trace (default: auto-detect)")
    parser.add_argument("--tx", type=str, help="Specific transaction hash to trace (traces only this tx)")
    parser.add_argument("--output-dir", type=str, default="./traces", help="Output directory for traces")
    parser.add_argument("--max-txs", type=int, default=None, help="Maximum transactions to trace per block")
    parser.add_argument("--max-gas", type=int, default=None,
                        help="Skip transactions using more than this gas (e.g., 10000000)")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    parser.add_argument("--header", "-H", action="append", metavar="NAME:VALUE",
                        help="HTTP header to include in requests (can be specified multiple times)")

    # Client endpoint arguments
    parser.add_argument("--geth", type=str, help="Geth RPC endpoint URL")
    parser.add_argument("--besu", type=str, help="Besu RPC endpoint URL")
    parser.add_argument("--reth", type=str, help="Reth RPC endpoint URL")
    parser.add_argument("--nethermind", type=str, help="Nethermind RPC endpoint URL")
    parser.add_argument("--erigon", type=str, help="Erigon RPC endpoint URL")
    parser.add_argument("--ethrex", type=str, help="Ethrex RPC endpoint URL")

    args = parser.parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Parse headers
    global _request_headers
    if args.header:
        for header in args.header:
            if ":" not in header:
                parser.error(f"Invalid header format '{header}', expected 'Name: Value'")
            name, value = header.split(":", 1)
            _request_headers[name.strip()] = value.strip()

    # Build nodes dict from provided arguments
    nodes = {}
    for client in SUPPORTED_CLIENTS:
        url = getattr(args, client)
        if url:
            nodes[client] = url

    if len(nodes) < 2:
        parser.error("At least 2 client endpoints required for comparison")

    output_dir = Path(args.output_dir)

    # Step 1: Get block numbers from all nodes (in parallel)
    print("Fetching head block numbers from all nodes...")
    block_numbers = {}
    block_errors = {}

    def fetch_block_number(client_url):
        client, url = client_url
        try:
            block_num = get_block_number(url)
            return client, block_num, None
        except Exception as e:
            return client, None, str(e)

    with ThreadPoolExecutor(max_workers=len(nodes)) as executor:
        futures = [executor.submit(fetch_block_number, (c, u)) for c, u in nodes.items()]
        for future in as_completed(futures):
            client, block_num, error = future.result()
            if error:
                block_errors[client] = error
            else:
                block_numbers[client] = block_num

    # Print results sorted by client name
    print("\nClient head blocks:")
    for client in sorted(nodes.keys()):
        if client in block_numbers:
            print(f"  {client}: {block_numbers[client]}")
        else:
            print(f"  {client}: ERROR - {block_errors[client]}")

    if not block_numbers:
        print("\nERROR: Could not get block number from any node")
        sys.exit(1)

    # Step 2: Validate block range and determine target block
    if args.block:
        target_block = args.block
        print(f"\nTarget block (specified): {target_block}")
    else:
        min_block = min(block_numbers.values())
        max_block = max(block_numbers.values())
        block_diff = max_block - min_block

        if block_diff >= 16:
            print(f"\nERROR: Block difference ({block_diff}) >= 16, nodes too far out of sync")
            sys.exit(1)

        target_block = min_block
        print(f"\nTarget block (min of heads): {target_block}")

    # Step 3: Get transactions from blocks
    # Use first available node to get transactions
    first_client = next(iter(nodes))
    first_url = nodes[first_client]

    # Collect transactions: list of (block_number, tx_hash)
    tx_list = []

    if args.tx:
        # Single transaction mode
        tx_list = [(target_block, args.tx)]
        print(f"\nTracing specified transaction: {args.tx}")
    else:
        # Collect transactions from blocks until max_txs is reached
        current_block = target_block
        max_txs = args.max_txs or float('inf')

        print(f"\nCollecting transactions (max: {args.max_txs or 'unlimited'})...")

        while len(tx_list) < max_txs and current_block > 0 and not _shutdown_requested:
            try:
                block = get_block_with_transactions(first_url, current_block)
            except Exception as e:
                print(f"  Block {current_block}: ERROR - {e}")
                break

            transactions = block.get("transactions", [])
            block_tx_hashes = [tx["hash"] if isinstance(tx, dict) else tx for tx in transactions]

            # Add transactions from this block
            remaining = max_txs - len(tx_list)
            txs_to_add = block_tx_hashes[:int(remaining)]
            for tx_hash in txs_to_add:
                tx_list.append((current_block, tx_hash))

            print(f"  Block {current_block}: {len(txs_to_add)} txs (total: {len(tx_list)})")

            if len(tx_list) >= max_txs:
                break

            current_block -= 1

        if _shutdown_requested and not tx_list:
            print("\nShutdown requested during collection, exiting...")
            sys.exit(0)

    print(f"\nCollected {len(tx_list)} transaction(s) to trace")

    # Filter transactions by gas usage if --max-gas is specified
    if args.max_gas and tx_list:
        print(f"\nFiltering transactions by gas usage (max: {args.max_gas:,})...")

        def fetch_receipt(block_tx):
            block_num, tx_hash = block_tx
            try:
                receipt = get_transaction_receipt(first_url, tx_hash)
                gas_used = int(receipt["gasUsed"], 16)
                return block_num, tx_hash, gas_used, None
            except Exception as e:
                return block_num, tx_hash, None, str(e)

        filtered_tx_list = []
        skipped = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(fetch_receipt, (b, t)) for b, t in tx_list]
            for future in as_completed(futures):
                if _shutdown_requested:
                    break
                block_num, tx_hash, gas_used, error = future.result()
                if error:
                    # If we can't get receipt, include it anyway
                    filtered_tx_list.append((block_num, tx_hash))
                elif gas_used <= args.max_gas:
                    filtered_tx_list.append((block_num, tx_hash))
                else:
                    skipped += 1
                    print(f"  Skipped {tx_hash[:16]}... ({gas_used:,} gas)")

        print(f"  Kept {len(filtered_tx_list)}, skipped {skipped} high-gas transactions")
        tx_list = filtered_tx_list

    if not tx_list:
        print("ERROR: No transactions to trace")
        sys.exit(1)

    # Step 4: Trace transactions on all nodes
    all_comparisons = {}
    traced_tx_list = []  # Track actually traced transactions

    for tx_idx, (block_num, tx_hash) in enumerate(tx_list):
        # Check for shutdown request before starting next transaction
        if _shutdown_requested:
            print(f"\nStopping early due to shutdown request ({tx_idx}/{len(tx_list)} traced)")
            break

        print(f"\n[{tx_idx + 1}/{len(tx_list)}] Block {block_num} - {tx_hash}...")
        block_dir = output_dir / str(block_num)
        tx_dir = block_dir / tx_hash
        traces = {}

        def trace_on_node(client_url):
            client, url = client_url
            try:
                trace = trace_transaction(url, tx_hash)
                return client, trace, None
            except Exception as e:
                return client, None, str(e)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(trace_on_node, (c, u)): c for c, u in nodes.items()}

            for future in as_completed(futures):
                client, trace, error = future.result()
                if error:
                    print(f"    {client}: ERROR - {error[:60]}...")
                    traces[client] = {"error": error}
                    save_trace(tx_dir, client, None, error)
                else:
                    print(f"    {client}: OK ({len(trace.get('structLogs', []))} structLogs)")
                    traces[client] = trace
                    save_trace(tx_dir, client, trace)

        # Step 5: Compare traces
        comparison = compare_traces(traces)
        all_comparisons[tx_hash] = comparison
        traced_tx_list.append((block_num, tx_hash))

        # Save comparison
        comparison_file = tx_dir / "_comparison.json"
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2)

        # Print summary
        print_comparison_summary(tx_hash, comparison)

    # Final summary
    print(f"\n{'='*80}")
    if _shutdown_requested:
        print("FINAL SUMMARY (interrupted)")
    else:
        print("FINAL SUMMARY")
    print(f"{'='*80}")

    if traced_tx_list:
        blocks_covered = sorted(set(b for b, _ in traced_tx_list))
        if len(blocks_covered) == 1:
            print(f"Block: {blocks_covered[0]}")
        else:
            print(f"Blocks: {blocks_covered[-1]} - {blocks_covered[0]} ({len(blocks_covered)} blocks)")

        if _shutdown_requested:
            print(f"Transactions traced: {len(traced_tx_list)}/{len(tx_list)} (interrupted)")
        else:
            print(f"Transactions traced: {len(traced_tx_list)}")

        print(f"Output directory: {output_dir.absolute()}")

        txs_with_diffs = sum(1 for c in all_comparisons.values() if c["differences"])
        print(f"\nTransactions with differences: {txs_with_diffs}/{len(traced_tx_list)}")
    else:
        print("No transactions were traced.")

    # Aggregate difference types
    all_diff_types = defaultdict(int)
    for comparison in all_comparisons.values():
        for diff in comparison["differences"]:
            all_diff_types[diff["type"]] += 1

    if all_diff_types:
        print("\nDifference types found:")
        for diff_type, count in sorted(all_diff_types.items(), key=lambda x: -x[1]):
            print(f"  {diff_type}: {count}")


if __name__ == "__main__":
    main()
