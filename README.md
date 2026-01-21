# Ethereum Client Trace Comparison Tool

Compare `debug_traceTransaction` output across multiple Ethereum execution clients to identify behavioral differences and quirks.

## Requirements

```bash
# Install all dependencies (including AI analysis)
pip install -r requirements.txt

# Or install only required dependencies (no AI analysis)
pip install requests
```

The AI analysis feature requires `claude-agent-sdk` and `anyio`. If not installed, the tool will skip AI analysis automatically.

## Workflow

The comparison process has two stages:

1. **Collect traces** - Query multiple clients and store raw trace data
2. **Analyze traces** - Process collected data and generate reports

## Step 1: Collect Traces

Use `compare_traces.py` to collect `debug_traceTransaction` output from multiple clients.

```bash
# Compare two clients
python compare_traces.py \
  --geth http://geth:8545 \
  --reth http://reth:8545

# Compare all supported clients
python compare_traces.py \
  --geth http://geth:8545 \
  --besu http://besu:8545 \
  --reth http://reth:8545 \
  --nethermind http://nethermind:8545 \
  --erigon http://erigon:8545 \
  --ethrex http://ethrex:8545

# Trace a specific block
python compare_traces.py \
  --geth http://geth:8545 \
  --reth http://reth:8545 \
  --block 19000000

# Trace a specific transaction
python compare_traces.py \
  --geth http://geth:8545 \
  --reth http://reth:8545 \
  --tx 0x123...abc

# Limit transactions per block
python compare_traces.py \
  --geth http://geth:8545 \
  --reth http://reth:8545 \
  --max-txs 10

# Skip high-gas transactions to prevent OOM
python compare_traces.py \
  --geth http://geth:8545 \
  --reth http://reth:8545 \
  --max-gas 5000000

# Combine limits: trace up to 100 txs, skip those over 10M gas
python compare_traces.py \
  --geth http://geth:8545 \
  --reth http://reth:8545 \
  --max-txs 100 \
  --max-gas 10000000

# With authentication headers
python compare_traces.py \
  --geth http://geth:8545 \
  --reth http://reth:8545 \
  -H "Authorization: Bearer token123" \
  -H "X-API-Key: mykey"
```

### Options

| Flag | Description |
|------|-------------|
| `--geth URL` | Geth RPC endpoint |
| `--besu URL` | Besu RPC endpoint |
| `--reth URL` | Reth RPC endpoint |
| `--nethermind URL` | Nethermind RPC endpoint |
| `--erigon URL` | Erigon RPC endpoint |
| `--ethrex URL` | Ethrex RPC endpoint |
| `--block N` | Starting block number (default: auto-detect from node heads) |
| `--tx HASH` | Trace only this transaction |
| `--max-txs N` | Maximum transactions to trace (spans multiple blocks if needed) |
| `--max-gas N` | Skip transactions using more than N gas (prevents OOM on large txs) |
| `--output-dir DIR` | Output directory (default: `./traces`) |
| `--workers N` | Parallel workers (default: 5) |
| `--header`, `-H` | HTTP header for requests, format `Name: Value` (can be repeated) |

At least 2 client endpoints are required.

## Step 2: Analyze Traces

After collecting traces, run `analyze_traces.py` to generate reports.

```bash
# Use default directories (reads from ./traces, outputs to ./reports)
python analyze_traces.py

# Specify custom directories
python analyze_traces.py --input-dir ./my-traces --output-dir ./my-reports

# Disable AI analysis
python analyze_traces.py --no-ai
```

### Options

| Flag | Description |
|------|-------------|
| `--input-dir DIR` | Input directory containing traces (default: `./traces`) |
| `--output-dir DIR` | Output directory for reports (default: `./reports`) |
| `--no-ai` | Disable AI analysis even if claude-code-sdk is available |

This reads from the input directory and outputs:
- `summary.md` - Overall comparison summary
- `<client>.md` - Per-client quirk reports (e.g., `geth.md`, `reth.md`)
- `ai_analysis.md` - AI-generated analysis of differences (if enabled)

AI analysis is automatically enabled when `claude-code-sdk` is installed, and skipped otherwise.

## Directory Structure

```
traces/
├── <block_number>/
│   ├── <tx_hash_1>/
│   │   ├── geth.json           # Raw trace from geth
│   │   ├── besu.json           # Raw trace from besu
│   │   ├── reth.json           # Raw trace from reth
│   │   ├── ...                 # Other clients
│   │   └── _comparison.json    # Diff summary for this tx
│   ├── <tx_hash_2>/
│   │   └── ...
│   └── ...
└── <another_block>/
    └── ...
```

### File Contents

**`<client>.json`** - Raw `debug_traceTransaction` response:
```json
{
  "gas": 21000,
  "failed": false,
  "returnValue": "0x",
  "structLogs": [
    {
      "pc": 0,
      "op": "PUSH1",
      "gas": 79000,
      "gasCost": 3,
      "depth": 1,
      "stack": [],
      "memory": [],
      "storage": {}
    }
  ]
}
```

**`_comparison.json`** - Comparison results:
```json
{
  "nodes_compared": ["geth", "reth", "besu"],
  "errors": {},
  "differences": [
    {
      "type": "missing_key",
      "index": 42,
      "key": "memory",
      "missing_in": ["besu"],
      "present_in": ["geth", "reth"]
    }
  ]
}
```

### Difference Types

| Type | Description |
|------|-------------|
| `structlog_count_mismatch` | Clients returned different numbers of structLog entries |
| `missing_key` | A key present in some clients is missing in others |
| `type_mismatch` | Same key has different types across clients |
| `value_mismatch` | Same key has different values (after normalization) |
| `top_level_mismatch` | Top-level fields (gas, returnValue) differ |

## Example Session

```bash
# 1. Collect traces from 3 clients for block 19000000
python compare_traces.py \
  --geth http://localhost:8545 \
  --reth http://localhost:8546 \
  --besu http://localhost:8547 \
  --block 19000000 \
  --max-txs 50

# 2. Analyze the collected traces
python analyze_traces.py

# 3. View results
cat reports/summary.md
cat reports/geth.md
cat reports/ai_analysis.md
```

### Custom Directories

```bash
# Collect to custom directory
python compare_traces.py \
  --geth http://localhost:8545 \
  --reth http://localhost:8546 \
  --output-dir ./my-traces

# Analyze from custom directory, output reports elsewhere
python analyze_traces.py \
  --input-dir ./my-traces \
  --output-dir ./my-reports
```
