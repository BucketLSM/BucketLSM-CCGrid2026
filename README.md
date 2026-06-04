## BucketLSM: Bucket-Based Flush and Parallel Compaction for LSM-Tree

BucketLSM is a modified [RocksDB](https://github.com/facebook/rocksdb) fork that introduces bucket-based flush and parallel compaction optimizations for LSM-tree storage engines. By splitting memtable flushes into multiple L0 files aligned to key-range buckets, BucketLSM enables parallel L0-to-L1 compaction and reduces write stalls under heavy write workloads.

This project is based on RocksDB, developed and maintained by Facebook Database Engineering Team, and built on earlier work on [LevelDB](https://github.com/google/leveldb).

---

## Key Features

- **Bucket Flush**: Splits a single memtable flush into multiple L0 SST files based on global key-range partitions (buckets), creating naturally non-overlapping L0 file groups.
- **Parallel Bucket Flush**: Flushes each bucket segment concurrently using multiple threads.
- **Dynamic Bucket Adjustment**: Automatically splits/merges bucket boundaries based on sliding-window workload statistics.
- **Parallel L0-to-L1 Compaction**: Compacts multiple L0 groups to L1 concurrently, since groups have non-overlapping key ranges.
- **Weighted L0 Group Selection**: Scores L0 groups by both file count and L1 overlap ratio to pick the best compaction candidate.
- **Intra-L0 Compaction**: Merges overlapping files within L0 to reduce read amplification when L0-to-L1 compaction is blocked.
- **Range-Aware L0 Scan**: Lazily opens L0 file iterators based on key-range overlap, reducing unnecessary I/O during range scans.


## Custom Options

All custom options are defined in `DBOptions` (`include/rocksdb/options.h`) and can be passed as `db_bench` flags.

### Flush Options

| Option | Type | Default | db_bench Flag | Description |
|--------|------|---------|---------------|-------------|
| `use_new_flush` | int | `0` | `--use_new_flush=N` | Flush mode. **0** = original RocksDB flush (single L0 file per memtable). **1** = Bucket flush (splits memtable into multiple L0 files based on global key-range buckets). |
| `parallel_split_flush` | bool | `false` | `--parallel_split_flush` | When enabled, bucket segments are flushed in parallel using multiple threads. Requires `use_new_flush=1`. |
| `initial_split_count` | int | `3` | `--initial_split_count=N` | Number of buckets to create when L0 is empty (bootstrapping). The first memtable flush is split into this many segments. Requires `use_new_flush=1`. |
| `enable_bucket_adjustment` | bool | `true` | `--enable_bucket_adjustment` | Allow dynamic bucket boundary split/merge based on sliding-window statistics (file count and data volume per bucket). When disabled, bucket boundaries remain fixed after initial creation. Requires `use_new_flush=1`. |

### Compaction Options

| Option | Type | Default | db_bench Flag | Description |
|--------|------|---------|---------------|-------------|
| `parallel_l0_compaction` | bool | `true` | `--parallel_l0_compaction` | Enable concurrent L0-to-L1 compactions. Since bucket flush produces non-overlapping L0 groups, multiple groups can be compacted to L1 in parallel. Requires `use_new_flush=1`. |
| `enable_intra_l0_compaction` | bool | `true` | `--enable_intra_l0_compaction` | Enable L0-to-L0 compaction. When L0-to-L1 compaction is blocked, overlapping L0 files within the same group are merged to reduce read amplification. |
| `l0_weight_compaction` | bool | `true` | `--l0_weight_compaction` | Use weighted scoring for L0 group compaction selection. Score = `alpha * (N / N_max) + (1 - alpha) * (1 - R / R_max)`, where N = file count in the group and R = overlap ratio with L1. When disabled, simply picks the group with the most files. Requires `use_new_flush=1`. |
| `l0_file_count_weight` | double | `0.5` | `--l0_file_count_weight=F` | The alpha weight in the scoring formula above. Higher value prioritizes reducing L0 file count; lower value prioritizes minimizing L1 overlap (compaction I/O). Valid range: 0.0 to 1.0. Requires `l0_weight_compaction=true`. |

### Read Path Options

| Option | Type | Default | db_bench Flag | Description |
|--------|------|---------|---------------|-------------|
| `new_range_scan` | bool | `false` | `--new_range_scan` | Enable range-aware L0 scanning. L0 SST file iterators are opened lazily based on key-range overlap with the current scan position, reducing unnecessary I/O during range scans. |

### Example: db_bench Usage

```bash
# Bucket flush with 4 initial buckets, parallel flush and compaction
./db_bench \
  --use_new_flush=1 \
  --initial_split_count=4 \
  --parallel_split_flush=true \
  --parallel_l0_compaction=true \
  --enable_bucket_adjustment=true \
  --l0_weight_compaction=true \
  --l0_file_count_weight=0.6 \
  --benchmarks=fillrandom \
  --num=10000000

# Original RocksDB behavior (baseline comparison)
./db_bench \
  --use_new_flush=0 \
  --benchmarks=fillrandom \
  --num=10000000
```

### Example: Programmatic Usage (C++)

```cpp
#include "rocksdb/db.h"
#include "rocksdb/options.h"

rocksdb::Options options;
options.create_if_missing = true;

// Enable bucket flush with parallel compaction
options.use_new_flush = 1;
options.initial_split_count = 4;
options.parallel_split_flush = true;
options.parallel_l0_compaction = true;
options.enable_bucket_adjustment = true;
options.l0_weight_compaction = true;
options.l0_file_count_weight = 0.5;

rocksdb::DB* db;
rocksdb::Status s = rocksdb::DB::Open(options, "/tmp/bucketlsm_db", &db);
```

---

## Modified Files

All custom modifications are marked with `//jw` comments in the source code. Key files:

| File | Description |
|------|-------------|
| `include/rocksdb/options.h` | Public option declarations |
| `options/db_options.cc/h` | Option definitions and serialization |
| `db/flush_job.cc/h` | Bucket flush and parallel flush implementation |
| `db/column_family.cc/h` | `BucketManager` (bucket partitioning and dynamic adjustment) |
| `db/version_set.cc/h` | L0 group tracking and pending bytes estimation |
| `db/compaction/compaction_picker_level.cc` | L0 group-based compaction selection and weighted scoring |
| `db/compaction/compaction_outputs.cc/h` | Bucket/group boundary-aligned output splitting |
| `db/compaction/compaction_job.cc` | L0 coverage logging for compaction analysis |
| `db/db_l0_range_index.cc/h` | L0 file key-range index for range-aware scanning |
| `db/l0_group_helper.cc/h` | Helper for L0 overlap group analysis |
| `db/l0_group_tracker.cc/h` | L0 group lifecycle tracking |
| `tools/db_bench_tool.cc` | Benchmark flag support for custom options |

---

## License

RocksDB is dual-licensed under both the GPLv2 (found in the COPYING file in the root directory) and Apache 2.0 License (found in the LICENSE.Apache file in the root directory).  You may select, at your option, one of the above-listed licenses.
