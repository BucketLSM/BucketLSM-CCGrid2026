#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <algorithm>

namespace ROCKSDB_NAMESPACE {

class FileMetaData;

// User key range of a single L0 file (kept in memory persistently)
struct L0FileRange {
  std::string smallest_ukey;
  std::string largest_ukey;
  uint64_t file_number = 0;
};

class L0RangeIndex {
 public:
  L0RangeIndex() = default;

  // Incrementally update the index from the current Version's L0 file list.
  // Copies smallest/largest user_key and file number from LevelFiles(0).
  void RefreshFrom(const std::vector<FileMetaData*>& l0_files);

  // Called while holding the lock before flush; copies ranges into out
  // so they can be used after the lock is released.
  void SnapshotRanges(std::vector<L0FileRange>* out) const;

  // For debugging/testing
  size_t Size() const { return map_.size(); }

 private:
  // file_number -> range
  std::unordered_map<uint64_t, L0FileRange> map_;
  // Sorted cache by ascending smallest key (refreshed on snapshot)
  mutable std::vector<L0FileRange> sorted_cache_;
  mutable bool cache_dirty_ = true;
  mutable std::mutex mutex_;
};

}  // namespace ROCKSDB_NAMESPACE