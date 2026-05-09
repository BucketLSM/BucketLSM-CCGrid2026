#include "db/db_l0_range_index.h"
#include "db/version_set.h"  // For FileMetaData

namespace ROCKSDB_NAMESPACE {

void L0RangeIndex::RefreshFrom(const std::vector<FileMetaData*>& l0_files) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Clear existing ranges
  map_.clear();

  // Add current L0 files
  for (const FileMetaData* file : l0_files) {
    L0FileRange range;
    range.smallest_ukey = file->smallest.user_key().ToString();
    range.largest_ukey = file->largest.user_key().ToString();
    range.file_number = file->fd.GetNumber();

    map_[file->fd.GetNumber()] = range;
  }

  cache_dirty_ = true;
}

void L0RangeIndex::SnapshotRanges(std::vector<L0FileRange>* out) const {
  std::lock_guard<std::mutex> lock(mutex_);

  out->clear();
  out->reserve(map_.size());

  // Update sorted cache if needed
  if (cache_dirty_) {
    sorted_cache_.clear();
    sorted_cache_.reserve(map_.size());

    for (const auto& pair : map_) {
      sorted_cache_.push_back(pair.second);
    }

    // Sort by smallest user key
    std::sort(sorted_cache_.begin(), sorted_cache_.end(),
              [](const L0FileRange& a, const L0FileRange& b) {
                return a.smallest_ukey < b.smallest_ukey;
              });

    cache_dirty_ = false;
  }

  // Copy to output
  *out = sorted_cache_;
}

}  // namespace ROCKSDB_NAMESPACE