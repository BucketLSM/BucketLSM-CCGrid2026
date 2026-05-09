//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <unordered_map>

#include "rocksdb/types.h"
#include "db/dbformat.h"
#include "db/version_edit.h"

namespace ROCKSDB_NAMESPACE {

class InternalKeyComparator;

// L0GroupTracker manages groups of L0 files for group-aware compaction
class L0GroupTracker {
 public:
  struct L0Group {
    int group_id;
    std::vector<uint64_t> file_numbers;  // SST file numbers in this group
    InternalKey smallest, largest;       // Key range covered by this group
    uint64_t total_size;                 // Total size of files in this group
    uint64_t creation_time;              // When this group was created

    L0Group() : group_id(-1), total_size(0), creation_time(0) {}

    L0Group(int id, const InternalKey& s, const InternalKey& l)
        : group_id(id), smallest(s), largest(l), total_size(0), creation_time(0) {}

    bool ShouldCompact() const {
      // Trigger compaction when group has 4+ files
      return file_numbers.size() >= 4;
    }

    bool Empty() const {
      return file_numbers.empty();
    }

    void AddFile(uint64_t file_number, const InternalKey& file_smallest,
                 const InternalKey& file_largest, uint64_t file_size,
                 const InternalKeyComparator& cmp) {
      file_numbers.push_back(file_number);
      total_size += file_size;

      if (file_numbers.size() == 1) {
        // First file in group
        smallest = file_smallest;
        largest = file_largest;
      } else {
        // Expand group boundaries
        if (cmp.Compare(file_smallest, smallest) < 0) {
          smallest = file_smallest;
        }
        if (cmp.Compare(file_largest, largest) > 0) {
          largest = file_largest;
        }
      }
    }
  };

  explicit L0GroupTracker(const InternalKeyComparator* cmp);
  ~L0GroupTracker() = default;

  // Called when new L0 files are created (e.g., from flush)
  void RegisterFlushResult(const std::vector<FileMetaData*>& flushed_files);

  // Called when files are removed (e.g., after compaction)
  void RemoveFiles(const std::vector<uint64_t>& file_numbers);

  // Get a group that should be compacted, nullptr if none
  L0Group* GetGroupForCompaction();

  // Get total number of groups
  size_t GetGroupCount() const { return groups_.size(); }

  // Get group information for logging/debugging
  std::vector<L0Group> GetAllGroups() const;

  // Clear all groups (for testing or reset)
  void Clear();

 private:
  const InternalKeyComparator* icmp_;
  std::vector<L0Group> groups_;
  int next_group_id_;
  mutable std::mutex mutex_;  // Protect concurrent access

  // Calculate overlap ratio between file and group
  double CalculateOverlap(const FileMetaData* file, const L0Group& group) const;

  // Find the best group for a file, or create new group
  L0Group* AssignFileToGroup(const FileMetaData* file);

  // Remove empty groups
  void CleanupEmptyGroups();
};

}  // namespace ROCKSDB_NAMESPACE