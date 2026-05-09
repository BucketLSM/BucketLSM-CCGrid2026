//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "db/l0_group_tracker.h"

#include <algorithm>
#include <chrono>
#include <mutex>

#include "db/dbformat.h"
#include "rocksdb/env.h"

namespace ROCKSDB_NAMESPACE {

L0GroupTracker::L0GroupTracker(const InternalKeyComparator* cmp)
    : icmp_(cmp), next_group_id_(1) {}

void L0GroupTracker::RegisterFlushResult(
    const std::vector<FileMetaData*>& flushed_files) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto* file : flushed_files) {
    if (file == nullptr) continue;

    L0Group* best_group = AssignFileToGroup(file);
    if (best_group != nullptr) {
      best_group->AddFile(file->fd.GetNumber(), file->smallest, file->largest,
                          file->fd.GetFileSize(), *icmp_);
    }
  }

  CleanupEmptyGroups();
}

void L0GroupTracker::RemoveFiles(const std::vector<uint64_t>& file_numbers) {
  std::lock_guard<std::mutex> lock(mutex_);

  for (uint64_t file_num : file_numbers) {
    for (auto& group : groups_) {
      auto it = std::find(group.file_numbers.begin(), group.file_numbers.end(), file_num);
      if (it != group.file_numbers.end()) {
        group.file_numbers.erase(it);
        // Note: We don't update group bounds here for simplicity
        // In practice, you might want to recalculate bounds
        break;
      }
    }
  }

  CleanupEmptyGroups();
}

L0GroupTracker::L0Group* L0GroupTracker::GetGroupForCompaction() {
  std::lock_guard<std::mutex> lock(mutex_);

  for (auto& group : groups_) {
    if (group.ShouldCompact()) {
      return &group;
    }
  }
  return nullptr;
}

std::vector<L0GroupTracker::L0Group> L0GroupTracker::GetAllGroups() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return groups_;
}

void L0GroupTracker::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  groups_.clear();
  next_group_id_ = 1;
}

double L0GroupTracker::CalculateOverlap(const FileMetaData* file,
                                        const L0Group& group) const {
  if (group.Empty()) {
    return 0.0;
  }

  // Simple overlap calculation: check if there's any key range overlap
  bool has_overlap = icmp_->Compare(file->smallest, group.largest) <= 0 &&
                     icmp_->Compare(group.smallest, file->largest) <= 0;

  if (!has_overlap) {
    return 0.0;
  }

  // For simplicity, return a fixed overlap ratio when there is overlap
  // In a more sophisticated implementation, you could calculate actual
  // overlap ratio based on key distribution
  return 0.7;  // 70% overlap when ranges intersect
}

L0GroupTracker::L0Group* L0GroupTracker::AssignFileToGroup(const FileMetaData* file) {
  L0Group* best_group = nullptr;
  double max_overlap = 0.0;
  const double overlap_threshold = 0.5;  // 50% overlap required to join group

  // Find the group with maximum overlap
  for (auto& group : groups_) {
    if (group.Empty()) continue;

    double overlap = CalculateOverlap(file, group);
    if (overlap > max_overlap) {
      max_overlap = overlap;
      best_group = &group;
    }
  }

  // If overlap is significant enough, add to existing group
  if (max_overlap > overlap_threshold) {
    return best_group;
  }

  // Create new group for this file
  groups_.emplace_back(next_group_id_++, file->smallest, file->largest);
  groups_.back().creation_time =
      static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());

  return &groups_.back();
}

void L0GroupTracker::CleanupEmptyGroups() {
  groups_.erase(
      std::remove_if(groups_.begin(), groups_.end(),
                     [](const L0Group& group) { return group.Empty(); }),
      groups_.end());
}

}  // namespace ROCKSDB_NAMESPACE