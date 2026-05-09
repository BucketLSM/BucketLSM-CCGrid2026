#include "db/l0_group_helper.h"
#include "db/version_set.h"

namespace ROCKSDB_NAMESPACE {

// Helper function to get max file count among L0 groups for hybrid flush algo
int GetMaxL0GroupFileCount(const VersionStorageInfo* vstorage) {
  const auto& l0_files = vstorage->LevelFiles(0);
  if (l0_files.empty()) return 0;

  // Get user comparator
  const Comparator* ucmp = vstorage->InternalComparator()->user_comparator();

  // Build L0 groups by merging overlapping file ranges
  std::vector<std::pair<std::string, std::string>> groups;
  for (const auto* f : l0_files) {
    std::string file_lo = f->smallest.user_key().ToString();
    std::string file_hi = f->largest.user_key().ToString();

    bool merged = false;
    for (auto& g : groups) {
      // Check if this file overlaps with the group
      if (ucmp->Compare(file_lo, g.second) <= 0 &&
          ucmp->Compare(file_hi, g.first) >= 0) {
        // Expand the group
        if (ucmp->Compare(file_lo, g.first) < 0) g.first = file_lo;
        if (ucmp->Compare(file_hi, g.second) > 0) g.second = file_hi;
        merged = true;
        break;
      }
    }

    if (!merged) {
      groups.emplace_back(file_lo, file_hi);
    }
  }

  // Count files in each group and find max
  int max_count = 0;
  for (const auto& g : groups) {
    int count = 0;
    for (const auto* f : l0_files) {
      std::string file_lo = f->smallest.user_key().ToString();
      std::string file_hi = f->largest.user_key().ToString();

      // Check if file is in this group
      if (ucmp->Compare(file_lo, g.second) <= 0 &&
          ucmp->Compare(file_hi, g.first) >= 0) {
        count++;
      }
    }
    max_count = std::max(max_count, count);
  }

  return max_count;
}

}  // namespace ROCKSDB_NAMESPACE