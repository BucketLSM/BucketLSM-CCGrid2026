#pragma once

namespace ROCKSDB_NAMESPACE {

class VersionStorageInfo;

// Helper function to get max file count among L0 groups for hybrid flush algo
int GetMaxL0GroupFileCount(const VersionStorageInfo* vstorage);

}  // namespace ROCKSDB_NAMESPACE