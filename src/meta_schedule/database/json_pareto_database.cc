#include <tvm/ffi/reflection/registry.h>

#include <algorithm>
#include <limits>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../module_equality.h"
#include "../utils.h"
namespace tvm {
namespace meta_schedule {

/**************** Implementation of Pareto Optimization Utilities ****************/

double GetMeanFromFloatImmArray(const ffi::Optional<ffi::Array<FloatImm>>& arr,
                                double default_value) {
  if (!arr.defined() || arr.value().empty()) {
    return default_value;
  }
  double sum = 0.0;
  for (const FloatImm& val : arr.value()) {
    sum += val->value;
  }
  return sum / arr.value().size();
}

std::vector<int> NonDominatedSort(const std::vector<double>& obj1_values,
                                  const std::vector<double>& obj2_values) {
  int n = obj1_values.size();
  ICHECK_EQ(obj2_values.size(), static_cast<size_t>(n))
      << "Objective value vectors must have the same size";
  
  std::vector<int> ranks(n, 0);
  std::vector<bool> processed(n, false);
  
  int current_rank = 1;
  int remaining = n;
  
  while (remaining > 0) {
    // Find all non-dominated solutions (current Pareto front)
    std::vector<int> front;
    for (int i = 0; i < n; ++i) {
      if (processed[i]) continue;
      
      bool is_dominated = false;
      // Check against all other unprocessed solutions
      for (int j = 0; j < n; ++j) {
        if (i == j || processed[j]) continue;
        if (ParetoDominates(obj1_values[j], obj2_values[j], obj1_values[i], obj2_values[i])) {
          is_dominated = true;
          break;
        }
      }
      
      if (!is_dominated) {
        front.push_back(i);
      }
    }
    
    // Safety check: front should never be empty if there are unprocessed solutions
    if (front.empty()) {
      // Assign all remaining solutions to current rank to avoid infinite loop
      for (int i = 0; i < n; ++i) {
        if (!processed[i]) {
          ranks[i] = current_rank;
          processed[i] = true;
          remaining--;
        }
      }
    } else {
      // Assign rank to the front
      for (int idx : front) {
        ranks[idx] = current_rank;
        processed[idx] = true;
        remaining--;
      }
    }
    
    current_rank++;
  }
  
  return ranks;
}

std::vector<double> CalculateCrowdingDistance(const std::vector<double>& obj1_values,
                                              const std::vector<double>& obj2_values) {
  int n = obj1_values.size();
  ICHECK_EQ(obj2_values.size(), static_cast<size_t>(n))
      << "Objective value vectors must have the same size";
  
  std::vector<double> distances(n, 0.0);
  
  if (n <= 2) {
    // Boundary points get infinite distance
    std::fill(distances.begin(), distances.end(), std::numeric_limits<double>::max());
    return distances;
  }
  
  // Find min and max for normalization
  double min_obj1 = *std::min_element(obj1_values.begin(), obj1_values.end());
  double max_obj1 = *std::max_element(obj1_values.begin(), obj1_values.end());
  double min_obj2 = *std::min_element(obj2_values.begin(), obj2_values.end());
  double max_obj2 = *std::max_element(obj2_values.begin(), obj2_values.end());
  
  double obj1_range = max_obj1 - min_obj1;
  double obj2_range = max_obj2 - min_obj2;
  
  // Create index arrays for sorting
  std::vector<int> obj1_indices(n);
  std::vector<int> obj2_indices(n);
  for (int i = 0; i < n; ++i) {
    obj1_indices[i] = i;
    obj2_indices[i] = i;
  }
  
  // Sort by objective 1 (ascending - lower is better)
  std::sort(obj1_indices.begin(), obj1_indices.end(),
            [&obj1_values](int i, int j) { return obj1_values[i] < obj1_values[j]; });
  // Sort by objective 2 (ascending - lower is better)
  std::sort(obj2_indices.begin(), obj2_indices.end(),
            [&obj2_values](int i, int j) { return obj2_values[i] < obj2_values[j]; });
  
  // Boundary points get infinite distance
  if (obj1_range > 1e-10) {
    distances[obj1_indices[0]] = std::numeric_limits<double>::max();
    distances[obj1_indices[n - 1]] = std::numeric_limits<double>::max();
  }
  if (obj2_range > 1e-10) {
    distances[obj2_indices[0]] = std::numeric_limits<double>::max();
    distances[obj2_indices[n - 1]] = std::numeric_limits<double>::max();
  }
  
  // Calculate distance for non-boundary points
  if (obj1_range > 1e-10) {
    for (int i = 1; i < n - 1; ++i) {
      int idx = obj1_indices[i];
      distances[idx] += (obj1_values[obj1_indices[i + 1]] - obj1_values[obj1_indices[i - 1]]) / obj1_range;
    }
  }
  if (obj2_range > 1e-10) {
    for (int i = 1; i < n - 1; ++i) {
      int idx = obj2_indices[i];
      distances[idx] += (obj2_values[obj2_indices[i + 1]] - obj2_values[obj2_indices[i - 1]]) / obj2_range;
    }
  }
  
  return distances;
}

bool TuningRecordDominates(const TuningRecord& a, const TuningRecord& b) {
  if (!a->IsValid() || !b->IsValid()) {
    return false;
  }
  double a_time = GetMeanFromFloatImmArray(a->run_secs, std::numeric_limits<double>::max());
  double b_time = GetMeanFromFloatImmArray(b->run_secs, std::numeric_limits<double>::max());
  double a_bw = GetMeanFromFloatImmArray(a->bw_mbps, std::numeric_limits<double>::max());
  double b_bw = GetMeanFromFloatImmArray(b->bw_mbps, std::numeric_limits<double>::max());
  
  // Both latency and bandwidth should be MINIMIZED (lower is better)
  return ParetoDominates(a_time, a_bw, b_time, b_bw);
}

std::vector<double> CalculateCrowdingDistanceForRecords(
    const std::vector<TuningRecord>& records) {
  int n = records.size();
  std::vector<double> obj1_values(n);
  std::vector<double> obj2_values(n);
  
  for (int i = 0; i < n; ++i) {
    obj1_values[i] = GetMeanFromFloatImmArray(records[i]->run_secs, std::numeric_limits<double>::max());
    obj2_values[i] = GetMeanFromFloatImmArray(records[i]->bw_mbps, std::numeric_limits<double>::max());
  }
  
  return CalculateCrowdingDistance(obj1_values, obj2_values);
}

/*!
 * \brief A record with its Pareto rank and crowding distance.
 */
struct RankedRecord {
  TuningRecord record;
  int rank;
  double crowding_distance;

  RankedRecord(const TuningRecord& rec, int r, double cd) : record(rec), rank(r), crowding_distance(cd) {}

  // Sort by rank (ascending), then by crowding distance (descending), then by record pointer for stability
  bool operator<(const RankedRecord& other) const {
    if (rank != other.rank) {
      return rank < other.rank;
    }
    if (crowding_distance != other.crowding_distance) {
      return crowding_distance > other.crowding_distance;  // Higher distance is better
    }
    // If rank and crowding distance are equal, use record pointer for stable ordering
    return record.get() < other.record.get();
  }
};

/**************** JSONParetoDatabaseNode ****************/

class JSONParetoDatabaseNode : public DatabaseNode {
 public:
  explicit JSONParetoDatabaseNode(ffi::String mod_eq_name = "structural")
      : DatabaseNode(mod_eq_name),
        workloads2idx_(0, WorkloadHash(), WorkloadEqual(GetModuleEquality())),
        workload_records_(0, WorkloadHash(), WorkloadEqual(GetModuleEquality())) {}

  /*! \brief The path to the workload table */
  ffi::String path_workload;
  /*! \brief The path to the tuning record table */
  ffi::String path_tuning_record;
  /*! \brief All the workloads in the database */
  std::unordered_map<Workload, int, WorkloadHash, WorkloadEqual> workloads2idx_;
  /*! \brief All the tuning records per workload, sorted by Pareto rank and crowding distance */
  std::unordered_map<Workload, std::set<RankedRecord>, WorkloadHash, WorkloadEqual> workload_records_;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<JSONParetoDatabaseNode>()
        .def_ro("path_workload", &JSONParetoDatabaseNode::path_workload)
        .def_ro("path_tuning_record", &JSONParetoDatabaseNode::path_tuning_record);
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("meta_schedule.JSONParetoDatabase", JSONParetoDatabaseNode,
                                    DatabaseNode);

 public:
  bool HasWorkload(const IRModule& mod) final {
    return workloads2idx_.find(Workload(mod, GetModuleEquality().Hash(mod))) !=
           workloads2idx_.end();
  }

  Workload CommitWorkload(const IRModule& mod) final {
    // Try to insert `mod` into `workloads_`
    auto [it, inserted] =
        this->workloads2idx_.emplace(Workload(mod, GetModuleEquality().Hash(mod)), -1);
    Workload workload = it->first;
    // If `mod` is new in `workloads2idx_`, append it to the workload file
    if (inserted) {
      it->second = static_cast<int>(this->workloads2idx_.size()) - 1;
      // Only append to file if path is set (not during initialization)
      if (!this->path_workload.empty()) {
        JSONFileAppendLine(this->path_workload, JSONDumps(workload->AsJSON()));
      }
    }
    return it->first;
  }

  /*!
   * \brief Add a record to the database and update Pareto ranking.
   * \param record The record to add.
   * \param append_to_file Whether to append the record to the file.
   */
  void AddRecord(const TuningRecord& record, bool append_to_file) {
    if (!record->IsValid()) {
      return;
    }
    Workload workload = record->workload;
    
    // Ensure workload is in workloads2idx_ (for consistency)
    if (workloads2idx_.find(workload) == workloads2idx_.end()) {
      auto [it, inserted] = workloads2idx_.emplace(workload, -1);
      if (inserted) {
        it->second = static_cast<int>(workloads2idx_.size()) - 1;
      }
    }
    
    // Get existing records for this workload
    auto& records_set = workload_records_[workload];
    
    // Convert to vector for easier manipulation
    std::vector<TuningRecord> existing_records;
    existing_records.reserve(records_set.size());
    for (const auto& ranked_rec : records_set) {
      existing_records.push_back(ranked_rec.record);
    }
    
    // Add the new record
    existing_records.push_back(record);
    
    // Extract objective values for non-dominated sorting
    int n = existing_records.size();
    std::vector<double> obj1_values(n);  // run_secs (time) - minimize
    std::vector<double> obj2_values(n);   // bw_mbps (bandwidth) - minimize (lower is better)
    for (int i = 0; i < n; ++i) {
      obj1_values[i] = GetMeanFromFloatImmArray(existing_records[i]->run_secs, std::numeric_limits<double>::max());
      obj2_values[i] = GetMeanFromFloatImmArray(existing_records[i]->bw_mbps, std::numeric_limits<double>::max());
    }
    
    // Recalculate Pareto ranks for all records using modularized function
    std::vector<int> ranks = NonDominatedSort(obj1_values, obj2_values);
    
    // Calculate crowding distances for each rank
    std::unordered_map<int, std::vector<TuningRecord>> records_by_rank;
    std::unordered_map<int, std::vector<int>> indices_by_rank;
    for (int i = 0; i < n; ++i) {
      records_by_rank[ranks[i]].push_back(existing_records[i]);
      indices_by_rank[ranks[i]].push_back(i);
    }
    
    std::vector<double> crowding_distances(n, 0.0);
    for (auto& [rank, rank_records] : records_by_rank) {
      std::vector<double> rank_distances = CalculateCrowdingDistanceForRecords(rank_records);
      const auto& rank_indices = indices_by_rank[rank];
      for (size_t i = 0; i < rank_indices.size(); ++i) {
        crowding_distances[rank_indices[i]] = rank_distances[i];
      }
    }
    
    // Rebuild the sorted set
    records_set.clear();
    for (int i = 0; i < n; ++i) {
      records_set.emplace(existing_records[i], ranks[i], crowding_distances[i]);
    }
    
    // Append to file if requested
    if (append_to_file && !this->path_tuning_record.empty()) {
      // Ensure workload is in workloads2idx_ (commit it if not already)
      auto [it, inserted] = this->workloads2idx_.emplace(workload, -1);
      if (inserted) {
        it->second = static_cast<int>(this->workloads2idx_.size()) - 1;
        if (!this->path_workload.empty()) {
          JSONFileAppendLine(this->path_workload, JSONDumps(workload->AsJSON()));
        }
      }
      JSONFileAppendLine(this->path_tuning_record,
                         JSONDumps(ffi::Array<Any>{
                             /*workload_index=*/Integer(it->second),
                             /*tuning_record=*/record->AsJSON()  //
                         }));
    }
  }

 public:
  void CommitTuningRecord(const TuningRecord& record) final {
    AddRecord(record, /*append_to_file=*/true);
  }

  ffi::Array<TuningRecord> GetTopK(const Workload& workload, int top_k) final {
    CHECK_GE(top_k, 0) << "ValueError: top_k must be non-negative";
    if (top_k == 0) {
      return {};
    }
    
    auto it = workload_records_.find(workload);
    if (it == workload_records_.end()) {
      return {};
    }
    
    ffi::Array<TuningRecord> results;
    results.reserve(top_k);
    
    // Get records with rank 1 (Pareto front), sorted by crowding distance (descending)
    for (const auto& ranked_rec : it->second) {
      if (ranked_rec.rank == 1) {
        results.push_back(ranked_rec.record);
        if (results.size() == static_cast<size_t>(top_k)) {
          break;
        }
      } else {
        // Since records are sorted by rank, we can stop once we see rank > 1
        break;
      }
    }
    
    return results;
  }

  ffi::Array<TuningRecord> GetAllTuningRecords() final {
    ffi::Array<TuningRecord> results;
    for (const auto& [workload, records_set] : workload_records_) {
      for (const auto& ranked_rec : records_set) {
        results.push_back(ranked_rec.record);
      }
    }
    return results;
  }

  int64_t Size() final {
    int64_t total = 0;
    for (const auto& [workload, records_set] : workload_records_) {
      total += records_set.size();
    }
    return total;
  }
};

Database Database::JSONParetoDatabase(ffi::String path_workload, ffi::String path_tuning_record,
                                      bool allow_missing, ffi::String mod_eq_name) {
  int num_threads = std::thread::hardware_concurrency();
  ObjectPtr<JSONParetoDatabaseNode> n = ffi::make_object<JSONParetoDatabaseNode>(mod_eq_name);
  std::vector<Workload> workloads;
  // Load `n->workloads2idx_` from `path_workload`
  {
    std::vector<Any> json_objs = JSONFileReadLines(path_workload, num_threads, allow_missing);
    int n_objs = json_objs.size();
    n->workloads2idx_.reserve(n_objs);
    workloads.reserve(n_objs);
    for (int i = 0; i < n_objs; ++i) {
      Workload workload = Workload::FromJSON(json_objs[i].cast<ObjectRef>());
      auto recalc_hash = n->GetModuleEquality().Hash(workload->mod);
      // Todo(tvm-team): re-enable the shash check when we get environment
      // independent structural hash values.
      if (recalc_hash != workload->shash) {
        ObjectPtr<WorkloadNode> wkl = ffi::make_object<WorkloadNode>(*workload.get());
        wkl->shash = recalc_hash;
        workload = Workload(wkl);
      }
      n->workloads2idx_.emplace(workload, i);
      workloads.push_back(workload);
    }
  }
  // Load `n->workload_records_` from `path_tuning_record`
  {
    std::vector<Any> json_objs = JSONFileReadLines(path_tuning_record, num_threads, allow_missing);
    std::vector<TuningRecord> records;
    records.resize(json_objs.size(), TuningRecord{ffi::UnsafeInit()});
    support::parallel_for_dynamic(
        0, json_objs.size(), num_threads, [&](int thread_id, int task_id) {
          auto json_obj = json_objs[task_id].cast<ObjectRef>();
          Workload workload{ffi::UnsafeInit()};
          try {
            const ffi::ArrayObj* arr = json_obj.as<ffi::ArrayObj>();
            ICHECK_EQ(arr->size(), 2);
            int64_t workload_index = arr->at(0).cast<IntImm>()->value;
            ICHECK(workload_index >= 0 && static_cast<size_t>(workload_index) < workloads.size());
            workload = workloads[workload_index];
            records[task_id] = TuningRecord::FromJSON(arr->at(1).cast<ObjectRef>(), workload);
          } catch (std::runtime_error& e) {
            LOG(FATAL) << "ValueError: Unable to parse TuningRecord, on line " << (task_id + 1)
                       << " of file " << path_tuning_record << ". The workload is:\n"
                       << (workload.defined() ? workload->mod->Script() : "(null)")
                       << "\nThe JSONObject of TuningRecord is:\n"
                       << json_obj << "\nThe error message is:\n"
                       << e.what();
          }
        });
    // Group records by workload for efficient bulk loading
    std::unordered_map<Workload, std::vector<TuningRecord>, WorkloadHash, WorkloadEqual>
        records_by_workload(0, WorkloadHash(), WorkloadEqual(n->GetModuleEquality()));
    for (const TuningRecord& record : records) {
      if (record->IsValid()) {
        records_by_workload[record->workload].push_back(record);
      }
    }
    // Add all records for each workload at once
    for (auto& [workload, workload_records] : records_by_workload) {
      for (const TuningRecord& record : workload_records) {
        n->AddRecord(record, /*append_to_file=*/false);
      }
    }
  }
  n->path_workload = path_workload;
  n->path_tuning_record = path_tuning_record;
  return Database(n);
}

TVM_FFI_STATIC_INIT_BLOCK() { JSONParetoDatabaseNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("meta_schedule.DatabaseJSONParetoDatabase", Database::JSONParetoDatabase);
}

}  // namespace meta_schedule
}  // namespace tvm
