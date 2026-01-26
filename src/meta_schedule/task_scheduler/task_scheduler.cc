/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/reflection/registry.h>

#include "../utils.h"
#include "tvm/ffi/object.h"

namespace tvm {
namespace meta_schedule {

TVM_FFI_STATIC_INIT_BLOCK() {
  TaskRecordNode::RegisterReflection();
  TaskSchedulerNode::RegisterReflection();
  PyTaskSchedulerNode::RegisterReflection();
}

TaskRecord::TaskRecord(TuneContext ctx, double task_weight) {
  ObjectPtr<TaskRecordNode> n = ffi::make_object<TaskRecordNode>();
  n->ctx = ctx;
  n->task_weight = task_weight;
  n->flop = 1.0;
  auto _ = Profiler::TimedScope("InitializeTask");
  CHECK(ctx->mod.defined()) << "ValueError: Require `context.mod`, but it is not defined";
  CHECK(ctx->space_generator.defined())
      << "ValueError: Require `context.space_generator`, but it is not defined";
  CHECK(ctx->search_strategy.defined())
      << "ValueError: Require `context.search_strategy`, but it is not defined";
  TVM_PY_LOG(INFO, ctx->logger) << "\n" << ctx->mod;
  ctx->Initialize();
  n->flop = std::max(1.0, tir::EstimateTIRFlops(ctx->mod.value()));
  this->data_ = std::move(n);
}

void SendToBuilder(TaskRecordNode* self, const Builder& builder) {
  auto _ = Profiler::TimedScope("SendToBuilder");
  ffi::Array<MeasureCandidate> candidates = self->measure_candidates.value();
  Target target = self->ctx->target.value();
  ffi::Array<BuilderInput> inputs;
  inputs.reserve(candidates.size());
  for (const MeasureCandidate& candidate : candidates) {
    inputs.push_back(BuilderInput(candidate->sch->mod(), target));
  }
  self->builder_results = builder->Build(inputs);
}

void SendToRunner(TaskRecordNode* self, const Runner& runner) {
  auto _ = Profiler::TimedScope("SendToRunner");
  ffi::Array<MeasureCandidate> candidates = self->measure_candidates.value();
  ffi::Array<BuilderResult> builder_results = self->builder_results.value();
  Target target = self->ctx->target.value();
  ICHECK_EQ(candidates.size(), builder_results.size());
  int n = candidates.size();
  int n_build_errors = 0;
  ffi::Array<RunnerInput> inputs;
  inputs.reserve(n);
  for (int i = 0; i < n; ++i) {
    const MeasureCandidate& candidate = candidates[i];
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.has_value()) {
      ++n_build_errors;
      continue;
    }
    inputs.push_back(RunnerInput(/*artifact_path=*/builder_result->artifact_path.value(),
                                 /*device_type=*/target->kind->name,
                                 /*args_info=*/candidate->args_info));
  }
  ffi::Array<RunnerFuture> futures = runner->Run(inputs);
  if (n_build_errors == 0) {
    self->runner_futures = futures;
    return;
  }
  ffi::Array<RunnerFuture> results;
  results.reserve(n);
  for (int i = 0, j = 0; i < n; ++i) {
    const BuilderResult& builder_result = builder_results[i];
    if (builder_result->error_msg.has_value()) {
      results.push_back(RunnerFuture(
          /*f_done=*/[]() -> bool { return true; },
          /*f_result=*/
          [msg = builder_result->error_msg]() -> RunnerResult {
            return RunnerResult(std::nullopt, std::nullopt, msg);
          }));
    } else {
      results.push_back(futures[j++]);
    }
  }
  self->runner_futures = results;
}

void TaskCleanUp(TaskRecordNode* self, int task_id, const ffi::Array<RunnerResult>& results) {
  ICHECK_EQ(self->builder_results.value().size(), results.size());
  ICHECK_EQ(self->runner_futures.value().size(), results.size());
  int n = results.size();
  std::string name = self->ctx->task_name.value();
  const ffi::Function& logger = self->ctx->logger;
  for (int i = 0; i < n; ++i) {
    const BuilderResult& builder_result = self->builder_results.value()[i];
    const MeasureCandidate& candidate = self->measure_candidates.value()[i];
    const RunnerResult& runner_result = results[i];
    ffi::Optional<ffi::String> error_msg = std::nullopt;
    int trials = self->latency_ms.size() + 1;
    double run_ms = 1e9;
    double bandwidth_mbps = 1e9;
    if ((error_msg = builder_result->error_msg)) {
      ++self->build_error_count;
    } else if ((error_msg = runner_result->error_msg)) {
      ++self->run_error_count;
    } else {
      run_ms = GetRunMsMedian(runner_result);
      bandwidth_mbps = GetBandwidthMbpsMedian(runner_result);
    }
    self->latency_ms.push_back(run_ms);
    self->bandwidth_mbps.push_back(bandwidth_mbps);
    if (error_msg) {
      const tir::Schedule& sch = candidate->sch;
      std::string err = error_msg.value();
      TVM_PY_LOG(INFO, logger) << std::fixed << std::setprecision(4)  //
                               << "[Task #" << task_id << ": " << name << "] Trial #" << trials
                               << ": Error in "
                               << (builder_result->error_msg.has_value() ? "building" : "running")
                               << ":\n"
                               << err << "\n"
                               << sch->mod() << "\n"
                               << Concat(sch->trace().value()->AsPython(false), "\n");
    } else {
      double best_ms = *std::min_element(self->latency_ms.begin(), self->latency_ms.end());
      TVM_PY_LOG(INFO, logger) << std::fixed << std::setprecision(4)  //
                               << "[Task #" << task_id << ": " << name << "] Trial #" << trials
                               << ": GFLOPs: " << (self->flop / run_ms / 1e6)
                               << ". Time: " << (run_ms * 1e3) << " us"
                               << ". Best GFLOPs: " << (self->flop / best_ms / 1e6);
    }
  }
  self->measure_candidates = std::nullopt;
  self->builder_results = std::nullopt;
  self->runner_futures = std::nullopt;
}

void TaskSchedulerNode::SetReferencePoint(TuneContext ctx, Builder builder, Runner runner) {
  TVM_PY_LOG(INFO, ctx->logger) << "Setting reference point for task: " << ctx->task_name;

  // Create a TaskRecord and keep it alive (don't use temporary)
  TaskRecord task_record = TaskRecord(ctx, 1.0);
  TaskRecordNode* task = task_record.get();
  IRModule mod = ctx->mod.value();

  // Generate an unoptimized reference point (worst possible point for hypervolume calculation)
  // Strategy: Start from the original module with NO schedule rules applied (empty trace),
  // then apply only postprocessors to ensure validity. This gives us the most basic,
  // unoptimized code with minimal bindings (e.g., GPU thread/block bindings) needed for validity.
  // Using a fixed seed (0) ensures deterministic behavior.

  // Create an empty trace - no schedule rules, no optimizations
  tir::Trace empty_trace = tir::Trace();

  // Apply only postprocessors with a fixed seed for deterministic behavior
  ThreadedTraceApply pp(ctx->space_generator.value()->postprocs.value());
  TRandState fixed_seed = 0;  // Fixed seed for determinism
  ffi::Optional<tir::Schedule> sch_pp = pp.Apply(mod, empty_trace, &fixed_seed);

  if (!sch_pp.has_value()) {
    TVM_PY_LOG(ERROR, ctx->logger)
        << "Error in setting reference point: Cannot apply postprocessors "
        << "to create a valid unoptimized implementation\n"
        << pp.SummarizeFailures();
    return;
  }

  // Send to builder
  ffi::Array<BuilderInput> builder_inputs;
  builder_inputs.push_back(BuilderInput(sch_pp.value()->mod(), ctx->target.value()));
  ffi::Array<BuilderResult> builder_results = builder->Build(builder_inputs);

  // Send to runner
  ffi::Array<RunnerInput> runner_inputs;
  for (const BuilderResult& builder_result : builder_results) {
    if (builder_result->error_msg.has_value()) {
      TVM_PY_LOG(ERROR, ctx->logger)
          << "Error in setting reference point: " << builder_result->error_msg.value();
      return;
    }

    runner_inputs.push_back(RunnerInput(builder_result->artifact_path.value(),
                                        ctx->target.value()->kind->name,
                                        ArgInfo::FromEntryFunc(mod, true)));
  }

  ffi::Array<RunnerFuture> futures = runner->Run(runner_inputs);

  // Join runner futures and process results
  ffi::Array<RunnerResult> results;
  results.reserve(futures.size());
  for (RunnerFuture future : futures) {
    auto result = future->Result();
    if (result->error_msg.has_value()) {
      TVM_PY_LOG(ERROR, ctx->logger)
          << "Error in setting reference point: " << result->error_msg.value();
      return;
    }
    results.push_back(result);
  }

  // Process results to extract latency and bandwidth
  // Create a dummy measure candidate for TaskCleanUp
  ffi::Array<MeasureCandidate> measure_candidates;
  measure_candidates.push_back(MeasureCandidate(sch_pp.value(), ArgInfo::FromEntryFunc(mod, true)));
  task->measure_candidates = measure_candidates;
  task->builder_results = builder_results;
  task->runner_futures = futures;

  // Process results to populate task->latency_ms and task->bandwidth_mbps
  TaskCleanUp(task, 0, results);

  if (task->latency_ms.empty() || task->bandwidth_mbps.empty()) {
    TVM_PY_LOG(ERROR, ctx->logger) << "Error in setting reference point: No valid results obtained";
    return;
  }

  double latency_ms = *std::max_element(task->latency_ms.begin(), task->latency_ms.end());
  double bandwidth_mbps =
      *std::max_element(task->bandwidth_mbps.begin(), task->bandwidth_mbps.end());
  this->latency_reference_point = latency_ms;
  this->bandwidth_reference_point = bandwidth_mbps;

  TVM_PY_LOG(INFO, ctx->logger) << "Reference point set: latency=" << latency_ms
                                << "ms, bandwidth=" << bandwidth_mbps << "MB/s";
}

void TaskSchedulerNode::Tune(ffi::Array<TuneContext> ctxs, ffi::Array<FloatImm> task_weights,
                             int max_trials_global, int max_trials_per_task,
                             int num_trials_per_iter, Builder builder, Runner runner,
                             ffi::Array<MeasureCallback> measure_callbacks,
                             ffi::Optional<Database> database,
                             ffi::Optional<CostModel> latency_cost_model,
                             ffi::Optional<CostModel> bandwidth_cost_model) {
  CHECK_EQ(ctxs.size(), task_weights.size()) << "ValueError: `task_weights` must have the same "
                                                "length as `ctxs`";
  int n_tasks = this->remaining_tasks_ = ctxs.size();
  this->measure_callbacks_ = measure_callbacks;
  this->database_ = database;
  this->latency_cost_model_ = latency_cost_model;
  this->bandwidth_cost_model_ = bandwidth_cost_model;
  this->tasks_.clear();
  this->tasks_.reserve(n_tasks);
  for (int i = 0; i < n_tasks; ++i) {
    const TuneContext& ctx = ctxs[i];
    double weight = task_weights[i]->value;
    TVM_PY_LOG(INFO, this->logger) << "Initializing Task #" << i << ": " << ctx->task_name;
    TVM_PY_LOG(INFO, ctx->logger) << "Initializing Task #" << i << ": " << ctx->task_name;
    this->tasks_.push_back(TaskRecord(ctx, weight));
    ffi::Array<tir::Schedule> design_spaces =
        ctx->space_generator.value()->GenerateDesignSpace(ctx->mod.value());
    TVM_PY_LOG(INFO, ctx->logger) << "Total " << design_spaces.size()
                                  << " design space(s) generated";
    for (int i = 0, n = design_spaces.size(); i < n; ++i) {
      tir::Schedule sch = design_spaces[i];
      tir::Trace trace = sch->trace().value();
      trace = trace->Simplified(true);
      TVM_PY_LOG(INFO, ctx->logger) << "Design space #" << i << ":\n"
                                    << sch->mod() << "\n"
                                    << Concat(trace->AsPython(false), "\n");
    }
    ctx->search_strategy.value()->PreTuning(max_trials_per_task, num_trials_per_iter, design_spaces,
                                            database, latency_cost_model, bandwidth_cost_model);
  }

  int num_trials_already = 0;
  for (int task_id; num_trials_already < max_trials_global && (task_id = NextTaskId()) != -1;) {
    TVM_PY_LOG(INFO, this->logger)
        << "TaskScheduler picks Task #" << task_id << ": " << tasks_[task_id]->ctx->task_name;
    TaskRecordNode* task = tasks_[task_id].get();
    ICHECK(!task->is_terminated);
    ICHECK(!task->runner_futures.defined());
    if (static_cast<int>(task->latency_ms.size()) >= max_trials_per_task) {
      TerminateTask(task_id);
      continue;
    }
    if (ffi::Optional<ffi::Array<MeasureCandidate>> candidates = task->measure_candidates =
            task->ctx->search_strategy.value()->GenerateMeasureCandidates()) {
      int num_candidates = candidates.value().size();
      num_trials_already += num_candidates;
      TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to builder";
      SendToBuilder(task, builder);
      TVM_PY_LOG(INFO, this->logger) << "Sending " << num_candidates << " sample(s) to runner";
      SendToRunner(task, runner);
    } else {
      TerminateTask(task_id);
    }
  }
  for (int task_id = 0; task_id < n_tasks; ++task_id) {
    TaskRecordNode* task = this->tasks_[task_id].get();
    if (!task->is_terminated) {
      if (task->runner_futures.defined()) {
        JoinRunningTask(task_id);
      }
      TerminateTask(task_id);
    }
    task->ctx->search_strategy.value()->PostTuning();
  }
}

ffi::Array<RunnerResult> TaskSchedulerNode::JoinRunningTask(int task_id) {
  TaskRecordNode* task = this->tasks_[task_id].get();
  ICHECK(task->runner_futures.defined());
  ffi::Array<RunnerResult> results;
  {
    auto _ = Profiler::TimedScope("JoinRunnerFutures");
    ffi::Array<RunnerFuture> futures = task->runner_futures.value();
    results.reserve(futures.size());
    for (RunnerFuture future : futures) {
      results.push_back(future->Result());
    }
  }
  ICHECK(task->measure_candidates.defined());
  task->ctx->search_strategy.value()->NotifyRunnerResults(task->measure_candidates.value(),
                                                          results);
  ICHECK(task->builder_results.defined());
  ICHECK_EQ(results.size(), task->measure_candidates.value().size());
  ICHECK_EQ(results.size(), task->builder_results.value().size());
  for (const MeasureCallback& callback : this->measure_callbacks_) {
    callback->Apply(ffi::GetRef<TaskScheduler>(this), task_id, task->measure_candidates.value(),
                    task->builder_results.value(), results);
  }
  TaskCleanUp(task, task_id, results);
  TVM_PY_LOG_CLEAR_SCREEN(this->logger);
  TVM_PY_LOG(INFO, this->logger) << "[Updated] Task #" << task_id << ": " << task->ctx->task_name;
  this->PrintTuningStatistics();
  return results;
}

void TaskSchedulerNode::TouchTask(int task_id) {
  TaskRecordNode* task = this->tasks_[task_id].get();
  if (!task->is_terminated && task->runner_futures.defined()) {
    for (const RunnerFuture future : task->runner_futures.value()) {
      if (!future->Done()) {
        return;
      }
    }
    this->JoinRunningTask(task_id);
  }
}

void TaskSchedulerNode::TerminateTask(int task_id) {
  TaskRecordNode* task = this->tasks_[task_id].get();
  ICHECK(!task->is_terminated);
  task->is_terminated = true;
  --this->remaining_tasks_;
  TVM_PY_LOG_CLEAR_SCREEN(this->logger);
  TVM_PY_LOG(INFO, this->logger) << "Task #" << task_id
                                 << " has finished. Remaining task(s): " << this->remaining_tasks_;
  this->PrintTuningStatistics();
}

/*!
 * \brief Calculate hypervolume for 2D Pareto front.
 * Hypervolume measures the area dominated by the Pareto front relative to a reference point.
 * \param latency_values Vector of latency values (to minimize).
 * \param bandwidth_values Vector of bandwidth values (to minimize).
 * \param ref_latency Reference point for latency (should be worse than all points).
 * \param ref_bandwidth Reference point for bandwidth (should be worse than all points).
 * \return The hypervolume value.
 */
static double CalculateHypervolume2D(const std::vector<double>& latency_values,
                                     const std::vector<double>& bandwidth_values,
                                     double ref_latency, double ref_bandwidth) {
  if (latency_values.empty() || bandwidth_values.empty()) {
    LOG(INFO) << "[Hypervolume] Empty input: latency_size=" << latency_values.size()
              << ", bandwidth_size=" << bandwidth_values.size();
    return 0.0;
  }
  ICHECK_EQ(latency_values.size(), bandwidth_values.size());

  int n = latency_values.size();
  LOG(INFO) << "[Hypervolume] Input: n=" << n << ", ref_latency=" << ref_latency
            << ", ref_bandwidth=" << ref_bandwidth;

  // Find Pareto front (rank 1 solutions)
  std::vector<int> ranks = NonDominatedSort(latency_values, bandwidth_values);
  std::vector<std::pair<double, double>> pareto_front;
  for (int i = 0; i < n; ++i) {
    if (ranks[i] == 1) {
      pareto_front.push_back({latency_values[i], bandwidth_values[i]});
    }
  }

  LOG(INFO) << "[Hypervolume] Pareto front size: " << pareto_front.size() << " out of " << n;
  if (pareto_front.empty()) {
    LOG(INFO) << "[Hypervolume] Empty Pareto front, returning 0.0";
    return 0.0;
  }

  // Sort Pareto front by latency (ascending), then by bandwidth (ascending) for ties
  std::sort(pareto_front.begin(), pareto_front.end(),
            [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
              if (a.first != b.first) {
                return a.first < b.first;
              }
              return a.second < b.second;
            });

  LOG(INFO) << "[Hypervolume] Pareto front points (latency, bandwidth):";
  for (size_t i = 0; i < pareto_front.size(); ++i) {
    LOG(INFO) << "  [" << i << "] (" << pareto_front[i].first << ", " << pareto_front[i].second
              << ")";
  }

  // Calculate hypervolume using the "sweep" algorithm for 2D minimization
  // Hypervolume is the area of the region dominated by the Pareto front
  // Algorithm: sweep from reference point, accumulating rectangle areas
  double hypervolume = 0.0;
  double prev_latency = ref_latency;
  double worst_bandwidth = ref_bandwidth;  // Track worst bandwidth seen so far

  LOG(INFO) << "[Hypervolume] Starting calculation: prev_latency=" << prev_latency
            << ", worst_bandwidth=" << worst_bandwidth;

  int points_processed = 0;
  int points_skipped = 0;
  for (const auto& point : pareto_front) {
    double latency = point.first;
    double bandwidth = point.second;

    // Ensure point is within reference bounds
    if (latency > ref_latency || bandwidth > ref_bandwidth) {
      LOG(INFO) << "[Hypervolume] Skipping point (" << latency << ", " << bandwidth
                << ") - outside reference bounds (ref: " << ref_latency << ", " << ref_bandwidth
                << ")";
      points_skipped++;
      continue;
    }

    // Calculate rectangle: from (prev_latency, worst_bandwidth) to (latency, bandwidth)
    // This represents the new area dominated by this point
    double width = prev_latency - latency;
    double height = worst_bandwidth - bandwidth;

    LOG(INFO) << "[Hypervolume] Processing point (" << latency << ", " << bandwidth
              << "): width=" << width << ", height=" << height;

    if (width > 0 && height > 0) {
      double area = width * height;
      hypervolume += area;
      LOG(INFO) << "[Hypervolume] Added area: " << area << ", total: " << hypervolume;
      points_processed++;
    } else {
      LOG(INFO) << "[Hypervolume] Skipping point - invalid dimensions (width=" << width
                << ", height=" << height << ")";
    }

    // Update for next iteration
    prev_latency = latency;
    worst_bandwidth = std::min(worst_bandwidth, bandwidth);
  }

  LOG(INFO) << "[Hypervolume] Final result: " << hypervolume << " (processed: " << points_processed
            << ", skipped: " << points_skipped << ")";
  return hypervolume;
}

void TaskSchedulerNode::PrintTuningStatistics() {
  std::ostringstream os;
  int n_tasks = this->tasks_.size();
  int total_trials = 0;
  double total_latency = 0.0;
  double total_hypervolume = 0.0;

  // Check if reference points are set
  bool has_reference_point =
      this->latency_reference_point.has_value() && this->bandwidth_reference_point.has_value();

  support::TablePrinter p;
  auto header_row = p.Row();
  header_row << "ID"
             << "Name"
             << "FLOP"
             << "Weight"
             << "Max Speed (GFLOPS)"
             << "Min Latency (us)"
             << "Weighted Latency (us)"
             << "Min Bandwidth (MB/s)";
  if (has_reference_point) {
    header_row << "Hypervolume";
  }
  header_row << "Trials"
             << "Done";
  p.Separator();
  for (int i = 0; i < n_tasks; ++i) {
    const TaskRecordNode* task = this->tasks_[i].get();
    auto row = p.Row();
    int trials = task->latency_ms.size();
    row << /*id=*/i << /*name=*/task->ctx->task_name.value()  //
        << /*flops=*/static_cast<int64_t>(task->flop)
        << /*weight=*/static_cast<int>(task->task_weight);

    double latency_ms = 1e9;
    if (!task->latency_ms.empty()) {
      latency_ms = *std::min_element(task->latency_ms.begin(), task->latency_ms.end());
    }
    if (latency_ms >= 1e9) {
      row << /*speed=*/"N/A" << /*latency=*/"N/A" << /*weighted_latency=*/"N/A";
    } else {
      latency_ms *= 1000.0;
      double speed = task->flop / latency_ms / 1000.0;
      double weighted_latency = latency_ms * task->task_weight;
      row << /*speed=*/speed << /*latency=*/latency_ms << /*weighted_latency=*/weighted_latency;
      total_latency += weighted_latency;
      total_trials += trials;
    }

    double bandwidth_mbps = 1e9;
    if (!task->bandwidth_mbps.empty()) {
      bandwidth_mbps = *std::min_element(task->bandwidth_mbps.begin(), task->bandwidth_mbps.end());
    }
    if (bandwidth_mbps >= 1e9) {
      row << /*bandwidth*/ "N/A";
    } else {
      row << bandwidth_mbps;
    }

    // Calculate hypervolume only if reference point is set
    if (has_reference_point) {
      double hypervolume = 0.0;
      if (!task->latency_ms.empty() && !task->bandwidth_mbps.empty() &&
          task->latency_ms.size() == task->bandwidth_mbps.size()) {
        // Convert latency_ms to seconds for consistency
        std::vector<double> latency_sec(task->latency_ms.size());
        for (size_t j = 0; j < task->latency_ms.size(); ++j) {
          latency_sec[j] = task->latency_ms[j] / 1000.0;  // Convert ms to seconds
        }

        // Use the reference point set by SetReferencePoint
        // Reference point is stored in milliseconds, convert to seconds
        double ref_latency_sec = this->latency_reference_point.value() / 1000.0;
        double ref_bandwidth_mbps = this->bandwidth_reference_point.value();

        // Calculate hypervolume
        hypervolume = CalculateHypervolume2D(latency_sec, task->bandwidth_mbps, ref_latency_sec,
                                             ref_bandwidth_mbps);
        total_hypervolume += hypervolume;
      }

      row << hypervolume;
    }

    row << trials;
    if (task->is_terminated) {
      row << "Y";
    } else {
      row << "";
    }
  }
  p.Separator();

  os << "\nTotal trials: " << total_trials  //
     << "\nTotal latency (us): " << total_latency;
  if (has_reference_point) {
    os << "\nTotal hypervolume: " << total_hypervolume;
    os << "\nReference point: Latency=" << this->latency_reference_point.value() << "ms, Bandwidth=" << this->bandwidth_reference_point.value() << "MB/s";
  }
  os << "\n";

  if (using_ipython()) {
    print_interactive_table(p.AsStr());
    std::cout << os.str() << std::endl << std::flush;
    TVM_PY_LOG(DEBUG, this->logger) << "\n" << p.AsStr() << os.str();
  } else {
    TVM_PY_LOG(INFO, this->logger) << "\n" << p.AsStr() << os.str();
  }
}

TaskScheduler TaskScheduler::PyTaskScheduler(
    ffi::Function logger, PyTaskSchedulerNode::FNextTaskId f_next_task_id,
    PyTaskSchedulerNode::FJoinRunningTask f_join_running_task, PyTaskSchedulerNode::FTune f_tune) {
  CHECK(f_next_task_id != nullptr) << "ValueError: next_task_id is not defined";
  ObjectPtr<PyTaskSchedulerNode> n = ffi::make_object<PyTaskSchedulerNode>();
  n->logger = logger;
  n->f_next_task_id = f_next_task_id;
  n->f_join_running_task = f_join_running_task;
  n->f_tune = f_tune;
  return TaskScheduler(n);
}

int PyTaskSchedulerNode::NextTaskId() {
  CHECK(f_next_task_id != nullptr) << "PyTaskScheduler's NextTaskId method not implemented!";
  return f_next_task_id();
}

ffi::Array<RunnerResult> PyTaskSchedulerNode::JoinRunningTask(int task_id) {
  if (f_join_running_task == nullptr) {
    return TaskSchedulerNode::JoinRunningTask(task_id);
  } else {
    return f_join_running_task(task_id);
  }
}

void PyTaskSchedulerNode::SetReferencePoint(TuneContext task, Builder builder, Runner runner) {
  if (f_set_reference_point == nullptr) {
    TaskSchedulerNode::SetReferencePoint(task, builder, runner);
  } else {
    f_set_reference_point(task, builder, runner);
  }
}

void PyTaskSchedulerNode::Tune(ffi::Array<TuneContext> tasks, ffi::Array<FloatImm> task_weights,
                               int max_trials_global, int max_trials_per_task,
                               int num_trials_per_iter, Builder builder, Runner runner,
                               ffi::Array<MeasureCallback> measure_callbacks,
                               ffi::Optional<Database> database,
                               ffi::Optional<CostModel> latency_cost_model,
                               ffi::Optional<CostModel> bandwidth_cost_model) {
  if (f_tune == nullptr) {
    TaskSchedulerNode::Tune(tasks, task_weights, max_trials_global, max_trials_per_task,
                            num_trials_per_iter, builder, runner, measure_callbacks, database,
                            latency_cost_model, bandwidth_cost_model);
  } else {
    f_tune(tasks, task_weights, max_trials_global, max_trials_per_task, num_trials_per_iter,
           builder, runner, measure_callbacks, database, latency_cost_model, bandwidth_cost_model);
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("meta_schedule.TaskSchedulerPyTaskScheduler", TaskScheduler::PyTaskScheduler)
      .def_method("meta_schedule.TaskSchedulerSetReferencePoint",
                  &TaskSchedulerNode::SetReferencePoint)
      .def_method("meta_schedule.TaskSchedulerTune", &TaskSchedulerNode::Tune)
      .def_method("meta_schedule.TaskSchedulerJoinRunningTask", &TaskSchedulerNode::JoinRunningTask)
      .def_method("meta_schedule.TaskSchedulerNextTaskId", &TaskSchedulerNode::NextTaskId)
      .def_method("meta_schedule.TaskSchedulerTerminateTask", &TaskSchedulerNode::TerminateTask)
      .def_method("meta_schedule.TaskSchedulerTouchTask", &TaskSchedulerNode::TouchTask)
      .def_method("meta_schedule.TaskSchedulerPrintTuningStatistics",
                  &TaskSchedulerNode::PrintTuningStatistics);
}

}  // namespace meta_schedule
}  // namespace tvm
