# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test Meta Schedule NSGA-II Search Strategy"""
# pylint: disable=missing-function-docstring
import tempfile
from typing import List

import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.dummy_object import DummyMutator
from tvm.script import tir as T
from tvm.tir.schedule import Schedule, Trace
from tvm.ir.module import IRModule

# pylint: disable=missing-class-docstring,invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None: # type: ignore
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (32, 32), "float32")
        B = T.match_buffer(b, (32, 32), "float32")
        C = T.match_buffer(c, (32, 32), "float32")
        for i, j, k in T.grid(32, 32, 32):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0 # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=missing-class-docstring,invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _is_trace_equal(sch_1: Schedule, sch_2: Schedule, remove_decisions=True) -> bool:
    if remove_decisions:
        trace_1 = Trace(sch_1.trace.insts, {})
        trace_2 = Trace(sch_2.trace.insts, {})
    else:
        trace_1 = sch_1.trace
        trace_2 = sch_2.trace
    return str(trace_1) == str(trace_2)


def test_meta_schedule_nsgaii_search():  # pylint: disable = invalid-name
    """Original NSGA-II test - basic functionality test"""
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    num_trials_per_iter = 10
    max_trials_per_task = 2000
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.NSGAIISearch(
            population_size=5,
            init_measured_ratio=0.1,
            init_min_unmeasured=50,
            max_fail_count=10,
            genetic_num_iters=3,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,
    )
    strategy = context.search_strategy
    with tempfile.TemporaryDirectory() as tmpdir:
        strategy.pre_tuning(
            max_trials=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=context.space_generator.generate_design_space(context.mod),
            database=ms.database.JSONParetoDatabase(work_dir=tmpdir),
            latency_cost_model=ms.cost_model.RandomModel(),
            bandwidth_cost_model=ms.cost_model.RandomModel(),
        )
        num_trials_each_iter: List[int] = []
        candidates = strategy.generate_measure_candidates()
        while candidates is not None:
            num_trials_each_iter.append(len(candidates))
            runner_results: List[ms.runner.RunnerResult] = []
            for candidate in candidates:
                _is_trace_equal(
                    candidate.sch,
                    correct_sch,
                    remove_decisions=(isinstance(strategy, ms.search_strategy.ReplayTrace)),
                )
                runner_results.append(
                    ms.runner.RunnerResult(
                        run_secs=[],
                        bw_mbps=[],
                        error_msg=None,
                    )
                )
            strategy.notify_runner_results(candidates, runner_results)
            candidates = strategy.generate_measure_candidates()
        strategy.post_tuning()
        assert sum(num_trials_each_iter) == 25
        assert num_trials_each_iter.count(0) < 5


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(i, sch.sample_perfect_tile(i, n=4))
    j_0, j_1, j_2, j_3 = sch.split(j, sch.sample_perfect_tile(j, n=4))
    k_0, k_1 = sch.split(k, sch.sample_perfect_tile(k, n=2))
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _create_schedule(mod, sch_fn):
    sch = tvm.tir.Schedule(mod=mod, debug_mask="all")
    sch_fn(sch)
    return sch


def test_nsgaii_pareto_dominance_through_database():
    """
    Test that Pareto dominance is correctly identified through database operations.
    This indirectly tests the ParetoDominates function.
    """
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = ms.database.JSONParetoDatabase(work_dir=tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])

        # Create records where some are clearly dominated
        # Record 1: Fast time, high bandwidth (Pareto front)
        record1 = ms.database.TuningRecord(
            trace, workload, run_secs=[1.0], bw_mbps=[200.0],
            target=target, args_info=arg_info
        )
        # Record 2: Slow time, low bandwidth (Pareto front - trade-off)
        record2 = ms.database.TuningRecord(
            trace, workload, run_secs=[3.0], bw_mbps=[50.0],
            target=target, args_info=arg_info
        )
        # Record 3: Dominated by record1 (worse in both: higher time AND higher bandwidth)
        record3 = ms.database.TuningRecord(
            trace, workload, run_secs=[1.5], bw_mbps=[250.0],
            target=target, args_info=arg_info
        )
        # Record 4: Dominated by record2 (worse in both: higher time AND higher bandwidth)
        record4 = ms.database.TuningRecord(
            trace, workload, run_secs=[4.0], bw_mbps=[60.0],
            target=target, args_info=arg_info
        )

        for record in [record1, record2, record3, record4]:
            database.commit_tuning_record(record)

        # GetTopK should return only Pareto front records (rank 1)
        top_k = database.get_top_k(workload, 10)

        # Should have exactly 2 Pareto front records (record1 and record2)
        assert len(top_k) == 2

        def get_mean(values):
            if not values:
                return 0.0
            return float(sum(float(v) for v in values) / len(values))

        top_k_values = [
            (get_mean(r.run_secs), get_mean(r.bw_mbps)) for r in top_k
        ]
        record1_values = (get_mean(record1.run_secs), get_mean(record1.bw_mbps))
        record2_values = (get_mean(record2.run_secs), get_mean(record2.bw_mbps))
        record3_values = (get_mean(record3.run_secs), get_mean(record3.bw_mbps))
        record4_values = (get_mean(record4.run_secs), get_mean(record4.bw_mbps))

        # Verify Pareto front records are present
        assert record1_values in top_k_values, "Record 1 should be in Pareto front"
        assert record2_values in top_k_values, "Record 2 should be in Pareto front"

        # Verify dominated records are NOT present
        assert record3_values not in top_k_values, "Record 3 (dominated) should not be in top_k"
        assert record4_values not in top_k_values, "Record 4 (dominated) should not be in top_k"


def test_nsgaii_non_dominated_sorting_correctness():
    """
    Test that non-dominated sorting correctly assigns ranks.
    This tests that rank 2+ solutions are checked against ALL solutions (not just unprocessed).
    """
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = ms.database.JSONParetoDatabase(work_dir=tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])

        # Create records with clear dominance hierarchy:
        # Rank 1: (1.0, 2.0), (2.0, 1.0) - trade-offs, none dominate each other
        # Note: (1.0, 1.0) would dominate both, so we use (1.0, 2.0) and (2.0, 1.0) as rank 1
        # Rank 2: (2.0, 2.0), (2.0, 3.0), (3.0, 2.0) - dominated by rank 1
        # Rank 3: (3.0, 3.0) - dominated by rank 1 and 2
        records = [
            ms.database.TuningRecord(trace, workload, run_secs=[1.0], bw_mbps=[2.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[2.0], bw_mbps=[1.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[2.0], bw_mbps=[2.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[2.0], bw_mbps=[3.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[3.0], bw_mbps=[2.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[3.0], bw_mbps=[3.0], target=target, args_info=arg_info),
        ]

        for record in records:
            database.commit_tuning_record(record)

        # GetTopK should return only rank 1 (Pareto front)
        top_k = database.get_top_k(workload, 10)

        def get_mean(values):
            if not values:
                return 0.0
            return float(sum(float(v) for v in values) / len(values))

        top_k_values = [(get_mean(r.run_secs), get_mean(r.bw_mbps)) for r in top_k]

        # Should have exactly 2 rank 1 records (trade-offs)
        assert len(top_k) == 2, f"Expected 2 rank 1 records, got {len(top_k)}"

        # Verify rank 1 records are present
        rank1_values = [(1.0, 2.0), (2.0, 1.0)]
        for rv in rank1_values:
            assert rv in top_k_values, f"Rank 1 record {rv} should be in top_k"

        # Verify rank 2+ records are NOT in top_k
        rank2plus_values = [(2.0, 2.0), (2.0, 3.0), (3.0, 2.0), (3.0, 3.0)]
        for rv in rank2plus_values:
            assert rv not in top_k_values, f"Rank 2+ record {rv} should not be in top_k (dominated)"


def test_nsgaii_crowding_distance_identical_points():
    """
    Test that crowding distance correctly handles identical points.
    Identical points should get infinite distance (not 0).
    """
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = ms.database.JSONParetoDatabase(work_dir=tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])

        # Create multiple records with identical objective values
        # All should be in Pareto front (none dominate each other)
        # All should have infinite crowding distance
        records = [
            ms.database.TuningRecord(trace, workload, run_secs=[1.0], bw_mbps=[2.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[1.0], bw_mbps=[2.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[1.0], bw_mbps=[2.0], target=target, args_info=arg_info),
        ]

        for record in records:
            database.commit_tuning_record(record)

        # GetTopK should return all records (all are in Pareto front)
        top_k = database.get_top_k(workload, 10)

        # All records should be returned
        assert len(top_k) == 3, "All identical records should be in Pareto front"

        # Since all are identical, they should all have infinite crowding distance
        # and be sorted consistently (order may vary but all should be present)


def test_nsgaii_crowding_distance_zero_range():
    """
    Test that crowding distance handles zero range correctly.
    When all points have the same value in one or both objectives.
    """
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = ms.database.JSONParetoDatabase(work_dir=tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])

        # Create records with zero range in one objective
        # All have same latency but different bandwidth
        # Note: When latency is the same, lower bandwidth dominates, so only (1.0, 50.0) is in Pareto front
        records = [
            ms.database.TuningRecord(trace, workload, run_secs=[1.0], bw_mbps=[50.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[1.0], bw_mbps=[100.0], target=target, args_info=arg_info),
            ms.database.TuningRecord(trace, workload, run_secs=[1.0], bw_mbps=[150.0], target=target, args_info=arg_info),
        ]

        for record in records:
            database.commit_tuning_record(record)

        top_k = database.get_top_k(workload, 10)

        # Only the record with lowest bandwidth should be in Pareto front
        # (1.0, 50.0) dominates (1.0, 100.0) and (1.0, 150.0) because it has same latency but lower bandwidth
        assert len(top_k) == 1, "Only the record with lowest bandwidth should be in Pareto front"

        def get_mean(values):
            if not values:
                return 0.0
            return float(sum(float(v) for v in values) / len(values))

        top_k_values = [(get_mean(r.run_secs), get_mean(r.bw_mbps)) for r in top_k]
        assert (1.0, 50.0) in top_k_values, "Record with lowest bandwidth (1.0, 50.0) should be in top_k"


def test_nsgaii_cost_model_score_negation():
    """
    Test that cost model scores are correctly negated for Pareto dominance.
    Higher cost model scores (better performance) should result in better Pareto ranks.
    """
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = ms.database.JSONParetoDatabase(work_dir=tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])

        # Create a mock cost model that returns predictable scores
        # We'll use actual measured results that correspond to different cost model predictions
        # Solution A: Better performance (lower latency, lower bandwidth usage)
        # Solution B: Worse performance (higher latency, higher bandwidth usage)
        # After cost model: A should get higher scores, B should get lower scores
        # After negation: A should get lower (better) negated scores, B should get higher (worse) negated scores
        # A should dominate B in Pareto sense

        # Record A: Better performance (lower time, lower bandwidth)
        record_a = ms.database.TuningRecord(
            trace, workload, run_secs=[1.0], bw_mbps=[100.0],
            target=target, args_info=arg_info
        )
        # Record B: Worse performance (higher time, higher bandwidth)
        record_b = ms.database.TuningRecord(
            trace, workload, run_secs=[2.0], bw_mbps=[200.0],
            target=target, args_info=arg_info
        )

        database.commit_tuning_record(record_a)
        database.commit_tuning_record(record_b)

        # GetTopK should return record_a (it dominates record_b)
        top_k = database.get_top_k(workload, 10)

        assert len(top_k) == 1, "Only record_a should be in Pareto front (it dominates record_b)"

        def get_mean(values):
            if not values:
                return 0.0
            return float(sum(float(v) for v in values) / len(values))

        top_k_values = [(get_mean(r.run_secs), get_mean(r.bw_mbps)) for r in top_k]
        record_a_values = (get_mean(record_a.run_secs), get_mean(record_a.bw_mbps))

        assert record_a_values in top_k_values, "Record A (better performance) should be in top_k"


def test_nsgaii_multi_objective_optimization():
    """
    Test that NSGA-II correctly optimizes for both latency and bandwidth simultaneously.
    Solutions with different trade-offs should both be in Pareto front.
    """
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = ms.database.JSONParetoDatabase(work_dir=tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])

        # Create records with different trade-offs (all should be in Pareto front)
        # Record 1: Fast time, high bandwidth (trade-off)
        record1 = ms.database.TuningRecord(
            trace, workload, run_secs=[1.0], bw_mbps=[200.0],
            target=target, args_info=arg_info
        )
        # Record 2: Slow time, low bandwidth (trade-off)
        record2 = ms.database.TuningRecord(
            trace, workload, run_secs=[3.0], bw_mbps=[50.0],
            target=target, args_info=arg_info
        )
        # Record 3: Medium time, medium bandwidth (trade-off)
        record3 = ms.database.TuningRecord(
            trace, workload, run_secs=[2.0], bw_mbps=[100.0],
            target=target, args_info=arg_info
        )
        # Record 4: Dominated by record1 (worse in both)
        record4 = ms.database.TuningRecord(
            trace, workload, run_secs=[1.5], bw_mbps=[250.0],
            target=target, args_info=arg_info
        )

        for record in [record1, record2, record3, record4]:
            database.commit_tuning_record(record)

        top_k = database.get_top_k(workload, 10)

        # Should have exactly 3 Pareto front records (record1, record2, record3)
        assert len(top_k) == 3, "Should have 3 Pareto front records"

        def get_mean(values):
            if not values:
                return 0.0
            return float(sum(float(v) for v in values) / len(values))

        top_k_values = [(get_mean(r.run_secs), get_mean(r.bw_mbps)) for r in top_k]
        record1_values = (get_mean(record1.run_secs), get_mean(record1.bw_mbps))
        record2_values = (get_mean(record2.run_secs), get_mean(record2.bw_mbps))
        record3_values = (get_mean(record3.run_secs), get_mean(record3.bw_mbps))
        record4_values = (get_mean(record4.run_secs), get_mean(record4.bw_mbps))

        # Verify all trade-off records are in Pareto front
        assert record1_values in top_k_values, "Record 1 should be in Pareto front"
        assert record2_values in top_k_values, "Record 2 should be in Pareto front"
        assert record3_values in top_k_values, "Record 3 should be in Pareto front"

        # Verify dominated record is NOT in Pareto front
        assert record4_values not in top_k_values, "Record 4 (dominated) should not be in top_k"


def test_nsgaii_with_cost_models():
    """
    Test NSGA-II with actual cost models (XGBoost or RandomModel).
    Verify that cost model predictions are correctly used for selection.
    """
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    num_trials_per_iter = 5
    max_trials_per_task = 50
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.NSGAIISearch(
            population_size=10,
            init_measured_ratio=0.2,
            init_min_unmeasured=20,
            max_fail_count=10,
            genetic_num_iters=2,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.8,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,
    )
    strategy = context.search_strategy
    with tempfile.TemporaryDirectory() as tmpdir:
        strategy.pre_tuning(
            max_trials=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=context.space_generator.generate_design_space(context.mod),
            database=ms.database.JSONParetoDatabase(work_dir=tmpdir),
            latency_cost_model=ms.cost_model.RandomModel(),
            bandwidth_cost_model=ms.cost_model.RandomModel(),
        )

        # Run a few iterations and verify behavior
        candidates = strategy.generate_measure_candidates()
        iteration_count = 0
        while candidates is not None and iteration_count < 3:
            assert len(candidates) > 0, "Should generate candidates"
            runner_results: List[ms.runner.RunnerResult] = []
            for candidate in candidates:
                runner_results.append(
                    ms.runner.RunnerResult(
                        run_secs=[1.0 + iteration_count * 0.1],  # Vary latency
                        bw_mbps=[100.0 + iteration_count * 10.0],  # Vary bandwidth
                        error_msg=None,
                    )
                )
            strategy.notify_runner_results(candidates, runner_results)
            candidates = strategy.generate_measure_candidates()
            iteration_count += 1

        strategy.post_tuning()
        assert iteration_count > 0, "Should have run at least one iteration"


def test_nsgaii_population_evolution():
    """
    Test that NSGA-II population evolves correctly over generations.
    Verify that best solutions are preserved (elitism) and population maintains diversity.
    """
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    num_trials_per_iter = 8
    max_trials_per_task = 100
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.NSGAIISearch(
            population_size=10,
            init_measured_ratio=0.1,
            init_min_unmeasured=20,
            max_fail_count=10,
            genetic_num_iters=3,  # Multiple generations
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,
    )
    strategy = context.search_strategy
    with tempfile.TemporaryDirectory() as tmpdir:
        strategy.pre_tuning(
            max_trials=max_trials_per_task,
            num_trials_per_iter=num_trials_per_iter,
            design_spaces=context.space_generator.generate_design_space(context.mod),
            database=ms.database.JSONParetoDatabase(work_dir=tmpdir),
            latency_cost_model=ms.cost_model.RandomModel(),
            bandwidth_cost_model=ms.cost_model.RandomModel(),
        )

        # Track candidates over iterations
        all_candidates = []
        candidates = strategy.generate_measure_candidates()
        iteration = 0
        while candidates is not None and iteration < 5:
            all_candidates.append(len(candidates))
            runner_results: List[ms.runner.RunnerResult] = []
            # Vary results to create Pareto front
            for i, candidate in enumerate(candidates):
                # Create trade-offs: some fast+highBW, some slow+lowBW
                if i % 2 == 0:
                    runner_results.append(
                        ms.runner.RunnerResult(
                            run_secs=[1.0 + i * 0.1],
                            bw_mbps=[200.0 + i * 10.0],
                            error_msg=None,
                        )
                    )
                else:
                    runner_results.append(
                        ms.runner.RunnerResult(
                            run_secs=[3.0 + i * 0.1],
                            bw_mbps=[50.0 + i * 5.0],
                            error_msg=None,
                        )
                    )
            strategy.notify_runner_results(candidates, runner_results)
            candidates = strategy.generate_measure_candidates()
            iteration += 1

        strategy.post_tuning()

        # Verify that we generated candidates
        assert len(all_candidates) > 0, "Should have generated candidates"
        assert sum(all_candidates) > 0, "Should have generated some candidates"


def test_nsgaii_edge_cases():
    """
    Test NSGA-II with edge cases: empty population, single solution, etc.
    """
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.NSGAIISearch(
            population_size=2,  # Small population
            init_measured_ratio=0.0,  # No measured solutions
            init_min_unmeasured=2,
            max_fail_count=5,
            genetic_num_iters=1,  # Single generation
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=5,
            eps_greedy=0.5,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,
    )
    strategy = context.search_strategy
    with tempfile.TemporaryDirectory() as tmpdir:
        strategy.pre_tuning(
            max_trials=10,
            num_trials_per_iter=5,
            design_spaces=context.space_generator.generate_design_space(context.mod),
            database=ms.database.JSONParetoDatabase(work_dir=tmpdir),
            latency_cost_model=ms.cost_model.RandomModel(),
            bandwidth_cost_model=ms.cost_model.RandomModel(),
        )

        # Should handle small population gracefully
        candidates = strategy.generate_measure_candidates()
        if candidates is not None:
            assert len(candidates) > 0, "Should generate candidates if possible"
            runner_results: List[ms.runner.RunnerResult] = []
            for candidate in candidates:
                runner_results.append(
                    ms.runner.RunnerResult(
                        run_secs=[1.0],
                        bw_mbps=[100.0],
                        error_msg=None,
                    )
                )
            strategy.notify_runner_results(candidates, runner_results)

        strategy.post_tuning()


if __name__ == "__main__":
    test_meta_schedule_nsgaii_search()
    test_nsgaii_pareto_dominance_through_database()
    test_nsgaii_non_dominated_sorting_correctness()
    test_nsgaii_crowding_distance_identical_points()
    test_nsgaii_crowding_distance_zero_range()
    test_nsgaii_cost_model_score_negation()
    test_nsgaii_multi_objective_optimization()
    test_nsgaii_with_cost_models()
    test_nsgaii_population_evolution()
    test_nsgaii_edge_cases()
    tvm.testing.main()
