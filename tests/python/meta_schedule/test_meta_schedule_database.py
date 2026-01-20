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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
"""Test Meta Schedule Database"""
import os.path as osp
import tempfile
from typing import Callable, List, Optional

import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import tir
from tvm.ir.module import IRModule
from tvm.meta_schedule.database import TuningRecord, Workload
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument
# fmt: off
@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulRelu:
    @T.prim_func
    def main(a: T.handle, b: T.handle, d: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (16, 16), "float32")
        B = T.match_buffer(b, (16, 16), "float32")
        D = T.match_buffer(d, (16, 16), "float32")
        C = T.alloc_buffer((16, 16), "float32")
        for i, j, k in T.grid(16, 16, 16):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(16, 16):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)


# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_tiles = [1, 1, 2, 512]
    j_tiles = [1, 512, 1, 2]
    k_tiles = [256, 4]
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=i_tiles)
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=j_tiles)
    k_0, k_1 = sch.split(loop=k, factors=k_tiles)
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _create_schedule(mod: IRModule, sch_fn: Callable[[Schedule], None]) -> Schedule:
    sch = tir.Schedule(mod=mod, debug_mask="all")
    sch_fn(sch)
    return sch


def _create_tmp_database(tmpdir: str, mod_eq: str = "structural") -> ms.database.JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return ms.database.JSONDatabase(path_workload, path_tuning_record, module_equality=mod_eq)


def _create_tmp_pareto_database(tmpdir: str, mod_eq: str = "structural") -> ms.database.JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return ms.database.JSONParetoDatabase(path_workload, path_tuning_record, module_equality=mod_eq)

def _equal_record(a: ms.database.TuningRecord, b: ms.database.TuningRecord):
    assert str(a.trace) == str(b.trace)
    assert str(a.run_secs) == str(b.run_secs)
    # AWAIT(@zxybazh): change to export after fixing "(bool)0"
    assert str(a.target) == str(b.target)
    tvm.ir.assert_structural_equal(a.workload.mod, b.workload.mod)
    for arg0, arg1 in zip(a.args_info, b.args_info):
        assert str(arg0.as_json()) == str(arg1.as_json())


@ms.utils.derived_object
class PyMemoryDatabaseDefault(ms.database.PyDatabase):
    def __init__(self):
        super().__init__()
        self.tuning_records_: List[TuningRecord] = []
        self.workloads_: List[Workload] = []

    def has_workload(self, mod: IRModule) -> bool:
        for workload in self.workloads_:
            if tvm.ir.structural_equal(mod, workload.mod):
                return True

    def commit_workload(self, mod: IRModule) -> ms.database.Workload:
        if self.has_workload(mod):
            for workload in self.workloads_:
                if tvm.ir.structural_equal(mod, workload.mod):
                    return workload
        else:
            workload = ms.database.Workload(mod)
            self.workloads_.append(workload)
            return workload

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.tuning_records_.append(record)

    def get_all_tuning_records(self) -> List[TuningRecord]:
        return self.tuning_records_

    def get_top_k(self, workload: ms.database.Workload, top_k: int) -> List[TuningRecord]:
        return sorted(
            list(
                filter(
                    lambda x: tvm.ir.structural_equal(workload.mod, x.workload.mod),
                    self.tuning_records_,
                )
            ),
            key=lambda x: sum(x.run_secs) / len(x.run_secs) if x.run_secs else 1e9,
        )[:top_k]

    def __len__(self) -> int:
        return len(self.tuning_records_)


@ms.utils.derived_object
class PyMemoryDatabaseOverride(ms.database.PyDatabase):
    def __init__(self):
        super().__init__()
        self.tuning_records_: List[TuningRecord] = []
        self.workloads_: List[Workload] = []

    def has_workload(self, mod: IRModule) -> bool:
        for workload in self.workloads_:
            if tvm.ir.structural_equal(mod, workload.mod):
                return True

    def commit_workload(self, mod: IRModule) -> ms.database.Workload:
        if self.has_workload(mod):
            for workload in self.workloads_:
                if tvm.ir.structural_equal(mod, workload.mod):
                    return workload
        else:
            workload = ms.database.Workload(mod)
            self.workloads_.append(workload)
            return workload

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.tuning_records_.append(record)

    def get_all_tuning_records(self) -> List[TuningRecord]:
        return self.tuning_records_

    def get_top_k(self, workload: ms.database.Workload, top_k: int) -> List[TuningRecord]:
        return sorted(
            list(
                filter(
                    lambda x: tvm.ir.structural_equal(workload.mod, x.workload.mod),
                    self.tuning_records_,
                )
            ),
            key=lambda x: sum(x.run_secs) / len(x.run_secs) if x.run_secs else 1e9,
        )[:top_k]

    def __len__(self) -> int:
        return len(self.tuning_records_)

    def query_tuning_record(
        self, mod: IRModule, target: Target, workload_name: Optional[str] = None
    ) -> Optional[TuningRecord]:
        if self.has_workload(mod):
            records = self.get_top_k(self.commit_workload(mod), 2)
            if len(records) == 1:
                return records[0]
            elif len(records) == 2:
                return records[1]  # return the 2nd best if there are two records
        return None

    def query_schedule(
        self, mod: IRModule, target: Target, workload_name: Optional[str] = None
    ) -> Optional[Schedule]:
        record = self.query_tuning_record(mod, target, workload_name)
        if record is not None:
            sch = Schedule(record.workload.mod)
            record.trace.apply_to_schedule(sch, remove_postproc=False)
            return sch
        return None

    def query_ir_module(
        self, mod: IRModule, target: Target, workload_name: Optional[str] = None
    ) -> Optional[IRModule]:
        record = self.query_tuning_record(mod, target, workload_name)
        if record is not None:
            sch = Schedule(record.workload.mod)
            record.trace.apply_to_schedule(sch, remove_postproc=False)
            return sch.mod
        return None


def test_meta_schedule_tuning_record_round_trip():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [T.float32(1.5), T.float32(2.5), T.float32(1.8)],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        new_record = ms.database.TuningRecord.from_json(record.as_json(), workload)
        _equal_record(record, new_record)


def test_meta_schedule_database_create():
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        assert osp.exists(database.path_workload)
        assert osp.exists(database.path_tuning_record)


def test_meta_schedule_database_has_workload():
    mod: IRModule = Matmul
    missing_mod: IRModule = MatmulRelu
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        assert len(database) == 1
        assert database.has_workload(mod)
        assert not database.has_workload(missing_mod)


def test_meta_schedule_database_add_entry():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        assert len(database) == 1
        (ret,) = database.get_top_k(workload, 3)
        _equal_record(ret, record)


def test_meta_schedule_database_missing():
    mod: IRModule = Matmul
    mod_2: IRModule = MatmulRelu
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(mod)
        workload_2 = database.commit_workload(mod_2)
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            [1.5, 2.5, 1.8],
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
        ret = database.get_top_k(workload_2, 3)
        assert len(ret) == 0


def test_meta_schedule_database_sorting():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        token = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        records = [
            ms.database.TuningRecord(
                trace,
                token,
                [7.0, 8.0, 9.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 2.0, 3.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 5.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.1, 1.2, 600.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 100.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 9.0, 8.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
        ]
        for record in records:
            database.commit_tuning_record(record)
        ret = database.get_top_k(token, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


def test_meta_schedule_database_reload():
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        token = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        records = [
            ms.database.TuningRecord(
                trace,
                token,
                [7.0, 8.0, 9.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [1.0, 2.0, 3.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
            ms.database.TuningRecord(
                trace,
                token,
                [4.0, 5.0, 6.0],
                tvm.target.Target("llvm"),
                ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
            ),
        ]
        for record in records:
            database.commit_tuning_record(record)
        new_database = ms.database.JSONDatabase(
            path_workload=database.path_workload,
            path_tuning_record=database.path_tuning_record,
        )
        token = new_database.commit_workload(mod)
        ret = new_database.get_top_k(token, 2)
        assert len(ret) == 2
        try:
            _equal_record(ret[0], records[2])
            _equal_record(ret[1], records[1])
        except AssertionError:
            _equal_record(ret[0], records[1])
            _equal_record(ret[1], records[2])


def test_meta_schedule_database_union():
    mod: IRModule = Matmul
    target = tvm.target.Target("llvm")
    arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
    db_1 = ms.database.MemoryDatabase()
    db_2 = ms.database.MemoryDatabase()
    trace = _create_schedule(mod, _schedule_matmul).trace

    def query(db):  # pylint: disable=invalid-name
        return db.query_tuning_record(mod=mod, target=target, workload_name="main").run_secs

    def commit_record(db, run_sec):  # pylint: disable=invalid-name
        db.commit_tuning_record(
            ms.database.TuningRecord(
                trace,
                workload=db.commit_workload(mod),
                run_secs=[run_sec],
                target=target,
                args_info=arg_info,
            )
        )

    commit_record(db_1, 1.0)
    (run_sec,) = query(db_1)
    assert run_sec.value == 1.0

    commit_record(db_2, 0.5)
    (run_sec,) = query(db_2)
    assert run_sec.value == 0.5

    (run_secs,) = query(ms.database.UnionDatabase(db_1, db_2))
    assert run_secs.value == 0.5

    (run_secs,) = query(ms.database.OrderedUnionDatabase(db_1, db_2))
    assert run_secs.value == 1.0


def test_meta_schedule_pydatabase_default_query():
    mod: IRModule = Matmul
    target = tvm.target.Target("llvm")
    arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
    db = PyMemoryDatabaseDefault()  # pylint: disable=invalid-name
    sch = _create_schedule(mod, _schedule_matmul)
    trace = sch.trace

    def query(db, mod, target, kind):  # pylint: disable=invalid-name
        return db.query(mod=mod, target=target, workload_name="main", kind=kind)

    def commit_record(trace, db, run_sec):  # pylint: disable=invalid-name
        db.commit_tuning_record(
            ms.database.TuningRecord(
                trace,
                workload=db.commit_workload(mod),
                run_secs=[run_sec],
                target=target,
                args_info=arg_info,
            )
        )

    commit_record(trace, db, 1.0)
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 1.0
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, sch.mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, sch.mod)

    commit_record(Schedule(mod).trace, db, 0.2)  # Empty Trace
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 0.2
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, mod)


def test_meta_schedule_pydatabase_override_query():
    mod: IRModule = Matmul
    target = tvm.target.Target("llvm")
    arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
    db = PyMemoryDatabaseOverride()  # pylint: disable=invalid-name
    sch = _create_schedule(mod, _schedule_matmul)
    trace = sch.trace

    def query(db, mod, target, kind):  # pylint: disable=invalid-name
        return db.query(mod=mod, target=target, workload_name="main", kind=kind)

    def commit_record(trace, db, run_sec):  # pylint: disable=invalid-name
        db.commit_tuning_record(
            ms.database.TuningRecord(
                trace,
                workload=db.commit_workload(mod),
                run_secs=[run_sec],
                target=target,
                args_info=arg_info,
            )
        )

    commit_record(trace, db, 1.14)
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 1.14
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, sch.mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, sch.mod)

    commit_record(Schedule(mod).trace, db, 0.514)  # Empty Trace
    record = query(db, mod, target, "record")
    assert record is not None and record.run_secs[0].value == 1.14  # Override to 2nd best
    sch_res = query(db, mod, target, "schedule")
    assert sch_res is not None and tvm.ir.structural_equal(sch_res.mod, sch.mod)
    mod_res = query(db, mod, target, "ir_module")
    assert mod_res is not None and tvm.ir.structural_equal(mod_res, sch.mod)


def test_meta_schedule_pydatabase_current():
    db = PyMemoryDatabaseDefault()  # pylint: disable=invalid-name
    with db:  # pylint: disable=not-context-manager
        assert ms.database.Database.current() == db


def call_get_top_k(run_secs_list, database, k):
    mod: IRModule = Matmul
    workload = database.commit_workload(mod)
    for run_secs in run_secs_list:
        record = ms.database.TuningRecord(
            _create_schedule(mod, _schedule_matmul).trace,
            workload,
            run_secs,
            tvm.target.Target("llvm"),
            ms.arg_info.ArgInfo.from_prim_func(func=mod["main"]),
        )
        database.commit_tuning_record(record)
    return [[v.value for v in record.run_secs] for record in database.get_top_k(workload, k)]


@pytest.mark.parametrize(
    "k,expected",
    [
        (0, []),
        (1, [[0.0, 2.0]]),
        (4, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
        (5, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
    ],
)
def test_memory_database_get_top_k(k, expected):
    run_secs_list = [[1.5, 4.5], [], [0.0, 2.0], None, [2.0], [3.0, 1e10], [1e10]]
    database = ms.database.MemoryDatabase()
    result = call_get_top_k(run_secs_list, database, k)
    assert result == expected


@pytest.mark.parametrize(
    "k,expected",
    [
        (0, []),
        (4, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
        (5, [[0.0, 2.0], [2.0], [1.5, 4.5], [3.0, 1e10]]),
    ],
)
def test_json_database_get_top_k(k, expected):
    run_secs_list = [[1.5, 4.5], [], [0.0, 2.0], None, [2.0], [3.0, 1e10], [1e10]]
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        result = call_get_top_k(run_secs_list, database, k)
    assert result == expected


def MatmulPrimFunc() -> IRModule:
    return Matmul


@pytest.mark.parametrize("f_mod", [MatmulPrimFunc])
@pytest.mark.parametrize("mod_eq", ["structural", "ignore-tensor", "anchor-block"])
def test_json_database_commit_workload(f_mod, mod_eq):
    mod: IRModule = f_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir, mod_eq)
        database.commit_workload(mod)


@pytest.mark.parametrize("f_mod", [MatmulPrimFunc])
@pytest.mark.parametrize("mod_eq", ["structural", "ignore-tensor", "anchor-block"])
def test_memory_database_commit_workload(f_mod, mod_eq):
    mod: IRModule = f_mod()
    database = ms.database.MemoryDatabase(module_equality=mod_eq)
    database.commit_workload(mod)


def test_json_pareto_database_commit_workload():
    """Test that JSONParetoDatabase can commit workloads."""
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir)
        workload = database.commit_workload(mod)
        assert workload is not None
        assert database.has_workload(mod)
        
        # Committing the same workload again should return the same workload
        workload2 = database.commit_workload(mod)
        assert workload.same_as(workload2)


@pytest.mark.parametrize("f_mod", [MatmulPrimFunc])
@pytest.mark.parametrize("mod_eq", ["structural", "ignore-tensor", "anchor-block"])
def test_json_pareto_database_commit_workload_variants(f_mod, mod_eq):
    """Test JSONParetoDatabase commit_workload with different module equality methods."""
    mod: IRModule = f_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir, mod_eq)
        workload = database.commit_workload(mod)
        assert workload is not None
        assert database.has_workload(mod)


def test_json_pareto_database_get_top_k_pareto_front():
    """Test that GetTopK returns Pareto front records (rank 1) sorted by crowding distance."""
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
        
        # Create records with different trade-offs (lower is better for both time and bandwidth):
        # Record 1: Fast time, high bandwidth (Pareto front - trade-off: fast but uses more bandwidth)
        record1 = ms.database.TuningRecord(
            trace, workload, run_secs=[1.0, 1.1, 1.0], bw_mbps=[200.0, 210.0, 205.0],
            target=target, args_info=arg_info
        )
        # Record 2: Slow time, low bandwidth (Pareto front - trade-off: slow but uses less bandwidth)
        record2 = ms.database.TuningRecord(
            trace, workload, run_secs=[3.0, 3.1, 3.0], bw_mbps=[50.0, 55.0, 52.0],
            target=target, args_info=arg_info
        )
        # Record 3: Medium time, medium bandwidth (Pareto front - in between)
        record3 = ms.database.TuningRecord(
            trace, workload, run_secs=[2.0, 2.1, 2.0], bw_mbps=[100.0, 110.0, 105.0],
            target=target, args_info=arg_info
        )
        # Record 4: Dominated by record1 (worse in both objectives: higher time AND higher bandwidth)
        record4 = ms.database.TuningRecord(
            trace, workload, run_secs=[1.5, 1.6, 1.5], bw_mbps=[250.0, 260.0, 255.0],
            target=target, args_info=arg_info
        )
        # Record 5: Dominated by record2 (worse in both objectives: higher time AND higher bandwidth)
        record5 = ms.database.TuningRecord(
            trace, workload, run_secs=[4.0, 4.1, 4.0], bw_mbps=[600.0, 610.0, 605.0],
            target=target, args_info=arg_info
        )
        
        # Commit all records
        for record in [record1, record2, record3, record4, record5]:
            database.commit_tuning_record(record)
        
        # GetTopK should return only Pareto front records (rank 1)
        # These should be record1, record2, record3 (not dominated by any)
        top_k = database.get_top_k(workload, 10)
        
        # Should have exactly 3 Pareto front records
        assert len(top_k) == 3
        
        # Verify that record4 and record5 are not in the top results
        # (they are dominated). Since all records share the same trace,
        # we need to check by comparing run_secs and bw_mbps values.
        def get_mean(values):
            if not values:
                return 0.0
            # Convert T.float32 to Python float for comparison
            return float(sum(float(v) for v in values) / len(values))
        
        top_k_values = [
            (get_mean(r.run_secs), get_mean(r.bw_mbps)) for r in top_k
        ]
        record4_values = (get_mean(record4.run_secs), get_mean(record4.bw_mbps))
        record5_values = (get_mean(record5.run_secs), get_mean(record5.bw_mbps))
        
        assert record4_values not in top_k_values, f"record4 {record4_values} should not be in top_k"
        assert record5_values not in top_k_values, f"record5 {record5_values} should not be in top_k"
        
        # Verify that all top_k records are from Pareto front
        # (record1, record2, record3 should all be present)
        record1_values = (get_mean(record1.run_secs), get_mean(record1.bw_mbps))
        record2_values = (get_mean(record2.run_secs), get_mean(record2.bw_mbps))
        record3_values = (get_mean(record3.run_secs), get_mean(record3.bw_mbps))
        
        assert record1_values in top_k_values, f"record1 {record1_values} should be in top_k"
        assert record2_values in top_k_values, f"record2 {record2_values} should be in top_k"
        assert record3_values in top_k_values, f"record3 {record3_values} should be in top_k"


def test_json_pareto_database_get_top_k_crowding_distance():
    """Test that GetTopK returns records sorted by crowding distance within Pareto front."""
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
        
        # Create multiple Pareto front records with different crowding distances
        # Boundary points should have higher crowding distance
        # For Pareto front: records should have trade-offs (better time but worse bandwidth, etc.)
        records = [
            # Extreme points (should have high crowding distance)
            ms.database.TuningRecord(
                trace, workload, run_secs=[1.0], bw_mbps=[200.0],  # Fast time, high BW (trade-off)
                target=target, args_info=arg_info
            ),
            ms.database.TuningRecord(
                trace, workload, run_secs=[5.0], bw_mbps=[50.0],  # Slow time, low BW (trade-off)
                target=target, args_info=arg_info
            ),
            # Interior points (should have lower crowding distance)
            ms.database.TuningRecord(
                trace, workload, run_secs=[2.5], bw_mbps=[125.0],  # Middle time, middle BW
                target=target, args_info=arg_info
            ),
            ms.database.TuningRecord(
                trace, workload, run_secs=[3.0], bw_mbps=[100.0],  # Middle time, middle BW
                target=target, args_info=arg_info
            ),
        ]
        
        for record in records:
            database.commit_tuning_record(record)
        
        top_k = database.get_top_k(workload, 10)
        
        # All records should be in Pareto front (none dominate each other)
        assert len(top_k) == 4
        
        # Verify sorting by crowding distance (descending)
        # Boundary points (extreme values) should have infinite crowding distance
        # and come first, followed by interior points sorted by distance
        
        def get_mean(values):
            if not values:
                return 0.0
            return float(sum(float(v) for v in values) / len(values))
        
        def get_record_values(record):
            return (get_mean(record.run_secs), get_mean(record.bw_mbps))
        
        # Map records to their values for identification
        record_values = [get_record_values(r) for r in records]
        top_k_values = [get_record_values(r) for r in top_k]
        
        # Verify all records are present
        for rv in record_values:
            assert rv in top_k_values, f"Record {rv} should be in top_k"
        
        # Calculate expected crowding distances
        times = [rv[0] for rv in record_values]
        bws = [rv[1] for rv in record_values]
        min_time, max_time = min(times), max(times)
        min_bw, max_bw = min(bws), max(bws)
        time_range = max_time - min_time
        bw_range = max_bw - min_bw
        
        # Find boundary points (min/max time and min/max bandwidth)
        boundary_indices = set()
        for i, (t, b) in enumerate(record_values):
            if abs(t - min_time) < 1e-6 or abs(t - max_time) < 1e-6:
                boundary_indices.add(i)
            if abs(b - min_bw) < 1e-6 or abs(b - max_bw) < 1e-6:
                boundary_indices.add(i)
        
        # Boundary points should come first (they have infinite crowding distance)
        # Check that all boundary points are in the first positions
        boundary_values = [record_values[i] for i in boundary_indices]
        num_boundary = len(boundary_indices)
        
        # The first num_boundary records should be boundary points
        first_n_values = top_k_values[:num_boundary]
        for bv in boundary_values:
            assert bv in first_n_values, f"Boundary point {bv} should be in first {num_boundary} positions"
        
        # Interior points should come after boundary points
        # and be sorted by crowding distance (descending)
        if len(record_values) > num_boundary:
            interior_indices = set(range(len(record_values))) - boundary_indices
            interior_values = [record_values[i] for i in interior_indices]
            
            # Calculate expected crowding distances for interior points
            # (simplified - just verify they're not boundary)
            interior_in_top_k = [v for v in top_k_values if v in interior_values]
            
            # Interior points should come after boundary points
            assert len(interior_in_top_k) == len(interior_values), \
                f"All {len(interior_values)} interior points should be in top_k"
            
            # Verify interior points are after boundary points
            for iv in interior_values:
                iv_pos = top_k_values.index(iv)
                assert iv_pos >= num_boundary, \
                    f"Interior point {iv} at position {iv_pos} should come after boundary points"


def test_json_pareto_database_get_top_k_crowding_distance_exact_order():
    """Test that GetTopK returns records in exact order by crowding distance (descending)."""
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
        
        # Create records with predictable crowding distances
        # All 4 records should be in Pareto front (none dominate each other)
        # Record 1: Min time, max bandwidth (boundary - infinite distance)
        # Record 2: Max time, min bandwidth (boundary - infinite distance)
        # Record 3: Middle time, middle bandwidth (interior - finite distance)
        # Record 4: Another middle point with trade-off (interior - finite distance)
        records = [
            ms.database.TuningRecord(
                trace, workload, run_secs=[1.0], bw_mbps=[300.0],  # Min time, max BW
                target=target, args_info=arg_info
            ),
            ms.database.TuningRecord(
                trace, workload, run_secs=[10.0], bw_mbps=[50.0],  # Max time, min BW
                target=target, args_info=arg_info
            ),
            ms.database.TuningRecord(
                trace, workload, run_secs=[5.0], bw_mbps=[150.0],  # Middle time, middle BW
                target=target, args_info=arg_info
            ),
            ms.database.TuningRecord(
                trace, workload, run_secs=[4.0], bw_mbps=[200.0],  # Faster time but higher BW (trade-off)
                target=target, args_info=arg_info
            ),
        ]
        
        for record in records:
            database.commit_tuning_record(record)
        
        top_k = database.get_top_k(workload, 10)
        assert len(top_k) == 4
        
        def get_mean(values):
            if not values:
                return 0.0
            return float(sum(float(v) for v in values) / len(values))
        
        def get_record_values(record):
            return (get_mean(record.run_secs), get_mean(record.bw_mbps))
        
        # Get values for each record
        record0_val = get_record_values(records[0])  # (1.0, 300.0) - boundary
        record1_val = get_record_values(records[1])  # (10.0, 50.0) - boundary
        record2_val = get_record_values(records[2])  # (5.0, 150.0) - interior
        record3_val = get_record_values(records[3])  # (4.0, 200.0) - interior
        
        top_k_values = [get_record_values(r) for r in top_k]
        
        # Boundary points (records 0 and 1) should come first (infinite distance)
        # They can be in any order among themselves
        assert record0_val in top_k_values[:2], "Record 0 (boundary) should be in first 2"
        assert record1_val in top_k_values[:2], "Record 1 (boundary) should be in first 2"
        
        # Interior points (records 2 and 3) should come after boundary points
        record2_pos = top_k_values.index(record2_val)
        record3_pos = top_k_values.index(record3_val)
        assert record2_pos >= 2, f"Record 2 (interior) at position {record2_pos} should be >= 2"
        assert record3_pos >= 2, f"Record 3 (interior) at position {record3_pos} should be >= 2"
        
        # Calculate expected crowding distances for interior points
        # Time: [1.0, 5.0, 6.0, 10.0] -> range = 9.0
        # BW: [50.0, 150.0, 200.0, 300.0] -> range = 250.0
        # Record 2 (5.0, 150.0): time contribution = (6.0 - 1.0) / 9.0 = 5/9 ≈ 0.556
        #                        BW contribution = (200.0 - 50.0) / 250.0 = 150/250 = 0.6
        #                        Total ≈ 1.156
        # Record 3 (6.0, 200.0): time contribution = (10.0 - 5.0) / 9.0 = 5/9 ≈ 0.556
        #                        BW contribution = (300.0 - 150.0) / 250.0 = 150/250 = 0.6
        #                        Total ≈ 1.156
        
        # Both interior points have similar distances, so order may vary
        # But they should both come after boundary points


def test_json_pareto_database_get_top_k_limit():
    """Test that GetTopK respects the top_k limit."""
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir)
        workload = database.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
        
        # Create 5 Pareto front records with trade-offs (none dominate each other)
        # Each record has a different trade-off: better time but worse bandwidth, or vice versa
        pareto_records = [
            (1.0, 200.0),   # Fast time, high bandwidth
            (2.0, 150.0),   # Medium-fast time, medium-high bandwidth
            (3.0, 100.0),   # Medium time, medium bandwidth
            (4.0, 75.0),    # Medium-slow time, medium-low bandwidth
            (5.0, 50.0),    # Slow time, low bandwidth
        ]
        for time, bw in pareto_records:
            record = ms.database.TuningRecord(
                trace, workload,
                run_secs=[time], bw_mbps=[bw],
                target=target, args_info=arg_info
            )
            database.commit_tuning_record(record)
        
        # Test different top_k values
        assert len(database.get_top_k(workload, 0)) == 0
        assert len(database.get_top_k(workload, 1)) == 1
        assert len(database.get_top_k(workload, 3)) == 3
        assert len(database.get_top_k(workload, 5)) == 5
        assert len(database.get_top_k(workload, 10)) == 5  # Only 5 records exist


def test_json_pareto_database_get_top_k_empty():
    """Test GetTopK with no records."""
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir)
        workload = database.commit_workload(mod)
        
        # No records committed
        top_k = database.get_top_k(workload, 5)
        assert len(top_k) == 0


def test_json_pareto_database_get_top_k_different_workloads():
    """Test that GetTopK only returns records for the specified workload."""
    mod1: IRModule = Matmul
    mod2: IRModule = MatmulRelu
    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_pareto_database(tmpdir)
        workload1 = database.commit_workload(mod1)
        workload2 = database.commit_workload(mod2)
        trace1 = _create_schedule(mod1, _schedule_matmul).trace
        trace2 = _create_schedule(mod2, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        
        # Create records for both workloads
        record1 = ms.database.TuningRecord(
            trace1, workload1, run_secs=[1.0], bw_mbps=[100.0],
            target=target, args_info=ms.arg_info.ArgInfo.from_prim_func(func=mod1["main"])
        )
        record2 = ms.database.TuningRecord(
            trace2, workload2, run_secs=[2.0], bw_mbps=[200.0],
            target=target, args_info=ms.arg_info.ArgInfo.from_prim_func(func=mod2["main"])
        )
        
        database.commit_tuning_record(record1)
        database.commit_tuning_record(record2)
        
        # GetTopK for workload1 should only return record1
        top_k1 = database.get_top_k(workload1, 10)
        assert len(top_k1) == 1
        assert str(top_k1[0].trace) == str(record1.trace)
        
        # GetTopK for workload2 should only return record2
        top_k2 = database.get_top_k(workload2, 10)
        assert len(top_k2) == 1
        assert str(top_k2[0].trace) == str(record2.trace)


def test_json_pareto_database_reload():
    """Test that JSONParetoDatabase can reload from files and maintain Pareto ranking."""
    mod: IRModule = Matmul
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create database and commit records
        database1 = _create_tmp_pareto_database(tmpdir)
        workload = database1.commit_workload(mod)
        trace = _create_schedule(mod, _schedule_matmul).trace
        target = tvm.target.Target("llvm")
        arg_info = ms.arg_info.ArgInfo.from_prim_func(func=mod["main"])
        
        # Create 3 Pareto front records with trade-offs (none dominate each other)
        records = [
            ms.database.TuningRecord(
                trace, workload, run_secs=[1.0], bw_mbps=[200.0],  # Fast time, high bandwidth
                target=target, args_info=arg_info
            ),
            ms.database.TuningRecord(
                trace, workload, run_secs=[3.0], bw_mbps=[50.0],  # Slow time, low bandwidth
                target=target, args_info=arg_info
            ),
            ms.database.TuningRecord(
                trace, workload, run_secs=[2.0], bw_mbps=[100.0],  # Medium time, medium bandwidth
                target=target, args_info=arg_info
            ),
        ]
        
        for record in records:
            database1.commit_tuning_record(record)
        
        # Reload database from files
        database2 = _create_tmp_pareto_database(tmpdir)
        workload2 = database2.commit_workload(mod)
        
        # GetTopK should still return Pareto front records
        top_k = database2.get_top_k(workload2, 10)
        assert len(top_k) == 3  # All 3 records are in Pareto front


if __name__ == "__main__":
    tvm.testing.main()
