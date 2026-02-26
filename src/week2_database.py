"""
Week 2 - GPR Processing Database (SQLAlchemy ORM + SQLite)
데이터셋 메타데이터 및 전처리 파이프라인 이력 관리
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, ForeignKey
)
from sqlalchemy.orm import declarative_base, relationship, Session

Base = declarative_base()


# ─────────────────────────────────────────────
# ORM 테이블 정의
# ─────────────────────────────────────────────

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    format = Column(String)              # DT1, DZT, IDS .dt
    frequency_mhz = Column(Float)
    n_samples = Column(Integer)
    n_traces = Column(Integer)
    time_window_ns = Column(Float)
    dx_m = Column(Float)                 # 트레이스 간격
    data_min = Column(Float)
    data_max = Column(Float)
    data_mean = Column(Float)
    data_std = Column(Float)
    created_at = Column(DateTime, default=datetime.now)

    runs = relationship("ProcessingRun", back_populates="dataset",
                        cascade="all, delete-orphan")

    def __repr__(self):
        return (f"<Dataset(id={self.id}, name='{self.name}', "
                f"shape=({self.n_samples}x{self.n_traces}), "
                f"freq={self.frequency_mhz}MHz)>")


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.now)

    dataset = relationship("Dataset", back_populates="runs")
    steps = relationship("ProcessingStep", back_populates="run",
                         cascade="all, delete-orphan",
                         order_by="ProcessingStep.step_order")

    def __repr__(self):
        return (f"<ProcessingRun(id={self.id}, dataset_id={self.dataset_id}, "
                f"steps={len(self.steps)})>")


class ProcessingStep(Base):
    __tablename__ = "processing_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("processing_runs.id"), nullable=False)
    step_order = Column(Integer, nullable=False)
    step_name = Column(String, nullable=False)
    parameters = Column(Text)            # JSON 문자열
    elapsed_ms = Column(Float)

    run = relationship("ProcessingRun", back_populates="steps")

    def __repr__(self):
        return (f"<ProcessingStep(order={self.step_order}, "
                f"name='{self.step_name}', {self.elapsed_ms:.1f}ms)>")

    @property
    def params_dict(self):
        if self.parameters:
            return json.loads(self.parameters)
        return {}


# ─────────────────────────────────────────────
# GPRDatabase 클래스
# ─────────────────────────────────────────────

class GPRDatabase:
    """GPR 데이터셋 및 처리 이력 관리"""

    def __init__(self, db_path="G:/RAG_system/db/gpr_processing.db"):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.db_path = db_path

    def _session(self):
        return Session(self.engine)

    # ── 데이터셋 등록 ──

    def register_dataset(self, name, file_path, data, header=None,
                         format=None, frequency_mhz=None,
                         time_window_ns=None, dx_m=None):
        """
        데이터셋을 DB에 등록 (data에서 자동 통계 추출)
        Returns: Dataset.id
        """
        # 헤더에서 주파수/시간창 자동 추출 시도
        if header and frequency_mhz is None:
            for key in ['ANTENNA FREQUENCY', 'antfreq', 'MAX_FREQ']:
                if key in header:
                    val = header[key]
                    try:
                        freq_val = float(val)
                        # Hz 단위면 MHz로 변환
                        if freq_val > 1e6:
                            freq_val /= 1e6
                        frequency_mhz = freq_val
                    except (ValueError, TypeError):
                        pass
                    break

        if header and time_window_ns is None:
            for key in ['TOTAL TIME WINDOW']:
                if key in header:
                    try:
                        time_window_ns = float(header[key])
                    except (ValueError, TypeError):
                        pass
                    break

        ds = Dataset(
            name=name,
            file_path=str(file_path),
            format=format,
            frequency_mhz=frequency_mhz,
            n_samples=int(data.shape[0]),
            n_traces=int(data.shape[1]),
            time_window_ns=time_window_ns,
            dx_m=dx_m,
            data_min=float(np.min(data)),
            data_max=float(np.max(data)),
            data_mean=float(np.mean(data)),
            data_std=float(np.std(data)),
        )

        with self._session() as session:
            session.add(ds)
            session.commit()
            ds_id = ds.id
            print(f"  [DB] 데이터셋 등록: id={ds_id}, '{name}' "
                  f"({data.shape[0]}x{data.shape[1]})")
            return ds_id

    # ── 처리 이력 기록 ──

    def log_processing_run(self, dataset_id, description, steps):
        """
        파이프라인 실행 이력 기록
        steps: list of dict {'step_name', 'parameters', 'elapsed_ms'}
        Returns: ProcessingRun.id
        """
        run = ProcessingRun(
            dataset_id=dataset_id,
            description=description,
        )

        for i, step_info in enumerate(steps):
            params_json = json.dumps(step_info.get('parameters', {}),
                                     ensure_ascii=False)
            step = ProcessingStep(
                step_order=i + 1,
                step_name=step_info['step_name'],
                parameters=params_json,
                elapsed_ms=step_info.get('elapsed_ms', 0.0),
            )
            run.steps.append(step)

        with self._session() as session:
            session.add(run)
            session.commit()
            run_id = run.id
            total_ms = sum(s.get('elapsed_ms', 0) for s in steps)
            print(f"  [DB] 처리 이력 기록: run_id={run_id}, "
                  f"{len(steps)}단계, 총 {total_ms:.1f}ms")
            return run_id

    # ── 조회 ──

    def list_datasets(self):
        """등록된 모든 데이터셋 목록 반환"""
        with self._session() as session:
            datasets = session.query(Dataset).all()
            results = []
            for ds in datasets:
                results.append({
                    'id': ds.id,
                    'name': ds.name,
                    'format': ds.format,
                    'shape': (ds.n_samples, ds.n_traces),
                    'frequency_mhz': ds.frequency_mhz,
                    'n_runs': len(ds.runs),
                })
            return results

    def get_dataset_history(self, dataset_id):
        """특정 데이터셋의 처리 이력 조회"""
        with self._session() as session:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if not ds:
                print(f"  [DB] 데이터셋 id={dataset_id} 없음")
                return None

            history = {
                'dataset': ds.name,
                'shape': (ds.n_samples, ds.n_traces),
                'runs': []
            }
            for run in ds.runs:
                run_info = {
                    'run_id': run.id,
                    'description': run.description,
                    'created_at': str(run.created_at),
                    'steps': []
                }
                for step in run.steps:
                    run_info['steps'].append({
                        'order': step.step_order,
                        'name': step.step_name,
                        'parameters': step.params_dict,
                        'elapsed_ms': step.elapsed_ms,
                    })
                history['runs'].append(run_info)
            return history

    def print_summary(self):
        """DB 전체 요약 출력"""
        datasets = self.list_datasets()
        print(f"\n{'='*60}")
        print(f"  GPR Processing Database Summary")
        print(f"  DB: {self.db_path}")
        print(f"{'='*60}")
        print(f"  총 데이터셋: {len(datasets)}개\n")

        for ds in datasets:
            print(f"  [{ds['id']}] {ds['name']}")
            print(f"      포맷: {ds['format']}, "
                  f"Shape: {ds['shape']}, "
                  f"주파수: {ds['frequency_mhz']}MHz")
            print(f"      처리 이력: {ds['n_runs']}건")

            # 각 run의 상세 출력
            history = self.get_dataset_history(ds['id'])
            if history:
                for run in history['runs']:
                    total_ms = sum(s['elapsed_ms'] for s in run['steps'])
                    print(f"      ├─ Run #{run['run_id']}: "
                          f"{run['description']} ({total_ms:.1f}ms)")
                    for step in run['steps']:
                        print(f"      │  {step['order']}. {step['name']} "
                              f"({step['elapsed_ms']:.1f}ms)")
            print()

        print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Week 2 - GPR Database 테스트")
    print("=" * 50)

    db = GPRDatabase()

    # 테스트 데이터로 등록
    dummy_data = np.random.randn(1000, 225).astype(np.float32)
    ds_id = db.register_dataset(
        name="test_dataset",
        file_path="/tmp/test.dt1",
        data=dummy_data,
        format="DT1",
        frequency_mhz=100.0,
        time_window_ns=50.0,
        dx_m=0.25,
    )

    # 테스트 처리 이력
    db.log_processing_run(
        dataset_id=ds_id,
        description="테스트 파이프라인",
        steps=[
            {'step_name': 'DC Removal', 'parameters': {}, 'elapsed_ms': 1.2},
            {'step_name': 'Dewow', 'parameters': {'window': 50}, 'elapsed_ms': 2.5},
            {'step_name': 'Background Removal', 'parameters': {}, 'elapsed_ms': 0.8},
        ]
    )

    # 요약 출력
    db.print_summary()

    # 정리: 테스트 데이터 삭제
    with db._session() as session:
        ds = session.query(Dataset).filter_by(id=ds_id).first()
        if ds:
            session.delete(ds)
            session.commit()
    print("테스트 데이터 정리 완료")

    # DB 파일 존재 확인
    print(f"\nDB 파일: {db.db_path} (존재: {db.db_path.exists()})")
