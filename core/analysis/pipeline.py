# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Orchestrates all geometric analyzers into a single pipeline."""

import time
import concurrent.futures

import torch
from typing import Optional

from core.algebra import CliffordAlgebra

from ._types import AnalysisConfig, AnalysisReport, SamplingConfig
from .sampler import StatisticalSampler
from .dimension import EffectiveDimensionAnalyzer
from .signature import SignatureSearchAnalyzer
from .spectral import SpectralAnalyzer
from .symmetry import SymmetryDetector
from .commutator import CommutatorAnalyzer


class GeometricAnalyzer:
    """Top-level orchestrator for the geometric analysis toolkit.

    Runs a configurable subset of analyzers in the correct dependency
    order and returns an :class:`AnalysisReport`.

    **Input modes:**

    * ``data.ndim == 2`` and ``algebra is None`` -- **raw mode**: full
      pipeline from sampling through signature search to GA analyses.
    * ``data.ndim == 3`` and ``algebra is not None`` -- **pre-embedded
      mode**: skip sampling / dimension / signature; run spectral,
      symmetry, and commutator analyses directly.
    * ``data.ndim == 2`` and ``algebra is not None`` -- **raw + known
      algebra**: embed data, then run GA analyses.

    Args:
        config: Master analysis configuration.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def analyze(
        self,
        data: torch.Tensor,
        algebra: Optional[CliffordAlgebra] = None,
    ) -> AnalysisReport:
        """Run the full geometric analysis pipeline.

        Args:
            data: Raw ``[N, D]`` or pre-embedded ``[N, C, 2^n]`` tensor.
            algebra: Required when *data* is pre-embedded.  Optional
                when raw -- will be created from the signature search.

        Returns:
            :class:`AnalysisReport`.
        """
        cfg = self.config
        report = AnalysisReport()
        t0 = time.time()

        data = data.to(cfg.device)
        report.metadata["data_shape"] = list(data.shape)
        report.metadata["config_device"] = cfg.device

        if data.ndim == 3 and algebra is not None:
            # Pre-embedded mode
            report = self._run_ga_analyses(data, algebra, report)
        elif data.ndim == 2 and algebra is not None:
            # Raw + known algebra
            mv_data = self._embed_raw(data, algebra)
            report = self._run_ga_analyses(mv_data, algebra, report)
        elif data.ndim == 2:
            # Full raw mode
            report = self._run_full_pipeline(data, report)
        else:
            raise ValueError(f"Unexpected data shape {data.shape}. Expected [N, D] or [N, C, dim].")

        report.metadata["elapsed_seconds"] = round(time.time() - t0, 2)
        return report

    def _run_full_pipeline(self, data: torch.Tensor, report: AnalysisReport) -> AnalysisReport:
        cfg = self.config

        # 1. Sampling
        sampled, sample_meta = StatisticalSampler.sample(data, cfg.sampling)
        if isinstance(sampled, list):
            # bootstrap returns list -- use first for pipeline, rest for CI
            sampled = sampled[0]
        report.metadata["sampling"] = sample_meta

        # 2. Dimension analysis
        dim_result = None
        if cfg.run_dimension:
            da = EffectiveDimensionAnalyzer(
                device=cfg.device,
                energy_threshold=cfg.energy_threshold,
            )
            dim_result = da.analyze(sampled)
            report.dimension = dim_result

        # 3. Signature search
        sig_result = None
        if cfg.run_signature:
            ssa = SignatureSearchAnalyzer(device=cfg.device)
            sig_result = ssa.analyze(sampled, dim_result=dim_result)
            report.signature = sig_result

        # 4. Create algebra and embed (n >= 2 for meaningful GA structure)
        if sig_result is not None:
            p, q, r = sig_result.signature
        elif dim_result is not None:
            p, q, r = dim_result.intrinsic_dim, 0, 0
        else:
            p, q, r = min(sampled.shape[1], 6), 0, 0
        # Ensure n >= 2 so grade-2 (bivectors) exist for GA analyses
        if p + q + r < 2:
            p = max(p, 2 - q - r)

        algebra = CliffordAlgebra(p, q, r, device=cfg.device)
        mv_data = self._embed_raw(sampled, algebra)

        # 5. GA analyses (parallel)
        report = self._run_ga_analyses(mv_data, algebra, report)

        return report

    def _run_ga_analyses(
        self,
        mv_data: torch.Tensor,
        algebra: CliffordAlgebra,
        report: AnalysisReport,
    ) -> AnalysisReport:
        cfg = self.config

        tasks = {}
        if cfg.run_spectral:
            tasks["spectral"] = lambda: SpectralAnalyzer(algebra).analyze(mv_data)
        if cfg.run_symmetry:
            tasks["symmetry"] = lambda: SymmetryDetector(algebra, null_threshold=cfg.energy_threshold).analyze(mv_data)
        if cfg.run_commutator:
            tasks["commutator"] = lambda: CommutatorAnalyzer(algebra).analyze(mv_data)

        if len(tasks) <= 1:
            # Sequential
            for name, fn in tasks.items():
                setattr(report, name, fn())
        else:
            # Parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {name: pool.submit(fn) for name, fn in tasks.items()}
                for name, fut in futures.items():
                    setattr(report, name, fut.result())

        # Refine continuous symmetries with commutator result
        if cfg.run_symmetry and cfg.run_commutator and report.symmetry is not None and report.commutator is not None:
            if mv_data.ndim == 3:
                flat = mv_data.mean(dim=1)
            else:
                flat = mv_data
            detector = SymmetryDetector(algebra, null_threshold=cfg.energy_threshold)
            report.symmetry.continuous_symmetry_dim = detector.detect_continuous_symmetries(
                flat, commutator_result=report.commutator
            )

        return report

    @staticmethod
    def _embed_raw(data: torch.Tensor, algebra: CliffordAlgebra) -> torch.Tensor:
        """Embed raw ``[N, D]`` data as grade-1 multivectors ``[N, 1, dim]``."""
        n = algebra.n
        D = data.shape[1]
        if D > n:
            data = data[:, :n]
        elif D < n:
            pad = torch.zeros(data.shape[0], n - D, device=data.device, dtype=data.dtype)
            data = torch.cat([data, pad], dim=-1)
        mv = algebra.embed_vector(data.float())  # [N, dim]
        return mv.unsqueeze(1)  # [N, 1, dim]
