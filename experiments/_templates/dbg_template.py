# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""
==============================================================================
VERSOR EXPERIMENT: MATHEMATICAL DEBUGGER
==============================================================================

This script is designed to validate topological and algebraic phenomena rather
than to achieve State-of-the-Art (SOTA) on traditional benchmarks. Its focus
is to confirm that the Clifford Algebra framework computes known identities
and physical laws correctly, and to surface regressions when they do not.

Please kindly note that as an experimental module, formal mathematical proofs
and exhaustive literature reviews may still be in progress. Contributions that
tighten the validation suite — additional check_* methods, sharper tolerances,
cross-references to the literature — are warmly welcomed.

==============================================================================

Debugger Template — Rotor sanity checks in Cl(3,0).

Hypothesis
  The algebra kernel should satisfy the three lowest-level Spin-group
  identities without any learned model in the loop: unit rotor
  ``R * ~R == 1``, sandwich isometry ``|R x ~R|^2 == |x|^2``, and reverse
  involution ``reverse(reverse(mv)) == mv``. This file is the minimal smoke
  test for regressions in the framework core.

Execute Command
  uv run python -m experiments._templates.dbg_template
  uv run python -m experiments._templates.dbg_template --signature 3,1,0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from experiments._lib import (
    add_signature_arg,
    make_experiment_parser,
    print_banner,
    section_header,
    set_seed,
    setup_algebra,
)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class RotorSanityDebugger:
    """Three-check rotor validator.

    Each ``check_*`` method returns a ``dict`` with a ``'max_error'`` key so
    the report formatter can summarise every check uniformly.
    """

    def __init__(self, algebra, num_samples: int = 256, seed: int = 42):
        self.algebra = algebra
        self.num_samples = num_samples
        self.generator = torch.Generator().manual_seed(seed)

    # --- sampling ---------------------------------------------------------

    def _random_bivector(self) -> torch.Tensor:
        """Pure grade-2 multivector of shape ``[num_samples, dim]``."""
        mv = torch.randn(self.num_samples, self.algebra.dim, generator=self.generator) * 0.1
        mask = self.algebra.grade_masks[2].to(dtype=mv.dtype)
        return mv * mask

    def _random_vector(self) -> torch.Tensor:
        """Pure grade-1 multivector of shape ``[num_samples, dim]``."""
        mv = torch.randn(self.num_samples, self.algebra.dim, generator=self.generator) * 0.1
        mask = self.algebra.grade_masks[1].to(dtype=mv.dtype)
        return mv * mask

    def _random_multivector(self) -> torch.Tensor:
        return torch.randn(self.num_samples, self.algebra.dim, generator=self.generator) * 0.1

    # --- checks -----------------------------------------------------------

    def check_unit_rotor(self) -> Dict[str, float]:
        """``R = exp(-B/2)`` satisfies ``R * ~R == 1``."""
        bivector = self._random_bivector()
        rotor = self.algebra.exp(-0.5 * bivector)
        product = self.algebra.geometric_product(rotor, self.algebra.reverse(rotor))
        identity = torch.zeros_like(product)
        identity[..., 0] = 1.0
        errors = (product - identity).abs()
        return {'max_error': errors.max().item(), 'mean_error': errors.mean().item(), 'num_samples': self.num_samples}

    def check_sandwich_preserves_norm(self) -> Dict[str, float]:
        """``|R x ~R| == |x|`` for any grade-1 ``x`` and rotor ``R``."""
        bivector = self._random_bivector()
        rotor = self.algebra.exp(-0.5 * bivector)
        rotor_reverse = self.algebra.reverse(rotor)
        vector = self._random_vector()
        rotated = self.algebra.geometric_product(
            self.algebra.geometric_product(rotor, vector),
            rotor_reverse,
        )
        norm_original = self.algebra.norm_sq(vector).squeeze(-1)
        norm_rotated = self.algebra.norm_sq(rotated).squeeze(-1)
        errors = (norm_rotated - norm_original).abs()
        return {'max_error': errors.max().item(), 'mean_error': errors.mean().item(), 'num_samples': self.num_samples}

    def check_reverse_involution(self) -> Dict[str, float]:
        """``reverse(reverse(mv)) == mv`` on a dense multivector."""
        mv = self._random_multivector()
        twice = self.algebra.reverse(self.algebra.reverse(mv))
        errors = (twice - mv).abs()
        return {'max_error': errors.max().item(), 'mean_error': errors.mean().item(), 'num_samples': self.num_samples}

    # --- orchestration ----------------------------------------------------

    def run(self) -> Dict[str, Dict[str, float]]:
        return {
            'unit_rotor': self.check_unit_rotor(),
            'sandwich_norm': self.check_sandwich_preserves_norm(),
            'reverse_involution': self.check_reverse_involution(),
        }


def format_report(results: Dict[str, Dict[str, float]], tolerance: float = 1e-4) -> str:
    """Tabular summary. Rows end in ``OK`` or ``FAIL`` based on ``tolerance``."""
    lines = [section_header('Rotor Sanity Report')]
    header = f'  {"check":<22} {"max_error":>12} {"mean_error":>12} {"status":>8}'
    lines.append(header)
    lines.append('  ' + '-' * (len(header) - 2))
    all_passed = True
    for name, result in results.items():
        status = 'OK' if result['max_error'] < tolerance else 'FAIL'
        if status == 'FAIL':
            all_passed = False
        lines.append(f'  {name:<22} {result["max_error"]:>12.3e} {result["mean_error"]:>12.3e} {status:>8}')
    lines.append('  ' + '-' * (len(header) - 2))
    lines.append(f'  Overall: {"PASS" if all_passed else "FAIL"} (tolerance={tolerance:.1e})')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = make_experiment_parser(
        'Debugger template — rotor sanity checks.',
        include=('seed', 'device'),
    )
    add_signature_arg(parser, default=(3, 0, 0))
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    p, q, r = args.signature
    algebra = setup_algebra(p=p, q=q, r=r, device=args.device)

    print_banner(
        'Debugger Template — Rotor Sanity Checks',
        signature=f'Cl({p}, {q}, {r})',
        num_samples=args.num_samples,
        tolerance=args.tolerance,
    )

    debugger = RotorSanityDebugger(algebra, num_samples=args.num_samples, seed=args.seed)
    results = debugger.run()
    print(format_report(results, tolerance=args.tolerance))


if __name__ == '__main__':
    main()
