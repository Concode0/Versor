# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Phase 2: Mother algebra cross-term discovery mixin.

Contains graph-guided and legacy cross-term discovery logic.
"""

import logging

import numpy as np
import torch

from models.sr.net import SRGBN
from models.sr.translator import RotorTranslator
from models.sr.utils import standardize

logger = logging.getLogger(__name__)


class CrossTermMixin:
    """Phase 2 methods: mother algebra cross-term discovery."""

    def _mother_algebra_joint(self, groups, X_orig, y_orig, X_norm, prep=None):
        """Graph-guided cross-term discovery between variable groups.

        Instead of brute-force GP energy checking, uses the relationship
        graph's cross-group edges to identify which specific group pairs
        have significant coupling, then trains focused sub-mother algebras
        for each significant pair.

        Falls back to the legacy full-mother-algebra approach if the
        relationship graph is unavailable.

        Returns:
            (list[RotorTerm], float): Cross-terms and total cross-energy.
        """
        if len(groups) < 2:
            return [], 0.0

        from models.sr.grouper import VariableGrouper

        gcfg = self.grouping_config
        grouper = VariableGrouper(
            max_groups=self.max_groups,
            device=self.device,
            sample_size=gcfg.get("sample_size", 500),
            commutator_weight=gcfg.get("commutator_weight", 0.4),
            coherence_weight=gcfg.get("coherence_weight", 0.3),
            spectral_weight=gcfg.get("spectral_weight", 0.3),
        )

        # Graph-guided path: use cross-group edges to identify significant pairs
        graph = prep.relationship_graph if prep is not None else None
        if graph is not None and self.graph_guided:
            return self._mother_algebra_graph_guided(
                groups,
                X_orig,
                y_orig,
                X_norm,
                graph,
                grouper,
            )

        # Legacy fallback: brute-force GP energy check
        return self._mother_algebra_legacy(
            groups,
            X_orig,
            y_orig,
            X_norm,
            grouper,
        )

    def _mother_algebra_graph_guided(self, groups, X_orig, y_orig, X_norm, graph, grouper):
        """Targeted cross-term discovery using relationship graph edges.

        Only explores group pairs that the graph identifies as having
        significant cross-group coupling, and biases rotors toward the
        specific coupling planes.
        """
        cross_edges = graph.cross_group_edges()
        if not cross_edges:
            logger.info("  No cross-group edges, skipping Phase 2")
            return [], 0.0

        # Filter to significant edges
        significant = [e for e in cross_edges if e.strength > self.mother_cross_threshold]
        if not significant:
            logger.info(
                f"  {len(cross_edges)} cross-edges all below threshold ({self.mother_cross_threshold}), skipping"
            )
            return [], 0.0

        # Group significant edges by group-pair
        group_pairs = {}
        for e in significant:
            ga = graph.group_assignments.get(e.var_i)
            gb = graph.group_assignments.get(e.var_j)
            if ga is None or gb is None or ga == gb:
                continue
            pair_key = (min(ga, gb), max(ga, gb))
            group_pairs.setdefault(pair_key, []).append(e)

        if not group_pairs:
            return [], 0.0

        # Limit to top-k group pairs by max edge strength
        pair_list = sorted(
            group_pairs.items(),
            key=lambda kv: max(e.strength for e in kv[1]),
            reverse=True,
        )[: self.cross_edge_top_k]

        logger.info(
            f"  Graph-guided Phase 2: {len(pair_list)} group pairs from {len(significant)} significant cross-edges"
        )

        all_cross_terms = []
        total_cross_energy = 0.0

        for (g_a, g_b), pair_edges in pair_list:
            terms, energy = self._train_focused_cross_rotor(
                groups[g_a],
                groups[g_b],
                pair_edges,
                X_orig,
                y_orig,
                X_norm,
                grouper,
            )
            all_cross_terms.extend(terms)
            total_cross_energy += energy

        return all_cross_terms, total_cross_energy

    def _train_focused_cross_rotor(self, group_a, group_b, pair_edges, X_orig, y_orig, X_norm, grouper):
        """Train a joint rotor for a specific group pair.

        Builds a focused 2-group sub-mother algebra and biases the rotor
        toward the strongest cross-edge plane.

        Returns:
            (list[RotorTerm], float): Cross-terms and cross-energy.
        """
        from models.sr.errors import AlgebraConstructionError

        try:
            sub_mother, _ = grouper.build_mother_algebra([group_a, group_b])
        except (AlgebraConstructionError, RuntimeError, ValueError) as e:
            logger.warning(f"Sub-mother algebra failed for groups: {e}")
            return [], 0.0

        # Embed both groups into sub-mother
        N = X_orig.shape[0]
        mother_mvs = []
        for g in [group_a, group_b]:
            X_g = X_orig[:, g.var_indices]
            X_g_std = standardize(X_g)
            n_g = g.algebra.n
            if X_g_std.shape[1] < n_g:
                X_g_std = np.column_stack([X_g_std, np.zeros((N, n_g - X_g_std.shape[1]))])
            elif X_g_std.shape[1] > n_g:
                X_g_std = X_g_std[:, :n_g]

            local_mv = g.algebra.embed_vector(torch.tensor(X_g_std, dtype=torch.float32, device=self.device))
            mother_mv = grouper.inject_to_mother(local_mv, g, sub_mother)
            mother_mvs.append(mother_mv)

        # Check GP grade-2 energy between the two groups
        gp = sub_mother.geometric_product(mother_mvs[0], mother_mvs[1])
        bv_mask = sub_mother.grade_masks[2]
        if bv_mask.device != gp.device:
            bv_mask = bv_mask.to(gp.device)
        cross_bv = gp[..., bv_mask]
        cross_energy = cross_bv.pow(2).mean().item()

        if cross_energy < self.mother_cross_threshold:
            return [], cross_energy

        logger.info(
            f"  Cross-energy {cross_energy:.4f} for groups "
            f"{group_a.var_names} x {group_b.var_names}, training focused rotor"
        )

        try:
            n_combined = len(group_a.var_indices) + len(group_b.var_indices)
            auto = SRGBN.auto_config(N, n_combined, sub_mother.dim)
            model = SRGBN.single_rotor(
                sub_mother,
                n_combined,
                channels=max(auto["channels"], 8),
            )
            model = model.to(self.device)

            # Bias rotor toward strongest cross-edge plane
            if pair_edges:
                strongest = pair_edges[0]
                # Map global var indices to sub-mother local indices
                combined_indices = group_a.var_indices + group_b.var_indices
                g2l = {gi: li for li, gi in enumerate(combined_indices)}
                li = g2l.get(strongest.var_i)
                lj = g2l.get(strongest.var_j)
                if li is not None and lj is not None:
                    self._bias_rotor_to_plane(
                        model,
                        li,
                        lj,
                        strongest.edge_type,
                        sub_mother,
                    )

            residual_t = torch.tensor(
                y_orig,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
            res_mean = residual_t.mean()
            res_std = residual_t.std().clamp(min=1e-8)
            residual_norm = (residual_t - res_mean) / res_std

            # Build input from combined group variables
            combined_X = np.column_stack(
                [
                    X_orig[:, group_a.var_indices],
                    X_orig[:, group_b.var_indices],
                ]
            )
            X_combined_norm = standardize(torch.tensor(combined_X, dtype=torch.float32, device=self.device))

            model, _, _ = self._train_stage(
                model,
                X_combined_norm,
                residual_norm,
                sub_mother,
            )

            translator = RotorTranslator(sub_mother)
            cross_terms = translator.translate_direct(model)
            if not cross_terms:
                cross_terms = translator.translate(model)
            return cross_terms, cross_energy

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Focused cross-rotor training failed: {e}")
            return [], cross_energy

    def _mother_algebra_legacy(self, groups, X_orig, y_orig, X_norm, grouper):
        """Legacy brute-force cross-term discovery (fallback when no graph)."""
        from models.sr.errors import AlgebraConstructionError

        try:
            mother_alg, offsets = grouper.build_mother_algebra(groups)
        except (AlgebraConstructionError, RuntimeError, ValueError) as e:
            logger.warning(f"Mother algebra construction failed: {e}")
            return [], 0.0

        N = X_orig.shape[0]
        mother_mvs = []
        for g in groups:
            X_g = X_orig[:, g.var_indices]
            X_g_std = standardize(X_g)
            n_g = g.algebra.n
            if X_g_std.shape[1] < n_g:
                X_g_std = np.column_stack([X_g_std, np.zeros((N, n_g - X_g_std.shape[1]))])
            elif X_g_std.shape[1] > n_g:
                X_g_std = X_g_std[:, :n_g]

            local_mv = g.algebra.embed_vector(torch.tensor(X_g_std, dtype=torch.float32, device=self.device))
            mother_mv = grouper.inject_to_mother(local_mv, g, mother_alg)
            mother_mvs.append(mother_mv)

        cross_energy = 0.0
        if len(mother_mvs) >= 2:
            gp = mother_alg.geometric_product(mother_mvs[0], mother_mvs[1])
            bv_mask = mother_alg.grade_masks[2]
            if bv_mask.device != gp.device:
                bv_mask = bv_mask.to(gp.device)
            cross_bv = gp[..., bv_mask]
            cross_energy = cross_bv.pow(2).mean().item()

        if cross_energy < self.mother_cross_threshold:
            logger.info(f"  Cross-energy {cross_energy:.4f} < threshold, no cross-terms")
            return [], cross_energy

        logger.info(f"  Cross-energy {cross_energy:.4f} detected, training joint rotor")
        try:
            auto = SRGBN.auto_config(X_orig.shape[0], self.in_features, mother_alg.dim)
            model = SRGBN.single_rotor(mother_alg, self.in_features, channels=max(auto["channels"], 8))
            model = model.to(self.device)

            residual_t = torch.tensor(
                y_orig,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(-1)
            res_mean = residual_t.mean()
            res_std = residual_t.std().clamp(min=1e-8)
            residual_norm = (residual_t - res_mean) / res_std

            X_norm_std = standardize(X_norm) if isinstance(X_norm, torch.Tensor) else X_norm
            model, _, _ = self._train_stage(
                model,
                X_norm_std,
                residual_norm,
                mother_alg,
            )

            translator = RotorTranslator(mother_alg)
            cross_terms = translator.translate_direct(model)
            if not cross_terms:
                cross_terms = translator.translate(model)
            return cross_terms, cross_energy

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Joint rotor training failed: {e}")
            return [], cross_energy
