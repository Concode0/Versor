# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""GTM data — re-exports ARC dataset loaders."""

from .arc import ToyARCDataset, ARCDataset, collate_arc, get_arc_loaders

__all__ = [
    "ToyARCDataset",
    "ARCDataset",
    "collate_arc",
    "get_arc_loaders",
]
