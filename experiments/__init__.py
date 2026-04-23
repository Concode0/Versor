# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Experiments: mathematical debuggers and idea incubators.

Two personalities live here, separated by filename prefix:

* ``dbg_*`` — **Mathematical Debugger**. Validates a known algebraic or
  physical identity (Lorentz invariance, Yang–Mills action, 
  rotor round-trip). Success criterion: residuals below a fixed
  tolerance. No SOTA chasing — these exist to surface regressions when
  core/layers break an invariant.

* ``inc_*`` — **Idea Incubator**. Explores a radical, not-yet-published
  geometric hypothesis with a learned model (STA trajectory, pendulum
  dynamics, lattice morphing, GDO). Success criterion: the hypothesis
  produces a reproducible phenomenon worth writing up.

When in doubt: *are you checking that a known identity holds?* → ``dbg_``.
*Are you probing whether a novel architectural idea does something useful?* → ``inc_``.

New files should start from a template:

* ``experiments/_templates/inc_template.py`` — runnable Cl(2,0) rotation
  regressor (the smallest complete incubator).
* ``experiments/_templates/dbg_template.py`` — runnable Cl(3,0) rotor
  sanity suite (the smallest complete debugger).

Universal helpers (seed, algebra factory, argparse additions, plot saver)
live in ``experiments/_lib.py`` and are entirely opt-in; they do not
dictate model, loss, or training-loop structure.
"""
