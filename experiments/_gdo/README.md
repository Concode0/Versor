# Geometric Deterministic Optimizer (GDO)
### Idea Incubator: Exploring Topology-Aware Optimization

**A brief note from the author:**

This is not a formal documentation. To explain simply, the current state of GDO is a somewhat "overkill" combination of numerous geometric structures, assembled largely based on my raw geometric intuition.

True to the spirit of an **Idea Incubator**, this is a highly experimental space. Because it combines so many radical concepts at once, it requires an immense amount of hyperparameter tuning and the development of dynamic, automated parameter configurations. I expect this refinement process will take a considerable amount of time.

Therefore, if you are interested in exploring this codebase, **I strongly recommend taking it apart and utilizing it fragment by fragment**. Trying to digest the entire pipeline at once might be overwhelming. Examining individual modules (e.g., topology detection or dimensional lifting) on their own will likely be much easier to understand and apply to your own research.

Thank you so much for your interest!

---

## Core Components (For Fragmented Use)
If you are dissecting this module, here are the main geometric concepts you can extract:

* **Topology Detection (`topology.py`)**: Uses Morse theory concepts to read the loss surface (e.g., detecting saddle points or valleys).
* **Dimensional Lifting (`dimensional_lift.py`)**: Temporarily lifts parameter spaces to higher dimensions to create geometric shortcuts out of local minima.
* **Geometric Parameter Coloring (`parameter_groups.py`)**: Groups parameters topologically for efficient updates.

Feel free to extract and test any single component that catches your eye.

Thank you for your interest.