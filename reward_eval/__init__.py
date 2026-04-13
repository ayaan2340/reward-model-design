"""Reward model evaluation pipeline (manifest, inference, metrics).

Rollout HDF5 from ``robomimic.../run_trained_agent.py`` should store per-episode
``task_success``, ``done_mode``, and ``dones`` (see ``dataset_states_to_obs``).
``preprocess_manifest`` resolves labels via ``task_success``, then ``done_mode==0`` + ``dones``.
"""

__version__ = "0.1.0"

DEFAULT_SQUARE_INSTRUCTION = (
    "fit the square nut onto the square peg"
)
