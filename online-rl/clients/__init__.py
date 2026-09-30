"""Self-contained GUI env clients (session-protocol clients copied from
env_infra, plus the lease-based adapter the rollout uses).
"""
from .session_env_client import SessionGuiEnvClient

__all__ = ["SessionGuiEnvClient"]
