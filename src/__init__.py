"""
Skills and Subagents Framework

A modular framework for robotics skill execution and subagent coordination.
Enables concurrent task execution, skill composition, and inter-agent communication.

Modules:
    models: Core entity definitions (Skill, Subagent, Message)
    services: Business logic (SkillRegistry, AgentManager, Executor, Messenger)
    integrations: External system integrations (ROS 2)
    cli: Command-line interface tools
    lib: Utility functions and validators
"""

__version__ = "0.1.0"
__author__ = "AI Humanoid Robotics"

# Core exports (will be populated as modules are implemented)
__all__ = [
    "models",
    "services",
    "integrations",
    "cli",
    "lib",
]
