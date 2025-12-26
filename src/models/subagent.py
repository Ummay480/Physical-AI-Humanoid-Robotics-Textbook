"""
Subagent models and state definitions.

This module defines the Subagent entity and its associated state machine.
Subagents are autonomous execution contexts that run specific skills.
"""

from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID, uuid4


class AgentState(str, Enum):
    """
    Lifecycle states for subagents.

    State transitions:
        IDLE → RUNNING → COMPLETED
                ↓           ↓
              PAUSED → RUNNING
                ↓
              FAILED

    States:
        IDLE: Agent created but not yet executing
        RUNNING: Agent actively executing its skill
        PAUSED: Agent temporarily suspended (can be resumed)
        COMPLETED: Agent finished successfully
        FAILED: Agent encountered an error and terminated
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

    def can_transition_to(self, target: "AgentState") -> bool:
        """
        Check if transition to target state is valid.

        Args:
            target: Target state to transition to

        Returns:
            True if transition is valid, False otherwise
        """
        valid_transitions = {
            AgentState.IDLE: {AgentState.RUNNING, AgentState.FAILED},
            AgentState.RUNNING: {AgentState.PAUSED, AgentState.COMPLETED, AgentState.FAILED},
            AgentState.PAUSED: {AgentState.RUNNING, AgentState.FAILED},
            AgentState.COMPLETED: set(),  # Terminal state
            AgentState.FAILED: set(),  # Terminal state
        }

        return target in valid_transitions.get(self, set())

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (no further transitions)."""
        return self in {AgentState.COMPLETED, AgentState.FAILED}

    @property
    def is_active(self) -> bool:
        """Check if agent is actively executing."""
        return self == AgentState.RUNNING


class SubagentMetadata(BaseModel):
    """Metadata about a subagent's configuration and execution."""

    agent_id: UUID = Field(default_factory=uuid4, description="Unique agent identifier")
    skill_name: str = Field(..., description="Name of the skill this agent executes")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Agent creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Execution start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Execution completion timestamp")

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class SubagentStatus(BaseModel):
    """Current status and execution state of a subagent."""

    state: AgentState = Field(default=AgentState.IDLE, description="Current lifecycle state")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Execution progress (0.0 to 1.0)")
    message: Optional[str] = Field(None, description="Status message or error description")
    error_code: Optional[str] = Field(None, description="Error code if state is FAILED")

    class Config:
        use_enum_values = True


class SubagentResources(BaseModel):
    """Resource allocation and usage for a subagent."""

    max_memory_mb: float = Field(default=50.0, description="Maximum memory allocation (MB)")
    current_memory_mb: float = Field(default=0.0, description="Current memory usage (MB)")
    cpu_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="CPU usage percentage")
    timeout_seconds: Optional[float] = Field(None, description="Execution timeout in seconds")


class Subagent(BaseModel):
    """
    Subagent entity representing an autonomous execution context.

    A subagent executes a specific skill independently, with its own
    lifecycle, resources, and execution logs.

    Attributes:
        metadata: Agent identification and timestamps
        status: Current state and execution progress
        resources: Resource allocation and usage
        parameters: Skill execution parameters
        results: Execution results (populated on completion)
        logs: Execution log entries
    """

    metadata: SubagentMetadata = Field(default_factory=SubagentMetadata)
    status: SubagentStatus = Field(default_factory=SubagentStatus)
    resources: SubagentResources = Field(default_factory=SubagentResources)
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Skill execution parameters")
    results: Optional[Dict[str, Any]] = Field(None, description="Execution results")
    logs: list[str] = Field(default_factory=list, description="Execution log entries")

    @property
    def agent_id(self) -> UUID:
        """Get the agent's unique identifier."""
        return self.metadata.agent_id

    @property
    def state(self) -> AgentState:
        """Get the agent's current state."""
        return self.status.state

    def transition_to(self, new_state: AgentState, message: Optional[str] = None) -> None:
        """
        Transition agent to a new state.

        Args:
            new_state: Target state
            message: Optional status message

        Raises:
            ValueError: If transition is invalid
        """
        if not self.status.state.can_transition_to(new_state):
            raise ValueError(
                f"Invalid state transition: {self.status.state} → {new_state}"
            )

        self.status.state = new_state
        if message:
            self.status.message = message

        # Update timestamps
        if new_state == AgentState.RUNNING and not self.metadata.started_at:
            self.metadata.started_at = datetime.utcnow()
        elif new_state.is_terminal:
            self.metadata.completed_at = datetime.utcnow()

    def add_log(self, message: str) -> None:
        """Add a log entry with timestamp."""
        timestamp = datetime.utcnow().isoformat()
        self.logs.append(f"[{timestamp}] {message}")

    def set_results(self, results: Dict[str, Any]) -> None:
        """Set execution results."""
        self.results = results

    class Config:
        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
            AgentState: lambda v: v.value,
        }
