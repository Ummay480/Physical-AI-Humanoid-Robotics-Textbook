"""
Skill models and execution status definitions.

This module defines the Skill entity, execution parameters, and status tracking.
Skills are reusable capabilities with defined inputs, outputs, and execution logic.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ExecutionStatus(str, Enum):
    """
    Status of skill execution.

    States:
        PENDING: Execution queued but not yet started
        RUNNING: Currently executing
        SUCCESS: Completed successfully
        FAILED: Execution failed with error
        TIMEOUT: Execution exceeded time limit
        CANCELLED: Execution was cancelled by user/system
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal status (execution finished)."""
        return self in {
            ExecutionStatus.SUCCESS,
            ExecutionStatus.FAILED,
            ExecutionStatus.TIMEOUT,
            ExecutionStatus.CANCELLED,
        }

    @property
    def is_successful(self) -> bool:
        """Check if execution succeeded."""
        return self == ExecutionStatus.SUCCESS

    @property
    def is_active(self) -> bool:
        """Check if execution is currently active."""
        return self in {ExecutionStatus.PENDING, ExecutionStatus.RUNNING}


class RetryPolicy(BaseModel):
    """Retry policy for failed skill executions."""

    max_attempts: int = Field(default=1, ge=1, description="Maximum retry attempts")
    backoff_multiplier: float = Field(default=1.0, ge=1.0, description="Exponential backoff multiplier")
    initial_delay_seconds: float = Field(default=1.0, ge=0.0, description="Initial retry delay in seconds")

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given retry attempt.

        Args:
            attempt: Retry attempt number (1-indexed)

        Returns:
            Delay in seconds before retry
        """
        if attempt < 1:
            return 0.0
        return self.initial_delay_seconds * (self.backoff_multiplier ** (attempt - 1))


class ParameterSchema(BaseModel):
    """Schema definition for a skill parameter."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, number, boolean, object, array)")
    required: bool = Field(default=False, description="Whether parameter is required")
    description: Optional[str] = Field(None, description="Parameter description")
    default: Optional[Any] = Field(None, description="Default value if not provided")
    validation_rules: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema validation rules")


class SkillMetadata(BaseModel):
    """Metadata about a skill definition."""

    name: str = Field(..., description="Unique skill name", min_length=1)
    version: str = Field(default="1.0.0", description="Skill version (semver)")
    description: str = Field(..., description="Skill description and activation criteria")
    author: Optional[str] = Field(None, description="Skill author/creator")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Skill creation timestamp")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    @validator("version")
    def validate_semver(cls, v):
        """Validate semantic versioning format."""
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must follow semver format (e.g., 1.0.0)")
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class SkillConfiguration(BaseModel):
    """Configuration for skill execution behavior."""

    timeout_seconds: Optional[float] = Field(None, ge=0.0, description="Execution timeout in seconds")
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy, description="Retry policy for failures")
    parallel_execution: bool = Field(default=False, description="Whether skill can execute in parallel")
    resource_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource requirements (memory, CPU, etc.)"
    )


class SkillInterface(BaseModel):
    """Input/output interface definition for a skill."""

    parameters: List[ParameterSchema] = Field(default_factory=list, description="Input parameters")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Output schema")
    dependencies: List[str] = Field(default_factory=list, description="Required skill dependencies")

    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate input parameters against schema.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param_schema in self.parameters:
            if param_schema.required and param_schema.name not in params:
                return False, f"Missing required parameter: {param_schema.name}"

            if param_schema.name in params:
                # Basic type checking (can be extended with JSON Schema validator)
                value = params[param_schema.name]
                expected_type = param_schema.type

                type_map = {
                    "string": str,
                    "number": (int, float),
                    "boolean": bool,
                    "object": dict,
                    "array": list,
                }

                if expected_type in type_map:
                    expected_python_type = type_map[expected_type]
                    if not isinstance(value, expected_python_type):
                        return False, f"Parameter '{param_schema.name}' must be type {expected_type}"

        return True, None


class Skill(BaseModel):
    """
    Skill entity representing a reusable capability.

    A skill encapsulates execution logic with defined inputs, outputs,
    and configuration. Skills can be composed to create complex behaviors.

    Attributes:
        metadata: Skill identification and description
        interface: Input/output parameters and dependencies
        config: Execution configuration and policies
        implementation: Execution logic (Python callable or reference)
    """

    metadata: SkillMetadata
    interface: SkillInterface = Field(default_factory=SkillInterface)
    config: SkillConfiguration = Field(default_factory=SkillConfiguration)
    implementation: Optional[str] = Field(
        None,
        description="Implementation reference (module path or callable name)"
    )

    @property
    def name(self) -> str:
        """Get skill name."""
        return self.metadata.name

    @property
    def version(self) -> str:
        """Get skill version."""
        return self.metadata.version

    def can_execute_with(self, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Check if skill can execute with given parameters.

        Args:
            parameters: Proposed execution parameters

        Returns:
            Tuple of (can_execute, error_message)
        """
        return self.interface.validate_parameters(parameters)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ExecutionStatus: lambda v: v.value,
        }


class SkillExecution(BaseModel):
    """
    Record of a skill execution instance.

    Tracks the execution lifecycle, parameters, results, and status.
    """

    skill_name: str = Field(..., description="Name of executed skill")
    skill_version: str = Field(..., description="Version of executed skill")
    status: ExecutionStatus = Field(default=ExecutionStatus.PENDING, description="Current execution status")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Execution parameters")
    results: Optional[Dict[str, Any]] = Field(None, description="Execution results")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    attempt_count: int = Field(default=1, ge=1, description="Retry attempt number")

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def mark_started(self) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self, results: Dict[str, Any]) -> None:
        """Mark execution as successfully completed."""
        self.status = ExecutionStatus.SUCCESS
        self.results = results
        self.completed_at = datetime.utcnow()

    def mark_failed(self, error_message: str) -> None:
        """Mark execution as failed."""
        self.status = ExecutionStatus.FAILED
        self.error_message = error_message
        self.completed_at = datetime.utcnow()

    def mark_timeout(self) -> None:
        """Mark execution as timed out."""
        self.status = ExecutionStatus.TIMEOUT
        self.error_message = "Execution exceeded timeout limit"
        self.completed_at = datetime.utcnow()

    def mark_cancelled(self) -> None:
        """Mark execution as cancelled."""
        self.status = ExecutionStatus.CANCELLED
        self.error_message = "Execution was cancelled"
        self.completed_at = datetime.utcnow()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ExecutionStatus: lambda v: v.value,
        }
