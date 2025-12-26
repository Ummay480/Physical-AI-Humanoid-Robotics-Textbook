"""
Skill execution engine with async support, timeout handling, and retry logic.

Provides the core execution engine for running skills with parameter validation,
resource management, and result tracking.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime
import logging

from ..models.skill import Skill, SkillExecution, ExecutionStatus
from ..services.skill_registry import SkillRegistry, SkillNotFoundError
from ..lib.utils import get_current_timestamp

logger = logging.getLogger(__name__)


class SkillExecutionError(Exception):
    """Base exception for skill execution errors."""

    pass


class SkillTimeoutError(SkillExecutionError):
    """Raised when skill execution exceeds timeout."""

    def __init__(self, skill_name: str, timeout: float):
        self.skill_name = skill_name
        self.timeout = timeout
        super().__init__(f"Skill '{skill_name}' exceeded timeout of {timeout}s")


class SkillParameterError(SkillExecutionError):
    """Raised when skill parameters are invalid."""

    def __init__(self, message: str):
        super().__init__(f"Invalid parameters: {message}")


# Type alias for skill implementation functions
SkillImplementation = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


class SkillExecutor:
    """
    Execution engine for running skills.

    Handles parameter validation, timeout enforcement, retry logic,
    and execution tracking. Supports both sync and async skill implementations.
    """

    def __init__(self, registry: Optional[SkillRegistry] = None):
        """
        Initialize skill executor.

        Args:
            registry: Skill registry to use (creates new one if None)
        """
        self.registry = registry or SkillRegistry()
        self._implementations: Dict[str, SkillImplementation] = {}

        logger.info("Skill executor initialized")

    def register_implementation(
        self,
        skill_name: str,
        implementation: SkillImplementation,
    ) -> None:
        """
        Register an implementation function for a skill.

        Args:
            skill_name: Name of the skill
            implementation: Async function that implements the skill logic

        Example:
            async def my_skill_impl(params: Dict[str, Any]) -> Dict[str, Any]:
                # Skill logic here
                return {"result": "success"}

            executor.register_implementation("my-skill", my_skill_impl)
        """
        self._implementations[skill_name] = implementation
        logger.info(f"Registered implementation for skill: {skill_name}")

    async def execute(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        skill_version: Optional[str] = None,
    ) -> SkillExecution:
        """
        Execute a skill with given parameters.

        Args:
            skill_name: Name of skill to execute
            parameters: Execution parameters
            skill_version: Specific version to use (None = latest)

        Returns:
            SkillExecution record with results or error

        Raises:
            SkillNotFoundError: If skill not found in registry
            SkillParameterError: If parameters are invalid
        """
        # Get skill from registry
        skill = self.registry.get(skill_name, skill_version)

        # Create execution record
        execution = SkillExecution(
            skill_name=skill.name,
            skill_version=skill.version,
            parameters=parameters,
        )

        try:
            # Validate parameters
            can_execute, error = skill.can_execute_with(parameters)
            if not can_execute:
                raise SkillParameterError(error)

            # Execute with retry logic
            await self._execute_with_retry(skill, execution)

        except SkillTimeoutError as e:
            execution.mark_timeout()
            logger.error(f"Skill timeout: {skill_name} - {e}")

        except SkillParameterError as e:
            execution.mark_failed(str(e))
            logger.error(f"Parameter error: {skill_name} - {e}")

        except Exception as e:
            execution.mark_failed(f"Execution error: {str(e)}")
            logger.error(f"Skill execution failed: {skill_name} - {e}", exc_info=True)

        return execution

    async def _execute_with_retry(self, skill: Skill, execution: SkillExecution) -> None:
        """
        Execute skill with retry policy.

        Args:
            skill: Skill to execute
            execution: Execution record to update
        """
        retry_policy = skill.config.retry_policy
        last_error = None

        for attempt in range(1, retry_policy.max_attempts + 1):
            execution.attempt_count = attempt

            try:
                # Execute skill
                await self._execute_once(skill, execution)

                # Success - no need to retry
                return

            except SkillTimeoutError:
                # Timeout - don't retry, propagate immediately
                raise

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Skill execution attempt {attempt}/{retry_policy.max_attempts} failed: {e}"
                )

                # Calculate delay before next retry
                if attempt < retry_policy.max_attempts:
                    delay = retry_policy.get_delay(attempt)
                    logger.info(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise last_error

    async def _execute_once(self, skill: Skill, execution: SkillExecution) -> None:
        """
        Execute skill once (single attempt).

        Args:
            skill: Skill to execute
            execution: Execution record to update

        Raises:
            SkillTimeoutError: If execution exceeds timeout
            SkillExecutionError: If skill implementation not found
            Exception: Any exception from skill implementation
        """
        # Check if implementation is registered
        if skill.name not in self._implementations:
            raise SkillExecutionError(
                f"No implementation registered for skill: {skill.name}"
            )

        implementation = self._implementations[skill.name]

        # Mark as started
        execution.mark_started()
        logger.info(f"Executing skill: {skill.name} v{skill.version} (attempt {execution.attempt_count})")

        try:
            # Execute with timeout if specified
            if skill.config.timeout_seconds:
                results = await asyncio.wait_for(
                    implementation(execution.parameters),
                    timeout=skill.config.timeout_seconds,
                )
            else:
                results = await implementation(execution.parameters)

            # Mark as completed
            execution.mark_completed(results)
            logger.info(f"Skill completed: {skill.name} v{skill.version}")

        except asyncio.TimeoutError:
            raise SkillTimeoutError(skill.name, skill.config.timeout_seconds)

    def execute_sync(
        self,
        skill_name: str,
        parameters: Dict[str, Any],
        skill_version: Optional[str] = None,
    ) -> SkillExecution:
        """
        Synchronous wrapper for execute() method.

        Args:
            skill_name: Name of skill to execute
            parameters: Execution parameters
            skill_version: Specific version to use (None = latest)

        Returns:
            SkillExecution record with results or error

        Note:
            This creates a new event loop if one doesn't exist.
            Use execute() directly in async contexts.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute(skill_name, parameters, skill_version))

    async def execute_batch(
        self,
        executions: list[tuple[str, Dict[str, Any], Optional[str]]],
    ) -> list[SkillExecution]:
        """
        Execute multiple skills concurrently.

        Args:
            executions: List of (skill_name, parameters, version) tuples

        Returns:
            List of SkillExecution records

        Example:
            results = await executor.execute_batch([
                ("skill-1", {"param": "value"}, None),
                ("skill-2", {"param": "value"}, "1.0.0"),
            ])
        """
        tasks = [
            self.execute(skill_name, params, version)
            for skill_name, params, version in executions
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed executions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                skill_name, params, version = executions[i]
                execution = SkillExecution(
                    skill_name=skill_name,
                    skill_version=version or "unknown",
                    parameters=params,
                )
                execution.mark_failed(f"Batch execution error: {str(result)}")
                processed_results.append(execution)
            else:
                processed_results.append(result)

        return processed_results

    def get_execution_stats(self, executions: list[SkillExecution]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of executions.

        Args:
            executions: List of SkillExecution records

        Returns:
            Dictionary with execution statistics
        """
        if not executions:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "timeout": 0,
                "cancelled": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0,
            }

        total = len(executions)
        successful = sum(1 for e in executions if e.status == ExecutionStatus.SUCCESS)
        failed = sum(1 for e in executions if e.status == ExecutionStatus.FAILED)
        timeout = sum(1 for e in executions if e.status == ExecutionStatus.TIMEOUT)
        cancelled = sum(1 for e in executions if e.status == ExecutionStatus.CANCELLED)

        # Calculate average duration for completed executions
        durations = [e.duration_seconds for e in executions if e.duration_seconds is not None]
        avg_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "timeout": timeout,
            "cancelled": cancelled,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_duration": avg_duration,
        }
