"""
Skill loader for parsing skill definitions from YAML/JSON files.

Loads skill definitions and converts them to Skill entities.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..models.skill import (
    Skill,
    SkillMetadata,
    SkillInterface,
    SkillConfiguration,
    ParameterSchema,
    RetryPolicy,
)
from ..lib.validators import validate_skill_name, validate_semver

logger = logging.getLogger(__name__)


class SkillLoadError(Exception):
    """Base exception for skill loading errors."""

    pass


class InvalidSkillDefinitionError(SkillLoadError):
    """Raised when skill definition is invalid."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(f"{field}: {message}" if field else message)


class SkillLoader:
    """
    Loads skill definitions from YAML or JSON files.

    Parses skill configuration files and creates Skill entities
    with validation.
    """

    @staticmethod
    def load_from_file(file_path: str) -> Skill:
        """
        Load a skill from a YAML or JSON file.

        Args:
            file_path: Path to skill definition file (.yaml, .yml, or .json)

        Returns:
            Loaded Skill instance

        Raises:
            SkillLoadError: If file cannot be read or parsed
            InvalidSkillDefinitionError: If skill definition is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise SkillLoadError(f"File not found: {file_path}")

        if not path.is_file():
            raise SkillLoadError(f"Not a file: {file_path}")

        # Determine format from extension
        suffix = path.suffix.lower()
        if suffix in [".yaml", ".yml"]:
            return SkillLoader._load_yaml(path)
        elif suffix == ".json":
            return SkillLoader._load_json(path)
        else:
            raise SkillLoadError(f"Unsupported file format: {suffix}")

    @staticmethod
    def load_from_directory(directory_path: str, recursive: bool = False) -> List[Skill]:
        """
        Load all skills from a directory.

        Args:
            directory_path: Path to directory containing skill files
            recursive: If True, search subdirectories recursively

        Returns:
            List of loaded Skill instances

        Raises:
            SkillLoadError: If directory cannot be read
        """
        path = Path(directory_path)

        if not path.exists():
            raise SkillLoadError(f"Directory not found: {directory_path}")

        if not path.is_dir():
            raise SkillLoadError(f"Not a directory: {directory_path}")

        skills = []
        pattern = "**/*" if recursive else "*"

        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in [".yaml", ".yml", ".json"]:
                try:
                    skill = SkillLoader.load_from_file(str(file_path))
                    skills.append(skill)
                    logger.info(f"Loaded skill from {file_path}: {skill.name}")
                except Exception as e:
                    logger.error(f"Failed to load skill from {file_path}: {e}")
                    # Continue loading other skills

        logger.info(f"Loaded {len(skills)} skills from {directory_path}")
        return skills

    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> Skill:
        """
        Load a skill from a dictionary.

        Args:
            data: Skill definition as dictionary

        Returns:
            Loaded Skill instance

        Raises:
            InvalidSkillDefinitionError: If definition is invalid
        """
        try:
            # Parse metadata
            metadata = SkillLoader._parse_metadata(data.get("metadata", {}))

            # Parse interface
            interface = SkillLoader._parse_interface(data.get("interface", {}))

            # Parse configuration
            config = SkillLoader._parse_configuration(data.get("config", {}))

            # Get implementation reference
            implementation = data.get("implementation")

            # Create skill
            skill = Skill(
                metadata=metadata,
                interface=interface,
                config=config,
                implementation=implementation,
            )

            return skill

        except Exception as e:
            raise InvalidSkillDefinitionError(f"Failed to parse skill: {e}")

    @staticmethod
    def _load_yaml(path: Path) -> Skill:
        """Load skill from YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                raise InvalidSkillDefinitionError("YAML must contain a dictionary")

            return SkillLoader.load_from_dict(data)

        except yaml.YAMLError as e:
            raise SkillLoadError(f"Invalid YAML: {e}")
        except Exception as e:
            raise SkillLoadError(f"Failed to load YAML: {e}")

    @staticmethod
    def _load_json(path: Path) -> Skill:
        """Load skill from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                raise InvalidSkillDefinitionError("JSON must contain an object")

            return SkillLoader.load_from_dict(data)

        except json.JSONDecodeError as e:
            raise SkillLoadError(f"Invalid JSON: {e}")
        except Exception as e:
            raise SkillLoadError(f"Failed to load JSON: {e}")

    @staticmethod
    def _parse_metadata(data: Dict[str, Any]) -> SkillMetadata:
        """Parse metadata section."""
        # Required fields
        name = data.get("name")
        if not name:
            raise InvalidSkillDefinitionError("Missing required field", field="metadata.name")

        description = data.get("description")
        if not description:
            raise InvalidSkillDefinitionError(
                "Missing required field", field="metadata.description"
            )

        # Validate name format
        is_valid, error = validate_skill_name(name)
        if not is_valid:
            raise InvalidSkillDefinitionError(error, field="metadata.name")

        # Optional fields
        version = data.get("version", "1.0.0")
        is_valid, error = validate_semver(version)
        if not is_valid:
            raise InvalidSkillDefinitionError(error, field="metadata.version")

        author = data.get("author")
        tags = data.get("tags", [])

        if not isinstance(tags, list):
            raise InvalidSkillDefinitionError("Must be a list", field="metadata.tags")

        return SkillMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            tags=tags,
        )

    @staticmethod
    def _parse_interface(data: Dict[str, Any]) -> SkillInterface:
        """Parse interface section."""
        # Parse parameters
        parameters_data = data.get("parameters", [])
        if not isinstance(parameters_data, list):
            raise InvalidSkillDefinitionError("Must be a list", field="interface.parameters")

        parameters = [SkillLoader._parse_parameter(p) for p in parameters_data]

        # Parse outputs
        outputs = data.get("outputs", {})
        if not isinstance(outputs, dict):
            raise InvalidSkillDefinitionError("Must be an object", field="interface.outputs")

        # Parse dependencies
        dependencies = data.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise InvalidSkillDefinitionError("Must be a list", field="interface.dependencies")

        return SkillInterface(
            parameters=parameters,
            outputs=outputs,
            dependencies=dependencies,
        )

    @staticmethod
    def _parse_parameter(data: Dict[str, Any]) -> ParameterSchema:
        """Parse a parameter schema."""
        name = data.get("name")
        if not name:
            raise InvalidSkillDefinitionError("Missing required field: name")

        param_type = data.get("type")
        if not param_type:
            raise InvalidSkillDefinitionError(f"Missing type for parameter: {name}")

        valid_types = ["string", "number", "integer", "boolean", "object", "array"]
        if param_type not in valid_types:
            raise InvalidSkillDefinitionError(
                f"Invalid type '{param_type}' for parameter {name}. Must be one of: {valid_types}"
            )

        return ParameterSchema(
            name=name,
            type=param_type,
            required=data.get("required", False),
            description=data.get("description"),
            default=data.get("default"),
            validation_rules=data.get("validation_rules", {}),
        )

    @staticmethod
    def _parse_configuration(data: Dict[str, Any]) -> SkillConfiguration:
        """Parse configuration section."""
        timeout = data.get("timeout_seconds")
        if timeout is not None and (not isinstance(timeout, (int, float)) or timeout <= 0):
            raise InvalidSkillDefinitionError(
                "Must be a positive number", field="config.timeout_seconds"
            )

        # Parse retry policy
        retry_data = data.get("retry_policy", {})
        retry_policy = SkillLoader._parse_retry_policy(retry_data)

        parallel = data.get("parallel_execution", False)
        if not isinstance(parallel, bool):
            raise InvalidSkillDefinitionError(
                "Must be boolean", field="config.parallel_execution"
            )

        resources = data.get("resource_requirements", {})
        if not isinstance(resources, dict):
            raise InvalidSkillDefinitionError(
                "Must be an object", field="config.resource_requirements"
            )

        return SkillConfiguration(
            timeout_seconds=timeout,
            retry_policy=retry_policy,
            parallel_execution=parallel,
            resource_requirements=resources,
        )

    @staticmethod
    def _parse_retry_policy(data: Dict[str, Any]) -> RetryPolicy:
        """Parse retry policy."""
        max_attempts = data.get("max_attempts", 1)
        if not isinstance(max_attempts, int) or max_attempts < 1:
            raise InvalidSkillDefinitionError(
                "Must be a positive integer", field="retry_policy.max_attempts"
            )

        backoff = data.get("backoff_multiplier", 1.0)
        if not isinstance(backoff, (int, float)) or backoff < 1.0:
            raise InvalidSkillDefinitionError(
                "Must be >= 1.0", field="retry_policy.backoff_multiplier"
            )

        initial_delay = data.get("initial_delay_seconds", 1.0)
        if not isinstance(initial_delay, (int, float)) or initial_delay < 0:
            raise InvalidSkillDefinitionError(
                "Must be non-negative", field="retry_policy.initial_delay_seconds"
            )

        return RetryPolicy(
            max_attempts=max_attempts,
            backoff_multiplier=backoff,
            initial_delay_seconds=initial_delay,
        )


def load_skill(file_path: str) -> Skill:
    """
    Convenience function to load a single skill.

    Args:
        file_path: Path to skill definition file

    Returns:
        Loaded Skill instance
    """
    return SkillLoader.load_from_file(file_path)


def load_skills(directory_path: str, recursive: bool = False) -> List[Skill]:
    """
    Convenience function to load multiple skills from a directory.

    Args:
        directory_path: Path to directory containing skill files
        recursive: If True, search subdirectories recursively

    Returns:
        List of loaded Skill instances
    """
    return SkillLoader.load_from_directory(directory_path, recursive)
