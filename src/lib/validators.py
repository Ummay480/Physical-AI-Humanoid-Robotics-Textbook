"""
Validation utilities for skills and subagents framework.

Provides JSON Schema validation, parameter validation, and data integrity checks.
"""

from typing import Dict, Any, Optional, List, Union
import re


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class SchemaValidator:
    """
    JSON Schema-based validator for skill parameters and data structures.

    Provides validation against JSON Schema specifications without
    requiring external dependencies.
    """

    BASIC_TYPES = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
        "null": type(None),
    }

    @classmethod
    def validate(cls, data: Any, schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate data against a JSON Schema.

        Args:
            data: Data to validate
            schema: JSON Schema specification

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            cls._validate_value(data, schema, path="$")
            return True, None
        except ValidationError as e:
            return False, f"{e.field}: {e.message}" if e.field else e.message

    @classmethod
    def _validate_value(cls, value: Any, schema: Dict[str, Any], path: str) -> None:
        """
        Internal validation method with path tracking.

        Args:
            value: Value to validate
            schema: Schema specification
            path: JSON path for error reporting

        Raises:
            ValidationError: If validation fails
        """
        # Type validation
        if "type" in schema:
            cls._validate_type(value, schema["type"], path)

        # Enum validation
        if "enum" in schema:
            if value not in schema["enum"]:
                raise ValidationError(
                    f"Value must be one of {schema['enum']}, got {value}",
                    field=path
                )

        # String validations
        if isinstance(value, str):
            cls._validate_string(value, schema, path)

        # Number validations
        if isinstance(value, (int, float)):
            cls._validate_number(value, schema, path)

        # Array validations
        if isinstance(value, list):
            cls._validate_array(value, schema, path)

        # Object validations
        if isinstance(value, dict):
            cls._validate_object(value, schema, path)

    @classmethod
    def _validate_type(cls, value: Any, expected_type: Union[str, List[str]], path: str) -> None:
        """Validate value type."""
        types_to_check = [expected_type] if isinstance(expected_type, str) else expected_type

        for type_name in types_to_check:
            if type_name not in cls.BASIC_TYPES:
                continue

            expected_python_type = cls.BASIC_TYPES[type_name]
            if isinstance(value, expected_python_type):
                return

        raise ValidationError(
            f"Expected type {expected_type}, got {type(value).__name__}",
            field=path
        )

    @classmethod
    def _validate_string(cls, value: str, schema: Dict[str, Any], path: str) -> None:
        """Validate string constraints."""
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValidationError(
                f"String length {len(value)} is less than minimum {schema['minLength']}",
                field=path
            )

        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValidationError(
                f"String length {len(value)} exceeds maximum {schema['maxLength']}",
                field=path
            )

        if "pattern" in schema:
            pattern = schema["pattern"]
            if not re.match(pattern, value):
                raise ValidationError(
                    f"String does not match pattern: {pattern}",
                    field=path
                )

    @classmethod
    def _validate_number(cls, value: Union[int, float], schema: Dict[str, Any], path: str) -> None:
        """Validate number constraints."""
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(
                f"Value {value} is less than minimum {schema['minimum']}",
                field=path
            )

        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(
                f"Value {value} exceeds maximum {schema['maximum']}",
                field=path
            )

        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValidationError(
                f"Value {value} must be greater than {schema['exclusiveMinimum']}",
                field=path
            )

        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise ValidationError(
                f"Value {value} must be less than {schema['exclusiveMaximum']}",
                field=path
            )

        if "multipleOf" in schema and value % schema["multipleOf"] != 0:
            raise ValidationError(
                f"Value {value} is not a multiple of {schema['multipleOf']}",
                field=path
            )

    @classmethod
    def _validate_array(cls, value: list, schema: Dict[str, Any], path: str) -> None:
        """Validate array constraints."""
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValidationError(
                f"Array length {len(value)} is less than minimum {schema['minItems']}",
                field=path
            )

        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValidationError(
                f"Array length {len(value)} exceeds maximum {schema['maxItems']}",
                field=path
            )

        if "uniqueItems" in schema and schema["uniqueItems"]:
            if len(value) != len(set(str(v) for v in value)):
                raise ValidationError("Array items must be unique", field=path)

        # Validate items against schema
        if "items" in schema:
            item_schema = schema["items"]
            for idx, item in enumerate(value):
                cls._validate_value(item, item_schema, f"{path}[{idx}]")

    @classmethod
    def _validate_object(cls, value: dict, schema: Dict[str, Any], path: str) -> None:
        """Validate object constraints."""
        # Required properties
        if "required" in schema:
            for prop in schema["required"]:
                if prop not in value:
                    raise ValidationError(
                        f"Missing required property: {prop}",
                        field=f"{path}.{prop}"
                    )

        # Properties validation
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in value:
                    cls._validate_value(value[prop_name], prop_schema, f"{path}.{prop_name}")

        # Additional properties
        if "additionalProperties" in schema and schema["additionalProperties"] is False:
            allowed_props = set(schema.get("properties", {}).keys())
            actual_props = set(value.keys())
            extra_props = actual_props - allowed_props

            if extra_props:
                raise ValidationError(
                    f"Additional properties not allowed: {', '.join(extra_props)}",
                    field=path
                )


def validate_skill_name(name: str) -> tuple[bool, Optional[str]]:
    """
    Validate skill name format.

    Rules:
    - Must be lowercase alphanumeric with hyphens
    - Must start with letter
    - Length 3-50 characters
    - No consecutive hyphens

    Args:
        name: Skill name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not name:
        return False, "Skill name cannot be empty"

    if len(name) < 3:
        return False, "Skill name must be at least 3 characters"

    if len(name) > 50:
        return False, "Skill name must not exceed 50 characters"

    if not re.match(r"^[a-z][a-z0-9-]*$", name):
        return False, "Skill name must be lowercase alphanumeric with hyphens, starting with a letter"

    if "--" in name:
        return False, "Skill name cannot contain consecutive hyphens"

    if name.endswith("-"):
        return False, "Skill name cannot end with hyphen"

    return True, None


def validate_semver(version: str) -> tuple[bool, Optional[str]]:
    """
    Validate semantic version format (major.minor.patch).

    Args:
        version: Version string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not version:
        return False, "Version cannot be empty"

    parts = version.split(".")
    if len(parts) != 3:
        return False, "Version must have format major.minor.patch (e.g., 1.0.0)"

    for part in parts:
        if not part.isdigit():
            return False, f"Version component '{part}' must be numeric"

        if int(part) < 0:
            return False, "Version components must be non-negative"

    return True, None


def validate_topic_name(topic: str) -> tuple[bool, Optional[str]]:
    """
    Validate message topic name.

    Rules:
    - Alphanumeric with forward slashes and underscores
    - Must start with forward slash
    - No consecutive slashes
    - Length 2-100 characters

    Args:
        topic: Topic name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not topic:
        return False, "Topic name cannot be empty"

    if len(topic) < 2:
        return False, "Topic name must be at least 2 characters"

    if len(topic) > 100:
        return False, "Topic name must not exceed 100 characters"

    if not topic.startswith("/"):
        return False, "Topic name must start with forward slash"

    if "//" in topic:
        return False, "Topic name cannot contain consecutive slashes"

    if not re.match(r"^/[a-zA-Z0-9/_-]+$", topic):
        return False, "Topic name must contain only alphanumeric, slashes, underscores, and hyphens"

    return True, None


def validate_parameters(
    parameters: Dict[str, Any],
    schema_list: List[Dict[str, Any]]
) -> tuple[bool, Optional[str]]:
    """
    Validate parameters against a list of parameter schemas.

    Args:
        parameters: Parameters to validate
        schema_list: List of parameter schema specifications

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required parameters
    for param_schema in schema_list:
        param_name = param_schema.get("name")
        is_required = param_schema.get("required", False)

        if is_required and param_name not in parameters:
            return False, f"Missing required parameter: {param_name}"

    # Validate each provided parameter
    schema_by_name = {s["name"]: s for s in schema_list}

    for param_name, param_value in parameters.items():
        if param_name not in schema_by_name:
            return False, f"Unknown parameter: {param_name}"

        param_schema = schema_by_name[param_name]
        validation_rules = param_schema.get("validation_rules", {})

        if validation_rules:
            is_valid, error = SchemaValidator.validate(param_value, validation_rules)
            if not is_valid:
                return False, f"Parameter '{param_name}' validation failed: {error}"

    return True, None
