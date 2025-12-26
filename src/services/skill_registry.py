"""
Skill registry service for managing skill definitions.

Provides centralized skill registration, lookup, and metadata management.
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging

from ..models.skill import Skill, SkillMetadata
from ..lib.validators import validate_skill_name, validate_semver

logger = logging.getLogger(__name__)


class SkillRegistryError(Exception):
    """Base exception for skill registry errors."""

    pass


class SkillAlreadyRegisteredError(SkillRegistryError):
    """Raised when attempting to register a skill that already exists."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        super().__init__(f"Skill '{name}' version '{version}' is already registered")


class SkillNotFoundError(SkillRegistryError):
    """Raised when a requested skill is not found in the registry."""

    def __init__(self, name: str, version: Optional[str] = None):
        self.name = name
        self.version = version
        msg = f"Skill '{name}'" + (f" version '{version}'" if version else "") + " not found"
        super().__init__(msg)


class CircularDependencyError(SkillRegistryError):
    """Raised when circular dependencies are detected in skills."""

    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


class SkillRegistry:
    """
    Central registry for managing skill definitions.

    The registry maintains all registered skills, handles versioning,
    and provides lookup capabilities by name, tag, or dependency.

    Thread-safe for read operations. Write operations (register/unregister)
    should be coordinated externally if used in multi-threaded contexts.
    """

    def __init__(self):
        """Initialize an empty skill registry."""
        # Primary storage: skills[name][version] = Skill
        self._skills: Dict[str, Dict[str, Skill]] = defaultdict(dict)

        # Tag index: tags[tag] = set of (name, version) tuples
        self._tags: Dict[str, Set[tuple[str, str]]] = defaultdict(set)

        # Dependency index: dependents[skill_name] = set of skills that depend on it
        self._dependents: Dict[str, Set[tuple[str, str]]] = defaultdict(set)

        logger.info("Skill registry initialized")

    def register(self, skill: Skill, allow_overwrite: bool = False) -> None:
        """
        Register a skill in the registry.

        Args:
            skill: Skill to register
            allow_overwrite: If True, allow overwriting existing skill with same name/version

        Raises:
            SkillAlreadyRegisteredError: If skill already exists and overwrite not allowed
            ValueError: If skill name or version is invalid
            CircularDependencyError: If registering creates circular dependencies
        """
        # Validate skill name and version
        is_valid, error = validate_skill_name(skill.name)
        if not is_valid:
            raise ValueError(f"Invalid skill name: {error}")

        is_valid, error = validate_semver(skill.version)
        if not is_valid:
            raise ValueError(f"Invalid skill version: {error}")

        # Check if already registered
        if skill.name in self._skills and skill.version in self._skills[skill.name]:
            if not allow_overwrite:
                raise SkillAlreadyRegisteredError(skill.name, skill.version)
            logger.warning(f"Overwriting existing skill: {skill.name} v{skill.version}")

        # Check for circular dependencies
        self._check_circular_dependencies(skill)

        # Register the skill
        self._skills[skill.name][skill.version] = skill

        # Index tags
        for tag in skill.metadata.tags:
            self._tags[tag].add((skill.name, skill.version))

        # Index dependencies
        for dep_name in skill.interface.dependencies:
            self._dependents[dep_name].add((skill.name, skill.version))

        logger.info(f"Registered skill: {skill.name} v{skill.version}")

    def unregister(self, name: str, version: str) -> Skill:
        """
        Unregister a skill from the registry.

        Args:
            name: Skill name
            version: Skill version

        Returns:
            The unregistered Skill

        Raises:
            SkillNotFoundError: If skill not found
            SkillRegistryError: If other skills depend on this skill
        """
        skill = self.get(name, version)

        # Check if any skills depend on this one
        if name in self._dependents and self._dependents[name]:
            dependents = [f"{n} v{v}" for n, v in self._dependents[name]]
            raise SkillRegistryError(
                f"Cannot unregister {name} v{version}: required by {', '.join(dependents)}"
            )

        # Remove from primary storage
        del self._skills[name][version]
        if not self._skills[name]:  # No more versions
            del self._skills[name]

        # Remove from tag index
        for tag in skill.metadata.tags:
            self._tags[tag].discard((name, version))
            if not self._tags[tag]:
                del self._tags[tag]

        # Remove from dependency index
        for dep_name in skill.interface.dependencies:
            self._dependents[dep_name].discard((name, version))
            if not self._dependents[dep_name]:
                del self._dependents[dep_name]

        logger.info(f"Unregistered skill: {name} v{version}")
        return skill

    def get(self, name: str, version: Optional[str] = None) -> Skill:
        """
        Get a skill by name and optional version.

        Args:
            name: Skill name
            version: Skill version (if None, returns latest version)

        Returns:
            The requested Skill

        Raises:
            SkillNotFoundError: If skill not found
        """
        if name not in self._skills:
            raise SkillNotFoundError(name)

        if version is None:
            # Return latest version (highest semver)
            version = self._get_latest_version(name)

        if version not in self._skills[name]:
            raise SkillNotFoundError(name, version)

        return self._skills[name][version]

    def has(self, name: str, version: Optional[str] = None) -> bool:
        """
        Check if a skill exists in the registry.

        Args:
            name: Skill name
            version: Skill version (if None, checks if any version exists)

        Returns:
            True if skill exists, False otherwise
        """
        if name not in self._skills:
            return False

        if version is None:
            return True  # At least one version exists

        return version in self._skills[name]

    def list_all(self) -> List[Skill]:
        """
        Get all registered skills.

        Returns:
            List of all skills in the registry
        """
        skills = []
        for versions in self._skills.values():
            skills.extend(versions.values())
        return skills

    def list_by_tag(self, tag: str) -> List[Skill]:
        """
        Get all skills with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of skills with the tag
        """
        if tag not in self._tags:
            return []

        skills = []
        for name, version in self._tags[tag]:
            skills.append(self._skills[name][version])
        return skills

    def list_versions(self, name: str) -> List[str]:
        """
        Get all registered versions of a skill.

        Args:
            name: Skill name

        Returns:
            List of version strings, sorted

        Raises:
            SkillNotFoundError: If skill name not found
        """
        if name not in self._skills:
            raise SkillNotFoundError(name)

        versions = list(self._skills[name].keys())
        return sorted(versions, key=self._version_sort_key)

    def get_dependencies(self, name: str, version: Optional[str] = None) -> List[str]:
        """
        Get the dependencies of a skill.

        Args:
            name: Skill name
            version: Skill version (if None, uses latest)

        Returns:
            List of dependency skill names

        Raises:
            SkillNotFoundError: If skill not found
        """
        skill = self.get(name, version)
        return skill.interface.dependencies.copy()

    def get_dependents(self, name: str) -> List[tuple[str, str]]:
        """
        Get all skills that depend on a given skill.

        Args:
            name: Skill name

        Returns:
            List of (skill_name, version) tuples that depend on this skill
        """
        if name not in self._dependents:
            return []

        return list(self._dependents[name])

    def clear(self) -> None:
        """Remove all skills from the registry."""
        self._skills.clear()
        self._tags.clear()
        self._dependents.clear()
        logger.info("Skill registry cleared")

    def count(self) -> int:
        """
        Get total number of registered skills (all versions).

        Returns:
            Total skill count
        """
        return sum(len(versions) for versions in self._skills.values())

    def _get_latest_version(self, name: str) -> str:
        """Get the latest version of a skill."""
        versions = self.list_versions(name)
        return versions[-1]  # Sorted, so last is latest

    def _version_sort_key(self, version: str) -> tuple:
        """Convert version string to sortable tuple."""
        try:
            major, minor, patch = version.split(".")
            return (int(major), int(minor), int(patch))
        except (ValueError, AttributeError):
            return (0, 0, 0)

    def _check_circular_dependencies(self, skill: Skill) -> None:
        """
        Check if registering this skill would create circular dependencies.

        Args:
            skill: Skill to check

        Raises:
            CircularDependencyError: If circular dependency detected
        """
        for dep_name in skill.interface.dependencies:
            if not self.has(dep_name):
                continue  # Dependency not registered yet, no cycle possible

            # Check if dependency (or its dependencies) depend on this skill
            cycle = self._find_cycle(dep_name, skill.name, set())
            if cycle:
                raise CircularDependencyError([skill.name] + cycle)

    def _find_cycle(self, current: str, target: str, visited: Set[str]) -> Optional[List[str]]:
        """
        Recursively search for cycles in dependency graph.

        Args:
            current: Current skill being checked
            target: Target skill we're looking for (cycle back to this)
            visited: Set of already visited skills (to detect infinite loops)

        Returns:
            List representing the cycle path, or None if no cycle
        """
        if current in visited:
            return None  # Already checked this path

        visited.add(current)

        # Get all versions of the current skill
        if current not in self._skills:
            return None

        for version_dict in self._skills[current].values():
            for dep_name in version_dict.interface.dependencies:
                if dep_name == target:
                    return [current]  # Found cycle!

                # Recursively check this dependency
                cycle = self._find_cycle(dep_name, target, visited.copy())
                if cycle:
                    return [current] + cycle

        return None


# Singleton registry instance for convenience
_global_registry: Optional[SkillRegistry] = None


def get_global_registry() -> SkillRegistry:
    """
    Get the global skill registry singleton.

    Returns:
        Global SkillRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = SkillRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (primarily for testing)."""
    global _global_registry
    _global_registry = None
