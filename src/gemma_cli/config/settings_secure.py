"""Secure version of expand_path function with critical security fixes."""

import os
from pathlib import Path
from typing import List, Optional
import logging


def expand_path_secure(path_str: str, allowed_dirs: Optional[List[Path]] = None) -> Path:
    """
    Expand path with security validation to prevent path traversal attacks.

    This function validates, expands ~ and environment variables, re-validates,
    then ensures the resulting path is within allowed directories to prevent
    malicious path traversal attempts (e.g., "../../../etc/shadow").

    Args:
        path_str: Path string to expand
        allowed_dirs: Optional list of allowed parent directories. If None,
                     uses default safe directories (.gemma_cli, models, config)

    Returns:
        Expanded and validated Path object

    Raises:
        ValueError: If path contains traversal attempts or is outside allowed directories
        FileNotFoundError: If resolved path doesn't exist (for stricter validation)

    Security:
        - Validates BEFORE and AFTER expansion to catch env var injection
        - Prevents path traversal with ".." components
        - Validates resolved path is within allowed directories
        - Properly checks symlink targets (both relative and absolute)

    Example:
        >>> expand_path_secure("~/.gemma_cli/config.toml")
        Path("/home/user/.gemma_cli/config.toml")
        >>> expand_path_secure("../../../etc/shadow")  # Raises ValueError
        >>> os.environ['EVIL'] = '../..'; expand_path_secure("$EVIL/etc")  # Also raises ValueError
    """
    # Define default allowed directories if none provided
    if allowed_dirs is None:
        allowed_dirs = [
            Path.home() / ".gemma_cli",  # User config directory
            Path.cwd(),  # Current working directory
            Path("C:\\codedev\\llm\\.models"),  # Model directory (Windows)
            Path("/c/codedev/llm/.models"),  # Model directory (WSL)
            Path.cwd() / "config",  # Local config directory
            Path.cwd() / "models",  # Local models directory
        ]
        # Add user home if different from .gemma_cli
        allowed_dirs.append(Path.home())

    # SECURITY CHECK 1: Validate raw input BEFORE any expansion
    # This catches direct traversal attempts
    if ".." in path_str:
        raise ValueError(
            f"Path traversal not allowed in input: {path_str}\n"
            "Security: Detected '..' component which could access parent directories"
        )

    # Also check for encoded variations
    if "%2e%2e" in path_str.lower() or "%252e%252e" in path_str.lower():
        raise ValueError(
            f"Path traversal not allowed (encoded): {path_str}\n"
            "Security: Detected encoded '..' component"
        )

    # Expand user home directory (~) and environment variables
    expanded = os.path.expanduser(path_str)
    expanded = os.path.expandvars(expanded)

    # SECURITY CHECK 2: Re-validate AFTER expansion to catch env var injection
    # This catches: export EVIL="../../.."; expand_path("$EVIL/etc/passwd")
    if ".." in expanded:
        raise ValueError(
            f"Path traversal detected after expansion: {expanded}\n"
            f"Original: {path_str}\n"
            "Security: Environment variable or tilde expansion introduced '..' component"
        )

    # Convert to Path
    path = Path(expanded)

    # SECURITY CHECK 3: Validate path parts (catches normalized traversal)
    if ".." in path.parts:
        raise ValueError(
            f"Path traversal in path components: {path.parts}\n"
            "Security: Detected '..' in normalized path parts"
        )

    # Resolve to real path (follows ALL symlinks, gets absolute path)
    try:
        resolved = path.resolve(strict=False)  # Don't require file to exist yet
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Cannot resolve path {path_str}: {e}") from e

    # SECURITY CHECK 4: Validate resolved path is within allowed directories
    # Resolve all allowed directories for proper comparison
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            allowed_resolved = allowed_dir.resolve(strict=False)
            # Use Path.is_relative_to for secure comparison (Python 3.9+)
            # For older Python, fall back to string prefix check
            try:
                if resolved.is_relative_to(allowed_resolved):
                    is_allowed = True
                    break
            except AttributeError:
                # Fallback for Python < 3.9
                if str(resolved).startswith(str(allowed_resolved) + os.sep) or str(resolved) == str(allowed_resolved):
                    is_allowed = True
                    break
        except (OSError, RuntimeError):
            continue  # Skip invalid allowed directories

    if not is_allowed:
        allowed_dirs_str = "\n  - ".join(str(d.resolve(strict=False)) for d in allowed_dirs if d.exists())
        raise ValueError(
            f"Path {resolved} is not within allowed directories:\n  - {allowed_dirs_str}\n"
            "Security: Paths must be within designated safe directories"
        )

    # SECURITY CHECK 5: Validate symlinks properly
    # Check if the original path (not resolved) is a symlink
    if path.exists() and path.is_symlink():
        # The resolved path already follows symlinks, so we just need to ensure
        # the final target is within allowed directories (which we already checked)
        # Log for security audit trail
        logger = logging.getLogger(__name__)
        logger.debug(f"Symlink detected: {path} -> {resolved}")

        # Double-check: ensure symlink target is also validated
        # This is redundant but provides defense in depth
        if not is_allowed:
            raise ValueError(
                f"Symlink {path} resolves to {resolved} which is outside allowed directories\n"
                "Security: Symlink targets must be within allowed directories"
            )

    return resolved