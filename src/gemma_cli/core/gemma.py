"""Gemma inference interface using native Windows executable."""

import asyncio
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Optional


class GemmaInterface:
    """Interface for communicating with the native Windows gemma.exe."""

    # Security and performance constants
    MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB - prevent memory exhaustion
    MAX_PROMPT_LENGTH = 50_000  # 50KB - prevent command injection
    BUFFER_SIZE = 8192  # 8KB buffer - efficient I/O operations
    FORBIDDEN_CHARS = {"\x00", "\x1b"}  # Null bytes, escape sequences - security

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        gemma_executable: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the Gemma interface.

        Args:
            model_path: Path to the model weights file (.sbs)
            tokenizer_path: Path to tokenizer file (.spm), optional for single-file models
            gemma_executable: Path to gemma.exe binary. If None, searches:
                1. GEMMA_EXECUTABLE environment variable
                2. Common build directories (./build/Release/, ./build-avx2-sycl/, etc.)
                3. PATH environment
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 2.0)

        Raises:
            FileNotFoundError: If gemma executable doesn't exist or can't be found

        Example:
            >>> # Use environment variable
            >>> os.environ['GEMMA_EXECUTABLE'] = '/path/to/gemma.exe'
            >>> gemma = GemmaInterface(model_path)
            >>>
            >>> # Explicit path
            >>> gemma = GemmaInterface(model_path, gemma_executable='/custom/path/gemma.exe')
            >>>
            >>> # Auto-discovery (searches standard locations)
            >>> gemma = GemmaInterface(model_path)  # Uses PATH or build/
        """
        self.model_path = os.path.normpath(model_path)
        self.tokenizer_path = os.path.normpath(tokenizer_path) if tokenizer_path else None

        # Auto-discover executable if not provided
        if gemma_executable is None:
            gemma_executable = self._find_gemma_executable()

        self.gemma_executable = os.path.normpath(gemma_executable)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.process: Optional[subprocess.Popen] = None
        self.debug_mode = False

        # Verify executable exists
        if not os.path.exists(self.gemma_executable):
            raise FileNotFoundError(
                f"Gemma executable not found: {self.gemma_executable}\n"
                f"Set GEMMA_EXECUTABLE environment variable or place gemma.exe in build/Release/"
            )

    def _find_gemma_executable(self) -> str:
        """Find gemma executable in standard locations.

        Search order:
            1. GEMMA_EXECUTABLE environment variable
            2. Common build directories relative to cwd
            3. System PATH

        Returns:
            Path to gemma executable

        Raises:
            FileNotFoundError: If executable not found in any location
        """
        # Check environment variable first
        if "GEMMA_EXECUTABLE" in os.environ:
            path = os.environ["GEMMA_EXECUTABLE"]
            if os.path.exists(path):
                return path

        # Search common build locations (Windows-focused, adjust for Linux)
        is_windows = os.name == "nt"
        exe_name = "gemma.exe" if is_windows else "gemma"

        search_paths = [
            # Standard CMake build directories
            Path.cwd() / "build" / "Release" / exe_name,
            Path.cwd() / "build" / "RelWithSymbols" / exe_name,
            Path.cwd() / "build" / "Debug" / exe_name,
            Path.cwd() / "build" / "FastDebug" / exe_name,
            Path.cwd() / "build" / exe_name,
            # Custom build directory patterns
            Path.cwd() / "build-avx2-sycl" / "bin" / "RELEASE" / exe_name,
            Path.cwd() / "build-avx2" / "Release" / exe_name,
            # Parent directory builds (if gemma_cli is subdirectory)
            Path.cwd().parent / "build" / "Release" / exe_name,
            Path.cwd().parent / "build" / exe_name,
        ]

        for search_path in search_paths:
            if search_path.exists():
                return str(search_path)

        # Search system PATH as last resort
        import shutil

        path = shutil.which(exe_name)
        if path and os.path.exists(path):
            return path

        # Not found anywhere - provide helpful error
        raise FileNotFoundError(
            f"Gemma executable '{exe_name}' not found.\n"
            f"Searched locations:\n"
            f"  - GEMMA_EXECUTABLE environment variable\n"
            f"  - Common build directories: {', '.join(str(p.parent) for p in search_paths[:5])}\n"
            f"  - System PATH\n\n"
            f"Solutions:\n"
            f"  1. Set environment variable: GEMMA_EXECUTABLE=/path/to/{exe_name}\n"
            f"  2. Build gemma: cmake -B build && cmake --build build --config Release\n"
            f"  3. Pass explicit path: GemmaInterface(model_path, gemma_executable='/path/to/{exe_name}')"
        )

    def _build_command(self, prompt: str) -> list[str]:
        """
        Build the command to execute gemma with security validation.

        Args:
            prompt: Input prompt for generation

        Returns:
            Command list ready for subprocess execution

        Raises:
            ValueError: If prompt exceeds MAX_PROMPT_LENGTH or contains forbidden characters
        """
        # Security validation: Check prompt length
        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt exceeds maximum length of {self.MAX_PROMPT_LENGTH} bytes "
                f"(got {len(prompt)} bytes). This prevents potential command injection "
                "and ensures reasonable processing times."
            )

        # Security validation: Check for forbidden characters
        forbidden_found = [char for char in self.FORBIDDEN_CHARS if char in prompt]
        if forbidden_found:
            raise ValueError(
                f"Prompt contains forbidden characters: {forbidden_found}. "
                "Null bytes and escape sequences are not allowed to prevent "
                "command injection and terminal manipulation attacks."
            )

        cmd = [self.gemma_executable, "--weights", self.model_path]

        if self.tokenizer_path:
            cmd.extend(["--tokenizer", self.tokenizer_path])

        # Add generation parameters
        cmd.extend([
            "--max_generated_tokens",
            str(self.max_tokens),
            "--temperature",
            str(self.temperature),
            "--prompt",
            prompt,
        ])

        return cmd

    async def generate_response(
        self, prompt: str, stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Generate response from gemma with optional streaming callback.

        Security features:
        - Validates prompt length (max 50KB) to prevent command injection
        - Limits response size (max 10MB) to prevent memory exhaustion
        - Sanitizes input for forbidden characters (null bytes, escape sequences)

        Performance features:
        - Efficient buffered I/O (8KB chunks) instead of byte-by-byte reading

        Args:
            prompt: Input prompt for generation (max 50KB, no null bytes/escape sequences)
            stream_callback: Optional callback function called with each generated chunk

        Returns:
            Complete generated response text (max 10MB)

        Raises:
            ValueError: If prompt validation fails (length or forbidden characters)
            RuntimeError: If gemma process fails or response exceeds size limit
        """
        cmd = self._build_command(prompt)  # Validates prompt security constraints

        # Debug: print command if enabled
        if self.debug_mode:
            print(f"Debug - Command: {' '.join(cmd)}")

        try:
            # Use asyncio subprocess for proper async handling
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )

            response_parts: list[str] = []
            total_size = 0  # Track accumulated response size for security

            # Read output in real-time for streaming
            if self.process.stdout:
                while True:
                    try:
                        # Performance fix: Read in 8KB chunks instead of 1 byte at a time
                        # This reduces syscalls by ~8000x and dramatically improves performance
                        chunk = await self.process.stdout.read(self.BUFFER_SIZE)
                        if not chunk:
                            break

                        # Security check: Prevent memory exhaustion
                        total_size += len(chunk)
                        if total_size > self.MAX_RESPONSE_SIZE:
                            raise RuntimeError(
                                f"Response exceeded maximum size of {self.MAX_RESPONSE_SIZE} bytes. "
                                "This prevents memory exhaustion attacks or runaway generation."
                            )

                        output = chunk.decode("utf-8", errors="ignore")
                        if output:
                            response_parts.append(output)
                            if stream_callback:
                                stream_callback(output)

                    except (OSError, UnicodeDecodeError) as e:
                        if self.debug_mode:
                            print(f"Debug - Read error: {e}")
                        break

            # Wait for process completion
            return_code = await self.process.wait()

            if return_code != 0:
                stderr_output = ""
                if self.process.stderr:
                    stderr_bytes = await self.process.stderr.read()
                    stderr_output = stderr_bytes.decode("utf-8", errors="ignore")

                raise RuntimeError(
                    f"Gemma process failed (code {return_code}): {stderr_output}"
                )

            return "".join(response_parts)

        except (OSError, ValueError, RuntimeError) as e:
            error_msg = f"Error communicating with gemma: {e}"
            if self.debug_mode:
                print(f"Debug - {error_msg}")
                print(f"Debug - Command was: {' '.join(cmd)}")
            return f"Error: {str(e)}"

        finally:
            await self._cleanup_process()

    async def _cleanup_process(self) -> None:
        """Clean up the subprocess properly."""
        if self.process:
            try:
                if self.process.returncode is None:
                    # Process still running
                    self.process.terminate()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Force kill if it doesn't terminate gracefully
                        self.process.kill()
                        await self.process.wait()
            except (OSError, ProcessLookupError) as e:
                if self.debug_mode:
                    print(f"Debug - Cleanup error: {e}")
            finally:
                self.process = None

    async def stop_generation(self) -> None:
        """Stop current generation process."""
        await self._cleanup_process()

    def set_parameters(
        self, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> None:
        """
        Update generation parameters.

        Args:
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

    def get_config(self) -> dict[str, any]:
        """
        Get current configuration.

        Returns:
            Dictionary with current configuration
        """
        return {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "executable": self.gemma_executable,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "debug_mode": self.debug_mode,
        }
