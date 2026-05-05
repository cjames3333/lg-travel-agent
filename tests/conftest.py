import asyncio
import os
import socket
import subprocess
import sys
import time
import urllib.request
import pytest
from pathlib import Path

from dotenv import load_dotenv


# Add parent directory to path to import adk_travel_agent module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DEFAULT_MCP_PORT = int(os.getenv("PORT", 8007))


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_server(port: int, timeout: int = 15) -> bool:
    """Poll until the server responds on the given port or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
            return True
        except urllib.error.HTTPError:
            return True  # any HTTP response means the server is up
        except Exception:
            time.sleep(0.5)
    return False


def pytest_configure(config):
    set_env_vars_on_local_run()


@pytest.fixture(scope="session", autouse=True)
def start_mcp_server():
    """Start the weather MCP server once per test session.

    Skips spawning if a server is already listening on the port (e.g. a
    manually started instance), but always verifies the server is reachable
    before yielding so that LGS tests don't start against a dead endpoint.
    """
    server_process = None
    port = DEFAULT_MCP_PORT

    if _port_in_use(port):
        # Already running — don't spawn a second instance
        pass
    else:
        server_process = subprocess.Popen(
            [sys.executable, "weather-mcp-server.py"],
            cwd=Path(__file__).parent.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    if not _wait_for_server(port, timeout=15):
        if server_process:
            server_process.terminate()
        pytest.fail(
            f"Weather MCP server did not become ready on port {port} within 15 seconds. "
            "Start weather-mcp-server.py manually and re-run, or check for startup errors."
        )

    yield

    if server_process is not None:
        server_process.terminate()
        server_process.wait()


def set_env_vars_on_local_run():
    """Load environment variables from .env.test for local testing.
    In CI/CD, environment variables should already be set."""
    env_test_path = Path(__file__).parent.parent / '.env.test'
    if env_test_path.exists():
        load_dotenv(env_test_path)