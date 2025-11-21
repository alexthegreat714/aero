"""
Dashboard stub for Aero Agent.

Placeholder for future web-based dashboard implementation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class Dashboard:
    """
    Dashboard stub for Aero Agent.

    This is a placeholder for a future web-based dashboard.
    Planned features:
    - Real-time simulation visualization
    - Agent task monitoring
    - OCR result display
    - RAG document browser
    - System metrics

    Example (future):
        dashboard = Dashboard()
        dashboard.start(port=8080)
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        """
        Initialize the dashboard.

        Args:
            host: Dashboard host
            port: Dashboard port
        """
        self.host = host
        self.port = port
        self._running = False

        logger.info("Dashboard initialized (STUB)")
        logger.warning("Dashboard is a placeholder - no actual UI available")

    @property
    def is_running(self) -> bool:
        """Check if dashboard is running."""
        return self._running

    def start(self) -> None:
        """
        Start the dashboard server.

        STUB: Does nothing currently.
        """
        logger.info(f"Dashboard would start at http://{self.host}:{self.port} (STUB)")
        self._running = True

    def stop(self) -> None:
        """
        Stop the dashboard server.

        STUB: Does nothing currently.
        """
        logger.info("Dashboard stopped (STUB)")
        self._running = False

    def get_status(self) -> dict:
        """Get dashboard status."""
        return {
            "running": self._running,
            "host": self.host,
            "port": self.port,
            "status": "stub",
            "message": "Dashboard is a placeholder implementation",
        }

    def render_page(self, page: str) -> str:
        """
        Render a dashboard page.

        STUB: Returns placeholder HTML.

        Args:
            page: Page name to render

        Returns:
            HTML content
        """
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Aero Agent Dashboard</title>
            <style>
                body {{
                    font-family: system-ui, -apple-system, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                }}
                h1 {{ color: #333; }}
                .card {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 20px 0;
                }}
                .status {{ color: #4CAF50; }}
                .warning {{ color: #ff9800; }}
            </style>
        </head>
        <body>
            <h1>Aero Agent Dashboard</h1>
            <div class="card">
                <h2 class="warning">Placeholder Dashboard</h2>
                <p>This is a stub implementation. The full dashboard will include:</p>
                <ul>
                    <li>Real-time simulation visualization</li>
                    <li>Agent task monitoring</li>
                    <li>OCR result display</li>
                    <li>RAG document browser</li>
                    <li>System metrics and logs</li>
                </ul>
            </div>
            <div class="card">
                <h3>Current Page: {page}</h3>
                <p class="status">API Server: Running at /api</p>
            </div>
        </body>
        </html>
        """
