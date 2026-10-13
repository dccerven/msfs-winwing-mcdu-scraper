"""
Configuration management for MSFS WinWing CDU Scraper
"""

import yaml
import logging
from typing import Any, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for MCDU scraper"""

    # Grid specifications (CRITICAL - Must Match MobiFlight)
    CDU_COLUMNS = 24
    CDU_ROWS = 14
    CDU_CELLS = CDU_COLUMNS * CDU_ROWS  # 336 cells total

    # Font sizes
    FONT_SIZE_LARGE = 0
    FONT_SIZE_SMALL = 1

    # Colour codes, taken from MobiFlight's own FormatTable in
    # src/MobiFlightConnector/MobiFlight/Joysticks/WinCtrl/WinCtrlCduController.cs
    # and the reference headwind_a33_winwing_cdu.py script.
    # A code outside this table is rendered as grey by the device, not white.
    COLORS = {
        "a": "amber",
        "c": "cyan",
        "e": "grey",
        "g": "green",
        "k": "khaki",
        "m": "magenta",
        "o": "blue",
        "r": "red",
        "w": "white",
        "y": "yellow",
    }

    #: Where the scraped screen can be sent.
    OUTPUTS = ("mobiflight", "spad")

    #: SPAD.next defaults, matching dementedmonkey's CJ4/737 MCDU mod.
    SPAD_DEFAULT_URL = "ws://localhost:8088"
    SPAD_DEFAULT_MESSAGE_TYPE = "wt737"

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            # Look for config.yaml in current directory, then parent
            config_path = self._find_config_file()

        self.config_path = config_path
        self.config_data = self._load_config()
        self._validate_config()

    def _find_config_file(self) -> str:
        """Find config.yaml in current or parent directories"""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.cwd() / "config.yaml.example",
            Path(__file__).parent.parent / "config.yaml.example"
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found configuration file at: {path}")
                return str(path)

        raise FileNotFoundError(
            "No config.yaml found. Please copy config.yaml.example to config.yaml "
            "and configure your screen regions."
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_config(self):
        """Validate configuration has required fields.

        Which window to capture and which part of it to crop are chosen in
        the GUI, so they are not configuration any more.
        """
        for section in ('mobiflight', 'performance'):
            if section not in self.config_data:
                raise ValueError(
                    f"Missing required configuration section: {section}")
        self.get_output()       # raises on an unknown value

        logger.info("Configuration validation passed")

    def get_captain_url(self) -> str:
        """Get captain WebSocket URL."""
        url = self.config_data['mobiflight'].get('captain_url')
        if not url:
            raise ValueError(
                "Missing 'mobiflight.captain_url' in config. "
                "Please set it to the WinWing CDU captain WebSocket URI "
                "(e.g. ws://localhost:8320/winwing/cdu-captain)."
            )
        return url

    def get_copilot_url(self) -> str:
        """Get copilot WebSocket URL."""
        url = self.config_data['mobiflight'].get('copilot_url')
        if not url:
            raise ValueError(
                "Missing 'mobiflight.copilot_url' in config. "
                "Please set it to the WinWing CDU co-pilot WebSocket URI "
                "(e.g. ws://localhost:8320/winwing/cdu-co-pilot)."
            )
        return url

    def get_font(self) -> Optional[str]:
        """Font override, or None to use the aircraft profile's font.

        Each aircraft profile carries the right hardware font (AirbusThales,
        Boeing, ...), so most users should leave this unset. Setting it here
        forces one font regardless of profile.
        """
        return self.config_data['mobiflight'].get('font') or None

    def get_max_retries(self) -> int:
        """Get max WebSocket connection retries"""
        return self.config_data['mobiflight'].get('max_retries', 3)

    def get_output(self) -> str:
        """Where the screen goes by default: 'mobiflight' or 'spad'."""
        output = str(self.config_data.get('output') or 'mobiflight').lower()
        if output not in self.OUTPUTS:
            raise ValueError(
                f"Unknown 'output: {output}' in config. "
                f"Use one of: {', '.join(self.OUTPUTS)}.")
        return output

    def _spad(self) -> Dict[str, Any]:
        return self.config_data.get('spad') or {}

    def get_spad_url(self) -> str:
        """SPAD.next's MCDU WebSocket (its SimBridge-style relay)."""
        return self._spad().get('url') or self.SPAD_DEFAULT_URL

    def get_spad_message_type(self) -> str:
        """The aircraft type tag sent in ``update:<type>:`` messages."""
        return (self._spad().get('message_type')
                or self.SPAD_DEFAULT_MESSAGE_TYPE)

    def get_spad_aircraft(self) -> str:
        """Sent as the update's ``aircraft`` field (the sim's ATC MODEL)."""
        return str(self._spad().get('aircraft') or '')

    def get_capture_fps(self) -> int:
        """Get capture frame rate, clamped to [1, 120]."""
        raw = self.config_data['performance'].get('capture_fps', 30)
        return max(1, min(120, int(raw)))

    def get_enable_caching(self) -> bool:
        """Check if caching is enabled"""
        return self.config_data['performance'].get('enable_caching', True)
