"""Configuration management for the backtester."""

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment and defaults."""

    # OANDA API settings
    OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
    OANDA_ACCOUNT_ID: str = os.getenv("OANDA_ACCOUNT_ID", "")
    OANDA_ENV: Literal["practice", "live"] = os.getenv("OANDA_ENV", "practice")  # type: ignore

    # Default strategy parameters
    DEFAULT_RR: float = 2.0
    DEFAULT_ATR_TF: str = "M15"
    DEFAULT_ATR_MULT: float = 1.2
    DEFAULT_TIMEZONE: str = "America/New_York"
    DEFAULT_USE_MID_STOP: bool = False

    # Slippage and fees (configurable per instrument)
    SLIPPAGE_BPS: dict[str, float] = {
        "US500_USD": 0.5,  # 0.5 basis points
        "NAS100_USD": 0.5,
        "SPX500_USD": 0.5,
        "EUR_USD": 0.1,
        "default": 1.0,
    }

    COMMISSION_BPS: dict[str, float] = {
        "US500_USD": 0.0,  # OANDA typically includes in spread
        "NAS100_USD": 0.0,
        "SPX500_USD": 0.0,
        "EUR_USD": 0.0,
        "default": 0.0,
    }

    # Cache settings
    CACHE_DIR: str = "./cache"
    ENABLE_CACHE: bool = True

    @classmethod
    def validate(cls) -> None:
        """Validate that required settings are present."""
        if not cls.OANDA_API_KEY:
            raise ValueError("OANDA_API_KEY not set in environment")
        if not cls.OANDA_ACCOUNT_ID:
            raise ValueError("OANDA_ACCOUNT_ID not set in environment")
        if cls.OANDA_ENV not in ("practice", "live"):
            raise ValueError(f"OANDA_ENV must be 'practice' or 'live', got {cls.OANDA_ENV}")

    @classmethod
    def get_slippage_bps(cls, instrument: str) -> float:
        """Get slippage in basis points for an instrument."""
        return cls.SLIPPAGE_BPS.get(instrument, cls.SLIPPAGE_BPS["default"])

    @classmethod
    def get_commission_bps(cls, instrument: str) -> float:
        """Get commission in basis points for an instrument."""
        return cls.COMMISSION_BPS.get(instrument, cls.COMMISSION_BPS["default"])

