"""Snowflake data warehouse settings."""

from typing import Optional


class SnowflakeMixin:
    """Snowflake connection and feature toggle."""

    SNOWFLAKE_CONNECTION_TYPE: str = "native"
    SNOWFLAKE_ACCOUNT: Optional[str] = None
    SNOWFLAKE_USER: Optional[str] = None
    SNOWFLAKE_PASSWORD: Optional[str] = None
    SNOWFLAKE_WAREHOUSE: Optional[str] = None
    SNOWFLAKE_DATABASE: Optional[str] = None
    SNOWFLAKE_ROLE: Optional[str] = None
    SNOWFLAKE_SCHEMA: Optional[str] = None
    FEATURE_SNOWFLAKE: bool = False

    def validate_snowflake_config(self) -> bool:
        """Validate Snowflake configuration completeness."""
        if not self.FEATURE_SNOWFLAKE:  # type: ignore[attr-defined]
            return True
        config_names = [
            "SNOWFLAKE_ACCOUNT",
            "SNOWFLAKE_USER",
            "SNOWFLAKE_PASSWORD",
            "SNOWFLAKE_WAREHOUSE",
            "SNOWFLAKE_DATABASE",
        ]
        values = [
            self.SNOWFLAKE_ACCOUNT,
            self.SNOWFLAKE_USER,
            self.SNOWFLAKE_PASSWORD,  # type: ignore[attr-defined]
            self.SNOWFLAKE_WAREHOUSE,
            self.SNOWFLAKE_DATABASE,  # type: ignore[attr-defined]
        ]
        missing = [
            name
            for name, val in zip(config_names, values)
            if not val or val in ("x", "your-account", "your-user")
        ]
        if missing:
            raise ValueError(
                f"Missing or placeholder Snowflake configurations: {', '.join(missing)}\n"
                "Please set these in environment variables"
            )
        return True
