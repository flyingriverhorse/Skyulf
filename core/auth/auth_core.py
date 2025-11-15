"""
FastAPI Authentication Core Module

Modern JWT-based authentication system migrated from Flask.
Provides secure authentication with async support and dependency injection.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from passlib.context import CryptContext
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

# Password hashing context using modern practices
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Import configuration functions
try:
    from config import get_settings
except ImportError:
    # Fallback for when running as standalone module
    import sys
    from pathlib import Path
    fastapi_app_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(fastapi_app_dir))
    from config import get_settings

# Get settings instance for configuration values


def get_auth_settings():
    """Get authentication settings from config"""
    return get_settings()

# JWT configuration (now loaded from config)


def get_jwt_config():
    """Get JWT configuration from settings"""
    settings = get_auth_settings()
    return {
        "SECRET_KEY": settings.SECRET_KEY,
        "ALGORITHM": settings.ALGORITHM,
        "ACCESS_TOKEN_EXPIRE_MINUTES": settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        "REFRESH_TOKEN_EXPIRE_DAYS": settings.REFRESH_TOKEN_EXPIRE_DAYS,
    }

# Security configuration (now loaded from config)


def get_security_config():
    """Get security configuration from settings"""
    settings = get_auth_settings()
    return {
        "MAX_LOGIN_ATTEMPTS": settings.MAX_LOGIN_ATTEMPTS,
        "ACCOUNT_LOCKOUT_DURATION": timedelta(minutes=settings.ACCOUNT_LOCKOUT_DURATION_MINUTES),
    }


class TokenData(BaseModel):
    """Token data model for JWT processing."""
    username: Optional[str] = None
    scopes: List[str] = []


class Token(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    scope: str = ""


class UserInDB(BaseModel):
    """User model as stored in database."""
    id: Optional[int] = None  # Database ID for compatibility
    username: str
    email: EmailStr
    display_name: str
    hashed_password: str
    ad_groups: List[str] = []
    is_active: bool = True
    is_admin: bool = False
    created_date: datetime
    last_login: Optional[datetime] = None
    failed_attempts: int = 0
    account_locked: bool = False
    locked_until: Optional[datetime] = None


class User(BaseModel):
    """User model for API responses (no password)."""
    id: Optional[int] = None  # Database ID for compatibility
    username: str
    email: EmailStr
    display_name: str
    ad_groups: List[str] = []
    is_active: bool = True
    is_admin: bool = False
    created_date: datetime
    last_login: Optional[datetime] = None

    def has_permission(self, permission: str) -> bool:
        """
        Check if user has a specific permission.

        Args:
            permission: Permission to check (e.g., "admin", "user")

        Returns:
            True if user has permission, False otherwise
        """
        # Admin users have all permissions
        if self.is_admin:
            return True

        # Check based on permission type
        if permission == "admin":
            return self.is_admin
        elif permission == "user":
            return True  # All authenticated users have basic user permissions

        # For more complex permissions, convert to UserInDB temporarily
        # This is a temporary solution - in production, use proper permission system
        return False


class UserCreate(BaseModel):
    """User creation model."""
    username: str
    email: EmailStr
    display_name: str
    password: str
    ad_groups: List[str] = []


class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str


# Enhanced AD groups mapping (migrated from Flask)
AD_GROUPS: Dict[str, str] = {
    "User": "User",  # Can use application features
    "Admin": "Admin",  # Full admin access
}

# Dummy users database (migrated from Flask - replace with real DB)


def create_dummy_users() -> Dict[str, UserInDB]:
    """Create dummy users for development/testing."""
    users = {
        "user2": UserInDB(
            id=1,  # Assign unique ID
            username="user2",
            email="user2@123.com",
            display_name="Standard User",
            hashed_password=get_password_hash("password123"),
            ad_groups=[AD_GROUPS["Admin"]],
            created_date=datetime(2024, 1, 1),
            is_active=True,
            is_admin=False
        ),
        "admin1": UserInDB(
            id=2,  # Assign unique ID
            username="admin1",
            email="admin1@123.com",
            display_name="Admin User",
            hashed_password=get_password_hash("admin123"),
            ad_groups=[AD_GROUPS["Admin"]],
            created_date=datetime(2024, 1, 1),
            is_active=True,
            is_admin=True
        ),
    }

    try:
        settings = get_auth_settings()
        if getattr(settings, "AUTH_FALLBACK_ENABLED", False):
            username = settings.AUTH_FALLBACK_USERNAME or "admin"
            password = settings.AUTH_FALLBACK_PASSWORD or "admin123"
            display_name = settings.AUTH_FALLBACK_DISPLAY_NAME or username
            is_admin = bool(getattr(settings, "AUTH_FALLBACK_IS_ADMIN", True))

            ad_groups = [AD_GROUPS["User"]]
            if is_admin and AD_GROUPS["Admin"] not in ad_groups:
                ad_groups.append(AD_GROUPS["Admin"])

            users[username] = UserInDB(
                id=3,
                username=username,
                email=f"{username}@local.dev",
                display_name=display_name,
                hashed_password=get_password_hash(password),
                ad_groups=ad_groups,
                created_date=datetime(2024, 1, 1),
                is_active=True,
                is_admin=is_admin
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to register fallback auth user: %s", exc)

    return users


def get_password_hash(password: str) -> str:
    """Create password hash using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    secret_key: Optional[str] = None
) -> str:
    """
    Create JWT access token.

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        secret_key: Secret key for signing

    Returns:
        JWT token string
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)

    # Get config values
    jwt_config = get_jwt_config()

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=jwt_config["ACCESS_TOKEN_EXPIRE_MINUTES"])

    # Convert to timestamps for JWT library
    to_encode.update({"exp": expire.timestamp()})
    to_encode.update({"iat": now.timestamp()})
    to_encode.update({"type": "access"})

    if not secret_key:
        secret_key = secrets.token_urlsafe(32)  # Default fallback

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=jwt_config["ALGORITHM"])
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    secret_key: Optional[str] = None
) -> str:
    """
    Create JWT refresh token.

    Args:
        data: Data to encode in token
        expires_delta: Token expiration time
        secret_key: Secret key for signing

    Returns:
        JWT refresh token string
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)

    # Get config values
    jwt_config = get_jwt_config()

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(days=jwt_config["REFRESH_TOKEN_EXPIRE_DAYS"])

    # Convert to timestamps for JWT library
    to_encode.update({"exp": expire.timestamp()})
    to_encode.update({"iat": now.timestamp()})
    to_encode.update({"type": "refresh"})

    if not secret_key:
        secret_key = secrets.token_urlsafe(32)  # Default fallback

    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=jwt_config["ALGORITHM"])
    return encoded_jwt


def verify_token(token: str, secret_key: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
    """
    Verify and decode JWT token with proper expiration checking.

    Args:
        token: JWT token to verify
        secret_key: Secret key for verification
        token_type: Expected token type (access/refresh)

    Returns:
        Token payload if valid, None otherwise
    """
    try:
        # Get config values
        jwt_config = get_jwt_config()

        # Decode and verify the token (this will raise an exception if expired)
        payload = jwt.decode(token, secret_key, algorithms=[jwt_config["ALGORITHM"]])

        # Verify token type
        if payload.get("type") != token_type:
            logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
            return None

        # Check if token was issued in the future (clock skew protection)
        iat = payload.get("iat")
        if iat:
            current_timestamp = datetime.now(timezone.utc).timestamp()
            if current_timestamp < (iat - 300):  # Allow 5 minutes clock skew
                logger.warning("Token issued in the future, possible clock skew")
                return None

        return payload

    except ExpiredSignatureError:
        logger.info("Token signature has expired")
        return None
    except JWTError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


async def authenticate_user(username: str, password: str, users_db: Dict[str, UserInDB]) -> Optional[UserInDB]:
    """
    Authenticate user credentials.

    Args:
        username: Username to authenticate
        password: Plain text password
        users_db: User database

    Returns:
        User object if authenticated, None otherwise
    """
    user = users_db.get(username)
    if not user:
        try:
            from .service import get_user_by_username

            user = await get_user_by_username(username)
            if user:
                users_db[username] = user
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Dynamic user lookup failed for %s: %s", username, exc)

    if not user:
        logger.warning(f"Authentication failed: user '{username}' not found")
        return None

    # Check if account is locked
    if user.account_locked and user.locked_until:
        if datetime.now(timezone.utc) < user.locked_until:
            logger.warning(f"Authentication failed: account '{username}' is locked")
            return None
        else:
            # Unlock account if lockout period has expired
            user.account_locked = False
            user.locked_until = None
            user.failed_attempts = 0

    # Verify password
    if not verify_password(password, user.hashed_password):
        # Get security config
        security_config = get_security_config()

        # Increment failed attempts
        user.failed_attempts += 1

        # Lock account if max attempts reached
        if user.failed_attempts >= security_config["MAX_LOGIN_ATTEMPTS"]:
            user.account_locked = True
            user.locked_until = datetime.now(timezone.utc) + security_config["ACCOUNT_LOCKOUT_DURATION"]
            logger.warning(f"Account '{username}' locked due to too many failed attempts")

        logger.warning(f"Authentication failed: invalid password for user '{username}'")
        return None

    # Check if account is active
    if not user.is_active:
        logger.warning(f"Authentication failed: account '{username}' is inactive")
        return None

    # Successful authentication - reset failed attempts
    user.failed_attempts = 0
    user.account_locked = False
    user.locked_until = None
    user.last_login = datetime.now(timezone.utc)

    if user.id:
        try:
            from .service import record_successful_login

            await record_successful_login(user_id=user.id)
        except Exception as exc:  # pragma: no cover - telemetry only
            logger.debug("Skipping login metadata update for %s: %s", username, exc)

    logger.info(f"User '{username}' authenticated successfully")
    return user


def get_user_permissions(user: UserInDB) -> List[str]:
    """
    Get user permissions based on AD groups.

    Args:
        user: User object

    Returns:
        List of permission strings
    """
    permissions = []

    for group in user.ad_groups:
        if group == AD_GROUPS["User"]:
            permissions.extend([
                "read:data",
                "create:data_source",
                "update:own_data_source",
                "delete:own_data_source"
            ])
        elif group == AD_GROUPS["Admin"]:
            permissions.extend([
                "read:all",
                "create:all",
                "update:all",
                "delete:all",
                "admin:users",
                "admin:system"
            ])

    # Remove duplicates
    return list(set(permissions))


def create_tokens(user: UserInDB, secret_key: str) -> Token:
    """
    Create access and refresh tokens for user.

    Args:
        user: Authenticated user
        secret_key: Secret key for token signing

    Returns:
        Token response with access and refresh tokens
    """
    permissions = get_user_permissions(user)

    # Get config values
    jwt_config = get_jwt_config()

    # Data to encode in token
    token_data = {
        "sub": user.username,
        "email": user.email,
        "display_name": user.display_name,
        "scopes": permissions,
        "is_admin": user.is_admin
    }

    access_token_expires = timedelta(minutes=jwt_config["ACCESS_TOKEN_EXPIRE_MINUTES"])
    refresh_token_expires = timedelta(days=jwt_config["REFRESH_TOKEN_EXPIRE_DAYS"])

    access_token = create_access_token(
        data=token_data,
        expires_delta=access_token_expires,
        secret_key=secret_key
    )

    refresh_token = create_refresh_token(
        data={"sub": user.username},
        expires_delta=refresh_token_expires,
        secret_key=secret_key
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=int(access_token_expires.total_seconds()),
        scope=" ".join(permissions)
    )


# Initialize the users database
USERS_DB = create_dummy_users()
