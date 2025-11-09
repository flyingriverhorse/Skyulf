"""
User Management Pydantic Models

This module contains Pydantic models for user management API endpoints.
These models handle validation and serialization for user operations.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator
from enum import Enum


class UserRole(str, Enum):
    """User roles for role-based access control."""
    ADMIN = "admin"
    USER = "user"


class UserStatus(str, Enum):
    """User status options."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class UserBase(BaseModel):
    """Base model for user data."""
    username: str = Field(..., min_length=3, max_length=80)
    email: EmailStr
    full_name: Optional[str] = Field(None, max_length=200)

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if not v.isalnum() and '_' not in v:
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v.lower()


class UserCreate(UserBase):
    """Model for creating a new user."""
    password: str = Field(..., min_length=8, max_length=100)
    is_admin: Optional[bool] = False
    is_verified: Optional[bool] = False

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserUpdate(BaseModel):
    """Model for updating user data."""
    username: Optional[str] = Field(None, min_length=3, max_length=80)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=200)
    is_active: Optional[bool] = None
    is_admin: Optional[bool] = None
    is_verified: Optional[bool] = None

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if v is not None:
            if not v.isalnum() and '_' not in v:
                raise ValueError('Username can only contain letters, numbers, and underscores')
            return v.lower()
        return v


class UserPasswordUpdate(BaseModel):
    """Model for password updates."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v


class UserResponse(BaseModel):
    """Model for user response data."""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    is_verified: bool
    last_login: Optional[datetime]
    login_count: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserListResponse(BaseModel):
    """Model for user list with pagination."""
    users: list[UserResponse]
    total: int
    page: int
    per_page: int
    has_next: bool
    has_prev: bool


class UserStatsResponse(BaseModel):
    """Model for user statistics."""
    total_users: int
    active_users: int
    admin_users: int
    verified_users: int
    recent_signups: int  # Last 30 days
    recent_logins: int   # Last 24 hours


class UserActivityResponse(BaseModel):
    """Model for user activity data."""
    user_id: int
    username: str
    action: str
    timestamp: datetime
    details: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
