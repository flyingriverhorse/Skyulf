"""
User Management API Routes

This module provides FastAPI routes for user management operations.
All routes require proper authentication and authorization.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from core.database.engine import get_db
from core.auth.dependencies import get_current_admin_user, get_current_user
from core.database.models import User
from .models import (
    UserCreate, UserUpdate, UserPasswordUpdate, 
    UserResponse, UserListResponse, UserStatsResponse
)
from .service import UserService

# Create router for user management
user_router = APIRouter(
    prefix="/admin/users",
    tags=["User Management"],
    dependencies=[Depends(get_current_admin_user)]  # Require admin access
)


@user_router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(db: Session = Depends(get_db)):
    """Get user statistics for the admin dashboard."""
    return UserService.get_user_stats(db)


@user_router.get("/", response_model=UserListResponse)
async def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of users to return"),
    search: Optional[str] = Query(None, description="Search users by username, email, or name"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    is_admin: Optional[bool] = Query(None, description="Filter by admin status"),
    db: Session = Depends(get_db)
):
    """Get list of users with optional filtering and pagination."""
    return UserService.get_users(
        db=db,
        skip=skip,
        limit=limit,
        search=search,
        is_active=is_active,
        is_admin=is_admin
    )


@user_router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get specific user by ID."""
    user = UserService.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    return UserResponse.from_orm(user)


@user_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Create a new user."""
    user = UserService.create_user(db, user_data)
    return UserResponse.from_orm(user)


@user_router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db)
):
    """Update user information."""
    user = UserService.update_user(db, user_id, user_update)
    return UserResponse.from_orm(user)


@user_router.patch("/{user_id}/password")
async def update_user_password(
    user_id: int,
    password_update: UserPasswordUpdate,
    db: Session = Depends(get_db)
):
    """Update user password."""
    UserService.update_user_password(db, user_id, password_update)
    return {"message": "Password updated successfully"}


@user_router.patch("/{user_id}/toggle-status", response_model=UserResponse)
async def toggle_user_status(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Toggle user active status."""
    user = UserService.toggle_user_status(db, user_id)
    return UserResponse.from_orm(user)


@user_router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    hard_delete: bool = Query(False, description="Permanently delete user"),
    db: Session = Depends(get_db)
):
    """Delete a user (soft delete by default, hard delete if specified)."""
    if hard_delete:
        UserService.hard_delete_user(db, user_id)
        return {"message": "User permanently deleted"}
    else:
        UserService.delete_user(db, user_id)
        return {"message": "User deactivated"}


@user_router.get("/search/{query}")
async def search_users(
    query: str,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    db: Session = Depends(get_db)
):
    """Search users by username, email, or full name."""
    users = UserService.search_users(db, query, limit)
    return [UserResponse.from_orm(user) for user in users]


# Routes for current user (don't require admin)
current_user_router = APIRouter(
    prefix="/users/me",
    tags=["Current User"],
    dependencies=[Depends(get_current_user)]  # Only require regular user auth
)


@current_user_router.get("/", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Get current user's profile information."""
    return UserResponse.from_orm(current_user)


@current_user_router.put("/", response_model=UserResponse)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's profile information."""
    # Users can't change their admin status or active status
    user_update.is_admin = None
    user_update.is_active = None
    
    user = UserService.update_user(db, current_user.id, user_update)
    return UserResponse.from_orm(user)


@current_user_router.patch("/password")
async def update_current_user_password(
    password_update: UserPasswordUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's password."""
    UserService.update_user_password(db, current_user.id, password_update)
    return {"message": "Password updated successfully"}


# Public routes for user management (limited access)
public_user_router = APIRouter(
    prefix="/users",
    tags=["Public User Operations"]
)


@public_user_router.get("/check-username/{username}")
async def check_username_availability(
    username: str,
    db: Session = Depends(get_db)
):
    """Check if a username is available."""
    user = UserService.get_user_by_username(db, username)
    return {"available": user is None}


@public_user_router.get("/check-email/{email}")
async def check_email_availability(
    email: str,
    db: Session = Depends(get_db)
):
    """Check if an email is available."""
    user = UserService.get_user_by_email(db, email)
    return {"available": user is None}