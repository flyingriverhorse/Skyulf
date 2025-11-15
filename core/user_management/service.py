"""
User Management Service Layer

This module provides service layer functions for user management operations.
It handles business logic, database interactions, and data validation.
"""

from datetime import timedelta
from typing import Optional, List

from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from fastapi import HTTPException, status

from core.database.models import User
from core.utils.datetime import utcnow
from .models import (
    UserCreate, UserUpdate, UserPasswordUpdate,
    UserResponse, UserListResponse, UserStatsResponse
)

# Password hashing context using passlib
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
class UserService:
    """Service class for user management operations."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using passlib with bcrypt."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(password, hashed)

    @staticmethod
    def create_user(db: Session, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == user_data.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # Check if email already exists
        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )

        # Create new user
        hashed_password = UserService.hash_password(user_data.password)
        db_user = User(
            username=user_data.username,
            email=user_data.email,
            password_hash=hashed_password,
            full_name=user_data.full_name,
            is_admin=user_data.is_admin,
            is_verified=user_data.is_verified,
            is_active=True,
            login_count=0
        )

        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    @staticmethod
    def get_users(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        is_active: Optional[bool] = None,
        is_admin: Optional[bool] = None
    ) -> UserListResponse:
        """Get list of users with optional filtering and pagination."""
        query = db.query(User)

        # Apply filters
        if search:
            search_filter = or_(
                User.username.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%"),
                User.full_name.ilike(f"%{search}%")
            )
            query = query.filter(search_filter)

        if is_active is not None:
            query = query.filter(User.is_active == is_active)

        if is_admin is not None:
            query = query.filter(User.is_admin == is_admin)

        # Get total count
        total = query.count()

        # Apply pagination and ordering
        users = query.order_by(desc(User.created_at)).offset(skip).limit(limit).all()

        # Calculate pagination info
        has_next = (skip + limit) < total
        has_prev = skip > 0
        page = (skip // limit) + 1 if limit > 0 else 1

        return UserListResponse(
            users=[UserResponse.from_orm(user) for user in users],
            total=total,
            page=page,
            per_page=limit,
            has_next=has_next,
            has_prev=has_prev
        )

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def update_user(db: Session, user_id: int, user_update: UserUpdate) -> User:
        """Update user information."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Check for username conflicts (if updating username)
        if user_update.username and user_update.username != user.username:
            existing_user = db.query(User).filter(
                and_(User.username == user_update.username, User.id != user_id)
            ).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already exists"
                )

        # Check for email conflicts (if updating email)
        if user_update.email and user_update.email != user.email:
            existing_email = db.query(User).filter(
                and_(User.email == user_update.email, User.id != user_id)
            ).first()
            if existing_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists"
                )

        # Update user fields
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        user.updated_at = utcnow()
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def update_user_password(db: Session, user_id: int, password_update: UserPasswordUpdate) -> User:
        """Update user password."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Verify current password
        if not UserService.verify_password(password_update.current_password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

        # Update password
        user.password_hash = UserService.hash_password(password_update.new_password)
        user.updated_at = utcnow()
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def toggle_user_status(db: Session, user_id: int) -> User:
        """Toggle user active status."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user.is_active = not user.is_active
        user.updated_at = utcnow()
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def delete_user(db: Session, user_id: int) -> bool:
        """Delete a user (soft delete by deactivating)."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Soft delete by deactivating
        user.is_active = False
        user.updated_at = utcnow()
        db.commit()
        return True

    @staticmethod
    def hard_delete_user(db: Session, user_id: int) -> bool:
        """Permanently delete a user (use with caution)."""
        user = UserService.get_user_by_id(db, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        db.delete(user)
        db.commit()
        return True

    @staticmethod
    def get_user_stats(db: Session) -> UserStatsResponse:
        """Get user statistics."""
        now = utcnow()
        thirty_days_ago = now - timedelta(days=30)
        twenty_four_hours_ago = now - timedelta(hours=24)

        # Basic counts
        total_users = db.query(User).count()
        active_users = db.query(User).filter(User.is_active.is_(True)).count()
        admin_users = db.query(User).filter(User.is_admin.is_(True)).count()
        verified_users = db.query(User).filter(User.is_verified.is_(True)).count()

        # Recent activity
        recent_signups = db.query(User).filter(User.created_at >= thirty_days_ago).count()
        recent_logins = db.query(User).filter(User.last_login >= twenty_four_hours_ago).count()

        return UserStatsResponse(
            total_users=total_users,
            active_users=active_users,
            admin_users=admin_users,
            verified_users=verified_users,
            recent_signups=recent_signups,
            recent_logins=recent_logins
        )

    @staticmethod
    def update_last_login(db: Session, user_id: int) -> User:
        """Update user's last login timestamp."""
        user = UserService.get_user_by_id(db, user_id)
        if user:
            user.last_login = utcnow()
            user.login_count += 1
            db.commit()
            db.refresh(user)
        return user

    @staticmethod
    def search_users(db: Session, query: str, limit: int = 10) -> List[User]:
        """Search users by username, email, or full name."""
        search_filter = or_(
            User.username.ilike(f"%{query}%"),
            User.email.ilike(f"%{query}%"),
            User.full_name.ilike(f"%{query}%")
        )

        return db.query(User).filter(search_filter).limit(limit).all()
