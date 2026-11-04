from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from core import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from helpers import unauthorized , conflict
from models import RefreshToken, User
from schemas.auth import RegisterRequest, TokenResponse, UserUpdate




def register_user(db: Session, data: RegisterRequest) -> User:
    existing = db.scalar(select(User).where(User.email == data.email))
    if existing is not None:
        raise conflict("An account with this email already exists")

    user = User(
        email=data.email,
        full_name=data.full_name,
        password_hash=hash_password(data.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User:
    user = db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active or not verify_password(password, user.password_hash):
        raise unauthorized("Incorrect email or password")
    return user


def issue_tokens(db: Session, user: User, user_agent: str | None, ip: str | None) -> TokenResponse:
    access_token, expires_in = create_access_token(user.id)
    raw_refresh, refresh_hash, expires_at = generate_refresh_token()

    db.add(RefreshToken(
        user_id=user.id,
        token_hash=refresh_hash,
        user_agent=user_agent,
        ip_address=ip,
        expires_at=expires_at,
    ))
    db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=expires_in,
    )


def rotate_refresh_token(
    db: Session, raw_refresh_token: str, user_agent: str | None, ip: str | None
) -> TokenResponse:
    
    token_hash = hash_refresh_token(raw_refresh_token)
    stored = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))

    now = datetime.now(timezone.utc)
    if (
        stored is None
        or stored.revoked_at is not None
        or stored.expires_at < now
    ):
        raise unauthorized("Refresh token is invalid, expired, or already used")

    stored.revoked_at = now
    user = db.get(User, stored.user_id)
    if user is None or not user.is_active:
        db.commit()
        raise unauthorized("Account no longer active")

    db.commit()
    return issue_tokens(db, user, user_agent, ip)


def revoke_refresh_token(db: Session, raw_refresh_token: str) -> None:
    token_hash = hash_refresh_token(raw_refresh_token)
    stored = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == token_hash))
    if stored is not None and stored.revoked_at is None:
        stored.revoked_at = datetime.now(timezone.utc)
        db.commit()


def update_me(db: Session, user: User, data: UserUpdate) -> User:
    """PATCH /auth/me: change the display name and/or the email of the signed-in user."""
    if data.full_name is not None:
        full_name = data.full_name.strip()
        if not full_name:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail={
                "error": {"code": "EMPTY_NAME", "message": "Name cannot be empty"}
            })
        user.full_name = full_name

    if data.email is not None and data.email != user.email:
        taken = db.scalar(select(User).where(User.email == data.email, User.id != user.id))
        if taken is not None:
            raise conflict("An account with this email already exists")
        user.email = data.email

    db.commit()
    db.refresh(user)
    return user