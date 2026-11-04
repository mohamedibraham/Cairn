from fastapi import APIRouter, Depends, Request, status
from sqlalchemy.orm import Session

from db import get_db
from dependencies.auth import get_current_user
from models.user import User
from schemas.auth import LoginRequest, RefreshRequest, RegisterRequest, TokenResponse, UserRead, UserUpdate
from services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    user = auth_service.register_user(db, payload)
    return user


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = auth_service.authenticate_user(db, payload.email, payload.password)
    return auth_service.issue_tokens(
        db, user, request.headers.get("user-agent"), request.client.host if request.client else None
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(payload: RefreshRequest, request: Request, db: Session = Depends(get_db)):
    return auth_service.rotate_refresh_token(
        db,
        payload.refresh_token,
        request.headers.get("user-agent"),
        request.client.host if request.client else None,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(payload: RefreshRequest, db: Session = Depends(get_db)):
    auth_service.revoke_refresh_token(db, payload.refresh_token)


@router.get("/me", response_model=UserRead)
def me(user: User = Depends(get_current_user)):
    return user


@router.patch("/me", response_model=UserRead)
def update_me(payload: UserUpdate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return auth_service.update_me(db, user, payload)