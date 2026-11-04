import uuid
import jwt

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from core import decode_access_token
from db import get_db
from models import User
from helpers import unauthorized

bearer_scheme = HTTPBearer(auto_error=False)





def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise unauthorized("Missing bearer token")

    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise unauthorized("Access token expired")
    except jwt.PyJWTError:
        raise unauthorized("Invalid access token")

    try:
        user_id = uuid.UUID(payload["sub"])
    except (KeyError, ValueError):
        raise unauthorized("Malformed token payload")

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise unauthorized("User not found or inactive")

    return user
