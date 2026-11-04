from fastapi import  HTTPException, status

def unauthorized(reason: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {"code": "UNAUTHORIZED", "message": reason}},
        headers={"WWW-Authenticate": "Bearer"},
    )


def conflict(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"error": {"code": "EMAIL_TAKEN", "message": message}},
    )