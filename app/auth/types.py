"""
Type definitions for auth objects.
"""

from typing import TypedDict, Optional, Literal


class FirebaseUserDict(TypedDict):
    """Type definition for Firebase user object stored in local storage."""

    localId: str
    idToken: str
    refreshToken: str
    expiresIn: str
    login_type: str
    auth_provider: str
    email: Optional[str]
    name: Optional[str]
