from langgraph_sdk import Auth

auth = Auth()

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    identity:Auth.types.MinimalUserDict = {
        "display_name": "user-123",
        "identity": "user-123",
        "is_authenticated": True,
        "permissions": ["read", "write"]
    }
    return identity