import json
import httpx
import streamlit as st
import uuid

## -------------------------------------------------------------------------------------------------
## Firebase Auth API -------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------


# Firebase configuration
DEFAULT_FIREBASE_CONFIG = {
    "apiKey": "AIzaSyBce1O8ZVmYfWJQIe8MzyyLZ50qj4KZTlA",
    "authDomain": "quantifyx-d80bf.firebaseapp.com",
    "projectId": "quantifyx-d80bf",
    "storageBucket": "quantifyx-d80bf.firebasestorage.app",
    "messagingSenderId": "635203093236",
    "appId": "1:635203093236:web:6a075166006d0c3dbad4a1",
    "measurementId": "G-TQCKF7HH3S",
}


def sign_in_anonymous():
    """
    Sign in anonymously to Firebase.

    Returns:
        Dict containing user information including idToken and localId
    """
    request_ref = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={DEFAULT_FIREBASE_CONFIG['apiKey']}"
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"returnSecureToken": True})

    try:
        with httpx.Client() as client:
            response = client.post(request_ref, headers=headers, content=data)
            raise_detailed_error(response)
            response_data = response.json()

        # Add anonymous identifier to the response
        response_data["login_type"] = "anonymous"
        response_data["auth_provider"] = "anonymous"

        return response_data
    except Exception as e:
        print(f"Error in anonymous sign in: {str(e)}")
        return None


def sign_in_with_email_and_password(email, password):
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/verifyPassword?key={0}".format(
        DEFAULT_FIREBASE_CONFIG["apiKey"]
    )
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"email": email, "password": password, "returnSecureToken": True})

    with httpx.Client() as client:
        response = client.post(request_ref, headers=headers, content=data)
        raise_detailed_error(response)
        return response.json()


def get_account_info(id_token):
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getAccountInfo?key={0}".format(
        DEFAULT_FIREBASE_CONFIG["apiKey"]
    )
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"idToken": id_token})

    with httpx.Client() as client:
        response = client.post(request_ref, headers=headers, content=data)
        raise_detailed_error(response)
        return response.json()


def send_email_verification(id_token):
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getOobConfirmationCode?key={0}".format(
        DEFAULT_FIREBASE_CONFIG["apiKey"]
    )
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"requestType": "VERIFY_EMAIL", "idToken": id_token})

    with httpx.Client() as client:
        response = client.post(request_ref, headers=headers, content=data)
        raise_detailed_error(response)
        return response.json()


def send_password_reset_email(email):
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/getOobConfirmationCode?key={0}".format(
        DEFAULT_FIREBASE_CONFIG["apiKey"]
    )
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"requestType": "PASSWORD_RESET", "email": email})

    with httpx.Client() as client:
        response = client.post(request_ref, headers=headers, content=data)
        raise_detailed_error(response)
        return response.json()


def create_user_with_email_and_password(email, password):
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/signupNewUser?key={0}".format(
        DEFAULT_FIREBASE_CONFIG["apiKey"]
    )
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"email": email, "password": password, "returnSecureToken": True})

    with httpx.Client() as client:
        response = client.post(request_ref, headers=headers, content=data)
        raise_detailed_error(response)
        return response.json()


def delete_user_account(id_token):
    request_ref = "https://www.googleapis.com/identitytoolkit/v3/relyingparty/deleteAccount?key={0}".format(
        DEFAULT_FIREBASE_CONFIG["apiKey"]
    )
    headers = {"content-type": "application/json; charset=UTF-8"}
    data = json.dumps({"idToken": id_token})

    with httpx.Client() as client:
        response = client.post(request_ref, headers=headers, content=data)
        raise_detailed_error(response)
        return response.json()


def raise_detailed_error(response):
    """
    Raise detailed errors from httpx responses.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        # Get error message from the response
        error_json = {}
        try:
            error_json = response.json()
        except ValueError:
            error_json = {"error": {"message": response.text}}

        raise Exception(
            f"HTTP Error: {e}. Details: {error_json.get('error', {}).get('message', 'Unknown error')}"
        )
