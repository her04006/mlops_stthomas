import pytest
import os
import sys
# Add the project root to the path for importing from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi.testclient import TestClient
from src.main import app
from unittest.mock import patch

client = TestClient(app)

def test_redirect_to_docs():
    """Test that the root endpoint redirects to the docs."""
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"

def test_similar_responses_endpoint_status():
    """Test that the similar_responses endpoint returns 200 status."""
    with patch('src.retriever.retriever.get_similar_responses',
               return_value=["Test response 1", "Test response 2"]):
        response = client.post(
            "/similar_responses",
            json={"question": "Test question"}
        )
        assert response.status_code == 200

def test_similar_responses_endpoint_content():
    """Test that the similar_responses endpoint returns expected content structure."""
    test_responses = ["Test response 1", "Test response 2", "Test response 3"]
    
    with patch('src.retriever.retriever.get_similar_responses',
               return_value=test_responses):
        response = client.post(
            "/similar_responses",
            json={"question": "Test question"}
        )
        
        # Check response structure
        data = response.json()
        assert "answers" in data
        assert isinstance(data["answers"], list)
        assert len(data["answers"]) == len(test_responses)
        assert data["answers"] == test_responses

def test_similar_responses_with_long_question():
    """Test the endpoint with a long question."""
    long_question = "This is a very long question " * 20  # ~500 characters
    
    with patch('src.retriever.retriever.get_similar_responses',
               return_value=["Test response"]):
        response = client.post(
            "/similar_responses",
            json={"question": long_question}
        )
        assert response.status_code == 200
        assert response.json() == {"answers": ["Test response"]}

def test_similar_responses_with_empty_question():
    """Test the endpoint with an empty question."""
    # Empty string should be handled, but API might reject it depending on validation
    response = client.post(
        "/similar_responses",
        json={"question": ""}
    )
    # Status code could be 200 or 422 depending on if validation is implemented
    assert response.status_code in [200, 422]

def test_similar_responses_with_special_characters():
    """Test the endpoint with special characters in the question."""
    special_question = "What about <!@#$%^&*()_+?> special chars?"
    
    with patch('src.retriever.retriever.get_similar_responses',
               return_value=["Test response"]):
        response = client.post(
            "/similar_responses",
            json={"question": special_question}
        )
        assert response.status_code == 200
        assert response.json() == {"answers": ["Test response"]}


def test_similar_responses_invalid_json():
    """Test the endpoint with invalid JSON."""
    response = client.post(
        "/similar_responses",
        data="This is not valid JSON"
    )
    assert response.status_code == 422  # Unprocessable Entity

def test_similar_responses_empty_results():
    """Test behavior when the retriever returns no results."""
    with patch('src.retriever.retriever.get_similar_responses',
               return_value=[]):
        response = client.post(
            "/similar_responses",
            json={"question": "Question with no results"}
        )
        assert response.status_code == 200
        assert response.json() == {"answers": []}