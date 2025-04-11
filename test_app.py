import pytest
from sqlalchemy import create_engine, delete
from app import create_app, issues_table
import uuid
from prometheus_client import CollectorRegistry


test_engine = create_engine('sqlite:///test_issues.db')

@pytest.fixture
def client():
    """Fixture to set up a test client for the Flask app"""
    prometheus_registry = CollectorRegistry()
    app = create_app(test_engine, prometheus_registry)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def setup_test_database():
    """Fixture to set up the test database"""
    issues_table.metadata.create_all(test_engine)

    yield

    with test_engine.connect() as conn:
        conn.execute(delete(issues_table))
        conn.commit()

def test_predict_endpoint(client):
    """Test the /api/predict endpoint"""
    test_data = {
        "title": "Fix broken button",
        "body": "The submit button on the login page doesn't work."
    }
    response = client.post('/api/predict', json=test_data)

    assert response.status_code == 200
    response_data = response.get_json()
    assert "id" in response_data
    assert "label" in response_data
    assert response_data["label"]

def test_correct_endpoint(client):
    """Test the /api/correct endpoint"""

    issue_id = str(uuid.uuid4())
    predicted_label0 = "bug"
    corrected_label0 = "question"

    with test_engine.connect() as conn:
        conn.execute(issues_table.insert().values(
            id=issue_id,
            title="Example title",
            body="Example body",
            predicted_label=predicted_label0,
            corrected_label=None
        ))
        conn.commit()

    test_data = {
        "id": issue_id,
        "label": corrected_label0
    }
    response = client.post('/api/correct', json=test_data)

    assert response.status_code == 200
    response_data = response.get_json()
    assert response_data["id"] == issue_id
    assert response_data["corrected_label"] == corrected_label0
    assert response_data["predicted_label"] == predicted_label0

def test_correct_endpoint_invalid_id(client):
    """Test the /api/correct endpoint with an invalid ID"""

    test_data = {
        "id": "nonexistent-id",
        "label": "enhancement"
    }
    response = client.post('/api/correct', json=test_data)

    assert response.status_code == 404
    response_data = response.get_json()
    assert "error" in response_data
    assert response_data["error"] == "Issue ID not found"
