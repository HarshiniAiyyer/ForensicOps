import pytest
from foreapp import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home(client):
    response = client.get('/')
    
    # Basic checks
    assert response.status_code == 200
    
    # Different ways to check response content
    # 1. Check raw bytes (response.data)
    assert b'index.html' in response.data
    
    # 2. Check decoded text (response.text)
    assert 'Medicaid' in response.text
    assert 'Health Insurance Coverage Change' in response.text
    
    # 3. Check if template was rendered
    assert response.is_json == False  # Should be HTML, not JSON
