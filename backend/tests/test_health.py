import pytest

pytestmark = pytest.mark.asyncio


async def test_health_returns_ok(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_ping_reports_service_metadata(client):
    response = await client.get("/api/v1/ping")

    assert response.status_code == 200
    body = response.json()
    assert body["pong"] is True
    assert body["service"] == "animal-face-cbir"
    assert "version" in body and isinstance(body["version"], str)


async def test_openapi_schema_is_published(client):
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert schema["info"]["title"] == "Animal Face CBIR API"
