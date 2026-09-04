import fetch_historical_odds
import fetch_receptions_odds


def test_historical_client_honors_explicit_api_key(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "configured-test-key")
    client = fetch_historical_odds.OddsAPIClient("explicit-test-key")
    assert client.api_key == "explicit-test-key"


def test_receptions_client_honors_explicit_api_key(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "configured-test-key")
    client = fetch_receptions_odds.OddsAPIClient("explicit-test-key")
    assert client.api_key == "explicit-test-key"
