import requests
import pandas as pd
import os
import time
import datetime
from typing import List, Dict, Optional

class OddsAPIClient:
    """Client for interacting with The Odds API."""
    
    BASE_URL = "https://api.the-odds-api.com/v4/historical/sports"
    SPORT = "americanfootball_nfl"
    REGIONS = "us"
    ODDS_FORMAT = "american"
    DATE_FORMAT = "iso"
    
    def __init__(self, api_key: str):
        self.api_key = '8db99b0d1a04d209bbc64119dcb102b1'

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Helper to make API requests with error handling and rate limiting."""
        url = f"{self.BASE_URL}/{self.SPORT}/{endpoint}"
        params['apiKey'] = self.api_key
        params['regions'] = self.REGIONS
        params['oddsFormat'] = self.ODDS_FORMAT
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Basic rate limiting
            time.sleep(0.2) 
            
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if response.status_code == 422:
                print(f"  (Market likely unavailable for this date)")
            return None

    def get_events(self, date_str: str) -> List[Dict]:
        """Fetches events (games) for a specific date."""
        params = {
            'markets': 'h2h', # Minimal market to get event IDs
            'date': date_str
        }
        data = self._make_request("odds", params)
        if data and 'data' in data:
            return data['data']
        return []

    def get_player_props(self, event_id: str, date_str: str, market: str) -> List[Dict]:
        """Fetches specific player props for an event."""
        params = {
            'markets': market,
            'date': date_str,
            'bookmakers': 'draftkings' # Focusing on a major bookmaker for consistency
        }
        endpoint = f"events/{event_id}/odds"
        data = self._make_request(endpoint, params)
        if data and 'data' in data:
            return data['data']
        return []

def parse_receptions_odds(event_data: Dict, season: int, week: int) -> List[Dict]:
    """Parses the nested JSON response into flat records."""
    rows = []
    
    game_id = event_data.get('id')
    home_team = event_data.get('home_team')
    away_team = event_data.get('away_team')
    commence_time = event_data.get('commence_time')
    
    # We expect 'bookmakers' list in the event_data
    bookmakers = event_data.get('bookmakers', [])
    
    for bookmaker in bookmakers:
        for market in bookmaker.get('markets', []):
            if market['key'] != 'player_receptions':
                continue
                
            # Group outcomes by player to handle Over/Under pairs
            player_outcomes = {}
            
            for outcome in market['outcomes']:
                player_name = outcome.get('description', outcome.get('name'))
                label = outcome.get('name') # 'Over' or 'Under'
                
                if player_name not in player_outcomes:
                    player_outcomes[player_name] = {'Over': {}, 'Under': {}}
                
                if label in ['Over', 'Under']:
                    player_outcomes[player_name][label] = {
                        'line': outcome.get('point'),
                        'odds': outcome.get('price')
                    }
            
            # Create rows from grouped data
            for player, lines in player_outcomes.items():
                row = {
                    'Player': player,
                    'Team': 'N/A', # API doesn't always provide player team directly here
                    'HomeTeam': home_team,
                    'AwayTeam': away_team,
                    'Season': season,
                    'Week': week,
                    'GameDate': commence_time,
                    'Over_Line': lines['Over'].get('line'),
                    'Over_Odds': lines['Over'].get('odds'),
                    'Under_Line': lines['Under'].get('line'),
                    'Under_Odds': lines['Under'].get('odds'),
                    'Bookmaker': bookmaker['title']
                }
                rows.append(row)
                
    return rows

def main():
    API_KEY = '8db99b0d1a04d209bbc64119dcb102b1'
    SEASON = 2024
    OUTPUT_DIR = f'vegas/{SEASON}'
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    client = OddsAPIClient(API_KEY)
    
    # Define the schedule/dates to query. 
    # Since 2025 is in the future relative to the training data context but maybe current for the user,
    # we'll set up a structure that can be easily expanded.
    # For now, let's assume we want to check a specific set of weeks or dates.
    # If this is for *historical* data of a completed season, we'd iterate all weeks.
    # If 2025 is the *current* season, we might only have data for past weeks.
    
    # 2024 NFL season started September 5, 2024 (Thursday night)
    # Week 1 Sunday games were September 8, 2024
    # Generate query dates for each week (Thu/Fri/Sat/Sun/Mon to catch all games)
    weeks = []
    first_sunday = datetime.date(2024, 9, 8)
    
    for w in range(1, 19): # All 18 weeks
        sunday_date = first_sunday + datetime.timedelta(weeks=(w-1))
        
        # Query multiple days to catch Thu/Fri/Sat/Sun/Mon games
        # Thursday = -3, Friday = -2, Saturday = -1, Sunday = 0, Monday = +1
        query_dates = []
        for day_offset in [-3, -2, -1, 0, 1]:  # Thu, Fri, Sat, Sun, Mon
            query_day = sunday_date + datetime.timedelta(days=day_offset)
            query_dates.append(query_day.strftime('%Y-%m-%dT15:00:00Z'))
        
        weeks.append((w, query_dates))
        
    print(f"Starting fetch for {SEASON} season...")
    
    for week_num, query_dates in weeks:
        print(f"Processing Week {week_num}...")
        
        week_data = []
        processed_games = set()
        
        for date_str in query_dates:
            print(f"  Querying date: {date_str}")
            events = client.get_events(date_str)
            
            if not events:
                print(f"    No events found.")
                continue
                
            # Parse query date to compare with commence_time
            query_dt = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            
            for event in events:
                event_id = event['id']
                if event_id in processed_games:
                    continue
                
                # Check commence time
                # We only want to fetch odds if the game is "soon" (e.g., within 48 hours)
                # This prevents fetching Sunday games on Thursday (saving tokens & getting better lines later)
                # But ensures we fetch Thursday games on Thursday.
                commence_time_str = event['commence_time']
                # commence_time is usually ISO like '2025-09-05T00:20:00Z'
                # Handle potentially different formats if needed, but usually it's strict ISO
                try:
                    commence_dt = datetime.datetime.strptime(commence_time_str, '%Y-%m-%dT%H:%M:%SZ')
                except ValueError:
                    # Fallback or skip if format is weird
                    print(f"    Warning: Could not parse commence_time {commence_time_str}")
                    continue
                
                # Calculate hours until game
                diff = commence_dt - query_dt
                hours_until = diff.total_seconds() / 3600
                
                # Logic:
                # If it's the Thursday query, fetch games starting within ~36 hours (Thursday night is ~9-10 hours away, Friday/Saturday games?)
                # If it's the Sunday query, fetch everything remaining (Sunday/Monday).
                # Let's just use a threshold: if game is > 60 hours away, skip it.
                # Thursday to Sunday is 72 hours. So 60 hours is a safe cutoff.
                if hours_until > 60:
                    continue
                    
                print(f"    Fetching odds for {event['home_team']} vs {event['away_team']} (Starts in {hours_until:.1f}h)")
                
                event_odds = client.get_player_props(event_id, date_str, 'player_receptions')
                
                if event_odds:
                    rows = parse_receptions_odds(event_odds, SEASON, week_num)
                    week_data.extend(rows)
                
                processed_games.add(event_id)
            
        if week_data:
            df = pd.DataFrame(week_data)
            output_file = os.path.join(OUTPUT_DIR, f'week_{week_num}_receptions.csv')
            df.to_csv(output_file, index=False)
            print(f"  Saved {len(df)} rows to {output_file}")
        else:
            print(f"  No receptions data found for Week {week_num}")

if __name__ == "__main__":
    main()
