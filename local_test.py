import json
from nfl_td_lambda.app import lambda_handler # Imports your main function from app.py

def run_local_test():
    """
    Simulates the AWS Lambda environment to run your app.py script locally.

    This function calls your lambda_handler and prints the output in a
    readable format.
    """
    print("--- Starting Local Test ---")
    print("This will connect to your S3 bucket using your local AWS credentials.")
    
    # In Lambda, these arguments are provided by the service.
    # For a local test, we can pass empty objects.
    dummy_event = {}
    dummy_context = {}

    # Call your main handler function
    response = lambda_handler(dummy_event, dummy_context)

    print("\n--- Test Complete ---")
    
    # Check the response from your function
    status_code = response.get('statusCode')
    print(f"Response Status Code: {status_code}")

    if status_code == 200:
        print("\n--- Top 25 Predicted Scorers ---")
        # The body of the response is a JSON string, so we need to parse it
        predictions = json.loads(response.get('body', '[]'))
        
        # Print the results in a clean table-like format
        header = f"{'Player':<25} {'Team':<5} {'Pos':<5} {'Model Prob':<12} {'Price':<7} {'Market Prob':<12} {'Edge'}"
        print(header)
        print("-" * len(header))
        
        for player in predictions[:25]:
            prob = player.get('predicted_touchdown_probability', 0)
            market_prob = player.get('market_implied_prob', 0)
            edge = player.get('model_edge', 0)
            
            # Format price to include a '+' for positive odds
            price = player.get('price', 0)
            price_str = f"+{price}" if price > 0 else str(price)

            print(f"{player.get('player_display_name', ''):<25} "
                  f"{player.get('team', ''):<5} "
                  f"{player.get('position', ''):<5} "
                  f"{prob:<12.3f} "
                  f"{price_str:<7} "
                  f"{market_prob:<12.3f} "
                  f"{edge:+.3f}")
    else:
        print("\n--- Error ---")
        # Print the error message returned by the Lambda function
        error_body = json.loads(response.get('body', '{}'))
        print(error_body.get('error', 'An unknown error occurred.'))


if __name__ == '__main__':
    run_local_test()
