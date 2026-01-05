
import polars as pl
from skyulf.profiling.analyzer import EDAAnalyzer

def test_sentiment_frontend():
    print("--- Sentiment Analysis Frontend Data Test ---")
    
    # 1. Create Synthetic Text Data
    data = {
        "text_col": [
            "I love this product! It is amazing.",
            "This is the worst experience ever.",
            "It is okay, nothing special.",
            "Absolutely fantastic service.",
            "Terrible, do not buy.",
            "Just a normal day.",
            None,
            "Another neutral statement."
        ]
    }
    
    df = pl.DataFrame(data)
    print("Dataset created with mixed sentiment text.")
    
    # 2. Run Analyzer
    analyzer = EDAAnalyzer(df)
    
    # Check semantic type detection
    semantic_type = analyzer._get_semantic_type(df["text_col"])
    print(f"Detected Semantic Type: {semantic_type}")
    
    if semantic_type != "Text":
        print("WARNING: Column was not detected as Text! Sentiment analysis will be skipped.")
        # Force it for testing logic
        print("Forcing analysis on text_col...")
        
    # Run Sentiment Analysis directly
    sentiment = analyzer._analyze_sentiment(df["text_col"])
    
    print("\n[Sentiment Result]")
    print(sentiment)
    
    # Check format for Frontend
    if sentiment:
        keys = list(sentiment.keys())
        print(f"Keys: {keys}")
        
        expected_keys = ["positive", "neutral", "negative"]
        has_all_keys = all(k in keys for k in expected_keys)
        
        if has_all_keys:
            print("SUCCESS: Keys match frontend expectation (lowercase).")
        else:
            print(f"FAILURE: Keys do not match. Expected {expected_keys}, got {keys}")
            
        # Check values are ratios (0-1)
        values = list(sentiment.values())
        is_ratio = all(0 <= v <= 1 for v in values)
        if is_ratio:
             print("SUCCESS: Values are ratios.")
        else:
             print(f"FAILURE: Values are not ratios. Got {values}")
    else:
        print("FAILURE: No sentiment result returned.")

if __name__ == "__main__":
    test_sentiment_frontend()
