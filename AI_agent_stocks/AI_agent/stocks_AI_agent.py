#use python 3.9.7
#!pip install openai yfinance pandas -q
import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # Replace with your key
print("✅ API Setup complete!")

"""
Financial Agent with Custom Stock Prediction
============================================
Uses local CSV data from ../stock_data/ folder
Saves conversation logs to agent_logs/ folder
"""

import pandas as pd
import numpy as np
import json
from openai import OpenAI
from datetime import datetime
import os
from glob import glob
from scipy import stats

client = OpenAI()

pay_model = "gpt-5"

# =============================================================================
# DIRECTORIES
# =============================================================================
DATA_DIR = '../stock_data'
LOG_DIR = 'agent_logs'

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# YOUR CUSTOM PREDICTION FUNCTIONS
# =============================================================================

def read_stock_data(ticker, data_dir=DATA_DIR):
    """
    Read stock data from CSV file in the data directory.
    """
    # Try exact match first
    exact_path = os.path.join(data_dir, f"{ticker}.csv")
    if os.path.exists(exact_path):
        df = pd.read_csv(exact_path, parse_dates=['Date'], index_col='Date')
        return df.sort_index()
    
    # Try pattern match (ticker_date.csv)
    pattern = os.path.join(data_dir, f"{ticker}_*.csv")
    files = glob(pattern)
    if files:
        files.sort(reverse=True)
        df = pd.read_csv(files[0], parse_dates=['Date'], index_col='Date')
        return df.sort_index()
    
    # Try case-insensitive search
    for f in os.listdir(data_dir):
        if f.lower().startswith(ticker.lower()) and f.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, f), parse_dates=['Date'], index_col='Date')
            return df.sort_index()
    
    raise FileNotFoundError(f"No CSV file found for ticker '{ticker}' in {data_dir}")


def normalize_series(series):
    """Normalize a series to 0-1 range."""
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series * 0
    return (series - min_val) / (max_val - min_val)


def find_similar_period(target_df, source_df, target_start, step_size=5):
    """
    Find the period in source_df that best matches target_df's pattern.
    """
    target_start = pd.to_datetime(target_start)
    target_data = target_df[target_df.index >= target_start]['Close']
    target_len = len(target_data)
    
    if target_len < 10:
        raise ValueError("Target period too short (need at least 10 data points)")
    
    target_normalized = normalize_series(target_data).values
    
    best_corr = -1
    best_start_idx = 0
    results = []
    
    for i in range(0, len(source_df) - target_len, step_size):
        source_window = source_df['Close'].iloc[i:i + target_len]
        source_normalized = normalize_series(source_window).values
        
        if len(source_normalized) == len(target_normalized):
            corr, _ = stats.pearsonr(target_normalized, source_normalized)
            rmse = np.sqrt(np.mean((target_normalized - source_normalized) ** 2))
            
            results.append({
                'start_idx': i,
                'start_date': source_df.index[i],
                'end_date': source_df.index[i + target_len - 1],
                'correlation': corr,
                'rmse': rmse
            })
            
            if corr > best_corr:
                best_corr = corr
                best_start_idx = i
    
    if not results:
        raise ValueError("Could not find any matching periods")
    
    results.sort(key=lambda x: x['correlation'], reverse=True)
    best_result = results[0]
    best_result['target_len'] = target_len
    
    return best_result, results[:5]


def predict_future_trend(source_df, best_result, future_days=60):
    """
    Predict future trend based on what happened after the similar period.
    """
    end_idx = best_result['start_idx'] + best_result['target_len']
    future_end_idx = min(end_idx + future_days, len(source_df))
    
    if future_end_idx <= end_idx:
        return None
    
    future_data = source_df.iloc[end_idx:future_end_idx].copy()
    
    if len(future_data) > 0:
        future_data['predicted_price'] = future_data['Close']
        
    return future_data


def run_prediction(target_ticker, target_start, source_ticker, future_days=60, 
                   step_size=5, data_dir=DATA_DIR, show_diagnostics=False):
    """
    Complete prediction pipeline.
    """
    target_df = read_stock_data(target_ticker, data_dir)
    source_df = read_stock_data(source_ticker, data_dir)
    
    best_result, top_results = find_similar_period(
        target_df, source_df, target_start, step_size
    )
    
    target_start_dt = pd.to_datetime(target_start)
    target_data = target_df[target_df.index >= target_start_dt]
    target_last_price = target_data['Close'].iloc[-1]
    
    prediction_df = predict_future_trend(source_df, best_result, future_days)
    
    if prediction_df is not None and len(prediction_df) > 0:
        source_price_at_match_end = source_df['Close'].iloc[best_result['start_idx'] + best_result['target_len'] - 1]
        scale_factor = target_last_price / source_price_at_match_end
        prediction_df['predicted_price'] = prediction_df['Close'] * scale_factor
    
    best_result['dtw_distance'] = best_result['rmse'] * best_result['target_len']
    
    return {
        'result': 'success',
        'best_results': best_result,
        'top_matches': top_results,
        'prediction_df': prediction_df,
        'volume_df': prediction_df,
        'target_last_price': target_last_price,
        'target_ticker': target_ticker,
        'source_ticker': source_ticker,
    }


# =============================================================================
# AGENT TOOL FUNCTIONS
# =============================================================================

def get_stock_price(ticker: str) -> dict:
    """Get current stock price from local CSV data."""
    try:
        df = read_stock_data(ticker)
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        
        return {
            "ticker": ticker.upper(),
            "current_price": round(float(latest['Close']), 2),
            "previous_close": round(float(prev['Close']), 2),
            "date": str(df.index[-1].date()),
            "high": round(float(latest['High']), 2),
            "low": round(float(latest['Low']), 2),
            "volume": int(latest['Volume']) if 'Volume' in latest else None,
            "source": "local_csv"
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def get_stock_history(ticker: str, period: str = "1mo") -> dict:
    """Get historical stock data from local CSV."""
    try:
        df = read_stock_data(ticker)
        
        period_days = {
            "1d": 1, "5d": 5, "1mo": 21, "3mo": 63, 
            "6mo": 126, "1y": 252, "2y": 504, "max": len(df)
        }
        days = period_days.get(period, 21)
        df = df.tail(days)
        
        if df.empty:
            return {"error": "No data found", "ticker": ticker}
        
        return {
            "ticker": ticker.upper(),
            "period": period,
            "data_points": len(df),
            "start_date": str(df.index[0].date()),
            "end_date": str(df.index[-1].date()),
            "start_price": round(float(df['Close'].iloc[0]), 2),
            "end_price": round(float(df['Close'].iloc[-1]), 2),
            "high": round(float(df['High'].max()), 2),
            "low": round(float(df['Low'].min()), 2),
            "avg_volume": int(df['Volume'].mean()) if 'Volume' in df.columns else None,
            "price_change_pct": round(float((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100), 2),
            "source": "local_csv"
        }
    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def predict_stock_today(ticker: str) -> dict:
    """Predict today's stock price based on local CSV data."""
    try:
        df = read_stock_data(ticker)
        
        if len(df) < 30:
            return {"error": f"Not enough data for '{ticker}' (need at least 30 days)"}
        
        current_price = df['Close'].iloc[-1]
        
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        
        sma_10 = df['SMA_10'].iloc[-1]
        sma_30 = df['SMA_30'].iloc[-1]
        returns_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
        
        if sma_10 > sma_30 and returns_5d > 0:
            predicted_direction = "UP"
            predicted_change_pct = min(returns_5d * 0.3, 3.0)
            confidence = "HIGH"
        elif sma_10 < sma_30 and returns_5d < 0:
            predicted_direction = "DOWN"
            predicted_change_pct = max(returns_5d * 0.3, -3.0)
            confidence = "HIGH"
        elif sma_10 > sma_30:
            predicted_direction = "UP"
            predicted_change_pct = 0.5
            confidence = "MEDIUM"
        elif sma_10 < sma_30:
            predicted_direction = "DOWN"
            predicted_change_pct = -0.5
            confidence = "MEDIUM"
        else:
            predicted_direction = "NEUTRAL"
            predicted_change_pct = 0
            confidence = "LOW"
        
        predicted_price = current_price * (1 + predicted_change_pct / 100)
        
        return {
            "ticker": ticker.upper(),
            "prediction_for": "Next trading day",
            "current_price": round(float(current_price), 2),
            "predicted_price": round(float(predicted_price), 2),
            "predicted_direction": predicted_direction,
            "predicted_change_percent": round(float(predicted_change_pct), 2),
            "confidence": confidence,
            "signals": {
                "sma_10": round(float(sma_10), 2),
                "sma_30": round(float(sma_30), 2),
                "sma_signal": "BULLISH" if sma_10 > sma_30 else "BEARISH",
                "5_day_momentum": round(float(returns_5d), 2)
            },
            "analysis_date": str(df.index[-1].date()),
            "method": "Moving Average Crossover + Momentum",
            "source": "local_csv",
            "disclaimer": "This is a statistical prediction, not financial advice."
        }
        
    except Exception as e:
        return {"error": str(e)}


def predict_stock_pattern(target_ticker: str, target_start: str, source_ticker: str, 
                          future_days: int = 60) -> dict:
    """Agent-compatible wrapper for the pattern matching prediction system."""
    try:
        output = run_prediction(
            target_ticker=target_ticker,
            target_start=target_start,
            source_ticker=source_ticker,
            future_days=future_days,
            step_size=5,
            data_dir=DATA_DIR,
            show_diagnostics=False
        )
        
        result = {
            "status": "success",
            "target_ticker": target_ticker.upper(),
            "target_start": target_start,
            "source_ticker": source_ticker.upper(),
            "future_days": future_days,
            "source": "local_csv"
        }
        
        if output.get('best_results'):
            best = output['best_results']
            result["best_match"] = {
                "start_date": str(best.get('start_date', 'N/A'))[:10],
                "end_date": str(best.get('end_date', 'N/A'))[:10],
                "correlation": round(float(best['correlation']), 4) if best.get('correlation') else None,
                "rmse": round(float(best['rmse']), 4) if best.get('rmse') else None,
            }
        
        if output.get('prediction_df') is not None and len(output['prediction_df']) > 0:
            pred_df = output['prediction_df']
            start_price = float(pred_df['predicted_price'].iloc[0])
            end_price = float(pred_df['predicted_price'].iloc[-1])
            change_pct = (end_price / start_price - 1) * 100
            
            result["prediction"] = {
                "start_date": str(pred_df.index[0])[:10],
                "end_date": str(pred_df.index[-1])[:10],
                "start_price": round(start_price, 2),
                "end_price": round(end_price, 2),
                "price_change_pct": round(change_pct, 2),
                "min_price": round(float(pred_df['predicted_price'].min()), 2),
                "max_price": round(float(pred_df['predicted_price'].max()), 2),
                "num_days": len(pred_df),
            }
            
            if change_pct > 5:
                result["prediction"]["trend"] = "BULLISH"
            elif change_pct < -5:
                result["prediction"]["trend"] = "BEARISH"
            else:
                result["prediction"]["trend"] = "NEUTRAL"
        
        result["disclaimer"] = "Pattern matching prediction based on historical similarity. Not financial advice."
        
        return result
        
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": f"Stock data file not found: {str(e)}",
            "suggestion": f"Ensure CSV files exist in {DATA_DIR}/ directory",
            "available_files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
        }
    except Exception as e:
        return {
            "status": "error", 
            "error": str(e),
            "target_ticker": target_ticker,
            "source_ticker": source_ticker
        }


def list_available_stocks() -> dict:
    """List all available stock tickers from local CSV files."""
    try:
        if not os.path.exists(DATA_DIR):
            return {"error": f"Data directory not found: {DATA_DIR}"}
        
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        tickers = list(set([f.split('_')[0].split('.')[0].upper() for f in files]))
        tickers.sort()
        
        return {
            "available_tickers": tickers,
            "count": len(tickers),
            "data_directory": DATA_DIR
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_stock_technicals(ticker: str) -> dict:
    """Fetch technical indicators from local CSV."""
    try:
        df = read_stock_data(ticker)
        
        if len(df) < 200:
            return {"error": f"Not enough data for full technical analysis (have {len(df)} days, need 200)"}
        
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()
        
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        
        latest = df.iloc[-1]
        
        def safe(v):
            return round(float(v), 2) if pd.notna(v) else None
        
        return {
            "ticker": ticker.upper(),
            "date": str(df.index[-1].date()),
            "current_price": safe(latest["Close"]),
            "SMA_50": safe(latest["SMA_50"]),
            "SMA_200": safe(latest["SMA_200"]),
            "RSI_14": safe(latest["RSI"]),
            "52_week_high": safe(df["Close"].tail(252).max()),
            "52_week_low": safe(df["Close"].tail(252).min()),
            "sma_signal": "BULLISH (Golden Cross)" if latest["SMA_50"] > latest["SMA_200"] else "BEARISH (Death Cross)",
            "rsi_signal": "OVERBOUGHT" if latest["RSI"] > 70 else ("OVERSOLD" if latest["RSI"] < 30 else "NEUTRAL"),
            "source": "local_csv"
        }
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

AVAILABLE_TOOLS = {
    "get_stock_price": get_stock_price,
    "get_stock_history": get_stock_history,
    "predict_stock_today": predict_stock_today,
    "predict_stock_pattern": predict_stock_pattern,
    "analyze_stock_technicals": analyze_stock_technicals,
    "list_available_stocks": list_available_stocks,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current/latest stock price from local CSV data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL, TSLA)"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_history",
            "description": "Get historical stock data from local CSV files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "period": {"type": "string", "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y"], "description": "Time period"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_stock_today",
            "description": "Predict next day's stock price using moving averages and momentum from local data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_stock_pattern",
            "description": "Predict future stock movement using pattern matching. Finds similar historical patterns in a source stock to predict target stock's future trend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_ticker": {"type": "string", "description": "Stock ticker to predict (e.g., 'ORCL')"},
                    "target_start": {"type": "string", "description": "Start date in YYYY-MM-DD format (e.g., '2023-08-01')"},
                    "source_ticker": {"type": "string", "description": "Stock to search for similar patterns (e.g., 'AAPL')"},
                    "future_days": {"type": "integer", "description": "Days to predict ahead (default: 60)"}
                },
                "required": ["target_ticker", "target_start", "source_ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_stock_technicals",
            "description": "Get technical indicators (SMA, RSI, 52-week high/low) from local data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_stocks",
            "description": "List all available stock tickers from local CSV files.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# =============================================================================
# CONVERSATION LOGGER
# =============================================================================

class ConversationLogger:
    """Handles saving conversation logs to files."""
    
    def __init__(self, log_dir=LOG_DIR):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a new log file for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"conversation_{self.session_id}.json")
        self.txt_log_file = os.path.join(log_dir, f"conversation_{self.session_id}.txt")
        
        self.conversation_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "exchanges": []
        }
        
        print(f"📝 Logging to: {self.log_file}")
    
    def log_exchange(self, user_input: str, agent_response: str, tool_calls: list = None):
        """Log a single exchange (user input + agent response)."""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "agent": agent_response,
            "tool_calls": tool_calls or []
        }
        
        self.conversation_data["exchanges"].append(exchange)
        self._save_json()
        self._save_txt(exchange)
    
    def log_tool_call(self, tool_name: str, arguments: dict, result: str):
        """Log a tool call (for detailed logging)."""
        return {
            "tool": tool_name,
            "arguments": arguments,
            "result": result[:500] + "..." if len(result) > 500 else result  # Truncate long results
        }
    
    def _save_json(self):
        """Save conversation to JSON file."""
        self.conversation_data["last_updated"] = datetime.now().isoformat()
        self.conversation_data["total_exchanges"] = len(self.conversation_data["exchanges"])
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_data, f, indent=2, ensure_ascii=False)
    
    def _save_txt(self, exchange: dict):
        """Append exchange to human-readable text file."""
        with open(self.txt_log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Time: {exchange['timestamp']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"👤 USER:\n{exchange['user']}\n\n")
            
            if exchange.get('tool_calls'):
                f.write(f"🔧 TOOLS CALLED:\n")
                for tc in exchange['tool_calls']:
                    f.write(f"   - {tc['tool']}({tc['arguments']})\n")
                f.write("\n")
            
            f.write(f"🤖 AGENT:\n{exchange['agent']}\n")
    
    def get_log_path(self):
        """Return the path to the current log file."""
        return self.log_file
    
    def get_session_summary(self):
        """Get a summary of the current session."""
        return {
            "session_id": self.session_id,
            "log_file": self.log_file,
            "txt_file": self.txt_log_file,
            "total_exchanges": len(self.conversation_data["exchanges"]),
            "start_time": self.conversation_data["start_time"]
        }


# =============================================================================
# FINANCIAL AGENT CLASS WITH LOGGING
# =============================================================================

class FinancialAgent:
    def __init__(self, model: str = pay_model):
        self.model = model
        self.conversation_history = []
        self.logger = ConversationLogger()
        self.current_tool_calls = []  # Track tool calls for current exchange
        
        self.system_prompt = """You are an expert financial analyst AI agent with stock prediction capabilities.

You use LOCAL CSV data from the ../stock_data/ directory (not live internet data).

Your tools include:
1. list_available_stocks - See what stocks are available in local data
2. get_stock_price - Get latest price from local CSV
3. get_stock_history - Get historical data from local CSV
4. predict_stock_today - Predict next day using moving averages
5. predict_stock_pattern - Predict using pattern matching between two stocks
6. analyze_stock_technicals - Get technical indicators (SMA, RSI)

For pattern matching predictions, you need:
- target_ticker: the stock to predict
- target_start: start date (YYYY-MM-DD)
- source_ticker: stock to find similar patterns in

Always explain predictions clearly and include disclaimers that these are not financial advice."""

        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })

    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name not in AVAILABLE_TOOLS:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = AVAILABLE_TOOLS[tool_name](**arguments)
            result_str = json.dumps(result, indent=2)
            
            # Log the tool call
            self.current_tool_calls.append(
                self.logger.log_tool_call(tool_name, arguments, result_str)
            )
            
            return result_str
        except Exception as e:
            return json.dumps({"error": str(e)})

    def chat(self, user_input: str) -> str:
        # Reset tool calls for this exchange
        self.current_tool_calls = []
        
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        for iteration in range(10):
            print(f"\n🔄 Iteration {iteration + 1}")
            
            response = client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                #temperature=0.3,
                #reasoning_effort="none"
            )
            
            message = response.choices[0].message

            if message.tool_calls:
                print(f"🔧 Calling {len(message.tool_calls)} tool(s):")
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"   → {tool_name}({arguments})")
                    
                    result = self._execute_tool(tool_name, arguments)
                    print(f"   ✓ Result received")

                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result
                    })
            else:
                # Final response
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content
                })
                
                # Log the complete exchange
                self.logger.log_exchange(
                    user_input=user_input,
                    agent_response=message.content,
                    tool_calls=self.current_tool_calls
                )
                
                print(f"\n✅ Done! (Logged to {self.logger.txt_log_file})")
                return message.content

        return "Max iterations reached."

    def reset(self):
        """Reset conversation but keep the same log file."""
        self.conversation_history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        print("🔄 Conversation reset (log continues in same file).")
    
    def new_session(self):
        """Start a completely new session with new log file."""
        self.conversation_history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        self.logger = ConversationLogger()
        print(f"🆕 New session started. Logging to: {self.logger.log_file}")
    
    def get_log_info(self):
        """Get information about current logging session."""
        return self.logger.get_session_summary()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("🤖 Financial Agent (Local CSV Data)")
    print(f"📁 Data Directory: {DATA_DIR}")
    print(f"📝 Log Directory: {LOG_DIR}")
    print("="*60)
    
    # Check data directory
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        print(f"📊 Found {len(files)} CSV files")
    else:
        print(f"⚠️  Data directory not found: {DATA_DIR}")
    
    agent = FinancialAgent()
    
    print("\n📌 Example queries:")
    print("   • What stocks are available?")
    print("   • Get Google price")
    print("   • Predict NVIDIA using Apple patterns from 2025-08-01")
    print("   • Analyze MSFT technicals")
    print("\n📌 Commands:")
    print("   • 'reset' - Reset conversation (same log file)")
    print("   • 'new' - Start new session (new log file)")
    print("   • 'log' - Show log file info")
    print("   • 'quit' - Exit")
    
    print("\n" + "-"*60)
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print(f"\n📝 Conversation saved to:")
            print(f"   JSON: {agent.logger.log_file}")
            print(f"   TXT:  {agent.logger.txt_log_file}")
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'reset':
            agent.reset()
            continue
        
        if user_input.lower() == 'new':
            agent.new_session()
            continue
        
        if user_input.lower() == 'log':
            info = agent.get_log_info()
            print(f"\n📝 Log Info:")
            print(f"   Session ID: {info['session_id']}")
            print(f"   JSON file: {info['log_file']}")
            print(f"   TXT file: {info['txt_file']}")
            print(f"   Exchanges: {info['total_exchanges']}")
            continue
        
        if not user_input:
            continue
        
        response = agent.chat(user_input)
        print(f"\n🤖 Agent:\n{response}")
