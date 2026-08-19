Run python stock_download.py

Edit it to select interested stocks

This can void the AI agent download unnecessary data on the internet.

All stock data will be saved in folder stock_data/

Then 
cd AI_agent
Run
python stocks_AI_agent.py

Edit it 
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"  # Replace with your OPENAI API key
Select your AI model

You: Hello
AI Agent: Hi there! How can I help with your stocks today?

The dialogs are saved in folder agent_logs/

To add new stocks,
Run python stock_download.py
