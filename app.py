import requests
import os
from flask import Flask, render_template, request, jsonify
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

llm_config = {
    "model": "gpt-4o",
    "api_key": OPENAI_API_KEY,
    "cache_seed": None
}

message_history = {
    "weather_agent": [],
    "currency_agent": [],
    "user_agent": []
}

app = Flask(__name__)

def save_history(history):
    global message_history
    message_history = history
    
def get_history():
    global message_history
    return message_history

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/chat'
           , methods=['POST'])
def chat():
    
    print("HISTORY", get_history())
    message = request.json["message"]
    weather_agent = AssistantAgent(
        name="WeatherAgent",
        llm_config=llm_config,
        system_message="You are an helpful AI assistant capable of finding the weather for any location in the world. You can also provide a brief description of the weather for a location.",
        description="An Weather AI assistant capable of finding the weather for any location in the world.",
        human_input_mode="NEVER"
    )
    def find_weather(location: str) -> dict:
        url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}"
        response = requests.get(url)
        data = response.json()
        return {
            "location": data["location"]["name"],
            "region": data["location"]["region"],
            "country": data["location"]["country"],
            "temperature": data["current"]["temp_c"],
            "condition": data["current"]["condition"]["text"]
        }
    weather_agent.register_for_llm(name="find_weather", description="Finds the weather based on location and return the location, region, country, temperature in celcius and condition there.")(find_weather)

    currency_agent = AssistantAgent(
        name="CurrencyAgent",
        llm_config=llm_config,
        system_message="You are an helpful AI assistant capable of converting currencies.",
        description="An Currency AI assistant capable of converting currencies.",
        human_input_mode="NEVER"
    )
    def get_currency_exchange_rate(value: int, base_currency_code: str, target_currency_code: str) -> dict:
        url = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json"
        response = requests.get(url)
        data = response.json()
        base_currency_in_usd = value / data["usd"][base_currency_code.lower()]
        target_value = base_currency_in_usd * data["usd"][target_currency_code.lower()]
        return {
            "base_currency_code": base_currency_code,
            "target_currency_code": target_currency_code,
            "value": value,
            "target_value": target_value
        }
    currency_agent.register_for_llm(name="get_currency_exchange_rate", description="Converts the value from base currency to target currency and return the base currency code, target currency code, value and target value.")(get_currency_exchange_rate)

    def should_terminate_user(message):
        return "tool_calls" not in message and message["role"] != "tool"

    user_agent = UserProxyAgent(
        name="UserAgent",
        llm_config=llm_config,
        description="A human user capable of interacting with AI agents.",
        code_execution_config=False,
        human_input_mode="NEVER",
        is_termination_msg=should_terminate_user
    )
    user_agent.register_for_execution(name="find_weather")(find_weather)
    user_agent.register_for_execution(name="get_currency_exchange_rate")(get_currency_exchange_rate)

    group_chat = GroupChat(agents=[user_agent, weather_agent, currency_agent], messages=[], max_round=120)
    
    group_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    
    history = get_history()
    weather_agent._oai_messages = { group_manager: history["weather_agent"] }
    currency_agent._oai_messages = { group_manager: history["currency_agent"] }
    user_agent._oai_messages = { group_manager: history["user_agent"] }
    
    user_agent.initiate_chat(group_manager, message=message, clear_history=False)
    
    save_history({
        "weather_agent": weather_agent.chat_messages.get(group_manager),
        "currency_agent": currency_agent.chat_messages.get(group_manager),
        "user_agent": user_agent.chat_messages.get(group_manager)
    })
    
    return jsonify(group_chat.messages[-1])

if __name__ == '__main__':
   app.run()

# chat()