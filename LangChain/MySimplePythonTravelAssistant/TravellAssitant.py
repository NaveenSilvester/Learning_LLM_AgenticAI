import datetime

class ChatbotState:
    """Manages the current state of the conversation and collected data."""
    def __init__(self):
        self.current_agent = "welcome"
        self.destination = None
        self.start_date = None
        self.end_date = None
        self.interests = []
        self.conversation_history = []

    def update_state(self, new_agent=None, data=None):
        if new_agent:
            self.current_agent = new_agent
        if data:
            for key, value in data.items():
                setattr(self, key, value)
        self.conversation_history.append(data) # Store for context if needed

class WelcomeAgent:
    def process(self, user_input, state: ChatbotState):
        if state.current_agent == "welcome":
            response = "Hello! I'm your trip planner chatbot. What destination are you thinking of?"
            state.update_state(new_agent="info_gathering_destination")
        else:
            response = "Welcome back! What else can I help you with?"
        return response

class InformationGatheringAgent:
    def process(self, user_input, state: ChatbotState):
        if state.current_agent == "info_gathering_destination":
            state.update_state(new_agent="info_gathering_dates", data={"destination": user_input})
            return f"Great! So you're thinking of {user_input}. What are your preferred travel dates? (e.g., 'August 1st to August 10th')"
        elif state.current_agent == "info_gathering_dates":
            # Basic date parsing (can be made robust with date libraries)
            try:
                # This is a very simplistic parser. Real-world needs a robust NLP solution.
                dates_str = user_input.lower().replace("from", "").replace("to", "").split("and")
                start_date_str = dates_str[0].strip()
                end_date_str = dates_str[1].strip() if len(dates_str) > 1 else start_date_str

                # Dummy date parsing (you'd use `dateutil.parser` or similar)
                state.update_state(data={"start_date": start_date_str, "end_date": end_date_str})
                state.update_state(new_agent="info_gathering_interests")
                return "Got it. And what kind of activities or interests do you have for this trip? (e.g., 'adventure', 'relaxation', 'culture')"
            except Exception:
                return "I didn't quite catch the dates. Could you please provide them in a format like 'August 1st to August 10th'?"
        elif state.current_agent == "info_gathering_interests":
            interests = [i.strip() for i in user_input.split(',')]
            state.update_state(data={"interests": interests})
            state.update_state(new_agent="itinerary_generation")
            return "Thanks for sharing your interests! Let me cook up a little plan for you..."
        else:
            return "I'm not sure what information you're trying to provide right now."

class ItineraryGenerationAgent:
    def process(self, user_input, state: ChatbotState):
        if state.current_agent == "itinerary_generation":
            destination = state.destination
            start_date = state.start_date
            end_date = state.end_date
            interests = ", ".join(state.interests) if state.interests else "general activities"

            itinerary = f"For your trip to {destination} from {start_date} to {end_date}, with your interest in {interests}, here's a basic idea:\n\n"
            itinerary += "Day 1: Arrive, settle in, maybe explore the local area.\n"
            if "adventure" in state.interests:
                itinerary += "Day 2: Seek out some outdoor adventure, like hiking or a local excursion.\n"
            if "relaxation" in state.interests:
                itinerary += "Day 2: Find a nice spot to relax, perhaps a spa or a beach.\n"
            if "culture" in state.interests:
                itinerary += "Day 2: Visit historical sites or museums.\n"
            itinerary += "Remaining Days: Continue exploring based on your interests, try local cuisine.\n"
            itinerary += "Last Day: Enjoy a final activity and prepare for departure.\n\n"
            itinerary += "How does that sound? Do you need anything else, or are you good to go?"
            state.update_state(new_agent="farewell")
            return itinerary
        return "I'm ready to generate an itinerary, but I'm missing some details. Let's restart the planning process."


class FarewellAgent:
    def process(self, user_input, state: ChatbotState):
        user_input_lower = user_input.lower()
        if any(word in user_input_lower for word in ["bye", "goodbye", "that's all", "no thanks"]):
            state.update_state(new_agent="exit") # Signal to end conversation
            return "Great! Have a wonderful trip! Goodbye for now!"
        elif any(word in user_input_lower for word in ["yes", "more help", "another trip"]):
            state.update_state(new_agent="welcome") # Restart the process
            return "Alright, let's plan another one! What's your next destination?"
        else:
            return "I'm here to help if you need anything else. Just let me know!"


class FallbackAgent:
    def process(self, user_input, state: ChatbotState):
        return "I'm sorry, I didn't understand that. I can help you plan a trip. Could you please tell me your destination?"

class Chatbot:
    def __init__(self):
        self.state = ChatbotState()
        self.agents = {
            "welcome": WelcomeAgent(),
            "info_gathering_destination": InformationGatheringAgent(),
            "info_gathering_dates": InformationGatheringAgent(),
            "info_gathering_interests": InformationGatheringAgent(),
            "itinerary_generation": ItineraryGenerationAgent(),
            "farewell": FarewellAgent(),
            "fallback": FallbackAgent()
        }

    def chat(self, user_input):
        current_agent_name = self.state.current_agent
        agent = self.agents.get(current_agent_name)

        if agent:
            response = agent.process(user_input, self.state)
        else:
            response = self.agents["fallback"].process(user_input, self.state)

        # Special handling for agent transitions or ending conversation
        if self.state.current_agent == "itinerary_generation" and "let me cook up a little plan" in response:
            # Itinerary agent has started its process, next turn is to present
            pass # The agent itself sets the next state to "farewell"

        elif self.state.current_agent == "info_gathering_interests" and "Thanks for sharing your interests!" in response:
            pass # The agent itself sets the next state to "itinerary_generation"

        # If after processing, the agent hasn't explicitly set a new next step
        # and the current state is not "exit", we might default to fallback or
        # allow the agent logic to dictate. For clarity in this example, agents
        # explicitly manage transitions.

        return response

# --- Running the Chatbot ---
if __name__ == "__main__":
    chatbot = Chatbot()
    print(chatbot.chat("")) # Initial greeting

    while True:
        user_message = input("You: ")
        if user_message.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        response = chatbot.chat(user_message)
        print(f"Chatbot: {response}")

        if chatbot.state.current_agent == "exit":
            print("Chatbot: Goodbye!")
            break