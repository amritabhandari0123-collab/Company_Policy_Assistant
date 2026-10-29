import os
import pandas as pd
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load private Slack tokens
load_dotenv()

app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Load the existing company policies
policies = pd.read_csv("company_policies.csv")


@app.event("app_mention")
def answer_policy_question(event, say):
    question = event.get("text", "").lower()

    # Find the most relevant policy
    best_policy = None
    best_score = 0

    for _, row in policies.iterrows():
        title = str(row["title"])
        policy_text = str(row["policy_text"])

        keywords = [
            word for word in title.lower().split()
            if len(word) > 3
        ]

        score = sum(1 for word in keywords if word in question)

        if score > best_score:
            best_score = score
            best_policy = row

    if best_policy is not None:
        response = (
            f"*Relevant Policy:* {best_policy['title']}\n"
            f"{best_policy['policy_text']}"
        )
    else:
        response = (
            "I couldn't find a matching company policy "
            "for your question."
        )

    say(text=response, thread_ts=event["ts"])


if __name__ == "__main__":
    print("Company Policy Slack Bot is running...")
    SocketModeHandler(
        app,
        os.environ["SLACK_APP_TOKEN"]
    ).start()