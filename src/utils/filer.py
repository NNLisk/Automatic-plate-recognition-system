import os

def get_new_session_id():
    sessions_directory = "data/inference/sessions"
    existing = [int(d) for d in os.listdir(sessions_directory) if d.isdigit()]
    nextID = max(existing, default=0) + 1

    return nextID

def make_new_session():
    sessionID = get_new_session_id()
    sessionPath = f"data/inference/sessions/{sessionID}"
    os.makedirs(sessionPath, exist_ok=True)
    print(f"new session made with id {sessionID}")
    return sessionPath, sessionID

