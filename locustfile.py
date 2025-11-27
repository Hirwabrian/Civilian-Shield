from locust import HttpUser, task, between

class CivilianShieldUser(HttpUser):
    # --- 1. CONFIGURATION (Must be here) ---
    # This sets the base URL for all requests
    host = "https://civilian-shield-armstrong.streamlit.app"
    
    # Simulate a wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)

    @task
    def load_main_page(self):
        # --- 2. ACTION ---
        # Now we simply request the root path "/"
        # Locust will automatically append this to the 'host' defined above.
        self.client.get("/")