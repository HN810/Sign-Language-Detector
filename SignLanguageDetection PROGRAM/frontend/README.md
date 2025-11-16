Frontend folder

This folder contains a lightweight browser UI that shows your webcam feed and can optionally call a local Python server to run predictions using your trained `model.p`.

Files
- index.html — the frontend page (open with Live Server or any static server)
- style.css — styles
- app.js — JavaScript logic for camera & calling backend
- server.py — optional Flask server that accepts a base64 image and returns a predicted label (see instructions)

Run the frontend
- Open `frontend/index.html` in Live Server (VSCode) or any static server. The UI will request camera access.

Optional: run the backend to get real predictions

1. Ensure you have `model.p` created by running `training.py` (put it in the project root; the server will try to load the model from the parent folder).

2. Install Python dependencies in your project venv (or globally):

   pip install -r requirements.txt

3. Run the server (from this folder):

   python server.py

4. Health & debug

- The server exposes a simple health endpoint at `/health` which returns whether the model is loaded and if mediapipe is available. Use this to check availability.
- If you plan to run the server on a host so other people can reach it, set `HOST=0.0.0.0` and (optionally) `PORT` environment variables before starting. Example:

   # Windows PowerShell
   $env:HOST='0.0.0.0'; $env:PORT='5000'; python server.py

   # This allows container platforms and cloud hosts to route traffic to the app.

Deployment suggestions (keep your backend running for demos)

- Docker: build the image in `frontend/` and push to a container host (Render, Fly, Railway, etc.). A `Dockerfile` is included.
- Managed services: Render/Railway/Heroku can run the Flask app and keep it available 24/7. Use the Dockerfile or create a simple service that runs `python server.py` and set `HOST=0.0.0.0`.
- If you deploy a public backend, be careful with exposing `model.p` and consider adding authentication or rate-limiting.

Frontend notes

- The frontend calls `http://127.0.0.1:5000/predict` by default. If you deploy the backend publicly, edit `app.js` to point to the hosted URL (or use an environment-driven configuration during your deployment build).

If you want, I can:
- Improve the frontend design or add a small mapping UI to link folder names to letters.
- Add a GitHub Actions workflow to build & deploy a Docker image to a chosen host (Render/Heroku) so the backend is always available for demos.
