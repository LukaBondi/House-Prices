// Local uvicorn + static file server (python -m http.server).
// Docker overrides this file with config.docker.js → /api (nginx proxy).
window.API_BASE = "http://localhost:8000";
