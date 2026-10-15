// Vercel serves the dashboard only. Set this to the public URL of the
// separately hosted FastAPI service after deploying the backend.
// Local static servers use the FastAPI dev server on port 8001.
const isLocalHost = ["localhost", "127.0.0.1"].includes(window.location.hostname);
window.RAILCONNECT_API_URL = isLocalHost ? "http://localhost:8001" : "";
