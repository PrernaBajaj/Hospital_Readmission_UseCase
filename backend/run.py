"""One command to start the app:  python backend/run.py"""
import uvicorn
if __name__ == "__main__":
    uvicorn.run("app:app", app_dir=__file__.rsplit("/", 1)[0], host="0.0.0.0", port=8000)
