import os

if __package__:
    from .api import app
else:
    from api import app


def main() -> None:
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug, use_reloader=False)


if __name__ == "__main__":
    main()