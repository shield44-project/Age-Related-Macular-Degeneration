if __package__:
    from .api import app
else:
    from api import app


def main() -> None:
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()