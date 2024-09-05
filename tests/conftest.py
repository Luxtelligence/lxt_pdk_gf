import os


def pytest_configure():
    os.environ["JUPYTER_PLATFORM_DIRS"] = "1"
