"""Sync changelog."""

import re
from pathlib import Path


def remove_links_from_markdown(input_file: str, output_file: str) -> None:
    """Removes all links from a Markdown file and writes the cleaned content to another file.

    Args:
        input_file (str): Path to the input Markdown file.
        output_file (str): Path to the output Markdown file.
    """
    # Read the input file
    input_file = Path(input_file).read_text()

    # Use regex to replace links [text](link) with just text
    cleaned_content = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", input_file, flags=re.DOTALL)

    # Write the cleaned content to the output file
    Path(output_file).write_text(cleaned_content)


if __name__ == "__main__":
    # Specify input and output file paths
    input_file = "CHANGELOG.md"  # Replace with your input file
    output_file = "docs/changelog.md"  # Replace with your desired output file

    # Remove links from the Markdown file
    remove_links_from_markdown(input_file, output_file)

    print(f"Links removed. Cleaned content written to '{output_file}'")
