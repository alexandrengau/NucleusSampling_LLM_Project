import json


def parse_text_to_json(text):
    # Split the input text into paragraphs, removing leading/trailing spaces and replacing newlines
    paragraphs = [p.strip().replace('\n', ' ').replace('\"', "'").replace('  ', ' ') for p in text.split('\n\n') if p.strip()]

    # Remove paragraphs after "End of the Project Gutenberg EBook..."
    end_marker = "End of the Project Gutenberg EBook of The Old Man of the Sea, by W.W. Jacobs"
    if end_marker in paragraphs:
        end_index = paragraphs.index(end_marker)
        paragraphs = paragraphs[:end_index]

    # Remove paragraphs before "THE OLD MAN OF THE SEA"
    start_marker = "THE OLD MAN OF THE SEA"
    if start_marker in paragraphs:
        start_index = paragraphs.index(start_marker) + 1
        paragraphs = paragraphs[start_index:]

    # Remove paragraphs with format "[Illustration: ...]"
    paragraphs = [p for p in paragraphs if not p.startswith("[Illustration:")]

    # Create a dictionary containing the list of paragraphs
    json_data = {'paragraphs': paragraphs}

    # Convert the dictionary to a JSON-formatted string with indentation and non-ASCII characters
    json_string = json.dumps(json_data, indent=2, ensure_ascii=False)

    return json_string


if __name__ == "__main__":
    # Specify the path to the input text file
    file_path = './book_metadata/old-man-and-the-sea.txt'

    # Read the content of the input text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Parse the text and convert it to a JSON-formatted string
    json_content = parse_text_to_json(text_content)

    # Write the JSON content to an output file
    with open('./book_metadata/parsed_book.json', 'w', encoding='utf-8') as output_file:
        output_file.write(json_content)

    # Print a message indicating successful parsing and saving of the JSON file
    print("Parsed book and saved JSON to ./book_metadata/parsed_book.json")
