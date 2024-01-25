import json
import xml.etree.ElementTree as ET


# Define a function to extract sentence pairs from a TMX file
def extract_sentence_pairs(tmx_file_path):
    # Parse the TMX file using ElementTree
    tree = ET.parse(tmx_file_path)
    root = tree.getroot()

    # Define the XML namespace (used for handling XML namespace in the TMX file)
    ns = {"xml": "http://www.w3.org/XML/1998/namespace"}

    # Initialize an empty list to store extracted sentence pairs
    sentence_pairs = []

    # Iterate through translation units ('tu' elements) in the TMX file
    for tu in root.findall(".//tu", namespaces=ns):
        # Extract source and target segments for each translation unit
        source_segment = tu.find(".//tuv[@xml:lang='en']/seg", namespaces=ns).text
        target_segment = tu.find(".//tuv[@xml:lang='fr']/seg", namespaces=ns).text

        # Append the extracted pair to the list
        sentence_pairs.append({"source": source_segment, "target": target_segment})

    # Return the list of extracted sentence pairs
    return sentence_pairs


# Replace 'en-fr.tmx' with the path to your TMX file
tmx_file_path = "./en-fr_sentence_pairs/en-fr.tmx"
# Call the function to extract sentence pairs
sentence_pairs = extract_sentence_pairs(tmx_file_path)

# Save the extracted sentence pairs to a JSON file
json_file_path = "./en-fr_sentence_pairs/en-fr_sentence_pairs.json"
with open(json_file_path, "w", encoding="utf-8") as json_file:
    # Serialize the list of sentence pairs to JSON with indentation
    json.dump(sentence_pairs, json_file, ensure_ascii=False, indent=2)

# Print a message indicating the successful save of the JSON file
print(f"Sentence pairs saved to {json_file_path}")
