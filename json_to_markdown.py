import json
import os
import markdown

output_path = "./markdown"
# walk through the cache folder and iterature through json files
for root, dirs, files in os.walk('./cache'):
    for file in files:
        if file.endswith(".json"):
            # convert content to markdown
            file_path = os.path.join(root, file)
            with open(file_path) as json_file:
                markdown_content = json.load(json_file)["content"]
                # write to output/filename.md
                with open(os.path.join(output_path, file.strip("json") + "md"), "w") as output_file:
                    output_file.write(markdown_content)

