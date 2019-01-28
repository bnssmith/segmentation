import sys
import os
import json

def main():
    json_folder = sys.argv[1]
    img_folder = sys.argv[2]
    output_file = sys.argv[3]

    json_paths = {}
    for root, folders, files in os.walk(json_folder):
        for f in files:
            if f.lower().endswith('.json'):
                img_id = f.split(".")[0]
                json_paths[img_id] = os.path.join(root, f)

    img_paths = {}
    for root, folders, files in os.walk(img_folder):
        for f in files:
            if f.lower().endswith('.jpg') or f.lower().endswith(".png"):
                img_id = f.split(".")[0]
                img_paths[img_id] = os.path.join(root, f)

    ids = list(set(json_paths.keys()) & set(img_paths.keys()))
    ids.sort()

    ls = []
    for i in ids:
        json_path = json_paths[i]
        img_path = img_paths[i]

        ls.append([
            json_path,
            img_path
        ])


    print(len(ls))
    with open(output_file, 'w') as f:
        json.dump(ls, f)


if __name__ == "__main__":
    main()
