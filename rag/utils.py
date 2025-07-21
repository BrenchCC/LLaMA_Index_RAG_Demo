import json


def print_prtty_json(json_object):
    json_formatted_str = json.dumps(json_object, indent=2, ensure_ascii=False)
    print(json_formatted_str)
