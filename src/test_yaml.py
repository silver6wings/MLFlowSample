import yaml

with open("test_mlprojects.yaml", 'r') as stream:
    try:
        parsed_yaml=yaml.safe_load(stream)
        print(parsed_yaml)
    except yaml.YAMLError as exc:
        print(exc)