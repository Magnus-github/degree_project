def str_to_class(string: str):
    module_name, object_name = string.split(":")
    module = __import__(module_name, fromlist=[object_name])
    return getattr(module, object_name)