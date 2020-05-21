import json

class GenerationConfig:
    def __init__(self, model_type="",
                 model_name_or_path="",
                 cached_dir="",
                 max_length="",
                 temperature="",
                 top_k="",
                 top_p="",
                 stop_token="",
                 do_sample=""):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.cached_dir = cached_dir
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.stop_token = stop_token
        self.do_sample = do_sample

    @classmethod
    def from_dict(cls, json_object):
        config = GenerationConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))

    def __str__(self):
        return str(self.__dict__)


if __name__ == "__main__":
    config = GenerationConfig.from_json_file("../config_files/simple_generation_config.json")
    print(config.max_length)
    print(config.model_name_or_path)