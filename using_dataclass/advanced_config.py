from dataclasses import dataclass, fields, field, asdict
import json, yaml
import argparse
from datetime import datetime
from typing import Optional

@dataclass
class GPUConfig:
    cuda: bool
    seed: int


@dataclass
class TextGenerationConfig:
    model_type: str
    model_name_or_path: str = field(metadata={"help": "the path for model or the name of it e.g. GPT, GPT2"})
    cached_dir: str
    max_length: int
    temperature: float
    top_k: int
    top_p: int
    stop_token: str
    do_sample: bool
    created_at: datetime
    gpu_config: GPUConfig
    padding_text: str#Optional[str] = field(default="ff", metadata={"help": "the character for padding the text"})
    num_beams: Optional[int] = field(default=3, metadata={"help": "number of beams in beam search"})

    @classmethod
    def from_json(cls, json_file):
        config_json = json.load(open(json_file))
        return cls(**config_json)

    @classmethod
    def from_yaml(cls, yaml_file):
        config_yaml = yaml.load(open(yaml_file), yaml.FullLoader)
        return cls(**config_yaml)

    def to_argparse(self, **kwargs):
        parser = argparse.ArgumentParser(description=self.__doc__)
        for attr, f in zip(self.__dict__, fields(self)):
            value = self.__dict__[attr]
            value_or_class = f.type
            if f.metadata:
                help = f.metadata['help']
                parser.add_argument("--" + str(attr.replace('_', '-')), default=value, type=value_or_class, help=help)
            else:
                parser.add_argument("--" + str(attr.replace('_', '-')), default=value, type=value_or_class)

        arg = parser.parse_args()
        return arg

    def __post_init__(self):
        if type(self.created_at) is str:
            self.created_at = datetime.strptime(self.created_at, '%b %d %Y %I:%M%p')

        if type(self.gpu_config) is dict:
            self.gpu_config = GPUConfig(**self.gpu_config)

    def help(self, f=None):
        if f:
            return {f.name: f.metadata['help'] for f in fields(self) if 'help' in f.metadata}[f]
        else:
            return {f.name: f.metadata['help'] for f in fields(self) if 'help' in f.metadata}


if __name__ == "__main__":
    config = TextGenerationConfig.from_json("../config_files/advanced_generation_config.json")
    #config = TextGenerationConfig.from_yaml("../config_files/advanced_generation_config.yml")

    print(fields(config))
    print(config.model_name_or_path)
    print(config.created_at)

    print(config.gpu_config.cuda)

    print(config.help())
    print(config.help("model_name_or_path"))

    config_dict = asdict(config)
    print(config_dict["do_sample"])

    args = config.to_argparse()
    print(args.stop_token)