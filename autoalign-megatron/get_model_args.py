from dataclasses import asdict, dataclass
import json
import os

from pydantic import BaseModel
import pydantic


@dataclass
class ModelConfig(BaseModel):
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    ffn_hidden_size: int
    num_query_groups: int
    max_position_embeddings: int
    norm_epsilon: float
    add_qkv_bias: bool
    untie_embeddings_and_output_weights: bool

    @classmethod
    def from_file(cls, file_path: str, model_family: str):
        assert os.path.exists(file_path), f"File {file_path} does not exist"
        with open(file_path, 'r') as file:
            config = json.load(file)

        # Dynamically get the list of parameter names
        param_names = {field for field in ModelConfig.model_fields}

        # Map and pop keys that need renaming
        key_mapping = {
            'num_hidden_layers': 'num_layers',
            'rms_norm_eps': 'norm_epsilon',
            'num_key_value_heads': 'num_query_groups',
            "intermediate_size": "ffn_hidden_size",
        }

        default_config = {"add_qkv_bias": True, "untie_embeddings_and_output_weights": False}
        if model_family == "Llama":
            default_config["add_qkv_bias"] = False
        elif model_family == "Qwen":
            default_config["untie_embeddings_and_output_weights"] = not config[
                "tie_word_embeddings"
            ]

        for old_key, new_key in key_mapping.items():
            if old_key in config:
                if old_key == 'num_key_value_heads':
                    config[new_key] = config['num_attention_heads'] // config.pop(old_key)
                else:
                    config[new_key] = config.pop(old_key)

        # Remove unused keys
        config = {k: v for k, v in config.items() if k in param_names}

        # check pydantic version for compatibility
        if pydantic.__version__ >= "2.0.0":
            return cls.model_validate(config | default_config)
        else:
            return cls(**config, **default_config)

    def to_shell_variables(self):
        variables = []
        for k, v in asdict(self).items():
            if type(v) is bool and v:
                variables.append("--" + k.replace('_', '-'))
            elif type(v) is int:
                variables.append("--" + k.replace('_', '-') + '=' + str(v))

        return "\n".join(variables)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate model configuration shell variables.')

    parser.add_argument('--config-path', type=str, help='The path to the config file')
    parser.add_argument(
        '--model-family', type=str, help='The model family name', choices=["Qwen", "Llama"]
    )
    args = parser.parse_args()
    config_path = args.config_path
    if args.model_family is None:
        if 'llama' in config_path.lower():
            args.model_family = "Llama"
        elif 'qwen' in config_path.lower():
            args.model_family = "Qwen"
        else:
            raise ValueError(
                f"Could not infer model family from config path {config_path}, please specify the model family: Qwen or Llama"
            )

    config = ModelConfig.from_file(config_path, args.model_family)
    print(config.to_shell_variables())
