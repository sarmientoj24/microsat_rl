from omegaconf import OmegaConf


def get_yaml_args(method, train):
    conf = OmegaConf.load(f'./config/{train}/{method}.yaml')
    
    # YAML doesnt have a None value, only string
    for keys, value in conf.items():
        if value == "None":
            conf[keys] = None
    
    return conf