from GZSSAR import model

from . import attention, former, simple


def get_fusion_module(module_name, **kwargs):
    if module_name in dir(simple):
        return getattr(simple, module_name)(**kwargs)
    elif module_name in dir(attention):
        return getattr(attention, module_name)(**kwargs)
    elif module_name in dir(former):
        return getattr(former, module_name)(**kwargs)
    else:
        raise ValueError(f'{module_name} not found.')
