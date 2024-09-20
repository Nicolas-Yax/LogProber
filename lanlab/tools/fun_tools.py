import inspect

def get_args_names(f,return_default_values=False):
    """ Return the list of the names of arguments with a default value"""
    sig = inspect.signature(f)
    arg_names = []
    kwarg_names = []
    kwarg_default_values = []
    for _arg_name,param in enumerate(sig.parameters):
        if param.default == inspect._empty:
            arg_names.append(_arg_name)
        else:
            kwarg_names.append(_arg_name)
            kwarg_default_values.append(param.default)
    if return_default_values:
        return arg_names,kwarg_names,kwarg_default_values
    return arg_names,kwarg_names
