import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def split_twin_args(args, kwargs):
    # Split arguments between the subfigure and the standalone figure
    split_args = {}
    split_args["args"] = [list(args).copy(), list(args).copy()]
    split_args["kwargs"] = [kwargs.copy(), kwargs.copy()]

    for arg_type, arg_obj in {"args": enumerate(args), "kwargs":kwargs.items()}.items():
        for i, arg in arg_obj:
            if isinstance(arg, TwinStandaloneObj):
                split_args[arg_type][0][i] = arg.subobj
                split_args[arg_type][1][i] = arg.alnobj
            elif isinstance(arg, mpl.patches.Rectangle):
                new_rect = mpl.patches.Rectangle(arg.get_xy(), arg.get_width(), arg.get_height())
                new_rect.update_from(arg)
                split_args[arg_type][1][i] = new_rect
            else:
                pass

    return split_args["args"], split_args["kwargs"]

class TwinStandaloneObj:
    def __init__(self, subobj, alnobj):
        self.subobj = subobj
        self.alnobj = alnobj
    
    def __getattr__(self, name):
        # Get the attribute from the wrapped object
        subattr = getattr(self.subobj, name)
        
        # If the attribute is a method, wrap it
        if callable(subattr):
            def wrapped_method(*args, **kwargs):
                # Split the function arguments
                [subargs, alnargs], [subkwargs, alnkwargs] = split_twin_args(args, kwargs)

                # Perform the additional task
                alnattr = getattr(self.alnobj, name)
                
                # Call the method for the standalone figure
                subobj = subattr(*subargs, **subkwargs)
                alnobj = alnattr(*alnargs, **alnkwargs)

                if isinstance(subobj, (list, np.ndarray, plt.Figure, plt.Axes, plt.GridSpec)):
                    # Combine them into a common axis object
                    if not isinstance(subobj, (list, np.ndarray)):
                        return TwinStandaloneObj(subobj, alnobj)
                    elif isinstance(subobj, list) or len(subobj.shape) == 1:
                        return np.array([TwinStandaloneObj(x,y) for x,y in zip(subobj, alnobj)])
                    else:
                        return np.array([[TwinStandaloneObj(x,y) for x,y in zip(_subobj, _alnobj)] for _subobj,_alnobj in zip(subobj, alnobj)])
                else:
                    return subobj
            return wrapped_method
        else:
            # If it's not a method, return it directly (like attributes)
            return subattr

class TwinStandaloneFigure:
    def __init__(self, subfig, standalone_figure_kwargs={}):
        self.subfig = subfig
        self.alnfig = plt.figure(**standalone_figure_kwargs)

    def __getattr__(self, name):
        # Get the attribute from the wrapped object
        subattr = getattr(self.subfig, name)

        if callable(subattr):
            def wrapped_method(*args, **kwargs):
                # Perform the additional task
                [subargs, alnargs], [subkwargs, alnkwargs] = split_twin_args(args, kwargs)

                # Perform the additional task
                alnattr = getattr(self.alnfig, name)
                
                # Call the method for the standalone figure
                subobj = subattr(*subargs, **subkwargs)
                alnobj = alnattr(*alnargs, **alnkwargs)

                if isinstance(subobj, (list, np.ndarray, plt.Figure, plt.Axes, plt.GridSpec)):
                    # Combine them into a common axis object
                    if not isinstance(subobj, (list, np.ndarray)):
                        return TwinStandaloneObj(subobj, alnobj)
                    elif len(subobj.shape) == 1:
                        return np.array([TwinStandaloneObj(x,y) for x,y in zip(subobj, alnobj)])
                    else:
                        return np.array([[TwinStandaloneObj(x,y) for x,y in zip(_subobj, _alnobj)] for _subobj,_alnobj in zip(subobj, alnobj)])
                else:
                    return subobj
            return wrapped_method
        else:
            # If it's not a method, return it directly (like attributes)
            return subattr