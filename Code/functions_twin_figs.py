import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
                # Perform the additional task
                alnattr = getattr(self.alnobj, name)
                
                # Call the method for the standalone figure
                subobj = subattr(*args, **kwargs)
                alnobj = alnattr(*args, **kwargs)

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
        
        # If the attribute is a method, wrap it
        if callable(subattr):
            def wrapped_method(*args, **kwargs):
                # Perform the additional task
                alnattr = getattr(self.alnfig, name)
                
                # Call the method for the standalone figure
                subobj = subattr(*args, **kwargs)
                alnobj = alnattr(*args, **kwargs)

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