"""Provides an interface to the necessary functions from the R package
'fancyscaffold.'
"""

try:
    import rpy2.robjects.packages as packages
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
except ModuleNotFoundError:
    packages = None
    pandas2ri = None

def import_fancyscaffold():
    """Tests if fancyscaffold is installed. If not, the function tells the user
    to install it from Github.
    """
    try:
        fancyscaffold = packages.importr('fancyscaffold')
    except rpy2.rinterface_lib.embedded.RRuntimeError as e:
        raise ImportError('fancyscaffold is required for this functionality, '
                          'but has not been installed. Install fancyscaffold '
                          'on "https://github.com/sflippl/fancyscaffold".')\
            from e
    except NameError as e:
    raise ImportError('Rpy2 is required for this functionality, but has '
                      'not been installed. Install it using conda, pip, '
                      'or any other preferred package manager.')\
        from e
    return fancyscaffold
