from fuzzylearn import __version__
import pkg_resources
fuzzylearn_version = pkg_resources.get_distribution('fuzzylearn').version

def test_version():
    """test version"""
    print(fuzzylearn_version)
    assert __version__ == fuzzylearn_version


if __name__=="__main__":
    print(__version__)

