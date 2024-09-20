from lanlab.tools.configurable import *
import pytest 

#Write tests for configurable
def test_configurable():
    c = Configurable()
    assert c.config == c.config_class()
    