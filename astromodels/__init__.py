# -*- coding: utf-8 -*-
import os
import inspect
from . import models1D
from . import models2D

from .models1D import *
from .models2D import *

__pkg_dir__ = os.path.dirname(inspect.getfile(inspect.currentframe()))
__data_dir__ = os.path.join(__pkg_dir__, "data")

