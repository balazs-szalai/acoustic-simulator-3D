# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 23:48:32 2024

@author: balazs
"""

from acoustic_simulator.model_3D.model_builder import Model
from acoustic_simulator.model_3D.transform import (render_model_homogenious_light,
                                                   render_content_homogenious_light,
                                                   cartesian_to_spherical_py,
                                                   spherical_to_cartesian_py)
from acoustic_simulator.model_3D.geometry import (part_ids,
                                                  in_cylinder,
                                                  add_cylinder,
                                                  in_block,
                                                  add_block,
                                                  in_model)