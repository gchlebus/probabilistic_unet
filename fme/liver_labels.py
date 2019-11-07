# Copyright (c) Fraunhofer MEVIS, Germany. All rights reserved.
# **InsertLicense** code

__author__ = 'gchlebus'

from data.cityscapes.cityscapes_labels import Label

labels = [
  Label("background", 0, 0, "bg", 0, False, False, (0, 0, 0)),
  Label("liver", 1, 1, "liver", 0, False, False, (255, 255, 255)),
]
