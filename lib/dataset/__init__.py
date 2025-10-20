# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.panoptic import Panoptic as panoptic
from dataset.shelf_synthetic import ShelfSynthetic as shelf_synthetic
from dataset.campus_synthetic import CampusSynthetic as campus_synthetic
from dataset.shelf_synthetic_mpda import ShelfSynthetic_mpda as shelf_synthetic_mpda
from dataset.campus_synthetic_mpda import CampusSynthetic_mpda as campus_synthetic_mpda
from dataset.shelf import Shelf as shelf
from dataset.campus import Campus as campus
from dataset.etri import Etri as etri
from dataset.panoptic_mpda import Panoptic_mpda as panoptic_mpda
from dataset.panoptic_mpda import JointsDataset_mpda as JointsDataset_mpda
from dataset.hospital import Hospital as hospital