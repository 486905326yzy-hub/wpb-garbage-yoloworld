#!/usr/bin/env python3
import os
import tempfile
import unittest

from garbage_yoloworld.core import PromptVocabulary, canonical_nms


class CoreTest(unittest.TestCase):
    def test_prompt_mapping_and_cross_alias_nms(self):
        config = """\
output_class: garbage
classes:
  bottle:
    min_confidence: 0.1
    prompts: [bottle, discarded bottle]
  paper:
    prompts: [paper ball]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
            stream.write(config)
            path = stream.name
        try:
            vocabulary = PromptVocabulary(path)
            first = vocabulary.candidate([0, 0, 100, 100], 0.8, "bottle")
            duplicate = vocabulary.candidate([2, 2, 98, 98], 0.7, "discarded bottle")
            other = vocabulary.candidate([150, 150, 200, 200], 0.6, "paper ball")
            detections = canonical_nms([duplicate, other, first], 0.45, 10)
            self.assertEqual(len(detections), 2)
            self.assertEqual(detections[0]["class_name"], "garbage")
            self.assertEqual(detections[0]["score"], 0.8)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
