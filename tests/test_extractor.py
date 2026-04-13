import random
import unittest

from mcap_image_extractor.extractor import (
    ExtractionConfig,
    McapImageExtractor,
    build_image_filename,
)


class ExtractorHelperTests(unittest.TestCase):
    def setUp(self):
        self.extractor = McapImageExtractor(
            ExtractionConfig(
                bag_paths=[],
                output_dir='out',
                image_topics=['/cam/left', '/cam/right'],
            )
        )

    def test_build_image_filename_preserves_exact_topic_and_timestamp(self):
        filename = build_image_filename(
            '/cam/left/image_raw',
            1_713_182_400_123_456_789,
            'png',
        )
        self.assertEqual(
            filename,
            '%2Fcam%2Fleft%2Fimage_raw__1713182400.123456789.png',
        )

    def test_select_nearest_unique_indices(self):
        indices = self.extractor._select_nearest_unique_indices(
            [10, 20, 30, 40],
            [12, 35],
        )
        self.assertEqual(indices, [0, 2])

    def test_allocate_random_counts_respects_capacity(self):
        allocations = self.extractor._allocate_random_counts(
            5,
            {'bag_a': 4, 'bag_b': 6},
        )
        self.assertEqual(sum(allocations.values()), 5)
        self.assertLessEqual(allocations['bag_a'], 4)
        self.assertLessEqual(allocations['bag_b'], 6)

    def test_build_bag_random_plan_selects_requested_count_per_topic(self):
        random.seed(0)
        plan = self.extractor._build_bag_random_plan(
            {
                '/cam/left': [10, 20, 30, 40],
                '/cam/right': [11, 19, 31, 41],
            },
            2,
        )
        self.assertEqual(len(plan['/cam/left']), 2)
        self.assertEqual(len(plan['/cam/right']), 2)


if __name__ == '__main__':
    unittest.main()
