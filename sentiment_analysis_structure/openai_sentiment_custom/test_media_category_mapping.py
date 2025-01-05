import unittest

class TestMediaCategoryMapping(unittest.TestCase):
    def setUp(self):
        self.categories = {
            'Scientific News Outlets': ['Nature', 'SciAm', 'STAT', 'NewScientist'],
            # ... other categories
        }
        self.media_outlet_to_category = {}
        for category, outlets in self.categories.items():
            for outlet in outlets:
                self.media_outlet_to_category[outlet] = category

    def test_existing_media_outlet(self):
        outlet = 'Nature'
        expected_category = 'Scientific News Outlets'
        self.assertEqual(self.media_outlet_to_category.get(outlet), expected_category)

    def test_unknown_media_outlet(self):
        outlet = 'UnknownOutlet'
        expected_category = None
        self.assertIsNone(self.media_outlet_to_category.get(outlet))

if __name__ == '__main__':
    unittest.main()
