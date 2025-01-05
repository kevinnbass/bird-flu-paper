# test_media_category_mapping.py

import unittest
import pandas as pd
from main import media_outlet_to_category  # Adjust the import based on your project structure

class TestMediaCategoryMapping(unittest.TestCase):
    def setUp(self):
        # Sample categories and mapping
        self.categories = {
            'Scientific News Outlets': ['Nature', 'SciAm', 'STAT', 'NewScientist'],
            'Left News Outlets': ['TheAtlantic', 'The Daily Beast', 'The Intercept', 'Mother Jones', 'MSNBC', 'Slate', 'Vox'],
            'Lean Left News Outlets': ['AP', 'Axios', 'CNN', 'Guardian', 'Business Insider', 'NBCNews', 'NPR', 'NYTimes', 'Politico', 'ProPublica', 'WaPo', 'Health News Online Report'],
            'Centrist News Outlets': ['Reuters', 'MarketWatch', 'Financial Times'],
            'Lean Right News Outlets': ['TheDispatch', 'EpochTimes', 'FoxBusiness', 'WSJ', 'National Review', 'WashTimes'],
            'Right News Outlets': ['Breitbart', 'TheBlaze', 'Daily Mail', 'DailyWire', 'FoxNews', 'NYPost', 'Newsmax']
        }
        
        # Create reverse mapping
        self.media_outlet_to_category = {}
        for category, outlets in self.categories.items():
            for outlet in outlets:
                self.media_outlet_to_category[outlet] = category

    def test_existing_media_outlet(self):
        # Test mapping for existing media outlet
        outlet = 'Nature'
        expected_category = 'Scientific News Outlets'
        actual_category = self.media_outlet_to_category.get(outlet)
        self.assertEqual(actual_category, expected_category)

    def test_unknown_media_outlet(self):
        # Test mapping for unknown media outlet
        outlet = 'UnknownOutlet'
        expected_category = None
        actual_category = self.media_outlet_to_category.get(outlet)
        self.assertIsNone(actual_category)

    def test_dataframe_mapping(self):
        # Test DataFrame mapping
        data = {
            'media_outlet': ['Nature', 'TheAtlantic', 'UnknownOutlet']
        }
        df = pd.DataFrame(data)
        df['media_category'] = df['media_outlet'].map(self.media_outlet_to_category)
        df['media_category'] = df['media_category'].fillna('Unknown')
        
        expected = ['Scientific News Outlets', 'Left News Outlets', 'Unknown']
        actual = df['media_category'].tolist()
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()
