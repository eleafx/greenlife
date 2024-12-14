# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import requests
from transformers import pipeline
import numpy as np
import os
import json
from datetime import datetime, timedelta
import time
import threading

class FAODataManager:
    def __init__(self):
        self.cache_duration = timedelta(hours=24)
        self.last_update = None
        self.data = {}
        self.lock = threading.Lock()

    def fetch_fao_data(self):
        """Fetch emission data from FAOSTAT public API"""
        try:
            # FAOSTAT public API endpoint
            base_url = "https://fenixservices.fao.org/faostat/api/v1/en/data/GE"

            params = {
                'area': ['5000'],  # World
                'element': ['5312'],  # GHG emissions (CO2eq)
                'item': 'all',
                'year': '2021',  # Most recent year
                'show_codes': True,
                'show_unit': True,
                'output_type': 'json'
            }

            print("Fetching data from FAOSTAT...")
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()  # Added to raise exception for bad status codes


            if response.status_code == 200:
                data = response.json()
                emissions_data = {}

                for record in data.get('data', []):
                    try:
                        item_name = record.get('item_name_e', '').lower()
                        value = float(record.get('value', 0))
                        unit = record.get('unit', '')

                        # Convert to kg CO2e per kg product
                        if unit == 'gigagrams':
                            value = value / 1000000  # Convert to kg per kg product

                        if item_name and value > 0:
                            # Clean and standardize the item name
                            cleaned_name = self.clean_product_name(item_name)
                            if cleaned_name:
                                if cleaned_name in emissions_data:
                                    # Average if we have multiple values
                                    emissions_data[cleaned_name] = (emissions_data[cleaned_name] + value) / 2
                                else:
                                    emissions_data[cleaned_name] = value

                    except (ValueError, TypeError) as e:
                        print(f"Error processing FAO record: {e}")
                        continue

                print(f"Successfully loaded {len(emissions_data)} items from FAOSTAT")
                return emissions_data
            else:
                print(f"FAOSTAT API request failed: {response.status_code}")
                return self.get_default_data()

        except requests.exceptions.RequestException as e:
            print(f"Error accessing FAO API: {e}")
            return self.get_default_data()


    def clean_product_name(self, name):
        """Clean and standardize product names"""
        if not name:
            return ""

        name = name.lower().strip()

        # Remove specific FAO terminology
        remove_terms = [
            'production', 'total', 'raw', 'fresh', 
            'dried', 'processed', '(total)', ',', '.'
        ]

        for term in remove_terms:
            name = name.replace(term, '')

        # Clean up extra spaces
        name = ' '.join(name.split())

        # Map common FAO names to standard names
        name_mapping = {
            'bovine meat': 'beef',
            'poultry meat': 'chicken',
            'swine meat': 'pork',
            'sheep meat': 'lamb',
            'marine fish': 'fish',
            'hen eggs': 'eggs'
        }

        return name_mapping.get(name, name)

    def get_default_data(self):
        """Provide default emission values if API fails"""
        return {
            'beef': 60.0,
            'lamb': 24.0,
            'pork': 7.2,
            'chicken': 6.9,
            'fish': 5.4,
            'eggs': 4.8,
            'milk': 3.2,
            'cheese': 13.5,
            'rice': 2.7,
            'wheat': 0.8,
            'potatoes': 0.5,
            'vegetables': 2.0,
            'fruits': 1.1,
            'beans': 2.0,
            'nuts': 2.3
        }

    def get_data(self):
        """Get FAO data with caching"""
        with self.lock:
            current_time = datetime.now()

            # Check if cache needs updating
            if (not self.last_update or 
                current_time - self.last_update > self.cache_duration or 
                not self.data):

                new_data = self.fetch_fao_data()
                if new_data:
                    self.data = new_data
                    self.last_update = current_time
                    # Save to local cache file
                    self.save_to_cache()
                else:
                    # Try loading from cache file
                    self.load_from_cache()

            return self.data.copy()

    def save_to_cache(self):
        """Save data to local cache file"""
        try:
            cache_dir = 'cache'
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, 'fao_cache.json')

            cache_data = {
                'timestamp': self.last_update.isoformat(),
                'data': self.data
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)

        except Exception as e:
            print(f"Error saving to cache: {e}")

    def load_from_cache(self):
        """Load data from local cache file"""
        try:
            cache_file = os.path.join('cache', 'fao_cache.json')

            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                self.last_update = datetime.fromisoformat(cache_data['timestamp'])
                self.data = cache_data['data']
                return True

        except Exception as e:
            print(f"Error loading from cache: {e}")
            return False

class GreenLifeAssistant:
    def __init__(self):
        try:
            # Initialize FAO data manager
            self.fao_manager = FAODataManager()

            # Initialize other components
            self.classifier = pipeline("zero-shot-classification",
                                     model="facebook/bart-large-mnli")
        except Exception as e:
            st.error(f"Error initializing GreenLife Assistant: {str(e)}")
            self.classifier = None


        # Add impact thresholds first
        self.impact_thresholds = {
            'transport': {
                'low': 2.0,      # kg CO2 per trip
                'medium': 5.0,
                'high': 10.0
            },
            'meal': {
                'low': 3.0,      # kg CO2 per meal
                'medium': 7.0,
                'high': 15.0
            }
        }    
        self.food_cache = {}  # Cache for API results
        self.load_emission_data() # Load food emission data
        self.load_food_measurements() # Load food measurements
        self.load_cooking_emissions() # Load cooking emissions data

        # Add transport emissions data
        self.transport_emissions = {
            'car': 0.192,
            'bus': 0.089,
            'train': 0.041,
            'bike': 0,
            'walk': 0,
            'motorcycle': 0.103,
            'electric_car': 0.053
        }
       
       # Add sustainable alternatives and tips
        self.sustainable_alternatives = {
            'transport': {
                'car': [
                    "Consider cycling for short distances under 5km",
                    "Try carpooling with colleagues or neighbors",
                    "Use public transport for longer journeys",
                    "Combine multiple errands into one trip"
                ],
                'bus': [
                    "Great choice! Could you walk part of the journey?",
                    "Consider cycling for shorter segments"
                ],
                'train': [
                    "Excellent choice for long-distance travel!",
                    "Consider bringing a reusable water bottle"
                ],
                'bike': [
                    "Perfect zero-emission transport choice!",
                    "Remember to maintain your bike for optimal performance"
                ],
                'walk': [
                    "Perfect zero-emission choice!",
                    "Consider tracking your steps for additional health benefits"
                ]
            },
            'meal': {
                'high_impact': [
                    "Try plant-based alternatives for protein",
                    "Consider reducing portion sizes of high-impact ingredients",
                    "Look for locally sourced alternatives",
                    "Try meat-free Mondays as a start"
                ],
                'medium_impact': [
                    "Consider seasonal vegetables",
                    "Try incorporating more legumes",
                    "Look for locally sourced ingredients"
                ],
                'low_impact': [
                    "Great choice of low-impact ingredients!",
                    "Consider growing some herbs at home",
                    "Try composting food waste"
                ]
            }
        }

        # Initialize the database
        self.load_emission_data()


    def get_food_data(self, food_name):
        """Get food data from Open Food Facts API"""
        if not food_name:
            return None

        if not isinstance(food_name, str):
            raise ValueError("Food name must be a string")

        if food_name in self.food_cache:
           return self.food_cache[food_name]
        try:
            url = f"https://world.openfoodfacts.org/cgi/search.pl"
            params = {
            'search_terms': food_name,
            'search_simple': 1,
            'action': 'process',
            'json': 1,
            'page_size': 1  # Get only the best match
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['products']:
                    product = data['products'][0]

                    # Extract relevant nutritional information
                    food_info = {
                        'product_name': product.get('product_name_en', food_name),
                        'categories': product.get('categories', ''),
                        'ingredients': product.get('ingredients_text', ''),
                        'nutrition_per_100g': {
                            'energy': product.get('nutriments', {}).get('energy-kcal_100g', 0),
                            'proteins': product.get('nutriments', {}).get('proteins_100g', 0),
                            'carbohydrates': product.get('nutriments', {}).get('carbohydrates_100g', 0),
                            'fat': product.get('nutriments', {}).get('fat_100g', 0)
                        },
                        'ecoscore': product.get('ecoscore_grade', 'unknown')
                    }

                    # Cache the result
                    self.food_cache[food_name] = food_info
                    return food_info

            return None
        except Exception as e:
            st.error(f"Error fetching food data: {str(e)}")
            return None


    def get_food_emissions_database(self):
        """Get food emissions data from Open Food Facts with better search strategy"""
        try:
            emissions_data = {}

            # Define search strategy for raw ingredients
            search_categories = [
                {'category': 'fresh-foods', 'tag_type': 'categories'},
                {'category': 'vegetables', 'tag_type': 'categories'},
                {'category': 'fruits', 'tag_type': 'categories'},
                {'category': 'legumes', 'tag_type': 'categories'},
                {'category': 'cereals-and-potatoes', 'tag_type': 'categories'},
                {'category': 'raw', 'tag_type': 'states'}
            ]

            base_url = "https://world.openfoodfacts.org/cgi/search.pl"

            for search in search_categories:
                params = {
                    'action': 'process',
                    'tagtype_0': search['tag_type'],
                    'tag_contains_0': 'contains',
                    'tag_0': search['category'],
                    'tagtype_1': 'states',  # Add state filter
                    'tag_contains_1': 'contains',
                    'tag_1': 'en:raw',  # Filter for raw ingredients
                    'json': 1,
                    'page_size': 100,
                    'fields': 'product_name,product_name_en,categories,ecoscore_grade,states'
                }

                try:
                    response = requests.get(base_url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()

                        for product in data.get('products', []):
                            # Prefer English name, fallback to general name
                            product_name = (product.get('product_name_en') or 
                                          product.get('product_name', '')).lower()

                            # Check if it's a raw/fresh product
                            categories = product.get('categories', '').lower()
                            states = product.get('states', '').lower()

                            if ('raw' in states or 'fresh' in categories) and product_name:
                                # Clean the product name
                                cleaned_name = self.clean_product_name(product_name)

                                if cleaned_name:  # Only process if we have a valid name
                                    ecoscore = product.get('ecoscore_grade')
                                    if ecoscore:
                                        emissions = self.estimate_emissions_from_ecoscore(ecoscore)

                                        # Store or update emissions data
                                        if cleaned_name in emissions_data:
                                            emissions_data[cleaned_name] = (emissions_data[cleaned_name] + emissions) / 2
                                        else:
                                            emissions_data[cleaned_name] = emissions

                                        print(f"Found: {cleaned_name} (ecoscore: {ecoscore})")

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching {search['category']}: {str(e)}")
                    continue

            # Also search specifically for common ingredients
            common_ingredients = [
                'potato', 'carrot', 'tomato', 'onion', 'rice',
                'beans', 'peas', 'cucumber', 'lettuce', 'spinach'
            ]

            for ingredient in common_ingredients:
                params = {
                    'action': 'process',
                    'search_terms': ingredient,
                    'tagtype_0': 'states',
                    'tag_contains_0': 'contains',
                    'tag_0': 'en:raw',  # Only raw ingredients
                    'json': 1,
                    'page_size': 50,
                    'fields': 'product_name,product_name_en,categories,ecoscore_grade,states'
                }

                try:
                    response = requests.get(base_url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()

                        for product in data.get('products', []):
                            product_name = (product.get('product_name_en') or 
                                          product.get('product_name', '')).lower()

                            if product_name:
                                cleaned_name = self.clean_product_name(product_name)
                                ecoscore = product.get('ecoscore_grade')

                                if cleaned_name and ecoscore:
                                    emissions = self.estimate_emissions_from_ecoscore(ecoscore)
                                    if cleaned_name in emissions_data:
                                        emissions_data[cleaned_name] = (emissions_data[cleaned_name] + emissions) / 2
                                    else:
                                        emissions_data[cleaned_name] = emissions

                                    print(f"Found: {cleaned_name} (ecoscore: {ecoscore})")

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching {ingredient}: {str(e)}")
                    continue

            print(f"\nTotal products found: {len(emissions_data)}")
            print("Sample of database items:", list(emissions_data.items())[:5])

            return emissions_data

        except Exception as e:
            st.error(f"Error in get_food_emissions_database: {str(e)}")
            return {}
    
    def clean_product_name(self, name):
        """Clean product name focusing on raw ingredients"""
        if not name:
            return ""

        # Convert to lowercase and remove extra spaces
        name = name.lower().strip()

        # Skip processed foods and snacks
        skip_terms = ['chips', 'crisps', 'snack', 'pringles', 'processed', 
                     'prepared', 'frozen', 'ready', 'instant', 'flavored']
        for term in skip_terms:
            if term in name:
                return ""

        # Basic cleaning
        remove_words = [
            'organic', 'bio', 'fresh', 'raw', 'whole', 'natural',
            'premium', 'quality', 'grade', 'brand', 'packaged',
            'washed', 'unwashed', 'clean', 'dirty', 'loose',
            'selected', 'choice', 'finest'
        ]

        name_parts = name.split()
        name_parts = [word for word in name_parts if word not in remove_words]
        name = ' '.join(name_parts)

        # Remove measurements and other unnecessary information
        import re
        patterns = [
            r'\d+[gkm]?g\b',
            r'\d+\s*pack\b',
            r'\d+\s*piece\w*\b',
            r'\(.*?\)',
            r'™',
            r'®',
        ]

        for pattern in patterns:
            name = re.sub(pattern, '', name)

        # Basic singularization
        if name.endswith('oes'):
            name = name[:-2]
        elif name.endswith('s'):
            name = name[:-1]

        return name.strip()    

    def estimate_emissions_from_ecoscore(self, ecoscore):
        """Estimate emissions based on ecoscore grade with more granular factors"""
        emissions_factors = {
            'a': 1.5,    # Very low environmental impact
            'b': 3.0,    # Low environmental impact
            'c': 5.0,    # Medium environmental impact
            'd': 7.5,    # High environmental impact
            'e': 10.0,   # Very high environmental impact
            'unknown': 5.0
        }

        return emissions_factors.get(str(ecoscore).lower(), emissions_factors['unknown'])

    def load_emission_data(self):
        """Load emission data from multiple sources with fallback"""
        try:
            # Initialize food emissions dictionary
            self.food_emissions = {}

            #  Load FAO data as primary source
            print("Loading FAO emission data...")
            fao_data = self.fao_manager.get_data()
            if fao_data:
                self.food_emissions.update(fao_data)
                print(f"Loaded {len(fao_data)} items from FAO database")


            # Try getting data from Open Food Facts as supplementary source
            print("Fetching data from Open Food Facts...")
            off_emissions = self.get_food_emissions_database()
            if off_emissions:
                print(f"Found {len(off_emissions)} items from Open Food Facts")

                # Merge with priority to FAO data
                for food, emission in off_emissions.items():
                    if food not in self.food_emissions:
                        self.food_emissions[food] = emission

            # Add verified default values
            verified_emissions = {
                'beef': 60.0,      # kg CO2e per kg
                'lamb': 24.0,
                'cheese': 13.5,
                'pork': 7.2,
                'chicken': 6.9,
                'eggs': 4.8,
                'egg': 4.8,       
                'rice': 2.7,
                'milk': 3.2,
                'vegetables': 2.0,
                'fruits': 1.1,
                'beans': 2.0,
                'nuts': 2.3,
                'tofu': 2.0,
                'bread': 1.3,
                'pasta': 1.2,
                'oil': 3.0
            }

            # Update with verified values where needed
            print("Updating with verified values...")
            for food, emission in verified_emissions.items():
                current = self.food_emissions.get(food)
                if not current or abs(current - emission) > emission * 0.5:
                    self.food_emissions[food] = emission

            # Create variations of food names (singular/plural)
            food_variations = {}
            for food, emission in self.food_emissions.items():
                food_lower = food.lower()
                food_variations[food_lower] = emission
                # Add singular version if plural
                if food_lower.endswith('s'):
                    food_variations[food_lower[:-1]] = emission
                # Add plural version if singular
                else:
                    food_variations[f"{food_lower}s"] = emission

            # Update with variations
            self.food_emissions.update(food_variations)

            # Print final database stats
            print(f"Final database size: {len(self.food_emissions)} items")
            print("Sample of database items:", list(self.food_emissions.items())[:5])

            # Store the last update time
            self.last_update = datetime.now()

        except Exception as e:
            st.error(f"Error loading emission data: {str(e)}")
            # Fallback to verified emissions if everything else fails
            self.food_emissions = verified_emissions

    def add_verified_defaults(self):
        """Add scientifically verified default values"""
        verified_emissions = {
            'beef': 60.0,      # kg CO2e per kg (IPCC data)
            'lamb': 24.0,
            'cheese': 13.5,
            'pork': 7.2,
            'chicken': 6.9,
            'eggs': 4.8,
            'rice': 2.7,
            'milk': 3.2,
            'vegetables': 2.0,
            'fruits': 1.1,
            'beans': 2.0,
            'nuts': 2.3,
            'tofu': 2.0
        }

        # Update only if we don't have data or if current value seems incorrect
        for food, emission in verified_emissions.items():
            current = self.food_emissions.get(food)
            if not current or abs(current - emission) > emission * 0.5:
                self.food_emissions[food] = emission

    def calculate_transport_emissions(self, distance, mode):
        """Calculate transport emissions using stored factors"""
        if distance < 0:
           raise ValueError("Distance cannot be negative")

        if not mode or mode.lower() not in self.transport_emissions:
           raise ValueError(f"Invalid transport mode. Must be one of: {', '.join(self.transport_emissions.keys())}")

        return distance * self.transport_emissions[mode.lower()]
    
    def load_food_measurements(self):
        """Load standard food measurement conversions"""
        self.food_measurements = {
            # Standard conversions to grams
            'cup': {
                'rice': 200,
                'pasta': 100,
                'flour': 120,
                'sugar': 200,
                'milk': 240,
                'vegetables': 150,
                'fruits': 150,
                'cheese': 100,
                'nuts': 150,
                'default': 150
            },
            'tablespoon': {
                'oil': 15,
                'butter': 14,
                'sugar': 12,
                'flour': 8,
                'default': 15
            },
            'teaspoon': {
                'salt': 6,
                'sugar': 4,
                'spices': 2,
                'default': 5
            },
            'piece': {
                'bread': 30,    # 1 slice
                'egg': 60,      # 1 medium egg
                'apple': 180,   # 1 medium apple
                'banana': 120,  # 1 medium banana
                'chicken': 150, # 1 chicken breast
                'fish': 140,    # 1 fillet
                'default': 100
            },
            # Add serving sizes
            'serving': {
                'rice': 150,
                'pasta': 150,
                'meat': 150,
                'fish': 140,
                'vegetables': 150,
                'fruits': 150,
                'default': 150
            }
        }

    def parse_food_amount(self, amount_str):
        """Parse food amount from string with units"""
        try:
            # Regular expressions for matching common amount formats
            import re

            # Remove parentheses and their contents
            amount_str = re.sub(r'\([^)]*\)', '', amount_str)

            # Pattern for matching numbers (including fractions) and units
            pattern = r'((?:\d+\s*\/\s*\d+)|(?:\d*\.?\d+))?\s*(cup|cups|tbsp|tsp|tablespoon|teaspoon|piece|pieces|g|gram|grams|kg|kilogram|kilograms|serving|servings|oz|ounce|ounces|lb|pound|pounds)?'

            match = re.match(pattern, amount_str.strip().lower())
            if match:
                amount, unit = match.groups()

                # Convert amount to float
                if amount:
                    if '/' in str(amount):
                        num, denom = map(float, amount.split('/'))
                        amount = num / denom
                    else:
                        amount = float(amount)
                else:
                    amount = 1.0  # Default to 1 if no amount specified

                # Standardize units
                if unit:
                    if unit in ['cup', 'cups']:
                        return 'cup', amount
                    elif unit in ['tbsp', 'tablespoon']:
                        return 'tablespoon', amount
                    elif unit in ['tsp', 'teaspoon']:
                        return 'teaspoon', amount
                    elif unit in ['piece', 'pieces']:
                        return 'piece', amount
                    elif unit in ['g', 'gram', 'grams']:
                        return 'gram', amount
                    elif unit in ['kg', 'kilogram', 'kilograms']:
                        return 'gram', amount * 1000
                    elif unit in ['serving', 'servings']:
                        return 'serving', amount
                    elif unit in ['oz', 'ounce', 'ounces']:
                        return 'gram', amount * 28.35
                    elif unit in ['lb', 'pound', 'pounds']:
                        return 'gram', amount * 453.592

                return 'piece', amount  # Default to pieces if no unit specified

            return None, None
        except Exception as e:
            st.error(f"Error parsing amount: {str(e)}")
            return None, None

    def convert_to_grams(self, amount, unit, food_type):
        """Convert amount to grams based on food type"""
        try:
            if unit == 'gram':
                return amount

            # Get conversion factor based on food type and unit
            conversion = self.food_measurements.get(unit, {})
            factor = conversion.get(food_type, conversion.get('default', 100))

            return amount * factor
        except Exception as e:
            st.error(f"Error converting to grams: {str(e)}")
            return 0


    def load_cooking_emissions(self):
        """Load cooking method emission factors"""
        # Emissions in kg CO2e per hour of cooking
        self.cooking_emissions = {
            'boiling': 0.42,      # Electric stove boiling
            'simmering': 0.21,    # Electric stove low heat
            'frying': 0.45,       # Electric stove frying
            'baking': 0.75,       # Electric oven
            'microwave': 0.06,    # Microwave cooking
            'pressure_cooking': 0.25,  # Electric pressure cooker
            'slow_cooking': 0.15,  # Slow cooker
            'grilling': 0.68,     # Electric grill
            'steaming': 0.30,     # Electric steamer
            'raw': 0.0            # No cooking needed
        }

        # Average cooking times in hours
        self.cooking_times = {
            'rice': {
                'boiling': 0.33,
                'pressure_cooking': 0.17,
                'rice_cooker': 0.33
            },
            'pasta': {
                'boiling': 0.17
            },
            'vegetables': {
                'steaming': 0.17,
                'boiling': 0.17,
                'frying': 0.17,
                'microwave': 0.08,
                'raw': 0
            },
            'meat': {
                'frying': 0.25,
                'baking': 0.75,
                'grilling': 0.33,
                'pressure_cooking': 0.5
            },
            'fish': {
                'frying': 0.17,
                'baking': 0.33,
                'steaming': 0.25
            },
            'default': {
                'frying': 0.25,
                'boiling': 0.33,
                'baking': 0.5,
                'microwave': 0.08
            }
        }

        # Energy source factors (multipliers based on energy source)
        self.energy_source_factors = {
            'electricity': 1.0,
            'natural_gas': 0.6,    # Generally cleaner than electricity
            'induction': 0.8,      # More efficient than regular electric
            'solar': 0.1,          # Very low emissions
            'wood': 1.5            # Higher emissions
        } 

    def parse_cooking_method(self, description):
        """Extract cooking method from description"""
        cooking_keywords = {
            'fry': 'frying',
            'fried': 'frying',
            'stir-fry': 'frying',
            'sautee': 'frying',
            'boil': 'boiling',
            'boiled': 'boiling',
            'simmer': 'simmering',
            'simmered': 'simmering',
            'bake': 'baking',
            'baked': 'baking',
            'roast': 'baking',
            'roasted': 'baking',
            'microwave': 'microwave',
            'steamed': 'steaming',
            'steam': 'steaming',
            'grill': 'grilling',
            'grilled': 'grilling',
            'pressure cook': 'pressure_cooking',
            'raw': 'raw'
        }

        description = description.lower()
        for keyword, method in cooking_keywords.items():
            if keyword in description:
                return method
        return 'frying'  # default method if none specified           

    def analyze_meal(self, meal_description, cooking_method=None, energy_source='electricity'):
        try:
            ingredients = [i.strip() for i in meal_description.split(',')]

            # If cooking method not specified, try to parse from description
            if not cooking_method:
                cooking_method = self.parse_cooking_method(meal_description)

            total_emissions = 0
            identified_ingredients = []
            detailed_info = []

            for ingredient in ingredients:
                # Improved parsing
                ingredient = ingredient.strip()
                # Match pattern like "2 cups rice" or "300g chicken"
                parts = ingredient.split()

                if len(parts) >= 2:
                    # Try to find the unit in the ingredient string
                    amount_str = parts[0]  # First part is the amount

                    # Check if the second part is a unit
                    possible_units = ['cup', 'cups', 'tbsp', 'tsp', 'tablespoon', 'teaspoon', 
                                    'piece', 'pieces', 'g', 'gram', 'grams', 'kg', 'kilogram', 
                                    'serving', 'servings', 'oz', 'ounce', 'lb', 'pound']

                    if parts[1].lower() in possible_units:
                        # Unit is present, food name starts after unit
                        food = ' '.join(parts[2:])
                        amount_str = f"{parts[0]} {parts[1]}"
                    else:
                        # No unit present, assume pieces
                        food = ' '.join(parts[1:])
                        amount_str = f"{parts[0]} piece"
                else:
                    # Default to 1 piece if no amount specified
                    amount_str, food = '1 piece', ingredient

                # Get base ingredient emissions
                unit, amount = self.parse_food_amount(amount_str)
                if unit and amount:
                    grams = self.convert_to_grams(amount, unit, food.lower())

                    # Try different variations of the food name
                    food_name = food.lower()
                    food_singular = food_name[:-1] if food_name.endswith('s') else food_name
                    food_plural = f"{food_name}s" if not food_name.endswith('s') else food_name

                    emission_factor = (
                        self.food_emissions.get(food_name) or 
                        self.food_emissions.get(food_singular) or 
                        self.food_emissions.get(food_plural)
                    )
                    if emission_factor is None:
                        st.warning(f"No emission data found for {food}. Using default value of 3.0 kg CO2e/kg. This is an estimate and actual emissions may vary.")
                        emission_factor = 3.0
                    base_emission = (grams / 1000) * emission_factor

                    # Calculate cooking emissions
                    food_type = self.get_food_type(food.lower())
                    cooking_time = self.cooking_times.get(food_type, self.cooking_times['default']).get(cooking_method, 0.25)
                    cooking_emission = self.cooking_emissions.get(cooking_method, 0) * cooking_time

                    # Apply energy source factor
                    cooking_emission *= self.energy_source_factors.get(energy_source, 1.0)

                    total_emission = base_emission + cooking_emission

                    identified_ingredients.append((food, total_emission))
                    total_emissions += total_emission

                    detailed_info.append({
                        'ingredient': food,
                        'original_amount': f"{amount} {unit}",
                        'grams': grams,
                        'base_emission': base_emission,
                        'cooking_emission': cooking_emission,
                        'total_emission': total_emission,
                        'cooking_method': cooking_method,
                        'cooking_time': cooking_time
                    })

            return {
                'emissions': total_emissions,
                'ingredients': identified_ingredients,
                'detailed_info': detailed_info,
                'cooking_method': cooking_method,
                'energy_source': energy_source
            }
        
            

        except Exception as e:
            st.error(f"Error analyzing meal: {str(e)}")
            return {'emissions': 0, 'ingredients': [], 'detailed_info': []}

    def validate_input(self, meal_description, cooking_method, energy_source):
        """Validate user input"""
        if not meal_description or not meal_description.strip():
            return False, "Please enter a meal description"

        if cooking_method not in self.cooking_emissions:
            return False, "Invalid cooking method"

        if energy_source not in self.energy_source_factors:
            return False, "Invalid energy source"

        return True, ""    

    def get_food_type(self, food):
        """Categorize food into types for cooking time estimation"""
        food_categories = {
            'rice': ['rice', 'risotto'],
            'pasta': ['pasta', 'noodles', 'spaghetti'],
            'vegetables': ['vegetables', 'carrots', 'broccoli', 'spinach', 'lettuce'],
            'meat': ['beef', 'chicken', 'pork', 'lamb'],
            'fish': ['fish', 'salmon', 'tuna', 'cod']
        }

        for category, foods in food_categories.items():
            if food in foods:
                return category
        return 'default'
    

    def meal_analysis_page(self):        
        st.title("🥗 Sustainable Meal Planner")
        
        st.write("""
            Enter ingredients with amounts (e.g., '2 cups rice, 300g chicken, 1 tablespoon oil')
            Supported units: cups, tablespoons (tbsp), teaspoons (tsp), pieces, grams (g), kilograms (kg), servings
            """)

        meal_input = st.text_area("Enter ingredients")

        col1, col2 = st.columns(2)
        with col1:
            cooking_method = st.selectbox(
                "Cooking Method",
                options=['frying', 'boiling', 'baking', 'microwave', 'steaming', 
                        'grilling', 'pressure_cooking', 'simmering', 'raw'],
                help="Select the primary cooking method"
            )
        with col2:
            energy_source = st.selectbox(
                "Energy Source",
                options=['electricity', 'natural_gas', 'induction', 'solar', 'wood'],
                help="Select your cooking energy source"
            )    

        
        if st.button("Analyze Meal"):
            valid, error_message = self.validate_input(meal_input, cooking_method, energy_source)
            if not valid:
                st.error(error_message)
                return
            if meal_input:
                analysis = self.analyze_meal(
                    meal_input,
                    cooking_method=cooking_method,
                    energy_source=energy_source
                )
                # Display results
                st.header("Analysis Results")

                # Show ingredient breakdown
                st.subheader("Ingredients Breakdown")
                for info in analysis['detailed_info']:
                    with st.expander(f"{info['ingredient'].title()}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Amount: {info['original_amount']}")
                            st.write(f"Weight: {info['grams']}g")
                        with col2:
                            st.write(f"Base Emissions: {info['base_emission']:.2f} kg CO2e")
                            st.write(f"Cooking Emissions: {info['cooking_emission']:.2f} kg CO2e")
                            st.write(f"Total: {info['total_emission']:.2f} kg CO2e")

                # Show total emissions
                st.metric(
                    "Total Meal Emissions", 
                    f"{analysis['emissions']:.2f} kg CO2e",
                    help="Total carbon dioxide equivalent emissions for this meal"
                )

                # Generate and display recommendations
                recommendations = self.generate_meal_recommendations(
                    analysis['ingredients'],
                    cooking_method,
                    energy_source
                )

                self.display_recommendations(recommendations)


                # Create emissions chart
                emissions_data = pd.DataFrame(
                    analysis['ingredients'],
                    columns=['Ingredient', 'Emissions']
                )

                fig = px.pie(
                    emissions_data, 
                    values='Emissions', 
                    names='Ingredient',
                    title='Emissions Distribution by Ingredient'
                )
                st.plotly_chart(fig)
        else:
            st.info("Please enter ingredients to analyze")
    
    def get_meal_statistics(self, activities):
        """Calculate statistics for meal activities"""
        meal_activities = [a for a in activities if a['type'] == 'meal']

        if not meal_activities:
            return None

        stats = {
            'total_meals': len(meal_activities),
            'total_emissions': sum(a['emissions'] for a in meal_activities),
            'common_ingredients': {},
            'highest_impact_meals': [],
            'lowest_impact_meals': []
        }

        # Track ingredient frequencies and emissions
        for activity in meal_activities:
            if 'detailed_info' in activity:
                for info in activity['detailed_info']:
                    ingredient = info['ingredient']
                    if ingredient not in stats['common_ingredients']:
                        stats['common_ingredients'][ingredient] = {
                            'count': 0,
                            'total_emissions': 0
                        }
                    stats['common_ingredients'][ingredient]['count'] += 1
                    stats['common_ingredients'][ingredient]['total_emissions'] += info['total_emission']

        # Sort meals by impact
        sorted_meals = sorted(meal_activities, key=lambda x: x['emissions'])
        stats['lowest_impact_meals'] = sorted_meals[:3]
        stats['highest_impact_meals'] = sorted_meals[-3:]

        return stats
            

    def generate_meal_recommendations(self, identified_ingredients, cooking_method, energy_source):
        """Generate comprehensive meal recommendations including cooking methods"""
        recommendations = {
            'ingredient_alternatives': [],
            'cooking_method_tips': [],
            'energy_saving_tips': [],
            'total_potential_savings': 0.0
        }

        # 1. Ingredient-based recommendations
        for ingredient, emission in identified_ingredients:
            if emission > 10:  # High-emission ingredient
                alternatives = [
                    (food, emission) 
                    for food, emission in self.food_emissions.items()
                    if emission < self.food_emissions[ingredient] / 2
                ]
                if alternatives:
                    alt = min(alternatives, key=lambda x: x[1])
                    reduction = emission - alt[1]
                    reduction_percent = ((emission - alt[1]) / emission) * 100
                    recommendations['ingredient_alternatives'].append({
                        'original': ingredient,
                        'alternative': alt[0],
                        'impact': f"Reduces emissions by {reduction_percent:.1f}% (from {emission:.1f} to {alt[1]:.1f} kg CO2e)",
                        'tips': self.sustainable_alternatives['meal']['high_impact'][0],
                        'potential_saving': reduction
                    })
                    recommendations['total_potential_savings'] += reduction

        # 2. Cooking method recommendations
        cooking_emissions = {
            method: self.cooking_emissions[method] 
            for method in self.cooking_emissions 
            if method != cooking_method
        }

        current_method_emission = self.cooking_emissions[cooking_method]
        better_methods = {
            method: emission 
            for method, emission in cooking_emissions.items() 
            if emission < current_method_emission
        }

        if better_methods:
            best_method = min(better_methods.items(), key=lambda x: x[1])
            savings = current_method_emission - best_method[1]

            cooking_recommendation = {
                'current_method': cooking_method,
                'recommended_method': best_method[0],
                'potential_saving': savings,
                'tips': []
            }

            # Add specific cooking tips
            if cooking_method == 'frying':
                cooking_recommendation['tips'].extend([
                    "Consider steaming vegetables instead of frying",
                    "Use a lid while cooking to reduce energy consumption",
                    "Cut ingredients into smaller pieces to reduce cooking time"
                ])
            elif cooking_method == 'baking':
                cooking_recommendation['tips'].extend([
                    "Batch cook multiple items when using the oven",
                    "Use the microwave for small portions",
                    "Avoid preheating longer than necessary"
                ])
            elif cooking_method == 'boiling':
                cooking_recommendation['tips'].extend([
                    "Use a correctly sized pot with lid",
                    "Consider steaming instead of boiling",
                    "Use minimal water necessary"
                ])

            recommendations['cooking_method_tips'].append(cooking_recommendation)
            recommendations['total_potential_savings'] += savings

        # 3. Energy source recommendations
        current_factor = self.energy_source_factors[energy_source]
        better_sources = {
            source: factor 
            for source, factor in self.energy_source_factors.items() 
            if factor < current_factor
        }

        if better_sources:
            for source, factor in better_sources.items():
                potential_saving = (current_factor - factor) * current_method_emission
                recommendations['energy_saving_tips'].append({
                    'current_source': energy_source,
                    'recommended_source': source,
                    'potential_saving': potential_saving,
                    'tips': [
                        f"Consider switching to {source} for cooking",
                        f"Potential emission reduction: {potential_saving:.2f} kg CO2e per hour of cooking"
                    ]
                })
                recommendations['total_potential_savings'] += potential_saving

        return recommendations


    def calculate_transport_emissions(self, distance, mode):
        """Calculate transport emissions using stored factors"""
        try:
            return distance * self.transport_emissions.get(mode.lower(), 0)
        except TypeError:
            st.error("Invalid distance or transport mode")
            return 0

    def generate_feedback(self, activity_type, data):
        """Generate personalized feedback based on user activity"""
        feedback = {
            'impact_level': '',
            'positive_points': [],
            'suggestions': [],
            'long_term_tips': []
        }
        
        if activity_type == 'transport':
            distance = data['distance']
            mode = data['mode'].lower()
            emissions = data['emissions']
            
            # Determine impact level
            if emissions < self.impact_thresholds['transport']['low']:
                feedback['impact_level'] = 'low'
            elif emissions < self.impact_thresholds['transport']['medium']:
                feedback['impact_level'] = 'medium'
            else:
                feedback['impact_level'] = 'high'
            
            # Generate mode-specific feedback
            feedback['suggestions'].extend(self.sustainable_alternatives['transport'].get(mode, []))
            
            # Add distance-based suggestions
            if mode == 'car' and distance < 5:
                feedback['suggestions'].append(
                    f"Your journey was {distance}km - consider walking or cycling next time!"
                )
            
            # Add positive reinforcement
            if mode in ['bike', 'walk']:
                feedback['positive_points'].append(
                    f"Amazing! Your {mode} journey saved {self.calculate_transport_emissions(distance, 'car'):.2f}kg CO2 compared to driving!"
                )
            
        elif activity_type == 'meal':
            emissions = data['emissions']
            ingredients = data['ingredients']
            
            # Determine impact level
            if emissions < self.impact_thresholds['meal']['low']:
                feedback['impact_level'] = 'low'
                feedback['suggestions'].extend(self.sustainable_alternatives['meal']['low_impact'])
            elif emissions < self.impact_thresholds['meal']['medium']:
                feedback['impact_level'] = 'medium'
                feedback['suggestions'].extend(self.sustainable_alternatives['meal']['medium_impact'])
            else:
                feedback['impact_level'] = 'high'
                feedback['suggestions'].extend(self.sustainable_alternatives['meal']['high_impact'])
            
            # Add ingredient-specific feedback
            high_impact_ingredients = [
                ing for ing, emission in ingredients 
                if emission > self.impact_thresholds['meal']['medium']
            ]
            if high_impact_ingredients:
                feedback['suggestions'].append(
                    f"Consider reducing {', '.join(high_impact_ingredients)} in your meals."
                )
        
        return feedback

    def display_recommendations(self,recommendations):
        """Display comprehensive cooking recommendations"""
        st.header("💡 Sustainability Recommendations")

        # Show total potential impact
        st.metric(
            "Total Potential Emission Reduction",
            f"{recommendations['total_potential_savings']:.2f} kg CO2e",
            help="Total emissions you could save by following all recommendations"
        )

        # Ingredient alternatives
        if recommendations['ingredient_alternatives']:
            st.subheader("🥗 Ingredient Alternatives")
            for rec in recommendations['ingredient_alternatives']:
                with st.expander(f"Alternative for {rec['original'].title()}"):
                    st.write(f"🔄 Try: {rec['alternative'].title()}")
                    st.write(f"📊 Impact: {rec['impact']}")
                    st.write(f"💡 Tip: {rec['tips']}")

        # Cooking method recommendations
        if recommendations['cooking_method_tips']:
            st.subheader("👩‍🍳 Cooking Method Improvements")
            for rec in recommendations['cooking_method_tips']:
                with st.expander(f"Better than {rec['current_method'].title()}"):
                    st.write(f"🔄 Try: {rec['recommended_method'].title()}")
                    st.write(f"📊 Potential saving: {rec['potential_saving']:.2f} kg CO2e per hour")
                    for tip in rec['tips']:
                        st.write(f"💡 {tip}")

        # Energy source recommendations
        if recommendations['energy_saving_tips']:
            st.subheader("⚡ Energy Source Improvements")
            for rec in recommendations['energy_saving_tips']:
                with st.expander(f"Better than {rec['current_source'].title()}"):
                    st.write(f"🔄 Try: {rec['recommended_source'].title()}")
                    for tip in rec['tips']:
                        st.write(f"💡 {tip}")

    
    def get_achievement_badges(self, activities):
        """Calculate achievements based on user's activity history"""
        badges = []
        total_emissions = sum(activity['emissions'] for activity in activities)
        
        # Emissions reduction badges
        if total_emissions < 50:
            badges.append({
                'name': 'Climate Champion',
                'icon': '🏆',
                'description': 'Maintained very low carbon emissions!'
            })
        elif total_emissions < 100:
            badges.append({
                'name': 'Green Warrior',
                'icon': '🌱',
                'description': 'Keeping emissions under control!'
            })
        
        # Activity-specific badges
        transport_activities = [a for a in activities if a['type'] == 'transport']
        green_transport = [a for a in transport_activities if a['emissions'] < 1.0]
        if len(green_transport) >= 5:
            badges.append({
                'name': 'Green Commuter',
                'icon': '🚲',
                'description': 'Made 5+ low-emission journeys!'
            })

        meal_activities = [a for a in activities if a['type'] == 'meal']
        low_impact_meals = [a for a in meal_activities if a['emissions'] < self.impact_thresholds['meal']['low']]
        if len(low_impact_meals) >= 3:
            badges.append({
                'name': 'Sustainable Chef',
                'icon': '👨‍🍳',
                'description': 'Created 3+ low-impact meals!'
            })    
        if len(activities) >= 7:
            badges.append({
                'name': 'Consistent Guardian',
                'icon': '🌍',
                'description': 'Tracked activities for 7+ days!'
            }) 

        # Add cooking method badges
        eco_cooking = len([a for a in meal_activities 
                          if a.get('energy_source') in ['solar', 'induction'] 
                          or a.get('cooking_method') in ['microwave', 'raw']])
        if eco_cooking >= 3:
            badges.append({
                'name': 'Eco Chef',
                'icon': '♨️',
                'description': 'Used eco-friendly cooking methods 3+ times!'
            })       
        
        return badges
    
    def export_activity_data(self,activities):
        """Export activity data to CSV with detailed information"""
        if not activities:
            return None

        flattened_data = []
        for activity in activities:
            if activity['type'] == 'meal':
                base_data = {
                    'date': activity['date'],
                    'type': activity['type'],
                    'description': activity['description'],
                    'cooking_method': activity.get('cooking_method', ''),
                    'energy_source': activity.get('energy_source', ''),
                    'total_emission': activity['emissions']
                }

                # Add ingredient details
                for ingredient_info in activity.get('detailed_info', []):
                    row = base_data.copy()
                    row.update({
                        'ingredient': ingredient_info['ingredient'],
                        'amount': ingredient_info['original_amount'],
                        'base_emission': ingredient_info.get('base_emission', 0),
                        'cooking_emission': ingredient_info.get('cooking_emission', 0),
                        'ingredient_total_emission': ingredient_info.get('total_emission', 0)
                    })
                    flattened_data.append(row)
            else:
                flattened_data.append({
                    'date': activity['date'],
                    'type': activity['type'],
                    'mode': activity.get('mode', ''),
                    'distance': activity.get('distance', ''),
                    'emission': activity['emissions']
                })

        df = pd.DataFrame(flattened_data)
        return df.to_csv(index=False).encode('utf-8')

    
def settings_page():
    st.title("⚙️ Settings")

    # Emissions target
    new_target = st.number_input(
        "Daily Emissions Target (kg CO2)",
        min_value=1.0,
        max_value=100.0,
        value=st.session_state.daily_target
    )
    if new_target != st.session_state.daily_target:
        st.session_state.daily_target = new_target

    # Preferred units
    unit_system = st.selectbox(
        "Preferred Units",
        ["Metric (kg)", "Imperial (lbs)"],
        index=0 if st.session_state.settings['preferred_units'] == 'kg' else 1
    )
    st.session_state.settings['preferred_units'] = 'kg' if unit_system == "Metric (kg)" else 'lbs'

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = {
        'total_emissions': 0.0,
        'activities': [],
        'daily_target': 10.0,
        'last_update': None,
        'settings': {
            'preferred_units': 'kg',
            'notification_enabled': True
        }
    }

    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value    


def main():
    initialize_session_state()
    st.title("🌱 GreenLife Assistant")
    
    app = GreenLifeAssistant()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Choose a feature", 
        ["Track Activity", "Meal Analysis", "View Progress", "Settings"]
    )
    
    if page == "Settings":
       settings_page()

    elif page == "Track Activity":
        st.header("🚶 Track Daily Activity")
        
        activity_type = st.selectbox(
            "Activity Type", 
            ["Transport", "Meal"]
        )
        
        if activity_type == "Transport":
            distance = st.number_input("Distance (km)", 0.0)
            mode = st.selectbox(
                "Mode of Transport", 
                list(app.transport_emissions.keys())
            )
            
            if st.button("Calculate Impact"):
                emissions = app.calculate_transport_emissions(distance, mode)
                # Store more detailed activity data
                new_activity = {
                    'date': datetime.now(),
                    'type': 'transport',
                    'mode': mode,
                    'distance': distance,
                    'emissions': emissions
                }
                st.session_state.activities.append(new_activity)
                st.session_state.total_emissions += emissions
                
                st.success(f"Carbon footprint: {emissions:.2f} kg CO2")
                
                # Show comparison
                emissions_comparison = {
                    transport: app.calculate_transport_emissions(distance, transport)
                    for transport in app.transport_emissions.keys()
                }

                comparison_df = pd.DataFrame({
                    'Transport Mode': emissions_comparison.keys(),
                    'Emissions (kg CO2)': emissions_comparison.values()
                })
                comparison_df = comparison_df.sort_values('Emissions (kg CO2)', ascending=True)

                
                fig = px.bar(comparison_df, 
                           x='Transport Mode', 
                           y='Emissions (kg CO2)',
                           title='Emissions Comparison by Transport Mode')
                # Highlight the selected mode
                fig.update_traces(
                    marker_color=['red' if m == mode else 'blue' for m in comparison_df['Transport Mode']]
                )
                fig.add_annotation(
                    x=mode,
                    y=emissions_comparison[mode],
                    text="Your choice",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10
                )
                st.plotly_chart(fig)
                
        elif activity_type == "Meal":
            st.write("""
            Enter ingredients with amounts (e.g., '2 cups rice, 300g chicken, 1 tablespoon oil')
            Supported units: cups, tablespoons (tbsp), teaspoons (tsp), pieces, grams (g), kilograms (kg), servings
            """)

            meal_description = st.text_area("Describe your meal")

            col1, col2 = st.columns(2)
            with col1:
                cooking_method = st.selectbox(
                    "Cooking Method",
                    options=['frying', 'boiling', 'baking', 'microwave', 'steaming', 
                            'grilling', 'pressure_cooking', 'simmering', 'raw'],
                    help="Select the primary cooking method"
                )
            with col2:
                energy_source = st.selectbox(
                    "Energy Source",
                    options=['electricity', 'natural_gas', 'induction', 'solar', 'wood'],
                    help="Select your cooking energy source"
                )


            if meal_description:
                # Analyze ingredients and portions
                analysis = app.analyze_meal(
                    meal_description,
                    cooking_method=cooking_method,
                    energy_source=energy_source
                )

                if analysis['ingredients']:
                    # Display ingredients breakdown
                    st.subheader("Detected Ingredients:")
                    total_emissions = 0

                    for info in analysis['detailed_info']:
                        with st.expander(f"{info['ingredient'].title()}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"Amount: {info['original_amount']}")
                                st.write(f"Weight: {info['grams']}g")
                            with col2:
                                st.write(f"Base Emissions: {info['base_emission']:.2f} kg CO2e")
                                st.write(f"Cooking Emissions: {info['cooking_emission']:.2f} kg CO2e")
                                st.write(f"Total: {info['total_emission']:.2f} kg CO2e")

                        
                    if st.button("Track This Meal"):
                        # Store meal activity data
                        new_activity = {
                            'date': datetime.now(),
                            'type': 'meal',
                            'description': meal_description,
                            'cooking_method': cooking_method,
                            'energy_source': energy_source,
                            'emissions': analysis['emissions'],
                            'ingredients': [(info['ingredient'], info['total_emission']) for info in analysis['detailed_info']], 
                            'detailed_info': analysis['detailed_info']
                        }
                        st.session_state.activities.append(new_activity)
                        st.session_state.total_emissions += analysis['emissions']

                        # Display results
                        st.success(f"Meal tracked! Total emissions: {analysis['emissions']:.2f} kg CO2e")

                        # Create emissions visualization
                        ingredients_df = pd.DataFrame(
                            [(info['ingredient'], info['total_emission']) for info in analysis['detailed_info']], # Changed from 'emission' to 'total_emission'
                            columns=['Ingredient', 'Emissions']
                        )

                        fig = px.pie(
                            ingredients_df,
                            values='Emissions',
                            names='Ingredient',
                            title='Emissions Distribution by Ingredient'
                        )
                        st.plotly_chart(fig)

                else:
                    st.warning("No ingredients detected. Please try rephrasing your meal description.")
            else:
                st.info("Please describe your meal to analyze its environmental impact.")
    elif page == "Meal Analysis":
        app.meal_analysis_page()
    
    
    elif page == "View Progress":
        st.header("📊 Your Impact")

        if st.session_state.activities:
            # Get badges first
            badges = app.get_achievement_badges(st.session_state.activities)

            # Create a container for badges at the top
            st.subheader("🏆 Your Achievements")
            if badges:
                badge_cols = st.columns(len(badges))
                for idx, badge in enumerate(badges):
                    with badge_cols[idx]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 10px; border: 2px solid #4CAF50; border-radius: 10px; margin: 5px;">
                            <h3>{badge['icon']}</h3>
                            <h4>{badge['name']}</h4>
                            <p>{badge['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Keep tracking your activities to earn achievement badges!")

            # Add a progress tracker for next badge
            total_emissions = sum(activity['emissions'] for activity in st.session_state.activities)
            green_transport = len([a for a in st.session_state.activities 
                                 if a['type'] == 'transport' and a['emissions'] < 1.0])

            st.subheader("🎯 Progress to Next Badge")
            col1, col2 = st.columns(2)

            with col1:
                if total_emissions >= 100:
                    emissions_progress = 0
                    st.progress(0)
                    st.write("Target: Reduce emissions to under 50kg to earn Climate Champion!")
                elif total_emissions >= 50:
                    emissions_progress = (100 - total_emissions) / 50
                    st.progress(emissions_progress)
                    st.write(f"Progress to Green Warrior: {emissions_progress:.1f}%")
                else:
                    st.progress(1.0)
                    st.write("Congratulations! You've achieved Climate Champion status!")

            with col2:
                if green_transport < 5:
                    transport_progress = green_transport / 5
                    st.progress(transport_progress)
                    st.write(f"Progress to Green Commuter: {transport_progress * 100:.1f}%")
                    st.write(f"Need {5 - green_transport} more green journeys!")
                else:
                    st.progress(1.0)
                    st.write("Congratulations! You're a Green Commuter!")

            # Create DataFrame and basic metrics
            df = pd.DataFrame(st.session_state.activities)
            df['date'] = pd.to_datetime(df['date'])

            # Total emissions
            total_emissions = df['emissions'].sum()
            st.metric("Total CO2 Emissions", f"{total_emissions:.2f} kg")

            # Activity breakdown
            st.subheader("Activity Breakdown")
            activity_counts = df['type'].value_counts()
            if not activity_counts.empty:
                fig_breakdown = px.pie(
                    values=activity_counts.values, 
                    names=activity_counts.index,
                    title='Activities Distribution'
                )
                st.plotly_chart(fig_breakdown, key="activity_dist_pie")

            # Daily emissions chart
            daily_data = (df.groupby(df['date'].dt.date)['emissions']
                           .sum()
                           .reset_index())
            daily_data['date'] = pd.to_datetime(daily_data['date'])

            if not daily_data.empty:
                fig_daily = px.bar(
                    daily_data, 
                    x='date', 
                    y='emissions',
                    title='Your Daily Carbon Footprint',
                    labels={'date': 'Date', 'emissions': 'Emissions (kg CO2)'}
                )

                fig_daily.update_layout(
                    bargap=0.2,
                    xaxis_tickformat='%Y-%m-%d',
                    showlegend=False,
                    xaxis_title="Date",
                    yaxis_title="Emissions (kg CO2)",
                    height=400,
                    xaxis={
                        'type': 'category',
                        'tickmode': 'array',
                        'tickvals': daily_data.index,
                        'ticktext': daily_data['date'].dt.strftime('%Y-%m-%d')
                    }
                )

                fig_daily.update_traces(
                    width=0.8,
                    marker_line_width=1
                )

                st.plotly_chart(fig_daily, key="daily_emissions_chart")

            # Add activity log
            st.subheader("Recent Activities")
            for idx, activity in enumerate(reversed(st.session_state.activities[-5:])):
                with st.expander(f"Activity {len(st.session_state.activities) - idx}"):
                    st.write(f"Date: {activity['date'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"Type: {activity['type'].capitalize()}")
                    st.write(f"Emissions: {activity['emissions']:.2f} kg CO2")

                    if activity['type'] == 'transport':
                        if 'mode' in activity:
                            st.write(f"Mode: {activity['mode']}")
                        if 'distance' in activity:
                            st.write(f"Distance: {activity['distance']} km")
                    elif activity['type'] == 'meal':
                        if 'description' in activity:
                            st.write(f"Description: {activity['description']}")
                        if 'ingredients' in activity:
                            st.write("Ingredients:")
                            for ingredient, emission in activity['ingredients']:
                                st.write(f"- {ingredient}: {emission:.1f} kg CO2")

            # Add export button
            csv_data = app.export_activity_data(st.session_state.activities)
            st.download_button(
                label="Download Activity Data",
                data=csv_data,
                file_name="green_life_activities.csv",
                mime="text/csv"
            )

            # Add clear button
            if st.button("Clear All Activities"):
                st.session_state.activities = []
                st.session_state.total_emissions = 0
                st.success("All activities cleared!")
                st.experimental_rerun()
        else:
            st.info("No activities tracked yet. Start by tracking some activities!")
if __name__ == "__main__":
    main()