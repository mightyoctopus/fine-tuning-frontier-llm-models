import random
from typing import Optional
from unicodedata import category

from transformers import AutoTokenizer
import re

BASE_MODEL = "meta-llama/Llama-3.1-8B"
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7


class Item:
    """
    An Item is a cleaned, curated datapoint of a Product with a Price.
    """

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = "Price is $"
    QUESTION = "How much does this cost to the nearest dollar?"
    REMOVALS = [
        "Manufacturer",
        "Language",
        "Best Sellers Rank",
        "Is Discontinued By Manufacturer"
    ]

    title: str
    price: float
    category: Optional[str]
    token_count: int
    details: Optional[str]
    prompt: Optional[str]
    include = False

    def __init__(self, data, price):
        self.title = data["title"]
        self.price = float(price)
        self.token_count = 0
        self.include = False
        self.prompt = None
        self.details = None
        self.category = None
        self.main_category = data["main_category"]
        self.parse(data)


    def scrub_details(self):
        """
        Clean up the details string by removing unnecessary text that doesn't add value.
        """
        details = self.details
        for remove in Item.REMOVALS:
            details = details.replace(remove, "")

        return details

    def scrub(self, stuff):
        """
        Clean up the provided text by removing unnecessary characters and whitespace
        Also, remove words that are 7+ chars and contain numbers as there are likely
        irrelevant product numbers.
        """
        stuff = re.sub(r"[:\[\]{}\s]", " ", stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")
        words = stuff.split(" ")
        select = [word for word in words if len(word) < 7 or not any(char.isdigit() for char in word)]
        return " ".join(select)

    def parse(self, data):
        """
        Parse this datapoint and if it fits within the allowed Token range
        then set include to "True"
        """
        contents = "\n".join(data["description"])
        if contents:
            contents += "\n"

        features = "\n".join(data["features"])
        if features:
            contents += features + "\n"

        self.details = data["details"]
        if self.details:
            contents += self.scrub_details() + "\n"

        if len(contents) > MIN_CHARS:
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            tokens = Item.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > MIN_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                text = Item.tokenizer.decode(tokens)
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text):
        """
        Set the prompt instance variable to be a prompt appropriate for training
        """
        self.prompt = f"{self.QUESTION}\n\nMain_Category: {self.main_category}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"

        self.token_count = len(Item.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Return a prompt suitable for testing, with the actual price(answer) removed
        so that the LLM model response can be evaluated.
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Return a string version of the Item class.
        """
        return f"<{self.title} = ${self.price}>"



###===============================================================================###
### Test ###
###===============================================================================###

# amazon_data = {
#     "title": "Dell Laptop 14 inches",
#     "price": 1400,
#     "description": ["The Dell Inspiron 14 offers a perfect balance between portability and performance. With its sleek aluminum design and ultra-slim profile, it’s ideal for both students and professionals. The 14-inch Full HD display delivers vibrant colors and crisp visuals for work, streaming, or browsing."],
#     "features": "Intel Core i5 12th Gen processor for fast multitasking 8GB DDR4 RAM and 512GB SSD for smooth, reliable performance, Backlit keyboard and fingerprint reader for convenience and security",
#     "details": "Product Dimensions: 12.7 x 8.6 x 0.7 inches, Item Weight: 3.08 pounds, Operating System: Windows 11 Home, Manufacturer: Dell",
#     "main_category": "Electronics"
# }
#
# item = Item(amazon_data, 122)

### Test if details are properly scrubbed:
# print(item.prompt)
# print(item.scrub_details()) # Scrubbed details


### Test the scrub() method:
# scrubbed = item.scrub("[,,,this is a great product that escalates the efficiency at work{}   ]")
# print(scrubbed)
# print(type(scrubbed))


### Parse() test:

# item.parse(amazon_data)
# print(item.prompt)
