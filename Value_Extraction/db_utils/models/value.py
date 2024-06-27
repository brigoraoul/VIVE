class Value:

    def __init__(self, id, name, description=""):
        self.id = id
        self.name = name
        self.keywords = []
        self.description = description

    def add_keyword(self, keyword):
        self.keywords.append(keyword)

    def change_description(self, description):
        self.description = description

    def print(self):
        print("ID: ", self.id)
        print("Name: ", self.name)
        print("Keywords: ", self.keywords)
        print("Description: ", self.description)
