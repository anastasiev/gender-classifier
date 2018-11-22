class User:
    def __init__(self, username, fullname, avatar, photos):
        self.username = username
        self.fullname = fullname
        self.avatar = avatar
        self.photos = photos

    def __str__(self):
        return "[ " + self.username + "\n" + self.fullname + "\n" + self.avatar + "\n" + str(self.photos) + "\n ]"
