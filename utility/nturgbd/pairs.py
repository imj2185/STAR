class Pairs:
    def __init__(self):
        self.head = [(2, 3), (2, 20), (20, 4), (20, 8)]
        self.left_hand = [(4, 5), (5, 6), (6, 7), (7, 22), (22, 21)]
        self.right_hand = [(8, 9), (9, 10), (10, 11), (11, 24), (24, 23)]
        self.torso = [(20, 4), (20, 8), (20, 1), (1, 0), (0, 12), (0, 16)]
        self.left_leg = [(0, 12), (12, 13), (13, 14), (14, 15)]
        self.right_leg = [(0, 16), (16, 17), (17, 18), (18, 19)]
        self.parts_connection = [(9, 1), (5, 1), (13, 1), (17, 1), (2, 1), (9, 0), (5, 0), (13, 0), (10, 1), (10, 0),
                                 (6, 1),
                                 (6, 0)]
        self.total_collection = set(
            self.head + self.left_hand + self.right_hand + self.torso + self.left_leg + self.right_leg + self.parts_connection)


if __name__ == '__main__':
    pairs = Pairs()
    print(pairs.total_collection)
