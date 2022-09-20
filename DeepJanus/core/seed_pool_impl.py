from DeepJanus.core.seed_pool import SeedPool

class SeedPoolRandom(SeedPool):
    def __init__(self, problem, n):
        super().__init__(problem)
        self.n = n
        self.seeds = [problem.generate_random_member() for _ in range(self.n)]

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.seeds[item]
