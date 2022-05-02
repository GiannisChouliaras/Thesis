class Cases:
    def __init__(self, case: int) -> None:
        self.case = case

    @property
    def before_path(self) -> str:
        return f"../data/raw/CASE{self.case}/paired/before.jpg"

    @property
    def after_path(self) -> str:
        return f"../data/raw/CASE{self.case}/paired/after.jpg"
