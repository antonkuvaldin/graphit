from . import ProgressBar, isnotebook

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class TQDMProgressBar(ProgressBar):
    def __init__(self):
        super(TQDMProgressBar, self).__init__()
        if isnotebook():
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        self.tqdm = None

    def init_bar(self, size: int):
        self.tqdm = tqdm(total=size)

    def add(self, n):
        self.tqdm.update(n)

    def finish(self):
        self.add(self.tqdm.total - self.tqdm.n)
        self.set_description('Done')
        self.tqdm.refresh(nolock=True)
        self.tqdm.close()

    def set_description(self, text: str):
        if self.predescription is not None:
            text = self.predescription + ' ' + text
        self.tqdm.set_description(desc=text)
