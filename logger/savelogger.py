class SaveLogger:
    def __init__(self) -> None:
        self.save_dict = dict()
    
    def update_save(self, **kwargs):
        for key in kwargs:
            if key not in self.save_dict:
                self.save_dict[key] = []

            self.save_dict[key].append(kwargs[key])
    
    def save_to_disk(self):
        pass